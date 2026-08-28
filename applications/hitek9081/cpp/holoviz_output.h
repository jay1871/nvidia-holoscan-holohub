#pragma once

#include "holoscan/holoscan.hpp"
#include "spectral_kernels.cuh"
#include "spectral_pipeline.h"
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

class HolovizOutput {
 public:
  struct Config {
    int bin_start = 0;
    int bin_end = 0;
    float db_min = -120.0f;
    float db_max = -10.0f;
    std::vector<float> freq_tick_positions;
  };

  HolovizOutput() = default;
  ~HolovizOutput() { free(); }

  void initialize(const Config& cfg, int num_antennas, cudaStream_t stream) {
    cfg_ = cfg;
    num_bins_ = cfg.bin_end - cfg.bin_start;
    stream_ = stream;

    coords_d_.resize(num_antennas, nullptr);
    for (int a = 0; a < num_antennas; a++) {
      cudaMalloc(&coords_d_[a], num_bins_ * 2 * sizeof(float));
    }

    initGridOverlays();
  }

  void emit(const SpectralPipeline& pipeline, holoscan::OutputContext& op_output) {
    const int A = pipeline.num_antennas();

    for (int a = 0; a < A; a++) {
      normalize_to_coords(pipeline.mag_gpu(a), coords_d_[a],
                          pipeline.fft_size(), cfg_.db_min, cfg_.db_max,
                          cfg_.bin_start, cfg_.bin_end, 0, stream_);
    }

    holoscan::TensorMap out_msg;
    emitGridOverlays(out_msg);
    for (int a = 0; a < A; a++) {
      char name[64];
      snprintf(name, sizeof(name), "antenna_%d", a);
      emitTensor(out_msg, name, coords_d_[a], num_bins_, 2);
    }
    op_output.emit(out_msg, "out");
  }

 private:
  void emitTensor(holoscan::TensorMap& out, const char* name,
                  float* data, int64_t rows, int64_t cols) {
    DLManagedTensor* dl = new DLManagedTensor();
    dl->dl_tensor.data = data;
    dl->dl_tensor.device = DLDevice{kDLCUDA, 0};
    dl->dl_tensor.ndim = 2;
    dl->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    dl->dl_tensor.shape = new int64_t[2]{rows, cols};
    dl->dl_tensor.strides = nullptr;
    dl->dl_tensor.byte_offset = 0;
    dl->manager_ctx = nullptr;
    dl->deleter = [](DLManagedTensor* self) {
      delete[] self->dl_tensor.shape;
      delete self;
    };
    out[name] = std::make_shared<holoscan::Tensor>(dl);
  }

  void initGridOverlays() {
    constexpr int NUM_DB_TICKS = 10;

    const auto& xpos = cfg_.freq_tick_positions;
    const int num_freq_ticks = static_cast<int>(xpos.size());

    if (num_freq_ticks > 0) {
      std::vector<float> freq_lines(num_freq_ticks * 4);
      for (int t = 0; t < num_freq_ticks; t++) {
        freq_lines[t * 4 + 0] = xpos[t];
        freq_lines[t * 4 + 1] = 0.0f;
        freq_lines[t * 4 + 2] = xpos[t];
        freq_lines[t * 4 + 3] = 1.0f;
      }
      cudaMalloc(&ticks_d_, freq_lines.size() * sizeof(float));
      cudaMemcpy(ticks_d_, freq_lines.data(), freq_lines.size() * sizeof(float), cudaMemcpyHostToDevice);
      num_tick_points_ = num_freq_ticks * 2;

      num_labels_ = num_freq_ticks;
      for (int t = 0; t < num_freq_ticks; t++) {
        float pos[3] = {xpos[t], 0.97f, 0.02f};
        float* d = nullptr;
        cudaMalloc(&d, 3 * sizeof(float));
        cudaMemcpy(d, pos, 3 * sizeof(float), cudaMemcpyHostToDevice);
        label_coords_d_.push_back(d);
      }
    }

    float db_lines[NUM_DB_TICKS * 2 * 2];
    for (int t = 0; t < NUM_DB_TICKS; t++) {
      float y = static_cast<float>(t + 1) / static_cast<float>(NUM_DB_TICKS + 1);
      db_lines[t * 4 + 0] = 0.0f;
      db_lines[t * 4 + 1] = y;
      db_lines[t * 4 + 2] = 1.0f;
      db_lines[t * 4 + 3] = y;
    }
    cudaMalloc(&db_ticks_d_, sizeof(db_lines));
    cudaMemcpy(db_ticks_d_, db_lines, sizeof(db_lines), cudaMemcpyHostToDevice);
    num_db_ticks_ = NUM_DB_TICKS;

    for (int t = 0; t < NUM_DB_TICKS; t++) {
      float y = static_cast<float>(t + 1) / static_cast<float>(NUM_DB_TICKS + 1);
      float pos[3] = {0.01f, y, 0.02f};
      float* d = nullptr;
      cudaMalloc(&d, 3 * sizeof(float));
      cudaMemcpy(d, pos, 3 * sizeof(float), cudaMemcpyHostToDevice);
      db_label_coords_d_.push_back(d);
    }
  }

  void emitGridOverlays(holoscan::TensorMap& out) {
    emitTensor(out, "freq_ticks", ticks_d_, num_tick_points_, 2);
    for (int t = 0; t < num_labels_; t++) {
      char name[64];
      snprintf(name, sizeof(name), "label_%d", t);
      emitTensor(out, name, label_coords_d_[t], 1, 3);
    }
    emitTensor(out, "db_ticks", db_ticks_d_, num_db_ticks_ * 2, 2);
    for (int t = 0; t < num_db_ticks_; t++) {
      char name[64];
      snprintf(name, sizeof(name), "db_label_%d", t);
      emitTensor(out, name, db_label_coords_d_[t], 1, 3);
    }
  }

  void free() {
    for (auto p : coords_d_) { if (p) cudaFree(p); }
    if (ticks_d_) cudaFree(ticks_d_);
    if (db_ticks_d_) cudaFree(db_ticks_d_);
    for (auto p : label_coords_d_) { if (p) cudaFree(p); }
    for (auto p : db_label_coords_d_) { if (p) cudaFree(p); }
  }

  Config cfg_;
  int num_bins_ = 0;
  cudaStream_t stream_ = nullptr;

  std::vector<float*> coords_d_;
  float* ticks_d_ = nullptr;
  int num_tick_points_ = 0;
  int num_labels_ = 0;
  std::vector<float*> label_coords_d_;
  float* db_ticks_d_ = nullptr;
  int num_db_ticks_ = 0;
  std::vector<float*> db_label_coords_d_;
};
