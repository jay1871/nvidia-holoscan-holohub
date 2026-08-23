/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * SpectralOp — Windowed FFT with averaging and magnitude display
 *
 * Input:  TensorMap with per-antenna [fft_size, 2] int16 IQ tensors from Hitek9081RxOp
 * Output: TensorMap with display-ready [fft_size, 2] float32 normalized coords for Holoviz
 *
 * Processing per compute() call:
 *   1. For each antenna: window(int16→float) → cuFFT → accumulate power OR finalize
 *   2. Averaging: accumulate |FFT|²/N² over num_averages frames in linear power domain
 *   3. When avg_count reaches num_averages: 10*log10(avg) → DC block → normalize → emit
 *   4. Between emissions: re-emit previous result to keep Holoviz fed
 *
 * Grid overlays: 10 vertical frequency ticks + 10 horizontal dB reference lines
 */

#pragma once

#include "holoscan/holoscan.hpp"
#include "spectral_kernels.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <library_types.h>
#include <vector>
#include <string>
#include <cstdio>

namespace holoscan::ops {

class SpectralOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectralOp)
  SpectralOp() = default;

  ~SpectralOp() { freeResources(); }

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("iq_in")
        .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", static_cast<uint64_t>(2)),
                   holoscan::Arg("policy", static_cast<uint64_t>(0)));
    spec.output<holoscan::TensorMap>("out")
        .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", static_cast<uint64_t>(2)),
                   holoscan::Arg("policy", static_cast<uint64_t>(0)));

    spec.param(fft_size_, "fft_size", "FFT size", "Complex samples per FFT", 65536);
    spec.param(num_antennas_, "num_antennas", "Antennas", "Number of antenna channels", 1);
    spec.param(num_averages_, "num_averages", "FFT averages",
               "Number of FFT frames to average in linear power (1=no averaging)", 1);
    spec.param(db_min_, "db_min", "dB minimum", "Display floor in dB", -80.0f);
    spec.param(db_max_, "db_max", "dB maximum", "Display ceiling in dB", 20.0f);
    spec.param(bin_start_, "bin_start", "Display start bin", "First FFT bin to display", 0);
    spec.param(bin_end_, "bin_end", "Display end bin", "Last FFT bin to display (0=all)", 0);
    spec.param(iq_swap_, "iq_swap", "IQ swap", "Swap I/Q (0=I first, 1=Q first)", 0);
    spec.param(freq_tick_positions_, "freq_tick_positions", "Frequency tick positions",
               "Normalized [0,1] x-positions for frequency grid lines", std::vector<float>{});
  }

  void initialize() override {
    Operator::initialize();

    const int N = fft_size_.get();
    const int A = num_antennas_.get();

    resolved_bin_start_ = bin_start_.get();
    resolved_bin_end_ = (bin_end_.get() == 0) ? N : bin_end_.get();

    cudaStreamCreate(&stream_);

    // Hanning window (precomputed once)
    cudaMalloc(&window_d_, N * sizeof(float));
    generate_hanning_window(window_d_, N, stream_);

    // Batched FFT buffers (contiguous: A antennas × N complex samples)
    cudaMalloc(&fft_in_d_, A * N * sizeof(float2));
    cudaMalloc(&fft_out_d_, A * N * sizeof(float2));

    // Per-antenna post-FFT buffers (fp32 precision for accumulation/display)
    power_accum_d_.resize(A, nullptr);
    mag_d_.resize(A, nullptr);
    coords_d_.resize(A, nullptr);

    for (int a = 0; a < A; a++) {
      cudaMalloc(&power_accum_d_[a], N * sizeof(float));
      cudaMemsetAsync(power_accum_d_[a], 0, N * sizeof(float), stream_);
      cudaMalloc(&mag_d_[a], N * sizeof(float));
      cudaMalloc(&coords_d_[a], N * 2 * sizeof(float));
    }

    // Batched fp32 FFT plan: all antennas in a single cufftXtExec call.
    // fp16 execution loses signals at N=65536 (16 butterfly stages exceed 10-bit mantissa).
    long long int n_ll[1] = {static_cast<long long>(N)};
    size_t workSize = 0;
    cufftCreate(&fft_plan_);
    cufftXtMakePlanMany(fft_plan_, 1, n_ll,
                        NULL, 1, static_cast<long long>(N),
                        CUDA_C_32F,
                        NULL, 1, static_cast<long long>(N),
                        CUDA_C_32F,
                        static_cast<long long>(A),
                        &workSize,
                        CUDA_C_32F);
    cufftSetStream(fft_plan_, stream_);

    initGridOverlays();
    cudaStreamSynchronize(stream_);

    HOLOSCAN_LOG_INFO("SpectralOp::initialize() — fft={} antennas={} averages={} dB=[{},{}] iq_swap={} bins=[{},{}] mode=fp32-batched",
                      N, A, num_averages_.get(), db_min_.get(), db_max_.get(),
                      iq_swap_.get(), resolved_bin_start_, resolved_bin_end_);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& /*context*/) override {
    auto maybe_input = op_input.receive<holoscan::TensorMap>("iq_in");

    if (maybe_input) {
      const auto& input = maybe_input.value();
      const int N = fft_size_.get();
      const int A = num_antennas_.get();
      const int num_avg = num_averages_.get();

      // DC-block + window each antenna's int16 IQ into the batched fp32 buffer
      for (int a = 0; a < A; a++) {
        std::string key = "antenna_" + std::to_string(a);
        auto it = input.find(key);
        if (it == input.end()) continue;

        const auto& tensor = it->second;
        const int16_t* iq_ptr = static_cast<const int16_t*>(tensor->data());

        apply_window_int16_to_float2(iq_ptr, window_d_, fft_in_d_ + a * N,
                                     N, iq_swap_.get(), stream_);
      }

      // Single batched FFT for all antennas
      cufftXtExec(fft_plan_, fft_in_d_, fft_out_d_, CUFFT_FORWARD);

      // Post-process each antenna
      for (int a = 0; a < A; a++) {
        float2* fft_a = fft_out_d_ + a * N;
        if (num_avg > 1) {
          accumulate_power_fftshift_f32(fft_a, power_accum_d_[a], N, 0, stream_);
        } else {
          magnitude_db_fftshift_f32(fft_a, mag_d_[a], N, 0, stream_);
          normalize_to_coords(mag_d_[a], coords_d_[a], N, db_min_.get(), db_max_.get(), resolved_bin_start_, resolved_bin_end_, 0, stream_);
        }
      }

      avg_count_++;

      if (num_avg > 1 && avg_count_ >= num_avg) {
        // Finalize averaging
        for (int a = 0; a < A; a++) {
          power_to_db(power_accum_d_[a], mag_d_[a], N, num_avg, stream_);
          normalize_to_coords(mag_d_[a], coords_d_[a], N, db_min_.get(), db_max_.get(), resolved_bin_start_, resolved_bin_end_, 0, stream_);
          // Reset accumulator for next averaging window
          cudaMemsetAsync(power_accum_d_[a], 0, N * sizeof(float), stream_);
        }
        avg_count_ = 0;
        cudaStreamSynchronize(stream_);
        has_result_ = true;
      } else if (num_avg == 1) {
        cudaStreamSynchronize(stream_);
        has_result_ = true;
      }
      // else: still accumulating, don't update has_result_
    }

    // Emit current result (or re-emit previous to keep Holoviz fed)
    if (has_result_) {
      emit_result(op_output);
    }
  }

 private:
  void emit_result(OutputContext& op_output) {
    const int A = num_antennas_.get();
    const int num_bins = resolved_bin_end_ - resolved_bin_start_;

    holoscan::TensorMap out_msg;
    emitGridOverlays(out_msg);

    for (int a = 0; a < A; a++) {
      std::string name = "antenna_" + std::to_string(a);
      emitTensor(out_msg, name.c_str(), coords_d_[a], num_bins, 2);
    }

    op_output.emit(out_msg, "out");
  }

  // =========================================================================
  // Grid overlay helpers
  // =========================================================================

  void initGridOverlays() {
    constexpr int NUM_DB_TICKS = 10;

    // --- Frequency axis: vertical lines at dynamic integer-MHz positions ---
    const auto& xpos = freq_tick_positions_.get();
    const int num_freq_ticks = static_cast<int>(xpos.size());

    if (num_freq_ticks > 0) {
      std::vector<float> freq_lines(num_freq_ticks * 4);
      for (int t = 0; t < num_freq_ticks; t++) {
        float x = xpos[t];
        freq_lines[t * 4 + 0] = x;
        freq_lines[t * 4 + 1] = 0.0f;
        freq_lines[t * 4 + 2] = x;
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

    // --- dB axis: horizontal lines + label positions ---
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

  // =========================================================================
  // Tensor emission helpers
  // =========================================================================

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

  // =========================================================================

  void freeResources() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (window_d_) cudaFree(window_d_);
    if (fft_in_d_) cudaFree(fft_in_d_);
    if (fft_out_d_) cudaFree(fft_out_d_);
    if (ticks_d_) cudaFree(ticks_d_);
    if (db_ticks_d_) cudaFree(db_ticks_d_);
    for (auto p : label_coords_d_) { if (p) cudaFree(p); }
    for (auto p : db_label_coords_d_) { if (p) cudaFree(p); }
    for (size_t a = 0; a < power_accum_d_.size(); a++) {
      if (power_accum_d_[a]) cudaFree(power_accum_d_[a]);
      if (mag_d_[a]) cudaFree(mag_d_[a]);
      if (coords_d_[a]) cudaFree(coords_d_[a]);
    }
    if (stream_) cudaStreamDestroy(stream_);
  }

  // CUDA state
  cudaStream_t stream_ = nullptr;
  cufftHandle fft_plan_ = 0;
  float* window_d_ = nullptr;

  // Batched fp32 FFT buffers (contiguous: A*N float2 elements)
  float2* fft_in_d_ = nullptr;
  float2* fft_out_d_ = nullptr;

  // Per-antenna post-FFT buffers (fp32)
  std::vector<float*> power_accum_d_;
  std::vector<float*> mag_d_;
  std::vector<float*> coords_d_;

  // Static display elements
  float* ticks_d_ = nullptr;
  int num_tick_points_ = 0;
  int num_labels_ = 0;
  std::vector<float*> label_coords_d_;

  // dB reference lines
  float* db_ticks_d_ = nullptr;
  int num_db_ticks_ = 0;
  std::vector<float*> db_label_coords_d_;

  // Display bin range (resolved at init)
  int resolved_bin_start_ = 0;
  int resolved_bin_end_ = 0;

  // Averaging state
  int avg_count_ = 0;
  bool has_result_ = false;

  // Parameters
  Parameter<int> fft_size_;
  Parameter<int> num_antennas_;
  Parameter<int> num_averages_;
  Parameter<float> db_min_;
  Parameter<float> db_max_;
  Parameter<int> bin_start_;
  Parameter<int> bin_end_;
  Parameter<int> iq_swap_;
  Parameter<std::vector<float>> freq_tick_positions_;
};

}  // namespace holoscan::ops
