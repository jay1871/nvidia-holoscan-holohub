#pragma once

#include "spectral_kernels.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <library_types.h>
#include <vector>
#include <cstdint>

class SpectralPipeline {
 public:
  struct Config {
    int fft_size = 65536;
    int num_antennas = 1;
    int num_averages = 1;
    int adc_bits = 16;
    int iq_swap = 0;
  };

  SpectralPipeline() = default;
  ~SpectralPipeline() { free(); }

  void initialize(const Config& cfg) {
    cfg_ = cfg;
    N_ = cfg.fft_size;
    A_ = cfg.num_antennas;
    adc_fullscale_ = static_cast<float>(1 << (cfg.adc_bits - 1));

    cudaStreamCreate(&stream_);

    cudaMalloc(&window_d_, N_ * sizeof(float));
    generate_hanning_window(window_d_, N_, stream_);

    cudaMalloc(&fft_in_d_, A_ * N_ * sizeof(float2));
    cudaMalloc(&fft_out_d_, A_ * N_ * sizeof(float2));

    power_accum_d_.resize(A_, nullptr);
    mag_d_.resize(A_, nullptr);
    for (int a = 0; a < A_; a++) {
      cudaMalloc(&power_accum_d_[a], N_ * sizeof(float));
      cudaMemsetAsync(power_accum_d_[a], 0, N_ * sizeof(float), stream_);
      cudaMalloc(&mag_d_[a], N_ * sizeof(float));
    }

    long long int n_ll[1] = {static_cast<long long>(N_)};
    size_t workSize = 0;
    cufftCreate(&fft_plan_);
    cufftXtMakePlanMany(fft_plan_, 1, n_ll,
                        NULL, 1, static_cast<long long>(N_), CUDA_C_32F,
                        NULL, 1, static_cast<long long>(N_), CUDA_C_32F,
                        static_cast<long long>(A_), &workSize, CUDA_C_32F);
    cufftSetStream(fft_plan_, stream_);

    cudaStreamSynchronize(stream_);
  }

  void process(const int16_t* const* iq_ptrs, int num_valid) {
    const int num_avg = cfg_.num_averages;

    for (int a = 0; a < num_valid && a < A_; a++) {
      if (!iq_ptrs[a]) continue;
      apply_window_int16_to_float2(iq_ptrs[a], window_d_, fft_in_d_ + a * N_,
                                   N_, cfg_.iq_swap, stream_);
    }

    cufftXtExec(fft_plan_, fft_in_d_, fft_out_d_, CUFFT_FORWARD);

    for (int a = 0; a < A_; a++) {
      float2* fft_a = fft_out_d_ + a * N_;
      if (num_avg > 1) {
        accumulate_power_fftshift_f32(fft_a, power_accum_d_[a], N_, 0, stream_);
      } else {
        magnitude_db_fftshift_f32(fft_a, mag_d_[a], N_, 0, adc_fullscale_, stream_);
      }
    }

    avg_count_++;
    new_result_ = false;

    if (num_avg > 1 && avg_count_ >= num_avg) {
      for (int a = 0; a < A_; a++) {
        power_to_db(power_accum_d_[a], mag_d_[a], N_, num_avg, adc_fullscale_, stream_);
        cudaMemsetAsync(power_accum_d_[a], 0, N_ * sizeof(float), stream_);
      }
      avg_count_ = 0;
      cudaStreamSynchronize(stream_);
      new_result_ = true;
    } else if (num_avg == 1) {
      cudaStreamSynchronize(stream_);
      new_result_ = true;
    }
  }

  bool has_new_result() const { return new_result_; }
  float* mag_gpu(int antenna) const { return mag_d_[antenna]; }
  int fft_size() const { return N_; }
  int num_antennas() const { return A_; }
  cudaStream_t stream() const { return stream_; }

 private:
  void free() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (window_d_) cudaFree(window_d_);
    if (fft_in_d_) cudaFree(fft_in_d_);
    if (fft_out_d_) cudaFree(fft_out_d_);
    for (auto p : power_accum_d_) { if (p) cudaFree(p); }
    for (auto p : mag_d_) { if (p) cudaFree(p); }
    if (stream_) cudaStreamDestroy(stream_);
  }

  Config cfg_;
  int N_ = 0;
  int A_ = 0;
  float adc_fullscale_ = 32768.0f;

  cudaStream_t stream_ = nullptr;
  cufftHandle fft_plan_ = 0;
  float* window_d_ = nullptr;
  float2* fft_in_d_ = nullptr;
  float2* fft_out_d_ = nullptr;
  std::vector<float*> power_accum_d_;
  std::vector<float*> mag_d_;

  int avg_count_ = 0;
  bool new_result_ = false;
};
