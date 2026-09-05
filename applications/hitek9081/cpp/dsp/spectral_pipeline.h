// Windowed FFT → magnitude pipeline. No display dependencies.
// Operates entirely on GPU via a dedicated CUDA stream.
//
// Pipeline per frame:
//   int16 IQ → Hanning window → cuFFT (fp32 batched) → |X|² accumulation or dB conversion
//
// Averaging accumulates |FFT[k]|² in linear power domain (not dB) to preserve
// noise statistics. After num_averages frames: 10*log10(accum/N) → dBFS.
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
    int iq_swap = 0;  // 0=I,Q as-is  1=swap (reversed JESD lanes)  2=conjugate (upper sideband)
  };

  SpectralPipeline() = default;
  ~SpectralPipeline() { free(); }

  void initialize(const Config& cfg) {
    cfg_ = cfg;
    N_ = cfg.fft_size;   // FFT length in complex samples
    A_ = cfg.num_antennas;
    adc_fullscale_ = static_cast<float>(1 << (cfg.adc_bits - 1));  // 2^(bits-1), e.g. 32768 for 16-bit

    cudaStreamCreate(&stream_);

    cudaMalloc(&window_d_, N_ * sizeof(float));
    generate_hanning_window(window_d_, N_, stream_);

    cudaMalloc(&fft_in_d_, A_ * N_ * sizeof(float2));
    // Natural FFT order: bin 0 = DC. Used by DDC for subband extraction.
    cudaMalloc(&fft_out_d_, A_ * N_ * sizeof(float2));

    power_accum_d_.resize(A_, nullptr);
    mag_d_.resize(A_, nullptr);
    for (int a = 0; a < A_; a++) {
      cudaMalloc(&power_accum_d_[a], N_ * sizeof(float));
      cudaMemsetAsync(power_accum_d_[a], 0, N_ * sizeof(float), stream_);
      cudaMalloc(&mag_d_[a], N_ * sizeof(float));
    }

    // Batched FFT: all antennas in one call. fp32 chosen over fp16 because
    // 16+ butterfly stages at N=65536 exceed fp16's 10-bit mantissa.
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
        // Accumulate |FFT[k]|² — linear power preserves noise statistics for averaging
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
  // mag_d_: fftshifted dBFS magnitude. bin 0 = most-negative freq, bin N/2 = DC.
  float* mag_gpu(int antenna) const { return mag_d_[antenna]; }
  // fft_out_d_: raw FFT in natural order (bin 0 = DC). Used by DDC for subband extraction.
  float2* fft_out_gpu(int antenna) const { return fft_out_d_ + antenna * N_; }
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
  int N_ = 0;  // FFT size (complex samples)
  int A_ = 0;  // antenna count
  float adc_fullscale_ = 32768.0f;  // 2^(adc_bits-1), overwritten by initialize()

  cudaStream_t stream_ = nullptr;
  cufftHandle fft_plan_ = 0;
  float* window_d_ = nullptr;
  float2* fft_in_d_ = nullptr;
  float2* fft_out_d_ = nullptr;         // natural order (bin 0 = DC)
  std::vector<float*> power_accum_d_;   // linear |X|² accumulator per antenna
  std::vector<float*> mag_d_;           // fftshifted dBFS output per antenna

  int avg_count_ = 0;
  bool new_result_ = false;
};
