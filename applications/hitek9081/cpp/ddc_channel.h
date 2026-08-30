#pragma once

#include "ddc_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

class DDCChannel {
 public:
  struct Config {
    double center_freq_mhz = 0;     // absolute freq where user clicked
    double lo_freq_mhz = 0;         // ADC center frequency (from CLI --center-freq)
    double bandwidth_mhz = 1.0;     // selected DDC bandwidth
    double sample_rate_mhz = 983.04;
    int input_length = 0;           // raw IQ samples per frame (fft_size)
    int output_length = 4096;       // samples for AMC model
    int antenna = 0;
  };

  struct ClassificationResult {
    int class_id = -1;
    float confidence = 0.0f;
    float top3_confidence[3] = {};
    int top3_ids[3] = {};
  };

  DDCChannel() = default;
  ~DDCChannel() { deactivate(); }

  void activate(const Config& cfg) {
    if (active_) deactivate();
    cfg_ = cfg;

    // NCO frequency: offset from LO to shift target to DC
    double offset_mhz = cfg_.center_freq_mhz - cfg_.lo_freq_mhz;
    freq_offset_norm_ = offset_mhz / cfg_.sample_rate_mhz;

    // Decimation factor: sample_rate / bandwidth → gives output at ~bandwidth rate
    decim_factor_ = std::max(1, static_cast<int>(cfg_.sample_rate_mhz / cfg_.bandwidth_mhz));

    // FIR filter: cutoff at half the DDC bandwidth, normalized to input sample rate
    float cutoff_norm = static_cast<float>(cfg_.bandwidth_mhz / (2.0 * cfg_.sample_rate_mhz));
    num_taps_ = 63;  // odd-length FIR

    cudaStreamCreate(&stream_);

    // Allocate working buffers at input_length (full frame size)
    int N = cfg_.input_length;
    cudaMalloc(&mixed_d_, N * sizeof(float2));
    cudaMalloc(&filtered_d_, N * sizeof(float2));

    // Decimated length
    int decim_len = N / decim_factor_;
    // Output buffer for model (4096 samples, tiled from decimated output)
    cudaMalloc(&iq_out_d_, cfg_.output_length * sizeof(float2));
    cudaMalloc(&decimated_d_, decim_len * sizeof(float2));

    // FIR taps
    cudaMalloc(&taps_d_, num_taps_ * sizeof(float));
    ddc_generate_lowpass_taps(taps_d_, num_taps_, cutoff_norm, stream_);

    decim_output_len_ = decim_len;
    nco_phase_ = 0.0;

    cudaStreamSynchronize(stream_);
    active_ = true;
  }

  void deactivate() {
    if (!active_) return;
    active_ = false;
    classification_ = ClassificationResult{};
    ema_initialized_ = false;
    for (int i = 0; i < 31; i++) ema_probs_[i] = 0;

    if (mixed_d_) { cudaFree(mixed_d_); mixed_d_ = nullptr; }
    if (filtered_d_) { cudaFree(filtered_d_); filtered_d_ = nullptr; }
    if (decimated_d_) { cudaFree(decimated_d_); decimated_d_ = nullptr; }
    if (iq_out_d_) { cudaFree(iq_out_d_); iq_out_d_ = nullptr; }
    if (taps_d_) { cudaFree(taps_d_); taps_d_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
  }

  void process(const int16_t* raw_iq, cudaStream_t src_stream) {
    if (!active_) return;

    int N = cfg_.input_length;

    // Wait for IQ data
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaEventRecord(evt, src_stream);
    cudaStreamWaitEvent(stream_, evt, 0);
    cudaEventDestroy(evt);

    // 1. NCO mix: shift target frequency to DC
    ddc_nco_mix(raw_iq, mixed_d_, N, freq_offset_norm_, nco_phase_, stream_);
    nco_phase_ += 2.0 * M_PI * freq_offset_norm_ * N;
    // Keep phase bounded
    nco_phase_ = fmod(nco_phase_, 2.0 * M_PI);

    // 2. Lowpass filter
    ddc_fir_filter(mixed_d_, filtered_d_, N, taps_d_, num_taps_, stream_);

    // 3. Decimate
    ddc_decimate(filtered_d_, decimated_d_, N, decim_factor_, stream_);

    // 4. Fill output buffer — tile if decimated output < 4096, truncate if >
    int src_len = decim_output_len_;
    if (src_len >= cfg_.output_length) {
      cudaMemcpyAsync(iq_out_d_, decimated_d_, cfg_.output_length * sizeof(float2),
                      cudaMemcpyDeviceToDevice, stream_);
    } else {
      // Tile short output to fill 4096
      ddc_tile_to_length(decimated_d_, iq_out_d_, src_len, cfg_.output_length, stream_);
    }

    // 5. Normalize to unit power
    ddc_normalize_power(iq_out_d_, cfg_.output_length, stream_);
  }

  void set_classification(const ClassificationResult& result) { classification_ = result; }

  float* ema_probs() { return ema_probs_; }
  bool& ema_initialized() { return ema_initialized_; }

  bool is_active() const { return active_; }
  float2* iq_output_gpu() const { return iq_out_d_; }
  int output_length() const { return cfg_.output_length; }
  int antenna() const { return cfg_.antenna; }
  double center_freq_mhz() const { return cfg_.center_freq_mhz; }
  double bandwidth_mhz() const { return cfg_.bandwidth_mhz; }
  int decim_factor() const { return decim_factor_; }
  cudaStream_t stream() const { return stream_; }
  const Config& config() const { return cfg_; }
  const ClassificationResult& classification() const { return classification_; }

 private:
  Config cfg_;
  bool active_ = false;
  double freq_offset_norm_ = 0;
  double nco_phase_ = 0;
  int decim_factor_ = 1;
  int num_taps_ = 63;
  int decim_output_len_ = 0;

  cudaStream_t stream_ = nullptr;
  float2* mixed_d_ = nullptr;
  float2* filtered_d_ = nullptr;
  float2* decimated_d_ = nullptr;
  float2* iq_out_d_ = nullptr;
  float* taps_d_ = nullptr;

  ClassificationResult classification_;
  float ema_probs_[31] = {};
  bool ema_initialized_ = false;
};
