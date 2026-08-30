/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ddc_kernels.cuh"
#include <math_constants.h>
#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
// NCO mix: shift signal to baseband
// =============================================================================

__global__ void nco_mix_kernel(const int16_t* in, float2* out, int n,
                               double freq_norm, double phase_start) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float I = static_cast<float>(in[2 * i]);
  float Q = static_cast<float>(in[2 * i + 1]);

  double phase = phase_start + 2.0 * CUDART_PI * freq_norm * i;
  float cos_p = __cosf(static_cast<float>(phase));
  float sin_p = __sinf(static_cast<float>(phase));

  // Complex multiply: (I + jQ) * (cos - j*sin) = (I*cos + Q*sin) + j(Q*cos - I*sin)
  out[i].x = I * cos_p + Q * sin_p;
  out[i].y = Q * cos_p - I * sin_p;
}

void ddc_nco_mix(const int16_t* in, float2* out, int n,
                 double freq_offset_normalized, double phase_start,
                 cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  nco_mix_kernel<<<blocks, threads, 0, stream>>>(in, out, n,
      freq_offset_normalized, phase_start);
}

// =============================================================================
// FIR lowpass filter (complex signal, real taps)
// =============================================================================

__global__ void fir_filter_kernel(const float2* in, float2* out, int n,
                                  const float* taps, int num_taps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int half = num_taps / 2;
  float sum_i = 0, sum_q = 0;

  for (int t = 0; t < num_taps; t++) {
    int idx = i - half + t;
    if (idx >= 0 && idx < n) {
      sum_i += in[idx].x * taps[t];
      sum_q += in[idx].y * taps[t];
    }
  }

  out[i].x = sum_i;
  out[i].y = sum_q;
}

void ddc_fir_filter(const float2* in, float2* out, int n,
                    const float* taps, int num_taps, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fir_filter_kernel<<<blocks, threads, 0, stream>>>(in, out, n, taps, num_taps);
}

// =============================================================================
// Decimate
// =============================================================================

__global__ void decimate_kernel(const float2* in, float2* out, int out_len, int decim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len) return;
  out[i] = in[i * decim];
}

void ddc_decimate(const float2* in, float2* out, int n, int decim_factor,
                  cudaStream_t stream) {
  int out_len = n / decim_factor;
  int threads = 256;
  int blocks = (out_len + threads - 1) / threads;
  decimate_kernel<<<blocks, threads, 0, stream>>>(in, out, out_len, decim_factor);
}

// =============================================================================
// Tile short buffer to fill longer output (repeat/wrap)
// =============================================================================

__global__ void tile_kernel(const float2* src, float2* dst, int src_len, int dst_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dst_len) return;
  dst[i] = src[i % src_len];
}

void ddc_tile_to_length(const float2* src, float2* dst, int src_len, int dst_len,
                        cudaStream_t stream) {
  int threads = 256;
  int blocks = (dst_len + threads - 1) / threads;
  tile_kernel<<<blocks, threads, 0, stream>>>(src, dst, src_len, dst_len);
}

// =============================================================================
// Power normalization
// =============================================================================

__global__ void compute_power_kernel(const float2* data, float* power_out, int n) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float p = 0.0f;
  if (i < n) {
    p = data[i].x * data[i].x + data[i].y * data[i].y;
  }
  sdata[tid] = p;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(power_out, sdata[0]);
}

__global__ void normalize_kernel(float2* data, int n, float inv_rms) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  data[i].x *= inv_rms;
  data[i].y *= inv_rms;
}

void ddc_normalize_power(float2* data, int n, cudaStream_t stream) {
  float* power_d;
  cudaMalloc(&power_d, sizeof(float));
  cudaMemsetAsync(power_d, 0, sizeof(float), stream);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  compute_power_kernel<<<blocks, threads, 0, stream>>>(data, power_d, n);

  float total_power;
  cudaMemcpyAsync(&total_power, power_d, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float rms = sqrtf(total_power / n);
  if (rms < 1e-10f) rms = 1e-10f;
  float inv_rms = 1.0f / rms;

  normalize_kernel<<<blocks, threads, 0, stream>>>(data, n, inv_rms);
  cudaFree(power_d);
}

// =============================================================================
// Deinterleave float2 → [I,I,...,Q,Q,...] for TensorRT
// =============================================================================

__global__ void deinterleave_iq_kernel(const float2* src, float* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[i].x;
  dst[n + i] = src[i].y;
}

void ddc_deinterleave_iq(const float2* src, float* dst, int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deinterleave_iq_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// =============================================================================
// Spectrogram: STFT magnitude for 2D CNN input
// =============================================================================

__global__ void spectrogram_kernel(const float2* in, float* out, int n_iq,
                                   const float* window, int win_len,
                                   int spec_time, int spec_freq) {
  // Each thread computes one (time, freq) bin
  int t = blockIdx.x;
  int f = threadIdx.x;
  if (t >= spec_time || f >= spec_freq) return;

  int hop = n_iq / spec_time;
  int start = t * hop;
  int half_freq = spec_freq / 2;

  // Compute one FFT bin via DFT (small spec_freq, GPU-parallel across freq)
  // fftshift: map output bin f to DFT bin k
  int k = f - half_freq;

  float sum_re = 0, sum_im = 0;
  for (int n = 0; n < win_len && (start + n) < n_iq; n++) {
    float w = window[n];
    float re = in[start + n].x * w;
    float im = in[start + n].y * w;

    float phase = -2.0f * CUDART_PI_F * k * n / win_len;
    float cos_p = __cosf(phase);
    float sin_p = __sinf(phase);

    sum_re += re * cos_p - im * sin_p;
    sum_im += re * sin_p + im * cos_p;
  }

  float mag = sqrtf(sum_re * sum_re + sum_im * sum_im);
  mag = fmaxf(mag, 1e-10f);
  out[t * spec_freq + f] = 20.0f * log10f(mag);
}

__global__ void normalize_spectrogram_kernel(float* data, int total, float min_val, float range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  data[i] = (data[i] - min_val) / range;
}

void ddc_compute_spectrogram(const float2* in, float* out, int n_iq,
                             const float* window, int win_len,
                             int spec_time, int spec_freq,
                             cudaStream_t stream) {
  // One block per time step, spec_freq threads per block
  spectrogram_kernel<<<spec_time, spec_freq, 0, stream>>>(
      in, out, n_iq, window, win_len, spec_time, spec_freq);

  // Find min/max for normalization (small buffer — do on CPU)
  int total = spec_time * spec_freq;
  std::vector<float> h_spec(total);
  cudaMemcpyAsync(h_spec.data(), out, total * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float min_val = *std::min_element(h_spec.begin(), h_spec.end());
  float max_val = *std::max_element(h_spec.begin(), h_spec.end());
  float range = max_val - min_val;
  if (range < 1e-6f) range = 1.0f;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  normalize_spectrogram_kernel<<<blocks, threads, 0, stream>>>(out, total, min_val, range);
}

// =============================================================================
// Generate lowpass FIR taps (windowed sinc, Blackman window)
// =============================================================================

__global__ void lowpass_taps_kernel(float* taps, int num_taps, float cutoff_norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_taps) return;

  int M = num_taps - 1;
  float n = static_cast<float>(i) - static_cast<float>(M) / 2.0f;

  // Sinc
  float h;
  if (fabsf(n) < 1e-6f) {
    h = 2.0f * cutoff_norm;
  } else {
    h = sinf(2.0f * CUDART_PI_F * cutoff_norm * n) / (CUDART_PI_F * n);
  }

  // Blackman window
  float w = 0.42f - 0.5f * cosf(2.0f * CUDART_PI_F * i / M)
                   + 0.08f * cosf(4.0f * CUDART_PI_F * i / M);

  taps[i] = h * w;
}

void ddc_generate_lowpass_taps(float* taps, int num_taps, float cutoff_normalized,
                               cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_taps + threads - 1) / threads;
  lowpass_taps_kernel<<<blocks, threads, 0, stream>>>(taps, num_taps, cutoff_normalized);
}
