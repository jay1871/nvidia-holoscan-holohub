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
// NCO downconversion: multiply by exp(-j * 2 * pi * f_norm * n) to shift target to DC.
// Negative sign = downconversion convention (positive freq -> baseband).
// Phase accumulates across frames to maintain continuity.
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
// FIR lowpass filter (complex signal, real taps).
// Uses shared memory tile with halo for input overlap -- reduces global reads ~30x.
// Taps loaded into shared memory once per block (63 floats = 252 bytes).
// Boundary: truncation at edges (not circular). Acceptable for long signals.
// =============================================================================

static constexpr int FIR_BLOCK = 256;
static constexpr int FIR_MAX_TAPS = 127;

__global__ void fir_filter_kernel(const float2* in, float2* out, int n,
                                  const float* taps, int num_taps) {
  __shared__ float s_taps[FIR_MAX_TAPS];
  __shared__ float2 s_tile[FIR_BLOCK + FIR_MAX_TAPS];  // block + halo

  int tid = threadIdx.x;
  int gid = blockIdx.x * FIR_BLOCK + tid;
  int half = num_taps / 2;

  // Load taps into shared memory (all threads cooperate)
  if (tid < num_taps) s_taps[tid] = taps[tid];
  __syncthreads();

  // Load tile: main block + halo on both sides
  int tile_start = blockIdx.x * FIR_BLOCK - half;
  int tile_len = FIR_BLOCK + num_taps - 1;
  for (int i = tid; i < tile_len; i += FIR_BLOCK) {
    int gi = tile_start + i;
    if (gi >= 0 && gi < n)
      s_tile[i] = in[gi];
    else
      s_tile[i] = make_float2(0.0f, 0.0f);
  }
  __syncthreads();

  if (gid >= n) return;

  float sum_i = 0.0f, sum_q = 0.0f;
  for (int t = 0; t < num_taps; t++) {
    float2 val = s_tile[tid + t];
    float tap = s_taps[t];
    sum_i += val.x * tap;
    sum_q += val.y * tap;
  }

  out[gid].x = sum_i;
  out[gid].y = sum_q;
}

void ddc_fir_filter(const float2* in, float2* out, int n,
                    const float* taps, int num_taps, cudaStream_t stream) {
  int blocks = (n + FIR_BLOCK - 1) / FIR_BLOCK;
  fir_filter_kernel<<<blocks, FIR_BLOCK, 0, stream>>>(in, out, n, taps, num_taps);
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

void ddc_normalize_power(float2* data, int n, float* scratch, cudaStream_t stream) {
  cudaMemsetAsync(scratch, 0, sizeof(float), stream);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  compute_power_kernel<<<blocks, threads, 0, stream>>>(data, scratch, n);

  // CPU round-trip to read scalar power -- unavoidable without a two-kernel device-side approach.
  // Acceptable at ~7 Hz DDC frame rate; total stall <0.1ms per channel.
  float total_power;
  cudaMemcpyAsync(&total_power, scratch, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float rms = sqrtf(total_power / n);
  if (rms < 1e-10f) rms = 1e-10f;
  float inv_rms = 1.0f / rms;

  normalize_kernel<<<blocks, threads, 0, stream>>>(data, n, inv_rms);
}

// =============================================================================
// Deinterleave float2 -> [I,I,...,Q,Q,...] for TensorRT
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
