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

#include "spectral_kernels.cuh"
#include <cstdint>
#include <math_constants.h>

// ============================================================================
// Hanning window generation
// ============================================================================

__global__ void hanning_window_kernel(float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 0.5f * (1.0f - cosf(2.0f * CUDART_PI_F * i / n));
  }
}

void generate_hanning_window(float* out, int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hanning_window_kernel<<<blocks, threads, 0, stream>>>(out, n);
}



// ============================================================================
// Normalize to [0,1] coordinate pairs for Holoviz line plot
// ============================================================================

__global__ void normalize_to_coords_kernel(const float* in, float* out,
                                           int bin_start, int num_bins,
                                           float db_min, float db_max,
                                           int flip_x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_bins) {
    float x = static_cast<float>(i) / static_cast<float>(num_bins - 1);
    if (flip_x) x = 1.0f - x;
    float y = (in[bin_start + i] - db_min) / (db_max - db_min);
    y = fminf(fmaxf(y, 0.0f), 1.0f);
    y = 1.0f - y;
    out[2 * i] = x;
    out[2 * i + 1] = y;
  }
}

void normalize_to_coords(const float* in, float* out, int n,
                         float db_min, float db_max,
                         int bin_start, int bin_end,
                         int flip_x, cudaStream_t stream) {
  int num_bins = bin_end - bin_start;
  int threads = 256;
  int blocks = (num_bins + threads - 1) / threads;
  normalize_to_coords_kernel<<<blocks, threads, 0, stream>>>(in, out, bin_start, num_bins, db_min, db_max, flip_x);
}


// ============================================================================
// Convert accumulated power to dB (finalize averaging)
// ============================================================================

__global__ void power_to_db_kernel(const float* accum, float* out, int n, int num_averages) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float avg_power = accum[i] / static_cast<float>(num_averages);
    if (avg_power < 1e-40f) avg_power = 1e-40f;
    float ref = (float)n * 32768.0f * 0.5f;
    out[i] = 10.0f * log10f(avg_power / (ref * ref));
  }
}

void power_to_db(const float* accum, float* out, int n, int num_averages,
                 cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  power_to_db_kernel<<<blocks, threads, 0, stream>>>(accum, out, n, num_averages);
}

// ============================================================================
// FP16: DC mean reduction (parallel sum of int16 IQ block)
// ============================================================================

__global__ void dc_mean_reduce_kernel(const int16_t* in, float* mean_iq, int num_samples,
                                      int iq_swap) {
  __shared__ float sI[256];
  __shared__ float sQ[256];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float local_i = 0.0f, local_q = 0.0f;
  if (i < num_samples) {
    float first  = static_cast<float>(in[2 * i]);
    float second = static_cast<float>(in[2 * i + 1]);
    if (iq_swap == 1) {
      local_i = second;
      local_q = first;
    } else if (iq_swap == 2) {
      local_i = first;
      local_q = -second;
    } else {
      local_i = first;
      local_q = second;
    }
  }

  sI[tid] = local_i;
  sQ[tid] = local_q;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sI[tid] += sI[tid + s];
      sQ[tid] += sQ[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&mean_iq[0], sI[0]);
    atomicAdd(&mean_iq[1], sQ[0]);
  }
}

void dc_mean_int16(const int16_t* in, float* mean_iq, int num_samples,
                   int iq_swap, cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_samples + threads - 1) / threads;
  dc_mean_reduce_kernel<<<blocks, threads, 0, stream>>>(in, mean_iq, num_samples, iq_swap);
}

// ============================================================================
// Window + DC subtract + int16→float2 (fused second pass)
// ============================================================================

__global__ void window_dcblock_int16_to_float2_kernel(const int16_t* in, const float* window,
                                                      const float* mean_iq, float2* out,
                                                      int num_samples, int iq_swap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_samples) {
    float w = window[i];
    float first  = static_cast<float>(in[2 * i]);
    float second = static_cast<float>(in[2 * i + 1]);

    float val_i, val_q;
    if (iq_swap == 1) {
      val_i = second;
      val_q = first;
    } else if (iq_swap == 2) {
      val_i = first;
      val_q = -second;
    } else {
      val_i = first;
      val_q = second;
    }

    float mean_i = mean_iq[0] / static_cast<float>(num_samples);
    float mean_q = mean_iq[1] / static_cast<float>(num_samples);

    out[i].x = (val_i - mean_i) * w;
    out[i].y = (val_q - mean_q) * w;
  }
}

void apply_window_dcblock_int16_to_float2(const int16_t* in, const float* window,
                                          const float* mean_iq, float2* out,
                                          int num_samples, int iq_swap,
                                          cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_samples + threads - 1) / threads;
  window_dcblock_int16_to_float2_kernel<<<blocks, threads, 0, stream>>>(
      in, window, mean_iq, out, num_samples, iq_swap);
}

// ============================================================================
// Accumulate linear power from float2 FFT output (for FFT averaging)
// ============================================================================

__global__ void accumulate_power_fftshift_f32_kernel(const float2* in, float* accum, int n,
                                                     int reverse) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int src = reverse ? (n / 2 - i + n) % n : (i + n / 2) % n;
    float re = in[src].x;
    float im = in[src].y;
    accum[i] += re * re + im * im;
  }
}

void accumulate_power_fftshift_f32(const float2* in, float* accum, int n,
                                   int reverse, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  accumulate_power_fftshift_f32_kernel<<<blocks, threads, 0, stream>>>(in, accum, n, reverse);
}

// ============================================================================
// Magnitude in dB with fftshift from float2 FFT output
// ============================================================================

__global__ void magnitude_db_fftshift_f32_kernel(const float2* in, float* out, int n,
                                                 int reverse) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int src = reverse ? (n / 2 - i + n) % n : (i + n / 2) % n;
    float re = in[src].x;
    float im = in[src].y;
    float mag_sq = re * re + im * im;
    if (mag_sq < 1e-40f) mag_sq = 1e-40f;
    float ref = (float)n * 32768.0f * 0.5f;
    out[i] = 10.0f * log10f(mag_sq / (ref * ref));
  }
}

void magnitude_db_fftshift_f32(const float2* in, float* out, int n,
                               int reverse, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  magnitude_db_fftshift_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, n, reverse);
}

// ============================================================================
// Gather IQ from strided batch into contiguous accumulation buffer
// ============================================================================

__global__ void gather_iq_kernel(const uint8_t* src, int16_t* dst,
                                 int num_pkts, int samples_per_pkt,
                                 int stride_bytes, int dst_sample_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_samples = num_pkts * samples_per_pkt;
  if (idx >= total_samples) return;

  int pkt = idx / samples_per_pkt;
  int sample_in_pkt = idx % samples_per_pkt;

  // Source: pkt * stride_bytes + sample_in_pkt * 4 bytes (2x int16)
  const int16_t* pkt_start = reinterpret_cast<const int16_t*>(src + pkt * stride_bytes);
  int16_t i_val = pkt_start[sample_in_pkt * 2];
  int16_t q_val = pkt_start[sample_in_pkt * 2 + 1];

  // Destination: contiguous [I,Q,I,Q,...] starting at dst_sample_offset
  int dst_idx = (dst_sample_offset + idx) * 2;
  dst[dst_idx] = i_val;
  dst[dst_idx + 1] = q_val;
}

void gather_iq_from_batch(const void* src, int16_t* dst,
                          int num_pkts, int samples_per_pkt,
                          int stride_bytes, int dst_sample_offset,
                          cudaStream_t stream) {
  int total_samples = num_pkts * samples_per_pkt;
  int threads = 256;
  int blocks = (total_samples + threads - 1) / threads;
  gather_iq_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(src), dst,
      num_pkts, samples_per_pkt, stride_bytes, dst_sample_offset);
}
