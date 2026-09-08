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
    y = 1.0f - y;  // Holoviz Y: 0=top, 1=bottom -- invert so dB_max renders at top
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

__global__ void power_to_db_kernel(const float* accum, float* out, int n, int num_averages,
                                   float adc_fullscale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float avg_power = accum[i] / static_cast<float>(num_averages);
    if (avg_power < 1e-40f) avg_power = 1e-40f;  // floor below IEEE denorm to prevent log10(0)
    // ref = N * fullscale * 0.25: N = FFT processing gain, 0.25 = Hanning coherent gain (0.5) * single-sided (0.5)
    float ref = (float)n * adc_fullscale * 0.25f;
    out[i] = 10.0f * log10f(avg_power / (ref * ref));
  }
}

void power_to_db(const float* accum, float* out, int n, int num_averages,
                 float adc_fullscale, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  power_to_db_kernel<<<blocks, threads, 0, stream>>>(accum, out, n, num_averages, adc_fullscale);
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
// Window + DC subtract + int16->float2 (fused second pass)
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
// Window + int16->float2 (no DC subtraction)
// ============================================================================

// iq_swap: 0=I,Q as-is from ADC
//          1=swap Q,I (reversed JESD lane mapping)
//          2=conjugate (negate Q -- upper sideband conversion)
__global__ void window_int16_to_float2_kernel(const int16_t* in, const float* window,
                                              float2* out, int num_samples, int iq_swap) {
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

    out[i].x = val_i * w;
    out[i].y = val_q * w;
  }
}

void apply_window_int16_to_float2(const int16_t* in, const float* window, float2* out,
                                  int num_samples, int iq_swap, cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_samples + threads - 1) / threads;
  window_int16_to_float2_kernel<<<blocks, threads, 0, stream>>>(
      in, window, out, num_samples, iq_swap);
}

// ============================================================================
// Accumulate linear power from float2 FFT output (for FFT averaging)
// ============================================================================

__global__ void accumulate_power_fftshift_f32_kernel(const float2* in, float* accum, int n,
                                                     int reverse) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // fftshift: maps natural order (DC at bin 0) to display order (DC at bin N/2)
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
                                                 int reverse, float adc_fullscale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // fftshift: natural order (DC=0) -> display order (DC=N/2)
    int src = reverse ? (n / 2 - i + n) % n : (i + n / 2) % n;
    float re = in[src].x;
    float im = in[src].y;
    float mag_sq = re * re + im * im;
    if (mag_sq < 1e-40f) mag_sq = 1e-40f;  // floor below IEEE denorm to prevent log10(0)
    // ref = N * fullscale * 0.25: N = FFT gain, 0.25 = Hanning coherent gain (0.5) * single-sided (0.5)
    float ref = (float)n * adc_fullscale * 0.25f;
    out[i] = 10.0f * log10f(mag_sq / (ref * ref));
  }
}

void magnitude_db_fftshift_f32(const float2* in, float* out, int n,
                               int reverse, float adc_fullscale, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  magnitude_db_fftshift_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, n, reverse, adc_fullscale);
}

// ============================================================================
// Interpolate DC bin (cosmetic) -- fills the fftshifted DC region with neighbors
// ============================================================================

__global__ void interpolate_dc_kernel(float* data, int n, int half_width) {
  int center = n / 2;
  int lo = center - half_width - 1;
  int hi = center + half_width + 1;
  if (lo < 0) lo = 0;
  if (hi >= n) hi = n - 1;
  float neighbor_avg = (data[lo] + data[hi]) * 0.5f;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = center - half_width + idx;
  if (idx <= 2 * half_width && bin >= 0 && bin < n) {
    // Only fill if the bin is significantly below neighbors (DC notch).
    // If there's a real signal at DC, leave it alone.
    if (data[bin] < neighbor_avg - 6.0f) {
      data[bin] = neighbor_avg;
    }
  }
}

void interpolate_dc_bin(float* data, int n, int half_width, cudaStream_t stream) {
  int count = 2 * half_width + 1;
  interpolate_dc_kernel<<<1, count, 0, stream>>>(data, n, half_width);
}

// ============================================================================
// Peak-downsample: reduce N bins to out_bins by taking max per stride
// ============================================================================

// Peak (max-hold) downsample -- preserves signal peaks in display.
// Uses float stride so output bins map uniformly to the full input range.
// This ensures freq_axis[i] = freq_min + i * (freq_max - freq_min) / (out_bins - 1)
// correctly maps to the downsampled output.
__global__ void downsample_peak_kernel(const float* in, float* out,
                                       int in_bins, int out_bins, int bin_offset,
                                       float stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_bins) return;

  int start = bin_offset + static_cast<int>(i * stride);
  int end = bin_offset + static_cast<int>((i + 1) * stride);
  if (start >= bin_offset + in_bins) start = bin_offset + in_bins - 1;
  if (end > bin_offset + in_bins) end = bin_offset + in_bins;

  float peak = in[start];
  for (int j = start + 1; j < end; j++) {
    if (in[j] > peak) peak = in[j];
  }
  out[i] = peak;
}

void downsample_peak(const float* in, float* out, int in_bins, int out_bins,
                     int bin_offset, cudaStream_t stream) {
  float stride = static_cast<float>(in_bins) / static_cast<float>(out_bins);
  int threads = 256;
  int blocks = (out_bins + threads - 1) / threads;
  downsample_peak_kernel<<<blocks, threads, 0, stream>>>(in, out, in_bins, out_bins, bin_offset, stride);
}

// ============================================================================
// Gather IQ from strided batch into contiguous accumulation buffer
//
// iq_format:
//   0 = pair-interleaved: [I0,Q0,I1,Q1,I2,Q2,...] (2 int16 per sample)
//   1 = block-interleaved groups of 4: [I0,I1,I2,I3,Q0,Q1,Q2,Q3,...] (8 int16 per 4 samples)
// ============================================================================

__global__ void gather_iq_kernel(const uint8_t* src, int16_t* dst,
                                 int num_pkts, int samples_per_pkt,
                                 int stride_bytes, int dst_sample_offset,
                                 int iq_format) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_samples = num_pkts * samples_per_pkt;
  if (idx >= total_samples) return;

  int pkt = idx / samples_per_pkt;
  int sample_in_pkt = idx % samples_per_pkt;

  const int16_t* pkt_start = reinterpret_cast<const int16_t*>(src + pkt * stride_bytes);

  int16_t i_val, q_val;

  if (iq_format == 1) {
    int group = sample_in_pkt / 4;
    int pos   = sample_in_pkt % 4;
    i_val = pkt_start[group * 8 + pos];
    q_val = pkt_start[group * 8 + pos + 4];
  } else {
    i_val = pkt_start[sample_in_pkt * 2];
    q_val = pkt_start[sample_in_pkt * 2 + 1];
  }

  int dst_idx = (dst_sample_offset + idx) * 2;
  dst[dst_idx] = i_val;
  dst[dst_idx + 1] = q_val;
}

void gather_iq_from_batch(const void* src, int16_t* dst,
                          int num_pkts, int samples_per_pkt,
                          int stride_bytes, int dst_sample_offset,
                          int iq_format, cudaStream_t stream) {
  int total_samples = num_pkts * samples_per_pkt;
  int threads = 256;
  int blocks = (total_samples + threads - 1) / threads;
  gather_iq_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(src), dst,
      num_pkts, samples_per_pkt, stride_bytes, dst_sample_offset, iq_format);
}

// ============================================================================
// Scatter-gather: read IQ from an array of per-packet pointers (zero-copy path)
//
// Each pkt_ptrs[i] points directly into DAQIRI's host_pinned buffer pool.
// On unified-memory systems (DGX Spark), GPU kernels can dereference these
// pointers without any cudaMemcpy -- CPU and GPU share the same LPDDR5 RAM.
// ============================================================================

__global__ void gather_iq_scattered_kernel(
    void* const* pkt_ptrs, int16_t* dst,
    int num_pkts, int samples_per_pkt,
    int dst_sample_offset, int max_dst_samples, int iq_format) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_samples = num_pkts * samples_per_pkt;
  if (idx >= total_samples) return;
  if (dst_sample_offset + idx >= max_dst_samples) return;

  int pkt = idx / samples_per_pkt;
  int sample_in_pkt = idx % samples_per_pkt;

  const int16_t* pkt_data = static_cast<const int16_t*>(pkt_ptrs[pkt]);

  int16_t i_val, q_val;

  if (iq_format == 1) {
    int group = sample_in_pkt / 4;
    int pos   = sample_in_pkt % 4;
    i_val = pkt_data[group * 8 + pos];
    q_val = pkt_data[group * 8 + pos + 4];
  } else {
    i_val = pkt_data[sample_in_pkt * 2];
    q_val = pkt_data[sample_in_pkt * 2 + 1];
  }

  int dst_idx = (dst_sample_offset + idx) * 2;
  dst[dst_idx] = i_val;
  dst[dst_idx + 1] = q_val;
}

void gather_iq_from_scattered(void* const* pkt_ptrs, int16_t* dst,
                              int num_pkts, int samples_per_pkt,
                              int dst_sample_offset, int max_dst_samples,
                              int iq_format,
                              cudaStream_t stream) {
  int total_samples = num_pkts * samples_per_pkt;
  int threads = 256;
  int blocks = (total_samples + threads - 1) / threads;
  gather_iq_scattered_kernel<<<blocks, threads, 0, stream>>>(
      pkt_ptrs, dst, num_pkts, samples_per_pkt, dst_sample_offset, max_dst_samples, iq_format);
}
