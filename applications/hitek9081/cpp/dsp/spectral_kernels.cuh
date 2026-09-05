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

#pragma once

#include <cuda_runtime.h>

// Generate Hanning window coefficients on GPU.
// out: float array of length n
void generate_hanning_window(float* out, int n, cudaStream_t stream);


// Normalize dB values to [0,1] range and pack as (x,y) coordinate pairs for Holoviz.
// Only bins in [bin_start, bin_end) are mapped to x=[0,1].
// in:        float array of length n (magnitude in dB)
// out:       float array of length n*2 (interleaved x,y pairs)
// db_min:    minimum dB value mapped to y=1 (bottom)
// db_max:    maximum dB value mapped to y=0 (top)
// bin_start: first bin to display (inclusive)
// bin_end:   last bin to display (exclusive)
// flip_x:   if nonzero, reverse x direction (for spectrum mirroring with iq_swap)
void normalize_to_coords(const float* in, float* out, int n,
                         float db_min, float db_max,
                         int bin_start, int bin_end,
                         int flip_x, cudaStream_t stream);

// Finalize averaged power to dB: out[i] = 10*log10(accum[i] / num_averages).
// Floors at -200 dB to avoid log(0).
// accum: float[n] (accumulated power sum)
// out:   float[n] (magnitude in dB)
void power_to_db(const float* accum, float* out, int n, int num_averages,
                 float adc_fullscale, cudaStream_t stream);

// ---- Batched fp32 FFT pipeline kernels ----

// Compute per-block DC mean of interleaved int16 IQ samples (parallel reduction).
// in:  int16_t pairs of length num_samples*2
// mean_iq: float[2] output = {mean_I, mean_Q} (must be zeroed before call)
void dc_mean_int16(const int16_t* in, float* mean_iq, int num_samples,
                   int iq_swap, cudaStream_t stream);

// Window + DC mean subtraction + int16→float2 conversion in one pass.
// Subtracts mean_iq from each sample before windowing.
void apply_window_dcblock_int16_to_float2(const int16_t* in, const float* window,
                                          const float* mean_iq, float2* out,
                                          int num_samples, int iq_swap,
                                          cudaStream_t stream);

// Window + int16→float2 (no DC subtraction).
void apply_window_int16_to_float2(const int16_t* in, const float* window, float2* out,
                                  int num_samples, int iq_swap, cudaStream_t stream);

// Accumulate |FFT[bin]|² from float2 FFT output into fp32 power buffer (fftshifted).
// in:    float2[n] (FFT output, natural order)
// accum: float[n] (running sum, fftshifted indexing)
void accumulate_power_fftshift_f32(const float2* in, float* accum, int n,
                                   int reverse, cudaStream_t stream);

// Compute magnitude-squared in dB with fftshift from float2 input.
// in:  float2[n] (FFT output, natural order)
// out: float[n] (dB values, fftshifted)
void magnitude_db_fftshift_f32(const float2* in, float* out, int n,
                               int reverse, float adc_fullscale, cudaStream_t stream);

// Interpolate DC bin region for cosmetic display (removes DC notch artifact).
// Replaces bins [n/2 - half_width, n/2 + half_width] with average of neighbors.
void interpolate_dc_bin(float* data, int n, int half_width, cudaStream_t stream);

// Peak-downsample: reduce in_bins to out_bins (peak per stride). Reads from in+bin_offset.
void downsample_peak(const float* in, float* out, int in_bins, int out_bins,
                     int bin_offset, cudaStream_t stream);

// ---- Strided batch gather ----

// Gather IQ samples from a strided batch buffer into a contiguous accumulation buffer.
// Batch layout: num_pkts packets, each stride_bytes apart, IQ data at offset 0 of each.
// Extracts samples_per_pkt complex samples (= samples_per_pkt*2 int16 values) per packet.
// src:             batch buffer start (GPU)
// dst:             accumulation buffer destination (GPU, offset by dst_sample_offset)
// num_pkts:        packets to extract from this batch
// samples_per_pkt: complex samples per packet (e.g., 976)
// stride_bytes:    distance between packet starts in the batch (nom_payload_size)
// dst_sample_offset: sample index in dst where first packet's data goes
// iq_format: 0=pair-interleaved [I,Q,I,Q,...], 1=block-4 [IIII,QQQQ,...]
void gather_iq_from_batch(const void* src, int16_t* dst,
                          int num_pkts, int samples_per_pkt,
                          int stride_bytes, int dst_sample_offset,
                          int iq_format, cudaStream_t stream);

// Zero-copy scatter-gather: reads from per-packet pointers in DAQIRI's buffer pool.
// pkt_ptrs: array of payload pointers (host_pinned, GPU-accessible on unified memory)
void gather_iq_from_scattered(void* const* pkt_ptrs, int16_t* dst,
                              int num_pkts, int samples_per_pkt,
                              int dst_sample_offset, int max_dst_samples,
                              int iq_format,
                              cudaStream_t stream);
