/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// NCO mix: multiply int16 IQ by complex exponential to shift freq_offset_hz to DC.
// in:    int16_t[n*2] interleaved I,Q
// out:   float2[n] complex baseband
// freq_offset_normalized: offset / sample_rate (cycles per sample)
void ddc_nco_mix(const int16_t* in, float2* out, int n,
                 double freq_offset_normalized, double phase_start,
                 cudaStream_t stream);

// Lowpass FIR filter (complex in, complex out).
// Applies real-valued FIR taps to complex signal.
// in:    float2[n]
// out:   float2[n]
// taps:  float[num_taps] (symmetric FIR, pre-uploaded to GPU)
void ddc_fir_filter(const float2* in, float2* out, int n,
                    const float* taps, int num_taps, cudaStream_t stream);

// Decimate: pick every decim_factor-th sample.
// in:    float2[n]
// out:   float2[n / decim_factor]
void ddc_decimate(const float2* in, float2* out, int n, int decim_factor,
                  cudaStream_t stream);

// Tile src[src_len] into dst[dst_len] by repeating (wrapping).
void ddc_tile_to_length(const float2* src, float2* dst, int src_len, int dst_len,
                        cudaStream_t stream);

// Normalize complex IQ to unit power (divide by RMS).
// scratch: pre-allocated float[1] on GPU for power reduction (avoids per-call cudaMalloc).
void ddc_normalize_power(float2* data, int n, float* scratch, cudaStream_t stream);

// Deinterleave float2 → separate I/Q channels for TensorRT: [I0,I1,...,Q0,Q1,...]
void ddc_deinterleave_iq(const float2* src, float* dst, int n, cudaStream_t stream);

// Compute magnitude spectrogram from complex IQ for 2D CNN input.
// in:       float2[n_iq] complex IQ samples
// out:      float[spec_time * spec_freq] normalized magnitude spectrogram
// window:   float[win_len] pre-computed Hanning window (win_len = spec_freq * 2)
void ddc_compute_spectrogram(const float2* in, float* out, int n_iq,
                             const float* window, int win_len,
                             int spec_time, int spec_freq,
                             cudaStream_t stream);

// Generate lowpass FIR taps (windowed sinc) on GPU.
// cutoff_normalized: cutoff_freq / sample_rate (0 to 0.5)
void ddc_generate_lowpass_taps(float* taps, int num_taps, float cutoff_normalized,
                               cudaStream_t stream);
