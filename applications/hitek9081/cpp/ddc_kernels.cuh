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
#include <cufft.h>
#include <cstdint>

// =============================================================================
// DDC (Digital Down-Converter) CUDA Kernels — PLACEHOLDER
// =============================================================================
//
// Future algorithm:
//   1. FFT the input block (reuse cuFFT plan, same fft_size as spectral path)
//   2. Circular-shift to move center_frequency to DC (baseband)
//   3. Zero bins outside ±bandwidth/2 (brick-wall filter in freq domain)
//      OR: multiply by a raised-cosine window for smoother rolloff
//   4. IFFT the sub-band → narrowband time-domain IQ
//   5. Decimate by ratio = fft_size / num_extracted_bins
//   6. Output: float32 complex IQ at reduced sample rate
//
// Performance considerations:
//   - At ~16 Hz input rate (65536-pt blocks), even a full FFT+IFFT is trivial
//   - Can batch multiple antennas into a single cufftPlanMany call
//   - Overlap-save method could be used for continuous streaming without gaps
// =============================================================================

// Extract frequency bins around a center bin, applying a window for rolloff.
// in:       cufftComplex[fft_size] (FFT output, not fftshifted)
// out:      cufftComplex[num_bins] (extracted sub-band, shifted to baseband)
// center_bin: which bin corresponds to the desired center frequency
// num_bins:   number of bins to extract (= bandwidth / bin_resolution)
// fft_size:   total FFT length
//
// STUB — not yet implemented.
void ddc_extract_subband(const cufftComplex* in, cufftComplex* out,
                         int center_bin, int num_bins, int fft_size,
                         cudaStream_t stream);
