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

#include "ddc_kernels.cuh"

// =============================================================================
// DDC kernel implementations — STUB
// =============================================================================

void ddc_extract_subband(const cufftComplex* /*in*/, cufftComplex* /*out*/,
                         int /*center_bin*/, int /*num_bins*/, int /*fft_size*/,
                         cudaStream_t /*stream*/) {
  // TODO: implement frequency-domain extraction
  //
  // __global__ void extract_subband_kernel(...)
  // {
  //   int i = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (i >= num_bins) return;
  //
  //   // Map output bin i to input FFT bin (with wraparound for negative freqs)
  //   int src_bin = (center_bin - num_bins/2 + i + fft_size) % fft_size;
  //   out[i] = in[src_bin];
  // }
}
