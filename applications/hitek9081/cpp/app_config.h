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

#include <cstdint>
#include <vector>

struct AppConfig {
  double sr_mhz = 0;
  double center_freq_mhz = 0;
  int fft_size = 65536;
  int num_antennas = 1;
  int num_averages = 1;
  float db_min = -120.0f;
  float db_max = 0.0f;
  double freq_min_mhz = 0;
  double freq_max_mhz = 0;
  int iq_swap = 1;

  int iq_format = 1;  // 0=pair-interleaved [I,Q,I,Q,...], 1=block-4 [IIII,QQQQ,...]
#ifdef HAS_IMGUI
  bool use_imgui = true;
#else
  bool use_imgui = false;
#endif
  bool waterfall = false;

  // Computed from hsb config
  uint16_t max_packet_size = 1482;
  uint16_t header_size = 70;
  int samples_per_antenna = 992;
  uint32_t data_port = 4096;
  int batch_size = 20480;

  // Frequency tick positions for Holoviz grid
  std::vector<float> tick_x_norm;
  std::vector<double> tick_freqs_mhz;
};
