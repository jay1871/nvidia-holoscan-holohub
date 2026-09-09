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

#include "ddc_channel.h"
#include "amc_inference.h"
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

/**
 * @brief Thread-safe manager for multi-channel Digital Down-Converters (DDC)
 *        and Automatic Modulation Classification (AMC) inference.
 *
 * Decouples the UI sink (ImGuiSinkOp) from DSP operators (SpectralOp / DDCBankOp).
 */
class DDCManager {
 public:
  static constexpr int NUM_SLOTS = 16;
  static constexpr double AMC_INFER_INTERVAL_SEC = 0.066;  // ~15 Hz update rate

  DDCManager() = default;
  ~DDCManager() = default;

  bool initialize_amc(const std::string& model_path, int input_len = 4096) {
    if (model_path.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return amc_inference_.initialize(model_path, input_len);
  }

  void activate_channel(int slot, const DDCChannel::Config& cfg) {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      channels_[slot].activate(cfg);
      last_infer_time_[slot] = {};  // immediate inference on first frame
    }
  }

  void deactivate_channel(int slot) {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      channels_[slot].deactivate();
    }
  }

  bool is_active(int slot) const {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      return channels_[slot].is_active();
    }
    return false;
  }

  double channel_bw_mhz(int slot) const {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      return channels_[slot].bandwidth_mhz();
    }
    return 0.0;
  }

  double channel_center_mhz(int slot) const {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      return channels_[slot].center_freq_mhz();
    }
    return 0.0;
  }

  DDCChannel::ClassificationResult get_classification(int slot) const {
    if (slot >= 0 && slot < NUM_SLOTS) {
      std::lock_guard<std::mutex> lock(mutex_);
      return channels_[slot].classification();
    }
    return {};
  }

  void process(const int16_t* const* iq_ptrs, int num_antennas, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    for (int s = 0; s < NUM_SLOTS; s++) {
      if (channels_[s].is_active()) {
        int ant = channels_[s].antenna();
        if (ant < num_antennas && iq_ptrs[ant]) {
          channels_[s].process(iq_ptrs[ant], stream);
          if (amc_inference_.is_ready()) {
            std::chrono::duration<double> elapsed = now - last_infer_time_[s];
            if (elapsed.count() >= AMC_INFER_INTERVAL_SEC) {
              amc_inference_.infer(channels_[s]);
              last_infer_time_[s] = now;
            }
          }
        }
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  std::array<DDCChannel, NUM_SLOTS> channels_{};
  std::array<std::chrono::steady_clock::time_point, NUM_SLOTS> last_infer_time_{};
  AMCInference amc_inference_;
};
