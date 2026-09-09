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

#ifdef HAS_IMGUI

#include "holoscan/holoscan.hpp"
#include "imgui_display.h"
#include "ddc_manager.h"
#include <vector>
#include <memory>

namespace holoscan::ops {

class ImGuiSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImGuiSinkOp)
  ImGuiSinkOp() = default;

  // Setters must be called before first compute(). Order doesn't matter.
  void set_display(std::shared_ptr<ImGuiDisplay> display) { display_ = display; }
  void set_ddc_manager(std::shared_ptr<DDCManager> mgr) { ddc_manager_ = mgr; }
  void set_signal_params(double sample_rate_mhz, double center_freq_mhz, int fft_size) {
    sample_rate_mhz_ = sample_rate_mhz;
    center_freq_mhz_ = center_freq_mhz;
    fft_size_ = fft_size;
  }
  void set_is_alive(std::shared_ptr<holoscan::BooleanCondition> is_alive) { is_alive_ = is_alive; }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override;

 private:
  std::shared_ptr<ImGuiDisplay> display_;
  std::shared_ptr<DDCManager> ddc_manager_;
  std::shared_ptr<holoscan::BooleanCondition> is_alive_;
  double sample_rate_mhz_ = 0;
  double center_freq_mhz_ = 0;
  int fft_size_ = 0;
  std::vector<float> host_buf_;

  Parameter<int> num_antennas_;
  Parameter<int> num_bins_;
};

}  // namespace holoscan::ops

#endif  // HAS_IMGUI
