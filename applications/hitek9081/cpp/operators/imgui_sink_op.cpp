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

#include "imgui_sink_op.h"

#ifdef HAS_IMGUI

#include <cuda_runtime.h>
#include <cmath>

namespace holoscan::ops {

void ImGuiSinkOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("in");
  spec.param(num_antennas_, "num_antennas", "Antennas", "Number of antenna channels", 1);
  spec.param(num_bins_, "num_bins", "Display bins", "Number of frequency bins", 1024);
}

void ImGuiSinkOp::initialize() {
  Operator::initialize();
  host_buf_.resize(num_bins_.get());
}

void ImGuiSinkOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  if (!display_ || display_->should_close()) return;

  auto in = op_input.receive<holoscan::TensorMap>("in");
  if (!in) return;

  const auto& input = *in;
  const int A = num_antennas_.get();
  const int bins = num_bins_.get();

  for (int a = 0; a < A; a++) {
    std::string key = "antenna_" + std::to_string(a);
    auto it = input.find(key);
    if (it == input.end()) continue;

    const auto& tensor = it->second;
    const float* gpu_data = static_cast<const float*>(tensor->data());

    cudaMemcpy(host_buf_.data(), gpu_data, bins * sizeof(float), cudaMemcpyDeviceToHost);
    display_->submit_frame(a, host_buf_.data(), bins);
  }

  // Sync GUI DDC selections → DDC channel activation
  if (ddc_manager_) {
    for (int s = 0; s < DDCManager::NUM_SLOTS; s++) {
      const auto& slot = display_->get_ddc_slot(s);

      if (slot.ddc_running && !ddc_manager_->is_active(s)) {
        DDCChannel::Config ddc_cfg;
        ddc_cfg.center_freq_mhz = slot.center_mhz;
        ddc_cfg.lo_freq_mhz = center_freq_mhz_;
        ddc_cfg.bandwidth_mhz = slot.ddc_bw_mhz;
        ddc_cfg.sample_rate_mhz = sample_rate_mhz_;
        ddc_cfg.input_length = fft_size_;
        ddc_cfg.output_length = 4096;
        ddc_cfg.antenna = slot.antenna;
        ddc_manager_->activate_channel(s, ddc_cfg);
      // 0.001 MHz epsilon prevents float jitter from triggering DDC retune every frame
      } else if (slot.ddc_running && ddc_manager_->is_active(s) &&
                 (std::abs(ddc_manager_->channel_bw_mhz(s) - slot.ddc_bw_mhz) > 0.001 ||
                  std::abs(ddc_manager_->channel_center_mhz(s) - slot.center_mhz) > 0.001)) {
        DDCChannel::Config ddc_cfg;
        ddc_cfg.center_freq_mhz = slot.center_mhz;
        ddc_cfg.lo_freq_mhz = center_freq_mhz_;
        ddc_cfg.bandwidth_mhz = slot.ddc_bw_mhz;
        ddc_cfg.sample_rate_mhz = sample_rate_mhz_;
        ddc_cfg.input_length = fft_size_;
        ddc_cfg.output_length = 4096;
        ddc_cfg.antenna = slot.antenna;
        ddc_manager_->activate_channel(s, ddc_cfg);
      } else if (!slot.active && ddc_manager_->is_active(s)) {
        // Cleared — deactivate
        ddc_manager_->deactivate_channel(s);
      }

      // Push classification results back to display
      if (ddc_manager_->is_active(s)) {
        const auto result = ddc_manager_->get_classification(s);
        if (result.class_id >= 0) {
          display_->update_classification(s, result.class_id, result.confidence,
                                          result.top3_ids, result.top3_confidence,
                                          result.family_id, result.family_confidence);
        }
      }
    }
  }

  display_->frame_complete();
}

}  // namespace holoscan::ops

#endif  // HAS_IMGUI
