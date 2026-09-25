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

/*
 * SpectralOp -- Holoscan operator wrapping the SpectralPipeline DSP core
 *
 * Thin operator that:
 *   1. Receives int16 IQ TensorMap from Hitek9081RxOp
 *   2. Feeds it to SpectralPipeline (window -> FFT -> magnitude)
 *   3. Emits display-ready output via the appropriate adapter:
 *      - GPU-downsampled raw dB floats for ImGui
 */

#pragma once

#include "holoscan/holoscan.hpp"
#include "spectral_pipeline.h"
#include "spectral_kernels.cuh"
#include "ddc_manager.h"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <string>
#include <memory>

namespace holoscan::ops {

class SpectralOp : public Operator {
 public:
  static constexpr int MAX_DISPLAY_BINS = 2048;

  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectralOp)
  SpectralOp() = default;
  ~SpectralOp() override;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

  void set_ddc_manager(std::shared_ptr<DDCManager> mgr) { ddc_manager_ = mgr; }
  void set_sync_event(cudaEvent_t event) { sync_event_ = event; }

 private:
  void emit_raw_db(OutputContext& op_output);
  void emitTensor(holoscan::TensorMap& out, const char* name,
                  float* data, int64_t rows, int64_t cols);
  void freeResources();

  // DSP core (owns FFT, window, magnitude buffers)
  SpectralPipeline pipeline_;

  // Shared DDC manager
  std::shared_ptr<DDCManager> ddc_manager_;

  std::vector<float*> ds_d_;
  int ds_out_bins_ = 0;

  // Display bin range
  int resolved_bin_start_ = 0;
  int resolved_bin_end_ = 0;
  int resolved_num_bins_ = 0;

  bool has_result_ = false;
  cudaEvent_t sync_event_ = nullptr;

  // Parameters
  Parameter<int> fft_size_;
  Parameter<int> num_antennas_;
  Parameter<int> num_averages_;
  Parameter<float> db_min_;
  Parameter<float> db_max_;
  Parameter<int> bin_start_;
  Parameter<int> bin_end_;
  Parameter<int> iq_swap_;
  Parameter<int> adc_bits_;
  Parameter<bool> dc_block_;
  Parameter<std::string> amc_model_path_;
};

}  // namespace holoscan::ops
