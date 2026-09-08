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

#include "spectral_op.h"
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

namespace holoscan::ops {

SpectralOp::~SpectralOp() {
  freeResources();
}

void SpectralOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("iq_in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", static_cast<uint64_t>(2)),
                 holoscan::Arg("policy", static_cast<uint64_t>(0)));
  spec.output<holoscan::TensorMap>("out")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", static_cast<uint64_t>(2)),
                 holoscan::Arg("policy", static_cast<uint64_t>(0)));

  spec.param(fft_size_, "fft_size", "FFT size", "Complex samples per FFT", 65536);
  spec.param(num_antennas_, "num_antennas", "Antennas", "Number of antenna channels", 1);
  spec.param(num_averages_, "num_averages", "FFT averages",
             "Number of FFT frames to average in linear power (1=no averaging)", 1);
  spec.param(db_min_, "db_min", "dB minimum", "Display floor in dB", -80.0f);
  spec.param(db_max_, "db_max", "dB maximum", "Display ceiling in dB", 20.0f);
  spec.param(bin_start_, "bin_start", "Display start bin", "First FFT bin to display", 0);
  spec.param(bin_end_, "bin_end", "Display end bin", "Last FFT bin to display (0=all)", 0);
  spec.param(iq_swap_, "iq_swap", "IQ swap", "Swap I/Q (0=I first, 1=Q first)", 0);
  spec.param(adc_bits_, "adc_bits", "ADC bits", "ADC resolution in bits", 16);
  spec.param(output_mode_, "output_mode", "Output mode",
             "0=normalized coords for Holoviz, 1=raw dB floats for ImGui", 0);
  spec.param(amc_model_path_, "amc_model_path", "AMC model path",
             "Path to ONNX model for modulation classification", std::string(""));
  spec.param(freq_tick_positions_, "freq_tick_positions", "Frequency tick positions",
             "Normalized [0,1] x-positions for frequency grid lines", std::vector<float>{});
}

void SpectralOp::initialize() {
  Operator::initialize();

  const int N = fft_size_.get();
  const int A = num_antennas_.get();

  resolved_bin_start_ = bin_start_.get();
  resolved_bin_end_ = (bin_end_.get() == 0) ? N : bin_end_.get();
  resolved_num_bins_ = resolved_bin_end_ - resolved_bin_start_;

  // Initialize DSP pipeline
  SpectralPipeline::Config dsp_cfg;
  dsp_cfg.fft_size = N;
  dsp_cfg.num_antennas = A;
  dsp_cfg.num_averages = num_averages_.get();
  dsp_cfg.adc_bits = adc_bits_.get();
  dsp_cfg.iq_swap = iq_swap_.get();
  pipeline_.initialize(dsp_cfg);

  // Initialize display output
  if (output_mode_.get() == 0) {
    HolovizOutput::Config viz_cfg;
    viz_cfg.bin_start = resolved_bin_start_;
    viz_cfg.bin_end = resolved_bin_end_;
    viz_cfg.db_min = db_min_.get();
    viz_cfg.db_max = db_max_.get();
    viz_cfg.freq_tick_positions = freq_tick_positions_.get();
    holoviz_output_.initialize(viz_cfg, A, pipeline_.stream());
  } else {
    ds_out_bins_ = (resolved_num_bins_ > MAX_DISPLAY_BINS) ? MAX_DISPLAY_BINS : resolved_num_bins_;
    ds_d_.resize(A, nullptr);
    for (int a = 0; a < A; a++) {
      cudaMalloc(&ds_d_[a], ds_out_bins_ * sizeof(float));
    }
  }

  // Initialize AMC inference via DDCManager if model path provided
  if (ddc_manager_ && !amc_model_path_.get().empty()) {
    HOLOSCAN_LOG_INFO("Initializing AMC inference via DDCManager...");
    if (ddc_manager_->initialize_amc(amc_model_path_.get(), 4096)) {
      HOLOSCAN_LOG_INFO("AMC inference engine ready: {}", amc_model_path_.get());
    } else {
      HOLOSCAN_LOG_WARN("AMC model failed to load: {} -- classification disabled",
                        amc_model_path_.get());
    }
  }

  HOLOSCAN_LOG_INFO("SpectralOp::initialize() -- fft={} antennas={} averages={} dB=[{},{}] "
                    "iq_swap={} bins=[{},{}] mode={}",
                    N, A, num_averages_.get(), db_min_.get(), db_max_.get(),
                    iq_swap_.get(), resolved_bin_start_, resolved_bin_end_,
                    output_mode_.get() == 0 ? "holoviz" : "imgui");
}

void SpectralOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  auto maybe_input = op_input.receive<holoscan::TensorMap>("iq_in");

  if (maybe_input) {
    const auto& input = maybe_input.value();
    // RX operator emits async -- GPU-side wait for copies to complete
    if (sync_event_) {
      cudaStreamWaitEvent(pipeline_.stream(), sync_event_, 0);
    }
    const int A = pipeline_.num_antennas();
    const int16_t* iq_ptrs[4] = {};
    int valid = 0;
    for (int a = 0; a < A && a < 4; a++) {
      std::string key = "antenna_" + std::to_string(a);
      auto it = input.find(key);
      if (it != input.end()) {
        iq_ptrs[a] = static_cast<const int16_t*>(it->second->data());
        valid = a + 1;
      }
    }
    pipeline_.process(iq_ptrs, valid);
    bool has_new = pipeline_.has_new_result();
    if (has_new) has_result_ = true;

    // Process active DDC channels on raw IQ + run AMC classification
    if (ddc_manager_) {
      ddc_manager_->process(iq_ptrs, A, pipeline_.stream());
    }
  }

  if (output_mode_.get() == 0) {
    if (has_result_) holoviz_output_.emit(pipeline_, op_output);
  } else {
    if (pipeline_.has_new_result()) emit_raw_db(op_output);
  }
}

void SpectralOp::emit_raw_db(OutputContext& op_output) {
  const int A = pipeline_.num_antennas();

  for (int a = 0; a < A; a++) {
    downsample_peak(pipeline_.mag_gpu(a), ds_d_[a], resolved_num_bins_, ds_out_bins_,
                    resolved_bin_start_, pipeline_.stream());
  }

  holoscan::TensorMap out_msg;
  for (int a = 0; a < A; a++) {
    char name[64];
    snprintf(name, sizeof(name), "antenna_%d", a);
    emitTensor(out_msg, name, ds_d_[a], ds_out_bins_, 1);
  }
  op_output.emit(out_msg, "out");
}

void SpectralOp::emitTensor(holoscan::TensorMap& out, const char* name,
                            float* data, int64_t rows, int64_t cols) {
  DLManagedTensor* dl = new DLManagedTensor();
  dl->dl_tensor.data = data;
  dl->dl_tensor.device = DLDevice{kDLCUDA, 0};
  dl->dl_tensor.ndim = 2;
  dl->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  dl->dl_tensor.shape = new int64_t[2]{rows, cols};
  dl->dl_tensor.strides = nullptr;
  dl->dl_tensor.byte_offset = 0;
  dl->manager_ctx = nullptr;
  dl->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete self;
  };
  out[name] = std::make_shared<holoscan::Tensor>(dl);
}

void SpectralOp::freeResources() {
  for (auto p : ds_d_) {
    if (p) cudaFree(p);
  }
  ds_d_.clear();
}

}  // namespace holoscan::ops
