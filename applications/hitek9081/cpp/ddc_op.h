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
 * DDCOp — Digital Down-Converter operator (PLACEHOLDER)
 *
 * Receives raw wideband IQ data from Hitek9081RxOp, extracts a narrowband signal
 * centered on a configurable frequency, and outputs reduced-rate complex IQ
 * suitable for downstream ML inference.
 *
 * =============================================================================
 * External Control Interface
 * =============================================================================
 *
 * The DDC parameters (center_frequency_hz, bandwidth_hz) are intended to be
 * updated at runtime by an external controller. The architecture:
 *
 *   ┌─────────────────────┐         ┌──────────────────────────┐
 *   │  gRPC Server Thread │──write─→│  std::atomic<DDCConfig>  │
 *   │  (listens on port)  │         │  (lock-free, 1 cache line)│
 *   └─────────────────────┘         └──────────────────────────┘
 *                                              ↑ read (trivial, ~2ns)
 *                                   ┌──────────────────────────┐
 *                                   │  DDCOp::compute()         │
 *                                   │  (called at ~16 Hz)       │
 *                                   └──────────────────────────┘
 *
 * The gRPC service definition (future, in a .proto file):
 *
 *   service DDCControl {
 *     rpc SetConfig(DDCConfigRequest) returns (DDCConfigResponse);
 *     rpc GetConfig(Empty) returns (DDCConfigResponse);
 *   }
 *
 *   message DDCConfigRequest {
 *     double center_frequency_hz = 1;
 *     double bandwidth_hz = 2;
 *   }
 *
 * Implementation steps when ready:
 *   1. Add grpc/protobuf to CMakeLists.txt dependencies
 *   2. Generate service stubs from .proto
 *   3. Spawn grpc::Server on a dedicated thread in DDCOp::initialize()
 *   4. ServiceImpl::SetConfig() writes to config_ atomically
 *   5. DDCOp::compute() reads config_ — no locking, no blocking
 *   6. Shut down server in DDCOp destructor
 *
 * =============================================================================
 * DSP Algorithm (to implement)
 * =============================================================================
 *
 * Input:  [num_antennas × fft_size × 2] int16 IQ from Hitek9081RxOp
 * Output: [num_antennas × output_len × 2] float32 complex IQ (narrowband)
 *
 * Per antenna:
 *   1. Window + FFT (same as SpectralOp — could share plan or create own)
 *   2. Identify bin range: center_bin ± num_bins/2
 *      where center_bin = (center_freq - tuned_center) / bin_width + fft_size/2
 *      and   num_bins = bandwidth_hz / bin_width
 *   3. Extract sub-band bins (circular addressing for wraparound)
 *   4. IFFT the num_bins-length sub-band → time-domain narrowband IQ
 *   5. Output rate = sample_rate * (num_bins / fft_size)
 *
 * At 65536-pt FFT / 245.76 MHz → bin_width = 3.75 kHz
 * For 1 MHz bandwidth → ~267 bins → output rate = ~1 MHz
 *
 * =============================================================================
 */

#pragma once

#include "holoscan/holoscan.hpp"
#include "ddc_kernels.cuh"
#include <chrono>

namespace holoscan::ops {

struct DDCConfig {
  double center_frequency_hz = 0.0;
  double bandwidth_hz = 1.0e6;

  bool operator==(const DDCConfig& o) const {
    return center_frequency_hz == o.center_frequency_hz &&
           bandwidth_hz == o.bandwidth_hz;
  }
};

class DDCOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DDCOp)
  DDCOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("iq_in");
    // Future: output narrowband IQ for ML inference
    // spec.output<holoscan::TensorMap>("iq_out");

    spec.param(enabled_, "enabled", "Enabled",
               "When false, operator receives but discards data", false);
    spec.param(center_frequency_hz_, "center_frequency_hz", "Center frequency (Hz)",
               "Frequency of interest for extraction", 3.5e9);
    spec.param(bandwidth_hz_, "bandwidth_hz", "Bandwidth (Hz)",
               "Extraction bandwidth", 1.0e6);
  }

  void initialize() override {
    Operator::initialize();

    config_.center_frequency_hz = center_frequency_hz_.get();
    config_.bandwidth_hz = bandwidth_hz_.get();

    // Future: start gRPC server thread here
    // grpc_server_ = start_ddc_grpc_server(port, &config_);

    HOLOSCAN_LOG_INFO("DDCOp::initialize() — enabled={}, center={:.3f} MHz, bw={:.3f} kHz",
                      enabled_.get(),
                      config_.center_frequency_hz / 1e6,
                      config_.bandwidth_hz / 1e3);
  }

  void compute(InputContext& op_input, OutputContext& /*op_output*/,
               ExecutionContext& /*context*/) override {
    auto maybe_input = op_input.receive<holoscan::TensorMap>("iq_in");
    if (!maybe_input) return;

    if (!enabled_.get()) return;

    frames_received_++;

    // Periodic status log (every ~5 seconds at 16 Hz)
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_log_time_).count();
    if (elapsed >= 5000) {
      HOLOSCAN_LOG_INFO("DDCOp: {} frames received | center={:.3f} MHz | bw={:.3f} kHz [no-op]",
                        frames_received_, config_.center_frequency_hz / 1e6,
                        config_.bandwidth_hz / 1e3);
      last_log_time_ = now;
    }

    // TODO: when implementing DSP:
    // 1. Extract antenna tensors from input TensorMap
    // 2. For each antenna:
    //    a. Window + FFT (own cuFFT plan, own stream)
    //    b. ddc_extract_subband(fft_out, subband, center_bin, num_bins, fft_size, stream)
    //    c. IFFT subband → narrowband IQ (float32)
    // 3. Emit narrowband IQ TensorMap on "iq_out"
  }

 private:
  DDCConfig config_;
  int64_t frames_received_ = 0;
  std::chrono::steady_clock::time_point last_log_time_ = std::chrono::steady_clock::now();

  // Future: std::unique_ptr<grpc::Server> grpc_server_;

  Parameter<bool> enabled_;
  Parameter<double> center_frequency_hz_;
  Parameter<double> bandwidth_hz_;
};

}  // namespace holoscan::ops
