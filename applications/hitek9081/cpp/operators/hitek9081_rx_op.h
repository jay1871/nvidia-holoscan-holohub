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
 * Hitek9081RxOp -- DPDK packet reception with zero-copy scatter-gather
 *
 * Receives RoCE-framed UDP packets from the HSB FPGA via DAQIRI DPDK,
 * filters metadata packets, accumulates IQ samples per antenna into
 * FFT-sized GPU buffers, and emits raw int16 IQ for downstream processing.
 *
 * Zero-copy path: on DGX Spark (unified LPDDR5), DAQIRI's host_pinned
 * buffers are GPU-accessible. We store payload pointers and pass them
 * directly to a GPU scatter-gather kernel -- no memcpy, no cudaMemcpyAsync.
 *
 * Output: TensorMap with one [fft_size, 2] int16 tensor per antenna
 *         (raw I,Q pairs on GPU). Emitted at most once per compute() call.
 */

#pragma once

#include "holoscan/holoscan.hpp"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace daqiri {
struct BurstParams;
}

namespace holoscan::ops {

class Hitek9081RxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Hitek9081RxOp)
  Hitek9081RxOp() = default;
  ~Hitek9081RxOp() override;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

  cudaEvent_t emit_event() const { return emit_event_; }

 private:
  static constexpr int MAX_ANTENNAS = 4;
  static constexpr int num_concurrent = 32;    // max in-flight batches. Sized for 1.5M pkt/s at batch_size=4096
  static constexpr int MAX_BURSTS_PER_BATCH = 256;
  static constexpr int NUM_ACCUM_BUFS = 32;   // ring depth > num_concurrent to prevent overwrite during GPU read
  static constexpr int NUM_OUTPUT_BUFS = 4;

  struct BatchAggregationParams {
    std::array<daqiri::BurstParams*, MAX_BURSTS_PER_BATCH> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  void free_processed_packets();
  void drain_batch_queue();
  void process_batch();
  void accumulate_batch(int batch_idx);
  void emit_iq(int src_buf, OutputContext& op_output);
#ifdef HITEK_ENABLE_RAW_DUMP
  void dump_raw_samples(const uint8_t* pkt_ptr, int pkt_len, int ant);
#endif
  void freeResources();

  // RX state
  int port_id_ = -1;
  BatchAggregationParams cur_batch_{};
  int cur_batch_idx_ = 0;
  std::queue<BatchAggregationParams> batch_q_;
  int64_t ttl_payload_bytes_ = 0;
  int64_t ttl_data_frames_ = 0;
  int64_t aggr_pkts_recv_ = 0;
  int64_t ttl_packets_dropped_ = 0;
  int64_t ttl_metadata_pkts_ = 0;

  // Cached Parameter values -- set once in initialize(), used in hot path
  int cached_fft_size_ = 0;
  int cached_num_antennas_ = 0;
  int cached_samples_per_ant_ = 0;
  int cached_iq_format_ = 0;
  int cached_batch_size_ = 0;
  uint32_t last_psn_[MAX_ANTENNAS] = {};
  bool last_psn_valid_[MAX_ANTENNAS] = {};
  int32_t expected_psn_step_ant_[MAX_ANTENNAS] = {};
  int psn_log_count_ = 0;
  int64_t ttl_psn_gaps_ = 0;
  int64_t ttl_psn_gaps_ant_[MAX_ANTENNAS] = {};
#ifdef HITEK_ENABLE_RAW_DUMP
  int raw_dump_samples_[MAX_ANTENNAS] = {};
  FILE* raw_dump_file_[MAX_ANTENNAS] = {};
  FILE* raw_dump_pkt_file_[MAX_ANTENNAS] = {};
  std::chrono::steady_clock::time_point dump_start_;
  bool dump_start_set_ = false;
  bool dump_ready_ = false;
  bool dump_done_ = false;
  bool dump_ant_seen_[MAX_ANTENNAS] = {};
  int dump_ant_seen_count_ = 0;
#endif
  uint16_t nom_payload_size_ = 0;
  int payload_drop_log_count_ = 0;
  std::array<cudaEvent_t, num_concurrent> rx_events_{};

  // Direct per-batch, per-antenna packet pointers (pre-segregated, zero-copy)
  std::array<std::array<void**, MAX_ANTENNAS>, num_concurrent> h_ant_ptrs_{};
  std::array<int, MAX_ANTENNAS> batch_ant_pkt_count_{};

  // IQ accumulation -- per-antenna offsets
  cudaStream_t accum_stream_ = nullptr;
  cudaEvent_t emit_event_ = nullptr;
  std::vector<void*> accum_bufs_[NUM_ACCUM_BUFS];
  int active_buf_ = 0;
  int accum_offset_[MAX_ANTENNAS] = {};
  bool pending_emit_ = false;
  int pending_emit_buf_ = 0;

  // Output snapshot buffers
  std::vector<void*> output_bufs_[NUM_OUTPUT_BUFS];
  int active_output_buf_ = 0;

  // Parameters
  Parameter<std::string> interface_name_;
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> max_packet_size_;
  Parameter<uint16_t> header_size_;
  Parameter<int> fft_size_;
  Parameter<int> num_antennas_;
  Parameter<int> samples_per_antenna_;
  Parameter<int> iq_format_;
  Parameter<uint32_t> data_port_;
  Parameter<int> rx_core_;

  bool thread_pinned_ = false;
  int64_t ttl_psn_ooo_ant_[MAX_ANTENNAS] = {};
  int64_t last_report_psn_ooo_ant_[MAX_ANTENNAS] = {};

  // Throughput reporting
  std::chrono::steady_clock::time_point last_report_time_ = std::chrono::steady_clock::now();
  int64_t last_report_frames_ = 0;
  int64_t last_report_bytes_ = 0;
  int64_t last_report_psn_gaps_ = 0;
  int64_t last_report_psn_gaps_ant_[MAX_ANTENNAS] = {};
};

}  // namespace holoscan::ops
