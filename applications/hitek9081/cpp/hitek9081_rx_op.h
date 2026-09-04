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
 * Hitek9081RxOp — DPDK packet reception with zero-copy scatter-gather
 *
 * Receives RoCE-framed UDP packets from the HSB FPGA via DAQIRI DPDK,
 * filters metadata packets, accumulates IQ samples per antenna into
 * FFT-sized GPU buffers, and emits raw int16 IQ for downstream processing.
 *
 * Zero-copy path: on DGX Spark (unified LPDDR5), DAQIRI's host_pinned
 * buffers are GPU-accessible. We store payload pointers and pass them
 * directly to a GPU scatter-gather kernel — no memcpy, no cudaMemcpyAsync.
 *
 * Output: TensorMap with one [fft_size, 2] int16 tensor per antenna
 *         (raw I,Q pairs on GPU). Emitted at most once per compute() call.
 */

#pragma once

#include "holoscan/holoscan.hpp"
#include <daqiri/daqiri.h>
#include "kernels.cuh"
#include "spectral_kernels.cuh"
#include <dlpack/dlpack.h>
#include <algorithm>
#include <queue>
#include <array>
#include <vector>
#include <chrono>
#include <cstdio>
#include <ctime>

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                           \
  })

using namespace daqiri;

namespace holoscan::ops {

// RoCE v2 BTH offsets from buffer start (IBTA spec): Eth(14) + IP(20) + UDP(8) = 42
static constexpr int BTH_OPCODE_OFFSET = 42;
static constexpr int BTH_PSN_OFFSET = 50;   // BTH byte 8: 24-bit Packet Sequence Number
static constexpr uint8_t ROCE_WRITE_ONLY_IMM = 0x2B;  // metadata packet — skip
static constexpr int UDP_DST_OFFSET = 36;  // Eth(14) + IP(20) + UDP_src(2)


class Hitek9081RxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Hitek9081RxOp)
  Hitek9081RxOp() = default;

  ~Hitek9081RxOp() {
    HOLOSCAN_LOG_INFO("Hitek9081RxOp: {} data frames | {} payload bytes | {} dropped | "
                      "{} metadata skipped | {} PSN gaps",
                      ttl_data_frames_, ttl_payload_bytes_,
                      ttl_packets_dropped_, ttl_metadata_pkts_,
                      ttl_psn_gaps_);
    for (int a = 0; a < MAX_ANTENNAS; a++) {
      if (raw_dump_file_[a]) fclose(raw_dump_file_[a]);
      if (raw_dump_pkt_file_[a]) fclose(raw_dump_pkt_file_[a]);
    }
    drain_batch_queue();
    freeResources();
  }

  void initialize() override {
    Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ < 0) {
      HOLOSCAN_LOG_ERROR("Could not find DAQIRI port: {}", interface_name_.get());
      exit(1);
    }

    nom_payload_size_ = max_packet_size_.get() - header_size_.get();

    // Pointer arrays for zero-copy scatter-gather.
    // On DGX Spark (unified memory), DAQIRI's host_pinned buffers are
    // GPU-accessible so we store payload pointers directly.
    for (int n = 0; n < num_concurrent; n++) {
      CUDA_TRY(cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get()));
      CUDA_TRY(cudaMallocHost((void**)&h_pkt_antenna_[n], batch_size_.get()));
      cudaEventCreate(&rx_events_[n]);
    }
    for (int a = 0; a < MAX_ANTENNAS; a++) {
      CUDA_TRY(cudaMallocHost((void**)&h_ant_ptrs_[a], sizeof(void*) * batch_size_.get()));
    }

    const int N = fft_size_.get();
    const int A = num_antennas_.get();
    cudaStreamCreate(&accum_stream_);

    // Stream warmup — ensures CUDA context is active
    dummy_packet_reorder(NULL, NULL, 1, 1, accum_stream_);
    cudaStreamSynchronize(accum_stream_);

    for (int b = 0; b < NUM_ACCUM_BUFS; b++) {
      accum_bufs_[b].resize(A, nullptr);
      for (int a = 0; a < A; a++) {
        CUDA_TRY(cudaMalloc(&accum_bufs_[b][a], N * 2 * sizeof(int16_t)));
        cudaMemsetAsync(accum_bufs_[b][a], 0, N * 2 * sizeof(int16_t), accum_stream_);
      }
    }

    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
      output_bufs_[b].resize(A, nullptr);
      for (int a = 0; a < A; a++) {
        CUDA_TRY(cudaMalloc(&output_bufs_[b][a], N * 2 * sizeof(int16_t)));
      }
    }

    cudaStreamSynchronize(accum_stream_);
    HOLOSCAN_LOG_INFO("Hitek9081RxOp::initialize() — payload={} batch={} fft={} antennas={} bufs={}",
                      nom_payload_size_, batch_size_.get(), N, A, NUM_ACCUM_BUFS);
  }

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("iq_out")
        .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", static_cast<uint64_t>(4)),
                   holoscan::Arg("policy", static_cast<uint64_t>(0)));

    spec.param<std::string>(interface_name_, "interface_name", "Port name",
               "Name of the RX port from DAQIRI config", "hsb_rx_port");
    spec.param<uint32_t>(batch_size_, "batch_size", "Batch size",
               "Batch size in packets for each processing epoch", 1024);
    spec.param<uint16_t>(max_packet_size_, "max_packet_size", "Max packet size",
               "Maximum packet size including headers", 586);
    spec.param<uint16_t>(header_size_, "header_size", "Header size",
               "Bytes to skip to reach payload", 70);
    spec.param<int>(fft_size_, "fft_size", "FFT size",
               "Number of complex samples to accumulate before emitting", 65536);
    spec.param<int>(num_antennas_, "num_antennas", "Number of antennas",
               "Number of antennas in the packet", 1);
    spec.param<int>(samples_per_antenna_, "samples_per_antenna", "Samples per antenna",
               "Complex samples per antenna per packet", 128);
    spec.param<int>(iq_format_, "iq_format", "IQ format",
               "0=pair-interleaved [I,Q,I,Q,...], 1=block-4 [IIII,QQQQ,...]", 1);
    spec.param<uint32_t>(data_port_, "data_port", "Data port",
               "Base UDP dst port for channel 0", 4096);
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    free_processed_packets();

    BurstParams* burst;
    const auto num_rx_queues = get_num_rx_queues(port_id_);

    pending_emit_ = false;
    for (int q = 0; q < num_rx_queues; q++) {
      for (int drain = 0; drain < 8; drain++) {
        auto status = get_rx_burst(&burst, port_id_, q);
        if (status != Status::SUCCESS) break;
        free_processed_packets();

        auto burst_size = get_num_packets(burst);

        if (cur_batch_.num_bursts >= MAX_BURSTS_PER_BATCH) {
          free_all_packets_and_burst_rx(burst);
          ttl_packets_dropped_ += burst_size;
          continue;
        }
        cur_batch_.bursts[cur_batch_.num_bursts++] = burst;

        for (int p = 0; p < burst_size; p++) {
          auto pkt_ptr = reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, p));
          auto pkt_len = get_segment_packet_length(burst, 0, p);

          if (pkt_len > BTH_OPCODE_OFFSET &&
              pkt_ptr[BTH_OPCODE_OFFSET] == ROCE_WRITE_ONLY_IMM) {
            ttl_metadata_pkts_++;
            continue;
          }

          int ant_id = 0;
          if (pkt_len > UDP_DST_OFFSET + 1) {
            uint16_t udp_dst = (static_cast<uint16_t>(pkt_ptr[UDP_DST_OFFSET]) << 8) |
                                static_cast<uint16_t>(pkt_ptr[UDP_DST_OFFSET + 1]);
            ant_id = static_cast<int>(udp_dst) - static_cast<int>(data_port_.get());
            if (ant_id < 0 || ant_id >= MAX_ANTENNAS) ant_id = 0;
          }

          int payload_len = static_cast<int>(pkt_len) - static_cast<int>(header_size_.get());
          if (payload_len <= 0 || payload_len > nom_payload_size_) continue;

          uint32_t psn = 0;
          if (pkt_len > BTH_PSN_OFFSET + 3) {
            psn = (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 1]) << 16) |
                  (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 2]) << 8) |
                   static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 3]);
            if (psn_log_count_ < 20) {
              HOLOSCAN_LOG_INFO("PSN sample [{}] ant{}: PSN={}", psn_log_count_, ant_id, psn);
              psn_log_count_++;
            }
          }

          dump_raw_samples(pkt_ptr, pkt_len, ant_id);

          // Direct insertion — no reorder buffer
          h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_] = pkt_ptr + header_size_.get();
          h_pkt_antenna_[cur_batch_idx_][aggr_pkts_recv_] = static_cast<uint8_t>(ant_id);

          // High-water-mark PSN gap detection per antenna
          if (last_psn_valid_[ant_id]) {
            int32_t delta = static_cast<int32_t>((psn - last_psn_[ant_id]) & 0x00FFFFFF);
            if (delta > 0x007FFFFF) delta -= 0x01000000;
            if (delta > 0) {
              uint32_t missed = (delta / 2) - 1;
              if (missed > 0) {
                ttl_psn_gaps_++;
                ttl_psn_gaps_ant_[ant_id]++;
              }
              last_psn_[ant_id] = psn;
            }
          } else {
            last_psn_[ant_id] = psn;
          }
          last_psn_valid_[ant_id] = true;

          ttl_data_frames_++;
          ttl_payload_bytes_ += payload_len;
          aggr_pkts_recv_++;

          if (aggr_pkts_recv_ >= batch_size_.get()) {
            aggr_pkts_recv_ = 0;
            process_batch();
          }
        }
      }
    }

    if (pending_emit_) {
      emit_iq(pending_emit_buf_, op_output);
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_report_time_).count();
    if (elapsed >= 2000) {
      double secs = elapsed / 1000.0;
      int64_t delta_frames = ttl_data_frames_ - last_report_frames_;
      int64_t delta_bytes = ttl_payload_bytes_ - last_report_bytes_;
      double mfps = delta_frames / (secs * 1e6);
      double payload_gbps = delta_bytes * 8.0 / (secs * 1e9);
      double payload_mbps = delta_bytes / (secs * 1e6);
      const int A = num_antennas_.get();
      int64_t delta_gaps = ttl_psn_gaps_ - last_report_psn_gaps_;
      if (A >= 2) {
        int64_t d0 = ttl_psn_gaps_ant_[0] - last_report_psn_gaps_ant_[0];
        int64_t d1 = ttl_psn_gaps_ant_[1] - last_report_psn_gaps_ant_[1];
        HOLOSCAN_LOG_INFO("RX: {:.3f} Mfps | {:.2f} Gbps ({:.0f} MB/s) | {} frames | "
                          "PSN gaps: {} (ant0:{} ant1:{})",
                          mfps, payload_gbps, payload_mbps, ttl_data_frames_,
                          delta_gaps, d0, d1);
        for (int a = 0; a < A; a++)
          last_report_psn_gaps_ant_[a] = ttl_psn_gaps_ant_[a];
      } else {
        HOLOSCAN_LOG_INFO("RX: {:.3f} Mfps | {:.2f} Gbps ({:.0f} MB/s) | {} frames | PSN gaps: {}",
                          mfps, payload_gbps, payload_mbps, ttl_data_frames_, delta_gaps);
      }
      last_report_frames_ = ttl_data_frames_;
      last_report_bytes_ = ttl_payload_bytes_;
      last_report_psn_gaps_ = ttl_psn_gaps_;
      last_report_time_ = now;
    }
  }

 private:
  static constexpr int MAX_ANTENNAS = 4;
  static constexpr int num_concurrent = 32;    // max in-flight batches. Sized for 1.5M pkt/s at batch_size=4096
  static constexpr int MAX_BURSTS_PER_BATCH = 256;
  static constexpr int NUM_ACCUM_BUFS = 32;   // ring depth > num_concurrent to prevent overwrite during GPU read
  static constexpr int NUM_OUTPUT_BUFS = 4;

  struct BatchAggregationParams {
    std::array<BurstParams*, MAX_BURSTS_PER_BATCH> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  void free_processed_packets() {
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        for (int m = 0; m < batch.num_bursts; m++) {
          free_all_packets_and_burst_rx(batch.bursts[m]);
        }
        batch_q_.pop();
      } else {
        break;
      }
    }
  }

  void drain_batch_queue() {
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      cudaEventSynchronize(batch.evt);
      for (int m = 0; m < batch.num_bursts; m++) {
        free_all_packets_and_burst_rx(batch.bursts[m]);
      }
      batch_q_.pop();
    }
    // Free any partial batch that hasn't been submitted yet
    for (int m = 0; m < cur_batch_.num_bursts; m++) {
      free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
    }
    cur_batch_.num_bursts = 0;
  }

  void process_batch() {
    free_processed_packets();
    if (batch_q_.size() == static_cast<size_t>(num_concurrent)) {
      HOLOSCAN_LOG_ERROR("Fell behind — dropping batch!");
      for (int m = 0; m < cur_batch_.num_bursts; m++) {
        ttl_packets_dropped_ += get_num_packets(cur_batch_.bursts[m]);
        free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
      }
      cur_batch_.num_bursts = 0;
      return;
    }

    int idx = cur_batch_idx_;

    accumulate_batch(h_dev_ptrs_[idx], h_pkt_antenna_[idx]);

    cudaEventRecord(rx_events_[idx], accum_stream_);
    cur_batch_.evt = rx_events_[idx];
    batch_q_.push(cur_batch_);

    cur_batch_.num_bursts = 0;
    cur_batch_idx_ = (cur_batch_idx_ + 1) % num_concurrent;
  }

  void accumulate_batch(void** pkt_ptrs, uint8_t* pkt_antennas) {
    const int A = num_antennas_.get();
    const int S = samples_per_antenna_.get();
    const int N = fft_size_.get();
    const int total_pkts = batch_size_.get();

    // Partition packets by antenna ID (from UDP dst port)
    for (int a = 0; a < A; a++) ant_pkt_count_[a] = 0;
    for (int i = 0; i < total_pkts; i++) {
      int a = pkt_antennas[i];
      if (a < A) {
        h_ant_ptrs_[a][ant_pkt_count_[a]++] = pkt_ptrs[i];
      }
    }

    // Accumulate each antenna independently
    for (int a = 0; a < A; a++) {
      int remaining = ant_pkt_count_[a];
      int offset = 0;
      while (remaining > 0) {
        int needed = N - accum_offset_[a];
        int pkts_needed = (needed + S - 1) / S;
        int pkts_to_copy = std::min(pkts_needed, remaining);
        int samples_to_copy = std::min(pkts_to_copy * S, needed);

        gather_iq_from_scattered(h_ant_ptrs_[a] + offset,
                                 static_cast<int16_t*>(accum_bufs_[active_buf_][a]),
                                 pkts_to_copy, S,
                                 accum_offset_[a], iq_format_.get(), accum_stream_);

        accum_offset_[a] += samples_to_copy;
        offset += pkts_to_copy;
        remaining -= pkts_to_copy;

        if (accum_offset_[a] >= N) {
          accum_offset_[a] = 0;
        }
      }
    }

    // Emit when antenna 0 has filled — zero stale data in next buffer
    if (accum_offset_[0] == 0 && ant_pkt_count_[0] > 0) {
      pending_emit_ = true;
      pending_emit_buf_ = active_buf_;
      int next_buf = (active_buf_ + 1) % NUM_ACCUM_BUFS;
      for (int a = 0; a < A; a++) {
        cudaMemsetAsync(accum_bufs_[next_buf][a], 0,
                        N * 2 * sizeof(int16_t), accum_stream_);
      }
      active_buf_ = next_buf;
    }
  }

  void emit_iq(int src_buf, OutputContext& op_output) {
    const int A = num_antennas_.get();
    const int N = fft_size_.get();
    const int dst_buf = active_output_buf_;
    active_output_buf_ = (active_output_buf_ + 1) % NUM_OUTPUT_BUFS;

    cudaStreamSynchronize(accum_stream_);

    for (int a = 0; a < A; a++) {
      cudaMemcpy(output_bufs_[dst_buf][a], accum_bufs_[src_buf][a],
                 N * 2 * sizeof(int16_t), cudaMemcpyDeviceToDevice);
    }

    holoscan::TensorMap out_msg;
    for (int a = 0; a < A; a++) {
      std::string key = "antenna_" + std::to_string(a);

      DLManagedTensor* dl = new DLManagedTensor();
      dl->dl_tensor.data = output_bufs_[dst_buf][a];
      dl->dl_tensor.device = DLDevice{kDLCUDA, 0};
      dl->dl_tensor.ndim = 2;
      dl->dl_tensor.dtype = DLDataType{kDLInt, 16, 1};
      dl->dl_tensor.shape = new int64_t[2]{N, 2};
      dl->dl_tensor.strides = nullptr;
      dl->dl_tensor.byte_offset = 0;
      dl->manager_ctx = nullptr;
      dl->deleter = [](DLManagedTensor* self) {
        delete[] self->dl_tensor.shape;
        delete self;
      };
      out_msg[key] = std::make_shared<holoscan::Tensor>(dl);
    }

    op_output.emit(out_msg, "iq_out");
  }

  void dump_raw_samples(const uint8_t* pkt_ptr, int pkt_len, int ant) {
    if (ant < 0 || ant >= num_antennas_.get()) return;
    if (raw_dump_samples_[ant] >= fft_size_.get()) return;

    // 5-second delay for DPDK/JESD link stabilization
    if (!dump_start_set_) {
      dump_start_ = std::chrono::steady_clock::now();
      dump_start_set_ = true;
    }
    if (!dump_ready_) {
      if (std::chrono::steady_clock::now() - dump_start_ < std::chrono::seconds(5)) return;
      dump_ready_ = true;
      HOLOSCAN_LOG_INFO("=== Dump delay elapsed — capturing steady-state data ===");
    }

    int num_samp = samples_per_antenna_.get();
    const int16_t* samples = reinterpret_cast<const int16_t*>(pkt_ptr + header_size_.get());

    if (!raw_dump_file_[ant]) {
      char path[128];
      snprintf(path, sizeof(path), "/tmp/hitek9081_raw_iq_dump_ant%d.bin", ant);
      raw_dump_file_[ant] = fopen(path, "wb");
      if (raw_dump_file_[ant])
        HOLOSCAN_LOG_INFO("=== Dumping {} samples to {} ===", fft_size_.get(), path);
    }
    if (!raw_dump_pkt_file_[ant]) {
      char path[128];
      snprintf(path, sizeof(path), "/tmp/hitek9081_raw_pkt_dump_ant%d.pcap", ant);
      raw_dump_pkt_file_[ant] = fopen(path, "wb");
      if (raw_dump_pkt_file_[ant]) {
        struct { uint32_t magic, ver, tz, sig, snaplen, linktype; }
          hdr = { 0xa1b2c3d4, (2u << 16) | 4u, 0, 0, 65535, 1 };
        fwrite(&hdr, sizeof(hdr), 1, raw_dump_pkt_file_[ant]);
        HOLOSCAN_LOG_INFO("=== Dumping full packets to {} ===", path);
      }
    }
    if (raw_dump_file_[ant]) {
      int to_write = std::min(num_samp, fft_size_.get() - raw_dump_samples_[ant]);
      fwrite(samples, sizeof(int16_t), to_write * 2, raw_dump_file_[ant]);
      fflush(raw_dump_file_[ant]);
      raw_dump_samples_[ant] += to_write;
      if (raw_dump_samples_[ant] >= fft_size_.get()) {
        fclose(raw_dump_file_[ant]);
        raw_dump_file_[ant] = nullptr;
        HOLOSCAN_LOG_INFO("=== Raw IQ dump ant{} complete: {} complex samples ===",
                          ant, raw_dump_samples_[ant]);
      }
    }
    if (raw_dump_pkt_file_[ant]) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      uint32_t pkt_len32 = static_cast<uint32_t>(pkt_len);
      struct { uint32_t ts_sec, ts_usec, cap_len, orig_len; }
        phdr = { static_cast<uint32_t>(ts.tv_sec),
                 static_cast<uint32_t>(ts.tv_nsec / 1000),
                 pkt_len32, pkt_len32 };
      fwrite(&phdr, sizeof(phdr), 1, raw_dump_pkt_file_[ant]);
      fwrite(pkt_ptr, 1, pkt_len, raw_dump_pkt_file_[ant]);
      fflush(raw_dump_pkt_file_[ant]);
      if (raw_dump_samples_[ant] >= fft_size_.get()) {
        fclose(raw_dump_pkt_file_[ant]);
        raw_dump_pkt_file_[ant] = nullptr;
        HOLOSCAN_LOG_INFO("=== Full packet dump ant{} complete ===", ant);
      }
    }
  }

  void freeResources() {
    for (int n = 0; n < num_concurrent; n++) {
      if (h_dev_ptrs_[n]) cudaFreeHost(h_dev_ptrs_[n]);
      if (h_pkt_antenna_[n]) cudaFreeHost(h_pkt_antenna_[n]);
      if (rx_events_[n]) cudaEventDestroy(rx_events_[n]);
    }
    for (int a = 0; a < MAX_ANTENNAS; a++) {
      if (h_ant_ptrs_[a]) cudaFreeHost(h_ant_ptrs_[a]);
    }
    for (int b = 0; b < NUM_ACCUM_BUFS; b++) {
      for (auto p : accum_bufs_[b]) { if (p) cudaFree(p); }
    }
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
      for (auto p : output_bufs_[b]) { if (p) cudaFree(p); }
    }
    if (accum_stream_) cudaStreamDestroy(accum_stream_);
  }

  // RX state
  int port_id_;
  BatchAggregationParams cur_batch_{};
  int cur_batch_idx_ = 0;
  std::queue<BatchAggregationParams> batch_q_;
  int64_t ttl_payload_bytes_ = 0;
  int64_t ttl_data_frames_ = 0;
  int64_t aggr_pkts_recv_ = 0;
  int64_t ttl_packets_dropped_ = 0;
  int64_t ttl_metadata_pkts_ = 0;
  uint32_t last_psn_[MAX_ANTENNAS] = {};
  bool last_psn_valid_[MAX_ANTENNAS] = {};
  int64_t ttl_psn_gaps_ = 0;
  int64_t ttl_psn_gaps_ant_[MAX_ANTENNAS] = {};
  int psn_log_count_ = 0;
  int raw_dump_samples_[MAX_ANTENNAS] = {};
  FILE* raw_dump_file_[MAX_ANTENNAS] = {};
  FILE* raw_dump_pkt_file_[MAX_ANTENNAS] = {};
  std::chrono::steady_clock::time_point dump_start_;
  bool dump_start_set_ = false;
  bool dump_ready_ = false;
  uint16_t nom_payload_size_;
  std::array<void**, num_concurrent> h_dev_ptrs_{};
  std::array<uint8_t*, num_concurrent> h_pkt_antenna_{};
  std::array<cudaEvent_t, num_concurrent> rx_events_{};

  // Per-antenna packet demux scratch
  void** h_ant_ptrs_[MAX_ANTENNAS] = {};
  int ant_pkt_count_[MAX_ANTENNAS] = {};

  // IQ accumulation — per-antenna offsets
  cudaStream_t accum_stream_ = nullptr;
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

  // Throughput reporting
  std::chrono::steady_clock::time_point last_report_time_ = std::chrono::steady_clock::now();
  int64_t last_report_frames_ = 0;
  int64_t last_report_bytes_ = 0;
  int64_t last_report_psn_gaps_ = 0;
  int64_t last_report_psn_gaps_ant_[MAX_ANTENNAS] = {};
};

}  // namespace holoscan::ops
