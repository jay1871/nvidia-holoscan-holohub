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
 * Hitek9081RxOp — DPDK packet reception only
 *
 * Receives RoCE-framed UDP packets from the HSB FPGA via DAQIRI DPDK,
 * filters metadata packets, accumulates IQ samples per antenna into
 * FFT-sized GPU buffers, and emits raw int16 IQ for downstream processing.
 *
 * Output: TensorMap with one [fft_size, 2] int16 tensor per antenna
 *         (raw I,Q pairs on GPU). Emitted when fft_size samples are accumulated.
 *
 * Downstream consumers (via fan-out):
 *   - SpectralOp: windowed FFT + magnitude for display
 *   - DDCOp: narrowband extraction for ML inference (future)
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

// RoCE BTH layout — offsets from start of DAQIRI buffer (full Ethernet frame)
static constexpr int BTH_OPCODE_OFFSET = 42;  // Eth(14) + IP(20) + UDP(8) = start of BTH
static constexpr int BTH_PSN_OFFSET = 50;     // BTH+8: AckReq(1)+Reserved(7)+PSN(24)
static constexpr uint8_t ROCE_WRITE_ONLY_IMM = 0x2B;

class Hitek9081RxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Hitek9081RxOp)
  Hitek9081RxOp() = default;

  ~Hitek9081RxOp() {
    HOLOSCAN_LOG_INFO("Hitek9081RxOp: {} data frames | {} payload bytes | {} dropped | {} metadata skipped | {} PSN gaps ({} missing pkts)",
                      ttl_data_frames_, ttl_payload_bytes_,
                      ttl_packets_dropped_, ttl_metadata_pkts_,
                      ttl_psn_gaps_, ttl_psn_gap_pkts_);
    if (raw_dump_file_) fclose(raw_dump_file_);
    if (raw_dump_pkt_file_) fclose(raw_dump_pkt_file_);
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

    // RX batch buffers (for DPDK burst accumulation)
    for (int n = 0; n < num_concurrent; n++) {
      if (gpu_direct_.get()) {
        CUDA_TRY(cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get()));
      } else {
        CUDA_TRY(cudaMallocHost(&full_batch_data_h_[n], batch_size_.get() * nom_payload_size_));
      }
      CUDA_TRY(cudaMalloc(&full_batch_data_d_[n], batch_size_.get() * nom_payload_size_));
      cudaStreamCreate(&rx_streams_[n]);
      cudaEventCreate(&rx_events_[n]);
      dummy_packet_reorder(NULL, NULL, 1, 1, rx_streams_[n]);
      cudaStreamSynchronize(rx_streams_[n]);
    }

    // IQ accumulation buffers (ring of NUM_ACCUM_BUFS for zero-copy emit)
    const int N = fft_size_.get();
    const int A = num_antennas_.get();
    cudaStreamCreate(&accum_stream_);

    for (int b = 0; b < NUM_ACCUM_BUFS; b++) {
      accum_bufs_[b].resize(A, nullptr);
      for (int a = 0; a < A; a++) {
        cudaMalloc(&accum_bufs_[b][a], N * 2 * sizeof(int16_t));
        cudaMemsetAsync(accum_bufs_[b][a], 0, N * 2 * sizeof(int16_t), accum_stream_);
      }
    }

    // Output snapshot buffers — decoupled from the accum ring so downstream
    // never reads a buffer that's being overwritten.
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
      output_bufs_[b].resize(A, nullptr);
      for (int a = 0; a < A; a++) {
        cudaMalloc(&output_bufs_[b][a], N * 2 * sizeof(int16_t));
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
    spec.param<bool>(gpu_direct_, "gpu_direct", "GPUDirect enabled",
               "Whether GPUDirect is enabled for RX queues", true);
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
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    free_processed_packets();

    BurstParams *burst;
    const auto num_rx_queues = get_num_rx_queues(port_id_);

    for (int q = 0; q < num_rx_queues; q++) {
      auto status = get_rx_burst(&burst, port_id_, q);
      if (status != Status::SUCCESS) continue;

      auto burst_size = get_num_packets(burst);
      ttl_pkts_recv_ += burst_size;

      if (cur_batch_.num_bursts >= MAX_BURSTS_PER_BATCH) {
        free_all_packets_and_burst_rx(burst);
        ttl_packets_dropped_ += burst_size;
        continue;
      }
      cur_batch_.bursts[cur_batch_.num_bursts++] = burst;

      for (int p = 0; p < burst_size; p++) {
        auto pkt_ptr = reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, p));
        auto pkt_len = get_segment_packet_length(burst, 0, p);

        // Filter metadata packets (RoCE opcode 0x2B)
        if (pkt_len > BTH_OPCODE_OFFSET) {
          if (pkt_ptr[BTH_OPCODE_OFFSET] == ROCE_WRITE_ONLY_IMM) {
            ttl_metadata_pkts_++;
            continue;
          }
        }

        // Extract PSN (24-bit) from BTH bytes 9-11 (byte 8 is AckReq|Rsvd)
        if (pkt_len > BTH_PSN_OFFSET + 3) {
          uint32_t psn = (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 1]) << 16) |
                         (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 2]) << 8) |
                          static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 3]);
          if (psn_log_count_ < 10) {
            HOLOSCAN_LOG_INFO("PSN sample [{}]: raw bytes @{}: {:02x} {:02x} {:02x} {:02x} → PSN={}",
                              psn_log_count_, BTH_PSN_OFFSET,
                              pkt_ptr[BTH_PSN_OFFSET], pkt_ptr[BTH_PSN_OFFSET + 1],
                              pkt_ptr[BTH_PSN_OFFSET + 2], pkt_ptr[BTH_PSN_OFFSET + 3], psn);
            psn_log_count_++;
          }
          if (last_psn_valid_) {
            uint32_t expected = (last_psn_ + 2) & 0x00FFFFFF;
            if (psn != expected) {
              uint32_t gap = ((psn - last_psn_) / 2 - 1) & 0x00FFFFFF;
              ttl_psn_gaps_++;
              ttl_psn_gap_pkts_ += gap;
            }
          }
          last_psn_ = psn;
          last_psn_valid_ = true;
        }

        int payload_len = static_cast<int>(pkt_len) - static_cast<int>(header_size_.get());
        if (payload_len <= 0 || payload_len > nom_payload_size_) {
          aggr_pkts_recv_++;
          continue;
        }

        if (gpu_direct_.get()) {
          h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_] = pkt_ptr + header_size_.get();
        } else {
          auto offset = aggr_pkts_recv_ * nom_payload_size_;
          memcpy((char*)full_batch_data_h_[cur_batch_idx_] + offset,
                pkt_ptr + header_size_.get(), payload_len);
        }

        // DEBUG: accumulate one full FFT worth of samples (fft_size_) into dump
        if (raw_dump_samples_ < fft_size_.get()) {
          int num_samp = samples_per_antenna_.get();
          const int16_t* samples;
          std::vector<int16_t> tmp_host_buf;
          if (gpu_direct_.get()) {
            tmp_host_buf.resize(num_samp * 2);
            cudaMemcpy(tmp_host_buf.data(), pkt_ptr + header_size_.get(),
                       num_samp * 2 * sizeof(int16_t), cudaMemcpyDeviceToHost);
            samples = tmp_host_buf.data();
          } else {
            samples = reinterpret_cast<const int16_t*>(pkt_ptr + header_size_.get());
          }
          if (!raw_dump_file_) {
            raw_dump_file_ = fopen("/tmp/hitek9081_raw_iq_dump.bin", "wb");
            if (raw_dump_file_) {
              HOLOSCAN_LOG_INFO("=== Dumping {} samples to /tmp/hitek9081_raw_iq_dump.bin ===", fft_size_.get());
            }
          }
          if (!raw_dump_pkt_file_) {
            raw_dump_pkt_file_ = fopen("/tmp/hitek9081_raw_pkt_dump.pcap", "wb");
            if (raw_dump_pkt_file_) {
              // PCAP global header: magic, version 2.4, GMT offset 0, accuracy 0, snaplen, LINKTYPE_ETHERNET(1)
              struct { uint32_t magic, ver, tz, sig, snaplen, linktype; }
                hdr = { 0xa1b2c3d4, (2u<<16)|4u, 0, 0, 65535, 1 };
              fwrite(&hdr, sizeof(hdr), 1, raw_dump_pkt_file_);
              HOLOSCAN_LOG_INFO("=== Dumping full packets to /tmp/hitek9081_raw_pkt_dump.pcap ===");
            }
          }
          if (raw_dump_file_) {
            int to_write = std::min(num_samp, fft_size_.get() - raw_dump_samples_);
            fwrite(samples, sizeof(int16_t), to_write * 2, raw_dump_file_);
            raw_dump_samples_ += to_write;
            if (raw_dump_samples_ >= fft_size_.get()) {
              fclose(raw_dump_file_);
              raw_dump_file_ = nullptr;
              HOLOSCAN_LOG_INFO("=== Raw IQ dump complete: {} complex samples written ===", raw_dump_samples_);
            }
          }
          // Write PCAP packet record: 16-byte header + raw Ethernet frame
          if (raw_dump_pkt_file_) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            uint32_t pkt_len32 = static_cast<uint32_t>(pkt_len);
            struct { uint32_t ts_sec, ts_usec, cap_len, orig_len; }
              phdr = { static_cast<uint32_t>(ts.tv_sec),
                       static_cast<uint32_t>(ts.tv_nsec / 1000),
                       pkt_len32, pkt_len32 };
            fwrite(&phdr, sizeof(phdr), 1, raw_dump_pkt_file_);
            fwrite(pkt_ptr, 1, pkt_len, raw_dump_pkt_file_);
            if (raw_dump_samples_ >= fft_size_.get()) {
              fclose(raw_dump_pkt_file_);
              raw_dump_pkt_file_ = nullptr;
              HOLOSCAN_LOG_INFO("=== Full packet dump complete ===");
            }
          }
        }

        ttl_data_frames_++;
        ttl_payload_bytes_ += payload_len;
        aggr_pkts_recv_++;

        if (aggr_pkts_recv_ >= batch_size_.get()) {
          aggr_pkts_recv_ = 0;
          process_batch(op_output);
        }
      }
    }

    // Throughput logging
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_report_time_).count();
    if (elapsed >= 2000) {
      double secs = elapsed / 1000.0;
      int64_t delta_frames = ttl_data_frames_ - last_report_frames_;
      int64_t delta_bytes = ttl_payload_bytes_ - last_report_bytes_;
      int64_t delta_psn_gaps = ttl_psn_gaps_ - last_report_psn_gaps_;
      double mfps = delta_frames / (secs * 1e6);
      double payload_gbps = delta_bytes * 8.0 / (secs * 1e9);
      double payload_mbps = delta_bytes / (secs * 1e6);
      HOLOSCAN_LOG_INFO("RX: {:.3f} Mfps | {:.2f} Gbps ({:.0f} MB/s) | {} frames | PSN gaps: {}",
                        mfps, payload_gbps, payload_mbps, ttl_data_frames_, delta_psn_gaps);
      last_report_frames_ = ttl_data_frames_;
      last_report_bytes_ = ttl_payload_bytes_;
      last_report_psn_gaps_ = ttl_psn_gaps_;
      last_report_time_ = now;
    }
  }

 private:
  static constexpr int num_concurrent = 10;
  static constexpr int MAX_BURSTS_PER_BATCH = 256;
  static constexpr int NUM_ACCUM_BUFS = 32;
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
        for (auto m = 0; m < batch.num_bursts; m++) {
          free_all_packets_and_burst_rx(batch.bursts[m]);
        }
        batch_q_.pop();
      } else {
        break;
      }
    }
  }

  void process_batch(OutputContext& op_output) {
    free_processed_packets();
    if (batch_q_.size() == static_cast<size_t>(num_concurrent)) {
      HOLOSCAN_LOG_ERROR("Fell behind — dropping batch!");
      for (auto m = 0; m < cur_batch_.num_bursts; m++) {
        ttl_packets_dropped_ += get_num_packets(cur_batch_.bursts[m]);
        free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
      }
      cur_batch_.num_bursts = 0;
      return;
    }

    int idx = cur_batch_idx_;
    if (gpu_direct_.get()) {
      dummy_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[idx]),
                           h_dev_ptrs_[idx], nom_payload_size_,
                           batch_size_.get(), rx_streams_[idx]);
    } else {
      cudaMemcpyAsync(full_batch_data_d_[idx], full_batch_data_h_[idx],
                      batch_size_.get() * nom_payload_size_,
                      cudaMemcpyDefault, rx_streams_[idx]);
    }

    cudaEventRecord(rx_events_[idx], rx_streams_[idx]);
    cur_batch_.evt = rx_events_[idx];
    batch_q_.push(cur_batch_);

    // Accumulate IQ from this batch into the current accumulation buffer
    cudaStreamWaitEvent(accum_stream_, rx_events_[idx], 0);
    accumulate_batch(full_batch_data_d_[idx], op_output);

    cur_batch_.num_bursts = 0;
    cur_batch_idx_ = (++cur_batch_idx_ % num_concurrent);
  }

  void accumulate_batch(void* batch_gpu, OutputContext& op_output) {
    const int S = samples_per_antenna_.get();
    const int N = fft_size_.get();
    const int pkts_in_batch = batch_size_.get();
    int pkts_remaining = pkts_in_batch;
    int pkt_offset = 0;

    while (pkts_remaining > 0) {
      int samples_needed = N - accum_offset_;
      int pkts_needed = (samples_needed + S - 1) / S;
      int pkts_to_copy = std::min(pkts_needed, pkts_remaining);
      int samples_to_copy = std::min(pkts_to_copy * S, samples_needed);

      const uint8_t* src = static_cast<const uint8_t*>(batch_gpu) + pkt_offset * nom_payload_size_;
      for (int a = 0; a < num_antennas_.get(); a++) {
        gather_iq_from_batch(src, static_cast<int16_t*>(accum_bufs_[active_buf_][a]),
                             pkts_to_copy, S, nom_payload_size_,
                             accum_offset_, accum_stream_);
      }

      accum_offset_ += samples_to_copy;
      pkt_offset += pkts_to_copy;
      pkts_remaining -= pkts_to_copy;

      if (accum_offset_ >= N) {
        accum_offset_ = 0;
        emit_iq(op_output);
      }
    }
  }

  void emit_iq(OutputContext& op_output) {
    cudaStreamSynchronize(accum_stream_);

    const int A = num_antennas_.get();
    const int N = fft_size_.get();
    const int src_buf = active_buf_;
    const int dst_buf = active_output_buf_;
    active_output_buf_ = (active_output_buf_ + 1) % NUM_OUTPUT_BUFS;

    // Copy to a dedicated output buffer so the accum ring can be reused immediately
    for (int a = 0; a < A; a++) {
      cudaMemcpy(output_bufs_[dst_buf][a], accum_bufs_[src_buf][a],
                 N * 2 * sizeof(int16_t), cudaMemcpyDeviceToDevice);
    }

    // Advance accum ring buffer slot
    active_buf_ = (active_buf_ + 1) % NUM_ACCUM_BUFS;

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

  void freeResources() {
    for (int n = 0; n < num_concurrent; n++) {
      if (full_batch_data_d_[n]) cudaFree(full_batch_data_d_[n]);
      if (full_batch_data_h_[n]) cudaFreeHost(full_batch_data_h_[n]);
      if (h_dev_ptrs_[n]) cudaFreeHost(h_dev_ptrs_[n]);
      if (rx_streams_[n]) cudaStreamDestroy(rx_streams_[n]);
      if (rx_events_[n]) cudaEventDestroy(rx_events_[n]);
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
  int64_t ttl_pkts_recv_ = 0;
  int64_t aggr_pkts_recv_ = 0;
  int64_t ttl_packets_dropped_ = 0;
  int64_t ttl_metadata_pkts_ = 0;
  uint32_t last_psn_ = 0;
  bool last_psn_valid_ = false;
  int64_t ttl_psn_gaps_ = 0;
  int64_t ttl_psn_gap_pkts_ = 0;
  int psn_log_count_ = 0;
  int raw_dump_samples_ = 0;
  FILE* raw_dump_file_ = nullptr;
  FILE* raw_dump_pkt_file_ = nullptr;
  uint16_t nom_payload_size_;
  std::array<void**, num_concurrent> h_dev_ptrs_{};
  std::array<void*, num_concurrent> full_batch_data_d_{};
  std::array<void*, num_concurrent> full_batch_data_h_{};
  std::array<cudaStream_t, num_concurrent> rx_streams_{};
  std::array<cudaEvent_t, num_concurrent> rx_events_{};

  // IQ accumulation (ring buffer — internal, never exposed downstream)
  cudaStream_t accum_stream_ = nullptr;
  std::vector<void*> accum_bufs_[NUM_ACCUM_BUFS];
  int active_buf_ = 0;
  int accum_offset_ = 0;

  // Output snapshot buffers (published downstream — safe from accum overwrites)
  std::vector<void*> output_bufs_[NUM_OUTPUT_BUFS];
  int active_output_buf_ = 0;

  // Parameters
  Parameter<std::string> interface_name_;
  Parameter<bool> gpu_direct_;
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> max_packet_size_;
  Parameter<uint16_t> header_size_;
  Parameter<int> fft_size_;
  Parameter<int> num_antennas_;
  Parameter<int> samples_per_antenna_;

  // Throughput reporting
  std::chrono::steady_clock::time_point last_report_time_ = std::chrono::steady_clock::now();
  int64_t last_report_frames_ = 0;
  int64_t last_report_bytes_ = 0;
  int64_t last_report_psn_gaps_ = 0;
};

}  // namespace holoscan::ops
