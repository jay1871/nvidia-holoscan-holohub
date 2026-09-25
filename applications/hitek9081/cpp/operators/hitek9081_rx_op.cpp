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

#include "hitek9081_rx_op.h"
#include <daqiri/daqiri.h>
#include "spectral_kernels.cuh"
#include <dlpack/dlpack.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <pthread.h>
#include <sched.h>

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
static constexpr int BTH_PSN_OFFSET = 50;             // BTH byte 8: 24-bit Packet Sequence Number
static constexpr uint8_t ROCE_WRITE_ONLY_IMM = 0x2B;  // metadata packet -- skip
static constexpr int UDP_DST_OFFSET = 36;             // Eth(14) + IP(20) + UDP_src(2)

Hitek9081RxOp::~Hitek9081RxOp() {
  HOLOSCAN_LOG_INFO("Hitek9081RxOp: {} data frames | {} payload bytes | {} dropped | "
                    "{} metadata skipped | {} PSN gaps",
                    ttl_data_frames_, ttl_payload_bytes_,
                    ttl_packets_dropped_, ttl_metadata_pkts_,
                    ttl_psn_gaps_);
  for (int a = 0; a < MAX_ANTENNAS; a++) {
    if (raw_dump_file_[a]) {
      fclose(raw_dump_file_[a]);
      raw_dump_file_[a] = nullptr;
    }
    if (raw_dump_pkt_file_[a]) {
      fclose(raw_dump_pkt_file_[a]);
      raw_dump_pkt_file_[a] = nullptr;
    }
  }
  drain_batch_queue();
  freeResources();
}

void Hitek9081RxOp::setup(OperatorSpec& spec) {
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
  spec.param<int>(rx_core_, "rx_core", "RX CPU Core",
                  "CPU core to pin RX compute thread to (-1 to disable)", 18);
  spec.param<bool>(dump_packets_, "dump_packets", "Dump packets",
                   "Dump 10 packets to PCAP and raw binary after 5 seconds", false);
}

void Hitek9081RxOp::initialize() {
  Operator::initialize();
  freeResources();

  port_id_ = get_port_id(interface_name_.get());
  if (port_id_ < 0) {
    HOLOSCAN_LOG_ERROR("Could not find DAQIRI port: {}", interface_name_.get());
    exit(1);
  }

  nom_payload_size_ = max_packet_size_.get() - header_size_.get();

  for (int n = 0; n < num_concurrent; n++) {
    for (int a = 0; a < MAX_ANTENNAS; a++) {
      size_t ptr_cap = std::max<size_t>(static_cast<size_t>(batch_size_.get()) * 2, 65536UL);
      CUDA_TRY(cudaMallocHost((void**)&h_ant_ptrs_[n][a], sizeof(void*) * ptr_cap));
    }
    cudaEventCreate(&rx_events_[n]);
  }

  const int N = fft_size_.get();
  const int A = num_antennas_.get();
  cudaStreamCreate(&accum_stream_);
  cudaEventCreate(&emit_event_);

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

  cached_fft_size_ = N;
  cached_num_antennas_ = A;
  cached_samples_per_ant_ = samples_per_antenna_.get();
  cached_iq_format_ = iq_format_.get();
  cached_batch_size_ = batch_size_.get();
  cached_dump_packets_ = dump_packets_.get();

  for (int a = 0; a < MAX_ANTENNAS; a++) {
    last_psn_[a] = 0;
    last_psn_valid_[a] = false;
    expected_psn_step_ant_[a] = 0;
    ttl_psn_gaps_ant_[a] = 0;
    last_report_psn_gaps_ant_[a] = 0;
    ttl_psn_ooo_ant_[a] = 0;
    last_report_psn_ooo_ant_[a] = 0;
  }
  psn_log_count_ = 0;
  payload_drop_log_count_ = 0;
  ttl_psn_gaps_ = 0;
  last_report_psn_gaps_ = 0;

  HOLOSCAN_LOG_INFO("Hitek9081RxOp::initialize() -- payload={} batch={} fft={} antennas={} bufs={}",
                    nom_payload_size_, cached_batch_size_, N, A, NUM_ACCUM_BUFS);
}

void Hitek9081RxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
  if (!thread_pinned_) {
    int core = rx_core_.get();
    if (core >= 0) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core, &cpuset);
      if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
        HOLOSCAN_LOG_INFO("Pinned RX compute thread to isolated Core {}", core);
      } else {
        HOLOSCAN_LOG_WARN("Failed to pin RX compute thread to Core {}", core);
      }
    }
    thread_pinned_ = true;
  }

  free_processed_packets();

  const uint16_t hdr_size = header_size_.get();
  const int base_port = static_cast<int>(data_port_.get());
  const int batch_sz = batch_size_.get();
  const bool dumping = cached_dump_packets_ && (!dump_ready_ || (dump_ready_ && !dump_done_));

  BurstParams* burst;
  const auto num_rx_queues = get_num_rx_queues(port_id_);

  for (int drain = 0; drain < 8; drain++) {
    bool any_burst = false;
    for (int q = 0; q < num_rx_queues; q++) {
      auto status = get_rx_burst(&burst, port_id_, q);
      if (status != Status::SUCCESS) continue;
      any_burst = true;
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

        int ant_id = 0;
        if (pkt_len > UDP_DST_OFFSET + 1) {
          uint16_t udp_dst = (static_cast<uint16_t>(pkt_ptr[UDP_DST_OFFSET]) << 8) |
                              static_cast<uint16_t>(pkt_ptr[UDP_DST_OFFSET + 1]);
          ant_id = static_cast<int>(udp_dst) - base_port;
          if (ant_id < 0 || ant_id >= cached_num_antennas_) continue;
        }

        uint32_t psn = 0;
        if (pkt_len > BTH_PSN_OFFSET + 3) {
          psn = (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 1]) << 16) |
                (static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 2]) << 8) |
                 static_cast<uint32_t>(pkt_ptr[BTH_PSN_OFFSET + 3]);
        }

        if (pkt_len > BTH_OPCODE_OFFSET &&
            pkt_ptr[BTH_OPCODE_OFFSET] == ROCE_WRITE_ONLY_IMM) {
          ttl_metadata_pkts_++;
          last_psn_[ant_id] = psn;
          last_psn_valid_[ant_id] = true;
          continue;
        }

        int payload_len = static_cast<int>(pkt_len) - static_cast<int>(hdr_size);
        if (payload_len <= 0 || payload_len > nom_payload_size_) {
          if (payload_drop_log_count_ < 8) {
            HOLOSCAN_LOG_WARN("Dropping packet ant{}: pkt_len={} hdr_size={} payload_len={} nom_payload_size_={}",
                              ant_id, pkt_len, hdr_size, payload_len, nom_payload_size_);
            payload_drop_log_count_++;
          }
          continue;
        }

        if (dumping) dump_raw_samples(pkt_ptr, pkt_len, ant_id);

        h_ant_ptrs_[cur_batch_idx_][ant_id][batch_ant_pkt_count_[ant_id]++] = pkt_ptr + hdr_size;

        if (last_psn_valid_[ant_id]) {
          int32_t delta = static_cast<int32_t>((psn - last_psn_[ant_id]) & 0x00FFFFFF);
          if (delta > 0x007FFFFF) delta -= 0x01000000;
          if (delta > 0) {
            int32_t step = expected_psn_step_ant_[ant_id];
            if (step <= 0) {
              step = (delta <= 2) ? delta : 1;
              expected_psn_step_ant_[ant_id] = step;
            }
            int32_t missed = (delta / step) - 1;
            if (missed > 0) {
              ttl_psn_gaps_++;
              ttl_psn_gaps_ant_[ant_id]++;
            }
            last_psn_[ant_id] = psn;
          } else if (delta < 0) {
            ttl_psn_ooo_ant_[ant_id]++;
          }
        } else {
          last_psn_[ant_id] = psn;
        }
        last_psn_valid_[ant_id] = true;

        if (psn_log_count_ < 16) {
          HOLOSCAN_LOG_INFO("PSN sample [{}] ant{}: PSN={}", psn_log_count_, ant_id, psn);
          psn_log_count_++;
        }

        ttl_data_frames_++;
        ttl_payload_bytes_ += payload_len;
        aggr_pkts_recv_++;
      }

      if (aggr_pkts_recv_ >= batch_sz) {
        aggr_pkts_recv_ = 0;
        process_batch(op_output);
      }
    }
    if (!any_burst) break;
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
    if (A > 1) {
      std::string ant_gaps_str;
      std::string ooo_str;
      for (int a = 0; a < A; a++) {
        int64_t d = ttl_psn_gaps_ant_[a] - last_report_psn_gaps_ant_[a];
        int64_t ooo = ttl_psn_ooo_ant_[a] - last_report_psn_ooo_ant_[a];
        if (a > 0) {
          ant_gaps_str += " ";
          ooo_str += ":";
        }
        ant_gaps_str += "ant" + std::to_string(a) + ":" + std::to_string(d);
        ooo_str += std::to_string(ooo);
      }
      HOLOSCAN_LOG_INFO("RX: {:.3f} Mfps | {:.2f} Gbps ({:.0f} MB/s) | {} frames | "
                        "PSN gaps: {} ({}) | ooo:({})",
                        mfps, payload_gbps, payload_mbps, ttl_data_frames_,
                        delta_gaps, ant_gaps_str, ooo_str);
    } else {
      HOLOSCAN_LOG_INFO("RX: {:.3f} Mfps | {:.2f} Gbps ({:.0f} MB/s) | {} frames | PSN gaps: {}",
                        mfps, payload_gbps, payload_mbps, ttl_data_frames_, delta_gaps);
    }
    for (int a = 0; a < A; a++) {
      last_report_psn_gaps_ant_[a] = ttl_psn_gaps_ant_[a];
      last_report_psn_ooo_ant_[a] = ttl_psn_ooo_ant_[a];
    }
    last_report_frames_ = ttl_data_frames_;
    last_report_bytes_ = ttl_payload_bytes_;
    last_report_psn_gaps_ = ttl_psn_gaps_;
    last_report_time_ = now;
  }
}

void Hitek9081RxOp::free_processed_packets() {
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

void Hitek9081RxOp::drain_batch_queue() {
  while (batch_q_.size() > 0) {
    const auto batch = batch_q_.front();
    cudaEventSynchronize(batch.evt);
    for (int m = 0; m < batch.num_bursts; m++) {
      free_all_packets_and_burst_rx(batch.bursts[m]);
    }
    batch_q_.pop();
  }
  for (int m = 0; m < cur_batch_.num_bursts; m++) {
    free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
  }
  cur_batch_.num_bursts = 0;
  batch_ant_pkt_count_.fill(0);
}

void Hitek9081RxOp::process_batch(OutputContext& op_output) {
  free_processed_packets();
  if (batch_q_.size() == static_cast<size_t>(num_concurrent)) {
    HOLOSCAN_LOG_ERROR("Fell behind -- dropping batch!");
    for (int m = 0; m < cur_batch_.num_bursts; m++) {
      ttl_packets_dropped_ += get_num_packets(cur_batch_.bursts[m]);
      free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
    }
    cur_batch_.num_bursts = 0;
    batch_ant_pkt_count_.fill(0);
    return;
  }

  int idx = cur_batch_idx_;

  accumulate_batch(idx, op_output);

  cudaEventRecord(rx_events_[idx], accum_stream_);
  cur_batch_.evt = rx_events_[idx];
  batch_q_.push(cur_batch_);

  cur_batch_.num_bursts = 0;
  batch_ant_pkt_count_.fill(0);
  cur_batch_idx_ = (cur_batch_idx_ + 1) % num_concurrent;
}

void Hitek9081RxOp::accumulate_batch(int batch_idx, OutputContext& op_output) {
  const int A = cached_num_antennas_;
  const int S = cached_samples_per_ant_;
  const int N = cached_fft_size_;
  const int iq_fmt = cached_iq_format_;

  int pkt_offset[MAX_ANTENNAS] = {};

  while (true) {
    bool has_more = false;
    for (int a = 0; a < A; a++) {
      if (pkt_offset[a] < batch_ant_pkt_count_[a]) {
        has_more = true;
        break;
      }
    }
    if (!has_more) break;

    int needed_samples = N - accum_offset_[0];
    if (needed_samples <= 0) {
      emit_iq(active_buf_, op_output);
      active_buf_ = (active_buf_ + 1) % NUM_ACCUM_BUFS;
      for (int a = 0; a < A; a++) {
        accum_offset_[a] = 0;
      }
      needed_samples = N;
    }

    bool progress = false;
    for (int a = 0; a < A; a++) {
      int a_needed = N - accum_offset_[a];
      if (a_needed <= 0) continue;

      int pkts_needed = (a_needed + S - 1) / S;
      int pkts_avail = batch_ant_pkt_count_[a] - pkt_offset[a];
      if (pkts_avail <= 0) continue;

      int pkts_to_copy = std::min(pkts_needed, pkts_avail);
      int samples_to_copy = std::min(pkts_to_copy * S, a_needed);

      gather_iq_from_scattered(&h_ant_ptrs_[batch_idx][a][pkt_offset[a]],
                               static_cast<int16_t*>(accum_bufs_[active_buf_][a]),
                               pkts_to_copy, S,
                               accum_offset_[a], N, iq_fmt, accum_stream_);

      accum_offset_[a] += samples_to_copy;
      pkt_offset[a] += pkts_to_copy;
      progress = true;
    }

    if (!progress) break;

    bool all_full = true;
    for (int a = 0; a < A; a++) {
      if (accum_offset_[a] < N) {
        all_full = false;
        break;
      }
    }

    if (all_full) {
      emit_iq(active_buf_, op_output);
      active_buf_ = (active_buf_ + 1) % NUM_ACCUM_BUFS;
      for (int a = 0; a < A; a++) {
        accum_offset_[a] = 0;
      }
    }
  }
}

void Hitek9081RxOp::emit_iq(int src_buf, OutputContext& op_output) {
  const int A = cached_num_antennas_;
  const int N = cached_fft_size_;
  const int dst_buf = active_output_buf_;
  active_output_buf_ = (active_output_buf_ + 1) % NUM_OUTPUT_BUFS;

  for (int a = 0; a < A; a++) {
    cudaMemcpyAsync(output_bufs_[dst_buf][a], accum_bufs_[src_buf][a],
                    N * 2 * sizeof(int16_t), cudaMemcpyDeviceToDevice, accum_stream_);
  }
  cudaEventRecord(emit_event_, accum_stream_);

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

void Hitek9081RxOp::dump_raw_samples(const uint8_t* pkt_ptr, int pkt_len, int ant) {
  if (dump_done_) return;
  if (ant < 0 || ant >= cached_num_antennas_) return;

  // 5-second delay for DPDK/JESD link stabilization
  if (!dump_start_set_) {
    dump_start_ = std::chrono::steady_clock::now();
    dump_start_set_ = true;
  }
  if (!dump_ready_) {
    if (std::chrono::steady_clock::now() - dump_start_ < std::chrono::seconds(5)) return;
    dump_ready_ = true;
    HOLOSCAN_LOG_INFO("=== Dump delay elapsed (5s) -- capturing {} packets of steady-state data ===",
                      DUMP_PACKET_COUNT);
  }

  if (raw_dump_pkts_[ant] >= DUMP_PACKET_COUNT) return;

  if (!dump_ant_seen_[ant]) {
    dump_ant_seen_[ant] = true;
    dump_ant_seen_count_++;
  }

  int num_samp = cached_samples_per_ant_;
  const int16_t* samples = reinterpret_cast<const int16_t*>(pkt_ptr + header_size_.get());

  if (!raw_dump_file_[ant]) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/hitek9081_raw_iq_dump_ant%d.bin", ant);
    raw_dump_file_[ant] = fopen(path, "wb");
    if (raw_dump_file_[ant])
      HOLOSCAN_LOG_INFO("=== Dumping raw IQ to {} ===", path);
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
    fwrite(samples, sizeof(int16_t), num_samp * 2, raw_dump_file_[ant]);
    fflush(raw_dump_file_[ant]);
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
  }

  raw_dump_pkts_[ant]++;
  if (raw_dump_pkts_[ant] >= DUMP_PACKET_COUNT) {
    if (raw_dump_file_[ant]) {
      fclose(raw_dump_file_[ant]);
      raw_dump_file_[ant] = nullptr;
      HOLOSCAN_LOG_INFO("=== Raw IQ dump ant{} complete: {} packets ({} complex samples) ===",
                        ant, DUMP_PACKET_COUNT, DUMP_PACKET_COUNT * num_samp);
    }
    if (raw_dump_pkt_file_[ant]) {
      fclose(raw_dump_pkt_file_[ant]);
      raw_dump_pkt_file_[ant] = nullptr;
      HOLOSCAN_LOG_INFO("=== Full packet dump ant{} complete: {} packets ===",
                        ant, DUMP_PACKET_COUNT);
    }

    bool all_done = true;
    for (int a = 0; a < cached_num_antennas_; a++) {
      if (dump_ant_seen_[a] && raw_dump_pkts_[a] < DUMP_PACKET_COUNT) {
        all_done = false;
        break;
      }
    }
    if (all_done && dump_ant_seen_count_ > 0) {
      dump_done_ = true;
      HOLOSCAN_LOG_INFO("=== All antenna dumps complete ({} packets each) ===", DUMP_PACKET_COUNT);
    }
  }
}

void Hitek9081RxOp::freeResources() {
  for (int n = 0; n < num_concurrent; n++) {
    for (int a = 0; a < MAX_ANTENNAS; a++) {
      if (h_ant_ptrs_[n][a]) {
        cudaFreeHost(h_ant_ptrs_[n][a]);
        h_ant_ptrs_[n][a] = nullptr;
      }
    }
    if (rx_events_[n]) {
      cudaEventDestroy(rx_events_[n]);
      rx_events_[n] = nullptr;
    }
  }
  for (int b = 0; b < NUM_ACCUM_BUFS; b++) {
    for (auto p : accum_bufs_[b]) {
      if (p) cudaFree(p);
    }
    accum_bufs_[b].clear();
  }
  for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
    for (auto p : output_bufs_[b]) {
      if (p) cudaFree(p);
    }
    output_bufs_[b].clear();
  }
  if (emit_event_) {
    cudaEventDestroy(emit_event_);
    emit_event_ = nullptr;
  }
  if (accum_stream_) {
    cudaStreamDestroy(accum_stream_);
    accum_stream_ = nullptr;
  }
}

}  // namespace holoscan::ops
