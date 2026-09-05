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

#include "hsb_device_manager.h"
#include "amc_inference.h"
#include <holoscan/holoscan.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#define CudaCheck(FUNC)                                                         \
    {                                                                           \
        const CUresult result = FUNC;                                           \
        if (result != CUDA_SUCCESS) {                                           \
            const char* error_name = "";                                        \
            cuGetErrorName(result, &error_name);                                \
            const char* error_string = "";                                      \
            cuGetErrorString(result, &error_string);                            \
            std::stringstream buf;                                              \
            buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " \
                << result << " (" << error_name << "): " << error_string;       \
            throw std::runtime_error(buf.str().c_str());                        \
        }                                                                       \
    }

HsbDeviceManager::HsbDeviceManager() = default;

HsbDeviceManager::~HsbDeviceManager() {
  stop_streaming();
  if (cuda_initialized_ && cu_context_) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cuda_initialized_ = false;
  }
}

bool HsbDeviceManager::initialize(const YAML::Node& yaml,
                                  const std::string& hololink_ip_override,
                                  uint32_t packets_per_frame_override,
                                  AppConfig& cfg) {
  const auto hsb_cfg = yaml["hsb"];
  if (!hsb_cfg) {
    HOLOSCAN_LOG_ERROR("Missing 'hsb' section in YAML config");
    return false;
  }

  hololink_ip_ = hololink_ip_override.empty()
      ? hsb_cfg["hololink_ip"].as<std::string>("192.168.0.2")
      : hololink_ip_override;
  mtu_ = hsb_cfg["mtu"].as<uint32_t>(1500);
  data_port_ = hsb_cfg["data_port"].as<uint32_t>();
  const uint32_t custom_header_size = hsb_cfg["custom_header_size"].as<uint32_t>(0);

  // The FPGA RoCE packetizer operates in 128-byte pages and computes:
  // payload_size = ((mtu - 74) / 128) * 128 (e.g. 3968 bytes for MTU 4096)
  constexpr size_t HOLOLINK_PAGE_SIZE = 128;
  constexpr uint32_t ROCE_OVERHEAD = 74;  // Eth(14)+IP(20)+UDP(8)+BTH(12)+RETH(16)+ICRC(4)
  const size_t fpga_payload_size = ((mtu_ - ROCE_OVERHEAD) / HOLOLINK_PAGE_SIZE) * HOLOLINK_PAGE_SIZE;

  packet_payload_size_ = hsb_cfg["payload_size"].as<size_t>(fpga_payload_size);
  if (packet_payload_size_ != fpga_payload_size) {
    HOLOSCAN_LOG_WARN("YAML payload_size ({}) does not match FPGA packetizer size ({}) for MTU {}. Using {} to match FPGA.",
                      packet_payload_size_, fpga_payload_size, mtu_, fpga_payload_size);
    packet_payload_size_ = fpga_payload_size;
  }

  packets_per_frame_ = (packets_per_frame_override > 0)
      ? packets_per_frame_override
      : hsb_cfg["packets_per_frame"].as<uint32_t>(1000);
  frame_size_ = packet_payload_size_ * packets_per_frame_;

  // Compute derived RX operator parameters from hsb config
  constexpr uint32_t ROCE_HEADER_SIZE = 70;  // Eth(14)+IP(20)+UDP(8)+BTH(12)+RETH(16)
  constexpr uint32_t ICRC_SIZE = 4;
  const uint16_t max_packet_size = static_cast<uint16_t>(ROCE_HEADER_SIZE + packet_payload_size_ + ICRC_SIZE);
  const uint16_t header_size = static_cast<uint16_t>(ROCE_HEADER_SIZE + custom_header_size);

  cfg.num_antennas = yaml["hitek9081_rx"]["num_antennas"].as<int>(1);
  cfg.iq_format = yaml["hitek9081_rx"]["iq_format"].as<int>(1);
  cfg.max_packet_size = max_packet_size;
  cfg.header_size = header_size;
  cfg.data_port = data_port_;
  // Each channel arrives as separate packets on its own UDP port
  cfg.samples_per_antenna = static_cast<int>((packet_payload_size_ - custom_header_size) / 4);
  if (cfg.batch_size == 0) {
    cfg.batch_size = yaml["hitek9081_rx"]["batch_size"].as<int>(20480);
  }
  num_antennas_ = cfg.num_antennas;

  HOLOSCAN_LOG_INFO("HSB config: pkt_payload={} frame_size={} ({} pkts/frame) mtu={} data_port={} num_antennas={} batch_size={}",
                    packet_payload_size_, frame_size_, packets_per_frame_, mtu_, data_port_, cfg.num_antennas, cfg.batch_size);
  HOLOSCAN_LOG_INFO("Computed RX params: max_packet_size={} header_size={} samples_per_antenna={} nom_payload_size={}",
                    cfg.max_packet_size, cfg.header_size, cfg.samples_per_antenna,
                    cfg.max_packet_size - cfg.header_size);

  cfg.iq_swap = yaml["spectral"]["iq_swap"].as<int>(1);
  cfg.db_min = yaml["spectral"]["db_min"].as<float>(20.0f);
  cfg.db_max = yaml["spectral"]["db_max"].as<float>(120.0f);

  constexpr size_t HOLOLINK_METADATA_SIZE = 128;
  const size_t metadata_address = ((frame_size_ + HOLOLINK_PAGE_SIZE - 1) / HOLOLINK_PAGE_SIZE) * HOLOLINK_PAGE_SIZE;
  page_size_ = metadata_address + HOLOLINK_METADATA_SIZE;
  pages_ = 1;

  double nyquist_mhz = cfg.sr_mhz / 2.0;
  HOLOSCAN_LOG_INFO("Spectrum config: sr={:.3f} MHz, center={:.3f} MHz, Nyquist=±{:.3f} MHz, FFT={}, avg={}",
                    cfg.sr_mhz, cfg.center_freq_mhz, nyquist_mhz, cfg.fft_size, cfg.num_averages);

  // Initialize CUDA context
  HOLOSCAN_LOG_INFO("Initializing CUDA...");
  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cu_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
  cuda_initialized_ = true;

  // Enumerate HSB
  HOLOSCAN_LOG_INFO("Enumerating HSB at {} — {} ADC channel(s)...", hololink_ip_, cfg.num_antennas);
  hololink_channels_.clear();
  for (int i = 0; i < cfg.num_antennas; i++) {
    hololink::Metadata meta = hololink::Enumerator::find_channel(hololink_ip_);
    if (i == 0) {
      auto board_desc = meta.get<std::string>("board_description").value_or("unknown");
      auto fpga_uuid = meta.get<std::string>("fpga_uuid").value_or("unknown");
      HOLOSCAN_LOG_INFO("HSB Board: '{}' (FPGA UUID: {})", board_desc, fpga_uuid);
    }
    hololink::DataChannel::use_sensor(meta, i);
    hololink::DataChannel::use_mtu(meta, mtu_);
    hololink_channels_.push_back(std::make_unique<hololink::DataChannel>(meta));
  }
  hololink_ = hololink_channels_[0]->hololink();

  HOLOSCAN_LOG_INFO("Starting hololink control plane...");
  hololink_->start();
  HOLOSCAN_LOG_INFO("Resetting hololink...");
  hololink_->reset();

  // Pre-build TensorRT engine if model path is set
  std::string amc_model = yaml["spectral"]["amc_model_path"].as<std::string>("");
  if (!amc_model.empty()) {
    std::string engine_path = amc_model.substr(0, amc_model.rfind('.')) + ".engine";
    std::ifstream engine_check(engine_path);
    if (!engine_check.good()) {
      HOLOSCAN_LOG_INFO("======================================================");
      HOLOSCAN_LOG_INFO("TensorRT engine not cached — building now...");
      HOLOSCAN_LOG_INFO("This is a one-time operation (30-60 seconds on GB10).");
      HOLOSCAN_LOG_INFO("The FPGA is NOT streaming yet — no packets will be lost.");
      HOLOSCAN_LOG_INFO("======================================================");
      AMCInference prebuild;
      if (prebuild.initialize(amc_model, 4096)) {
        HOLOSCAN_LOG_INFO("TensorRT engine cached: {}", engine_path);
      } else {
        HOLOSCAN_LOG_WARN("TensorRT engine build failed — AMC will be disabled");
      }
    }
  }

  return true;
}

bool HsbDeviceManager::start_streaming() {
  if (is_streaming_) return true;

  for (int i = 0; i < num_antennas_; i++) {
    hololink_channels_[i]->authenticate(0, 0);
    hololink_channels_[i]->configure_roce(0, frame_size_, page_size_, pages_, data_port_ + i);
    HOLOSCAN_LOG_INFO("Channel {} configured: sensor={} udp_dst={}", i, i, data_port_ + i);
  }

  mxfe_config_ = std::make_shared<hololink::HTK_JESD_HPS_Config>(*hololink_);
  mxfe_config_->host_pause_mapping(1);
  mxfe_config_->apply();

  const uint32_t pkt_payload = std::min(mtu_, static_cast<uint32_t>(packet_payload_size_));
  const uint32_t pkt_pages = (pkt_payload + 127) / 128;
  HOLOSCAN_LOG_INFO("FPGA dataplane programmed: port={} frame_size={} ({} pkts/frame) page_size={} mtu={} pkt_pages={}",
                    data_port_, frame_size_, packets_per_frame_, page_size_, mtu_, pkt_pages);

  is_streaming_ = true;
  return true;
}

void HsbDeviceManager::stop_streaming() {
  if (!is_streaming_) return;

  for (auto& ch : hololink_channels_) {
    if (ch) ch->unconfigure();
  }

  if (hololink_) {
    hololink_->stop();
    HOLOSCAN_LOG_INFO("Hololink stopped.");
  }

  is_streaming_ = false;
}
