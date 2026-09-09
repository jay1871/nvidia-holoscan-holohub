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

#include "app_config.h"
#include <yaml-cpp/yaml.h>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/htk_jesd.hpp>
#include <cuda.h>
#include <atomic>
#include <memory>
#include <vector>
#include <string>

class HsbDeviceManager {
 public:
  HsbDeviceManager();
  ~HsbDeviceManager();

  // Non-copyable, non-movable device controller
  HsbDeviceManager(const HsbDeviceManager&) = delete;
  HsbDeviceManager& operator=(const HsbDeviceManager&) = delete;
  HsbDeviceManager(HsbDeviceManager&&) = delete;
  HsbDeviceManager& operator=(HsbDeviceManager&&) = delete;

  /**
   * @brief Discover HSB hardware, reset control plane, and populate AppConfig.
   *
   * @param yaml Parsed YAML configuration node.
   * @param hololink_ip_override Optional CLI override for FPGA IP.
   * @param packets_per_frame_override Optional CLI override for packet ratio.
   * @param cfg Output application configuration populated from hardware parameters.
   * @return true on successful bringup, false on error.
   */
  bool initialize(const YAML::Node& yaml,
                  const std::string& hololink_ip_override,
                  uint32_t packets_per_frame_override,
                  AppConfig& cfg);

  /**
   * @brief Program FPGA RoCE data plane and start hardware streaming.
   * @return true on success, false on error.
   */
  bool start_streaming();

  /**
   * @brief Stop FPGA streaming and unconfigure RoCE channels.
   */
  void stop_streaming();

  std::shared_ptr<hololink::Hololink> hololink() const { return hololink_; }
  bool is_streaming() const { return is_streaming_; }

 private:
  CUdevice cu_device_ = 0;
  CUcontext cu_context_ = nullptr;
  bool cuda_initialized_ = false;

  std::string hololink_ip_;
  uint32_t mtu_ = 4096;
  uint32_t data_port_ = 4096;
  size_t packet_payload_size_ = 3968;
  uint32_t packets_per_frame_ = 100000;
  size_t frame_size_ = 0;
  size_t page_size_ = 0;
  unsigned pages_ = 1;
  int num_antennas_ = 1;

  std::vector<std::unique_ptr<hololink::DataChannel>> hololink_channels_;
  std::shared_ptr<hololink::Hololink> hololink_;
  std::shared_ptr<hololink::HTK_JESD_HPS_Config> mxfe_config_;
  std::atomic<bool> is_streaming_{false};
};
