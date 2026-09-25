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
 * hitek9081
 *
 * Receive high-rate IQ data from a Holoscan Sensor Bridge (HSB) FPGA using
 * DAQIRI ibverbs, compute windowed FFT magnitude on GPU, and display the
 * spectrum in real-time via ImGui/ImPlot.
 *
 * Pipeline:
 *   Hitek9081RxOp --> SpectralOp --> ImGuiSinkOp
 *   (ibverbs RX,    (window, FFT,    (spectrum & waterfall display)
 *    raw IQ out)     averaging,
 *                    magnitude)
 *
 * CLI Usage:
 *   ./hitek9081 <config.yaml> --sr <MHz> --center-freq <MHz> [options]
 *
 *   Required:
 *     --sr MHz            Sample rate in MHz (e.g., 491.52)
 *     --center-freq MHz   Center frequency in MHz (e.g., 750, 2400)
 *
 *   Options:
 *     --fft-size N        FFT size, power of 2, 512-1048576 (default: 65536)
 *     --num-averages N    FFTs to average, power of 2, 1-32 (default: 1)
 *     --freq-min MHz      Display min frequency (default: center - sr/2)
 *     --freq-max MHz      Display max frequency (default: center + sr/2)
 *     --hololink IP       Override FPGA IP from YAML (default: hsb.hololink_ip)
 *     -h, --help          Show usage
 *
 *   Examples:
 *     ./hitek9081 hitek9081.yaml --sr 491.52 --center-freq 750 \
 *         --fft-size 131072 --num-averages 1
 *
 * YAML Configuration (hsb section):
 *   payload_size        FPGA payload bytes per packet (from pcap RETH DMA_len)
 *   mtu                 Max per-packet payload for Holoscan IP (default: 1500,
 *                       must be >= payload_size, hardware max: 4096)
 *   custom_header_size  Bytes before IQ data in payload (0 if pure IQ)
 *   data_port           UDP destination port (must match daqiri flow udp_dst)
 *   hololink_ip         FPGA data-plane IP address
 *
 * Computed values (logged at startup, derived from YAML):
 *   max_packet_size     = 70 + payload_size + 4 (total Ethernet frame)
 *   header_size         = 70 + custom_header_size (offset to IQ data)
 *   samples_per_antenna = (payload_size - custom_header_size) / (num_antennas * 4)
 *   pkt_pages           = ceil(min(mtu, payload_size) / 128)
 *
 * MTU and DP_PACKET_SIZE:
 *   The FPGA's DP_PACKET_SIZE register (HIF offset 0x04) controls the max
 *   payload per RoCE packet in 128-byte pages. The hololink library reads
 *   "mtu" from enumeration metadata in configure_common() to compute this.
 *   We inject hsb.mtu into channel_metadata BEFORE DataChannel construction
 *   so configure_roce() programs the correct value on the first VP_MASK enable.
 *   The hardware latches DP_PACKET_SIZE on the VP_MASK rising edge and does
 *   NOT re-read it on subsequent enables -- post-hoc overrides have no effect.
 *   Always set mtu >= payload_size to avoid fragmentation.
 */

#include "app.h"
#include "app_config.h"
#include "hsb_device_manager.h"
#include <holoscan/holoscan.hpp>
#include <daqiri/daqiri.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " <config.yaml> --sr <MHz> --center-freq <MHz> [options]\n"
              << "Receive HSB FPGA IQ data, compute FFT, display spectrum.\n\n"
              << "Required:\n"
              << "  --sr MHz            Sample rate in MHz\n"
              << "  --center-freq MHz   Center frequency in MHz\n\n"
              << "Options:\n"
              << "  --fft-size N        FFT size (power of 2, 512-1048576, default: 65536)\n"
              << "  --num-averages N    FFTs to average (power of 2, 1-32, default: 1)\n"
              << "  --batch-size N      RX aggregation batch size (default: from YAML, typically 20480)\n"
              << "  --packets-per-frame N Data packets per metadata packet (default: from YAML, typically 1000)\n"
              << "  --freq-min MHz      Display minimum frequency (default: center - sr/2)\n"
              << "  --freq-max MHz      Display maximum frequency (default: center + sr/2)\n"
              << "  --hololink IP       HSB data-plane IP (default: from YAML)\n"
              << "  --waterfall         Enable waterfall display in ImGui\n"
              << "  --dump-packets      Dump 10 packets to PCAP and raw binary after 5 seconds\n"
              << "  -h, --help          Show this help\n";
}

bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static App* g_active_app = nullptr;
static HsbDeviceManager* g_hsb = nullptr;
static std::atomic<bool> g_shutting_down{false};
static std::atomic<bool> g_daqiri_shutdown_done{false};

static void safe_daqiri_shutdown() {
  if (!g_daqiri_shutdown_done.exchange(true)) {
    daqiri::shutdown();
    HOLOSCAN_LOG_INFO("DAQIRI shut down.");
  }
}

void request_app_shutdown() {
  if (g_shutting_down.exchange(true)) return;

  HOLOSCAN_LOG_INFO("Shutdown initiated -- stopping FPGA streaming...");
  if (g_hsb) {
    g_hsb->stop_streaming();
  }

  if (g_active_app) {
    if (g_active_app->is_alive) {
      g_active_app->is_alive->disable_tick();
    }
    g_active_app->stop_execution();
  }

  // Watchdog thread ensures process completes exit within 1 second
  // if GXF scheduler threads are blocked in a poll or condition
  std::thread watchdog([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    HOLOSCAN_LOG_INFO("Shutdown watchdog: completing exit.");
    safe_daqiri_shutdown();
    std::exit(0);
  });
  watchdog.detach();
}

void signalHandler(int signal_num) {
  std::cerr << "\nSignal " << signal_num << " received -- shutting down...\n";
  request_app_shutdown();
}


int main(int argc, char** argv) {
  std::string hololink_ip_override;
  uint32_t packets_per_frame_override = 0;
  AppConfig cfg;
  bool sr_set = false;
  bool cf_set = false;

  const struct option long_options[] = {
      {"help", no_argument, nullptr, 'h'},
      {"hololink", required_argument, nullptr, 0},
      {"sr", required_argument, nullptr, 0},
      {"center-freq", required_argument, nullptr, 0},
      {"fft-size", required_argument, nullptr, 0},
      {"num-averages", required_argument, nullptr, 0},
      {"batch-size", required_argument, nullptr, 0},
      {"packets-per-frame", required_argument, nullptr, 0},
      {"freq-min", required_argument, nullptr, 0},
      {"freq-max", required_argument, nullptr, 0},
      {"waterfall", no_argument, nullptr, 0},
      {"dump-packets", no_argument, nullptr, 0},
      {0, 0, nullptr, 0}};

  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "h", long_options, &option_index);
    if (c == -1) break;
    if (c == 0) {
      std::string name = long_options[option_index].name;
      if (name == "hololink") {
        hololink_ip_override = optarg;
      } else if (name == "sr") {
        cfg.sr_mhz = std::stod(optarg);
        sr_set = true;
      } else if (name == "center-freq") {
        cfg.center_freq_mhz = std::stod(optarg);
        cf_set = true;
      } else if (name == "fft-size") {
        cfg.fft_size = std::stoi(optarg);
      } else if (name == "num-averages") {
        cfg.num_averages = std::stoi(optarg);
      } else if (name == "batch-size") {
        cfg.batch_size = std::stoi(optarg);
      } else if (name == "packets-per-frame") {
        packets_per_frame_override = static_cast<uint32_t>(std::stoul(optarg));
      } else if (name == "freq-min") {
        cfg.freq_min_mhz = std::stod(optarg);
      } else if (name == "freq-max") {
        cfg.freq_max_mhz = std::stod(optarg);
      } else if (name == "waterfall") {
        cfg.waterfall = true;
      } else if (name == "dump-packets") {
        cfg.dump_packets = true;
      }
    } else if (c == 'h') {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!sr_set || !cf_set) {
    HOLOSCAN_LOG_ERROR("--sr and --center-freq are required");
    print_usage(argv[0]);
    return 1;
  }

  if (!is_power_of_2(cfg.fft_size) || cfg.fft_size < 512 || cfg.fft_size > 1048576) {
    HOLOSCAN_LOG_ERROR("--fft-size must be a power of 2 between 512 and 1048576");
    return 1;
  }

  if (cfg.num_averages < 1 || cfg.num_averages > 32 || !is_power_of_2(cfg.num_averages)) {
    HOLOSCAN_LOG_ERROR("--num-averages must be a power of 2 between 1 and 32");
    return 1;
  }

  // Default frequency display range to full Nyquist band
  if (cfg.freq_min_mhz == 0 && cfg.freq_max_mhz == 0) {
    cfg.freq_min_mhz = cfg.center_freq_mhz - cfg.sr_mhz / 2.0;
    cfg.freq_max_mhz = cfg.center_freq_mhz + cfg.sr_mhz / 2.0;
  }

  if (optind >= argc) {
    HOLOSCAN_LOG_ERROR("Usage: {} <config.yaml> --sr <MHz> --center-freq <MHz>", argv[0]);
    return 1;
  }

  std::filesystem::path config_path(argv[optind]);
  if (!config_path.is_absolute()) {
    if (std::filesystem::exists(config_path)) {
      config_path = std::filesystem::absolute(config_path);
    } else {
      config_path = std::filesystem::canonical(argv[0]).parent_path() / config_path;
    }
  }

  // -------------------------------------------------------------------------
  // Initialize HSB hardware & configure pipeline
  // -------------------------------------------------------------------------
  YAML::Node yaml = YAML::LoadFile(config_path.string());
  HsbDeviceManager hsb;
  if (!hsb.initialize(yaml, hololink_ip_override, packets_per_frame_override, cfg)) {
    return 1;
  }

  // -------------------------------------------------------------------------
  // Initialize DAQIRI network engine
  // -------------------------------------------------------------------------
  if (daqiri::daqiri_init(config_path.string()) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to initialize DAQIRI");
    return 1;
  }
  HOLOSCAN_LOG_INFO("DAQIRI initialized from {}", config_path.string());

  // -------------------------------------------------------------------------
  // Start FPGA streaming
  // -------------------------------------------------------------------------
  if (!hsb.start_streaming()) {
    HOLOSCAN_LOG_ERROR("Failed to start FPGA streaming");
    safe_daqiri_shutdown();
    return 1;
  }

  g_hsb = &hsb;

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // -------------------------------------------------------------------------
  // Run Holoscan pipeline
  // -------------------------------------------------------------------------
  auto app = holoscan::make_application<App>(cfg);
  app->config(config_path.string());
  g_active_app = app.get();
  app->run();

  // -------------------------------------------------------------------------
  // Teardown
  // -------------------------------------------------------------------------
  request_app_shutdown();

  daqiri::print_stats();
  safe_daqiri_shutdown();

  return 0;
}
