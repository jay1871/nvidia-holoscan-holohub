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
 * DAQIRI DPDK, compute windowed FFT magnitude on GPU, and display the
 * spectrum in real-time via Holoviz.
 *
 * Pipeline:
 *   Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp
 *   (DPDK RX,       (window, FFT,    (line plot per antenna)
 *    raw IQ out)     averaging,
 *        │           magnitude)
 *        │
 *        └──→ DDCOp (placeholder: narrowband extraction for ML)
 *
 * CLI Usage:
 *   ./hitek9081 <config.yaml> --sr <MHz> --center-freq <MHz> [options]
 *
 *   Required:
 *     --sr MHz            Sample rate in MHz (e.g., 245.76, 750)
 *     --center-freq MHz   Center frequency in MHz (e.g., 3500, 1500)
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
 *     ./hitek9081 hitek9081.yaml --sr 245.76 --center-freq 3500 \
 *         --fft-size 65536 --num-averages 8
 *
 *     ./hitek9081 hitek9081.yaml --sr 750 --center-freq 1500 \
 *         --fft-size 65536 --num-averages 8 --freq-min 350 --freq-max 1050
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
 *   NOT re-read it on subsequent enables — post-hoc overrides have no effect.
 *   Always set mtu >= payload_size to avoid fragmentation.
 */

#include "hitek9081_rx_op.h"
#include "spectral_op.h"
#include "ddc_op.h"
#ifdef HAS_IMGUI
#include "imgui_display.h"
#include "imgui_sink_op.h"
#endif

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <daqiri/daqiri.h>

#include <cuda.h>
#include <yaml-cpp/yaml.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/jesd.hpp>
#include <hololink/core/htk_jesd.hpp>

#include "amc_inference.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <csignal>

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

struct AppConfig {
  double sr_mhz = 0;
  double center_freq_mhz = 0;
  int fft_size = 65536;
  int num_antennas = 1;
  int num_averages = 1;
  float db_min = -120.0f;
  float db_max = 0.0f;
  double freq_min_mhz = 0;
  double freq_max_mhz = 0;
  int iq_swap = 1;

  int iq_format = 1;  // 0=pair-interleaved [I,Q,I,Q,...], 1=block-4 [IIII,QQQQ,...]
  bool use_imgui = false;
  bool waterfall = false;

  // Computed from hsb config
  uint16_t max_packet_size = 1482;
  uint16_t header_size = 70;
  int samples_per_antenna = 128;
  uint32_t data_port = 4096;

  // Integer-MHz frequency ticks for the display grid (computed once, capped at 11)
  std::vector<double> tick_freqs_mhz;   // actual MHz values for labels
  std::vector<float>  tick_x_norm;      // normalized [0,1] x-positions for grid lines
};

class App : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::BooleanCondition> is_alive;
#ifdef HAS_IMGUI
  std::shared_ptr<ImGuiDisplay> imgui_display_;
#endif

  explicit App(const AppConfig& cfg) : cfg_(cfg) {}

  void compose() override {
    using namespace holoscan;

    scheduler(make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                   from_config("scheduler")));

    const auto [rx_en, tx_en] = daqiri::get_rx_tx_configs_enabled(config());
    if (!rx_en) {
      HOLOSCAN_LOG_ERROR("No RX interface configured in YAML");
      exit(1);
    }

    // =========================================================================
    // Operators
    // =========================================================================
    this->is_alive = make_condition<BooleanCondition>("is_alive", true);

    auto rx = make_operator<ops::Hitek9081RxOp>(
        "hitek9081_rx",
        from_config("hitek9081_rx"),
        Arg("fft_size", cfg_.fft_size),
        Arg("num_antennas", cfg_.num_antennas),
        Arg("max_packet_size", cfg_.max_packet_size),
        Arg("header_size", cfg_.header_size),
        Arg("samples_per_antenna", cfg_.samples_per_antenna),
        Arg("iq_format", cfg_.iq_format),
        Arg("data_port", cfg_.data_port),
        this->is_alive);

    // Compute display bin range from frequency zoom settings.
    double full_bw = cfg_.sr_mhz;
    double f_low = cfg_.center_freq_mhz - full_bw / 2.0;
    int bin_start = static_cast<int>((cfg_.freq_min_mhz - f_low) / full_bw * cfg_.fft_size);
    int bin_end = static_cast<int>((cfg_.freq_max_mhz - f_low) / full_bw * cfg_.fft_size);
    if (bin_start < 0) bin_start = 0;
    if (bin_end > cfg_.fft_size) bin_end = cfg_.fft_size;

    int num_bins = bin_end - bin_start;

    auto spectral = make_operator<ops::SpectralOp>(
        "spectral",
        from_config("spectral"),
        Arg("num_averages", cfg_.num_averages),
        Arg("fft_size", cfg_.fft_size),
        Arg("num_antennas", cfg_.num_antennas),
        Arg("bin_start", bin_start),
        Arg("bin_end", bin_end),
        Arg("output_mode", cfg_.use_imgui ? 1 : 0),
        Arg("freq_tick_positions", cfg_.tick_x_norm));

    auto ddc = make_operator<ops::DDCOp>(
        "ddc",
        from_config("ddc"));

    // =========================================================================
    // Display: ImGui or Holoviz
    // =========================================================================
#ifdef HAS_IMGUI
    if (cfg_.use_imgui) {
      int display_bins = (num_bins > ops::SpectralOp::MAX_DISPLAY_BINS)
          ? ops::SpectralOp::MAX_DISPLAY_BINS : num_bins;

      ImGuiDisplay::Config disp_cfg{};
      disp_cfg.freq_min_mhz = cfg_.freq_min_mhz;
      disp_cfg.freq_max_mhz = cfg_.freq_max_mhz;
      disp_cfg.center_freq_mhz = cfg_.center_freq_mhz;
      disp_cfg.db_min = cfg_.db_min;
      disp_cfg.db_max = cfg_.db_max;
      disp_cfg.num_antennas = cfg_.num_antennas;
      disp_cfg.num_bins = display_bins;
      disp_cfg.waterfall_mode = cfg_.waterfall;

      imgui_display_ = std::make_shared<ImGuiDisplay>(disp_cfg);

      auto imgui_sink = make_operator<ops::ImGuiSinkOp>(
          "imgui_sink",
          Arg("num_antennas", cfg_.num_antennas),
          Arg("num_bins", display_bins));
      imgui_sink->set_display(imgui_display_);
      imgui_sink->set_ddc_channels(&spectral->ddc_channel(0), ops::SpectralOp::NUM_DDC_SLOTS);
      imgui_sink->set_signal_params(cfg_.sr_mhz, cfg_.center_freq_mhz, cfg_.fft_size);

      spectral->set_rx_operator(rx.get());
      add_flow(rx, spectral, {{"iq_out", "iq_in"}});
      add_flow(spectral, imgui_sink, {{"out", "in"}});
    } else
#endif
    {
      auto input_specs = buildGridSpecs();

      std::vector<std::vector<float>> colors = {
          {0.0f, 1.0f, 0.0f, 1.0f},
          {0.0f, 1.0f, 1.0f, 1.0f},
          {1.0f, 0.0f, 0.0f, 1.0f},
          {1.0f, 1.0f, 0.0f, 1.0f},
      };
      for (int a = 0; a < cfg_.num_antennas; a++) {
        std::string name = "antenna_" + std::to_string(a);
        ops::HolovizOp::InputSpec spec(name, ops::HolovizOp::InputType::LINE_STRIP);
        spec.color_ = colors[a % colors.size()];
        spec.line_width_ = 2.0f;
        input_specs.push_back(spec);
      }

      double disp_bw = cfg_.freq_max_mhz - cfg_.freq_min_mhz;
      char title[512];
      snprintf(title, sizeof(title),
               "HTK9081 Spectrum | %.2f to %.2f MHz | BW=%.2f MHz | avg=%d",
               cfg_.freq_min_mhz, cfg_.freq_max_mhz, disp_bw, cfg_.num_averages);

      auto viz = make_operator<ops::HolovizOp>(
          "holoviz",
          Arg("tensors", input_specs),
          Arg("window_title", std::string(title)),
          Arg("width", static_cast<uint32_t>(1920)),
          Arg("height", static_cast<uint32_t>(1080)));

      spectral->set_rx_operator(rx.get());
      add_flow(rx, spectral, {{"iq_out", "iq_in"}});
      add_flow(spectral, viz, {{"out", "receivers"}});
    }
  }

 private:
  using InputSpec = holoscan::ops::HolovizOp::InputSpec;
  using InputType = holoscan::ops::HolovizOp::InputType;

  std::vector<InputSpec> buildGridSpecs() {
    std::vector<InputSpec> specs;
    constexpr int NUM_DB_TICKS = 10;

    // --- dB axis (horizontal lines + labels) ---
    InputSpec db_lines("db_ticks", InputType::LINES);
    db_lines.color_ = {1.0f, 1.0f, 0.0f, 0.2f};
    db_lines.line_width_ = 1.0f;
    specs.push_back(db_lines);

    float db_range = cfg_.db_max - cfg_.db_min;
    for (int t = 0; t < NUM_DB_TICKS; t++) {
      float frac = static_cast<float>(t + 1) / static_cast<float>(NUM_DB_TICKS + 1);
      float db_val = cfg_.db_max - frac * db_range;
      char name[64], text[32];
      snprintf(name, sizeof(name), "db_label_%d", t);
      snprintf(text, sizeof(text), "%.0f dBFS", db_val);

      InputSpec label(name, InputType::TEXT);
      label.color_ = {1.0f, 1.0f, 0.0f, 0.8f};
      label.text_ = {std::string(text)};
      specs.push_back(label);
    }

    // --- Frequency axis (vertical lines + labels at integer MHz) ---
    InputSpec freq_lines("freq_ticks", InputType::LINES);
    freq_lines.color_ = {1.0f, 1.0f, 1.0f, 0.3f};
    freq_lines.line_width_ = 1.0f;
    specs.push_back(freq_lines);

    for (size_t t = 0; t < cfg_.tick_freqs_mhz.size(); t++) {
      char name[64], text[32];
      snprintf(name, sizeof(name), "label_%zu", t);
      snprintf(text, sizeof(text), "%.0f", cfg_.tick_freqs_mhz[t]);

      InputSpec label(name, InputType::TEXT);
      label.color_ = {1.0f, 1.0f, 1.0f, 0.9f};
      label.text_ = {std::string(text)};
      specs.push_back(label);
    }

    return specs;
  }

  AppConfig cfg_;
};

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
              << "  --freq-min MHz      Display minimum frequency (default: center - sr/2)\n"
              << "  --freq-max MHz      Display maximum frequency (default: center + sr/2)\n"
              << "  --hololink IP       HSB data-plane IP (default: from YAML)\n"
              << "  -h, --help          Show this help\n";
}

bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static App* g_active_app = nullptr;

void signalHandler(int signal_num) {
  if (g_active_app != nullptr) {
	  std::cerr << "\nctl-c pressed\n";
	  g_active_app->is_alive->disable_tick();
  }
}


int main(int argc, char** argv) {
  std::string hololink_ip_override;
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
      {"freq-min", required_argument, nullptr, 0},
      {"freq-max", required_argument, nullptr, 0},
      {"imgui", no_argument, nullptr, 0},
      {"waterfall", no_argument, nullptr, 0},
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
      } else if (name == "freq-min") {
        cfg.freq_min_mhz = std::stod(optarg);
      } else if (name == "freq-max") {
        cfg.freq_max_mhz = std::stod(optarg);
      } else if (name == "imgui") {
        cfg.use_imgui = true;
      } else if (name == "waterfall") {
        cfg.waterfall = true;
        cfg.use_imgui = true;
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

  // Compute display bin range (same formula as compose() for tick alignment).
  double full_bw = cfg.sr_mhz;
  double f_low = cfg.center_freq_mhz - full_bw / 2.0;
  int bin_start = static_cast<int>((cfg.freq_min_mhz - f_low) / full_bw * cfg.fft_size);
  int bin_end = static_cast<int>((cfg.freq_max_mhz - f_low) / full_bw * cfg.fft_size);
  if (bin_start < 0) bin_start = 0;
  if (bin_end > cfg.fft_size) bin_end = cfg.fft_size;
  int num_bins = bin_end - bin_start;

  // Compute integer-MHz frequency ticks, capped at 11.
  // Pick the smallest step (1, 2, 5, 10, 20, 50 ...) that yields <= 11 ticks.
  {
    constexpr int MAX_TICKS = 11;
    // candidate steps in MHz
    const double steps[] = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000};
    double step = steps[0];
    for (double s : steps) {
      step = s;
      int count = static_cast<int>(std::floor(cfg.freq_max_mhz / s) -
                                   std::ceil(cfg.freq_min_mhz / s)) + 1;
      if (count <= MAX_TICKS) break;
    }
    double first = std::ceil(cfg.freq_min_mhz / step) * step;
    for (double f = first; f <= cfg.freq_max_mhz + 1e-9; f += step) {
      // Compute tick x-position using the same bin math as the CUDA kernel
      double bin_frac = (f - f_low) / full_bw * cfg.fft_size - bin_start;
      float x_norm = static_cast<float>(bin_frac / (num_bins - 1));
      cfg.tick_freqs_mhz.push_back(f);
      cfg.tick_x_norm.push_back(x_norm);
    }
  }

  if (optind >= argc) {
    HOLOSCAN_LOG_ERROR("Usage: {} <config.yaml> --sr <MHz> --center-freq <MHz>", argv[0]);
    return 1;
  }

  std::filesystem::path config_path(argv[optind]);
  if (!config_path.is_absolute()) {
    config_path = std::filesystem::canonical(argv[0]).parent_path() / config_path;
  }

  // -------------------------------------------------------------------------
  // Load FPGA configuration from YAML
  // -------------------------------------------------------------------------
  YAML::Node yaml = YAML::LoadFile(config_path.string());
  const auto hsb_cfg = yaml["hsb"];
  if (!hsb_cfg) {
    HOLOSCAN_LOG_ERROR("Missing 'hsb' section in YAML config");
    return 1;
  }

  const std::string hololink_ip = hololink_ip_override.empty()
      ? hsb_cfg["hololink_ip"].as<std::string>("192.168.0.2")
      : hololink_ip_override;
  const size_t frame_size = hsb_cfg["payload_size"].as<size_t>();
  const uint32_t mtu = hsb_cfg["mtu"].as<uint32_t>(1500);
  const uint32_t data_port = hsb_cfg["data_port"].as<uint32_t>();
  const uint32_t custom_header_size = hsb_cfg["custom_header_size"].as<uint32_t>(0);

  // Compute derived RX operator parameters from hsb config
  constexpr uint32_t ROCE_HEADER_SIZE = 70;  // Eth(14)+IP(20)+UDP(8)+BTH(12)+RETH(16)
  constexpr uint32_t ICRC_SIZE = 4;
  const uint16_t max_packet_size = static_cast<uint16_t>(ROCE_HEADER_SIZE + frame_size + ICRC_SIZE);
  const uint16_t header_size = static_cast<uint16_t>(ROCE_HEADER_SIZE + custom_header_size);

  cfg.num_antennas = yaml["hitek9081_rx"]["num_antennas"].as<int>(1);
  cfg.iq_format = yaml["hitek9081_rx"]["iq_format"].as<int>(1);
  cfg.max_packet_size = max_packet_size;
  cfg.header_size = header_size;
  cfg.data_port = data_port;
  // Each channel arrives as separate packets on its own UDP port
  cfg.samples_per_antenna = static_cast<int>((frame_size - custom_header_size) / 4);

  HOLOSCAN_LOG_INFO("HSB config: payload_size={} mtu={} custom_header_size={} data_port={} num_antennas={}",
                    frame_size, mtu, custom_header_size, data_port, cfg.num_antennas);
  HOLOSCAN_LOG_INFO("Computed RX params: max_packet_size={} header_size={} samples_per_antenna={} nom_payload_size={}",
                    cfg.max_packet_size, cfg.header_size, cfg.samples_per_antenna,
                    cfg.max_packet_size - cfg.header_size);

  cfg.iq_swap = yaml["spectral"]["iq_swap"].as<int>(1);
  cfg.db_min = yaml["spectral"]["db_min"].as<float>(20.0f);
  cfg.db_max = yaml["spectral"]["db_max"].as<float>(120.0f);

  constexpr size_t HOLOLINK_PAGE_SIZE = 128;
  constexpr size_t HOLOLINK_METADATA_SIZE = 128;
  const size_t metadata_address = ((frame_size + HOLOLINK_PAGE_SIZE - 1) / HOLOLINK_PAGE_SIZE) * HOLOLINK_PAGE_SIZE;
  const size_t page_size = metadata_address + HOLOLINK_METADATA_SIZE;
  const unsigned pages = 1;

  double nyquist_mhz = cfg.sr_mhz / 2.0;
  HOLOSCAN_LOG_INFO("Spectrum config: sr={:.3f} MHz, center={:.3f} MHz, Nyquist=±{:.3f} MHz, FFT={}, avg={}",
                    cfg.sr_mhz, cfg.center_freq_mhz, nyquist_mhz, cfg.fft_size, cfg.num_averages);

  // -------------------------------------------------------------------------
  // Phase 1: Hololink FPGA bringup
  // -------------------------------------------------------------------------
  HOLOSCAN_LOG_INFO("Initializing CUDA...");
  CudaCheck(cuInit(0));
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, 0));
  CUcontext cu_context;
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

  HOLOSCAN_LOG_INFO("Enumerating HSB at {} — {} ADC channel(s)...", hololink_ip, cfg.num_antennas);
  std::vector<std::unique_ptr<hololink::DataChannel>> hololink_channels;
  for (int i = 0; i < cfg.num_antennas; i++) {
    hololink::Metadata meta = hololink::Enumerator::find_channel(hololink_ip);
    if (i == 0) {
      auto board_desc = meta.get<std::string>("board_description").value_or("unknown");
      auto fpga_uuid = meta.get<std::string>("fpga_uuid").value_or("unknown");
      HOLOSCAN_LOG_INFO("HSB Board: '{}' (FPGA UUID: {})", board_desc, fpga_uuid);
    }
    hololink::DataChannel::use_sensor(meta, i);
    hololink::DataChannel::use_mtu(meta, mtu);
    hololink_channels.push_back(std::make_unique<hololink::DataChannel>(meta));
  }
  std::shared_ptr<hololink::Hololink> hololink = hololink_channels[0]->hololink();

  HOLOSCAN_LOG_INFO("Starting hololink control plane...");
  hololink->start();
  HOLOSCAN_LOG_INFO("Resetting hololink...");
  hololink->reset();

  // -------------------------------------------------------------------------
  // Phase 1.5: Pre-build TensorRT engine (before DAQIRI starts)
  // -------------------------------------------------------------------------
  {
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
  }

  // -------------------------------------------------------------------------
  // Phase 2: DAQIRI init
  // -------------------------------------------------------------------------
  if (daqiri::daqiri_init(config_path.string()) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to initialize DAQIRI");
    hololink->stop();
    return 1;
  }
  HOLOSCAN_LOG_INFO("DAQIRI initialized from {}", config_path.string());

  // -------------------------------------------------------------------------
  // Phase 3: Program FPGA dataplane
  // -------------------------------------------------------------------------
  for (int i = 0; i < cfg.num_antennas; i++) {
    hololink_channels[i]->authenticate(0, 0);
    hololink_channels[i]->configure_roce(0, frame_size, page_size, pages, data_port + i);
    HOLOSCAN_LOG_INFO("Channel {} configured: sensor={} udp_dst={}", i, i, data_port + i);
  }

  std::shared_ptr<hololink::HTK_JESD_HPS_Config> mxfe_config_;
  mxfe_config_ = std::make_shared<hololink::HTK_JESD_HPS_Config>(*hololink);
  mxfe_config_->host_pause_mapping(1);
  mxfe_config_->apply();


  // MTU was injected into channel_metadata before DataChannel construction,
  // so configure_roce() → configure_common() computes DP_PACKET_SIZE natively
  // using our MTU value instead of the hardcoded DEFAULT_MTU (1500).
  const uint32_t pkt_payload = std::min(mtu, static_cast<uint32_t>(frame_size));
  const uint32_t pkt_pages = (pkt_payload + 127) / 128;
  HOLOSCAN_LOG_INFO("FPGA dataplane programmed: port={} payload_size={} page_size={} mtu={} pkt_pages={}",
                    data_port, frame_size, page_size, mtu, pkt_pages);

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  // -------------------------------------------------------------------------
  // Phase 4: Run Holoscan pipeline
  // -------------------------------------------------------------------------
  auto app = holoscan::make_application<App>(cfg);
  app->config(config_path.string());
  g_active_app = app.get();
  app->run();

  // -------------------------------------------------------------------------
  // Phase 5: Teardown
  // -------------------------------------------------------------------------
  for (auto& ch : hololink_channels) ch->unconfigure();

  daqiri::print_stats();
  daqiri::shutdown();
  HOLOSCAN_LOG_INFO("DAQIRI shut down.");

  hololink->stop();
  HOLOSCAN_LOG_INFO("Hololink stopped.");

  CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
  return 0;
}
