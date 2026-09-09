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

#include "app.h"
#include "hitek9081_rx_op.h"
#include "spectral_op.h"
#include "ddc_manager.h"
#ifdef HAS_IMGUI
#include "imgui_display.h"
#include "imgui_sink_op.h"
#endif

#include <daqiri/daqiri.h>
#include <cstdio>
#include <string>
#include <vector>

App::App(const AppConfig& cfg) : cfg_(cfg) {}

void App::compose() {
  using namespace holoscan;

  scheduler(make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                 from_config("scheduler")));

  const auto [rx_en, tx_en] = daqiri::get_rx_tx_configs_enabled(config());
  if (!rx_en) {
    HOLOSCAN_LOG_ERROR("No RX interface configured in YAML");
    exit(1);
  }

  // Operators
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
      Arg("batch_size", static_cast<uint32_t>(cfg_.batch_size)),
      Arg("dump_packets", cfg_.dump_packets),
      this->is_alive);

  // Compute display bin range from frequency zoom settings
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

  auto ddc_manager = std::make_shared<DDCManager>();
  spectral->set_ddc_manager(ddc_manager);
  spectral->set_sync_event(rx->emit_event());

  // Display: ImGui or Holoviz
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
    imgui_display_->set_close_callback([this]() {
      HOLOSCAN_LOG_INFO("GUI closed -- stopping pipeline...");
      if (this->is_alive) {
        this->is_alive->disable_tick();
      }
    });

    auto imgui_sink = make_operator<ops::ImGuiSinkOp>(
        "imgui_sink",
        Arg("num_antennas", cfg_.num_antennas),
        Arg("num_bins", display_bins));
    imgui_sink->set_display(imgui_display_);
    imgui_sink->set_ddc_manager(ddc_manager);
    imgui_sink->set_signal_params(cfg_.sr_mhz, cfg_.center_freq_mhz, cfg_.fft_size);
    imgui_sink->set_is_alive(this->is_alive);

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

    add_flow(rx, spectral, {{"iq_out", "iq_in"}});
    add_flow(spectral, viz, {{"out", "receivers"}});
  }
}

std::vector<holoscan::ops::HolovizOp::InputSpec> App::buildGridSpecs() {
  using InputSpec = holoscan::ops::HolovizOp::InputSpec;
  using InputType = holoscan::ops::HolovizOp::InputType;

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
