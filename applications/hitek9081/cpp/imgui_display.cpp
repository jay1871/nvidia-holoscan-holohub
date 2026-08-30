#ifdef HAS_IMGUI

#include "imgui_display.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <string>

ImGuiDisplay::ImGuiDisplay(const Config& cfg) : cfg_(cfg) {
  for (auto& buf : buffers_) {
    buf.antenna_data.resize(cfg_.num_antennas);
    for (auto& v : buf.antenna_data) {
      v.resize(cfg_.num_bins, cfg_.db_min);
    }
  }

  freq_axis_.resize(cfg_.num_bins);
  float freq_step = static_cast<float>((cfg_.freq_max_mhz - cfg_.freq_min_mhz) / (cfg_.num_bins - 1));
  for (int i = 0; i < cfg_.num_bins; i++) {
    freq_axis_[i] = static_cast<float>(cfg_.freq_min_mhz) + i * freq_step;
  }

  if (cfg_.waterfall_mode) {
    waterfall_buf_.resize(cfg_.num_antennas);
    for (auto& buf : waterfall_buf_) {
      buf.resize(cfg_.waterfall_history * cfg_.num_bins, cfg_.db_min);
    }
    waterfall_render_buf_.resize(cfg_.waterfall_history * cfg_.num_bins);
  }

  thread_ = std::thread(&ImGuiDisplay::render_loop, this);
}

ImGuiDisplay::~ImGuiDisplay() {
  running_.store(false);
  if (thread_.joinable()) thread_.join();
}

void ImGuiDisplay::submit_frame(int antenna, const float* db_data, int num_bins) {
  if (antenna < 0 || antenna >= cfg_.num_antennas) return;
  int n = std::min(num_bins, cfg_.num_bins);
  std::memcpy(buffers_[write_idx_].antenna_data[antenna].data(), db_data, n * sizeof(float));
}

void ImGuiDisplay::frame_complete() {
  latest_ready_.store(write_idx_, std::memory_order_release);
  // Advance write_idx, skipping the buffer the reader currently holds
  int next = (write_idx_ + 1) % NUM_BUFS;
  if (next == reading_.load(std::memory_order_acquire)) {
    next = (next + 1) % NUM_BUFS;
  }
  write_idx_ = next;
}

bool ImGuiDisplay::should_close() const {
  return closed_.load(std::memory_order_relaxed);
}

void ImGuiDisplay::update_classification(int slot, int class_id, float confidence,
                                         const int* top3_ids, const float* top3_conf) {
  if (slot < 0 || slot >= NUM_DDC_SLOTS) return;
  auto& s = ddc_slots_[slot];
  s.class_id = class_id;
  s.confidence = confidence;
  for (int i = 0; i < 3; i++) {
    s.top3_ids[i] = top3_ids[i];
    s.top3_confidence[i] = top3_conf[i];
  }
}

void ImGuiDisplay::render_loop() {
  if (!glfwInit()) {
    fprintf(stderr, "ImGuiDisplay: glfwInit failed\n");
    closed_.store(true);
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  char title[256];
  snprintf(title, sizeof(title), "HTK9081 Spectrum | %.1f - %.1f MHz",
           cfg_.freq_min_mhz, cfg_.freq_max_mhz);

  GLFWwindow* window = glfwCreateWindow(1920, 1080, title, nullptr, nullptr);
  if (!window) {
    fprintf(stderr, "ImGuiDisplay: failed to create GLFW window\n");
    glfwTerminate();
    closed_.store(true);
    return;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);  // no vsync — render as fast as possible

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Disable ImPlot's built-in box select — we handle selection manually
  ImPlot::GetInputMap().Select = ImGuiMouseButton_Middle;  // move to unused button

  while (running_.load() && !glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("##main", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Lock the read buffer for this entire frame
    cur_read_idx_ = latest_ready_.load(std::memory_order_acquire);
    reading_.store(cur_read_idx_, std::memory_order_release);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float ddc_panel_width = avail.x * 0.30f;
    float main_width = avail.x - ddc_panel_width;

    ImGui::BeginChild("##main_plots", ImVec2(main_width, avail.y), false);
    render_main_panel();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("##ddc_panel", ImVec2(ddc_panel_width, avail.y), true);
    render_ddc_panel();
    ImGui::EndChild();

    reading_.store(-1, std::memory_order_release);

    ImGui::End();
    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.06f, 0.06f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  closed_.store(true);
}

void ImGuiDisplay::render_main_panel() {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  float plot_height = avail.y / cfg_.num_antennas;

  for (int a = 0; a < cfg_.num_antennas; a++) {
    if (cfg_.waterfall_mode) {
      render_waterfall_plot(a, avail.x, plot_height);
    } else {
      render_line_plot(a, avail.x, plot_height);
    }
  }
}

static constexpr int DEFAULT_BW_IDX = 8;  // 20 MHz default

void ImGuiDisplay::handle_selection(int antenna, double center_freq) {
  // Clamp to display range
  if (center_freq < cfg_.freq_min_mhz || center_freq > cfg_.freq_max_mhz) return;

  // Find the first inactive slot, or fall back to round-robin
  int target = -1;
  for (int i = 0; i < NUM_DDC_SLOTS; i++) {
    if (!ddc_slots_[i].active) { target = i; break; }
  }
  if (target < 0) {
    target = next_slot_;
    next_slot_ = (next_slot_ + 1) % NUM_DDC_SLOTS;
  }

  DDCSlot& slot = ddc_slots_[target];
  slot = DDCSlot{};  // clean reset
  slot.active = true;
  slot.ddc_running = true;
  slot.antenna = antenna;
  slot.center_mhz = center_freq;
  slot.bw_selection_idx = DEFAULT_BW_IDX;
  slot.ddc_bw_mhz = BW_PRESETS[DEFAULT_BW_IDX].bw_mhz;

  update_slot_zoom(target);
}

void ImGuiDisplay::update_slot_zoom(int s) {
  DDCSlot& slot = ddc_slots_[s];
  double half_bw = slot.ddc_bw_mhz / 2.0;
  slot.freq_min_mhz = slot.center_mhz - half_bw;
  slot.freq_max_mhz = slot.center_mhz + half_bw;

  // Clamp to display range
  if (slot.freq_min_mhz < cfg_.freq_min_mhz) slot.freq_min_mhz = cfg_.freq_min_mhz;
  if (slot.freq_max_mhz > cfg_.freq_max_mhz) slot.freq_max_mhz = cfg_.freq_max_mhz;

  double freq_step = (cfg_.freq_max_mhz - cfg_.freq_min_mhz) / (cfg_.num_bins - 1);
  slot.bin_start = std::max(0, (int)((slot.freq_min_mhz - cfg_.freq_min_mhz) / freq_step));
  slot.bin_end = std::min(cfg_.num_bins, (int)((slot.freq_max_mhz - cfg_.freq_min_mhz) / freq_step) + 1);
}

void ImGuiDisplay::render_line_plot(int antenna, float width, float height) {
  if (cur_read_idx_ < 0) return;

  const auto& frame = buffers_[cur_read_idx_];

  static const ImVec4 colors[] = {
    {0.0f, 1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, 0.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 0.0f, 1.0f},
  };

  char label[64];
  snprintf(label, sizeof(label), "Antenna %d##line%d", antenna, antenna);

  ImPlot::PushStyleColor(ImPlotCol_Line, colors[antenna % 4]);

  if (ImPlot::BeginPlot(label, ImVec2(width, height), ImPlotFlags_NoLegend | ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxes("Frequency (MHz)", "dBFS");
    ImPlot::SetupAxisLimits(ImAxis_X1, cfg_.freq_min_mhz, cfg_.freq_max_mhz, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, cfg_.db_min, cfg_.db_max, ImPlotCond_Always);

    ImPlot::PlotLine("##spectrum", freq_axis_.data(),
                     frame.antenna_data[antenna].data(), cfg_.num_bins);

    // Draw DDC bandwidth shaded regions
    static const ImVec4 ddc_colors[] = {
      {0.2f, 0.6f, 1.0f, 0.15f},   // DDC 0: blue, transparent
      {1.0f, 0.5f, 0.0f, 0.15f},   // DDC 1: orange, transparent
    };
    for (int d = 0; d < NUM_DDC_SLOTS; d++) {
      if (ddc_slots_[d].ddc_running && ddc_slots_[d].antenna == antenna) {
        double ddc_lo = ddc_slots_[d].center_mhz - ddc_slots_[d].ddc_bw_mhz / 2.0;
        double ddc_hi = ddc_slots_[d].center_mhz + ddc_slots_[d].ddc_bw_mhz / 2.0;
        float xs[2] = {(float)ddc_lo, (float)ddc_hi};
        float y_top[2] = {(float)cfg_.db_max, (float)cfg_.db_max};
        float y_bot[2] = {(float)cfg_.db_min, (float)cfg_.db_min};
        ImPlot::PushStyleColor(ImPlotCol_Fill, ddc_colors[d % 2]);
        char shade_label[32];
        snprintf(shade_label, sizeof(shade_label), "##ddc_shade_%d", d);
        ImPlot::PlotShaded(shade_label, xs, y_top, y_bot, 2);
        ImPlot::PopStyleColor();
      }
    }

    // Single click to set DDC center frequency
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      double click_freq = ImPlot::GetPlotMousePos().x;
      handle_selection(antenna, click_freq);
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor();
}

void ImGuiDisplay::render_waterfall_plot(int antenna, float width, float height) {
  if (cur_read_idx_ < 0) return;

  // Append to waterfall only when new data arrives, track timing
  if (antenna == 0 && cur_read_idx_ != last_waterfall_idx_) {
    static double last_time = 0;
    double now = glfwGetTime();
    if (last_time > 0) {
      double dt = now - last_time;
      // Exponential moving average of frame interval
      waterfall_frame_time_ = waterfall_frame_time_ > 0
          ? waterfall_frame_time_ * 0.9 + dt * 0.1
          : dt;
    }
    last_time = now;

    for (int a = 0; a < cfg_.num_antennas; a++) {
      float* row = waterfall_buf_[a].data() + waterfall_write_pos_ * cfg_.num_bins;
      std::memcpy(row, buffers_[cur_read_idx_].antenna_data[a].data(), cfg_.num_bins * sizeof(float));
    }
    waterfall_write_pos_ = (waterfall_write_pos_ + 1) % cfg_.waterfall_history;
    last_waterfall_idx_ = cur_read_idx_;
  }

  char label[64];
  snprintf(label, sizeof(label), "Antenna %d Waterfall##wf%d", antenna, antenna);

  if (ImPlot::BeginPlot(label, ImVec2(width, height), ImPlotFlags_NoLegend | ImPlotFlags_NoBoxSelect)) {
    double actual_span = (waterfall_frame_time_ > 0)
        ? waterfall_frame_time_ * cfg_.waterfall_history
        : cfg_.waterfall_time_span;

    ImPlot::SetupAxes("Frequency (MHz)", nullptr);
    ImPlot::SetupAxisLimits(ImAxis_X1, cfg_.freq_min_mhz, cfg_.freq_max_mhz, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, actual_span, ImPlotCond_Always);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoTickMarks);

    ImPlot::PushColormap(ImPlotColormap_Viridis);

    // Reorder circular buffer into pre-allocated render buffer
    int tail = cfg_.waterfall_history - waterfall_write_pos_;
    std::memcpy(waterfall_render_buf_.data(),
                waterfall_buf_[antenna].data() + waterfall_write_pos_ * cfg_.num_bins,
                tail * cfg_.num_bins * sizeof(float));
    if (waterfall_write_pos_ > 0) {
      std::memcpy(waterfall_render_buf_.data() + tail * cfg_.num_bins,
                  waterfall_buf_[antenna].data(),
                  waterfall_write_pos_ * cfg_.num_bins * sizeof(float));
    }

    ImPlot::PlotHeatmap("##wf", waterfall_render_buf_.data(),
                        cfg_.waterfall_history, cfg_.num_bins,
                        cfg_.db_min, cfg_.db_max,
                        nullptr,
                        ImPlotPoint(cfg_.freq_min_mhz, 0),
                        ImPlotPoint(cfg_.freq_max_mhz, actual_span));

    ImPlot::PopColormap();

    // Draw DDC bandwidth shaded regions on waterfall
    static const ImVec4 wf_ddc_colors[] = {
      {0.2f, 0.6f, 1.0f, 0.2f},
      {1.0f, 0.5f, 0.0f, 0.2f},
    };
    for (int d = 0; d < NUM_DDC_SLOTS; d++) {
      if (ddc_slots_[d].ddc_running && ddc_slots_[d].antenna == antenna) {
        double ddc_lo = ddc_slots_[d].center_mhz - ddc_slots_[d].ddc_bw_mhz / 2.0;
        double ddc_hi = ddc_slots_[d].center_mhz + ddc_slots_[d].ddc_bw_mhz / 2.0;
        float xs[2] = {(float)ddc_lo, (float)ddc_hi};
        float y_top[2] = {(float)actual_span, (float)actual_span};
        float y_bot[2] = {0.0f, 0.0f};
        ImPlot::PushStyleColor(ImPlotCol_Fill, wf_ddc_colors[d % 2]);
        char shade_label[32];
        snprintf(shade_label, sizeof(shade_label), "##wf_ddc_shade_%d", d);
        ImPlot::PlotShaded(shade_label, xs, y_top, y_bot, 2);
        ImPlot::PopStyleColor();
      }
    }

    // Single click to set DDC center frequency
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      double click_freq = ImPlot::GetPlotMousePos().x;
      handle_selection(antenna, click_freq);
    }

    ImPlot::EndPlot();
  }
}

void ImGuiDisplay::render_ddc_panel() {

  ImGui::Text("Software DDC Channels");
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::TextWrapped("Left-click drag to select zoom range, then choose bandwidth to activate DDC.");
  ImGui::Spacing();

  ImVec2 avail = ImGui::GetContentRegionAvail();
  float slot_height = avail.y / NUM_DDC_SLOTS;

  static const char* class_names[31] = {
    "OOK", "4PAM", "BPSK", "QPSK", "8PSK", "16PSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "32PSK",
    "2FSK", "4FSK", "8FSK", "16FSK",
    "AM-SSB", "AM-DSB", "FM", "GMSK", "MSK", "OQPSK",
    "CPFSK", "64PSK",
    "OFDM-QPSK", "OFDM-16QAM", "OFDM-64QAM", "OFDM-256QAM",
    "WCDMA", "MC-GSM", "Noise"
  };

  for (int s = 0; s < NUM_DDC_SLOTS; s++) {
    ImGui::PushID(s);

    // Color-coded header: red = zoom only, green = DDC active
    ImVec4 header_color = ddc_slots_[s].ddc_running
        ? ImVec4(0.1f, 0.6f, 0.1f, 1.0f)   // green
        : ddc_slots_[s].active
            ? ImVec4(0.7f, 0.15f, 0.15f, 1.0f)  // red
            : ImVec4(0.3f, 0.3f, 0.3f, 1.0f);   // gray

    char header[64];
    snprintf(header, sizeof(header), "DDC %d%s", s,
             ddc_slots_[s].ddc_running ? " [ACTIVE]" :
             ddc_slots_[s].active ? " [SELECT BW]" : "");

    ImGui::PushStyleColor(ImGuiCol_Header, header_color);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(header_color.x * 1.2f,
        header_color.y * 1.2f, header_color.z * 1.2f, 1.0f));

    if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ddc_slots_[s].active) {
        float col_width = ImGui::GetContentRegionAvail().x * 0.5f;

        // Left column: DDC info + BW selection
        ImGui::BeginChild(("##ddc_info_" + std::to_string(s)).c_str(),
                          ImVec2(col_width, 160), false);
        ImGui::Text("Antenna:  %d", ddc_slots_[s].antenna);
        ImGui::Text("Center:   %.4f MHz", ddc_slots_[s].center_mhz);
        ImGui::Text("Zoom:     %.2f - %.2f MHz", ddc_slots_[s].freq_min_mhz, ddc_slots_[s].freq_max_mhz);

        // Bandwidth preset dropdown
        ImGui::Spacing();
        const char* bw_label = ddc_slots_[s].bw_selection_idx >= 0
            ? BW_PRESETS[ddc_slots_[s].bw_selection_idx].label : "-- Select BW --";
        ImGui::SetNextItemWidth(col_width - 10);
        if (ImGui::BeginCombo("##bw", bw_label)) {
          double max_bw = cfg_.freq_max_mhz - cfg_.freq_min_mhz;
          for (int b = 0; b < NUM_BW_PRESETS; b++) {
            if (BW_PRESETS[b].bw_mhz <= max_bw) {
              bool selected = (ddc_slots_[s].bw_selection_idx == b);
              if (ImGui::Selectable(BW_PRESETS[b].label, selected)) {
                ddc_slots_[s].bw_selection_idx = b;
                ddc_slots_[s].ddc_bw_mhz = BW_PRESETS[b].bw_mhz;
                ddc_slots_[s].ddc_running = true;
                update_slot_zoom(s);
              }
            }
          }
          ImGui::EndCombo();
        }

        if (ddc_slots_[s].ddc_running) {
          ImGui::Text("DDC BW:   %.3f MHz", ddc_slots_[s].ddc_bw_mhz);
        }

        if (ImGui::Button("Clear")) {
          ddc_slots_[s] = DDCSlot{};
          ImGui::EndChild();
          ImGui::PopStyleColor(2);
          ImGui::PopID();
          ImGui::Spacing();
          ImGui::Separator();
          ImGui::Spacing();
          continue;
        }
        ImGui::EndChild();

        // Right column: Classification (only when DDC is running)
        ImGui::SameLine();
        ImGui::BeginChild(("##ddc_class_" + std::to_string(s)).c_str(),
                          ImVec2(col_width, 160), false);
        if (ddc_slots_[s].ddc_running) {
          if (ddc_slots_[s].class_id >= 0) {
            int cid = ddc_slots_[s].class_id;
            const char* name = (cid >= 0 && cid < 31) ? class_names[cid] : "???";
            ImGui::TextColored(ImVec4(0, 1, 1, 1), "Modulation: %s", name);
            ImGui::Text("Confidence: %.1f%%", ddc_slots_[s].confidence * 100.0f);
            ImGui::Spacing();
            ImGui::Text("Top-3:");
            for (int t = 0; t < 3; t++) {
              int tid = ddc_slots_[s].top3_ids[t];
              const char* tname = (tid >= 0 && tid < 31) ? class_names[tid] : "???";
              ImGui::Text(" %d. %s %.1f%%", t + 1, tname,
                          ddc_slots_[s].top3_confidence[t] * 100.0f);
            }
          } else {
            ImGui::TextDisabled("Classifying...");
          }
        } else {
          ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Select bandwidth");
          ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "to activate DDC");
        }
        ImGui::EndChild();

        // Zoomed FFT plot
        if (cur_read_idx_ >= 0 && ddc_slots_[s].bin_end > ddc_slots_[s].bin_start &&
            ddc_slots_[s].antenna < cfg_.num_antennas) {
          const auto& frame = buffers_[cur_read_idx_];
          int zoom_bins = ddc_slots_[s].bin_end - ddc_slots_[s].bin_start;
          const float* data = frame.antenna_data[ddc_slots_[s].antenna].data() + ddc_slots_[s].bin_start;
          const float* freq = freq_axis_.data() + ddc_slots_[s].bin_start;

          float plot_h = slot_height - 180.0f;
          if (plot_h < 80.0f) plot_h = 80.0f;

          char plot_label[64];
          snprintf(plot_label, sizeof(plot_label), "##ddc_plot_%d", s);

          ImVec4 line_color = ddc_slots_[s].ddc_running
              ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)    // green when active
              : ImVec4(1.0f, 0.3f, 0.3f, 1.0f);    // red when zoom only

          ImPlot::PushStyleColor(ImPlotCol_Line, line_color);
          if (ImPlot::BeginPlot(plot_label, ImVec2(-1, plot_h), ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("MHz", "dBFS");
            ImPlot::SetupAxisLimits(ImAxis_X1, ddc_slots_[s].freq_min_mhz, ddc_slots_[s].freq_max_mhz, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, cfg_.db_min, cfg_.db_max, ImPlotCond_Always);

            ImPlot::PlotLine("##zoom", freq, data, zoom_bins);
            ImPlot::EndPlot();
          }
          ImPlot::PopStyleColor();
        }
      } else {
        ImGui::TextDisabled("No selection — drag on a plot to assign");
      }
    }
    ImGui::PopStyleColor(2);
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
  }
}

#endif  // HAS_IMGUI
