#ifdef HAS_IMGUI

#include "imgui_display.h"
#include "amc_inference.h"
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

// 16 distinguishable DDC overlay colors (transparent for shading)
static const ImVec4 DDC_COLORS[16] = {
  {0.2f,0.6f,1.0f,0.2f}, {1.0f,0.5f,0.0f,0.2f}, {0.0f,0.8f,0.4f,0.2f}, {0.8f,0.2f,0.8f,0.2f},
  {1.0f,1.0f,0.0f,0.2f}, {0.0f,0.8f,0.8f,0.2f}, {1.0f,0.2f,0.2f,0.2f}, {0.6f,0.4f,0.2f,0.2f},
  {0.4f,0.4f,1.0f,0.2f}, {1.0f,0.7f,0.3f,0.2f}, {0.3f,1.0f,0.6f,0.2f}, {1.0f,0.3f,0.6f,0.2f},
  {0.7f,0.7f,0.0f,0.2f}, {0.0f,0.5f,0.5f,0.2f}, {0.5f,0.0f,0.0f,0.2f}, {0.4f,0.6f,0.4f,0.2f},
};
// Opaque versions for line plots
static const ImVec4 DDC_LINE_COLORS[16] = {
  {0.2f,0.6f,1.0f,1.0f}, {1.0f,0.5f,0.0f,1.0f}, {0.0f,0.8f,0.4f,1.0f}, {0.8f,0.2f,0.8f,1.0f},
  {1.0f,1.0f,0.0f,1.0f}, {0.0f,0.8f,0.8f,1.0f}, {1.0f,0.2f,0.2f,1.0f}, {0.6f,0.4f,0.2f,1.0f},
  {0.4f,0.4f,1.0f,1.0f}, {1.0f,0.7f,0.3f,1.0f}, {0.3f,1.0f,0.6f,1.0f}, {1.0f,0.3f,0.6f,1.0f},
  {0.7f,0.7f,0.0f,1.0f}, {0.0f,0.5f,0.5f,1.0f}, {0.5f,0.0f,0.0f,1.0f}, {0.4f,0.6f,0.4f,1.0f},
};

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
                                         const int* top3_ids, const float* top3_conf,
                                         int family_id, float family_confidence) {
  if (slot < 0 || slot >= NUM_DDC_SLOTS) return;
  auto& s = ddc_slots_[slot];
  s.class_id = class_id;
  s.confidence = confidence;
  s.family_id = family_id;
  s.family_confidence = family_confidence;
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
  snprintf(title, sizeof(title), "HTK9081 Spectrum + DDC | %.1f - %.1f MHz",
           cfg_.freq_min_mhz, cfg_.freq_max_mhz);

  GLFWwindow* window = glfwCreateWindow(1920, 1080, title, nullptr, nullptr);
  if (!window) {
    fprintf(stderr, "ImGuiDisplay: failed to create window\n");
    glfwTerminate();
    closed_.store(true);
    return;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  ImPlot::GetInputMap().Select = ImGuiMouseButton_Middle;

  while (running_.load() && !glfwWindowShouldClose(window)) {
    glfwPollEvents();

    cur_read_idx_ = latest_ready_.load(std::memory_order_acquire);
    reading_.store(cur_read_idx_, std::memory_order_release);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##main", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float ddc_width = avail.x * 0.35f;
    float main_width = avail.x - ddc_width;

    // Left: spectrum/waterfall (full height)
    ImGui::BeginChild("##spectrum", ImVec2(main_width, avail.y), false);
    render_main_panel();
    ImGui::EndChild();

    // Right: DDC table + detail
    ImGui::SameLine();
    ImGui::BeginChild("##ddc", ImVec2(ddc_width, avail.y), true);
    render_ddc_window_content();
    ImGui::EndChild();

    ImGui::End();
    ImGui::Render();

    reading_.store(-1, std::memory_order_release);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
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

static constexpr int DEFAULT_BW_IDX = 5;  // 20 MHz

void ImGuiDisplay::handle_selection(int antenna, double center_freq) {
  fprintf(stderr, "[DDC] handle_selection: antenna=%d center=%.3f MHz\n", antenna, center_freq);
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
    for (int d = 0; d < NUM_DDC_SLOTS; d++) {
      if (ddc_slots_[d].ddc_running && ddc_slots_[d].antenna == antenna) {
        double ddc_lo = ddc_slots_[d].center_mhz - ddc_slots_[d].ddc_bw_mhz / 2.0;
        double ddc_hi = ddc_slots_[d].center_mhz + ddc_slots_[d].ddc_bw_mhz / 2.0;
        float xs[2] = {(float)ddc_lo, (float)ddc_hi};
        float y_top[2] = {(float)cfg_.db_max, (float)cfg_.db_max};
        float y_bot[2] = {(float)cfg_.db_min, (float)cfg_.db_min};
        ImPlot::PushStyleColor(ImPlotCol_Fill, DDC_COLORS[d % 16]);
        char shade_label[32];
        snprintf(shade_label, sizeof(shade_label), "##ddc_shade_%d", d);
        ImPlot::PlotShaded(shade_label, xs, y_top, y_bot, 2);
        ImPlot::PopStyleColor();
      }
    }

    // Single click to set DDC center frequency
    bool hovered = ImPlot::IsPlotHovered();
    bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    if (hovered && clicked) {
      double click_freq = ImPlot::GetPlotMousePos().x;
      fprintf(stderr, "[DDC] Click on antenna %d at %.3f MHz\n", antenna, click_freq);
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
    for (int d = 0; d < NUM_DDC_SLOTS; d++) {
      if (ddc_slots_[d].ddc_running && ddc_slots_[d].antenna == antenna) {
        double ddc_lo = ddc_slots_[d].center_mhz - ddc_slots_[d].ddc_bw_mhz / 2.0;
        double ddc_hi = ddc_slots_[d].center_mhz + ddc_slots_[d].ddc_bw_mhz / 2.0;
        float xs[2] = {(float)ddc_lo, (float)ddc_hi};
        float y_top[2] = {(float)actual_span, (float)actual_span};
        float y_bot[2] = {0.0f, 0.0f};
        ImPlot::PushStyleColor(ImPlotCol_Fill, DDC_COLORS[d % 16]);
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

void ImGuiDisplay::render_ddc_window_content() {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  float table_height = selected_ddc_ >= 0 ? avail.y * 0.45f : avail.y;

  ImGui::BeginChild("##ddc_table_area", ImVec2(avail.x, table_height), false);
  render_ddc_table();
  ImGui::EndChild();

  if (selected_ddc_ >= 0 && selected_ddc_ < NUM_DDC_SLOTS &&
      ddc_slots_[selected_ddc_].active) {
    ImGui::Separator();
    ImGui::BeginChild("##ddc_detail_area", ImVec2(avail.x, avail.y - table_height - 5), false);
    render_ddc_detail(selected_ddc_);
    ImGui::EndChild();
  }
}

void ImGuiDisplay::render_ddc_table() {

  ImGui::Text("DDC Channels (%d active)", [&]{
    int n = 0; for (auto& s : ddc_slots_) if (s.ddc_running) n++; return n; }());
  ImGui::Separator();

  for (int ant = 0; ant < cfg_.num_antennas; ant++) {
    char header[64];
    snprintf(header, sizeof(header), "Antenna %d", ant);
    if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::BeginTable(("##ddc_tbl_" + std::to_string(ant)).c_str(), 5,
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("DDC", ImGuiTableColumnFlags_WidthFixed, 35);
        ImGui::TableSetupColumn("Center (MHz)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("BW", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("Family", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("Conf", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableHeadersRow();

        bool has_any = false;
        for (int s = 0; s < NUM_DDC_SLOTS; s++) {
          if (!ddc_slots_[s].active || ddc_slots_[s].antenna != ant) continue;
          has_any = true;

          ImGui::PushID(s);
          ImGui::TableNextRow();

          // Highlight selected row
          bool is_selected = (selected_ddc_ == s);
          if (is_selected) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1,
                ImGui::GetColorU32(ImVec4(0.2f, 0.4f, 0.2f, 1.0f)));
          }

          ImGui::TableNextColumn();
          char label[16];
          snprintf(label, sizeof(label), "%d", s);
          if (ImGui::Selectable(label, is_selected, ImGuiSelectableFlags_SpanAllColumns)) {
            selected_ddc_ = is_selected ? -1 : s;
          }

          ImGui::TableNextColumn();
          ImGui::Text("%.3f", ddc_slots_[s].center_mhz);

          ImGui::TableNextColumn();
          if (ddc_slots_[s].ddc_running) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "%.1f", ddc_slots_[s].ddc_bw_mhz);
          } else {
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "---");
          }

          ImGui::TableNextColumn();
          if (ddc_slots_[s].family_id >= 0 && ddc_slots_[s].family_id < AMCInference::NUM_FAMILIES) {
            ImGui::Text("%s", AMCInference::FAMILY_NAMES[ddc_slots_[s].family_id]);
          } else {
            ImGui::TextDisabled("---");
          }

          ImGui::TableNextColumn();
          if (ddc_slots_[s].family_confidence > 0) {
            ImGui::Text("%.0f%%", ddc_slots_[s].family_confidence * 100.0f);
          } else {
            ImGui::TextDisabled("---");
          }

          ImGui::PopID();
        }

        if (!has_any) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TableNextColumn();
          ImGui::TextDisabled("Click spectrum to assign DDC");
        }

        ImGui::EndTable();
      }
    }
  }
}

void ImGuiDisplay::render_ddc_detail(int s) {

  ImGui::Text("DDC %d Detail", s);
  ImGui::Separator();

  float col_width = ImGui::GetContentRegionAvail().x * 0.5f;

  // Left: info + controls
  ImGui::BeginChild(("##detail_info_" + std::to_string(s)).c_str(),
                    ImVec2(col_width, 130), false);
  ImGui::Text("Antenna:  %d", ddc_slots_[s].antenna);
  ImGui::Text("Center:   %.4f MHz", ddc_slots_[s].center_mhz);
  ImGui::Text("Zoom:     %.2f - %.2f MHz", ddc_slots_[s].freq_min_mhz, ddc_slots_[s].freq_max_mhz);

  const char* bw_label = ddc_slots_[s].bw_selection_idx >= 0
      ? BW_PRESETS[ddc_slots_[s].bw_selection_idx].label : "-- Select BW --";
  ImGui::SetNextItemWidth(col_width - 10);
  if (ImGui::BeginCombo("##bw_detail", bw_label)) {
    double max_bw = cfg_.freq_max_mhz - cfg_.freq_min_mhz;
    for (int b = 0; b < NUM_BW_PRESETS; b++) {
      if (BW_PRESETS[b].bw_mhz <= max_bw) {
        bool sel = (ddc_slots_[s].bw_selection_idx == b);
        if (ImGui::Selectable(BW_PRESETS[b].label, sel)) {
          ddc_slots_[s].bw_selection_idx = b;
          ddc_slots_[s].ddc_bw_mhz = BW_PRESETS[b].bw_mhz;
          ddc_slots_[s].ddc_running = true;
          update_slot_zoom(s);
        }
      }
    }
    ImGui::EndCombo();
  }

  if (ImGui::Button("Clear")) {
    ddc_slots_[s] = DDCSlot{};
    selected_ddc_ = -1;
    ImGui::EndChild();
    return;
  }
  ImGui::EndChild();

  // Right: classification
  ImGui::SameLine();
  ImGui::BeginChild(("##detail_class_" + std::to_string(s)).c_str(),
                    ImVec2(col_width, 130), false);
  if (ddc_slots_[s].ddc_running && ddc_slots_[s].class_id >= 0) {
    int fid = ddc_slots_[s].family_id;
    const char* fam = (fid >= 0 && fid < AMCInference::NUM_FAMILIES) ? AMCInference::FAMILY_NAMES[fid] : "???";
    ImGui::TextColored(ImVec4(0, 1, 1, 1), "%s (%.0f%%)", fam,
                       ddc_slots_[s].family_confidence * 100.0f);
    int cid = ddc_slots_[s].class_id;
    const char* detail = (cid >= 0 && cid < AMCInference::NUM_CLASSES) ? AMCInference::CLASS_NAMES[cid] : "???";
    ImGui::Text("Detail: %s (%.1f%%)", detail, ddc_slots_[s].confidence * 100.0f);
    ImGui::Spacing();
    ImGui::Text("Top-3:");
    for (int t = 0; t < 3; t++) {
      int tid = ddc_slots_[s].top3_ids[t];
      const char* tname = (tid >= 0 && tid < AMCInference::NUM_CLASSES) ? AMCInference::CLASS_NAMES[tid] : "???";
      ImGui::Text(" %d. %s %.1f%%", t + 1, tname, ddc_slots_[s].top3_confidence[t] * 100.0f);
    }
  } else if (ddc_slots_[s].ddc_running) {
    ImGui::TextDisabled("Classifying...");
  } else {
    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Select bandwidth to activate");
  }
  ImGui::EndChild();

  // Zoomed FFT plot
  if (cur_read_idx_ >= 0 && ddc_slots_[s].bin_end > ddc_slots_[s].bin_start &&
      ddc_slots_[s].antenna < cfg_.num_antennas) {
    const auto& frame = buffers_[cur_read_idx_];
    int zoom_bins = ddc_slots_[s].bin_end - ddc_slots_[s].bin_start;
    const float* data = frame.antenna_data[ddc_slots_[s].antenna].data() + ddc_slots_[s].bin_start;
    const float* freq = freq_axis_.data() + ddc_slots_[s].bin_start;

    ImVec2 plot_avail = ImGui::GetContentRegionAvail();
    float plot_h = plot_avail.y - 5;
    if (plot_h < 80) plot_h = 80;

    ImPlot::PushStyleColor(ImPlotCol_Line, DDC_LINE_COLORS[s % 16]);
    if (ImPlot::BeginPlot("##detail_plot", ImVec2(-1, plot_h),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoBoxSelect)) {
      ImPlot::SetupAxes("MHz", "dBFS");
      ImPlot::SetupAxisLimits(ImAxis_X1, ddc_slots_[s].freq_min_mhz,
                              ddc_slots_[s].freq_max_mhz, ImPlotCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, cfg_.db_min, cfg_.db_max, ImPlotCond_Always);

      ImPlot::PlotLine("##zoom", freq, data, zoom_bins);

      if (ddc_slots_[s].ddc_running && ImPlot::IsPlotHovered() &&
          ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ddc_slots_[s].center_mhz = ImPlot::GetPlotMousePos().x;
        update_slot_zoom(s);
      }

      ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor();
  }
}

#endif  // HAS_IMGUI
