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

void ImGuiDisplay::handle_selection(int antenna, double xmin, double xmax) {
  // Reject selections narrower than 0.1% of display bandwidth (accidental clicks)
  double min_bw = (cfg_.freq_max_mhz - cfg_.freq_min_mhz) * 0.001;
  if ((xmax - xmin) < min_bw) return;

  // Clamp to display range
  if (xmin < cfg_.freq_min_mhz) xmin = cfg_.freq_min_mhz;
  if (xmax > cfg_.freq_max_mhz) xmax = cfg_.freq_max_mhz;

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
  slot.active = true;
  slot.antenna = antenna;
  slot.freq_min_mhz = xmin;
  slot.freq_max_mhz = xmax;
  slot.center_mhz = (xmin + xmax) / 2.0;

  double freq_step = (cfg_.freq_max_mhz - cfg_.freq_min_mhz) / (cfg_.num_bins - 1);
  slot.bin_start = std::max(0, (int)((xmin - cfg_.freq_min_mhz) / freq_step));
  slot.bin_end = std::min(cfg_.num_bins, (int)((xmax - cfg_.freq_min_mhz) / freq_step) + 1);
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

    // Manual drag selection using raw mouse input within the plot
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      drag_start_freq_ = ImPlot::GetPlotMousePos().x;
      dragging_ = true;
      drag_antenna_ = antenna;
    }
    if (dragging_ && drag_antenna_ == antenna && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      double end_freq = ImPlot::GetPlotMousePos().x;
      double fmin = std::min(drag_start_freq_, end_freq);
      double fmax = std::max(drag_start_freq_, end_freq);
      handle_selection(antenna, fmin, fmax);
      dragging_ = false;
    }
    // Draw selection rectangle while dragging
    if (dragging_ && drag_antenna_ == antenna && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      double cur_freq = ImPlot::GetPlotMousePos().x;
      double fmin = std::min(drag_start_freq_, cur_freq);
      double fmax = std::max(drag_start_freq_, cur_freq);
      drag_rect_ymin_ = cfg_.db_min;
      drag_rect_ymax_ = cfg_.db_max;
      ImPlot::DragRect(antenna + 100, &fmin, &drag_rect_ymin_, &fmax, &drag_rect_ymax_,
                       ImVec4(1, 1, 0, 0.25f), ImPlotDragToolFlags_NoInputs);
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

    // Manual drag selection
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      drag_start_freq_ = ImPlot::GetPlotMousePos().x;
      dragging_ = true;
      drag_antenna_ = antenna;
    }
    if (dragging_ && drag_antenna_ == antenna && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      double end_freq = ImPlot::GetPlotMousePos().x;
      double fmin = std::min(drag_start_freq_, end_freq);
      double fmax = std::max(drag_start_freq_, end_freq);
      handle_selection(antenna, fmin, fmax);
      dragging_ = false;
    }
    if (dragging_ && drag_antenna_ == antenna && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      double cur_freq = ImPlot::GetPlotMousePos().x;
      double fmin = std::min(drag_start_freq_, cur_freq);
      double fmax = std::max(drag_start_freq_, cur_freq);
      drag_rect_ymin_ = 0.0;
      drag_rect_ymax_ = (double)cfg_.waterfall_history;
      ImPlot::DragRect(antenna + 100, &fmin, &drag_rect_ymin_, &fmax, &drag_rect_ymax_,
                       ImVec4(1, 1, 0, 0.25f), ImPlotDragToolFlags_NoInputs);
    }

    ImPlot::EndPlot();
  }
}

void ImGuiDisplay::render_ddc_panel() {

  ImGui::Text("Software DDC Channels");
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::TextWrapped("Left-click drag on any antenna plot to assign a DDC channel.");
  ImGui::Spacing();

  ImVec2 avail = ImGui::GetContentRegionAvail();
  float slot_height = avail.y / NUM_DDC_SLOTS;

  static const ImVec4 slot_colors[] = {
    {0.0f, 1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f, 1.0f},
  };

  for (int s = 0; s < NUM_DDC_SLOTS; s++) {
    ImGui::PushID(s);
    char header[64];
    snprintf(header, sizeof(header), "DDC %d", s);

    if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ddc_slots_[s].active) {
        ImGui::Text("Antenna:  %d", ddc_slots_[s].antenna);
        ImGui::Text("Center:   %.4f MHz", ddc_slots_[s].center_mhz);
        ImGui::Text("Range:    %.4f - %.4f MHz", ddc_slots_[s].freq_min_mhz, ddc_slots_[s].freq_max_mhz);
        ImGui::Text("BW:       %.4f MHz", ddc_slots_[s].freq_max_mhz - ddc_slots_[s].freq_min_mhz);
        ImGui::Text("Bins:     %d - %d (%d)", ddc_slots_[s].bin_start, ddc_slots_[s].bin_end,
                    ddc_slots_[s].bin_end - ddc_slots_[s].bin_start);

        if (ImGui::Button("Clear")) {
          ddc_slots_[s] = DDCSlot{};
          ImGui::PopID();
          ImGui::Spacing();
          ImGui::Separator();
          ImGui::Spacing();
          continue;
        }

        // Zoomed FFT plot
        if (cur_read_idx_ >= 0 && ddc_slots_[s].bin_end > ddc_slots_[s].bin_start &&
            ddc_slots_[s].antenna < cfg_.num_antennas) {
          const auto& frame = buffers_[cur_read_idx_];
          int zoom_bins = ddc_slots_[s].bin_end - ddc_slots_[s].bin_start;
          const float* data = frame.antenna_data[ddc_slots_[s].antenna].data() + ddc_slots_[s].bin_start;
          const float* freq = freq_axis_.data() + ddc_slots_[s].bin_start;

          float plot_h = slot_height - 120.0f;
          if (plot_h < 80.0f) plot_h = 80.0f;

          char plot_label[64];
          snprintf(plot_label, sizeof(plot_label), "##ddc_plot_%d", s);

          ImPlot::PushStyleColor(ImPlotCol_Line, slot_colors[s % 2]);
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
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
  }
}

#endif  // HAS_IMGUI
