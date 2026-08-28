#pragma once

#ifdef HAS_IMGUI

#include <atomic>
#include <array>
#include <vector>
#include <thread>
#include <mutex>

class ImGuiDisplay {
 public:
  struct Config {
    double freq_min_mhz;
    double freq_max_mhz;
    double center_freq_mhz;
    float db_min;
    float db_max;
    int num_antennas;
    int num_bins;
    bool waterfall_mode;
    int waterfall_history = 512;
    double waterfall_time_span = 60.0;  // total visible time in seconds
  };

  explicit ImGuiDisplay(const Config& cfg);
  ~ImGuiDisplay();

  void submit_frame(int antenna, const float* db_data, int num_bins);
  void frame_complete();
  bool should_close() const;

 private:
  void render_loop();
  void render_main_panel();
  void render_ddc_panel();
  void render_line_plot(int antenna, float width, float height);
  void render_waterfall_plot(int antenna, float width, float height);
  void handle_selection(int antenna, double xmin, double xmax);

  static constexpr int NUM_BUFS = 4;

  struct FrameBuffer {
    std::vector<std::vector<float>> antenna_data;
  };

  Config cfg_;
  std::array<FrameBuffer, NUM_BUFS> buffers_;
  std::atomic<int> latest_ready_{-1};
  std::atomic<int> reading_{-1};
  int write_idx_ = 0;

  // Waterfall circular buffer: [antenna][history * num_bins]
  std::vector<std::vector<float>> waterfall_buf_;
  std::vector<float> waterfall_render_buf_;  // pre-allocated reorder buffer
  int waterfall_write_pos_ = 0;

  // X-axis values for ImPlot (precomputed frequency array)
  std::vector<float> freq_axis_;


  static constexpr int NUM_DDC_SLOTS = 2;

  struct DDCSlot {
    bool active = false;
    int antenna = 0;
    double freq_min_mhz = 0;
    double freq_max_mhz = 0;
    double center_mhz = 0;
    int bin_start = 0;
    int bin_end = 0;
  };
  std::array<DDCSlot, NUM_DDC_SLOTS> ddc_slots_;
  int next_slot_ = 0;

  // Drag state for manual frequency selection
  bool dragging_ = false;
  int drag_antenna_ = -1;
  double drag_start_freq_ = 0;
  double drag_rect_ymin_ = 0;
  double drag_rect_ymax_ = 0;

  // Current frame's read index (set once per render frame)
  int cur_read_idx_ = -1;
  int last_waterfall_idx_ = -1;
  double waterfall_frame_time_ = 0;  // estimated seconds per waterfall row

  std::atomic<bool> running_{true};
  std::atomic<bool> closed_{false};
  std::thread thread_;
};

#endif  // HAS_IMGUI
