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
  void update_classification(int slot, int class_id, float confidence,
                             const int* top3_ids, const float* top3_conf);

 private:
  void render_loop();
  void render_main_panel();
  void render_ddc_panel();
  void render_line_plot(int antenna, float width, float height);
  void render_waterfall_plot(int antenna, float width, float height);
  void handle_selection(int antenna, double center_freq);
  void update_slot_zoom(int slot);

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
    bool active = false;        // has a zoom selection
    bool ddc_running = false;   // DDC activated with a preset BW
    int antenna = 0;
    double freq_min_mhz = 0;   // zoom view range (from drag)
    double freq_max_mhz = 0;
    double center_mhz = 0;     // center of drag
    double ddc_bw_mhz = 0;     // selected preset bandwidth (0 = not selected)
    int bin_start = 0;          // zoom view bins
    int bin_end = 0;
    int bw_selection_idx = -1;  // index into BW_PRESETS
    // Classification results
    int class_id = -1;
    float confidence = 0.0f;
    float top3_confidence[3] = {};
    int top3_ids[3] = {};
  };

  struct BWPreset {
    const char* label;
    double bw_mhz;
  };
  static constexpr BWPreset BW_PRESETS[] = {
    {"12.5 kHz",  0.0125},
    {"25 kHz",    0.025},
    {"200 kHz",   0.2},
    {"1.4 MHz",   1.4},
    {"3 MHz",     3.0},
    {"5 MHz",     5.0},
    {"10 MHz",    10.0},
    {"15 MHz",    15.0},
    {"20 MHz",    20.0},
    {"40 MHz",    40.0},
    {"50 MHz",    50.0},
    {"80 MHz",    80.0},
    {"100 MHz",   100.0},
    {"160 MHz",   160.0},
  };
  static constexpr int NUM_BW_PRESETS = sizeof(BW_PRESETS) / sizeof(BW_PRESETS[0]);
  std::array<DDCSlot, NUM_DDC_SLOTS> ddc_slots_;
  int next_slot_ = 0;

 public:
  const DDCSlot& get_ddc_slot(int s) const { return ddc_slots_[s]; }
 private:


  // Current frame's read index (set once per render frame)
  int cur_read_idx_ = -1;
  int last_waterfall_idx_ = -1;
  double waterfall_frame_time_ = 0;  // estimated seconds per waterfall row

  std::atomic<bool> running_{true};
  std::atomic<bool> closed_{false};
  std::thread thread_;
};

#endif  // HAS_IMGUI
