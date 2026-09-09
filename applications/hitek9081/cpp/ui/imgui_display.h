#pragma once

#ifdef HAS_IMGUI

#include <atomic>
#include <array>
#include <vector>
#include <thread>
#include <functional>

class ImGuiDisplay {
 public:
  using CloseCallback = std::function<void()>;

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

  // Called from Holoscan pipeline thread. Render thread reads via atomic latest_ready_.
  void submit_frame(int antenna, const float* db_data, int num_bins);
  void frame_complete();
  bool should_close() const;
  void request_close() { close_requested_.store(true); }
  void set_close_callback(CloseCallback cb) { close_cb_ = std::move(cb); }
  void update_classification(int slot, int class_id, float confidence,
                             const int* top3_ids, const float* top3_conf,
                             int family_id, float family_confidence);

 private:
  void render_loop();
  void render_main_panel();
  void render_ddc_window_content();
  void render_ddc_table();
  void render_ddc_detail(int slot);
  void render_line_plot(int antenna, float width, float height);
  void render_waterfall_plot(int antenna, float width, float height);
  void handle_selection(int antenna, double center_freq);
  void update_slot_zoom(int slot);

  // Quad-buffer: writer skips buffer held by reader. 4 needed because pipeline can lap render at high frame rates.
  static constexpr int NUM_BUFS = 4;

  struct FrameBuffer {
    std::vector<std::vector<float>> antenna_data;
  };

  Config cfg_;
  std::array<FrameBuffer, NUM_BUFS> buffers_;
  std::atomic<int> latest_ready_{-1};
  std::atomic<int> reading_{-1};
  std::atomic<uint64_t> pipeline_frames_{0};
  int write_idx_ = 0;

  // OpenGL texture-based waterfall
  std::vector<unsigned int> wf_tex_;
  bool wf_tex_initialized_ = false;
  std::vector<uint32_t> wf_row_rgba_;
  uint32_t viridis_lut_[256] = {};
  bool viridis_lut_initialized_ = false;
  int waterfall_write_pos_ = 0;
  void init_waterfall_textures();
  void destroy_waterfall_textures();

  // X-axis values for ImPlot (precomputed frequency array)
  std::vector<float> freq_axis_;


  static constexpr int NUM_DDC_SLOTS = 16;

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
    int family_id = -1;
    float family_confidence = 0.0f;
  };

  struct BWPreset {
    const char* label;
    double bw_mhz;
  };
  static constexpr BWPreset BW_PRESETS[] = {
    {"1.4 MHz",   1.4},
    {"3 MHz",     3.0},
    {"5 MHz",     5.0},
    {"10 MHz",    10.0},
    {"15 MHz",    15.0},
    {"20 MHz",    20.0},
    {"40 MHz",    40.0},
    {"50 MHz",    50.0},
    {"60 MHz",    60.0},
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
  int selected_ddc_ = -1;
  int last_waterfall_idx_ = -1;
  double waterfall_frame_time_ = 0;  // estimated seconds per waterfall row

  std::atomic<bool> running_{true};
  std::atomic<bool> closed_{false};
  std::atomic<bool> close_requested_{false};
  CloseCallback close_cb_;
  std::thread thread_;
};

#endif  // HAS_IMGUI
