#pragma once

#ifdef HAS_IMGUI

#include "holoscan/holoscan.hpp"
#include "imgui_display.h"
#include "ddc_channel.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cmath>

namespace holoscan::ops {

class SpectralOp;

class ImGuiSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImGuiSinkOp)
  ImGuiSinkOp() = default;

  void set_display(std::shared_ptr<ImGuiDisplay> display) { display_ = display; }
  void set_ddc_channels(DDCChannel* channels, int count) {
    ddc_channels_ = channels;
    num_ddc_ = count;
  }
  void set_signal_params(double sample_rate_mhz, double center_freq_mhz, int fft_size) {
    sample_rate_mhz_ = sample_rate_mhz;
    center_freq_mhz_ = center_freq_mhz;
    fft_size_ = fft_size;
  }

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in");
    spec.param(num_antennas_, "num_antennas", "Antennas", "Number of antenna channels", 1);
    spec.param(num_bins_, "num_bins", "Display bins", "Number of frequency bins", 1024);
  }

  void initialize() override {
    Operator::initialize();
    host_buf_.resize(num_bins_.get());
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    if (!display_ || display_->should_close()) return;

    auto in = op_input.receive<holoscan::TensorMap>("in");
    if (!in) return;

    const auto& input = *in;
    const int A = num_antennas_.get();
    const int bins = num_bins_.get();

    for (int a = 0; a < A; a++) {
      std::string key = "antenna_" + std::to_string(a);
      auto it = input.find(key);
      if (it == input.end()) continue;

      const auto& tensor = it->second;
      const float* gpu_data = static_cast<const float*>(tensor->data());

      cudaMemcpy(host_buf_.data(), gpu_data, bins * sizeof(float), cudaMemcpyDeviceToHost);
      display_->submit_frame(a, host_buf_.data(), bins);
    }

    // Sync GUI DDC selections → DDC channel activation
    if (ddc_channels_) {
      for (int s = 0; s < num_ddc_; s++) {
        const auto& slot = display_->get_ddc_slot(s);

        if (slot.ddc_running && !ddc_channels_[s].is_active()) {
          DDCChannel::Config ddc_cfg;
          ddc_cfg.center_freq_mhz = slot.center_mhz;
          ddc_cfg.lo_freq_mhz = center_freq_mhz_;
          ddc_cfg.bandwidth_mhz = slot.ddc_bw_mhz;
          ddc_cfg.sample_rate_mhz = sample_rate_mhz_;
          ddc_cfg.input_length = fft_size_;
          ddc_cfg.output_length = 4096;
          ddc_cfg.antenna = slot.antenna;
          ddc_channels_[s].activate(ddc_cfg);
        } else if (slot.ddc_running && ddc_channels_[s].is_active() &&
                   std::abs(ddc_channels_[s].bandwidth_mhz() - slot.ddc_bw_mhz) > 0.001) {
          DDCChannel::Config ddc_cfg;
          ddc_cfg.center_freq_mhz = slot.center_mhz;
          ddc_cfg.lo_freq_mhz = center_freq_mhz_;
          ddc_cfg.bandwidth_mhz = slot.ddc_bw_mhz;
          ddc_cfg.sample_rate_mhz = sample_rate_mhz_;
          ddc_cfg.input_length = fft_size_;
          ddc_cfg.output_length = 4096;
          ddc_cfg.antenna = slot.antenna;
          ddc_channels_[s].activate(ddc_cfg);
        } else if (!slot.active && ddc_channels_[s].is_active()) {
          // Cleared — deactivate
          ddc_channels_[s].deactivate();
        }

        // Push classification results back to display
        if (ddc_channels_[s].is_active()) {
          const auto& result = ddc_channels_[s].classification();
          if (result.class_id >= 0) {
            display_->update_classification(s, result.class_id, result.confidence,
                                            result.top3_ids, result.top3_confidence);
          }
        }
      }
    }

    display_->frame_complete();
  }

 private:
  std::shared_ptr<ImGuiDisplay> display_;
  DDCChannel* ddc_channels_ = nullptr;
  int num_ddc_ = 0;
  double sample_rate_mhz_ = 0;
  double center_freq_mhz_ = 0;
  int fft_size_ = 0;
  std::vector<float> host_buf_;

  Parameter<int> num_antennas_;
  Parameter<int> num_bins_;
};

}  // namespace holoscan::ops

#endif  // HAS_IMGUI
