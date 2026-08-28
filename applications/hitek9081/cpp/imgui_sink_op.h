#pragma once

#ifdef HAS_IMGUI

#include "holoscan/holoscan.hpp"
#include "imgui_display.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace holoscan::ops {

class ImGuiSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImGuiSinkOp)
  ImGuiSinkOp() = default;

  void set_display(std::shared_ptr<ImGuiDisplay> display) { display_ = display; }

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

    display_->frame_complete();
  }

 private:
  std::shared_ptr<ImGuiDisplay> display_;
  std::vector<float> host_buf_;

  Parameter<int> num_antennas_;
  Parameter<int> num_bins_;
};

}  // namespace holoscan::ops

#endif  // HAS_IMGUI
