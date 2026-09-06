/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "ddc_channel.h"
#include "ddc_kernels.cuh"
#include "amc_constants.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>

class AMCInference {
 public:
  static constexpr int NUM_CLASSES = AMC_NUM_CLASSES;
  static constexpr const char* CLASS_NAMES[NUM_CLASSES] = {
    "ook", "4ask", "8ask", "16ask", "32ask", "64ask",
    "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk",
    "8fsk", "8gfsk", "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
    "bpsk", "qpsk", "8psk", "16psk", "32psk", "64psk",
    "16qam", "32qam", "32qam_cross", "64qam", "128qam_cross", "256qam",
    "512qam_cross", "1024qam", "16apsk", "32apsk",
    "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
    "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048",
    "fm", "am-dsb-sc", "am-dsb", "am-lsb", "am-usb",
    "lfm-data", "lfm-radar", "chirpss", "tone",
    "adsb-long", "adsb-short", "btle", "gsm", "dvbs2", "dmr", "p25", "lora",
    "80211a", "zigbee",
  };

  // Family mapping: class index → family ID.
  // Indices must match TorchSig signal_lists.py SIGNALS_SHARED_LIST order exactly.
  enum Family { FAM_OOK = 0, FAM_ASK, FAM_FSK, FAM_PSK, FAM_QAM, FAM_OFDM,
                FAM_FM, FAM_AM, FAM_CHIRP, FAM_TONE,
                FAM_SPREAD, FAM_TDMA, FAM_SAT, FAM_BURST, NUM_FAMILIES };
  static_assert(NUM_FAMILIES == AMC_NUM_FAMILIES, "AMC_NUM_FAMILIES mismatch");
  static constexpr const char* FAMILY_NAMES[NUM_FAMILIES] = {
    "OOK", "ASK", "FSK", "PSK", "QAM", "OFDM", "FM", "AM", "CHIRP", "TONE",
    "SPREAD", "TDMA", "SAT", "BURST"
  };
  static constexpr int CLASS_FAMILY[NUM_CLASSES] = {
    FAM_OOK,                                                          // ook
    FAM_ASK, FAM_ASK, FAM_ASK, FAM_ASK, FAM_ASK,                    // 4-64ask
    FAM_FSK, FAM_FSK, FAM_FSK, FAM_FSK,                             // 2fsk/gfsk/msk/gmsk
    FAM_FSK, FAM_FSK, FAM_FSK, FAM_FSK,                             // 4fsk/gfsk/msk/gmsk
    FAM_FSK, FAM_FSK, FAM_FSK, FAM_FSK,                             // 8fsk/gfsk/msk/gmsk
    FAM_FSK, FAM_FSK, FAM_FSK, FAM_FSK,                             // 16fsk/gfsk/msk/gmsk
    FAM_PSK, FAM_PSK, FAM_PSK, FAM_PSK, FAM_PSK, FAM_PSK,          // bpsk-64psk
    FAM_QAM, FAM_QAM, FAM_QAM, FAM_QAM, FAM_QAM, FAM_QAM,        // 16qam-256qam
    FAM_QAM, FAM_QAM,                                                // 512qam_cross, 1024qam
    FAM_PSK, FAM_PSK,                                                // 16apsk, 32apsk
    FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM,  // ofdm-64 to ofdm-300
    FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM, FAM_OFDM,  // ofdm-512 to ofdm-2048
    FAM_FM,                                                           // fm
    FAM_AM, FAM_AM, FAM_AM, FAM_AM,                                  // am variants
    FAM_CHIRP, FAM_CHIRP, FAM_CHIRP,                                 // lfm-data, lfm-radar, chirpss
    FAM_TONE,                                                         // tone
    FAM_BURST, FAM_BURST,                                             // adsb-long, adsb-short
    FAM_BURST,                                                        // btle
    FAM_TDMA,                                                         // gsm
    FAM_SAT,                                                          // dvbs2
    FAM_TDMA, FAM_TDMA,                                               // dmr, p25
    FAM_SPREAD,                                                       // lora
    FAM_OFDM, FAM_SPREAD,                                             // 80211a, zigbee
  };

  AMCInference() = default;
  ~AMCInference() { destroy(); }

  // STFT dimensions — must match train_amc.py iq_to_spectrogram() parameters
  static constexpr int SPEC_TIME = 128;
  static constexpr int SPEC_FREQ = 64;

  // First call builds TRT engine from ONNX (30-60s on GB10), caches as .engine.
  // Subsequent calls load cached engine instantly. Delete .engine to force rebuild.
  bool initialize(const std::string& onnx_path, int input_length = 4096) {
    input_length_ = input_length;

    std::string engine_path = onnx_path.substr(0, onnx_path.rfind('.')) + ".engine";
    if (load_engine(engine_path)) {
      fprintf(stderr, "[AMC] Loaded cached TensorRT engine: %s\n", engine_path.c_str());
      return create_context();
    }

    fprintf(stderr, "[AMC] Building TensorRT engine from %s — this takes 30-60 seconds...\n",
            onnx_path.c_str());
    fflush(stderr);
    if (!build_engine(onnx_path)) return false;
    save_engine(engine_path);
    fprintf(stderr, "[AMC] Engine built and cached: %s\n", engine_path.c_str());
    return create_context();
  }

  void infer(DDCChannel& channel) {
    if (!context_ || !channel.is_active()) return;

    // Convert IQ to spectrogram [1, 1, 128, 64] for 2D CNN
    ddc_compute_spectrogram(channel.iq_output_gpu(), input_d_, input_length_,
                            stft_window_d_, SPEC_FREQ * 2,
                            SPEC_TIME, SPEC_FREQ, channel.stream());

    context_->setTensorAddress("spectrogram", input_d_);
    context_->setTensorAddress("class_probs", output_d_);
    context_->enqueueV3(channel.stream());

    cudaMemcpyAsync(output_h_, output_d_, NUM_CLASSES * sizeof(float),
                    cudaMemcpyDeviceToHost, channel.stream());
    cudaStreamSynchronize(channel.stream());

    softmax(output_h_, NUM_CLASSES);

    // Per-channel EMA smoothing
    float* ema = channel.ema_probs();
    bool& ema_init = channel.ema_initialized();
    if (!ema_init) {
      for (int i = 0; i < NUM_CLASSES; i++) ema[i] = output_h_[i];
      ema_init = true;
    } else {
      for (int i = 0; i < NUM_CLASSES; i++) {
        ema[i] = EMA_ALPHA * output_h_[i] + (1.0f - EMA_ALPHA) * ema[i];
      }
    }

    // Sort smoothed probabilities
    std::vector<std::pair<float, int>> sorted_probs;
    for (int i = 0; i < NUM_CLASSES; i++) {
      sorted_probs.push_back({ema[i], i});
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Hysteresis: only change class if new winner beats current by a margin
    DDCChannel::ClassificationResult result = channel.classification();
    int new_top = sorted_probs[0].second;
    float new_conf = sorted_probs[0].first;

    if (result.class_id < 0 ||
        (new_top != result.class_id && new_conf > result.confidence + HYSTERESIS_MARGIN)) {
      result.class_id = new_top;
    }
    result.confidence = new_conf;
    for (int i = 0; i < 3; i++) {
      result.top3_ids[i] = sorted_probs[i].second;
      result.top3_confidence[i] = sorted_probs[i].first;
    }

    // Family confidence: sum smoothed probabilities per family
    float family_conf[NUM_FAMILIES] = {};
    for (int i = 0; i < NUM_CLASSES; i++) {
      family_conf[CLASS_FAMILY[i]] += ema[i];
    }
    int best_family = 0;
    for (int f = 1; f < NUM_FAMILIES; f++) {
      if (family_conf[f] > family_conf[best_family]) best_family = f;
    }
    result.family_id = best_family;
    result.family_confidence = family_conf[best_family];

    channel.set_classification(result);
  }

  bool is_ready() const { return context_ != nullptr; }

 private:
  bool build_engine(const std::string& onnx_path) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE))) {
      int n = parser->getNbErrors();
      for (int i = 0; i < n; i++) {
        fprintf(stderr, "[TensorRT ONNX] Error %d: %s\n", i, parser->getError(i)->desc());
      }
      return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 28);

    // Set optimization profile for dynamic batch dimension
    auto profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    auto dims = input->getDimensions();
    // min=1, opt=1, max=1 (single inference, no batching needed)
    nvinfer1::Dims4 shape(1, dims.d[1], dims.d[2], 1);
    nvinfer1::Dims inputDims;
    inputDims.nbDims = dims.nbDims;
    for (int i = 0; i < dims.nbDims; i++) inputDims.d[i] = (dims.d[i] == -1) ? 1 : dims.d[i];
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, inputDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) return false;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    return engine_ != nullptr;
  }

  bool create_context() {
    if (!engine_) return false;
    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    // Spectrogram input buffer [SPEC_TIME * SPEC_FREQ]
    if (cudaMalloc(&input_d_, SPEC_TIME * SPEC_FREQ * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&output_d_, NUM_CLASSES * sizeof(float)) != cudaSuccess) return false;

    // STFT Hanning window (win_len = SPEC_FREQ * 2)
    int win_len = SPEC_FREQ * 2;
    if (cudaMalloc(&stft_window_d_, win_len * sizeof(float)) != cudaSuccess) return false;
    // Generate Hanning window on GPU
    std::vector<float> win(win_len);
    for (int i = 0; i < win_len; i++) {
      win[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * i / win_len));
    }
    cudaMemcpy(stft_window_d_, win.data(), win_len * sizeof(float), cudaMemcpyHostToDevice);

    return true;
  }

  bool load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime->deserializeCudaEngine(data.data(), size));
    return engine_ != nullptr;
  }

  void save_engine(const std::string& path) {
    if (!engine_) return;
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
  }

  static void softmax(float* data, int n) {
    float max_val = *std::max_element(data, data + n);
    float sum = 0;
    for (int i = 0; i < n; i++) {
      data[i] = std::exp(data[i] - max_val);
      sum += data[i];
    }
    for (int i = 0; i < n; i++) data[i] /= sum;
  }

  void destroy() {
    context_.reset();
    engine_.reset();
    if (input_d_) { cudaFree(input_d_); input_d_ = nullptr; }
    if (output_d_) { cudaFree(output_d_); output_d_ = nullptr; }
    if (stft_window_d_) { cudaFree(stft_window_d_); stft_window_d_ = nullptr; }
  }

  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kERROR) {
        fprintf(stderr, "[TensorRT %d] %s\n", static_cast<int>(severity), msg);
      }
    }
  };

  Logger logger_;
  int input_length_ = 4096;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  // EMA smoothing: 30% new frame + 70% history. Lower α = more stable, slower response.
  // Tuned for ~7 Hz FFT frame rate — 0.3 gives ~1s settling time.
  static constexpr float EMA_ALPHA = 0.3f;
  // Hysteresis: displayed class only changes if new winner exceeds current by this margin.
  // Prevents flickering on TDD signals where class confidence oscillates.
  static constexpr float HYSTERESIS_MARGIN = 0.05f;

  float* input_d_ = nullptr;
  float* output_d_ = nullptr;
  float* stft_window_d_ = nullptr;
  float output_h_[NUM_CLASSES] = {};
};
