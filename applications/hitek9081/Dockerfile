# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# HTK9081 Spectrum Analyzer Development Container
################################################################################
# This container layers application-specific AI inference dependencies on top
# of the shared HoloHub networking image (which provides DPDK, DAQIRI, and HSB).
################################################################################

# Base networking image containing DPDK, Mellanox OFED, DAQIRI, and Hololink
ARG NETWORKING_BASE=holohub:holoscan-networking
FROM ${NETWORKING_BASE}

ARG DEBIAN_FRONTEND=noninteractive

# Dependencies:
# - TensorRT development libraries for HTK9081 Automatic Modulation Classification (AMC)
#   Provides NvInfer.h, NvOnnxParser.h, libnvinfer.so, and libnvonnxparser.so
# - X11 & Virtual Framebuffer utilities for GUI and headless rendering
#   Provides xauth, x11-utils (xdpyinfo), xvfb (headless display), and mesa-utils (glxinfo)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnvinfer-dev \
        libnvinfer-plugin-dev \
        libnvonnxparsers-dev \
        xauth \
        x11-utils \
        xvfb \
        mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Ensure wrapper-pinned holoscan-cli is present for in-container builds
COPY --chmod=755 holohub /tmp/scripts/
RUN /tmp/scripts/holohub env-info

# Default working directory for HoloHub builds
WORKDIR /workspace/holohub
