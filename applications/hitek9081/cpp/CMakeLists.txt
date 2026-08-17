# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "90;120;121" CACHE STRING "" FORCE)

project(hitek9081 CXX CUDA)

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)

if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

# ---------------------------------------------------------------------------
# Fetch hololink source — only build the core library (for FPGA bringup).
# ---------------------------------------------------------------------------
find_package(Git REQUIRED)
include(FetchContent)
FetchContent_Declare(hololink
  GIT_REPOSITORY https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
  GIT_TAG        fc2a6cfd7e85cd04d771fbf03ac45e3e7c57b5e1
  GIT_SHALLOW    FALSE
  PATCH_COMMAND  /usr/bin/git checkout -- . && /usr/bin/git apply "${CMAKE_CURRENT_LIST_DIR}/htk_esom5u_core.patch"
)
FetchContent_GetProperties(hololink)
if(NOT hololink_POPULATED)
  FetchContent_Populate(hololink)
endif()

set(HOLOLINK_SRC "${hololink_SOURCE_DIR}/src")
find_package(fmt REQUIRED)

# hololink_core (Enumerator, DataChannel, Hololink — FPGA bringup)
add_library(hololink_core STATIC
  ${HOLOLINK_SRC}/hololink/core/arp_wrapper.cpp
  ${HOLOLINK_SRC}/hololink/core/data_channel.cpp
  ${HOLOLINK_SRC}/hololink/core/enumerator.cpp
  ${HOLOLINK_SRC}/hololink/core/hololink.cpp
  ${HOLOLINK_SRC}/hololink/core/jesd.cpp
  ${HOLOLINK_SRC}/hololink/core/htk_jesd.cpp
  ${HOLOLINK_SRC}/hololink/core/logging.cpp
  ${HOLOLINK_SRC}/hololink/core/metadata.cpp
  ${HOLOLINK_SRC}/hololink/core/networking.cpp
  ${HOLOLINK_SRC}/hololink/core/packetizer_program.cpp
  ${HOLOLINK_SRC}/hololink/core/reactor.cpp
  ${HOLOLINK_SRC}/hololink/core/timeout.cpp
)
target_include_directories(hololink_core PUBLIC ${HOLOLINK_SRC})
target_link_libraries(hololink_core PUBLIC CUDA::cuda_driver fmt::fmt-header-only)
set_property(TARGET hololink_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(hitek9081
  kernels.cu
  spectral_kernels.cu
  ddc_kernels.cu
  main.cpp
)
set_target_properties(hitek9081 PROPERTIES CUDA_ARCHITECTURES "90;120;121")
target_link_libraries(hitek9081
  PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    hololink_core
    CUDA::cuda_driver
    CUDA::cufft
)
holohub_link_daqiri(hitek9081)

# Configure DAQIRI DPDK manager
target_compile_definitions(hitek9081 PUBLIC DAQIRI_MGR_DPDK=1)

# Copy config file to build directory
add_custom_target(hitek9081_config
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/../hitek9081.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../hitek9081.yaml"
)
add_dependencies(hitek9081 hitek9081_config)

# Installation
install(TARGETS hitek9081
        DESTINATION examples/hitek9081
        COMPONENT hitek9081-cpp)

install(FILES ../hitek9081.yaml
        DESTINATION examples/hitek9081
        COMPONENT hitek9081-configs)
