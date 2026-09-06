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

// Shared constants between DDCChannel and AMCInference.
// Avoids circular header dependency.
#pragma once

static constexpr int AMC_NUM_CLASSES = 69;   // must match TorchSig MODULATION_CLASSES length
// Families: OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, SPREAD, TDMA, SAT, BURST
static constexpr int AMC_NUM_FAMILIES = 14;
