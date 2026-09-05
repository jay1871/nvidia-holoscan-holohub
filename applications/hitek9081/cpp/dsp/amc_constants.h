// Shared constants between DDCChannel and AMCInference.
// Avoids circular header dependency.
#pragma once

static constexpr int AMC_NUM_CLASSES = 69;     // must match TorchSig MODULATION_CLASSES length
static constexpr int AMC_NUM_FAMILIES = 11;    // OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, PROTOCOL
