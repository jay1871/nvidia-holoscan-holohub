# HTK9081 Spectrum Analyzer + AMC

Real-time wideband spectrum analysis with CUDA DDC and TensorRT modulation classification.

```
Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp / ImGuiSinkOp
(DPDK zero-copy     │ (FFT, avg,      (line plot / waterfall)
 scatter-gather)    │  magnitude)
                    ├──→ DDCChannel[0] ──→ TensorRT AMC ──→ ImGui DDC Panel
                    └──→ DDCChannel[1] ──→ TensorRT AMC ──→ ImGui DDC Panel
```

## Requirements

- DGX Spark (GB10, sm_121) or Hopper/Blackwell GPU with unified memory
- HSB FPGA via ConnectX NIC, hugepages, NIC bound to DPDK
- Isolated CPU cores: `isolcpus=16-19` in kernel cmdline
- TensorRT dev libs in container (`libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev`)
- X11 display for ImGui or Holoviz

## Build

```bash
cd ~/holohub
./holohub build-container hitek9081 --language cpp
./holohub run-container hitek9081 --language cpp \
  --docker-opts="-u root --privileged -w /workspace/holohub/"
./holohub build hitek9081 --language=cpp --parallel $(($(nproc)-1))
```

TensorRT must be in the container image. Add to Dockerfile if missing:
```dockerfile
RUN apt-get update && apt-get install -y \
    libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

## AMC Model

EfficientNet-B4 on 2D spectrograms, 31 modulation classes (PSK, QAM, FSK, AM, FM, OFDM, WCDMA, MC-GSM, Noise).

**Train and export** (run outside container, see [`scripts/README.md`](scripts/README.md) for full details):
```bash
python3 -m venv ~/amc_venv && source ~/amc_venv/bin/activate
pip install torch timm onnx onnxscript numpy scipy matplotlib
cd ~/holohub/applications/hitek9081
python scripts/train_amc.py --epochs 200 --backbone efficientnet_b4 \
    --samples-per-class 3000 --output models/amc_model.onnx
```

**Deploy:** Set path in YAML, delete stale `.engine` if updating model:
```yaml
spectral:
  amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_model.onnx"
```
```bash
rm -f models/amc_model.engine  # force TRT recompile on next launch
```

First launch compiles ONNX → TensorRT engine (30-60s, cached for subsequent runs). This runs before DAQIRI init — no packet loss.

## Configuration

`hitek9081.yaml`:

| Parameter | Description |
|-----------|-------------|
| `hsb.hololink_ip` | FPGA data-plane IP |
| `hsb.payload_size` | Payload bytes per packet |
| `hsb.mtu` | Max RoCE payload (≥ payload_size, max 4096) |
| `hsb.data_port` | UDP dst port (match flow rule) |
| `hitek9081_rx.batch_size` | Packets per batch |
| `hitek9081_rx.iq_format` | 0=pair-interleaved, 1=block-4 |
| `spectral.iq_swap` | 0=as-is, 1=swap, 2=conjugate |
| `spectral.adc_bits` | ADC resolution (e.g., 16) |
| `spectral.amc_model_path` | ONNX path (empty = AMC disabled) |
| `spectral.db_min/db_max` | Display dBFS range |

Computed at startup: `max_packet_size`, `header_size`, `samples_per_antenna`.

## Run

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr <MHz> --center-freq <MHz> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--sr MHz` | Sample rate (required) | — |
| `--center-freq MHz` | Center frequency (required) | — |
| `--fft-size N` | FFT size, power of 2 | 65536 |
| `--num-averages N` | Linear power averaging | 1 |
| `--freq-min/max MHz` | Display frequency range | full band |
| `--hololink IP` | Override FPGA IP | YAML |
| `--imgui` | ImGui display (required for DDC/AMC) | Holoviz |
| `--waterfall` | Scrolling spectrogram (implies `--imgui`) | line plot |

```bash
# ImGui + AMC
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 983.04 --center-freq 750 --fft-size 32768 --num-averages 4 \
  --freq-min 700 --freq-max 800 --imgui

# Waterfall
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 983.04 --center-freq 750 --fft-size 32768 --num-averages 4 \
  --freq-min 700 --freq-max 800 --waterfall
```

## DDC + AMC

Click on spectrum → sets DDC center. Select bandwidth from preset dropdown → DDC activates (panel turns green). Classification runs continuously via EMA-smoothed TensorRT inference.

DDC signal path (all CUDA):
```
Raw int16 IQ → NCO mix (shift to DC) → FIR lowpass → decimate → tile to 4096
  → STFT [128×64] spectrogram → TensorRT EfficientNet-B4 → 31-class softmax
```

Bandwidth presets: 12.5 kHz, 25 kHz, 200 kHz, 1.4–20 MHz (LTE), 40–160 MHz (5G/WiFi).

Panel color: gray (empty), red (zoom only), green (DDC active).
Active DDC bandwidths shown as colored overlays on spectrum/waterfall.

Classification stability: EMA (α=0.3) + 5% hysteresis margin. Per-slot state, reset on clear/retune.

## Architecture

```
spectral_pipeline.h    — DSP: window → FFT → magnitude (no display deps)
holoviz_output.h       — Holoviz: grid overlays + coord normalization
spectral_op.h          — Holoscan wrapper: DSP + DDC + AMC routing
ddc_channel.h          — DDC: NCO → FIR → decimate → normalize
ddc_kernels.cu/cuh     — CUDA: NCO, FIR, decimate, STFT, normalize
amc_inference.h        — TensorRT: ONNX load, engine cache, inference
imgui_display.h/cpp    — ImGui render thread (no CUDA deps)
imgui_sink_op.h        — GPU→CPU bridge + DDC sync
hitek9081_rx_op.h      — Zero-copy DPDK RX
spectral_kernels.cu    — FFT pipeline kernels
```

`spectral_pipeline.h` has no display dependencies — reusable for headless or GnuRadio integration.

## Zero-Copy RX

DGX Spark CPU/GPU share LPDDR5. DAQIRI `host_pinned` buffers are GPU-accessible:
```
DAQIRI pool → pointer store → GPU scatter-gather → accum ring → emit
```

## Scaling

Each RX queue ≈ 1.5M pkt/s on one isolated ARM core. Add queues in YAML for multi-queue (see commented examples). The RX operator iterates all queues — no code changes.

| Queues | Cores | Throughput |
|--------|-------|-----------|
| 1 | 16 | ~47 Gbps |
| 2 | 16-17 | ~94 Gbps |
| 4 | 16-19 | ~100 Gbps |

FPGA port striping: `src_port = 12288 + (pkt_count % num_queues)`, one flow rule per port.

## CUDA Architectures

sm_90 (Hopper), sm_120, sm_121 (Blackwell/Thor). Update `CUDA_ARCHITECTURES` in `cpp/CMakeLists.txt` for others. Mismatch causes silent kernel failures (error 222).
