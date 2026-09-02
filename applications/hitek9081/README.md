# HTK9081 Spectrum Analyzer + AMC

Real-time wideband spectrum analysis with CUDA DDC and TensorRT modulation classification.

```
Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp / ImGuiSinkOp
(DPDK zero-copy     │ (FFT, avg,      (line plot / waterfall)
 scatter-gather)    │  magnitude)
                    ├──→ DDCChannel[0..15] ──→ TensorRT AMC ──→ ImGui DDC Panel
```

## Requirements

- DGX Spark (GB10, sm_121) or Hopper/Blackwell GPU with unified memory
- HSB FPGA via ConnectX NIC, hugepages, NIC bound to DPDK
- Isolated CPU cores: `isolcpus=16-19` in kernel cmdline
- TensorRT dev libs in container (`libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev`)
- X11 display for ImGui or Holoviz

## Build

TensorRT dev libs are included in the networking container Dockerfile
(`pkg/holoscan-networking/Dockerfile`).

```bash
cd ~/holohub

# Step 1: Build container (includes TensorRT, DAQIRI, DPDK)
./holohub build-container hitek9081 --language cpp

# Step 2: Enter container
./holohub run-container hitek9081 --language cpp \
  --docker-opts="-u root --privileged -w /workspace/holohub/"

# Step 3: Build application (inside container)
./holohub build hitek9081 --language=cpp --parallel $(($(nproc)-1))
```

If TensorRT is missing in an existing container (CMake error `NVINFER_LIB NOTFOUND`):
```bash
apt-get update && apt-get install -y libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

## AMC Model

EfficientNet-B0 on 2D spectrograms. 69 signal classes from TorchSig grouped
into 11 families (OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, PROTOCOL).

Protocol signals: ADS-B, Bluetooth LE, GSM, DVB-S2, DMR, P25, LoRa, WiFi 802.11a, Zigbee.

Training runs on any machine with CUDA — ONNX output is platform-independent.
x86 is ~2× faster than ARM for training due to numpy/scipy throughput.

**Train** (see [`scripts/README.md`](scripts/README.md)):
```bash
python3 -m venv ~/amc_venv && source ~/amc_venv/bin/activate
pip install torch timm onnx onnxscript numpy scipy matplotlib
pip install git+https://github.com/TorchDSP/torchsig.git
cd ~/holohub/applications/hitek9081
python scripts/train_amc.py --epochs 250 --batch-size 256 \
    --backbone efficientnet_b0 --samples-per-class 5000 \
    --dynamic --output models/amc_model.onnx
```

**Deploy:**
```bash
# Copy model to DGX Spark
scp models/amc_model.onnx nvidia@<dgx-ip>:~/holohub/applications/hitek9081/models/
# Delete stale engine
rm models/amc_model.engine
```

Set path in YAML:
```yaml
spectral:
  amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_model.onnx"
```

First launch compiles TRT engine (30-60s, before DAQIRI — no packet loss).

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
| `spectral.iq_swap` | 0=as-is, 1=swap (JESD lanes), 2=conjugate (upper sideband) |
| `spectral.adc_bits` | ADC resolution (e.g., 16) |
| `spectral.amc_model_path` | ONNX path (empty = AMC disabled) |
| `spectral.db_min/db_max` | Display dBFS range |

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
```

## DDC + AMC

16 software DDC channels, any assignable to any antenna.

Layout: spectrum (65% left) + DDC table/detail panel (35% right). DDC table
grouped by antenna — click row to expand zoomed FFT + classification.
Click on zoomed FFT to recenter DDC.

DDC signal path (all CUDA):
```
Raw int16 IQ → NCO mix (shift to DC) → FIR lowpass (shared memory, 63-tap Blackman)
  → decimate → tile to 4096 → STFT [128×64] spectrogram
  → TensorRT EfficientNet-B0 → 69-class softmax → family grouping
```

Display shows **signal family** as primary classification (sum of subclass probs).
11 families: OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, PROTOCOL.

Bandwidth presets: 1.4, 3, 5, 10, 15, 20, 40, 50, 60, 80, 100, 160 MHz.

Active DDC bandwidths shown as colored overlays (16 distinct colors).

Classification: EMA (α=0.3) + 5% hysteresis. Per-slot state, reset on clear/retune.

## Architecture

```
spectral_pipeline.h    — DSP: window → FFT → magnitude (no display deps)
holoviz_output.h       — Holoviz: grid overlays + coord normalization
spectral_op.h          — Holoscan wrapper: DSP + DDC + AMC routing
ddc_channel.h          — DDC: NCO → FIR → decimate → normalize
ddc_kernels.cu/cuh     — CUDA: NCO, FIR (shared mem), decimate, STFT, normalize
amc_inference.h        — TensorRT: engine cache, inference, family grouping
amc_constants.h        — Shared constants (class/family counts)
imgui_display.h/cpp    — ImGui render thread (no CUDA deps)
imgui_sink_op.h        — GPU→CPU bridge + DDC sync
hitek9081_rx_op.h      — Zero-copy DPDK RX
spectral_kernels.cu    — FFT pipeline kernels (downsample uses float stride)
```

## Zero-Copy RX

DGX Spark CPU/GPU share LPDDR5. DAQIRI `host_pinned` buffers are GPU-accessible:
```
DAQIRI pool → pointer store → GPU scatter-gather → accum ring → emit
```

## Scaling

Each RX queue ≈ 1.5M pkt/s on one isolated ARM core. Add queues in YAML
for multi-queue (see commented examples). RX operator iterates all queues
automatically.

| Queues | Cores | Throughput |
|--------|-------|-----------|
| 1 | 16 | ~47 Gbps |
| 2 | 16-17 | ~94 Gbps |
| 4 | 16-19 | ~100 Gbps |

FPGA port striping: `src_port = 12288 + (pkt_count % num_queues)`.

## CUDA Architectures

sm_90 (Hopper), sm_120, sm_121 (Blackwell/Thor). Update `CUDA_ARCHITECTURES`
in `cpp/CMakeLists.txt` for others.
