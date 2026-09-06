# Spectrum Analyzer + DDC + AMC

Real-time wideband spectrum analysis with CUDA DDC and TensorRT modulation classification.

```text
Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp / ImGuiSinkOp
(DPDK zero-copy     │ (FFT, avg,      (line plot / waterfall)
 scatter-gather)    │  magnitude)
                    ├──→ DDCChannel[0..15] ──→ TensorRT AMC ──→ ImGui DDC Panel
```

## Requirements

- DGX Spark (GB10, sm_121) or Hopper/Blackwell GPU with unified memory
- HSB FPGA via ConnectX NIC, hugepages, NIC bound to DPDK
- Isolated CPU cores: `isolcpus=16-19` in kernel cmdline
- Display: X11 display (for live ImGui/Holoviz GUI) or built-in virtual framebuffer (`xvfb`) for headless operation

## Container Setup

The application uses a layered container architecture to bundle TensorRT and X11/virtual display dependencies on top of the shared HoloHub networking image:

1. **Base networking container** (`holohub:holoscan-networking`): Built from [`pkg/holoscan-networking/Dockerfile`](../../pkg/holoscan-networking/Dockerfile), providing Mellanox OFED/RDMA drivers, DPDK userspace drivers, and DAQIRI.
2. **Application container** (`holohub-hitek9081:main`): Built from [`applications/hitek9081/Dockerfile`](Dockerfile), layering TensorRT libraries (`libnvinfer-dev`, `libnvinfer-plugin-dev`, `libnvonnxparsers-dev`), X11 utilities (`xauth`, `x11-utils`, `mesa-utils`), and virtual framebuffer (`xvfb`).

```bash
cd ~/holohub

# Step 1: Build base networking container (if not already built)
./holohub build-container holoscan-networking

# Step 2: Build hitek9081 application container
./holohub build-container hitek9081
```

## Quick Start (Build & Run in Container)

All compilation and execution happens inside the container with root privileges (`--privileged` / `-u root`) to access ConnectX NIC hardware, hugepages, and real-time thread memory locking (`set_mempolicy`). HoloHub automatically forwards your host display socket so the GUI window renders directly on your desktop screen:

```bash
cd ~/holohub

# 1. Enter the container environment
./holohub run-container hitek9081 \
  --docker-opts="-u root --privileged -w /workspace/holohub"

# 2. Build the application (inside container)
./holohub build hitek9081
# Or directly using ninja:
ninja -C build/hitek9081

# 3. Run the application (inside container)
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  applications/hitek9081/hitek9081.yaml \
  --sr 491.52 --center-freq 750
```

> [!TIP]
> **Single-Command Host Execution**: For automated testing or scripts, you can build and run in one command from the host:
>
> ```bash
> ./holohub run --as-root --docker-opts="--privileged" hitek9081
> ```
>
> Pass custom parameters via `--run-args`:
>
> ```bash
> ./holohub run --as-root --docker-opts="--privileged" \
>   --run-args="--sr 491.52 --center-freq 750 --num-averages 4" \
>   hitek9081
> ```

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
| ----------- | ------------- |
| `hsb.hololink_ip` | FPGA data-plane IP |
| `hsb.payload_size` | RoCE data packet payload (3968 B = 31 pages × 128 B, 992 IQ samples) |
| `hsb.packets_per_frame` | Data packets per metadata packet (default: 1000, supports up to 100000:1) |
| `hsb.mtu` | Max RoCE payload (≥ payload_size, max 4096) |
| `hsb.data_port` | UDP dst port (match flow rule; ant0=4096, ant1=4097, etc.) |
| `hitek9081_rx.batch_size` | Packets per batch (default: 20480, tuned for 100 GbE line rate) |
| `hitek9081_rx.num_antennas` | Number of ADC channels (1, 2, or 4) |
| `hitek9081_rx.iq_format` | 0=pair-interleaved, 1=block-4 |
| `spectral.iq_swap` | 0=as-is, 1=swap (JESD lanes), 2=conjugate (upper sideband) |
| `spectral.adc_bits` | ADC resolution (e.g., 16) |
| `spectral.amc_model_path` | ONNX path (empty = AMC disabled) |
| `spectral.db_min/db_max` | Display dBFS range |

## Execution Options

Run inside the container shell:

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  applications/hitek9081/hitek9081.yaml \
  --sr <MHz> --center-freq <MHz> [options]
```

### Headless Execution (Virtual Framebuffer)

If running in a headless environment without an X11 server or physical display, use the container's built-in `xvfb-run` utility:

```bash
xvfb-run -a ./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  applications/hitek9081/hitek9081.yaml \
  --sr 491.52 --center-freq 750
```

### CLI Options

| Flag | Description | Default |
| ------ | ------------- | --------- |
| `--sr MHz` | Sample rate in MHz (required) | — |
| `--center-freq MHz` | Center frequency in MHz (required) | — |
| `--fft-size N` | FFT size, power of 2 (512–1048576) | 65536 |
| `--num-averages N` | Linear power averaging (1–32) | 1 |
| `--batch-size N` | RX aggregation burst size | YAML (20480) |
| `--packets-per-frame N` | Data packets per metadata packet | YAML (1000) |
| `--freq-min/max MHz` | Display frequency range | full band |
| `--hololink IP` | Override FPGA IP | YAML (192.168.0.2) |
| `--imgui` | ImGui display | Default |
| `--holoviz` | Use Holoviz renderer instead of ImGui | — |
| `--waterfall` | Scrolling spectrogram (implies `--imgui`) | line plot |

```bash
# 4-Channel Mode @ 491.52 MSPS with 4x FFT averaging
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  applications/hitek9081/hitek9081.yaml \
  --sr 491.52 --center-freq 750 --fft-size 65536 --num-averages 4
```

### Reducing Network & CPU Overhead (`packets_per_frame`)

The Hololink FPGA packetizer generates an End-of-Frame (EOF) metadata packet (`0x2B` RDMA WRITE ONLY WITH IMMEDIATE) when `frame_size` bytes have been transmitted.

- **Previous 1:1 Ratio**: Previously, `frame_size` was set equal to a single packet's payload size, causing the FPGA to emit an EOF metadata packet after **every single data packet** (1:1 ratio). At 4 channels @ 491.52 MSPS, this flooded the wire with ~2.0 million metadata packets/sec (over 4.0 Mfps total), wasting ~4 Gbps of link bandwidth and heavily loading DPDK RX Core 16.
- **Configurable `packets_per_frame` (1,000:1 to 100,000:1)**: Decouples single-packet payload ($3968\text{ B} = 992\text{ samples}$) from FPGA frame size ($\text{frame\_size} = \text{payload} \times \text{packets\_per\_frame}$).
  - **1,000:1 ratio**: Cuts total packet rate nearly in half (from ~4.02 Mfps down to **~1.98 Mfps** at 4 channels @ 491.52 MSPS), eliminates 99.9% of metadata packets (~1,980/s remaining), saves ~4 Gbps on the wire, and relieves DPDK Core 16 of ~2 million descriptor polls/sec.
  - **100,000:1 ratio**: Achievable via `--packets-per-frame 100000`, reducing metadata packets to ~20 packets/s across all 4 channels without affecting RF streaming.

### Performance & Frame Rates

The ImGui window title reports live GPU DSP pipeline throughput:
`Spectrum + DDC | <min> - <max> MHz | Spectrum: XX.X FPS`

## DDC + AMC

16 software DDC channels, any assignable to any antenna.

Layout: spectrum (65% left) + DDC table/detail panel (35% right). DDC table
grouped by antenna — click row to expand zoomed FFT + classification.
Click on zoomed FFT to recenter DDC.

DDC signal path (all CUDA):

```text
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

```text
cpp/
├── main.cpp                     — CLI parsing, signal handling, and execution harness
├── app.h / app.cpp              — Holoscan application orchestration and pipeline composition
├── app_config.h                 — Unified application and CLI configuration
├── hardware/
│   ├── hsb_device_manager.h     — FPGA discovery, RoCE config, JESD link synchronization
│   └── hsb_device_manager.cpp
├── operators/
│   ├── hitek9081_rx_op.h / .cpp — Zero-copy DPDK RX operator
│   ├── spectral_op.h / .cpp     — Holoscan operator: DSP execution + DDC/AMC routing
│   └── imgui_sink_op.h / .cpp   — GPU→CPU bridge + DDC sync
├── dsp/
│   ├── spectral_pipeline.h      — DSP core: window → FFT → magnitude (no display deps)
│   ├── spectral_kernels.cu/.cuh — FFT pipeline & scatter-gather CUDA kernels
│   ├── ddc_manager.h            — Thread-safe multi-channel DDC manager and AMC dispatch
│   ├── ddc_channel.h            — DDC channel: NCO → FIR → decimate → normalize
│   ├── ddc_kernels.cu/.cuh      — CUDA kernels: NCO, FIR, decimate, STFT, normalize
│   ├── amc_inference.h          — TensorRT: engine cache, inference, family grouping
│   └── amc_constants.h          — Shared constants (class/family counts)
└── ui/
    ├── imgui_display.h / .cpp   — ImGui render thread (no CUDA deps)
    └── holoviz_output.h         — Holoviz: grid overlays + coordinate normalization
```

## Zero-Copy RX

DGX Spark CPU/GPU share LPDDR5. DAQIRI `host_pinned` buffers are GPU-accessible:

```text
DAQIRI pool → pointer store → GPU scatter-gather → accum ring → emit
```

## Scaling

Each RX queue ≈ 1.5M pkt/s on one isolated ARM core. Add queues in YAML
for multi-queue (see commented examples). RX operator iterates all queues
automatically.

| Queues | Cores | Throughput |
| -------- | ------- | ----------- |
| 1 | 16 | ~47 Gbps |
| 2 | 16-17 | ~94 Gbps |
| 4 | 16-19 | ~100 Gbps |

FPGA port striping: `src_port = 12288 + (pkt_count % num_queues)`.

## CUDA Architectures

sm_90 (Hopper), sm_120, sm_121 (Blackwell/Thor). Update `CUDA_ARCHITECTURES`
in `cpp/CMakeLists.txt` for others.
