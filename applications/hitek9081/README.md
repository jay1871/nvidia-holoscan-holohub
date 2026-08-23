# HTK9081 Spectrum Analyzer

Receive high-rate IQ sensor data from a Holoscan Sensor Bridge (HSB) FPGA,
compute windowed FFT magnitude on the GPU with optional averaging, and display
the spectrum in real-time via Holoviz.

## Pipeline

```
Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp
(DPDK RX,          (window, FFT,    (line plot per antenna)
 zero-copy          averaging,
 scatter-gather)    magnitude)
      │
      └──→ DDCOp (placeholder: narrowband extraction for ML inference)
```

1. **Hololink bringup** — discovers the FPGA, starts the control plane, resets to a known state, configures HTK JESD via HPS
2. **DAQIRI init** — configures DPDK, binds the NIC, programs hardware flow rules
3. **FPGA dataplane programming** — tells the FPGA the host MAC/IP/port so it starts streaming
4. **Holoscan pipeline** — Hitek9081RxOp receives packets via zero-copy scatter-gather; SpectralOp computes FFT + averaging on GPU; Holoviz displays

## Requirements

- DGX Spark or system with ConnectX NIC and NVIDIA GPU (unified memory required for zero-copy)
- HSB FPGA connected via ConnectX NIC
- Hugepages allocated (`echo 2048 > /proc/sys/vm/nr_hugepages`)
- NIC bound to DPDK (`dpdk-devbind.py -b vfio-pci <PCI_ADDR>`)
- Isolated CPU cores for DPDK RX (see [Scaling to Higher Data Rates](#scaling-to-higher-data-rates))
- Display available (X11 or direct display for Holoviz)

**CUDA architectures:** The CMakeLists.txt compiles CUDA kernels for sm_90
(Hopper), sm_120, and sm_121 (Blackwell/Thor). If running on a different GPU,
update `CUDA_ARCHITECTURES` in `cpp/CMakeLists.txt`. Kernels will silently fail
with `cudaErrorUnsupportedPtxVersion` (error 222) if the target architecture is
missing.

## Build Instructions

### Step 1: Build and enter the networking container

```bash
cd ~/holohub
./holohub run-container hitek9081 \
  --language cpp \
  --docker-opts="-u root --privileged -w /workspace/holohub/"
```

### Step 2: Build the application inside the container - in holohub directory

```bash
./holohub build hitek9081 --language=cpp --parallel $(($(nproc)-1))
```

## Configuration

Edit `hitek9081.yaml` before running. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `hsb.hololink_ip` | HSB FPGA data-plane IP address |
| `hsb.payload_size` | FPGA payload bytes per packet (from pcap RETH DMA_len) |
| `hsb.mtu` | Max RoCE packet payload in bytes (must be >= payload_size, max 4096) |
| `hsb.custom_header_size` | Bytes before IQ data in payload (0 if pure IQ) |
| `hsb.data_port` | UDP destination port (must match flow rule `udp_dst`) |
| `daqiri.cfg.interfaces[].address` | NIC PCI BDF (e.g., `0002:01:00.0`) |
| `hitek9081_rx.batch_size` | Packets per processing batch (tune for latency vs overhead) |
| `hitek9081_rx.iq_format` | IQ packing: 0=pair-interleaved, 1=block-4 |
| `spectral.iq_swap` | IQ ordering: 0=I,Q as-is, 1=swap Q,I, 2=conjugate (negate Q) |
| `spectral.db_min` / `db_max` | Display dBFS range |

The following are **computed automatically** from the above and logged at startup:
- `max_packet_size` = 70 + payload_size + 4 (total Ethernet frame size)
- `header_size` = 70 + custom_header_size (bytes before IQ data in the frame)
- `samples_per_antenna` = (payload_size - custom_header_size) / (num_antennas × 4)

## Run Instructions

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr <sample_rate_MHz> --center-freq <center_freq_MHz> [options]
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--sr MHz` | Sample rate in MHz (e.g., 245.76 or 983.04) |
| `--center-freq MHz` | Center frequency in MHz (e.g., 750) |

**Optional arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--fft-size N` | FFT size (power of 2, 512–1048576) | 65536 |
| `--num-averages N` | FFTs to average in linear power (power of 2, 1–32) | 1 |
| `--freq-min MHz` | Display minimum frequency in MHz | center - sr/2 |
| `--freq-max MHz` | Display maximum frequency in MHz | center + sr/2 |
| `--hololink IP` | Override FPGA IP from YAML | from YAML |
| `-h`, `--help` | Show usage | — |

**Example (full band at 983.04 MSPS):**

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 983.04 --center-freq 750 --fft-size 32768 --num-averages 4
```

**Example (zoomed sub-band at 245.76 MSPS):**

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 245.76 --center-freq 750 --fft-size 65536 --num-averages 8 \
  --freq-min 700 --freq-max 800
```

## Zero-Copy Architecture

On DGX Spark, the CPU and GPU share the same LPDDR5 memory. DAQIRI's
`host_pinned` (`cudaMallocHost`) buffers are directly accessible from GPU
kernels without any copy operations:

```
DAQIRI buffer pool (host_pinned / cudaMallocHost)
        │
        │  pointer store (no memcpy)
        ▼
h_dev_ptrs[batch][pkt] = payload pointer
        │
        │  GPU scatter-gather kernel
        ▼
accum_bufs[ring] — contiguous [I,Q,I,Q,...] int16 on GPU
        │
        │  emit (D2D copy to output snapshot)
        ▼
SpectralOp → Holoviz
```

This eliminates both the per-packet CPU `memcpy` and the `cudaMemcpyAsync` H2D
transfer that would be required on discrete GPU systems. Packet buffers are held
until a CUDA event confirms the GPU kernel has finished reading from them.

## Scaling to Higher Data Rates

Each DAQIRI RX queue is serviced by a single CPU core. On DGX Spark's ARM cores,
one core can sustain approximately **1.5M packets/sec** of packet inspection and
pointer storage. At payload_size=3904 this corresponds to ~47 Gbps. The FPGA at
983.04 MSPS produces ~1.0M pkt/s (~31 Gbps), within a single core's capacity.

To reach higher rates (up to 100 Gbps), add more RX queues pinned to separate
isolated CPU cores:

### 1. Isolate CPU cores (kernel command line)

```
isolcpus=16-19
```

This prevents the kernel scheduler from placing other work on cores 16–19,
ensuring they are dedicated to DPDK packet reception.

### 2. Add RX queues in `hitek9081.yaml`

Each queue needs its own isolated core. Uncomment the additional queues in the
DAQIRI config:

```yaml
queues:
- name: "rx_q_0"
  id: 0
  cpu_core: 16        # isolated core
  batch_size: 20480
  memory_regions: ["HSB_RX_CPU"]
- name: "rx_q_1"
  id: 1
  cpu_core: 17        # isolated core
  batch_size: 20480
  memory_regions: ["HSB_RX_CPU"]
- name: "rx_q_2"
  id: 2
  cpu_core: 18        # isolated core
  batch_size: 20480
  memory_regions: ["HSB_RX_CPU"]
- name: "rx_q_3"
  id: 3
  cpu_core: 19        # isolated core
  batch_size: 20480
  memory_regions: ["HSB_RX_CPU"]
```

### 3. FPGA port striping

The FPGA round-robins packets across N source ports (where N = number of RX
queues). Each port maps to a dedicated queue via a flow rule. This distributes
the NIC-to-CPU load across multiple ARM cores — all queues feed into the same
`compute()` call, which iterates them and funnels packets into a single GPU
scatter-gather pipeline:

```
FPGA (100 sources, 983.04 MSPS)
  │
  │  packet_counter % 4 → selects source port
  ▼
udp_src 12288 ──→ flow rule ──→ rx_q_0 (core 16) ─┐
udp_src 12289 ──→ flow rule ──→ rx_q_1 (core 17) ─┤
udp_src 12290 ──→ flow rule ──→ rx_q_2 (core 18) ─┤── compute() drains all queues
udp_src 12291 ──→ flow rule ──→ rx_q_3 (core 19) ─┘        │
                                                            ▼
                                                    GPU scatter-gather
                                                    (single accum stream)
```

The queues exist solely to parallelize the CPU-side packet reception across
multiple ARM cores. The GPU kernel and accumulation logic are unchanged — they
see one merged stream of packet pointers regardless of how many queues delivered
them.

YAML flow rules for 4-port striping:

```yaml
flows:
- name: "q0_flow"
  id: 0
  action: { type: queue, id: 0 }
  match: { udp_src: 12288, udp_dst: 4096 }
- name: "q1_flow"
  id: 1
  action: { type: queue, id: 1 }
  match: { udp_src: 12289, udp_dst: 4096 }
- name: "q2_flow"
  id: 2
  action: { type: queue, id: 2 }
  match: { udp_src: 12290, udp_dst: 4096 }
- name: "q3_flow"
  id: 3
  action: { type: queue, id: 3 }
  match: { udp_src: 12291, udp_dst: 4096 }
```

FPGA firmware change: set `source_port = DATA_SOURCE_UDP_PORT + (pkt_count % num_queues)`.

No application code changes are needed — the RX operator already iterates all
queues in `compute()`.

### Throughput scaling

| Queues | Isolated cores | Approx. max throughput |
|--------|---------------|----------------------|
| 1 | 16 | ~47 Gbps (1.5M pkt/s) |
| 2 | 16, 17 | ~94 Gbps (3.0M pkt/s) |
| 4 | 16–19 | ~100 Gbps (limited by NIC line rate) |

## IQ Format

The JESD data converter can output samples in two packing formats. The
`hitek9081_rx.iq_format` YAML parameter selects the decode mode in the GPU
scatter-gather kernel:

| iq_format | Source layout (per group of 4 samples) | Description |
|-----------|----------------------------------------|-------------|
| 0 | `[I0,Q0,I1,Q1,I2,Q2,I3,Q3]` | Pair-interleaved (each sample is adjacent I,Q) |
| 1 | `[I0,I1,I2,I3,Q0,Q1,Q2,Q3]` | Block-interleaved (4 I values followed by 4 Q values) |

Both modes output contiguous pair-interleaved `[I,Q,I,Q,...]` int16 into the
GPU accumulation buffer. SpectralOp consumes this format directly.

The `spectral.iq_swap` parameter is applied in the FFT pipeline (window kernel)
and handles phase/sign conventions independent of the packing format.

## FFT Averaging

Set `--num-averages N` to average N FFT frames in the **linear power domain**
before converting to dB. N must be a power of 2, between 1 and 32. This reduces
noise floor variance by approximately `10*log10(N)` dB.

| num_averages | Noise reduction | Update rate (at 1.0 Mfps, 32768-pt) |
|-------------|-----------------|--------------------------------------|
| 1 | none | ~30 fps |
| 4 | ~6 dB | ~7 fps |
| 8 | ~9 dB | ~4 fps |
| 32 | ~15 dB | ~1 fps |

## Display Output

Holoviz opens a window showing the FFT magnitude spectrum:
- X-axis: frequency from `freq_min` to `freq_max` (MHz), defaults to full Nyquist band
- Y-axis: magnitude in dBFS (configurable range via `spectral.db_min`/`spectral.db_max`)
- Each antenna is a separate colored line (green, cyan, red, yellow)
- Frequency tick marks at integer-MHz intervals (auto-scaled)
- Hanning window applied before FFT (no DC removal — preserves signals at center frequency)

Press `Ctrl+C` to stop. DAQIRI statistics print on exit.

## Changing the Payload Size

When changing `hsb.payload_size`, verify:

1. `hsb.payload_size` — must match FPGA configuration
2. `hsb.mtu` — must be >= payload_size (max 4096)
3. `daqiri buf_size` — must be >= 70 + payload_size + 4 + ~128 headroom

All other RX parameters are computed automatically and printed at startup.

## Packet Structure Reference

The FPGA sends RoCE v2 framed UDP packets. DAQIRI delivers the full Ethernet
frame (header NOT stripped):

```
┌─────────────────────────────────────────────────────────────────┐
│ Eth(14) | IP(20) | UDP(8) | BTH(12) | RETH(16) | payload | ICRC(4)
│ ╰──────── RoCE headers (70 bytes) ─────────╯                     │
├─────────────────────────────────────────────────────────────────┤
│ BTH opcodes (byte offset 42):                                   │
│   0x2A = RDMA Write Only        (data packet)                   │
│   0x2B = RDMA Write Only + Imm  (metadata — skipped)            │
└─────────────────────────────────────────────────────────────────┘
```

## DDC Path (Placeholder)

The `DDCOp` receives the same raw IQ data as `SpectralOp` via Holoscan's
graph fan-out. Currently a no-op. When implemented, it will extract a
narrowband signal for downstream ML inference.
