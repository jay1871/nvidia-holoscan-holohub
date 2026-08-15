# HTK9081 Spectrum Analyzer

Receive high-rate IQ sensor data from a Holoscan Sensor Bridge (HSB) FPGA,
compute windowed FFT magnitude on the GPU with optional averaging, and display
the spectrum in real-time via Holoviz.

## Pipeline

```
Hitek9081RxOp ──→ SpectralOp ──→ HolovizOp
(DPDK RX,          (window, FFT,    (line plot per antenna)
 raw IQ out)        averaging,
      │             magnitude)
      │
      └──→ DDCOp (placeholder: narrowband extraction for ML inference)
```

1. **Hololink bringup** — discovers the FPGA, starts the control plane, resets to a known state, configures HTK JESD via HPS
2. **DAQIRI init** — configures DPDK, binds the NIC, programs hardware flow rules
3. **FPGA dataplane programming** — tells the FPGA the host MAC/IP/port so it starts streaming
4. **Holoscan pipeline** — Hitek9081RxOp receives packets and emits raw IQ; SpectralOp computes FFT + averaging on GPU; Holoviz displays

## Requirements

- DGX Spark or system with ConnectX NIC and NVIDIA GPU
- HSB FPGA connected via ConnectX NIC
- Hugepages allocated (`echo 2048 > /proc/sys/vm/nr_hugepages`)
- NIC bound to DPDK (`dpdk-devbind.py -b vfio-pci <PCI_ADDR>`)
- Display available (X11 or direct display for Holoviz)

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

| Parameter | Location | Description |
|-----------|----------|-------------|
| `hsb.hololink_ip` | YAML | HSB FPGA data-plane IP address |
| `hsb.payload_size` | YAML | FPGA payload bytes per packet (from pcap RETH DMA_len) |
| `hsb.mtu` | YAML | Max RoCE packet payload in bytes (must be >= payload_size, max 4096) |
| `hsb.custom_header_size` | YAML | Bytes before IQ data in payload (0 if pure IQ) |
| `hsb.data_port` | YAML | UDP destination port (must match flow rule `udp_dst`) |
| `daqiri.cfg.interfaces[].address` | YAML | NIC PCI BDF (e.g., `0002:01:00.0`) |
| `hitek9081_rx.num_antennas` | YAML | Number of antennas in the packet (currently 1) |
| `spectral.iq_swap` | YAML | IQ ordering: 0=I,Q as-is, 1=swap Q,I, 2=I,Q conjugate (negate Q) |
| `spectral.db_min` / `db_max` | YAML | Display dBFS range (e.g., -130 to -6) |

The following are **computed automatically** from the above and logged at startup:
- `max_packet_size` = 70 + payload_size + 4 (total Ethernet frame size)
- `header_size` = 70 + custom_header_size (bytes before IQ data in the frame)
- `samples_per_antenna` = (payload_size - custom_header_size) / (num_antennas × 4)
- `pkt_pages` = ceil(min(mtu, payload_size) / 128) (DP_PACKET_SIZE register value)

## Run Instructions

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr <sample_rate_MHz> --center-freq <center_freq_MHz> [options]
```

Example:

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 245.76 --center-freq 750.0 --fft-size 65536 \
  --freq-min 700.0 --freq-max 800.0 --num-averages 1
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--sr MHz` | Sample rate in MHz (e.g., 245.76) |
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

**Example (full band):**

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 245.76 --center-freq 750 --fft-size 65536 --num-averages 8
```

**Example (zoomed to a sub-band):**

```bash
./build/hitek9081/applications/hitek9081/cpp/hitek9081 \
  /workspace/holohub/applications/hitek9081/hitek9081.yaml \
  --sr 245.76 --center-freq 750 --fft-size 65536 --num-averages 8 \
  --freq-min 700 --freq-max 800
```

## Display Output

Holoviz opens a window showing the FFT magnitude spectrum:
- X-axis: frequency from `freq_min` to `freq_max` (MHz), defaults to full Nyquist band
- Y-axis: magnitude in dBFS (configurable range via `spectral.db_min`/`spectral.db_max`)
- Each antenna is a separate colored line (green, cyan, red, yellow)
- Frequency tick marks at integer-MHz intervals (auto-scaled: 1, 2, 5, 10, 20, 50… MHz steps, capped at 11 ticks)
- DC offset removed via time-domain mean subtraction (fused into window kernel)
- Window title shows frequency range, bandwidth, and averaging count

Press `Ctrl+C` to stop. DAQIRI statistics print on exit.

## FFT Averaging

Set `--num-averages N` to average N FFT frames in the **linear power domain**
before converting to dB. N must be a power of 2, between 1 and 32. This reduces
noise floor variance by approximately `10*log10(N)` dB.

| num_averages | Noise reduction | Update rate (at 1.08 Mfps, 65536-pt) |
|-------------|-----------------|---------------------------------------|
| 1 | none | ~16 fps |
| 8 | ~9 dB | ~2 fps |
| 32 | ~15 dB | ~0.5 fps |

## Throughput

With payload_size=3920 (980 complex IQ samples per packet, samples_per_antenna=980):

```
RX: 1.008 Mfps | 31.50 Gbps (3938 MB/s payload) | 9574400 frames
```

- **Mfps** — million data frames per second (sensor data packets only)
- **Gbps** — sensor payload data rate (IQ data only, no headers)

## DDC Path (Placeholder)

The `DDCOp` receives the same raw IQ data as `SpectralOp` via Holoscan's
graph fan-out. It is currently a no-op that logs periodic receipt messages.

When implemented, it will:
1. FFT the input block
2. Extract bins around `center_frequency_hz ± bandwidth_hz/2`
3. IFFT the sub-band to get narrowband time-domain IQ
4. Output reduced-rate float32 complex IQ for ML inference

External control is designed via a `DDCConfig` struct that a future gRPC
service thread will update atomically. The compute() path reads this config
with zero locking overhead.

## Changing the Payload Size

When changing `hsb.payload_size`, verify these fields:

1. `hsb.payload_size` — FPGA payload bytes per packet (from pcap RETH DMA_len)
2. `hsb.mtu` — must be >= payload_size to avoid fragmentation (max 4096)
3. `hsb.custom_header_size` — bytes before IQ data in the payload (0 if none)
4. `daqiri buf_size` — must be >= 70 + payload_size + 4 + ~128 headroom

All other RX parameters (`max_packet_size`, `header_size`, `samples_per_antenna`,
`pkt_pages`) are computed automatically and printed at startup.

### MTU and fragmentation

The Holoscan IP `DP_PACKET_SIZE` register controls the max per-packet payload
in 128-byte pages. The hololink library's `configure_common()` reads an `"mtu"`
field from enumeration metadata to compute this value. By default (no `"mtu"` in
metadata), it falls back to `DEFAULT_MTU` (1500), which yields a 1408-byte
per-packet cap — causing fragmentation for any payload > 1408 bytes.

`hitek9081` injects `hsb.mtu` from the YAML into the channel metadata
**before** constructing the `DataChannel`. This ensures `configure_roce()` writes
the correct `DP_PACKET_SIZE` on the first (and only) VP_MASK enable edge. The
FPGA hardware latches this register at that moment and does not re-read it on
subsequent enable cycles — post-hoc register overrides have no effect.

If `mtu >= payload_size`, the entire frame fits in a single RoCE Write Only
packet (opcode `0x2A`). If `mtu < payload_size`, the FPGA fragments into
multiple packets (Write First `0x06` + Write Last `0x08`), which the current
RX operator cannot reassemble — **always set `mtu >= payload_size`**.

The Holoscan IP hardware supports up to 4096 bytes per packet.

## Packet Structure Reference

The FPGA sends RoCE v2 framed UDP packets. DAQIRI delivers the full Ethernet
frame (Ethernet header is NOT stripped):

```
┌─────────────────────────────────────────────────────────────────┐
│ Full frame as delivered by DAQIRI (586 bytes for payload_size=512):│
│   Eth(14) | IP(20) | UDP(8) | BTH(12) | RETH(16) |               │
│   payload(512) | ICRC(4)                                          │
│   ╰─── RoCE headers (70) ───╯╰─ payload_size ─╯                  │
├─────────────────────────────────────────────────────────────────┤
│ BTH opcodes (at byte offset 42 from buffer start):              │
│   0x2A = RDMA Write Only        (single-packet data frame)      │
│   0x2B = RDMA Write Only + Imm  (metadata / frame-complete)     │
│   0x06 = RDMA Write First       (multi-packet start)            │
│   0x07 = RDMA Write Middle      (multi-packet continuation)     │
│   0x08 = RDMA Write Last        (multi-packet final chunk)      │
└─────────────────────────────────────────────────────────────────┘
```

The metadata packet (`0x2B`) is 206 bytes on wire and contains frame number,
timestamp, and CRC. It is detected and skipped by Hitek9081RxOp.
