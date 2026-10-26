# RoCEv2 to Raw UDP Migration Guide

This document details the architectural impact, protocol framing differences, software code modifications, and performance considerations for transitioning the `hitek9081` data plane from **RoCEv2** (RDMA over Converged Ethernet) to **raw UDP**.

---

## Executive Summary

Switching from RoCEv2 to pure raw UDP is **very low effort in the C++ application layer** (~30 to 50 lines of code across 3 files).

- **DAQIRI with `ibverbs` requires 0 architectural changes**: DAQIRI already operates in `stream_type: "raw"` Ethernet mode using Mellanox DevX flow steering on UDP destination/source ports (`udp_src` / `udp_dst`). It does not use InfiniBand QP connection semantics or inspect transport-layer BTH/RETH headers.
- **GPU DSP & UI require 0 changes**: `SpectralOp`, DDC channelizers, TensorRT AMC inference, and ImGui consume unified `[fft_size, 2]` int16 IQ tensors emitted by the RX operator.
- **Primary effort**: The main implementation is in the FPGA RTL/firmware (simplifying packetizer logic) and any corresponding Hololink control plane registers.

---

## Protocol Framing Comparison

### 1. Current Framing: RoCEv2 over UDP

```text
|<--------------------------- Header: 70 Bytes --------------------------->|<----- Payload ----->|<- 4B ->|
+-------------+-------------+------------+-------------+--------------+-------------------+--------+
| Eth Header  | IPv4 Header | UDP Header | IB BTH      | IB RETH      | IQ Samples        | ICRC   |
| 14 bytes    | 20 bytes    | 8 bytes    | 12 bytes    | 16 bytes     | 3968 bytes        | 4 byte |
+-------------+-------------+------------+-------------+--------------+-------------------+--------+
```

- **Ethernet (L2)**: 14 bytes (DST MAC, SRC MAC, EtherType `0x0800`).
- **IPv4 (L3)**: 20 bytes (Protocol `0x11` UDP).
- **UDP (L4)**: 8 bytes (Src Port `0x3000` / 12288, Dst Port `4096..4099`, Length, Checksum).
- **InfiniBand BTH**: 12 bytes (Opcode `0x2A` WRITE ONLY or `0x2B` WRITE ONLY IMM, Partition Key, Destination QP, 24-bit PSN).
- **InfiniBand RETH**: 16 bytes (Virtual Address, Remote Key, DMA Length).
- **Payload**: 3968 bytes (31 pages * 128 bytes = 992 complex IQ samples).
- **ICRC**: 4 bytes (Invariant CRC trailer).
- **Total Packet on Wire**: 70 + 3968 + 4 = **4042 bytes**.

### 2. Proposed Framing: Pure Raw UDP

```text
|<---------------- Header: 42 Bytes ---------------->|<-- (Optional Hdr) -->|<----- Payload ----->|
+--------------------+--------------------+------------+----------------------+-------------------+
| Ethernet Header    | IPv4 Header        | UDP Header | Custom FPGA Header   | IQ Samples        |
| 14 bytes           | 20 bytes           | 8 bytes    | 0 to 8 bytes (opt)   | e.g. 3968 bytes   |
+--------------------+--------------------+------------+----------------------+-------------------+
```

- **Header Size**: Exactly **42 bytes** (or `42 + custom_header_size`).
- **No BTH / RETH**: Saves 28 bytes of transport overhead per packet.
- **No ICRC**: Saves 4 bytes of trailer overhead per packet.
- **Total Overhead Reduction**: 32 bytes saved per packet (~0.8% throughput efficiency gain).

---

## Software Changes by Component

### 1. `hsb_device_manager.cpp` & `hsb_device_manager.h`

The device manager computes derived packet framing parameters and programs the FPGA:

```cpp
// Current (RoCE):
constexpr uint32_t ROCE_HEADER_SIZE = 70;  // Eth(14)+IP(20)+UDP(8)+BTH(12)+RETH(16)
constexpr uint32_t ICRC_SIZE = 4;
constexpr uint32_t ROCE_OVERHEAD = 74;     // ROCE_HEADER_SIZE + ICRC_SIZE

// Proposed (Raw UDP):
constexpr uint32_t UDP_HEADER_SIZE = 42;   // Eth(14)+IP(20)+UDP(8)
constexpr uint32_t ICRC_SIZE = 0;          // No CRC trailer
constexpr uint32_t UDP_OVERHEAD = 42;      // Total packet framing overhead
```

1. Change `header_size = static_cast<uint16_t>(UDP_HEADER_SIZE + custom_header_size)`.
2. Change `max_packet_size = static_cast<uint16_t>(UDP_HEADER_SIZE + packet_payload_size_ + ICRC_SIZE)`.
3. In `start_streaming()`: Replace `hololink_channels_[i]->configure_roce(...)` with the raw UDP control API or register sequence.

### 2. `hitek9081_rx_op.cpp` & `hitek9081_rx_op.h`

1. **Header Offsets**:
   - `UDP_DST_OFFSET` remains unchanged at **byte 36** (`Eth 14 + IP 20 + UDP_src 2`). Antenna demuxing by destination port works identically.
2. **Metadata Packet Filtering**:
   - Currently, `hitek9081_rx_op` checks for `ROCE_WRITE_ONLY_IMM` (`0x2B` at byte 42) to identify and drop 100000:1 EOF metadata packets.
   - If raw UDP does not send metadata packets, remove this check. If metadata is sent on a specific port or with a header flag, adapt the check accordingly.
3. **Packet Sequence Number (PSN) & Gap Detection**:
   - Currently extracts the 24-bit PSN from BTH bytes 8..10 (offset 50).
   - If the FPGA prepends a sequence counter to the UDP payload (e.g. 32-bit counter at byte 42), read:
     `uint32_t psn = *reinterpret_cast<const uint32_t*>(pkt_ptr + 42);`
   - If the FPGA sends pure IQ with no sequence number, the PSN gap detector can simply be disabled.
4. **Scatter-Gather Buffer Pointer**:
   - `h_ant_ptrs_` gathers `pkt_ptr + header_size`. Since `header_size` is passed dynamically via `AppConfig`, no kernel launch code changes are required.

### 3. `hitek9081.yaml`

1. **Buffer Size**:
   - `buf_size: 8192` easily accommodates raw UDP (needs `>= 42 + payload_size`).
2. **Configuration Comments**:
   - Update documentation comments to reflect pure UDP instead of RoCE.

---

## Memory Alignment & GPU Performance Considerations

### Natural 2-Byte Alignment for 16-Bit IQ

- Raw UDP header length: $14 + 20 + 8 = \mathbf{42 \text{ bytes}}$.
- 42 is an even number ($42 \pmod 2 = 0$).
- Because raw ADC samples are signed 16-bit integers (`int16_t`), **the first sample starts on a naturally aligned 2-byte boundary**.
- CUDA kernels reading `int16_t` (`i_val = pkt_data[...]`) execute with zero memory alignment faults.

### Recommendation: 4-Byte / 8-Byte Alignment Optimization

If you want to read IQ samples as packed 32-bit integers (`int32_t` / complex pairs) or vectorized 64-bit / 128-bit loads (`float2`, `int4`), the starting address must be a multiple of 4 or 8:

- $42 \pmod 4 = 2$ (misaligned for 32-bit loads).
- **Tip**: Have the FPGA prepend a **2-byte** or **6-byte** custom header:
  - **Option A (2-byte header)**: 16-bit Packet Sequence Number. Total header = $42 + 2 = \mathbf{44 \text{ bytes}}$ (divisible by 4).
  - **Option B (6-byte header)**: 32-bit PSN (4 B) + 16-bit Channel/Flags (2 B). Total header = $42 + 6 = \mathbf{48 \text{ bytes}}$ (divisible by 16 / 128-bit aligned).

---

## Sequence Number & Gap Detection at Line Rate

At 491.52 MSPS (4 channels), the pipeline receives **~1.982 million packets/sec** (~63 Gbps):

1. **Why sequence numbers matter**: Without a sequence number, packet drops inside the NIC ring, PCIe fabric, or kernel cannot be localized to a specific antenna or timestamp.
2. **Recommended header**:
   - A simple 32-bit monotonically increasing sequence number per channel enables:
     - Immediate detection of network drops (`delta > 1`).
     - Out-of-order detection (`delta < 0`).
     - Accurate loss accounting for mission-critical RF recording.

---

## Architecture Variant: 4 Antennas in 1 Packet on 1 UDP Port

When using an FPGA design that multiplexes all 4 antennas into a single packet arriving on a single UDP destination port (instead of 4 separate packets on 4 distinct UDP ports), the code modifications remain small (~40 to 75 lines of code across 3 files).

### Architecture Comparison

```text
CURRENT ARCHITECTURE (1 antenna per packet, 4 UDP ports):
Packet 0 (UDP 4096) -> [ RoCE Header (70B) | Antenna 0 IQ (3968B) | ICRC (4B) ]
Packet 1 (UDP 4097) -> [ RoCE Header (70B) | Antenna 1 IQ (3968B) | ICRC (4B) ]
Packet 2 (UDP 4098) -> [ RoCE Header (70B) | Antenna 2 IQ (3968B) | ICRC (4B) ]
Packet 3 (UDP 4099) -> [ RoCE Header (70B) | Antenna 3 IQ (3968B) | ICRC (4B) ]
=> Software demuxes antenna by reading UDP destination port at byte 36.

PROPOSED ARCHITECTURE (4 antennas in 1 packet, 1 UDP port):
Packet 0 (UDP 4096) -> [ RoCE Header (70B) | Antenna 0..3 Multiplexed IQ | ICRC (4B) ]
=> No port demuxing needed; software unpacks the 4 antennas from the single payload.
```

### Payload Layouts & Implementation Details

#### Layout Option A: Contiguous Sub-blocks (Slices)

The FPGA packs antenna samples sequentially within the payload:

`[ RoCE Header | Antenna 0 Block | Antenna 1 Block | Antenna 2 Block | Antenna 3 Block | ICRC ]`

- **Code Scope**: ~30 lines in `hitek9081_rx_op.cpp`.
- **CUDA Scatter-Gather Kernel**: **Zero changes required**.
- **Implementation**: In `Hitek9081RxOp::compute()`, each packet immediately populates pointers for all 4 antennas:

```cpp
uint8_t* payload = pkt_ptr + hdr_size;
const size_t ant_bytes = packet_payload_size_ / 4;

for (int a = 0; a < 4; a++) {
  h_ant_ptrs_[cur_batch_idx_][a][batch_ant_pkt_count_[a]++] = payload + (a * ant_bytes);
}
```

The existing `gather_iq_from_scattered` CUDA kernel executes unmodified because each pointer points directly to a contiguous sample block for that antenna.

#### Layout Option B: Sample-Interleaved (TDM / Channel Multiplexed)

The FPGA interleaves time samples across all 4 antennas:

`[ RoCE Header | (A0_s0, A1_s0, A2_s0, A3_s0), (A0_s1, A1_s1, A2_s1, A3_s1), ... | ICRC ]`

- **Code Scope**: ~50–70 lines total across `hitek9081_rx_op.cpp` and `spectral_kernels.cu`.
- **Pointer Storage**: Store a single packet pointer per packet (`h_pkt_ptrs_`) instead of 4 separate pointer arrays.
- **CUDA De-interleaving Kernel**: A single GPU kernel extracts all 4 antennas simultaneously in parallel:

```cuda
__global__ void gather_4ch_interleaved_kernel(
    void* const* pkt_ptrs,
    int16_t* dst_ant0, int16_t* dst_ant1,
    int16_t* dst_ant2, int16_t* dst_ant3,
    int num_pkts, int samples_per_ant_per_pkt, int dst_offset) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pkts * samples_per_ant_per_pkt) return;

  int pkt = idx / samples_per_ant_per_pkt;
  int s   = idx % samples_per_ant_per_pkt;

  // 4 antennas * 2 (I, Q) = 8 int16 values per time step (16 bytes)
  const int16_t* src = static_cast<const int16_t*>(pkt_ptrs[pkt]) + (s * 8);

  int dst_idx = (dst_offset + idx) * 2;
  dst_ant0[dst_idx]     = src[0]; dst_ant0[dst_idx + 1] = src[1];
  dst_ant1[dst_idx]     = src[2]; dst_ant1[dst_idx + 1] = src[3];
  dst_ant2[dst_idx]     = src[4]; dst_ant2[dst_idx + 1] = src[5];
  dst_ant3[dst_idx]     = src[6]; dst_ant3[dst_idx + 1] = src[7];
}
```

*Note: On NVIDIA architectures (including GB10), 16-byte reads (`src[0..7]`) coalesce into efficient 128-bit vector transactions.*

### Additional System Simplifications

1. **FPGA Hardware Setup (`hsb_device_manager.cpp`)**:
   - Only a single call to `hololink_channels_[0]->configure_roce(...)` on `data_port` is needed instead of looping over 4 separate ports.
   - Update `cfg.samples_per_antenna` from `(packet_payload_size_ - custom_header_size) / 4` to `(packet_payload_size_ - custom_header_size) / (4 * cfg.num_antennas)` since each packet now divides its payload across all antennas.
2. **Deterministic Multi-Channel Synchronization (`hitek9081_rx_op.cpp`)**:
   - Because all 4 antennas arrive within the same packet, inter-channel phase synchronization is guaranteed at the packet level.
   - Only a single Packet Sequence Number (PSN) stream needs to be tracked. A single drop tracks packet loss across all 4 channels simultaneously.
3. **Flow Steering Configuration (`hitek9081.yaml`)**:
   - Only one flow rule is required in the DAQIRI configuration matching the single UDP port (`4096`).
4. **Downstream Pipeline Independence**:
   - `SpectralOp`, DDC channelizers, TensorRT AMC, and ImGui consume the identical `TensorMap` format (`"antenna_0"` through `"antenna_3"`). Zero downstream code modifications are required.
