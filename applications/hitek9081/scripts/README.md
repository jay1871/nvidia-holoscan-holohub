# AMC Model Training & Export

EfficientNet-B4 on 2D spectrograms. Input: `[B, 1, 128, 64]`. Output: `[B, 31]`.

31 classes: OOK, 4PAM, BPSK, QPSK, 8PSK, 16PSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, 32PSK, 2FSK, 4FSK, 8FSK, 16FSK, AM-SSB, AM-DSB, FM, GMSK, MSK, OQPSK, CPFSK, 64PSK, OFDM-QPSK, OFDM-16QAM, OFDM-64QAM, OFDM-256QAM, WCDMA, MC-GSM, Noise.

## Setup

```bash
python3 -m venv ~/amc_venv
source ~/amc_venv/bin/activate
pip install torch timm onnx onnxscript numpy scipy matplotlib
```

## Train

```bash
cd ~/holohub/applications/hitek9081

# B4, 200 epochs, ~11 hours on GB10
python scripts/train_amc.py --epochs 200 --backbone efficientnet_b4 \
    --samples-per-class 3000 --output models/amc_model.onnx

# Quick test, ~1 hour
python scripts/train_amc.py --epochs 20 --samples-per-class 500 \
    --output models/amc_model.onnx
```

### Resume

`--epochs` is **total** target, not additional. Checkpoint at epoch 100 → pass `--epochs` > 100:

```bash
python scripts/train_amc.py --resume models/amc_best.pt --epochs 500 \
    --backbone efficientnet_b4 --output models/amc_model.onnx
```

Re-export without training (`--epochs` ≤ checkpoint epoch):
```bash
python scripts/train_amc.py --resume models/amc_best.pt --epochs 0 \
    --backbone efficientnet_b4 --output models/amc_model.onnx
```

Cannot resume across different backbones.

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 200 | Total epoch target |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | AdamW + CosineAnnealing |
| `--input-length` | 4096 | IQ samples before STFT |
| `--backbone` | efficientnet_b4 | timm model name |
| `--samples-per-class` | 5000 | Per class per epoch |
| `--snr-min` | -20 | Min SNR (dB) |
| `--snr-max` | 30 | Max SNR (dB) |
| `--checkpoint` | models/amc_best.pt | Best weights save path |
| `--resume` | — | Resume from checkpoint |
| `--output` | models/amc_model.onnx | ONNX export path |

### Backbone options

| Backbone | Params | ~Time (200ep, 3K/class) |
|----------|--------|------------------------|
| efficientnet_b0 | ~5M | ~3h |
| efficientnet_b2 | ~9M | ~6h |
| efficientnet_b4 | ~19M | ~11h |

### Output

```
models/amc_best.pt              # Checkpoint (weights + optimizer + scheduler + history)
models/amc_best_training.png    # Loss + accuracy curves
models/amc_model.onnx           # ONNX for TensorRT
```

## Export only (no training)

```bash
python scripts/train_amc.py --resume models/amc_best.pt --epochs 0 \
    --backbone efficientnet_b4 --output models/amc_model.onnx
```

## Deploy

1. `models/amc_model.onnx` on host filesystem (volume-mounted into container)
2. YAML: `spectral.amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_model.onnx"`
3. Delete `.engine` when updating ONNX: `rm models/amc_model.engine`
4. First launch compiles TRT engine (30-60s, before DAQIRI — no packet loss)

## Data pipeline match

Training and inference both compute:
```
IQ[4096] → STFT(Hanning, 128 time × 64 freq) → dB magnitude → normalize [0,1]
```
Training: numpy. Inference: CUDA `ddc_compute_spectrogram` kernel.

## Training data

Synthetic on-the-fly generation per sample:
- Random modulation from 31 classes
- Random SPS: {2, 4, 8, 12, 16, 24, 32} (bandwidth robustness)
- Channel: Rayleigh fading (2-5 taps), CFO ±5%, phase noise (Wiener), random phase/time offset, AWGN (-20 to +30 dB)

Signal generators:
- **OFDM**: CP, pilots, DC/guard nulls, QPSK/16QAM/64QAM/256QAM subcarriers
- **WCDMA**: 3-5 OVSF-spread code channels, complex scrambling, Gaussian pulse shaping
- **MC-GSM**: 2-8 independent GMSK carriers (BT=0.3), random power/frequency offset
- **Noise**: Unit-power complex Gaussian (TDD gaps, empty spectrum)
