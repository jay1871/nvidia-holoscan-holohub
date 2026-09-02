# AMC Model Training & Export

EfficientNet-B0 on 2D spectrograms using TorchSig's signal generation.

Input: `[B, 1, 128, 64]` magnitude spectrogram. Output: `[B, 69]` class probabilities.

69 classes from TorchSig grouped into 11 families: OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, PROTOCOL.

Protocol signals: ADS-B, Bluetooth LE, GSM, DVB-S2, DMR, P25, LoRa, WiFi 802.11a, Zigbee.

## Setup

```bash
python3 -m venv ~/amc_venv
source ~/amc_venv/bin/activate
pip install torch timm onnx onnxscript numpy scipy matplotlib
pip install git+https://github.com/TorchDSP/torchsig.git
```

Training runs on any machine with CUDA. ONNX output is platform-independent.

| Platform | ~Epoch time (B0, dynamic) |
|----------|--------------------------|
| DGX Spark (20 ARM cores, GB10) | ~4-5 min |
| x86 desktop (16-core, any GPU) | ~2-3 min |
| x86 server (32-core, any GPU) | ~1-2 min |

## Training Modes

### Dynamic (`--dynamic`) — recommended

Fresh TorchSig signals every batch. No memorization possible. Best generalization.

```bash
python scripts/train_amc.py --epochs 250 --batch-size 256 \
    --backbone efficientnet_b0 --samples-per-class 5000 \
    --dynamic --output models/amc_model.onnx
```

### Static (default)

Pre-computes spectrograms to `.npz`, trains with augmentation. Faster but overfits with B2/B4.

```bash
python scripts/train_amc.py --epochs 250 --batch-size 256 \
    --backbone efficientnet_b0 --samples-per-class 5000 \
    --output models/amc_model.onnx
```

## Resume

`--epochs` is **total** target, not additional:

```bash
python scripts/train_amc.py --resume models/amc_best.pt --epochs 500 \
    --backbone efficientnet_b0 --dynamic --output models/amc_model.onnx
```

Cannot resume across different backbones.

## Export only

```bash
python scripts/train_amc.py --resume models/amc_best.pt --epochs 0 \
    --backbone efficientnet_b0 --output models/amc_model.onnx
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 200 | Total epoch target |
| `--batch-size` | 64 | Batch size (256 for GPU) |
| `--lr` | 1e-3 | AdamW + ReduceLROnPlateau (patience=15) |
| `--input-length` | 4096 | IQ samples before STFT |
| `--backbone` | efficientnet\_b4 | timm model (b0 recommended) |
| `--samples-per-class` | 2000 | Per class per epoch |
| `--dynamic` | false | Fresh signals every batch |
| `--data-root` | datasets/amc | Static mode: cached spectrograms |
| `--regenerate` | false | Static mode: force regeneration |
| `--checkpoint` | models/amc\_best.pt | Best weights save path |
| `--resume` | — | Resume from checkpoint |
| `--output` | models/amc\_model.onnx | ONNX export path |

## Backbone

| Backbone | Params | Overfitting risk |
|----------|--------|-----------------|
| efficientnet\_b0 | ~5M | Low — recommended |
| efficientnet\_b2 | ~9M | Medium — dynamic only |
| efficientnet\_b4 | ~19M | High — dynamic only |

B0 recommended. Family-level accuracy is equivalent across backbones.

## Training details

- **SNR range**: -5 to +30 dB. Uniform distribution.
- **LR scheduler**: ReduceLROnPlateau, halves LR after 15 epochs of no val loss improvement (min 1e-6)
- **Early stopping**: 40 epochs patience on val loss
- **Metrics per epoch**: loss, train %, val %, val loss, macro F1, LR, ETA
- **Final report**: per-class F1 scores
- **Training plot**: Loss | Accuracy | F1 | LR (4 panels)

## Expected performance (B0, 250 epochs, dynamic)

| Metric | Value |
|--------|-------|
| Val accuracy (69 classes) | ~48-49% |
| Macro F1 | ~0.44 |
| Family accuracy (11 families) | ~80-85% |

Strong families: PROTOCOL (lora 0.87, dvbs2 0.81, 80211a 0.75), FSK (16fsk 0.98), PSK (32psk 0.85), QAM (32qam\_cross 0.76).

Weak: OFDM subclass distinction (<0.15 per subclass, ~65% as family), low-order FSK variants (4fsk, 2gmsk).

## Output

```
models/amc_best.pt              # Best checkpoint
models/amc_best_training.png    # Training curves
models/amc_model.onnx           # ONNX for TensorRT
```

## Deploy

1. Copy `models/amc_model.onnx` to DGX Spark
2. YAML: `spectral.amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_model.onnx"`
3. Delete stale engine: `rm models/amc_model.engine`
4. First launch builds TRT engine (30-60s, before DAQIRI)

## Data pipeline match

Training and C++ inference both compute:
```
IQ[4096] → STFT(Hanning, 128 time × 64 freq) → dB magnitude → normalize [0,1]
```

Training: numpy (dynamic) or pre-computed `.npz` (static).
Inference: CUDA `ddc_compute_spectrogram` kernel.

Class list version-dependent — TorchSig v2.x has 69 classes. The C++ side
(`amc_constants.h`) must match. If TorchSig version changes the class list,
update `AMC_NUM_CLASSES` and `CLASS_NAMES` in `amc_inference.h`.
