# AMC Model Training & Export

EfficientNet-B0 on 2D spectrograms using TorchSig's signal generation.

Input: `[B, 1, 128, 64]` magnitude spectrogram (configurable via `--spec-time` and `--spec-freq`). Output: `[B, 69]` class probabilities.

69 classes from TorchSig grouped into 14 families: OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, SPREAD, TDMA, SAT, BURST.

Protocol signals: SPREAD (LoRa, Zigbee), TDMA (GSM, DMR, P25), SAT (DVB-S2), BURST (ADS-B, Bluetooth LE), OFDM (WiFi 802.11a).

## Setup

```bash
python3 -m venv ~/amc_venv
source ~/amc_venv/bin/activate
pip install torch timm onnx onnxscript numpy scipy matplotlib
pip install git+https://github.com/TorchDSP/torchsig.git
```

Training runs on any machine with CUDA. ONNX output is platform-independent.

| Platform | ~Epoch time (B0, dynamic) |
| ---------- | -------------------------- |
| DGX Spark (20 ARM cores, GB10) | ~4-5 min |
| x86 desktop (16-core, any GPU) | ~2-3 min |
| x86 server (32-core, any GPU) | ~1-2 min |

## Training Modes

Run training from the `applications/hitek9081` directory:

```bash
cd ~/holohub/applications/hitek9081
```

### Dynamic (`--dynamic`) — recommended

Fresh TorchSig signals every batch. No memorization possible. Best generalization.

```bash
python scripts/train_amc.py --epochs 250 --batch-size 256 \
    --backbone efficientnet_b0 --samples-per-class 5000 \
    --dynamic --output models/amc_model.onnx
```

### Static (default)

Pre-computes spectrograms to `.npz` files under `--data-root` (`datasets/amc/`), then trains with data augmentation. Faster per epoch, but higher overfitting risk.

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
| ------ | --------- | ------------- |
| `--epochs` | 200 | Total epoch target |
| `--batch-size` | 64 | Batch size (256 recommended for GPU) |
| `--lr` | 1e-3 | AdamW + ReduceLROnPlateau (patience=15) |
| `--input-length` | 4096 | IQ samples before STFT |
| `--spec-time` | 128 | Spectrogram time bins |
| `--spec-freq` | 64 | Spectrogram frequency bins |
| `--backbone` | efficientnet\_b4 | timm model (`efficientnet_b0` strongly recommended) |
| `--samples-per-class` | 2000 | Per class per epoch |
| `--dynamic` | false | Fresh signals every batch |
| `--data-root` | datasets/amc | Static mode: cached spectrograms directory |
| `--regenerate` | false | Static mode: force regeneration of cached `.npz` |
| `--checkpoint` | models/amc\_best.pt | Best weights save path |
| `--resume` | — | Resume from checkpoint |
| `--output` | models/amc\_model.onnx | ONNX export path |

## Backbone

| Backbone | Params | Overfitting risk |
| ---------- | -------- | ----------------- |
| efficientnet\_b0 | ~5M | Low — recommended for live TRT inference |
| efficientnet\_b2 | ~9M | Medium — dynamic only |
| efficientnet\_b4 | ~19M | High — dynamic only |

B0 is strongly recommended. Family-level accuracy is equivalent across backbones while inference latency is significantly lower.

## Training details

- **SNR range**: -5 to +30 dB. Uniform distribution.
- **LR scheduler**: ReduceLROnPlateau, halves LR after 15 epochs of no val loss improvement (min 1e-6)
- **Early stopping**: 40 epochs patience on val loss
- **Metrics per epoch**: loss, train %, val %, val loss, macro F1, LR, ETA
- **Final report**: per-class F1 scores
- **Training plot**: Loss | Accuracy | F1 | LR (4 panels)

## Expected performance (B0, 250 epochs, dynamic)

| Metric | Value |
| -------- | ------- |
| Val accuracy (69 classes) | ~48-49% |
| Macro F1 | ~0.44 |
| Family accuracy (14 families) | ~80-85% |

Strong families: SPREAD (lora 0.87), SAT (dvbs2 0.81), FSK (16fsk 0.98), PSK (32psk 0.85), QAM (32qam\_cross 0.76).

Weak: OFDM subclass distinction (<0.15 per subclass, ~65% as family), low-order FSK variants (4fsk, 2gmsk).

## Output

```text
models/amc_best.pt              # Best checkpoint weights
models/amc_best_training.png    # Training curves (Loss, Acc, F1, LR)
models/amc_model.onnx           # ONNX model ready for TensorRT
datasets/amc/*.npz              # Cached datasets (static mode only)
```

## Deploy

1. Copy `models/amc_model.onnx` to DGX Spark:

   ```bash
   scp models/amc_model.onnx nvidia@<dgx-ip>:~/holohub/applications/hitek9081/models/
   ```

2. Set path in YAML (`hitek9081.yaml`):

   ```yaml
   spectral:
     amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_model.onnx"
   ```

3. Delete stale engine if updating an existing model:

   ```bash
   rm -f models/amc_model.engine
   ```

4. First launch compiles the TRT engine (30–60s on GB10, cached as `.engine` for subsequent runs).

## Data pipeline match

Training and C++ inference both compute:

```text
IQ[4096] → STFT(Hanning, 128 time × 64 freq) → dB magnitude → normalize [0,1]
```

Training: numpy (dynamic) or pre-computed `.npz` (static).
Inference: CUDA `ddc_compute_spectrogram` kernel.

- **Spectrogram Dimensions**: The C++ inference engine uses `SPEC_TIME = 128` and `SPEC_FREQ = 64` by default (defined in [`cpp/dsp/amc_inference.h`](../cpp/dsp/amc_inference.h)). If training with custom `--spec-time` or `--spec-freq`, update `SPEC_TIME` and `SPEC_FREQ` in `amc_inference.h` and the STFT kernel in [`cpp/dsp/ddc_kernels.cu`](../cpp/dsp/ddc_kernels.cu) to match the exported ONNX model input dimensions.
- **Class list**: Version-dependent — TorchSig v2.x has 69 classes. The C++ side ([`cpp/dsp/amc_constants.h`](../cpp/dsp/amc_constants.h)) must match. If the TorchSig version changes the class list, update `AMC_NUM_CLASSES` and `CLASS_NAMES` in [`cpp/dsp/amc_constants.h`](../cpp/dsp/amc_constants.h).
