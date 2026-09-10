# Raw IQ AMC Model Training & Export

Trains 1D deep neural networks (**ConvNeXt-1D** and **ResNet-1D**) directly on raw complex IQ waveform buffers `[B, 2, 4096]` (I and Q channels) using dynamic TorchSig signal synthesis.

By operating directly on raw IQ samples, the models preserve complete phase trajectories, constellation geometries, and symbol transition dynamics in the complex domain.

## Key Architecture Advantages

- **Phase & Constellation Preservation**: High-order modulations (16QAM, 64QAM, 256QAM) are resolved by allowing convolutions to directly capture constellation grids and transitions in the complex plane.
- **Large 1D Receptive Field**: ConvNeXt-1D uses depthwise 1D convolutions with kernel size k=15, spanning multiple symbol periods and subcarrier cycles simultaneously in early layers.
- **Zero Memorization Risk**: Signals are generated on the fly every batch with randomized SNR (-5 dB to +30 dB), multipath fading, and carrier frequency offsets.
- **Native DDC Output Match**: The HTK9081 C++ DDC channels output normalized complex IQ samples directly on GPU memory, feeding straight into TensorRT without intermediate transformation.

## Backbones

| Backbone     | Architecture                                | Params | TRT FP16 Latency (GB10) | Expected Top-1 (Class) | Expected Top-1 (Family) | Best For                                     |
| ------------ | ------------------------------------------- | ------ | ----------------------- | ---------------------- | ----------------------- | -------------------------------------------- |
| `convnext1d` | Depthwise k=15, Inverted Bottleneck, GELU   | ~7.1M  | ~0.28 ms                | ~64-70%                | ~90-95%                 | Recommended -- highest accuracy on OFDM, QAM |
| `resnet1d`   | ResNet-18 1D, Residual Shortcuts, BatchNorm | ~3.9M  | ~0.18 ms                | ~59-64%                | ~87-91%                 | Lightweight -- ultra-low latency             |

## Setup

```bash
python3 -m venv ~/amc_venv
source ~/amc_venv/bin/activate
pip install torch onnx onnxscript numpy scipy matplotlib
pip install git+https://github.com/TorchDSP/torchsig.git
```

## Training

Run training from the `applications/hitek9081` directory:

```bash
cd ~/holohub/applications/hitek9081
```

### 1. ConvNeXt-1D (Recommended -- Highest Accuracy)

```bash
python scripts/train_amc_raw-iq.py     --epochs 250     --batch-size 128     --backbone convnext1d     --samples-per-class 2000     --snr-min -5.0     --snr-max 30.0     --checkpoint models/amc_raw_iq_convnext_best.pt     --output models/amc_raw_iq_convnext.onnx
```

### 2. ResNet-1D (Fastest / Lowest Latency)

```bash
python scripts/train_amc_raw-iq.py     --epochs 250     --batch-size 128     --backbone resnet1d     --samples-per-class 2000     --snr-min -5.0     --snr-max 30.0     --checkpoint models/amc_raw_iq_resnet_best.pt     --output models/amc_raw_iq_resnet.onnx
```

### 3. Resuming & Fine-Tuning

- **Extending Epochs**: Continue training towards a new total `--epochs` target using the decayed checkpoint learning rate:

  ```bash
  python scripts/train_amc_raw-iq.py \
      --resume models/amc_raw_iq_convnext_best.pt \
      --epochs 500 \
      --backbone convnext1d \
      --output models/amc_raw_iq_convnext.onnx
  ```

- **Fine-Tuning (`--override-lr`)**: Resets early stopping counters and overrides the learning rate with `--lr` to explore a new distribution or SNR range:

  ```bash
  python scripts/train_amc_raw-iq.py \
      --resume models/amc_raw_iq_convnext_best.pt \
      --override-lr --lr 1e-5 \
      --epochs 550 \
      --output models/amc_raw_iq_convnext_finetuned.onnx
  ```

  > [!TIP]
  > When resuming, keep `--samples-per-class` and `--batch-size` matched to the previous run to avoid distribution shift and optimizer gradient mismatch.

### 4. Standalone ONNX Export (Without Training)

Export any existing PyTorch checkpoint directly to an ONNX model:

```bash
python scripts/train_amc_raw-iq.py \
    --export models/amc_raw_iq_convnext_best.pt \
    --output models/amc_raw_iq_convnext_best.onnx
```

## CLI Arguments

| Flag                  | Default                     | Description                                                             |
| --------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `--epochs`            | `250`                       | Total epoch target                                                      |
| `--batch-size`        | `128`                       | Batch size (128-256 recommended for GPU)                                |
| `--lr`                | `1e-3`                      | Initial learning rate (AdamW + ReduceLROnPlateau)                       |
| `--workers`           | `max cores - 2`             | DataLoader worker processes (e.g., 18 on 20-core host)                  |
| `--input-length`      | `4096`                      | Number of complex IQ samples per signal (matches DDC output buffer)     |
| `--backbone`          | `convnext1d`                | Model architecture: `convnext1d` or `resnet1d`                          |
| `--samples-per-class` | `2000`                      | Dynamic training samples synthesized per class per epoch                |
| `--snr-min`           | `-5.0`                      | Minimum SNR in dB                                                       |
| `--snr-max`           | `30.0`                      | Maximum SNR in dB                                                       |
| `--override-lr`       | `False`                     | When resuming, override optimizer LR with `--lr` and reset early stop   |
| `--checkpoint`        | `models/amc_raw_iq_best.pt` | Path to save best model checkpoint                                      |
| `--resume`            | `None`                      | Checkpoint to resume training from                                      |
| `--export`            | `None`                      | Export an existing checkpoint directly to ONNX without training         |
| `--output`            | `models/amc_raw_iq.onnx`    | ONNX export path                                                        |

## Evaluation & Signal Families

The training script automatically logs both fine-grained modulation accuracy and 14-family grouping accuracy every epoch:

- **14 Families**: OOK, ASK, FSK, PSK, QAM, OFDM, FM, AM, CHIRP, TONE, SPREAD, TDMA, SAT, BURST.
- **Family Accuracy Calculation**: Evaluates the sum of subclass probabilities belonging to each family, aligning with the real-time C++ inference engine in [`cpp/dsp/amc_inference.h`](../cpp/dsp/amc_inference.h).
- **Empirical Benchmarks (Full 69 Classes, -5 dB to +30 dB SNR)**:
  - **Top-1 Class Accuracy**: **55.9%**
  - **Top-1 Family Accuracy**: **77.3%**
  - **Key Protocols**: DVB-S2 **0.960**, AM-USB **0.968**, 16FSK **0.991**, 32PSK **0.967**, 802.11a (Wi-Fi) **0.940**, Zigbee **0.921**, GSM **0.904**, Bluetooth LE **0.898**, LoRa **0.890**, P25 **0.840**, DMR **0.816**, LFM-Radar **0.816**.
- **Macro F1**: Unweighted mean of per-class F1 scores across all classes.
- **Per-Class Report**: Printed on training completion showing individual precision/recall/F1 for every modulation.

## Generated Artifacts

```text
models/amc_raw_iq_best.pt              # Best PyTorch model weights & training metadata
models/amc_raw_iq_best_training.png    # 4-panel curves (Loss, Class Acc, Family Acc, LR)
models/amc_raw_iq.onnx                 # Standalone ONNX model (Input: [B, 2, 4096])
```

## Deploying to TensorRT

1. Copy the exported ONNX model to the target deployment system:

   ```bash
   scp models/amc_raw_iq.onnx nvidia@<dgx-ip>:~/holohub/applications/hitek9081/models/
   ```

2. Point your application configuration (`hitek9081.yaml`) to the raw-IQ model:

   ```yaml
   spectral:
     amc_model_path: "/workspace/holohub/applications/hitek9081/models/amc_raw_iq.onnx"
   ```

3. Delete stale engine files to force TensorRT to build an engine matching the `[B, 2, 4096]` input tensor:

   ```bash
   rm -f models/amc_raw_iq*.engine
   ```

4. Recommended application runtime parameters:

   ```bash
   ./holohub run --as-root --docker-opts="--privileged" \
     --run-args="--sr 491.52 --center-freq 2400 --fft-size 131072 --num-averages 1" \
     hitek9081
   ```

   - **`--fft-size 131072` (128k) or `262144` (256k)**: Delivers continuous, non-tiled 4096-sample captures for wideband signals (such as 20 MHz Wi-Fi 802.11a and LTE) at fs = 491.52 MSPS.
   - **`--num-averages 1` (Default)**: Maximizes detection probability and peak signal power for transient packet bursts (Wi-Fi beacons, Bluetooth LE, ADS-B).
