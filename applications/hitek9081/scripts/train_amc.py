#!/usr/bin/env python3
"""Train EfficientNet-B4 AMC on TorchSig spectrograms.

Uses TorchSig's signal generators for realistic modulation data.
Converts IQ → STFT → spectrogram [1,128,64] for 2D CNN input.
Exports ONNX on completion for TensorRT deployment.

Setup:
    python3 -m venv ~/amc_venv && source ~/amc_venv/bin/activate
    pip install torch timm onnx onnxscript numpy scipy matplotlib
    pip install git+https://github.com/TorchDSP/torchsig.git

Train:
    python train_amc.py --epochs 200 --backbone efficientnet_b4 --output models/amc_model.onnx

Resume:
    python train_amc.py --resume models/amc_best.pt --epochs 500 --output models/amc_model.onnx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except ImportError:
    logging.error("timm not installed: pip install timm")
    sys.exit(1)

try:
    from torchsig.signals.signal_lists import TorchSigSignalLists
    from torchsig.transforms.transforms import ComplexTo2D
    from torchsig.utils.defaults import TorchSigDefaults
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.utils.data_loading import WorkerSeedingDataLoader
    HAS_TORCHSIG = True
except ImportError:
    HAS_TORCHSIG = False
    logging.warning("TorchSig not installed: pip install git+https://github.com/TorchDSP/torchsig.git")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import warnings
warnings.filterwarnings("ignore", message="generated signal is too large")

# ---------------------------------------------------------------------------
# Signal class list — TorchSig's full set (must match dataset indices exactly)
# ---------------------------------------------------------------------------
if HAS_TORCHSIG:
    MODULATION_CLASSES = list(TorchSigSignalLists.all_signals)
else:
    MODULATION_CLASSES = [
        "ook", "4ask", "8ask", "16ask", "32ask", "64ask",
        "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk",
        "8fsk", "8gfsk", "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
        "bpsk", "qpsk", "8psk", "16psk", "32psk", "64psk",
        "16qam", "32qam", "32qam_cross", "64qam", "128qam_cross", "256qam",
        "512qam_cross", "1024qam", "16apsk", "32apsk",
        "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
        "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048",
        "fm", "am-dsb-sc", "am-dsb", "am-lsb", "am-usb",
        "lfm-data", "lfm-radar", "chirpss", "tone",
        "adsb-long", "adsb-short", "btle", "gsm", "dvbs2", "dmr", "p25", "lora",
        "80211a", "zigbee",
    ]

NUM_CLASSES = len(MODULATION_CLASSES)


def compute_macro_f1(targets: list[int], preds: list[int], num_classes: int) -> float:
    """Macro-averaged F1 score."""
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for t, p in zip(targets, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(np.mean(f1))


def compute_per_class_f1(targets: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    """Per-class F1 scores."""
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for t, p in zip(targets, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# ---------------------------------------------------------------------------
# Spectrogram conversion
# ---------------------------------------------------------------------------

def iq_to_spectrogram(iq_complex: np.ndarray, spec_time: int = 128, spec_freq: int = 64) -> np.ndarray:
    """STFT magnitude spectrogram [1, T, F]. Hanning window, dB scale, normalized [0,1]."""
    n = len(iq_complex)
    hop = max(1, n // spec_time)
    win_len = spec_freq * 2
    window = np.hanning(win_len)
    spec = np.zeros((spec_time, spec_freq), dtype=np.float32)

    for t in range(spec_time):
        start = t * hop
        end = start + win_len
        if end > n:
            seg = np.zeros(win_len, dtype=np.complex64)
            valid = n - start
            if valid > 0:
                seg[:valid] = iq_complex[start:n]
        else:
            seg = iq_complex[start:end]

        windowed = seg * window
        fft_out = np.fft.fftshift(np.fft.fft(windowed))
        center = len(fft_out) // 2
        half = spec_freq // 2
        mag = np.abs(fft_out[center - half:center + half])
        mag = np.maximum(mag, 1e-10)
        spec[t, :] = 20.0 * np.log10(mag)

    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 1e-6:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)

    return spec.reshape(1, spec_time, spec_freq)


# ---------------------------------------------------------------------------
# Dataset wrapper — TorchSig IQ → spectrogram
# ---------------------------------------------------------------------------

def torchsig_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate TorchSigIterableDataset (data, class_index) → spectrogram tensors."""
    specs = []
    targets = []
    for sample in batch:
        data, class_idx = sample
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        iq = data[0] + 1j * data[1]
        power = np.mean(np.abs(iq) ** 2)
        if power > 1e-10:
            iq = iq / np.sqrt(power)
        spec = iq_to_spectrogram(iq, spec_time=128, spec_freq=64)
        specs.append(torch.from_numpy(spec))
        targets.append(int(class_idx))
    return torch.stack(specs), torch.tensor(targets, dtype=torch.long)


class PrecomputedSpectrogramDataset(Dataset):
    """Loads pre-computed spectrograms from .npz, applies lightweight augmentation."""

    def __init__(self, spec_path: str, augment: bool = True) -> None:
        data = np.load(spec_path)
        self.specs = data["specs"]      # [N, 1, 128, 64] float32
        self.labels = data["labels"]    # [N] int
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = self.specs[idx].copy()  # [1, 128, 64]

        if self.augment:
            # Random brightness/contrast shift on spectrogram
            if np.random.random() < 0.5:
                spec = spec * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)
                spec = np.clip(spec, 0, 1)

            # Time masking
            if np.random.random() < 0.5:
                t_start = np.random.randint(0, 118)
                t_width = np.random.randint(1, 11)
                spec[0, t_start:t_start + t_width, :] = 0

            # Frequency masking
            if np.random.random() < 0.5:
                f_start = np.random.randint(0, 56)
                f_width = np.random.randint(1, 9)
                spec[0, :, f_start:f_start + f_width] = 0

            # Random time roll (circular shift)
            if np.random.random() < 0.3:
                shift = np.random.randint(-20, 20)
                spec[0] = np.roll(spec[0], shift, axis=0)

        return torch.from_numpy(spec), int(self.labels[idx])


def generate_spectrogram_dataset(npz_path: str, num_samples: int, n_iq_samples: int,
                                  seed: int = 42) -> None:
    """Generate TorchSig IQ → pre-compute spectrograms → save as .npz."""
    dataset_metadata_overrides = {
        "num_iq_samples_dataset": n_iq_samples,
        "fft_size": 256,
        "fft_stride": 256,
        "num_signals_max": 1,
        "num_signals_min": 1,
        "noise_power_db": 0,
        "snr_db_min": -5,           # weight toward low SNR for real-world performance
        "snr_db_max": 30,
        "signal_center_freq_min": 1000,
        "signal_center_freq_max": 2000,
        "sample_rate": 10000,
        "frequency_min": 1000,
        "frequency_max": 2000,
        "cochannel_overlap_probability": 0.0,
        "signal_duration_in_samples_min": n_iq_samples // 2,
        "signal_duration_in_samples_max": n_iq_samples,
        "bandwidth_min": 1000,
        "bandwidth_max": 2000,
    }
    base_meta = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base_meta)
    dataset_metadata.update(dataset_metadata_overrides)

    transforms = [ComplexTo2D()]
    dataset = TorchSigIterableDataset(
        metadata=dataset_metadata, transforms=transforms,
        target_labels=["class_index"], signal_generators="all")

    loader = WorkerSeedingDataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    loader.seed(seed)

    specs = np.zeros((num_samples, 1, 128, 64), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int32)

    log.info("Generating %d spectrograms → %s", num_samples, npz_path)
    t0 = time.time()

    for i, sample in enumerate(loader):
        if i >= num_samples:
            break
        data, class_idx = sample[0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        iq = data[0] + 1j * data[1]

        power = np.mean(np.abs(iq) ** 2)
        if power > 1e-10:
            iq = iq / np.sqrt(power)

        specs[i] = iq_to_spectrogram(iq, spec_time=128, spec_freq=64)
        labels[i] = int(class_idx)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            log.info("  %d/%d (%.0f/s, ETA %.0fm)", i + 1, num_samples, rate, eta / 60)

    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, specs=specs, labels=labels)
    elapsed = time.time() - t0
    size_mb = Path(npz_path).stat().st_size / (1024 * 1024)
    log.info("Saved: %s (%.1f MB, %d samples, %.0fs)", npz_path, size_mb, num_samples, elapsed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpectrogramAMC(nn.Module):
    """EfficientNet backbone for spectrogram AMC. in_chans=1 (magnitude only)."""

    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = "efficientnet_b4") -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone, pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Training plot
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, output_path: str) -> None:
    """Loss, accuracy, and F1 curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        has_f1 = "val_f1" in history and len(history["val_f1"]) > 0
        has_lr = "lr" in history and len(history["lr"]) > 0
        ncols = 2 + int(has_f1) + int(has_lr)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], 'b-', label='Train Acc')
        axes[1].plot(epochs, history["val_acc"], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        ax_idx = 2
        if has_f1:
            f1_epochs = range(1, len(history["val_f1"]) + 1)
            axes[ax_idx].plot(f1_epochs, history["val_f1"], 'g-', label='Macro F1')
            axes[ax_idx].set_xlabel('Epoch')
            axes[ax_idx].set_ylabel('F1 Score')
            axes[ax_idx].set_title('Validation Macro F1')
            axes[ax_idx].set_ylim(0, 1)
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_lr:
            lr_epochs = range(1, len(history["lr"]) + 1)
            axes[ax_idx].plot(lr_epochs, history["lr"], 'm-', label='Learning Rate')
            axes[ax_idx].set_xlabel('Epoch')
            axes[ax_idx].set_ylabel('LR')
            axes[ax_idx].set_title('Learning Rate')
            axes[ax_idx].set_yscale('log')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        log.info("Training plot saved: %s", output_path)
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


# ---------------------------------------------------------------------------
# Dataset generation via TorchSig
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total_start = time.time()

    model = SpectrogramAMC(num_classes=NUM_CLASSES, backbone=args.backbone).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model: %s | Classes: %d | Parameters: %s", args.backbone, NUM_CLASSES, f"{param_count:,}")

    start_epoch = 0
    ckpt = None
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 0)
            log.info("  Resumed at epoch %d, best acc: %.1f%%", start_epoch, ckpt.get("best_acc", 0))
        else:
            model.load_state_dict(ckpt)
            log.info("  Loaded weights (no epoch info)")

    if not HAS_TORCHSIG:
        log.error("TorchSig required: pip install git+https://github.com/TorchDSP/torchsig.git")
        sys.exit(1)

    train_samples = args.samples_per_class * NUM_CLASSES

    dataset_metadata_overrides = {
        "num_iq_samples_dataset": args.input_length,
        "fft_size": 256,
        "fft_stride": 256,
        "num_signals_max": 1,
        "num_signals_min": 1,
        "noise_power_db": 0,
        "snr_db_min": -5,
        "snr_db_max": 30,
        "signal_center_freq_min": 1000,
        "signal_center_freq_max": 2000,
        "sample_rate": 10000,
        "frequency_min": 1000,
        "frequency_max": 2000,
        "cochannel_overlap_probability": 0.0,
        "signal_duration_in_samples_min": args.input_length // 2,
        "signal_duration_in_samples_max": args.input_length,
        "bandwidth_min": 1000,
        "bandwidth_max": 2000,
    }
    base_meta = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base_meta)
    dataset_metadata.update(dataset_metadata_overrides)
    transforms = [ComplexTo2D()]

    if args.dynamic:
        # Dynamic: fresh signals every batch — no memorization possible
        train_dataset = TorchSigIterableDataset(
            metadata=dataset_metadata, transforms=transforms,
            target_labels=["class_index"], signal_generators="all")
        val_dataset = TorchSigIterableDataset(
            metadata=dataset_metadata, transforms=transforms,
            target_labels=["class_index"], signal_generators="all")

        num_workers = min(16, os.cpu_count() or 4)
        train_batches_per_epoch = train_samples // args.batch_size
        val_batches_per_epoch = max(20, train_batches_per_epoch // 5)

        train_loader = WorkerSeedingDataLoader(
            train_dataset, batch_size=args.batch_size,
            collate_fn=torchsig_collate_fn, num_workers=num_workers)
        val_loader = WorkerSeedingDataLoader(
            val_dataset, batch_size=args.batch_size,
            collate_fn=torchsig_collate_fn, num_workers=max(2, num_workers // 2))

        log.info("Dynamic mode: %d train batches/epoch, %d val batches, %d workers",
                 train_batches_per_epoch, val_batches_per_epoch, num_workers)
    else:
        # Static: pre-computed spectrograms + augmentation
        train_npz = str(Path(args.data_root) / "train_specs.npz")
        val_npz = str(Path(args.data_root) / "val_specs.npz")
        val_samples = max(NUM_CLASSES * 10, train_samples // 5)

        if not Path(train_npz).exists() or args.regenerate:
            generate_spectrogram_dataset(train_npz, train_samples, args.input_length, seed=12345)
        if not Path(val_npz).exists() or args.regenerate:
            generate_spectrogram_dataset(val_npz, val_samples, args.input_length, seed=67890)

        train_ds = PrecomputedSpectrogramDataset(train_npz, augment=True)
        val_ds = PrecomputedSpectrogramDataset(val_npz, augment=False)
        log.info("Static mode: %d spectrograms (augmented) | Val: %d | %d classes",
                 len(train_ds), len(val_ds), NUM_CLASSES)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

        train_batches_per_epoch = 0  # use full dataset
        val_batches_per_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 40
    history: dict[str, list] = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "lr": []}

    if args.resume and isinstance(ckpt, dict):
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_acc = ckpt.get("best_acc", 0)
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        if "history" in ckpt:
            history = ckpt["history"]
            history.setdefault("val_f1", [])
            history.setdefault("lr", [])

    log.info("Training %d epochs (%d samples/epoch, %d classes)",
             args.epochs - start_epoch, train_samples, NUM_CLASSES)

    all_preds: list[int] = []
    all_targets: list[int] = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        batch_count = 0

        if args.dynamic:
            train_loader.seed(epoch * 10000)

        for batch_idx, (data, target) in enumerate(train_loader):
            if train_batches_per_epoch > 0 and batch_idx >= train_batches_per_epoch:
                break
            if batch_idx % 100 == 0:
                limit = train_batches_per_epoch if train_batches_per_epoch > 0 else len(train_loader)
                print(f"\r  batch {batch_idx}/{limit}", end="", flush=True)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            batch_count += 1

        print("\r" + " " * 40 + "\r", end="", flush=True)
        avg_loss = train_loss / max(1, batch_count)
        train_acc = 100.0 * correct / max(1, total)
        elapsed = time.time() - t0

        # Validation + F1
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        if args.dynamic:
            val_loader.seed(epoch * 10000 + 99999)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if val_batches_per_epoch > 0 and batch_idx >= val_batches_per_epoch:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss_sum += criterion(output, target).item()
                val_batches += 1
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                all_preds.extend(pred.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss_sum / val_batches
        macro_f1 = compute_macro_f1(all_targets, all_preds, NUM_CLASSES)

        # ReduceLROnPlateau: step with val loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(macro_f1)
        history["lr"].append(current_lr)

        total_elapsed = time.time() - total_start
        eta = total_elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)

        log.info("Epoch %d/%d | Loss: %.4f | Train: %.1f%% | Val: %.1f%% (loss %.4f) | F1: %.3f | LR: %.1e | %.1fs | ETA: %.1fh",
                 epoch + 1, args.epochs, avg_loss, train_acc, val_acc, val_loss, macro_f1,
                 current_lr, elapsed, eta / 3600)

        # Save best model (by val accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "best_val_loss": val_loss,
                "early_stop_counter": early_stop_counter,
                "history": history,
                "class_list": MODULATION_CLASSES,
                "backbone": args.backbone,
            }, args.checkpoint)
            log.info("  → Saved best model (%.1f%%)", val_acc)

        # Early stopping on val loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                log.info("Early stopping at epoch %d — val loss not improving for %d epochs",
                         epoch + 1, early_stop_patience)
                break

    total_time = time.time() - total_start
    log.info("Training complete. Best val: %.1f%% | Time: %.2fh | Checkpoint: %s",
             best_acc, total_time / 3600, args.checkpoint)

    # Per-class F1 report
    if all_preds and all_targets:
        per_class = compute_per_class_f1(all_targets, all_preds, NUM_CLASSES)
        log.info("Per-class F1 (final epoch):")
        for c in range(NUM_CLASSES):
            log.info("  %16s  F1=%.3f", MODULATION_CLASSES[c], per_class[c])
        log.info("  %16s  F1=%.3f", "MACRO", float(np.mean(per_class)))

    plot_path = args.checkpoint.replace(".pt", "_training.png")
    plot_training_history(history, plot_path)

    # Export ONNX
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state"])
    else:
        model.load_state_dict(ckpt_data)
    model.eval()
    dummy = torch.randn(1, 1, 128, 64).to(device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.cpu(), dummy.cpu(), args.output,
        input_names=["spectrogram"],
        output_names=["class_probs"],
        dynamic_axes={"spectrogram": {0: "batch"}, "class_probs": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    try:
        import onnx
        m = onnx.load(args.output)
        onnx.save(m, args.output, save_as_external_data=False)
    except Exception:
        pass

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    log.info("ONNX exported: %s (%.1f MB)", args.output, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AMC on TorchSig data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-length", type=int, default=4096,
                        help="IQ samples per signal (TorchSig num_iq_samples_dataset)")
    parser.add_argument("--backbone", default="efficientnet_b4",
                        help="timm model: efficientnet_b0, efficientnet_b2, efficientnet_b4")
    parser.add_argument("--samples-per-class", type=int, default=2000,
                        help="Training samples per modulation class")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic generation: fresh signals every batch (no memorization, slower)")
    parser.add_argument("--data-root", default="datasets/amc",
                        help="Root for cached TorchSig datasets (static mode only)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate static datasets")
    parser.add_argument("--checkpoint", default="models/amc_best.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default="models/amc_model.onnx")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
