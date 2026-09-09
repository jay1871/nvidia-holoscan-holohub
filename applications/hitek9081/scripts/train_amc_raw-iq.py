#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train ConvNeXt-1D / ResNet-1D AMC directly on raw IQ data using TorchSig.

Feeds raw complex IQ samples [B, 2, N] (In-phase and Quadrature channels)
directly into 1D deep neural networks, preserving complete phase trajectories,
constellation geometries, and symbol transitions directly in the complex domain.

Dynamically synthesizes fresh TorchSig signals every batch with randomized SNR,
multipath fading, and frequency offsets to eliminate data memorization.

Setup:
    python3 -m venv ~/amc_venv && source ~/amc_venv/bin/activate
    pip install torch onnx onnxscript numpy scipy matplotlib
    pip install git+https://github.com/TorchDSP/torchsig.git

Train ConvNeXt-1D (recommended, highest accuracy):
    python scripts/train_amc_raw-iq.py --epochs 250 --batch-size 128 \
        --backbone convnext1d --output models/amc_raw_iq.onnx

Train ResNet-1D (lightweight, lowest latency):
    python scripts/train_amc_raw-iq.py --epochs 250 --batch-size 128 \
        --backbone resnet1d --output models/amc_raw_iq.onnx

Resume:
    python scripts/train_amc_raw-iq.py --resume models/amc_best.pt --epochs 500
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.signals.signal_lists import TorchSigSignalLists
    from torchsig.transforms.transforms import ComplexTo2D
    from torchsig.utils.data_loading import WorkerSeedingDataLoader
    from torchsig.utils.defaults import TorchSigDefaults

    HAS_TORCHSIG = True
except ImportError:
    HAS_TORCHSIG = False
    log.warning("TorchSig not installed: pip install git+https://github.com/TorchDSP/torchsig.git")

warnings.filterwarnings("ignore", message="generated signal is too large")

# ---------------------------------------------------------------------------
# Signal class list -- TorchSig's full 69-class set
# ---------------------------------------------------------------------------
MODULATION_CLASSES = [
        "ook",
        "4ask",
        "8ask",
        "16ask",
        "32ask",
        "64ask",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "bpsk",
        "qpsk",
        "8psk",
        "16psk",
        "32psk",
        "64psk",
        "16qam",
        "32qam",
        "32qam_cross",
        "64qam",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "16apsk",
        "32apsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
        "fm",
        "am-dsb-sc",
        "am-dsb",
        "am-lsb",
        "am-usb",
        "lfm-data",
        "lfm-radar",
        "chirpss",
        "tone",
        "adsb-long",
        "adsb-short",
        "btle",
        "gsm",
        "dvbs2",
        "dmr",
        "p25",
        "lora",
        "80211a",
        "zigbee",
    ]

NUM_CLASSES = len(MODULATION_CLASSES)

# ---------------------------------------------------------------------------
# 14 Family Grouping (aligned with C++ amc_inference.h and amc_constants.h)
# ---------------------------------------------------------------------------
FAMILY_NAMES = [
    "OOK",
    "ASK",
    "FSK",
    "PSK",
    "QAM",
    "OFDM",
    "FM",
    "AM",
    "CHIRP",
    "TONE",
    "SPREAD",
    "TDMA",
    "SAT",
    "BURST",
]

NUM_FAMILIES = len(FAMILY_NAMES)


def get_class_family_index(class_name: str) -> int:
    """Map a signal class name to one of the 14 families."""
    name = class_name.lower().replace("_", "-")
    if name == "ook":
        return 0  # OOK
    elif "ask" in name:
        return 1  # ASK
    elif "fsk" in name or "msk" in name:
        return 2  # FSK
    elif "apsk" in name or "psk" in name:
        return 3  # PSK
    elif "qam" in name:
        return 4  # QAM
    elif "ofdm" in name or name == "80211a":
        return 5  # OFDM
    elif name == "fm":
        return 6  # FM
    elif "am" in name:
        return 7  # AM
    elif "lfm" in name or "chirp" in name:
        return 8  # CHIRP
    elif "tone" in name:
        return 9  # TONE
    elif "lora" in name or "zigbee" in name:
        return 10  # SPREAD
    elif "gsm" in name or "dmr" in name or "p25" in name:
        return 11  # TDMA
    elif "dvbs2" in name:
        return 12  # SAT
    elif "adsb" in name or "btle" in name:
        return 13  # BURST
    return 0


# Mapping from class index to family index (aligned with C++ amc_inference.h)
CLASS_FAMILY = [get_class_family_index(c) for c in MODULATION_CLASSES]


def compute_family_predictions(
    probs: torch.Tensor, class_family: list[int], num_families: int = NUM_FAMILIES
) -> torch.Tensor:
    """Sum class probabilities per family and return the argmax family index."""
    b = probs.shape[0]
    fam_probs = torch.zeros((b, num_families), device=probs.device, dtype=probs.dtype)
    for c_idx, f_idx in enumerate(class_family):
        fam_probs[:, f_idx] += probs[:, c_idx]
    return torch.argmax(fam_probs, dim=-1)


# ---------------------------------------------------------------------------
# Metrics: Macro F1 and Family Accuracy
# ---------------------------------------------------------------------------


def compute_per_class_f1(targets: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    """Compute per-class F1 score."""
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
        denom_prec = tp[c] + fp[c]
        denom_rec = tp[c] + fn[c]
        prec = tp[c] / denom_prec if denom_prec > 0 else 0.0
        rec = tp[c] / denom_rec if denom_rec > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# ---------------------------------------------------------------------------
# Raw IQ Dataset Collate Function
# ---------------------------------------------------------------------------


def make_raw_iq_collate_fn(cfo_max: float = 0.0, phase_jitter: bool = True):
    """Create collate function for raw IQ [2, N] tensors with power normalization and CFO augmentation."""

    def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        iqs = []
        targets = []
        for sample in batch:
            data, class_idx = sample
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            # data is [2, N] where channel 0 is I and channel 1 is Q
            data = data.float()
            N = data.shape[-1]

            if cfo_max > 0.0:
                # Random CFO in [-cfo_max, +cfo_max] cycles per sample
                delta_f = (torch.rand(1).item() * 2.0 - 1.0) * cfo_max
                phi0 = torch.rand(1).item() * 2.0 * math.pi if phase_jitter else 0.0
                t = torch.arange(N, dtype=torch.float32)
                phase = 2.0 * math.pi * delta_f * t + phi0
                cos_p = torch.cos(phase)
                sin_p = torch.sin(phase)
                i_rot = data[0] * cos_p - data[1] * sin_p
                q_rot = data[0] * sin_p + data[1] * cos_p
                data = torch.stack([i_rot, q_rot])
            elif phase_jitter:
                phi0 = torch.rand(1).item() * 2.0 * math.pi
                cos_p = math.cos(phi0)
                sin_p = math.sin(phi0)
                i_rot = data[0] * cos_p - data[1] * sin_p
                q_rot = data[0] * sin_p + data[1] * cos_p
                data = torch.stack([i_rot, q_rot])

            power = torch.mean(data[0] ** 2 + data[1] ** 2)
            if power > 1e-10:
                data = data / torch.sqrt(power)
            iqs.append(data)
            targets.append(int(class_idx))
        return torch.stack(iqs), torch.tensor(targets, dtype=torch.long)

    return collate_fn


# ---------------------------------------------------------------------------
# 1D Neural Network Architectures
# ---------------------------------------------------------------------------


class BasicBlock1D(nn.Module):
    """Residual block for 1D raw IQ signals."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet1D(nn.Module):
    """ResNet-18 adapted for 1D raw IQ waveforms [B, 2, N]."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 2,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        channels: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList()
        in_c = channels[0]
        for c, l_count in zip(channels, layers):
            blocks = [BasicBlock1D(in_c, c, stride=1 if in_c == c else 2)]
            for _ in range(l_count - 1):
                blocks.append(BasicBlock1D(c, c, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            in_c = c

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class ConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt block with depthwise convolution and inverted bottleneck."""

    def __init__(self, dim: int, kernel_size: int = 15) -> None:
        super().__init__()
        # Depthwise 1D convolution with large receptive field
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        # Pointwise 1x1 convolutions (inverted bottleneck C -> 4C -> C)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXt1D(nn.Module):
    """ConvNeXt adapted for 1D raw IQ signals [B, 2, N]."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 2,
        depths: tuple[int, ...] = (2, 2, 4, 2),
        dims: tuple[int, ...] = (64, 128, 256, 512),
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem: Conv1d with stride 4 downsamples 4096 -> 1024
        stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(1, dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            down = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock1D(dims[i], kernel_size=kernel_size) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Dropout(p=0.2),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        return self.head(x)


def create_model(backbone: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Model factory for raw IQ classifiers."""
    backbone_lower = backbone.lower()
    if backbone_lower in ("convnext1d", "convnext", "convnext_1d"):
        return ConvNeXt1D(num_classes=num_classes)
    elif backbone_lower in ("resnet1d", "resnet", "resnet_1d", "resnet18_1d"):
        return ResNet1D(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose 'convnext1d' or 'resnet1d'.")


# ---------------------------------------------------------------------------
# Training Plot Generation
# ---------------------------------------------------------------------------


def plot_training_history(history: dict, output_path: str) -> None:
    """Plot Loss, Class Accuracy, Family Accuracy, and Learning Rate curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        has_f1 = "val_f1" in history and len(history["val_f1"]) > 0
        has_lr = "lr" in history and len(history["lr"]) > 0
        has_fam = "val_fam_acc" in history and len(history["val_fam_acc"]) > 0

        ncols = 2 + int(has_fam) + int(has_f1) + int(has_lr)
        _fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))

        epochs = range(1, len(history["train_loss"]) + 1)

        ax_idx = 0
        axes[ax_idx].plot(epochs, history["train_loss"], "b-", label="Train Loss")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].set_title("Cross-Entropy Loss")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

        axes[ax_idx].plot(epochs, history["train_acc"], "b-", label="Train Acc")
        axes[ax_idx].plot(epochs, history["val_acc"], "r-", label="Val Acc (69-class)")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Accuracy (%)")
        axes[ax_idx].set_title("Modulation Accuracy (69 Classes)")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

        if has_fam:
            axes[ax_idx].plot(epochs, history["val_fam_acc"], "m-", label="Val Acc (14-family)")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel("Accuracy (%)")
            axes[ax_idx].set_title("Family Accuracy (14 Families)")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_f1:
            f1_epochs = range(1, len(history["val_f1"]) + 1)
            axes[ax_idx].plot(f1_epochs, history["val_f1"], "g-", label="Macro F1")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel("F1 Score")
            axes[ax_idx].set_title("Validation Macro F1")
            axes[ax_idx].set_ylim(0, 1)
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_lr:
            lr_epochs = range(1, len(history["lr"]) + 1)
            axes[ax_idx].plot(lr_epochs, history["lr"], "c-", label="Learning Rate")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel("LR")
            axes[ax_idx].set_title("Learning Rate")
            axes[ax_idx].set_yscale("log")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        log.info("Training plot saved: %s", output_path)
    except (ImportError, OSError) as err:
        log.warning("Could not generate training plot: %s", err)


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Execute dynamic raw-IQ training and export ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Started: %s", datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"))
    total_start = time.time()

    model = create_model(args.backbone, num_classes=NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: %s | Classes: %d | Families: %d | Parameters: %s",
        args.backbone,
        NUM_CLASSES,
        NUM_FAMILIES,
        f"{param_count:,}",
    )

    start_epoch = 0
    ckpt = None
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 0)
            log.info(
                "  Resumed at epoch %d, best acc: %.1f%%", start_epoch, ckpt.get("best_acc", 0)
            )
        else:
            model.load_state_dict(ckpt)
            log.info("  Loaded model weights (no metadata)")

    if not HAS_TORCHSIG:
        log.error("TorchSig required: pip install git+https://github.com/TorchDSP/torchsig.git")
        sys.exit(1)

    train_samples = args.samples_per_class * NUM_CLASSES

    # Dynamic dataset metadata overrides
    dataset_metadata_overrides = {
        "num_iq_samples_dataset": args.input_length,
        "fft_size": 256,
        "fft_stride": 256,
        "num_signals_max": 1,
        "num_signals_min": 1,
        "noise_power_db": 0,
        "snr_db_min": args.snr_min,
        "snr_db_max": args.snr_max,
        "signal_center_freq_min": 1000,
        "signal_center_freq_max": 2000,
        "sample_rate": 10000,
        "frequency_min": 1000,
        "frequency_max": 2000,
        "cochannel_overlap_probability": 0.0,
        "signal_duration_in_samples_min": args.input_length // 2,
        "signal_duration_in_samples_max": args.input_length,
        "bandwidth_min": args.bandwidth_min,
        "bandwidth_max": args.bandwidth_max,
    }
    base_meta = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base_meta)
    dataset_metadata.update(dataset_metadata_overrides)
    transforms = [ComplexTo2D()]

    # Dynamic datasets: fresh signals every batch
    train_dataset = TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=transforms,
        target_labels=["class_index"],
        signal_generators="all",
    )
    val_dataset = TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=transforms,
        target_labels=["class_index"],
        signal_generators="all",
    )

    max_cores = os.cpu_count() or 4
    if args.workers is not None:
        num_workers = max(1, args.workers)
    else:
        num_workers = max(1, max_cores - 2)
    train_batches_per_epoch = train_samples // args.batch_size
    val_batches_per_epoch = max(20, train_batches_per_epoch // 5)

    collate = make_raw_iq_collate_fn(cfo_max=args.cfo_max, phase_jitter=args.phase_jitter)
    train_loader = WorkerSeedingDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = WorkerSeedingDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    log.info(
        "Dynamic Mode: %d train batches/epoch (%d samples), %d val batches, %d workers (max cores: %d)",
        train_batches_per_epoch,
        train_samples,
        val_batches_per_epoch,
        num_workers,
        max_cores,
    )
    log.info(
        "Config: SNR [%.1f, %.1f] dB | BW [%.0f, %.0f] Hz | CFO max: %.3f | Phase jitter: %s",
        args.snr_min,
        args.snr_max,
        args.bandwidth_min,
        args.bandwidth_max,
        args.cfo_max,
        args.phase_jitter,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    best_fam_acc = 0.0
    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 40
    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_fam_acc": [],
        "val_f1": [],
        "lr": [],
    }

    if args.resume and isinstance(ckpt, dict):
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if args.override_lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
                log.info("  Overrode optimizer LR to: %g", args.lr)
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if args.override_lr:
                scheduler.best = float("inf")
                scheduler.num_bad_epochs = 0
        if args.override_lr:
            best_acc = 0.0
            best_fam_acc = 0.0
            best_val_loss = float("inf")
            early_stop_counter = 0
            log.info("  Fine-tuning mode: reset baseline best_val_loss and best_acc for new distribution")
        else:
            best_acc = ckpt.get("best_acc", 0.0)
            best_fam_acc = ckpt.get("best_fam_acc", 0.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            early_stop_counter = ckpt.get("early_stop_counter", 0)
        if "history" in ckpt and not args.override_lr:
            history = ckpt["history"]

    all_preds: list[int] = []
    all_targets: list[int] = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= train_batches_per_epoch:
                break
            if batch_idx % 100 == 0:
                print(f"\r  batch {batch_idx}/{train_batches_per_epoch}", end="", flush=True)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(target)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += len(target)

        train_loss = total_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        # Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_fam_correct = 0
        val_total = 0
        all_preds.clear()
        all_targets.clear()

        with torch.no_grad():
            for val_idx, (val_data, val_target) in enumerate(val_loader):
                if val_idx >= val_batches_per_epoch:
                    break
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_logits = model(val_data)
                v_loss = criterion(val_logits, val_target)
                val_loss_sum += v_loss.item() * len(val_target)

                probs = torch.softmax(val_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                fam_preds = compute_family_predictions(probs, CLASS_FAMILY, NUM_FAMILIES)
                target_fams = torch.tensor(
                    [CLASS_FAMILY[t] for t in val_target], device=device, dtype=torch.long
                )

                val_correct += (preds == val_target).sum().item()
                val_fam_correct += (fam_preds == target_fams).sum().item()
                val_total += len(val_target)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(val_target.cpu().tolist())

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = 100.0 * val_correct / max(1, val_total)
        val_fam_acc = 100.0 * val_fam_correct / max(1, val_total)
        elapsed = time.time() - t0

        per_class_f1 = compute_per_class_f1(all_targets, all_preds, NUM_CLASSES)
        macro_f1 = float(np.mean(per_class_f1))

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_fam_acc"].append(val_fam_acc)
        history["val_f1"].append(macro_f1)
        history["lr"].append(current_lr)

        total_elapsed = time.time() - total_start
        eta = total_elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)

        log.info(
            "Epoch %d/%d | Loss: %.4f | Train: %.1f%% | Val: %.1f%% (Fam: %.1f%%, loss %.4f) | "
            "F1: %.3f | LR: %.1e | %.1fs | ETA: %.1fh",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_acc,
            val_fam_acc,
            val_loss,
            macro_f1,
            current_lr,
            elapsed,
            eta / 3600,
        )

        # Save best model checkpoint
        if val_acc > best_acc or epoch == start_epoch:
            best_acc = val_acc
            best_fam_acc = val_fam_acc
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "best_fam_acc": best_fam_acc,
                    "best_val_loss": val_loss,
                    "early_stop_counter": early_stop_counter,
                    "history": history,
                    "class_list": MODULATION_CLASSES,
                    "family_list": FAMILY_NAMES,
                    "class_family": CLASS_FAMILY,
                    "backbone": args.backbone,
                    "input_length": args.input_length,
                },
                args.checkpoint,
            )
            log.info(
                "  -> Saved best model (Class Acc: %.1f%%, Family Acc: %.1f%%)",
                val_acc,
                val_fam_acc,
            )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                log.info(
                    "Early stopping at epoch %d -- val loss not improving for %d epochs",
                    epoch + 1,
                    early_stop_patience,
                )
                break

    total_time = time.time() - total_start
    log.info(
        "Training complete. Best Class Acc: %.1f%% | Best Family Acc: %.1f%% | Time: %.2fh",
        best_acc,
        best_fam_acc,
        total_time / 3600,
    )

    # Per-class & per-family F1 summary
    if all_preds and all_targets:
        per_class = compute_per_class_f1(all_targets, all_preds, NUM_CLASSES)
        log.info("Per-class F1 summary:")
        for c in range(NUM_CLASSES):
            log.info(
                "  %16s [%s] F1=%.3f",
                MODULATION_CLASSES[c],
                FAMILY_NAMES[CLASS_FAMILY[c]],
                per_class[c],
            )
        log.info("  %16s  MACRO F1=%.3f", "OVERALL", float(np.mean(per_class)))

    # Save training history curve
    plot_path = args.checkpoint.replace(".pt", "_training.png")
    plot_training_history(history, plot_path)

    # Export ONNX model for TensorRT deployment
    log.info("Exporting ONNX model to %s...", args.output)
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state"])
    else:
        model.load_state_dict(ckpt_data)

    model.eval()
    dummy = torch.randn(1, 2, args.input_length).to(device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        args.output,
        input_names=["iq"],
        output_names=["class_probs"],
        dynamic_axes={"iq": {0: "batch"}, "class_probs": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )

    try:
        import onnx

        m = onnx.load(args.output)
        onnx.save(m, args.output, save_as_external_data=False)
    except (ImportError, OSError, ValueError) as err:
        log.debug("ONNX external data consolidation skipped: %s", err)

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    log.info(
        "ONNX exported: %s (%.1f MB, input [B, 2, %d])",
        args.output,
        size_mb,
        args.input_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train raw IQ AMC on TorchSig data")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--workers",
        "--num-workers",
        dest="workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes (default: max CPU cores - 2)",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=4096,
        help="Number of complex IQ samples per signal (matches DDC output buffer)",
    )
    parser.add_argument(
        "--backbone",
        default="convnext1d",
        choices=["convnext1d", "resnet1d"],
        help="1D neural network architecture: 'convnext1d' (default) or 'resnet1d'",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2000,
        help="Dynamic training samples synthesized per class per epoch",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=-5.0,
        help="Minimum SNR in dB for synthetic signals",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=30.0,
        help="Maximum SNR in dB for synthetic signals",
    )
    parser.add_argument(
        "--bandwidth-min",
        type=float,
        default=1000.0,
        help="Minimum signal bandwidth in Hz (sample_rate=10000, default 1000 = 10%% fs)",
    )
    parser.add_argument(
        "--bandwidth-max",
        type=float,
        default=4000.0,
        help="Maximum signal bandwidth in Hz (sample_rate=10000, default 4000 = 40%% fs)",
    )
    parser.add_argument(
        "--cfo-max",
        type=float,
        default=0.0,
        help="Maximum normalized carrier frequency offset (e.g. 0.05 = +/-5%% of sample rate)",
    )
    parser.add_argument(
        "--phase-jitter",
        dest="phase_jitter",
        action="store_true",
        default=True,
        help="Apply random initial phase rotation [0, 2pi) to signals (default: True)",
    )
    parser.add_argument(
        "--no-phase-jitter",
        dest="phase_jitter",
        action="store_false",
        help="Disable random initial phase rotation",
    )
    parser.add_argument(
        "--override-lr",
        action="store_true",
        help="When resuming, override optimizer learning rate with --lr instead of checkpoint LR",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/amc_raw_iq_best.pt",
        help="Path to save best training checkpoint",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--output",
        default="models/amc_raw_iq.onnx",
        help="Path to export final ONNX model",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
