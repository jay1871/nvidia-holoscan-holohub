#!/usr/bin/env python3
"""Train EfficientNet-B4 AMC on synthetic 2D spectrograms (31 modulation classes).

Generates IQ → STFT → spectrogram [1,128,64] on-the-fly.
Supports resume from checkpoint. Exports ONNX on completion.

    python train_amc.py --epochs 200 --backbone efficientnet_b4 --output models/amc_model.onnx
    python train_amc.py --resume models/amc_best.pt --epochs 1000 --output models/amc_model.onnx
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except ImportError:
    logging.error("timm not installed: pip install timm")
    sys.exit(1)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =============================================================================
# Modulation signal generators
# =============================================================================

MODULATION_CLASSES = [
    "OOK", "4PAM", "BPSK", "QPSK", "8PSK", "16PSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "32PSK",
    "2FSK", "4FSK", "8FSK", "16FSK",
    "AM-SSB", "AM-DSB", "FM", "GMSK", "MSK", "OQPSK",
    "CPFSK", "64PSK",
    "OFDM-QPSK", "OFDM-16QAM", "OFDM-64QAM", "OFDM-256QAM",
    "WCDMA", "MC-GSM", "Noise"
]
NUM_CLASSES = len(MODULATION_CLASSES)


def generate_psk(n_samples: int, M: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """M-PSK: uniform phase constellation, repeat-upsampled."""
    n_symbols = n_samples // sps
    symbols = np.random.randint(0, M, n_symbols)
    phases = 2 * np.pi * symbols / M
    baseband = np.exp(1j * phases)
    sig = np.repeat(baseband, sps)[:n_samples]
    return add_noise(sig, snr_db)


def generate_qam(n_samples: int, M: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """M-QAM: square constellation, normalized to unit average power."""
    k = int(np.sqrt(M))
    n_symbols = n_samples // sps
    symbols_i = np.random.randint(0, k, n_symbols) * 2 - (k - 1)
    symbols_q = np.random.randint(0, k, n_symbols) * 2 - (k - 1)
    baseband = (symbols_i + 1j * symbols_q) / np.sqrt(2 * (M - 1) / 3)
    sig = np.repeat(baseband, sps)[:n_samples]
    return add_noise(sig, snr_db)


def generate_fsk(n_samples: int, M: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """M-FSK: per-symbol frequency deviation scaled by 1/sps."""
    sym_len = max(sps * 2, 8)
    n_symbols = n_samples // sym_len
    symbols = np.random.randint(0, M, n_symbols)
    freq_offsets = (symbols - (M - 1) / 2) * (0.4 / sps)
    sig = np.zeros(n_samples, dtype=np.complex128)
    t = np.arange(sym_len, dtype=np.float64) / sym_len
    for i, f in enumerate(freq_offsets):
        start = i * sym_len
        end = min(start + sym_len, n_samples)
        sig[start:end] = np.exp(2j * np.pi * f * t[:end - start])
    return add_noise(sig, snr_db)


def generate_am(n_samples: int, mode: str, snr_db: float, sps: int = 4) -> np.ndarray:
    """AM-DSB or AM-SSB via Hilbert transform."""
    t = np.arange(n_samples, dtype=np.float64) / n_samples
    msg_freq = max(1, 20 // sps)
    msg = np.sin(2 * np.pi * msg_freq * t) + 0.5 * np.sin(2 * np.pi * msg_freq * 2.6 * t)
    msg = msg / np.max(np.abs(msg))
    if mode == "dsb":
        sig = (1 + 0.8 * msg) * np.exp(2j * np.pi * 0.1 * t)
    else:  # ssb
        analytic = hilbert(msg)
        sig = analytic * np.exp(2j * np.pi * 0.1 * t)
    return add_noise(sig.astype(np.complex128), snr_db)


def generate_fm(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """FM: cumulative phase from multi-tone message."""
    t = np.arange(n_samples, dtype=np.float64) / n_samples
    msg_freq = max(1, 12 // sps)
    msg = np.sin(2 * np.pi * msg_freq * t) + 0.3 * np.sin(2 * np.pi * msg_freq * 2.3 * t)
    phase = np.cumsum(msg) * 0.5 / n_samples * 2 * np.pi * 50
    sig = np.exp(1j * phase)
    return add_noise(sig, snr_db)


def generate_cpfsk(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Continuous-phase FSK: h=0.5, binary symbols."""
    n_symbols = n_samples // sps
    symbols = np.random.choice([-1, 1], n_symbols)
    phase = np.cumsum(np.repeat(symbols, sps)[:n_samples]) * np.pi * 0.5 / sps
    sig = np.exp(1j * phase)
    return add_noise(sig, snr_db)


def generate_gmsk(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """GMSK: Gaussian-filtered MSK, BT=0.3."""
    n_symbols = n_samples // sps
    symbols = np.random.choice([-1, 1], n_symbols)
    bt_product = 0.3
    sigma = sps / (2 * np.pi * bt_product)
    filtered = gaussian_filter1d(np.repeat(symbols.astype(float), sps)[:n_samples], sigma=sigma)
    phase = np.cumsum(filtered) * np.pi * 0.5 / sps
    sig = np.exp(1j * phase)
    return add_noise(sig, snr_db)


def generate_msk(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Minimum Shift Keying: h=0.5, no Gaussian filter."""
    n_symbols = n_samples // sps
    symbols = np.random.choice([-1, 1], n_symbols)
    phase = np.cumsum(np.repeat(symbols, sps)[:n_samples]) * np.pi * 0.5 / sps
    sig = np.exp(1j * phase)
    return add_noise(sig, snr_db)


def generate_oqpsk(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Offset QPSK: Q channel delayed by half symbol."""
    n_symbols = n_samples // sps
    symbols_i = np.random.choice([-1, 1], n_symbols)
    symbols_q = np.random.choice([-1, 1], n_symbols)
    i_sig = np.repeat(symbols_i, sps)[:n_samples].astype(float)
    q_sig = np.repeat(symbols_q, sps)[:n_samples].astype(float)
    q_sig = np.roll(q_sig, sps // 2)
    sig = (i_sig + 1j * q_sig) / np.sqrt(2)
    return add_noise(sig, snr_db)


def generate_ook(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """On-Off Keying: binary amplitude, 2×sps symbol length."""
    sym_len = max(sps * 2, 8)
    n_symbols = n_samples // sym_len
    symbols = np.random.choice([0, 1], n_symbols)
    sig = np.repeat(symbols.astype(float), sym_len)[:n_samples] + 0j
    return add_noise(sig, snr_db)


def generate_pam(n_samples: int, M: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """M-PAM: real-valued amplitude levels, normalized."""
    n_symbols = n_samples // sps
    symbols = np.random.randint(0, M, n_symbols) * 2 - (M - 1)
    symbols = symbols / (M - 1)
    sig = np.repeat(symbols, sps)[:n_samples] + 0j
    return add_noise(sig, snr_db)


def add_noise(sig: np.ndarray, snr_db: float) -> np.ndarray:
    """AWGN at specified SNR (dB)."""
    sig = sig.astype(np.complex64)
    power = np.mean(np.abs(sig) ** 2)
    if power < 1e-10:
        power = 1.0
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
    )
    return (sig + noise).astype(np.complex64)


def generate_ofdm(n_samples: int, subcarrier_mod: str, snr_db: float, sps: int = 4) -> np.ndarray:
    """OFDM with CP, pilots, DC/guard nulls. nfft scales with sps."""
    # OFDM parameters scaled to samples-per-symbol
    nfft = min(256, max(64, sps * 16))
    cp_len = nfft // 4
    symbol_len = nfft + cp_len
    n_symbols = (n_samples // symbol_len) + 2

    # Subcarrier modulation constellation
    if subcarrier_mod == "QPSK":
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif subcarrier_mod == "16QAM":
        pts = np.array([-3, -1, 1, 3])
        constellation = (pts[:, None] + 1j * pts[None, :]).flatten() / np.sqrt(10)
    elif subcarrier_mod == "64QAM":
        pts = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        constellation = (pts[:, None] + 1j * pts[None, :]).flatten() / np.sqrt(42)
    elif subcarrier_mod == "256QAM":
        pts = np.arange(-15, 16, 2)
        constellation = (pts[:, None] + 1j * pts[None, :]).flatten()
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    else:
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    # Active subcarriers (exclude DC, guard bands)
    guard = nfft // 8
    active_subs = list(range(guard, nfft // 2)) + list(range(nfft // 2 + 1, nfft - guard))
    pilot_spacing = 6
    pilot_subs = active_subs[::pilot_spacing]
    data_subs = [s for s in active_subs if s not in pilot_subs]

    sig_out = np.zeros(0, dtype=np.complex64)
    for _ in range(n_symbols):
        freq_domain = np.zeros(nfft, dtype=np.complex64)

        # Data subcarriers
        data_symbols = constellation[np.random.randint(0, len(constellation), len(data_subs))]
        freq_domain[data_subs] = data_symbols

        # Pilot subcarriers (BPSK, known pattern)
        pilot_vals = np.random.choice([-1, 1], len(pilot_subs)).astype(np.complex64)
        freq_domain[pilot_subs] = pilot_vals

        # IFFT → time domain
        time_domain = np.fft.ifft(freq_domain).astype(np.complex64)

        # Add cyclic prefix
        symbol = np.concatenate([time_domain[-cp_len:], time_domain])
        sig_out = np.concatenate([sig_out, symbol])

    return add_noise(sig_out[:n_samples], snr_db)


def generate_wcdma(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """WCDMA downlink: OVSF spreading + scrambling, 3-5 code channels."""
    spread_factor = max(4, min(256, sps * 4))

    # Generate multiple code channels
    n_chips = n_samples
    sig = np.zeros(n_chips, dtype=np.complex64)

    # 3-4 traffic channels + pilot
    n_channels = np.random.randint(3, 6)
    for ch in range(n_channels):
        # Random QPSK symbols
        n_sym = n_chips // spread_factor + 1
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], n_sym) / np.sqrt(2)

        # Spreading with random PN code
        spread_code = np.random.choice([-1, 1], spread_factor)
        spread_sig = np.repeat(symbols, spread_factor)[:n_chips] * np.tile(spread_code, n_chips // spread_factor + 1)[:n_chips]

        # Scrambling with complex PN sequence
        scramble = (np.random.choice([-1, 1], n_chips) + 1j * np.random.choice([-1, 1], n_chips)) / np.sqrt(2)
        spread_sig = spread_sig * scramble

        # Random power per channel
        power = np.random.uniform(0.3, 1.0)
        sig += power * spread_sig.astype(np.complex64)

    # Root raised cosine pulse shape (simplified — just lowpass)
    sig_real = gaussian_filter1d(sig.real, sigma=0.8)
    sig_imag = gaussian_filter1d(sig.imag, sigma=0.8)
    sig = (sig_real + 1j * sig_imag).astype(np.complex64)

    return add_noise(sig[:n_samples], snr_db)


def generate_mc_gsm(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Multi-carrier GSM: 2-8 independent GMSK carriers, BT=0.3."""
    n_carriers = np.random.randint(2, 9)
    sig = np.zeros(n_samples, dtype=np.complex64)

    carrier_spacing = 1.0 / (n_carriers + 1)  # normalized spacing

    for c in range(n_carriers):
        # Generate GMSK for this carrier
        n_sym = n_samples // sps + 1
        symbols = np.random.choice([-1, 1], n_sym)
        bt_product = 0.3
        sigma = sps / (2 * np.pi * bt_product)
        filtered = gaussian_filter1d(np.repeat(symbols.astype(float), sps)[:n_samples], sigma=sigma)
        phase = np.cumsum(filtered) * np.pi * 0.5 / sps
        carrier_sig = np.exp(1j * phase).astype(np.complex64)

        # Frequency shift this carrier
        freq_offset = (c - n_carriers / 2.0) * carrier_spacing
        t = np.arange(n_samples, dtype=np.float32)
        carrier_sig = carrier_sig * np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)

        # Random power per carrier
        power = np.random.uniform(0.5, 1.0)
        sig += power * carrier_sig

    return add_noise(sig[:n_samples], snr_db)


def generate_noise(n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Unit-power complex Gaussian noise. Represents TDD gaps / empty spectrum."""
    sig = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    sig = sig / np.sqrt(2)  # unit power noise
    return sig


def apply_channel_impairments(sig: np.ndarray, snr_db: float) -> np.ndarray:
    """Rayleigh fading → CFO → phase noise → time offset → AWGN."""
    n = len(sig)

    # 1. Multipath Rayleigh fading (2-5 taps)
    num_taps = np.random.randint(2, 6)
    delays = np.sort(np.random.randint(0, 20, num_taps))
    powers = np.exp(-0.5 * delays / delays[-1]) if delays[-1] > 0 else np.ones(num_taps)
    powers = powers / np.sum(powers)
    coeffs = np.sqrt(powers) * (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    faded = np.zeros(n, dtype=np.complex64)
    for i, (delay, coeff) in enumerate(zip(delays, coeffs)):
        if delay < n:
            faded[delay:] += coeff * sig[:n - delay]
    sig = faded

    # 2. Carrier Frequency Offset (CFO): random offset up to ±5% of symbol rate
    cfo_normalized = np.random.uniform(-0.05, 0.05)
    t = np.arange(n, dtype=np.float32)
    sig = sig * np.exp(2j * np.pi * cfo_normalized * t / n * n).astype(np.complex64)

    # 3. Phase noise (Wiener process): random walk on phase
    phase_noise_std = np.random.uniform(0.001, 0.02)  # radians per sample
    phase_walk = np.cumsum(np.random.randn(n) * phase_noise_std)
    sig = sig * np.exp(1j * phase_walk).astype(np.complex64)

    # 4. Random phase offset
    phase_offset = np.random.uniform(0, 2 * np.pi)
    sig = sig * np.exp(1j * phase_offset).astype(np.complex64)

    # 5. Random time offset (fractional sample delay via linear interpolation)
    frac_delay = np.random.uniform(0, 1)
    if frac_delay > 0.01:
        idx = np.arange(n, dtype=np.float32) - frac_delay
        idx_floor = np.floor(idx).astype(int)
        idx_frac = idx - idx_floor
        idx_floor = np.clip(idx_floor, 0, n - 2)
        sig = sig[idx_floor] * (1 - idx_frac) + sig[idx_floor + 1] * idx_frac

    # 6. AWGN
    sig = add_noise(sig, snr_db)

    return sig


def ensure_length(sig: np.ndarray, n_samples: int) -> np.ndarray:
    """Truncate or tile-repeat to exactly n_samples."""
    sig = np.asarray(sig, dtype=np.complex64)
    if len(sig) >= n_samples:
        return sig[:n_samples]
    # Pad by repeating
    repeats = (n_samples // len(sig)) + 1
    return np.tile(sig, repeats)[:n_samples]


def generate_signal(class_idx: int, n_samples: int, snr_db: float, sps: int = 4) -> np.ndarray:
    """Dispatch to modulation-specific generator, enforce exact output length."""
    name = MODULATION_CLASSES[class_idx]

    generators = {
        "OOK":       lambda: generate_ook(n_samples, snr_db, sps),
        "4PAM":      lambda: generate_pam(n_samples, 4, snr_db, sps),
        "BPSK":      lambda: generate_psk(n_samples, 2, snr_db, sps),
        "QPSK":      lambda: generate_psk(n_samples, 4, snr_db, sps),
        "8PSK":      lambda: generate_psk(n_samples, 8, snr_db, sps),
        "16PSK":     lambda: generate_psk(n_samples, 16, snr_db, sps),
        "32PSK":     lambda: generate_psk(n_samples, 32, snr_db, sps),
        "64PSK":     lambda: generate_psk(n_samples, 64, snr_db, sps),
        "16QAM":     lambda: generate_qam(n_samples, 16, snr_db, sps),
        "32QAM":     lambda: generate_qam(n_samples, 32, snr_db, sps),
        "64QAM":     lambda: generate_qam(n_samples, 64, snr_db, sps),
        "128QAM":    lambda: generate_qam(n_samples, 128, snr_db, sps),
        "256QAM":    lambda: generate_qam(n_samples, 256, snr_db, sps),
        "2FSK":      lambda: generate_fsk(n_samples, 2, snr_db, sps),
        "4FSK":      lambda: generate_fsk(n_samples, 4, snr_db, sps),
        "8FSK":      lambda: generate_fsk(n_samples, 8, snr_db, sps),
        "16FSK":     lambda: generate_fsk(n_samples, 16, snr_db, sps),
        "AM-SSB":    lambda: generate_am(n_samples, "ssb", snr_db, sps),
        "AM-DSB":    lambda: generate_am(n_samples, "dsb", snr_db, sps),
        "FM":        lambda: generate_fm(n_samples, snr_db, sps),
        "GMSK":      lambda: generate_gmsk(n_samples, snr_db, sps),
        "MSK":       lambda: generate_msk(n_samples, snr_db, sps),
        "OQPSK":     lambda: generate_oqpsk(n_samples, snr_db, sps),
        "CPFSK":     lambda: generate_cpfsk(n_samples, snr_db, sps),
        "OFDM-QPSK": lambda: generate_ofdm(n_samples, "QPSK", snr_db, sps),
        "OFDM-16QAM": lambda: generate_ofdm(n_samples, "16QAM", snr_db, sps),
        "OFDM-64QAM": lambda: generate_ofdm(n_samples, "64QAM", snr_db, sps),
        "OFDM-256QAM": lambda: generate_ofdm(n_samples, "256QAM", snr_db, sps),
        "WCDMA":     lambda: generate_wcdma(n_samples, snr_db, sps),
        "MC-GSM":    lambda: generate_mc_gsm(n_samples, snr_db, sps),
        "Noise":     lambda: generate_noise(n_samples, snr_db, sps),
    }
    gen = generators.get(name, lambda: generate_psk(n_samples, 4, snr_db, sps))
    return ensure_length(gen(), n_samples)


# =============================================================================
# Dataset
# =============================================================================

class AMCDataset(Dataset):
    """On-the-fly synthetic modulation dataset."""

    def __init__(self, samples_per_class=1000, n_iq_samples=4096, snr_range=(0, 30)):
        self.samples_per_class = samples_per_class
        self.n_iq_samples = n_iq_samples
        self.snr_range = snr_range
        self.total = NUM_CLASSES * samples_per_class

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        class_idx = idx % NUM_CLASSES
        snr_db = np.random.uniform(*self.snr_range)

        # Random oversampling ratio (2×–32× samples per symbol)
        sps = np.random.choice([2, 4, 8, 12, 16, 24, 32])

        # Generate clean signal at the selected oversampling ratio
        sig = generate_signal(class_idx, self.n_iq_samples, snr_db=100, sps=sps)

        # Apply realistic channel impairments + AWGN
        sig = apply_channel_impairments(sig, snr_db)

        # Normalize to unit power (same as DDC output)
        power = np.mean(np.abs(sig) ** 2)
        if power > 1e-10:
            sig = sig / np.sqrt(power)

        # Convert to spectrogram [1, 128, 64]
        spec = iq_to_spectrogram(sig, spec_time=128, spec_freq=64)

        return torch.from_numpy(spec), class_idx


# =============================================================================
# Model (same as export script)
# =============================================================================

class SpectrogramAMC(nn.Module):
    """EfficientNet-B4 for spectrogram-based modulation classification.

    Input: [B, 1, spec_time, spec_freq] magnitude spectrogram
    Output: [B, num_classes] logits
    """
    def __init__(self, num_classes=31, backbone="efficientnet_b4"):
        super().__init__()
        self.model = timm.create_model(
            backbone, pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.model(x)


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
        # Take center spec_freq bins
        center = len(fft_out) // 2
        half = spec_freq // 2
        mag = np.abs(fft_out[center - half:center + half])
        mag = np.maximum(mag, 1e-10)
        spec[t, :] = 20.0 * np.log10(mag)

    # Normalize spectrogram to [0, 1]
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 1e-6:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)

    return spec.reshape(1, spec_time, spec_freq)  # [1, T, F]


# =============================================================================
# Training loop
# =============================================================================

def plot_training_history(history, output_path):
    """Plot loss and accuracy curves, save to PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss
        ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history["train_acc"], 'b-', label='Train Acc')
        ax2.plot(epochs, history["val_acc"], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        log.info("Training plot saved: %s", output_path)
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total_start = time.time()

    model = SpectrogramAMC(num_classes=NUM_CLASSES, backbone=args.backbone).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{param_count:,}")

    start_epoch = 0
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

    train_ds = AMCDataset(samples_per_class=args.samples_per_class,
                          n_iq_samples=args.input_length,
                          snr_range=(args.snr_min, args.snr_max))
    val_ds = AMCDataset(samples_per_class=args.samples_per_class // 5,
                        n_iq_samples=args.input_length,
                        snr_range=(args.snr_min, args.snr_max))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    if args.resume and isinstance(ckpt, dict):
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_acc = ckpt.get("best_acc", 0)
        if "history" in ckpt:
            history = ckpt["history"]

    log.info("Training %d epochs (%d samples/epoch, SNR [%d, %d] dB)",
             args.epochs - start_epoch, args.samples_per_class * NUM_CLASSES,
             args.snr_min, args.snr_max)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
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

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        val_acc = 100.0 * val_correct / val_total

        # Track history
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        total_elapsed = time.time() - total_start
        eta = total_elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)

        log.info("Epoch %d/%d | Loss: %.4f | Train: %.1f%% | Val: %.1f%% | %.1fs/epoch | ETA: %.1fh",
                 epoch + 1, args.epochs, avg_loss, train_acc, val_acc, elapsed, eta / 3600)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "history": history,
            }, args.checkpoint)
            log.info("  → Saved best model (%.1f%%)", val_acc)

    total_time = time.time() - total_start
    log.info("Training complete. Best val: %.1f%% | Time: %.2fh | Checkpoint: %s",
             best_acc, total_time / 3600, args.checkpoint)

    # Save training plot
    plot_path = args.checkpoint.replace(".pt", "_training.png")
    plot_training_history(history, plot_path)

    # Export to ONNX
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state"])
    else:
        model.load_state_dict(ckpt_data)
    model.eval()
    dummy = torch.randn(1, 1, 128, 64).to(device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        model.cpu(), dummy.cpu(), args.output,
        input_names=["spectrogram"],
        output_names=["class_probs"],
        dynamic_axes={"spectrogram": {0: "batch"}, "class_probs": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # Embed weights
    try:
        import onnx
        m = onnx.load(args.output)
        onnx.save(m, args.output, save_as_external_data=False)
    except Exception:
        pass

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    log.info("ONNX exported: %s (%.1f MB)", args.output, size_mb)


def main():
    parser = argparse.ArgumentParser(description="Train AMC EfficientNet-1D")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--backbone", default="efficientnet_b4",
                        help="timm model: efficientnet_b0, efficientnet_b2, efficientnet_b4")
    parser.add_argument("--samples-per-class", type=int, default=5000,
                        help="Training samples per modulation class")
    parser.add_argument("--snr-min", type=float, default=-20,
                        help="Minimum SNR in dB for training data")
    parser.add_argument("--snr-max", type=float, default=30,
                        help="Maximum SNR in dB for training data")
    parser.add_argument("--checkpoint", default="models/amc_best.pt")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint (.pt file) to continue training")
    parser.add_argument("--output", default="models/amc_model.onnx")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
