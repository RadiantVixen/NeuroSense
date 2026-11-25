"""Signal-quality screening for raw EEG (descriptive, no data is modified).

These metrics are used in Stage 1 to *describe* data quality. They are not
artifact rejection; thresholds for rejection are decided in Stage 2.

Conventions: ``x`` is an array of shape (n_samples, n_channels).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

MAD_TO_SD = 1.4826  # MAD * 1.4826 estimates sigma for Gaussian data


def highpass(x: np.ndarray, fs: float, cutoff: float = 1.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass, used only to remove DC/drift before
    computing amplitude statistics (drift would otherwise dominate the MAD)."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def max_flat_run(x: np.ndarray) -> np.ndarray:
    """Longest run (in samples) of exactly repeated consecutive values, per channel.

    A run of length L means L+1 identical samples. Short runs are expected from
    ADC quantisation; long runs indicate dropout, clipping or a disconnected electrode.
    """
    same = np.diff(x, axis=0) == 0
    out = np.zeros(x.shape[1], dtype=int)
    for ch in range(x.shape[1]):
        padded = np.concatenate(([0], same[:, ch].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        if edges.size:
            out[ch] = int((edges[1::2] - edges[::2]).max())
    return out


def robust_z(x: np.ndarray) -> np.ndarray:
    """|x - median| / (1.4826 * MAD), per channel. Robust to the outliers it detects."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * MAD_TO_SD
    mad = np.where(mad == 0, np.finfo(float).eps, mad)
    return np.abs(x - med) / mad


def extreme_fraction(x_hp: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Fraction of samples with robust z-score > k, per channel (on high-passed data)."""
    return (robust_z(x_hp) > k).mean(axis=0)


def out_of_range_fraction(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fraction of raw samples outside [low, high], per channel."""
    return ((x < low) | (x > high)).mean(axis=0)


def line_noise_ratio(x: np.ndarray, fs: float, f0: float, nperseg: int = 1024) -> float:
    """Channel-averaged PSD at ``f0`` divided by the median PSD of flanking bins
    (excluding +/-4 bins around f0). ~1 means no spectral peak at f0."""
    f, p = welch(x - x.mean(axis=0), fs=fs, nperseg=nperseg, axis=0)
    p = p.mean(axis=1)
    i = int(np.argmin(np.abs(f - f0)))
    flank = np.r_[max(i - 12, 0): max(i - 4, 0), i + 5: min(i + 13, len(f))]
    return float(p[i] / np.median(p[flank]))
