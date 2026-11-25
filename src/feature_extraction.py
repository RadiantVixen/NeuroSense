"""Window-level EEG features.

Input: windows array ``W`` of shape (n_windows, n_samples, n_channels); bad
channels are NaN and propagate to NaN features (imputed later, inside CV).

Per channel
-----------
Time domain
  log_var       log10 variance (Hjorth activity)
  mobility      sqrt(var(x') / var(x))            ~ mean frequency
  complexity    mobility(x') / mobility(x)        ~ bandwidth / deviation from a pure sine
  skew, kurt    3rd standardised moment, excess kurtosis (spikiness, artifacts)
  log_ll        log10 mean |x[t+1] - x[t]|          (line length)
Frequency domain (Welch PSD, 2 s Hann segments, 75 % overlap -> 0.5 Hz bins)
  abs_<band>    log10 absolute band power
  rel_<band>    band power / total 1-45 Hz power
  spec_ent      normalised Shannon entropy of the 1-45 Hz PSD
  r_theta_alpha, r_theta_beta, r_engagement   log10 band-power ratios

Regional
  fT_pA         log10 frontal theta / parietal alpha (task-load index)
  faa_F4F3      frontal alpha asymmetry log10(alpha F4) - log10(alpha F3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from data_loader import CHANNELS, FS

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
TOTAL_BAND = (1.0, 45.0)
WELCH_NPERSEG = 2 * FS
WELCH_NOVERLAP = int(1.5 * FS)
FRONTAL = ("AF3", "F3", "F4", "AF4")
PARIETAL = ("P7", "P8")


def window_psd(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per window and channel -> (freqs, P[n_windows, n_freqs, n_channels])."""
    f, p = welch(W, fs=FS, window="hann", nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
                 axis=1, detrend="constant", scaling="density")
    return f, p


def band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate the PSD over [lo, hi) with the rectangle rule (bins are uniform, df = fs/nperseg).

    Half-open intervals make adjacent bands partition the total band exactly.
    """
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(axis=1) * (f[1] - f[0])


def spectral_entropy(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """H = -sum p_k log p_k / log N over the 1-45 Hz bins, with p_k the PSD normalised to sum 1.

    0 = all power in one bin (pure tone); 1 = flat (white) spectrum.
    """
    m = (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])
    q = p[:, m]
    q = q / q.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(q > 0, q * np.log(q), 0.0), axis=1)
    h = h / np.log(m.sum())
    h[np.isnan(p[:, m]).any(axis=1)] = np.nan
    return h


def time_features(W: np.ndarray) -> dict[str, np.ndarray]:
    d1 = np.diff(W, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = W.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mob = np.sqrt(v1 / v0)
    return {
        "log_var": np.log10(v0),
        "mobility": mob,
        "complexity": np.sqrt(v2 / v1) / mob,
        "skew": skew(W, axis=1),
        "kurt": kurtosis(W, axis=1),
        "log_ll": np.log10(np.abs(d1).mean(axis=1)),
    }


def spectral_features(f: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    bp = {b: band_power(f, p, lo, hi) for b, (lo, hi) in BANDS.items()}
    total = band_power(f, p, *TOTAL_BAND)
    out = {f"abs_{b}": np.log10(v) for b, v in bp.items()}
    out |= {f"rel_{b}": v / total for b, v in bp.items()}
    out["spec_ent"] = spectral_entropy(f, p)
    out["r_theta_alpha"] = np.log10(bp["theta"] / bp["alpha"])
    out["r_theta_beta"] = np.log10(bp["theta"] / bp["beta"])
    out["r_engagement"] = np.log10(bp["beta"] / (bp["alpha"] + bp["theta"]))  # Pope et al. 1995
    return out


def extract(W: np.ndarray) -> pd.DataFrame:
    """Feature table (n_windows x n_features) with columns ``<channel>_<feature>`` + regional features."""
    with np.errstate(invalid="ignore", divide="ignore"):
        feats = time_features(W)
        f, p = window_psd(W)
        feats |= spectral_features(f, p)
        cols = {f"{ch}_{name}": v[:, i] for name, v in feats.items() for i, ch in enumerate(CHANNELS)}
        idx = lambda chs: [CHANNELS.index(c) for c in chs]
        theta = band_power(f, p, *BANDS["theta"])
        alpha = band_power(f, p, *BANDS["alpha"])
        cols["fT_pA"] = np.log10(np.nanmean(theta[:, idx(FRONTAL)], axis=1) / np.nanmean(alpha[:, idx(PARIETAL)], axis=1))
        cols["faa_F4F3"] = np.log10(alpha[:, CHANNELS.index("F4")]) - np.log10(alpha[:, CHANNELS.index("F3")])
    return pd.DataFrame(cols)


def feature_groups(columns) -> dict[str, list[str]]:
    """Map group name -> column names, for ablations and importance aggregation."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        name = c.split("_", 1)[1] if c.split("_", 1)[0] in CHANNELS else c
        if name.startswith(("abs_", "rel_")):
            key = name
        elif name.startswith("r_") or name in ("fT_pA", "faa_F4F3"):
            key = "ratios"
        elif name == "spec_ent":
            key = "spec_ent"
        else:
            key = "time"
        groups.setdefault(key, []).append(c)
    return groups


def channel_of(column: str) -> str | None:
    head = column.split("_", 1)[0]
    return head if head in CHANNELS else None
