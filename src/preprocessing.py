"""EEG preprocessing: filtering, artifact masking, bad-channel detection, segmentation.

Every operation here is *label-blind* and acts on one recording at a time, so it
cannot leak information across subjects or folds. Nothing is learned from the
data distribution of other recordings; anything learned (scaling, feature
selection) happens later inside the cross-validated model pipeline.

Pipeline for one recording ``raw`` (samples x channels):

1. ``bad_sample_mask``   - flag raw samples at the ADC rails or inside long flat runs,
                           dilated by a margin to cover filter ringing.
2. ``filter_recording``  - remove DC offset, 50 Hz notch, zero-phase 1-45 Hz band-pass.
3. ``detect_bad_channels`` - channels whose robust amplitude is abnormal *within* the recording.
4. ``segment``           - fixed-length windows; each window gets reject flags.

See ``notebooks/02_signal_processing.ipynb`` for the derivations and parameter choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from data_loader import CHANNELS, FS, Recording


@dataclass(frozen=True)
class PreprocessConfig:
    hp_hz: float = 1.0           # high-pass cut-off: removes DC/drift, keeps delta >= 1 Hz
    lp_hz: float = 45.0          # low-pass: headset roll-off makes > ~42 Hz unreliable
    filter_order: int = 4        # per pass; filtfilt doubles the effective order
    notch_hz: float | None = 50.0
    notch_q: float = 30.0        # -3 dB bandwidth = f0/Q ~ 1.7 Hz
    rail_low: float = 100.0      # raw values this close to 0 / full scale are saturated
    rail_high: float = 8300.0
    flat_min_samples: int = 32   # >= 0.25 s of identical samples = dropout / clipping
    mask_margin_s: float = 1.0   # dilate bad samples to cover filter transients
    window_s: float = 4.0
    step_s: float = 2.0
    edge_trim_s: float = 2.0     # skip recording edges (filter edge effects)
    abs_thresh: float = 200.0    # reject window if any good channel exceeds |x| (filtered units)
    bad_channel_ratio: float = 4.0  # channel robust SD > ratio * median channel robust SD
    reference: str = "original"  # "original" or "average" (common average reference)


def _flat_run_mask(x: np.ndarray, min_len: int) -> np.ndarray:
    """True for samples belonging to runs of >= ``min_len`` + 1 identical consecutive values."""
    mask = np.zeros(x.shape, dtype=bool)
    same = np.diff(x, axis=0) == 0
    for ch in range(x.shape[1]):
        padded = np.concatenate(([0], same[:, ch].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= min_len:
                mask[start : stop + 1, ch] = True
    return mask


def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
    """Binary dilation along time by ``margin`` samples (per channel)."""
    if margin <= 0 or not mask.any():
        return mask
    kernel = np.ones(2 * margin + 1)
    out = np.empty_like(mask)
    for ch in range(mask.shape[1]):
        out[:, ch] = np.convolve(mask[:, ch].astype(float), kernel, mode="same") > 0
    return out


def bad_sample_mask(raw: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Per-sample, per-channel mask of saturated / dropped-out raw samples."""
    rail = (raw <= cfg.rail_low) | (raw >= cfg.rail_high)
    flat = _flat_run_mask(raw, cfg.flat_min_samples)
    return _dilate(rail | flat, int(cfg.mask_margin_s * FS))


def filter_recording(raw: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """DC removal -> optional notch -> zero-phase Butterworth band-pass.

    ``sosfiltfilt`` runs the filter forward and backward: the phase responses
    cancel (zero phase lag, waveform shape preserved) and the magnitude response
    is squared (|H|^2), i.e. -6 dB at the cut-offs and twice the roll-off.
    Second-order sections are used for numerical stability of the low 1/64
    normalised cut-off.
    """
    x = raw - np.median(raw, axis=0)  # median is robust to saturated segments
    if cfg.notch_hz is not None:
        b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=FS)
        x = filtfilt(b, a, x, axis=0)
    sos = butter(cfg.filter_order, [cfg.hp_hz, cfg.lp_hz], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def detect_bad_channels(x_filt: np.ndarray, sample_mask: np.ndarray,
                        cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Boolean per channel: robust SD (on unmasked samples) far above the recording's typical channel.

    Relative criterion within one recording, so a subject with generally high
    amplitudes is not penalised; also flags channels that are >50 % masked.
    """
    rsd = np.empty(x_filt.shape[1])
    for ch in range(x_filt.shape[1]):
        v = x_filt[~sample_mask[:, ch], ch]
        if v.size < FS * 10:
            rsd[ch] = np.inf
            continue
        rsd[ch] = 1.4826 * np.median(np.abs(v - np.median(v)))
    mostly_masked = sample_mask.mean(axis=0) > 0.5
    return (rsd > cfg.bad_channel_ratio * np.median(rsd[np.isfinite(rsd)])) | mostly_masked


def rereference(x: np.ndarray, good: np.ndarray, mode: str) -> np.ndarray:
    """Common average reference over good channels only (bad channels would contaminate the mean)."""
    if mode == "original":
        return x
    if mode == "average":
        return x - x[:, good].mean(axis=1, keepdims=True)
    raise ValueError(mode)


@dataclass
class ProcessedRecording:
    subject: int
    condition: str
    x: np.ndarray              # filtered (samples, channels), float32
    bad_channels: np.ndarray   # (channels,) bool
    windows: pd.DataFrame      # one row per window, with reject flags
    sample_mask: np.ndarray = field(repr=False)

    def window_data(self, keep_only: bool = True) -> np.ndarray:
        """Stack windows -> (n_windows, n_samples, n_channels); bad channels set to NaN."""
        w = self.windows[~self.windows.rejected] if keep_only else self.windows
        n = int(self.windows.attrs["win_samples"])
        out = np.stack([self.x[s : s + n] for s in w.start_sample]) if len(w) else np.empty((0, n, self.x.shape[1]))
        out = out.astype(np.float64)
        out[:, :, self.bad_channels] = np.nan
        return out


def segment(x: np.ndarray, sample_mask: np.ndarray, bad_channels: np.ndarray,
            cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    n = int(cfg.window_s * FS)
    step = int(cfg.step_s * FS)
    trim = int(cfg.edge_trim_s * FS)
    good = ~bad_channels
    rows = []
    for start in range(trim, x.shape[0] - trim - n + 1, step):
        seg = x[start : start + n, good]
        m = sample_mask[start : start + n, good]
        rows.append(dict(start_sample=start, start_s=start / FS,
                         reject_mask=bool(m.any()),
                         reject_amplitude=bool(np.abs(seg).max() > cfg.abs_thresh) if seg.size else True,
                         max_abs=float(np.abs(seg).max()) if seg.size else np.nan))
    df = pd.DataFrame(rows)
    df["rejected"] = df.reject_mask | df.reject_amplitude
    df.attrs["win_samples"] = n
    return df


def preprocess(rec: Recording, cfg: PreprocessConfig = PreprocessConfig()) -> ProcessedRecording:
    raw = rec.data
    mask = bad_sample_mask(raw, cfg)
    x = filter_recording(raw, cfg)
    bad = detect_bad_channels(x, mask, cfg)
    x = rereference(x, ~bad, cfg.reference)
    windows = segment(x, mask, bad, cfg)
    windows.insert(0, "subject", rec.subject)
    windows.insert(1, "condition", rec.condition)
    return ProcessedRecording(rec.subject, rec.condition, x.astype(np.float32), bad, windows, mask)
