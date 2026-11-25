"""Analyses that test whether the rest-vs-task classifier relies on brain activity
or on ocular/muscle artifacts and recording drift.

* ``ocular_ica``          ICA (FastICA) with eye components removed, chosen by
                          correlation with frontal EOG proxies.
* ``artifact_features``   a small, hand-built set of artifact markers per window
                          (blink count, EOG-proxy power, forehead/temporal 20-45 Hz power).
* ``window_array``        window extraction at fixed start times, so every variant is
                          scored on exactly the same windows as the main model.
* ``paired_bootstrap``    participant-level bootstrap CIs for an AUC and for the
                          difference between two AUCs on the same windows.

The EOG proxies are built from the frontal EEG channels themselves (STEW has no EOG
channel), so they also contain some frontal brain activity; removing components that
match them can remove neural signal as well as eye signal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from data_loader import CHANNELS, FS

ICA_SEED = 0
EOG_LOWPASS_HZ = 4.0
ICA_R_THRESHOLD = 0.7      # remove a component only if |r| with an EOG proxy >= 0.7
BLINK_Z = 5.0              # blink = |VEOG| peak above 5 robust SDs of the recording's VEOG
BLINK_MIN_SEP_S = 0.3
WIN = 4 * FS

_ix = {c: i for i, c in enumerate(CHANNELS)}
_lp = butter(4, EOG_LOWPASS_HZ, btype="low", fs=FS, output="sos")


def _mean_good(x: np.ndarray, chans: list[str], good: np.ndarray) -> np.ndarray | None:
    idx = [_ix[c] for c in chans if good[_ix[c]]]
    return x[:, idx].mean(axis=1) if idx else None


def eog_proxies(x: np.ndarray, good: np.ndarray) -> dict[str, np.ndarray]:
    """VEOG ~ mean(AF3, AF4); HEOG ~ F7 - F8; both low-passed at 4 Hz. Missing if channels are bad."""
    out = {}
    v = _mean_good(x, ["AF3", "AF4"], good)
    if v is not None:
        out["veog"] = sosfiltfilt(_lp, v)
    if good[_ix["F7"]] and good[_ix["F8"]]:
        out["heog"] = sosfiltfilt(_lp, x[:, _ix["F7"]] - x[:, _ix["F8"]])
    return out


def ocular_ica(x: np.ndarray, good: np.ndarray, sample_ok: np.ndarray,
               threshold: float = ICA_R_THRESHOLD, seed: int = ICA_SEED) -> tuple[np.ndarray, dict]:
    """Remove at most one ICA component per EOG proxy (the best match, if |r| >= threshold).

    ICA is fitted on good channels and unmasked samples of the band-passed signal;
    the cleaned signal is reconstructed for all samples. Bad channels are returned unchanged.
    """
    gi = np.flatnonzero(good)
    ica = FastICA(n_components=len(gi), whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ica.fit(x[sample_ok][:, gi])
    S = ica.transform(x[:, gi])
    Slp = sosfiltfilt(_lp, S, axis=0)
    info = {"n_components": len(gi), "removed": [], "r": {}}
    drop = set()
    for name, proxy in eog_proxies(x, good).items():
        r = np.array([abs(np.corrcoef(Slp[:, k], proxy)[0, 1]) for k in range(S.shape[1])])
        k = int(np.argmax(r))
        info["r"][name] = float(r[k])
        if r[k] >= threshold:
            drop.add(k)
    S[:, sorted(drop)] = 0.0
    info["removed"] = sorted(drop)
    y = x.copy()
    y[:, gi] = ica.inverse_transform(S)
    return y, info


def blink_peaks(veog: np.ndarray) -> np.ndarray:
    """Sample indices of blinks: peaks of |VEOG - median| above BLINK_Z robust SDs (polarity-free)."""
    a = np.abs(veog - np.median(veog))
    rsd = 1.4826 * np.median(np.abs(veog - np.median(veog)))
    peaks, _ = find_peaks(a, height=BLINK_Z * rsd, distance=int(BLINK_MIN_SEP_S * FS))
    return peaks


def _band(W: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f, p = welch(W, fs=FS, nperseg=2 * FS, noverlap=int(1.5 * FS), axis=1)
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(axis=1) * (f[1] - f[0])  # (n_windows, n_channels)


def artifact_features(x: np.ndarray, good: np.ndarray, starts: np.ndarray) -> pd.DataFrame:
    """Six artifact markers per window. NaN where the required channels are bad."""
    prox = eog_proxies(x, good)
    W = window_array(x, good, starts)
    n = len(starts)
    out = {}
    if "veog" in prox:
        pk = blink_peaks(prox["veog"])
        out["blink_count"] = np.array([((pk >= s) & (pk < s + WIN)).sum() for s in starts], float)
        out["veog_logpow"] = np.log10([prox["veog"][s:s + WIN].var() for s in starts])
    else:
        out["blink_count"] = out["veog_logpow"] = np.full(n, np.nan)
    out["heog_logpow"] = np.log10([prox["heog"][s:s + WIN].var() for s in starts]) if "heog" in prox else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hf = _band(W, 20, 45)
        dl = _band(W, 1, 4)
        fr = [_ix[c] for c in ("AF3", "AF4", "F7", "F8")]
        tp = [_ix[c] for c in ("T7", "T8")]
        out["frontal_hf_logpow"] = np.log10(np.nanmean(hf[:, fr], axis=1))
        out["temporal_hf_logpow"] = np.log10(np.nanmean(hf[:, tp], axis=1))
        out["frontal_delta_logpow"] = np.log10(np.nanmean(dl[:, fr], axis=1))
    return pd.DataFrame(out)


def window_array(x: np.ndarray, good: np.ndarray, starts) -> np.ndarray:
    """(n_windows, 512, 14) windows at the given start samples; bad channels NaN."""
    W = np.stack([x[s:s + WIN] for s in starts]).astype(np.float64)
    W[:, :, ~good] = np.nan
    return W


def paired_bootstrap(subjects: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray | None = None,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """AUC of scores ``a`` (and ``b``, and a - b) with 95% CIs from resampling participants."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    idx = {s: np.flatnonzero(subjects == s) for s in subs}
    out = {"auc_a": roc_auc_score(y, a)}
    if b is not None:
        out["auc_b"] = roc_auc_score(y, b)
        out["diff"] = out["auc_a"] - out["auc_b"]
    va, vb, vd = [], [], []
    for _ in range(n_boot):
        ii = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        va.append(roc_auc_score(y[ii], a[ii]))
        if b is not None:
            vb.append(roc_auc_score(y[ii], b[ii])); vd.append(va[-1] - vb[-1])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["ci_a"] = ci(va)
    if b is not None:
        out["ci_b"], out["ci_diff"] = ci(vb), ci(vd)
    return out
