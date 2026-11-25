"""Personalisation strategies supported by STEW (one rest + one test recording per subject).

Common protocol (so every approach is scored on identical data):

* **calibration windows**: windows lying entirely within the first ``CAL_END_S`` seconds
  of a recording (start + 4 s <= 60 s);
* **evaluation windows**: windows starting at or after ``EVAL_START_S`` (>= 64 s), so
  calibration and evaluation windows never overlap in time (4 s gap).

Approaches
----------
A  subject-independent         train on other subjects; raw features
B1 unsupervised subject z-score  every subject's features z-scored with *all* of its own
                                 windows (labels unused; transductive - needs unlabeled data
                                 from both states of the new user)
B2 rest-baseline normalisation   features z-scored w.r.t. the subject's rest calibration
                                 windows (a resting baseline recorded at headset set-up)
C  subject-dependent             train only on the target's calibration windows (both states)
D  hybrid                        other subjects + target's calibration windows (up-weighted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_S = 4.0
CAL_END_S = 60.0
EVAL_START_S = 64.0


def calibration_mask(meta: pd.DataFrame) -> np.ndarray:
    return (meta.start_s + WINDOW_S <= CAL_END_S).values


def evaluation_mask(meta: pd.DataFrame) -> np.ndarray:
    return (meta.start_s >= EVAL_START_S).values


def _zscore(X: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    mu = ref.mean()
    sd = ref.std(ddof=1).replace(0, np.nan)
    return (X - mu) / sd


def subject_zscore(X: pd.DataFrame, subjects: np.ndarray) -> pd.DataFrame:
    """B1: per-subject standardisation using all of that subject's windows (label-free)."""
    out = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        out.loc[m] = _zscore(X.loc[m], X.loc[m]).values
    return out


def baseline_normalize(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """B2: per-subject standardisation w.r.t. the rest-condition calibration windows."""
    cal_rest = calibration_mask(meta) & (meta.condition == "lo").values
    out = X.copy()
    for s in meta.subject.unique():
        m = (meta.subject == s).values
        ref = X.loc[m & cal_rest]
        out.loc[m] = _zscore(X.loc[m], ref).values if len(ref) >= 5 else np.nan
    return out
