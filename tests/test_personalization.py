import numpy as np
import pandas as pd

from personalization import baseline_normalize, calibration_mask, evaluation_mask, subject_zscore


def meta_and_X():
    rows, X = [], []
    rng = np.random.default_rng(0)
    for s in (1, 2):
        for c in ("lo", "hi"):
            for start in np.arange(2.0, 144.0, 2.0):
                rows.append({"subject": s, "condition": c, "start_s": start})
                X.append([10 * s + (c == "hi") + rng.normal(0, 1), rng.normal(0, 1)])
    return pd.DataFrame(rows), pd.DataFrame(X, columns=["a", "b"])


def test_calibration_and_evaluation_windows_do_not_overlap():
    meta, _ = meta_and_X()
    cal, ev = calibration_mask(meta), evaluation_mask(meta)
    assert not (cal & ev).any()
    assert (meta.start_s[cal] + 4 <= 60).all() and (meta.start_s[ev] >= 64).all()
    # the latest calibration window ends at least 4 s before the earliest evaluation window starts
    assert meta.start_s[ev].min() - (meta.start_s[cal].max() + 4) >= 4


def test_subject_zscore_removes_subject_offset():
    meta, X = meta_and_X()
    Z = subject_zscore(X, meta.subject.values)
    for s in (1, 2):
        m = meta.subject == s
        assert np.allclose(Z.loc[m].mean(), 0, atol=1e-9)
        assert np.allclose(Z.loc[m].std(ddof=1), 1)


def test_baseline_normalize_centres_rest_calibration():
    meta, X = meta_and_X()
    Z = baseline_normalize(X, meta)
    ref = calibration_mask(meta) & (meta.condition == "lo").values
    for s in (1, 2):
        m = (meta.subject == s).values
        assert np.allclose(Z.loc[m & ref].mean(), 0, atol=1e-9)
        # test windows are shifted up by ~1 unit of feature a
        assert Z.loc[m & (meta.condition == "hi").values, "a"].mean() > 0.5
