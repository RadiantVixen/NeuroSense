import numpy as np

from confounds import artifact_features, blink_peaks, ocular_ica, paired_bootstrap, window_array
from data_loader import CHANNELS, FS

rng = np.random.default_rng(0)
T = np.arange(FS * 60) / FS
IX = {c: i for i, c in enumerate(CHANNELS)}


def blink_train(times, amp=150.0):
    s = np.zeros_like(T)
    for t0 in times:
        s += amp * np.exp(-0.5 * ((T - t0) / 0.08) ** 2)
    return s


def synthetic_with_blinks(times):
    """Independent 'brain' sources + one blink source projecting mostly to frontal channels."""
    x = rng.normal(0, 5, (T.size, len(CHANNELS)))
    x += 8 * np.sin(2 * np.pi * 10 * T)[:, None] * np.linspace(0.2, 1, len(CHANNELS))[None, :]
    gain = np.zeros(len(CHANNELS))
    for c, g in {"AF3": 1.0, "AF4": 1.0, "F7": 0.6, "F8": 0.6, "F3": 0.5, "F4": 0.5}.items():
        gain[IX[c]] = g
    return x + blink_train(times)[:, None] * gain[None, :], gain


def test_blink_peaks_detects_both_polarities():
    times = [5, 12, 20, 33, 47]
    for sign in (1, -1):
        veog = sign * blink_train(times) + rng.normal(0, 3, T.size)
        pk = blink_peaks(veog)
        assert len(pk) == len(times)
        assert np.allclose(np.sort(pk) / FS, times, atol=0.05)


def test_ocular_ica_removes_blink_component_and_keeps_alpha():
    times = np.arange(3, 58, 4.0)
    x, gain = synthetic_with_blinks(times)
    good = np.ones(len(CHANNELS), bool)
    y, info = ocular_ica(x, good, np.ones(T.size, bool))
    assert len(info["removed"]) >= 1 and info["r"]["veog"] > 0.9
    af3 = IX["AF3"]
    near = np.abs(T[:, None] - times[None, :]).min(axis=1) < 0.15
    assert np.abs(y[near, af3]).max() < 0.3 * np.abs(x[near, af3]).max()  # blinks largely gone
    o1 = IX["O1"]  # a posterior channel with no blink gain is barely changed
    assert np.corrcoef(x[:, o1], y[:, o1])[0, 1] > 0.95


def test_ocular_ica_removes_nothing_without_eye_signal():
    x = rng.normal(0, 5, (T.size, len(CHANNELS)))
    y, info = ocular_ica(x, np.ones(len(CHANNELS), bool), np.ones(T.size, bool))
    assert info["removed"] == [] and np.allclose(x, y, atol=1e-6)


def test_artifact_features_count_blinks_per_window():
    times = [1.0, 2.0, 9.0]  # two blinks in window [0, 4) s, one in [8, 12) s
    x, _ = synthetic_with_blinks(times)
    good = np.ones(len(CHANNELS), bool)
    df = artifact_features(x, good, np.array([0, 4 * FS, 8 * FS]))
    assert df.blink_count.tolist() == [2, 0, 1]
    assert df.shape[1] == 6 and not df.isna().any().any()


def test_window_array_sets_bad_channels_nan():
    x = rng.normal(size=(FS * 20, len(CHANNELS)))
    good = np.ones(len(CHANNELS), bool); good[3] = False
    W = window_array(x, good, [0, 512])
    assert W.shape == (2, 512, 14) and np.isnan(W[:, :, 3]).all() and not np.isnan(W[:, :, 2]).any()


def test_paired_bootstrap_difference_ci():
    subj = np.repeat(np.arange(30), 20)
    y = np.tile(np.r_[np.zeros(10), np.ones(10)], 30)
    good_scores = y + rng.normal(0, 0.5, y.size)
    noise = rng.normal(0, 1, y.size)
    out = paired_bootstrap(subj, y, good_scores, noise, n_boot=300)
    assert out["auc_a"] > 0.8 and 0.35 < out["auc_b"] < 0.65
    assert out["ci_diff"][0] > 0  # difference clearly positive
    assert out["ci_a"][0] < out["auc_a"] < out["ci_a"][1]
