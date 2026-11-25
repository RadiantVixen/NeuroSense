import numpy as np
from scipy.signal import welch

from data_loader import FS, Recording
from preprocessing import (PreprocessConfig, _dilate, _flat_run_mask, bad_sample_mask,
                           detect_bad_channels, filter_recording, preprocess, rereference)

T = np.arange(FS * 150) / FS
RNG = np.random.default_rng(0)


def synthetic(n_ch=14):
    """Offset + drift + 10 Hz alpha + 50 Hz line noise + white noise."""
    x = 4200 + 30 * T[:, None] / 150 + 10 * np.sin(2 * np.pi * 10 * T)[:, None] \
        + 20 * np.sin(2 * np.pi * 50 * T)[:, None] + RNG.normal(0, 3, (T.size, n_ch))
    return np.round(x / 0.51) * 0.51  # quantise like the ADC


def band_power(x, lo, hi):
    f, p = welch(x, fs=FS, nperseg=512, axis=0)
    m = (f >= lo) & (f <= hi)
    return p[m].sum(axis=0).mean()


def test_filter_removes_dc_drift_and_line_noise_keeps_alpha():
    raw = synthetic()
    y = filter_recording(raw)
    core = y[FS * 5 : -FS * 5]
    assert abs(core.mean()) < 0.5
    ratio_alpha = band_power(core, 9.5, 10.5) / band_power(raw[FS * 5 : -FS * 5] - raw.mean(0), 9.5, 10.5)
    assert 0.9 < ratio_alpha < 1.1
    assert band_power(core, 49, 51) < 1e-3 * band_power(raw - raw.mean(0), 49, 51)


def test_filter_is_zero_phase():
    s = np.sin(2 * np.pi * 10 * T)[:, None] * np.ones((1, 2))
    y = filter_recording(s * 10, PreprocessConfig(notch_hz=None))
    core = slice(FS * 10, -FS * 10)
    lag = np.argmax(np.correlate(y[core, 0], s[core, 0], "full")) - (s[core, 0].size - 1)
    assert lag == 0


def test_flat_run_and_dilate():
    x = np.arange(100.0)[:, None]
    x[40:80] = 5.0
    m = _flat_run_mask(x, 32)
    assert m[40:80, 0].all() and not m[:40, 0].any() and not m[80:, 0].any()
    d = _dilate(m, 5)
    assert d[35:85, 0].all() and not d[:35, 0].any()


def test_rail_samples_masked():
    raw = synthetic()
    raw[1000:1010, 3] = 0.0
    m = bad_sample_mask(raw)
    assert m[1000:1010, 3].all() and not m[:, 0].any()


def test_bad_channel_detected():
    raw = synthetic()
    raw[:, 5] += RNG.normal(0, 200, T.size)
    x = filter_recording(raw)
    bad = detect_bad_channels(x, np.zeros_like(x, bool))
    assert bad.tolist() == [i == 5 for i in range(14)]


def test_common_average_reference_zero_mean_across_good_channels():
    x = RNG.normal(size=(1000, 4))
    good = np.array([True, True, False, True])
    y = rereference(x, good, "average")
    assert np.allclose(y[:, good].mean(axis=1), 0)


def test_preprocess_windows_and_rejection():
    raw = synthetic()
    raw[FS * 60 : FS * 61, 2] += 2000  # gross artifact -> windows covering 60-61 s rejected
    p = preprocess(Recording(1, "hi", raw))
    w = p.windows
    assert (w.start_s.diff().dropna() == 2.0).all()
    assert w.start_s.min() == 2.0 and w.start_s.max() + 4 <= 148
    hit = (w.start_s < 61) & (w.start_s + 4 > 60)
    assert w.loc[hit, "rejected"].all()
    assert not w.loc[(w.start_s + 4 < 58) | (w.start_s > 63), "rejected"].any()
    assert p.window_data().shape == ((~w.rejected).sum(), 512, 14)
