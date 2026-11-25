import numpy as np

from data_loader import CHANNELS, FS
from feature_extraction import (BANDS, band_power, extract, feature_groups, spectral_entropy,
                                time_features, window_psd)

N = 4 * FS
t = np.arange(N) / FS
rng = np.random.default_rng(0)


def sines(freq, amp=10.0, n_win=3):
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.repeat(x[None, :, None], n_win, axis=0).repeat(len(CHANNELS), axis=2)


def test_parseval_total_power_matches_variance():
    W = rng.normal(0, 5, (4, N, 2))
    f, p = window_psd(W)
    total = band_power(f, p, 0, FS / 2 + 1)
    assert np.allclose(total, W.var(axis=1), rtol=0.1)


def test_sine_power_lands_in_correct_band():
    W = sines(10.0, amp=10.0)
    df = extract(W)
    assert np.allclose(df["O1_rel_alpha"], 1.0, atol=0.02)
    # power of a sine = A^2/2 = 50
    assert np.allclose(df["O1_abs_alpha"], np.log10(50), atol=0.05)
    assert (df["O1_r_theta_alpha"] < -3).all()


def test_bands_partition_total():
    W = rng.normal(0, 1, (2, N, 3))
    f, p = window_psd(W)
    s = sum(band_power(f, p, lo, hi) for lo, hi in BANDS.values())
    assert np.allclose(s, band_power(f, p, 1, 45))


def test_spectral_entropy_bounds():
    f, p_noise = window_psd(rng.normal(0, 1, (2, N, 1)))
    _, p_sine = window_psd(sines(10.0, n_win=2)[:, :, :1])
    assert (spectral_entropy(f, p_noise) > 0.95).all()
    assert (spectral_entropy(f, p_sine) < 0.3).all()


def test_hjorth_mobility_of_sine():
    # for a sampled sine, var(diff)/var(x) = 2 - 2 cos(w)  ->  mobility = 2 sin(w/2)
    w = 2 * np.pi * 10 / FS
    tf = time_features(sines(10.0)[:, :, :1])
    assert np.allclose(tf["mobility"], 2 * np.sin(w / 2), rtol=1e-2)
    assert np.allclose(tf["complexity"], 1.0, rtol=1e-2)


def test_nan_channel_propagates_only_to_its_features():
    W = rng.normal(0, 1, (2, N, len(CHANNELS)))
    W[:, :, CHANNELS.index("F3")] = np.nan
    df = extract(W)
    f3 = [c for c in df if c.startswith("F3_")]
    assert df[f3].isna().all().all()
    others = [c for c in df if not c.startswith("F3_") and c != "faa_F4F3"]
    assert not df[others].isna().any().any()  # fT_pA uses nanmean over frontal channels


def test_feature_count_and_groups():
    df = extract(rng.normal(0, 1, (1, N, len(CHANNELS))))
    assert df.shape[1] == 14 * 20 + 2
    groups = feature_groups(df.columns)
    assert sum(len(v) for v in groups.values()) == df.shape[1]
