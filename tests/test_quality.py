import numpy as np

from quality import extreme_fraction, highpass, line_noise_ratio, max_flat_run, robust_z

FS = 128


def test_max_flat_run_counts_repeated_samples():
    x = np.array([1, 2, 2, 2, 3, 4, 4], dtype=float)[:, None]
    assert max_flat_run(x)[0] == 2  # 3 identical samples -> 2 zero diffs


def test_max_flat_run_no_repeats():
    assert max_flat_run(np.arange(10.0)[:, None])[0] == 0


def test_robust_z_ignores_outliers_in_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (10_000, 1))
    x[:50] = 1e6
    z = robust_z(x)
    # scale estimate stays ~1 despite huge outliers
    assert np.median(z[50:]) < 1.0
    assert extreme_fraction(x, k=6)[0] >= 0.005


def test_highpass_removes_dc_and_drift():
    t = np.arange(FS * 60) / FS
    x = (4000 + 50 * t + 10 * np.sin(2 * np.pi * 10 * t))[:, None]
    y = highpass(x, FS, 1.0)
    core = y[FS * 5 : -FS * 5, 0]
    assert abs(core.mean()) < 0.5
    assert np.isclose(core.std(), 10 / np.sqrt(2), rtol=0.05)


def test_line_noise_ratio_detects_peak():
    rng = np.random.default_rng(1)
    t = np.arange(FS * 60) / FS
    noise = rng.normal(0, 1, (t.size, 3))
    assert line_noise_ratio(noise, FS, 50) < 2
    assert line_noise_ratio(noise + 5 * np.sin(2 * np.pi * 50 * t)[:, None], FS, 50) > 50
