"""Verify the documented STEW structure against the actual files.

Skipped automatically when the raw dataset is not present (e.g. on CI).
"""

import numpy as np
import pytest

from data_loader import (
    CHANNELS, CONDITIONS, DEFAULT_DATA_DIR, EXPECTED_SAMPLES, N_SUBJECTS,
    load_ratings, load_recording, recording_path,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "ratings.txt").exists(), reason="raw STEW data not available"
)


def test_all_files_present():
    missing = [
        recording_path(s, c).name
        for s in range(1, N_SUBJECTS + 1)
        for c in CONDITIONS
        if not recording_path(s, c).exists()
    ]
    assert missing == []


@pytest.mark.parametrize("subject", [1, 24, 48])
@pytest.mark.parametrize("condition", ["lo", "hi"])
def test_recording_shape_and_finite(subject, condition):
    rec = load_recording(subject, condition)
    assert rec.data.shape == (EXPECTED_SAMPLES, len(CHANNELS))
    assert np.isfinite(rec.data).all()
    assert rec.duration_s == 150.0


def test_ratings_match_documentation():
    r = load_ratings()
    assert list(r.columns) == ["rating_rest", "rating_test"]
    assert sorted(set(range(1, N_SUBJECTS + 1)) - set(r.index)) == [5, 24, 42]
    assert r.loc[1].tolist() == [2, 8]  # example row quoted in the documentation
    assert r.isin(range(1, 10)).all().all()  # documented 1-9 scale
