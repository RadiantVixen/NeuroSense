"""Loading and validation utilities for the STEW dataset.

Facts below are taken from the official dataset page
(IEEE DataPort, DOI 10.21227/44r8-ya50) and verified against the files
in ``tests/test_data_loader.py``:

* 48 subjects, files ``subXX_lo.txt`` (rest) and ``subXX_hi.txt``
  (SIMKAP multitasking test).
* Each file: rows = samples, columns = 14 Emotiv EPOC channels in the
  order given by :data:`CHANNELS`. 128 Hz, 2.5 min -> 19 200 samples.
* ``ratings.txt``: CSV ``subject, rating_rest, rating_test`` on a 1-9
  scale. Ratings for subjects 5, 24 and 42 are unavailable.

The documentation does not state the physical unit or the reference
electrode. Emotiv EPOC raw values are commonly reported in microvolts
with a ~4200 DC offset, but that is an assumption we treat as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "STEW"

FS = 128  # Hz, from dataset documentation
N_SUBJECTS = 48
CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
CONDITIONS = {"lo": "rest", "hi": "test"}  # documented meaning of suffixes
EXPECTED_SAMPLES = int(2.5 * 60 * FS)  # 19 200


@dataclass(frozen=True)
class Recording:
    subject: int
    condition: str  # "lo" (rest) or "hi" (multitasking test)
    data: np.ndarray  # shape (n_samples, n_channels), float64

    @property
    def duration_s(self) -> float:
        return self.data.shape[0] / FS


def recording_path(subject: int, condition: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {list(CONDITIONS)}, got {condition!r}")
    return Path(data_dir) / f"sub{subject:02d}_{condition}.txt"


def load_recording(subject: int, condition: str, data_dir: Path = DEFAULT_DATA_DIR) -> Recording:
    """Load one whitespace-delimited recording as a (samples, channels) array."""
    path = recording_path(subject, condition, data_dir)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f"{path.name}: expected {len(CHANNELS)} columns, got shape {data.shape}")
    return Recording(subject=subject, condition=condition, data=data)


def load_all(data_dir: Path = DEFAULT_DATA_DIR) -> dict[tuple[int, str], Recording]:
    """Load every recording, keyed by ``(subject, condition)``."""
    return {
        (s, c): load_recording(s, c, data_dir)
        for s in range(1, N_SUBJECTS + 1)
        for c in CONDITIONS
    }


def load_ratings(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Return subjective ratings indexed by subject: columns ``rating_rest``, ``rating_test``."""
    df = pd.read_csv(
        Path(data_dir) / "ratings.txt",
        header=None,
        names=["subject", "rating_rest", "rating_test"],
        skipinitialspace=True,
    )
    return df.set_index("subject").sort_index()


def to_dataframe(rec: Recording) -> pd.DataFrame:
    """Recording as a DataFrame with a time index in seconds."""
    t = np.arange(rec.data.shape[0]) / FS
    return pd.DataFrame(rec.data, columns=CHANNELS, index=pd.Index(t, name="time_s"))
