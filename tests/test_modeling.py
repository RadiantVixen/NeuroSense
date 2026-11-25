import numpy as np
import pandas as pd

from evaluation import classification_metrics, recording_level
from modeling import fixed_cv, model_specs, nested_cv, outer_cv


def toy(n_sub=16, n_win=10, effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for s in range(n_sub):
        offset = rng.normal(0, 3, 5)  # large subject offsets, like real EEG
        for lab in (0, 1):
            for _ in range(n_win):
                X.append(offset + rng.normal(0, 1, 5) + effect * lab * np.array([1, 0, 0, 0, 0]))
                rows.append({"subject": s, "condition": "hi" if lab else "lo", "label": lab})
    return pd.DataFrame(rows), pd.DataFrame(X, columns=list("abcde"))


def test_outer_folds_never_share_subjects():
    meta, X = toy()
    for tr, te in outer_cv(4).split(X, meta.label, meta.subject):
        assert not set(meta.subject.iloc[tr]) & set(meta.subject.iloc[te])


def test_nested_cv_returns_one_prediction_per_window_and_learns_signal():
    meta, X = toy(effect=3.0)
    spec = model_specs()["logreg"]
    oof = nested_cv(spec, X, meta.label.values, meta.subject.values, outer_splits=4, inner_splits=3, n_jobs=1)
    assert len(oof) == len(X) and oof.index.is_unique
    m = classification_metrics(meta.label, oof.pred, oof.score)
    assert m["roc_auc"] > 0.75


def test_no_signal_gives_chance():
    meta, X = toy(effect=0.0, n_sub=24)
    oof = fixed_cv(model_specs()["logreg"].estimator.set_params(clf__C=0.1), X, meta.label.values, meta.subject.values, 4)
    assert 0.35 < classification_metrics(meta.label, oof.pred, oof.score)["roc_auc"] < 0.65


def test_recording_level_aggregation():
    meta = pd.DataFrame({"subject": [1, 1, 1, 1], "condition": ["lo", "lo", "hi", "hi"], "label": [0, 0, 1, 1]})
    oof = pd.DataFrame({"score": [0.2, 0.6, 0.7, 0.9]})
    r = recording_level(meta, oof, 0.5)
    r = r.set_index("condition")
    assert np.isclose(r.loc["lo", "score"], 0.4) and np.isclose(r.loc["hi", "score"], 0.8)
    assert r.loc["lo", "pred"] == 0 and r.loc["hi", "pred"] == 1
    assert r.n_windows.tolist() == [2, 2]
