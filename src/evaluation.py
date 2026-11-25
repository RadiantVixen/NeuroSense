"""Evaluation metrics at window, recording and subject level."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)


def classification_metrics(y, pred, score) -> dict:
    """Positive class = 1 = test (multitasking)."""
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": recall_score(1 - np.asarray(y), 1 - np.asarray(pred), zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),  # rows = true [rest, test]
        "n": int(len(y)),
    }


def recording_level(meta: pd.DataFrame, oof: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate window scores to one score per recording (mean), then threshold."""
    d = meta[["subject", "condition", "label"]].join(oof[["score"]])
    r = d.groupby(["subject", "condition"]).agg(label=("label", "first"), score=("score", "mean"),
                                                n_windows=("score", "size")).reset_index()
    r["pred"] = (r.score > threshold).astype(int)
    return r


def per_subject(meta: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Window-level balanced accuracy and AUC within each held-out subject."""
    d = meta[["subject", "label"]].join(oof[["score", "pred"]])
    out = []
    for s, g in d.groupby("subject"):
        out.append({"subject": s, "balanced_accuracy": balanced_accuracy_score(g.label, g.pred),
                    "roc_auc": roc_auc_score(g.label, g.score) if g.label.nunique() == 2 else np.nan,
                    "n_windows": len(g)})
    return pd.DataFrame(out)


def regression_metrics(y, yhat) -> dict:
    return {"mae": mean_absolute_error(y, yhat), "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "r2": r2_score(y, yhat), "n": int(len(y))}


def bootstrap_ci_by_subject(meta: pd.DataFrame, oof: pd.DataFrame, metric, n_boot: int = 1000,
                            seed: int = 0) -> tuple[float, float]:
    """95 % CI resampling *subjects* (the independent units), not windows."""
    rng = np.random.default_rng(seed)
    d = meta[["subject", "label"]].join(oof)
    subs = d.subject.unique()
    by = {s: g for s, g in d.groupby("subject")}
    vals = []
    for _ in range(n_boot):
        g = pd.concat([by[s] for s in rng.choice(subs, len(subs), replace=True)])
        vals.append(metric(g))
    return tuple(np.percentile(vals, [2.5, 97.5]))
