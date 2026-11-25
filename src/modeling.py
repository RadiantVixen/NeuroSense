"""Model definitions and subject-independent nested cross-validation.

Leakage rules enforced here:
* Outer split = GroupKFold over subjects: no subject contributes windows to both
  the training and the test side of any fold.
* Every learned transform (median imputation, standardisation) is a step of the
  sklearn ``Pipeline``, so it is fitted on the training fold only.
* Hyper-parameters are tuned by an inner GroupKFold run *on the training subjects
  of each outer fold*; the outer test subjects are never seen during tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 0


@dataclass
class ModelSpec:
    name: str
    estimator: Pipeline
    grid: dict
    description: str


def _prep():
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def model_specs() -> dict[str, ModelSpec]:
    specs = [
        ModelSpec("chance", Pipeline([("clf", DummyClassifier(strategy="prior"))]), {},
                  "predicts the training-set class prior; AUC = 0.5 by construction"),
        ModelSpec("logreg", Pipeline(_prep() + [("clf", LogisticRegression(max_iter=5000))]),
                  {"clf__C": [1e-3, 1e-2, 1e-1, 1.0]},
                  "L2-regularised logistic regression"),
        ModelSpec("svm_rbf", Pipeline(_prep() + [("clf", SVC(kernel="rbf", gamma="scale"))]),
                  {"clf__C": [0.1, 1.0, 10.0]},
                  "support vector machine, Gaussian (RBF) kernel"),
        ModelSpec("random_forest", Pipeline([("impute", SimpleImputer(strategy="median")),
                                             ("clf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                                                            n_jobs=4, random_state=SEED))]),
                  {"clf__min_samples_leaf": [1, 10, 50]},
                  "random forest, 400 trees"),
        ModelSpec("grad_boost", Pipeline([("clf", HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                                                                 random_state=SEED))]),
                  {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 31]},
                  "histogram gradient boosting (handles NaN natively)"),
    ]
    return {s.name: s for s in specs}


def decision_scores(est, X) -> np.ndarray:
    """Continuous score where larger = more 'test'. Probabilities when available, else SVM margin."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def score_threshold(model_name: str) -> float:
    return 0.0 if model_name == "svm_rbf" else 0.5


def outer_cv(n_splits: int = 8) -> GroupKFold:
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def nested_cv(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
              outer_splits: int = 8, inner_splits: int = 4, n_jobs: int = 8,
              train_mask: np.ndarray | None = None, test_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Out-of-fold scores from subject-independent nested CV.

    The outer split is always computed on all rows (so folds are identical across analyses);
    ``train_mask`` / ``test_mask`` optionally restrict which rows of each side are used.
    """
    rows = []
    for fold, (tr, te) in enumerate(outer_cv(outer_splits).split(X, y, groups)):
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        if test_mask is not None:
            te = te[test_mask[te]]
        est = clone(spec.estimator)
        if spec.grid:
            search = GridSearchCV(est, spec.grid, cv=GroupKFold(n_splits=inner_splits), scoring="roc_auc",
                                  n_jobs=n_jobs, refit=True)
            search.fit(X.iloc[tr], y[tr], groups=groups[tr])
            est, best = search.best_estimator_, search.best_params_
        else:
            est.fit(X.iloc[tr], y[tr]); best = {}
        s = decision_scores(est, X.iloc[te])
        rows.append(pd.DataFrame({"row": te, "fold": fold, "score": s,
                                  "pred": (s > score_threshold(spec.name)).astype(int),
                                  "best_params": [best] * len(te)}))
    return pd.concat(rows).sort_values("row").set_index("row")


def fixed_cv(estimator, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, outer_splits: int = 8,
             threshold: float = 0.5) -> pd.DataFrame:
    """Out-of-fold scores with fixed hyper-parameters (for ablations and permutation tests)."""
    rows = []
    for fold, (tr, te) in enumerate(outer_cv(outer_splits).split(X, y, groups)):
        est = clone(estimator).fit(X.iloc[tr], y[tr])
        s = decision_scores(est, X.iloc[te])
        rows.append(pd.DataFrame({"row": te, "fold": fold, "score": s, "pred": (s > threshold).astype(int)}))
    return pd.concat(rows).sort_values("row").set_index("row")
