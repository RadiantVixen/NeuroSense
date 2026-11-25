"""Stage 7 - export validated results to JSON for the static website.

Every number exported here is read from the result files written by notebooks
01-06 (or recomputed deterministically from the raw data for signal examples).
Nothing is invented. Run from the repository root:

    python src/export_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, freqz, iirnotch, sosfreqz, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import CHANNELS, CONDITIONS, FS, PROJECT_ROOT, load_recording, load_ratings  # noqa: E402
from feature_extraction import BANDS  # noqa: E402
from preprocessing import PreprocessConfig, filter_recording, preprocess  # noqa: E402

RES = PROJECT_ROOT / "results"
SITE_DATA = PROJECT_ROOT / "website" / "data"
CFG = PreprocessConfig()


def r(x, nd=6):
    """Round floats recursively for compact JSON.

    6 decimals: the site rounds again to 2-3 decimals, and storing only 4 caused
    double-rounding errors (e.g. 0.66854 -> 0.6685 -> 0.668 instead of 0.669).
    """
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def load_json(p: Path):
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- metrics.json
def build_metrics() -> dict:
    ev = load_json(RES / "05_evaluation" / "summary.json")
    tr = load_json(RES / "04_modeling" / "summary.json")
    pe = load_json(RES / "06_personalization" / "summary.json")
    pre = load_json(RES / "02_preprocessing" / "summary.json")
    feat = load_json(RES / "03_features" / "summary.json")
    expl = load_json(RES / "01_exploration" / "summary.json")
    imp = {}
    for f in sorted((RES / "05_evaluation").glob("importance_*_*.csv")):
        model, kind = f.stem.removeprefix("importance_").rsplit("_", 1)  # e.g. grad_boost / group
        d = pd.read_csv(f, index_col=0)
        imp.setdefault(model, {})[kind] = [{"name": i, "mean": v["mean"], "sd": v["sd"]} for i, v in d.iterrows()]
    ps = pd.read_csv(RES / "05_evaluation" / "per_subject_auc.csv", index_col=0)
    per_subject = {m: {"subject": ps.index.tolist(), "auc": ps[m].tolist()} for m in ps.columns}
    pers = pd.read_csv(RES / "06_personalization" / "per_subject_auc.csv", index_col=0)
    pers_subject = {k: pers[k].dropna().tolist() for k in pers.columns}
    coef = pd.read_csv(RES / "05_evaluation" / "logreg_coef_mean.csv", index_col=0)["coef"]
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(20)
    return r({
        "dataset": expl["dataset"] | {"ratings": expl["ratings"]},
        "preprocessing": {k: pre[k] for k in ("config", "rejection_rate", "kept_windows", "n_windows_total", "windows_per_recording")}
            | {"n_bad_channels": len(pre["bad_channels"]), "n_channel_recordings": 96 * len(CHANNELS)},
        "features": feat,
        "models": tr["models"],
        "cv": tr["cv"],
        "classification": ev["metrics"],
        "per_subject_tests": ev["per_subject_tests"],
        "permutation_test": ev["permutation_test"],
        "ablation": ev["ablation"],
        "artifact_score_correlation": ev["artifact_score_correlation"],
        "best_model": ev["best_model"],
        "feature_importance": imp,
        "logreg_top_coefficients": [{"feature": k, "coef": v} for k, v in top.items()],
        "rating_regression": tr["rating_regression"],
        "classifier_score_vs_rating": tr["classifier_score_vs_rating_spearman"],
        "personalization": pe,
        "confounds": load_json(RES / "07_confounds" / "summary.json"),
        "per_subject_auc": per_subject,
        "personalization_per_subject": pers_subject,
    })


# --------------------------------------------------------------------------- predictions.json
def build_predictions(best_model: str) -> dict:
    rec = pd.read_csv(RES / "05_evaluation" / "recording_predictions.csv")
    rec = rec[rec.model == best_model]
    per_sub = pd.read_csv(RES / "05_evaluation" / "per_subject_auc.csv", index_col=0)
    pers = pd.read_csv(RES / "06_personalization" / "per_subject_auc.csv", index_col=0)
    ratings = load_ratings()
    oof = pd.read_csv(RES / "04_modeling" / "oof_predictions.csv.gz")
    oof = oof[oof.model == best_model]
    subjects = []
    for s in sorted(rec.subject.unique()):
        d = rec[rec.subject == s].set_index("condition")
        w = oof[oof.subject == s]
        subjects.append({
            "subject": int(s),
            "rest_score": d.loc["lo", "score"], "test_score": d.loc["hi", "score"],
            "rest_pred": int(d.loc["lo", "pred"]), "test_pred": int(d.loc["hi", "pred"]),
            "rest_windows": int(d.loc["lo", "n_windows"]), "test_windows": int(d.loc["hi", "n_windows"]),
            "auc": per_sub.loc[s, best_model] if s in per_sub.index else None,
            "rating_rest": int(ratings.loc[s, "rating_rest"]) if s in ratings.index else None,
            "rating_test": int(ratings.loc[s, "rating_test"]) if s in ratings.index else None,
            "personalization_auc": {k: pers.loc[s, k] for k in pers.columns} if s in pers.index else None,
            "timeline": {c: {"t": w[w.condition == c].start_s.add(2).tolist(), "score": w[w.condition == c].score.tolist()}
                         for c in CONDITIONS},
        })
    return r({"model": best_model, "threshold": 0.0 if best_model == "svm_rbf" else 0.5,
              "score_kind": "SVM margin" if best_model == "svm_rbf" else "probability of 'test'",
              "note": "Out-of-fold predictions: every score comes from a model that never saw this subject.",
              "subjects": subjects}, 4)


# --------------------------------------------------------------------------- visualizations.json + eeg_examples.json
def pick_example_subjects(n_clean: int = 3) -> list[int]:
    w = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "windows.csv")
    rate = w.groupby("subject").rejected.mean().sort_values()
    bad = set(pd.read_csv(PROJECT_ROOT / "data" / "processed" / "bad_channels.csv").subject)
    clean = [int(s) for s in rate.index if s not in bad][:n_clean]  # no excluded channels in either recording
    return clean + [int(rate.index[-1])]  # plus the subject with the most rejected windows


def build_eeg_examples(subjects: list[int], seconds: float = 10.0, start_s: float = 60.0) -> dict:
    out = {"fs": FS, "channels": list(CHANNELS), "start_s": start_s, "seconds": seconds,
           "note": "Recorded STEW data (offline playback), not live acquisition. 'raw' has only the per-channel median removed.",
           "recordings": {}}
    a, b = int(start_s * FS), int((start_s + seconds) * FS)
    for s in subjects:
        for c in CONDITIONS:
            rec = load_recording(s, c)
            raw = rec.data - np.median(rec.data, axis=0)
            p = preprocess(rec, CFG)
            f, pr = welch(raw, fs=FS, nperseg=4 * FS, axis=0)
            _, pf = welch(p.x.astype(float), fs=FS, nperseg=4 * FS, axis=0)
            m = f <= 64
            out["recordings"][f"{s}_{c}"] = {
                "subject": s, "condition": c, "label": CONDITIONS[c],
                "rejected_fraction": float(p.windows.rejected.mean()),
                "bad_channels": [CHANNELS[i] for i in np.flatnonzero(p.bad_channels)],
                "raw": [np.round(raw[a:b, i]).astype(int).tolist() for i in range(len(CHANNELS))],
                "filtered": [np.round(p.x[a:b, i].astype(float)).astype(int).tolist() for i in range(len(CHANNELS))],
                "psd_freqs": np.round(f[m], 2).tolist(),
                "psd_raw": [np.round(np.log10(pr[m, i] + 1e-12), 3).tolist() for i in range(len(CHANNELS))],
                "psd_filtered": [np.round(np.log10(pf[m, i] + 1e-12), 3).tolist() for i in range(len(CHANNELS))],
            }
    return out


def build_visualizations() -> dict:
    # group-average PSD (filtered data, all subjects), mean and 95% CI of log10 PSD across subjects
    logp = {c: [] for c in CONDITIONS}
    for s in range(1, 49):
        for c in CONDITIONS:
            x = filter_recording(load_recording(s, c).data, CFG)
            f, p = welch(x, fs=FS, nperseg=4 * FS, axis=0, average="median")
            logp[c].append(np.log10(p.mean(axis=1)))
    m = f <= 50
    group_psd = {"freqs": f[m].tolist()}
    for c in CONDITIONS:
        a = np.array(logp[c])[:, m]
        se = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])
        group_psd[c] = {"mean": a.mean(0).tolist(), "lo": (a.mean(0) - 1.96 * se).tolist(), "hi": (a.mean(0) + 1.96 * se).tolist()}

    # filter frequency response (zero-phase => squared magnitude)
    sos = butter(CFG.filter_order, [CFG.hp_hz, CFG.lp_hz], btype="bandpass", fs=FS, output="sos")
    ff, h = sosfreqz(sos, worN=1024, fs=FS)
    bn, an = iirnotch(CFG.notch_hz, CFG.notch_q, fs=FS)
    _, hn = freqz(bn, an, worN=1024, fs=FS)
    gain = 10 * np.log10(np.maximum((np.abs(h) * np.abs(hn)) ** 2, 1e-12))

    contrast = pd.read_csv(RES / "01_exploration" / "band_power_contrast.csv")
    eta = pd.read_csv(RES / "03_features" / "variance_decomposition.csv", index_col=0)
    eta_group = eta.groupby("group")[["subject", "condition"]].median().sort_values("condition", ascending=False)

    feats = pd.read_csv(PROJECT_ROOT / "data" / "features" / "features.csv.gz")
    show = ["fT_pA", "O1_rel_alpha", "F3_rel_theta", "AF3_r_theta_alpha", "F8_abs_theta", "T7_rel_gamma"]
    dists = {}
    for col in show:
        v = feats[col].dropna()
        edges = np.linspace(*np.percentile(v, [1, 99]), 41)
        dists[col] = {"edges": edges.tolist()} | {
            c: (np.histogram(feats.loc[feats.condition == c, col].dropna(), bins=edges)[0] /
                feats.loc[feats.condition == c, col].notna().sum()).tolist() for c in CONDITIONS}

    pre = load_json(RES / "02_preprocessing" / "summary.json")
    qc = pd.read_csv(RES / "01_exploration" / "quality_recording_level.csv")
    ratings = load_ratings().reset_index()
    return r({
        "bands": BANDS,
        "group_psd": group_psd,
        "filter_response": {"freqs": ff.tolist(), "gain_db": gain.tolist(), "hp_hz": CFG.hp_hz, "lp_hz": CFG.lp_hz, "notch_hz": CFG.notch_hz},
        "band_contrast": contrast[["band", "channel", "median_log_ratio", "frac_subj_increase", "p_fdr", "significant"]].to_dict(orient="records"),
        "variance_decomposition": {"groups": eta_group.index.tolist(), "subject": eta_group.subject.tolist(), "condition": eta_group.condition.tolist()},
        "feature_distributions": dists,
        "rejection_sweep": pre["threshold_sweep"],
        "line_noise": qc[["subject", "condition", "line50", "line60"]].to_dict(orient="records"),
        "ratings": ratings.to_dict(orient="records"),
    }, 4)


def main():
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    preds = build_predictions(metrics["best_model"])
    vis = build_visualizations()
    examples = build_eeg_examples(pick_example_subjects())
    for name, obj in [("metrics", metrics), ("predictions", preds), ("visualizations", vis)]:
        txt = json.dumps(obj, separators=(",", ":"))
        (RES / f"{name}.json").write_text(txt)
        (SITE_DATA / f"{name}.json").write_text(txt)
    (SITE_DATA / "eeg_examples.json").write_text(json.dumps(r(examples, 3), separators=(",", ":")))
    import site_render  # write the headline numbers into index.html as static HTML
    site_render.main([])
    for p in sorted(SITE_DATA.iterdir()):
        print(f"{p.relative_to(PROJECT_ROOT)}: {p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
