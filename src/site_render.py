"""Write key results into website/index.html as static HTML.

The interactive charts need JavaScript, but the headline numbers, the metrics
table and the findings must be readable without it (crawlers, link previews,
text browsers). This module fills them in from website/data/metrics.json.

Two kinds of slots in index.html:
* ``<x data-bind="key">...</x>``          inner text replaced by ``values[key]``
* ``<!-- BEGIN:name -->...<!-- END:name -->``  content replaced by ``blocks[name]``

``render`` is a pure function; ``python src/site_render.py`` re-renders the page
from the committed JSON (no raw data needed) and ``--check`` exits non-zero if the
committed page is out of date.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "website" / "index.html"
METRICS = ROOT / "website" / "data" / "metrics.json"

MODEL_LABEL = {
    "chance": "Chance (training prior)", "classic_index": "Frontal θ / parietal α index",
    "logreg": "Logistic regression", "svm_rbf": "SVM (RBF kernel)",
    "random_forest": "Random forest", "grad_boost": "Gradient boosting",
}
MODEL_ORDER = ["chance", "classic_index", "logreg", "svm_rbf", "random_forest", "grad_boost"]
LEARNED = ["logreg", "svm_rbf", "random_forest", "grad_boost"]


class MissingValue(ValueError):
    pass


def f(x, d=3) -> str:
    if x is None:
        raise MissingValue("a required value is null")
    return f"{x:.{d}f}"


def pct(x, d=1) -> str:
    if x is None:
        raise MissingValue("a required value is null")
    return f"{100 * x:.{d}f}%"


def bind_values(m: dict) -> dict[str, str]:
    best = m["best_model"]
    cls = m["classification"][best]
    w, rec = cls["window"], cls["recording"]
    rt = m["dataset"]["ratings"]
    return {
        "hero.recAcc": pct(rec["accuracy"]),
        "hero.recAccLabel": f"recording-level accuracy on held-out participants ({MODEL_LABEL[best].lower()})",
        "hero.auc": f(w["roc_auc"], 2),
        "hero.aucLabel": f"window-level ROC-AUC (95% CI {f(w['auc_ci'][0], 2)}–{f(w['auc_ci'][1], 2)})",
        "about.nRated": str(rt["n_subjects_rated"]),
        "about.restMed": f(rt["rest"]["50%"], 0),
        "about.testMed": f(rt["test"]["50%"], 0),
        "proc.badCh": f"{m['preprocessing']['n_bad_channels']} of {m['preprocessing']['n_channel_recordings']}",
        "proc.winPerRec": str(m["preprocessing"]["windows_per_recording"]),
        "methods.iaf": (f"{f(m['features']['iaf_rest_hz']['min'], 1)}–{f(m['features']['iaf_rest_hz']['max'], 1)} Hz "
                        f"in the {m['features']['iaf_rest_hz']['n_detected']} participants where one was detectable"),
    }


def metrics_table(m: dict) -> str:
    cl = m["classification"]
    rows = []
    for k in MODEL_ORDER:
        if k not in cl:
            continue
        w, r = cl[k]["window"], cl[k]["recording"]
        cls = ' class="best"' if k == m["best_model"] else ""
        rows.append(
            f"<tr{cls}><td>{MODEL_LABEL[k]}</td><td>{f(w['accuracy'])}</td><td>{f(w['balanced_accuracy'])}</td>"
            f"<td>{f(w['precision'])}</td><td>{f(w['recall'])}</td><td>{f(w['specificity'])}</td><td>{f(w['f1'])}</td>"
            f"<td>{f(w['roc_auc'])} <span class=\"ci\">[{f(w['auc_ci'][0], 2)}, {f(w['auc_ci'][1], 2)}]</span></td>"
            f"<td>{f(r['accuracy'])}</td><td>{f(r['roc_auc'])}</td></tr>")
    n_win = sum(map(sum, cl[m["best_model"]]["window"]["cm"]))
    return (
        "<table aria-label=\"Model metrics\"><thead><tr><th>Model</th><th>Accuracy</th><th>Bal. acc.</th><th>Precision</th>"
        "<th>Recall</th><th>Specificity</th><th>F1</th><th>ROC-AUC [95% CI]</th><th>Rec. acc.</th><th>Rec. AUC</th></tr></thead><tbody>"
        + "".join(rows)
        + f"</tbody><caption>Window columns: {n_win} held-out 4 s windows. Recording columns: 96 recordings, window scores averaged. "
          "Positive class = multitasking test. CIs resample participants (1000 bootstrap draws).</caption></table>")


SECTION_LABEL = {
    "reference": "Reference", "a. channels": "a · Channel subsets", "b. bands": "b · Frequency bands removed (re-filtered)",
    "c. ocular ICA": "c · Eye components removed with ICA", "d. artifact-only": "d · Artifact markers only",
    "e. time split": "e · First half → second half",
}


def sgn(v, d=3) -> str:
    """Signed number without a misleading '-0.000'."""
    return f"{0.0:.{d}f}" if round(v, d) == 0 else f"{v:+.{d}f}"


def _ci(lo, hi, d=3, signed=False) -> str:
    fmt = (lambda v: sgn(v, d)) if signed else (lambda v: f"{v:.{d}f}")
    return f'<span class="ci">[{fmt(lo)}, {fmt(hi)}]</span>'


def confound_table(m: dict) -> str:
    rows, last, notes = [], None, []
    for r in m["confounds"]["table"]:
        if r.get("note"):
            notes.append(f"<li><strong>{html.escape(r['analysis'].split(' ')[0])}</strong>: {html.escape(r['note'])}</li>")
        if r["section"] != last:
            rows.append(f'<tr><td class="section-head" colspan="6">{html.escape(SECTION_LABEL.get(r["section"], r["section"]))}</td></tr>')
            last = r["section"]
        auc = f"{f(r['auc'])} {_ci(r['auc_lo'], r['auc_hi'])}" if r["auc"] is not None else "—"
        delta = (f"{sgn(r['delta_auc'])} {_ci(r['delta_lo'], r['delta_hi'], signed=True)}"
                 if r["delta_auc"] is not None and r["section"] != "reference" else "—")
        rec = f(r["rec_acc"]) if r["rec_acc"] is not None else "—"
        rows.append(f"<tr><td>{html.escape(r['analysis'])}</td><td>{r['n_features']}</td><td>{auc}</td><td>{delta}</td>"
                    f"<td>{rec}</td><td>{f(r['subject_auc_median'])}</td></tr>")
    return ('<table aria-label="Artifact analyses"><thead><tr><th>Analysis</th><th>Features</th><th>Window AUC [95% CI]</th>'
            '<th>Δ AUC vs. reference [95% CI]</th><th>Rec. acc.</th><th>Median AUC per person*</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>"
            '<p class="table-note">Logistic regression, nested participant-grouped CV, same windows and outer folds as the main model unless noted. '
            "CIs: participant bootstrap, 1000 draws; Δ CIs are paired. A dash means the quantity does not apply to that design. "
            "*For e3, median AUC per recording.</p>"
            f'<ul class="table-note">{"".join(notes)}</ul>')


LOCATION = {
    "AF3": "left forehead, near the eye", "AF4": "right forehead, near the eye", "F7": "left frontal, beside the eye",
    "F8": "right frontal, beside the eye", "F3": "left frontal", "F4": "right frontal", "FC5": "left fronto-central",
    "FC6": "right fronto-central", "T7": "left temple", "T8": "right temple", "P7": "left parietal", "P8": "right parietal",
    "O1": "left occipital", "O2": "right occipital",
}


class ClaimNoLongerHolds(AssertionError):
    """Raised when re-run results contradict the wording of a finding; edit the text, don't silence this."""


def check(cond: bool, claim: str):
    if not cond:
        raise ClaimNoLongerHolds(claim)


def _row(m, prefix):
    rows = [r for r in m["confounds"]["table"] if r["analysis"].startswith(prefix)]
    if len(rows) != 1:
        raise KeyError(f"expected one confound row starting with {prefix!r}, found {len(rows)}")
    return rows[0]


def _n(x, d=3):
    return f'<span class="num">{f(x, d)}</span>'


def _auc_ci(r):
    return f"{_n(r['auc'])} (95% CI {f(r['auc_lo'], 2)}–{f(r['auc_hi'], 2)})"


def findings(m: dict) -> list[dict]:
    cl, cf = m["classification"], m["confounds"]
    lr, idx = cl["logreg"], cl["classic_index"]
    aucs = {k: cl[k]["window"]["roc_auc"] for k in LEARNED}
    holm = {t["model"]: t.get("p_vs_logreg_holm") for t in m["per_subject_tests"]}
    eta = m["features"]["eta2_median"]
    ch = m["feature_importance"]["logreg"]["channel"]
    grp = m["feature_importance"]["logreg"]["group"]
    reg = next(r for r in m["rating_regression"] if r["model"] == "ridge")
    base = next(r for r in m["rating_regression"] if r["model"] == "mean_baseline")
    sr = m["classifier_score_vs_rating"]
    ref, post, noeye, ica = _row(m, "All "), _row(m, "Posterior only"), _row(m, "Without AF3"), _row(m, "ICA, eye")
    nod, nog, art = _row(m, "Without δ (4"), _row(m, "Without γ (1"), _row(m, "Artifact markers only")
    e1, e2 = _row(m, "e1 "), _row(m, "e2 ")
    uni = cf["artifact_univariate_auc"]; bl = cf["blinks_per_minute"]; e3 = cf["drift_control_e3"]
    pers = {r["approach"]: r for r in m["personalization"]["results"]}

    check(min(aucs.values()) - idx["window"]["roc_auc"] > 0.15, "learned models clearly beat the index")
    check(all(p is not None and p > 0.05 for k, p in holm.items() if k in LEARNED and k != "logreg"), "models indistinguishable")
    check(ch[0]["name"] == "F8" and ch[1]["name"] == "T8", "F8 then T8 are the most important channels")
    check(reg["r2"] <= 0.05, "ratings not predicted")
    check(art["auc_lo"] > 0.6 and art["auc"] < ref["auc_lo"], "artifact-only above chance but below reference")
    check(post["auc_lo"] > 0.7 and ica["auc_lo"] > 0.7 and post["delta_hi"] < 0 and ica["delta_hi"] < 0,
          "brain-oriented subsets lower than reference but well above chance")
    check(nog["delta_lo"] < 0 < nog["delta_hi"], "removing gamma changes nothing")
    check(nod["delta_hi"] < 0, "removing delta lowers AUC")
    check(abs(e1["delta_auc"]) < 0.02 and e1["delta_lo"] < 0 < e1["delta_hi"], "time split across people unchanged")
    check(e3["median_auc"] < 0.75 and e2["auc"] > 0.95, "within-person near ceiling, drift control moderate")
    check(bl["wilcoxon_p"] > 0.05, "blink rate does not differ")

    top_ch = "; ".join(f'{c["name"]}, {LOCATION[c["name"]]} ({f(c["mean"], 3)})' for c in ch[:3])
    brain_drop = [-post["delta_auc"], -ica["delta_auc"]]
    return [
        {"title": "The full feature set outperforms the classic θ/α workload index",
         "html": f"<p>The frontal-θ / parietal-α ratio, a standard EEG workload index, achieves a window AUC of {_n(idx['window']['roc_auc'])} on held-out people "
                 f"and classifies {pct(idx['recording']['accuracy'])} of recordings correctly. Logistic regression on all 282 features achieves "
                 f"{_n(lr['window']['roc_auc'])} and {pct(lr['recording']['accuracy'])}.</p>"},
        {"title": "Model choice has no significant effect",
         "html": f"<p>Logistic regression, SVM, random forest and gradient boosting achieve between {_n(min(aucs.values()))} and {_n(max(aucs.values()))} AUC. "
                 f"Paired per-participant tests against logistic regression give Holm-corrected "
                 f"{'; '.join(f'p = {f(holm[k], 2)} ({MODEL_LABEL[k]})' for k in LEARNED if k != 'logreg')}. "
                 f"Additional model flexibility yields no gain; performance is limited by inter-individual variability: participant identity explains a median "
                 f"{pct(eta['subject'], 0)} of a feature's variance, rest vs. test {pct(eta['condition'], 1)}.</p>"},
        {"title": "Right frontal and temporal channels contribute most",
         "html": f"<p>Permuting one channel's features at a time, the largest AUC drops occur for {top_ch}. "
                 f"By feature type, relative θ power matters most (AUC drop {f(grp[0]['mean'], 3)}). Section 05 examines the extent to which this reflects ocular and muscle activity.</p>"},
        {"title": "Ocular artifacts explain part, but not all, of the performance",
         "html": f"<p>Six artifact markers alone achieve {_auc_ci(art)}. The horizontal eye-movement proxy (F7 − F8, 1–4 Hz) is the strongest single marker, "
                 f"with AUC {f(uni['heog_logpow'], 2)}; blink rate does not differ between conditions (median {f(bl['rest_median'], 1)} vs "
                 f"{f(bl['test_median'], 1)} per minute, p = {f(bl['wilcoxon_p'], 2)}). Removing eye components with ICA gives {_auc_ci(ica)}, and using only "
                 f"the four posterior channels gives {_auc_ci(post)}: {f(min(brain_drop), 3)}–{f(max(brain_drop), 3)} below the full model, but well above chance. "
                 f"Removing γ, where jaw and forehead muscle activity dominates, has no measurable effect (Δ {sgn(nog['delta_auc'])}, 95% CI "
                 f"{sgn(nog['delta_lo'])} to {sgn(nog['delta_hi'])}).</p>"},
        {"title": "Self-reported workload could not be predicted from EEG",
         "html": f"<p>Within the test condition, ridge regression on EEG features did not outperform a mean-only baseline in predicting 1–9 ratings: "
                 f"R² = {f(reg['r2'], 2)} (mean-only baseline {f(base['r2'], 2)}), MAE {f(reg['mae'], 2)}, n = {int(reg['n'])}. The classifier output was also "
                 f"uncorrelated with the ratings (Spearman ρ = {f(sr['rho'], 2)}, p = {f(sr['p'], 2)}). The EEG distinguishes the two conditions but shows no evidence "
                 f"of tracking perceived task difficulty. With n = {int(reg['n'])}, this test has limited power and does not rule out a relationship.</p>"},
        {"title": "Within-participant accuracy is inflated by recording drift",
         "html": f"<p>Trained on the first half of each participant's recordings and tested on the second half, the model achieves {_n(e2['auc'])}. "
                 f"However, a model distinguishing early from late windows <em>within a single recording</em>, where the condition is constant, also performs above chance "
                 f"(median AUC {f(e3['median_auc'], 2)} over {e3['n_recordings']} recordings; {e3['n_ge_0_8']} of them ≥ 0.8), indicating that recording drift makes segments "
                 f"partly identifiable. Across people, training on first halves and testing on second halves leaves AUC unchanged "
                 f"({f(e1['auc'])} vs {f(e1['ref_auc'])}). Per-participant normalisation did not improve performance: median per-person AUC "
                 f"{f(pers['A_subject_independent']['subject_auc_median'], 2)} without it, {f(pers['B1_subject_zscore']['subject_auc_median'], 2)} with a z-score, "
                 f"{f(pers['B2_rest_baseline']['subject_auc_median'], 2)} with a resting baseline.</p>"},
    ]


def confound_reading(m: dict) -> str:
    ref, post, ica, ica30 = _row(m, "All "), _row(m, "Posterior only"), _row(m, "ICA, eye"), _row(m, "ICA + without")
    art, nod = _row(m, "Artifact markers only"), _row(m, "Without δ (4")
    i = m["confounds"]["ica"]
    drops = [-r["delta_auc"] for r in (post, ica, ica30, _row(m, "Without AF3"))]
    return (f"<p>Removing eye-related signal in any of four ways (posterior channels only, no eye-adjacent channels, ICA, ICA without γ) lowers the AUC by "
            f"{f(min(drops), 3)}–{f(max(drops), 3)}. The most conservative of these, ICA-cleaned data with γ removed, still reaches {f(ica30['auc'])} "
            f"(95% CI {f(ica30['auc_lo'], 2)}–{f(ica30['auc_hi'], 2)}), against {f(ref['auc'])} for the full model. Six artifact markers alone achieve "
            f"{f(art['auc'])}.</p><p>Ocular activity therefore accounts for part of the full model's performance, but the corrected models retain most of it. "
            f"The remaining signal cannot be confirmed as neural: ICA removes only components matching the frontal eye proxies, and STEW provides no EOG channel for validation.</p>"
            f"<p>ICA removed on average {f(i['mean_removed']['lo'], 2)} components from rest recordings and {f(i['mean_removed']['hi'], 2)} from test recordings, "
            f"indicating no condition bias in the correction. It lowered 1–4 Hz power at AF3 by a median {f(-i['frontal_delta_change_db_median']['AF3']['lo'], 1)} dB "
            f"(rest) and {f(-i['frontal_delta_change_db_median']['AF3']['hi'], 1)} dB (test), and at O1 by {f(-i['frontal_delta_change_db_median']['O1']['lo'], 2)} dB.</p>")


def blocks(m: dict) -> dict[str, str]:
    out = {"metrics-table": metrics_table(m)}
    # optional blocks are added by later stages (findings, confound table)
    if "confounds" in m:
        out["confounds"] = confound_table(m)
        out["confound-reading"] = confound_reading(m)
        out["insights"] = "".join(
            f'<article class="card finding"><h3>{html.escape(x["title"])}</h3>{x["html"]}</article>' for x in findings(m))
    return out


def render(page: str, m: dict) -> str:
    for key, val in bind_values(m).items():
        pat = re.compile(r'(<(\w+)[^>]*\bdata-bind="' + re.escape(key) + r'"[^>]*>)(.*?)(</\2>)', re.S)
        page, n = pat.subn(lambda mo: mo.group(1) + html.escape(val) + mo.group(4), page)
        if n == 0:
            raise KeyError(f"data-bind slot {key!r} not found in index.html")
    for name, content in blocks(m).items():
        pat = re.compile(r"(<!-- BEGIN:" + re.escape(name) + r" -->)(.*?)(<!-- END:" + re.escape(name) + r" -->)", re.S)
        page, n = pat.subn(lambda mo: mo.group(1) + content + mo.group(3), page)
        if n == 0:
            raise KeyError(f"block {name!r} not found in index.html")
    return page


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    m = json.loads(METRICS.read_text())
    page = INDEX.read_text()
    new = render(page, m)
    if "--check" in argv:
        if new != page:
            print("website/index.html is out of date with website/data/metrics.json; run: python src/site_render.py")
            return 1
        print("website/index.html is in sync with metrics.json")
        return 0
    INDEX.write_text(new)
    print("rendered", INDEX.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
