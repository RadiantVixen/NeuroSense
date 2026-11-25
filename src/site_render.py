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
        "hero.recAccLabel": f"of recordings classified correctly, participants not seen in training ({MODEL_LABEL[best].lower()})",
        "hero.auc": f(w["roc_auc"], 2),
        "hero.aucLabel": f"ROC-AUC on 4 s windows (95% CI {f(w['auc_ci'][0], 2)}–{f(w['auc_ci'][1], 2)})",
        "about.nRated": str(rt["n_subjects_rated"]),
        "about.restMed": f(rt["rest"]["50%"], 0),
        "about.testMed": f(rt["test"]["50%"], 0),
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


def blocks(m: dict) -> dict[str, str]:
    out = {"metrics-table": metrics_table(m)}
    # optional blocks are added by later stages (findings, confound table)
    if "findings" in m:
        out["insights"] = "".join(
            f'<article class="card finding"><h3>{html.escape(x["title"])}</h3>{x["html"]}</article>' for x in m["findings"])
    if "confound_table_html" in m:
        out["confounds"] = m["confound_table_html"]
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
