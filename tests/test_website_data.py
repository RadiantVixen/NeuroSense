"""Guards the published site: the JSON it fetches exists, the headline values are
not null, and the static HTML carries real numbers (no '—' placeholders)."""

import json
import re
from pathlib import Path

import pytest

import site_render

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"
FILES = ["metrics", "predictions", "visualizations", "eeg_examples"]


@pytest.fixture(scope="module")
def metrics():
    return json.loads((SITE / "data" / "metrics.json").read_text())


@pytest.mark.parametrize("name", FILES)
def test_json_exists_and_parses(name):
    p = SITE / "data" / f"{name}.json"
    assert p.is_file(), f"missing {p.relative_to(ROOT)}; run python src/export_results.py"
    assert json.loads(p.read_text())


def test_fetch_paths_are_relative_and_match_files():
    js = (SITE / "script.js").read_text()
    fetched = set(re.findall(r"loadJSON\(\"(\w+)\"\)", js)) | set(re.findall(r"\"(\w+)\"\]\.map\(loadJSON\)", js))
    fetched |= {n for group in re.findall(r"\[([^\]]*)\]\.map\(loadJSON\)", js) for n in re.findall(r"\"(\w+)\"", group)}
    assert fetched == set(FILES)
    assert "fetch(`data/${name}.json`)" in js  # relative, so it works under /NeuroSense/ and /NeuroSense/website/
    for n in fetched:  # GitHub Pages is case-sensitive
        assert (SITE / "data" / f"{n}.json").name in {p.name for p in (SITE / "data").iterdir()}


def test_hero_values_not_null(metrics):
    values = site_render.bind_values(metrics)  # raises MissingValue on a null
    for k, v in values.items():
        assert v and v not in {"—", "nan", "None"}, k


def test_static_html_in_sync_with_json(metrics):
    page = (SITE / "index.html").read_text()
    assert site_render.render(page, metrics) == page, "run: python src/site_render.py"
    for key in site_render.bind_values(metrics):
        m = re.search(r'data-bind="' + re.escape(key) + r'"[^>]*>([^<]*)<', page)
        assert m and m.group(1).strip() not in ("", "—"), key
    table = re.search(r"<!-- BEGIN:metrics-table -->(.*?)<!-- END:metrics-table -->", page, re.S).group(1)
    assert table.count("<tr") >= 6


def test_null_hero_value_is_rejected(metrics):
    bad = json.loads(json.dumps(metrics))
    bad["classification"][bad["best_model"]]["recording"]["accuracy"] = None
    with pytest.raises(site_render.MissingValue):
        site_render.bind_values(bad)
