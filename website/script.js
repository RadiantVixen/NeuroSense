/* NeuroSense front-end. Every number shown is read from data/*.json, which is
   produced by src/export_results.py from the validated notebook outputs. */
"use strict";

const C = {
  rest: "#338ced", test: "#e76624",          // condition identity (validated pair)
  aqua: "#00a888", violet: "#9887ef",        // generic 2-series pair (validated)
  cyan: "#2fd3e6",                           // single-series accent
  text: "#e6edf7", text2: "#93a4c0", text3: "#64748f",
  grid: "#1c2a47", axis: "#31415f", surface: "#0f1b33",
};
const COND = { lo: "rest", hi: "test" };
const MODEL_LABEL = {
  chance: "Chance (prior)", classic_index: "Classic θ/α index", logreg: "Logistic regression",
  svm_rbf: "SVM (RBF)", random_forest: "Random forest", grad_boost: "Gradient boosting",
};
const APPROACH_LABEL = {
  A_subject_independent: "A · subject-independent", B1_subject_zscore: "B1 · unsupervised z-score",
  B2_rest_baseline: "B2 · rest baseline", C_subject_dependent: "C · within-subject", D_hybrid: "D · hybrid",
};
const GROUP_LABEL = {
  time: "time-domain (Hjorth, moments, line length)", spec_ent: "spectral entropy", ratios: "band ratios & indices",
  abs_delta: "δ absolute", abs_theta: "θ absolute", abs_alpha: "α absolute", abs_beta: "β absolute", abs_gamma: "γ absolute",
  rel_delta: "δ relative", rel_theta: "θ relative", rel_alpha: "α relative", rel_beta: "β relative", rel_gamma: "γ relative",
};
const FEATURE_DESC = {
  fT_pA: "log₁₀ of frontal θ (AF3, F3, F4, AF4) over parietal α (P7, P8) — the classic EEG workload index.",
  O1_rel_alpha: "Share of 1–45 Hz power in the α band at left occipital O1; α typically falls with visual attention.",
  F3_rel_theta: "Share of power in θ at left frontal F3; frontal θ rises with working-memory load.",
  AF3_r_theta_alpha: "log₁₀ θ/α ratio at AF3 (frontal pole — also exposed to eye movements).",
  F8_abs_theta: "log₁₀ absolute θ power at F8, next to the right eye — the single most condition-related feature, likely partly ocular.",
  T7_rel_gamma: "Share of power in γ (30–45 Hz) at left temporal T7 — mostly jaw/temporal muscle activity at the scalp.",
};
const MONTAGE = {
  AF3: [-0.3, 0.83], AF4: [0.3, 0.83], F7: [-0.71, 0.55], F8: [0.71, 0.55], F3: [-0.36, 0.56], F4: [0.36, 0.56],
  FC5: [-0.62, 0.27], FC6: [0.62, 0.27], T7: [-0.93, 0], T8: [0.93, 0], P7: [-0.72, -0.56], P8: [0.72, -0.56],
  O1: [-0.29, -0.88], O2: [0.29, -0.88],
};
const EYE_ADJ = new Set(["AF3", "AF4", "F7", "F8"]);

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
const median = (a) => { const s = [...a].filter((v) => v != null).sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

/* ------------------------------------------------------------------ plotting helpers */
function isObj(v) { return v && typeof v === "object" && !Array.isArray(v); }
function merge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b || {})) out[k] = isObj(v) && isObj(a[k]) ? merge(a[k], v) : v;
  return out;
}
const AXIS = { gridcolor: C.grid, linecolor: C.axis, zerolinecolor: C.axis, tickcolor: C.axis, tickfont: { color: C.text2, size: 11 }, title: { font: { color: C.text2, size: 12 } }, automargin: true };
function baseLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: C.text2, size: 12 },
    margin: { l: 56, r: 16, t: 28, b: 48 },
    xaxis: { ...AXIS }, yaxis: { ...AXIS },
    legend: { orientation: "h", x: 0, y: 1.14, font: { color: C.text2, size: 12 }, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: "#0b1529", bordercolor: C.axis, font: { color: C.text, family: "Inter, system-ui, sans-serif" } },
    showlegend: false,
  };
}
const CONFIG = { displaylogo: false, responsive: true, displayModeBar: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian"] };
const TOOLBAR = new Set(["chart-eeg", "chart-ex-psd"]);
function plot(id, data, layout = {}) {
  const el = document.getElementById(id);
  if (!el || !window.Plotly) return;
  return Plotly.react(el, data, merge(baseLayout(), layout), TOOLBAR.has(id) ? { ...CONFIG, displayModeBar: "hover" } : CONFIG);
}
function segToggle(group, attr, onChange) {
  $$(`button[data-${attr}]`, group).forEach((b) => b.addEventListener("click", () => {
    $$(`button[data-${attr}]`, group).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    onChange(b.dataset[attr]);
  }));
}
function fillSelect(sel, items) {
  sel.innerHTML = items.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
}

/* ------------------------------------------------------------------ data */
const DATA = {};
async function loadJSON(name) {
  const r = await fetch(`data/${name}.json`);
  if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ hero */
function drawHeroTraces(eeg) {
  // Static, faint drawing of *recorded* EEG (filtered) - decoration, not a live signal.
  const cv = $("#hero-wave"); if (!cv || !eeg) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const rec = Object.values(eeg.recordings)[0];
  const chans = ["O1", "F3", "P8", "AF4", "T7"].map((c) => eeg.channels.indexOf(c));
  chans.forEach((ci, k) => {
    const sig = rec.filtered[ci]; const n = sig.length;
    const y0 = h * (0.2 + 0.15 * k);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w, y = y0 - sig[i] * 0.9;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = k % 2 ? "rgba(47,211,230,0.35)" : "rgba(59,157,255,0.35)"; ctx.lineWidth = 1; ctx.stroke();
  });
}
/* ------------------------------------------------------------------ about */
function renderRatings(v) {
  const rows = v.ratings; const rng = mulberry(7);
  const traces = [];
  const jit = rows.map(() => (rng() - 0.5) * 0.25);
  rows.forEach((r, i) => traces.push({ x: [0, 1], y: [r.rating_rest + jit[i], r.rating_test + jit[i]], mode: "lines", line: { color: "rgba(147,164,192,0.22)", width: 1 }, hoverinfo: "skip" }));
  for (const [c, key] of [["lo", "rating_rest"], ["hi", "rating_test"]]) {
    traces.push({
      x: rows.map(() => (c === "lo" ? 0 : 1)), y: rows.map((r, i) => r[key] + jit[i]), mode: "markers", name: COND[c],
      marker: { color: C[COND[c]], size: 8, line: { color: C.surface, width: 2 } },
      customdata: rows.map((r) => [r.subject, r[key]]), hovertemplate: `participant %{customdata[0]}<br>${COND[c]} rating: %{customdata[1]}<extra></extra>`,
    });
  }
  plot("chart-ratings", traces, {
    xaxis: { tickvals: [0, 1], ticktext: ["rest", "multitasking test"], range: [-0.35, 1.35], showgrid: false },
    yaxis: { title: { text: "self-rated workload (1–9)" }, range: [0.3, 9.7], dtick: 1 }, margin: { t: 10 },
  });
}
function renderMontage() {
  const th = Array.from({ length: 101 }, (_, i) => (i / 100) * 2 * Math.PI);
  const names = Object.keys(MONTAGE);
  const tr = [
    { x: th.map(Math.cos), y: th.map(Math.sin), mode: "lines", line: { color: C.axis, width: 1.5 }, hoverinfo: "skip", showlegend: false },
    { x: [-0.12, 0, 0.12], y: [0.99, 1.13, 0.99], mode: "lines", line: { color: C.axis, width: 1.5 }, hoverinfo: "skip", showlegend: false },
  ];
  for (const [grp, sel, col, nm] of [["eye", (n) => EYE_ADJ.has(n), C.violet, "eye-adjacent"], ["other", (n) => !EYE_ADJ.has(n), C.aqua, "other"]]) {
    const ns = names.filter(sel);
    tr.push({
      x: ns.map((n) => MONTAGE[n][0]), y: ns.map((n) => MONTAGE[n][1]), mode: "markers+text", name: nm, text: ns, textposition: "bottom center",
      textfont: { color: C.text, size: 11 }, marker: { size: 13, color: col, line: { color: C.surface, width: 2 } }, hovertemplate: "%{text}<extra>" + nm + "</extra>",
    });
  }
  plot("chart-montage", tr, {
    showlegend: true, legend: { y: 1.12 }, margin: { l: 10, r: 10, t: 30, b: 10 },
    xaxis: { visible: false, range: [-1.25, 1.25], scaleanchor: "y" }, yaxis: { visible: false, range: [-1.15, 1.25] },
  });
}

/* ------------------------------------------------------------------ explorer */
const EX = { key: null, ver: "filtered", chans: null };
function initExplorer(eeg) {
  const keys = Object.keys(eeg.recordings);
  const subs = [...new Set(keys.map((k) => +k.split("_")[0]))];
  const worst = subs[subs.length - 1];
  fillSelect($("#ex-rec"), keys.map((k) => {
    const r = eeg.recordings[k];
    return [k, `Participant ${r.subject} · ${r.label}${r.subject === worst ? " (most rejected windows)" : ""}`];
  }));
  EX.key = keys[0]; EX.chans = new Set(eeg.channels);
  $("#ex-channels").innerHTML = eeg.channels.map((c) => `<button type="button" class="chip" data-ch="${c}" aria-pressed="true">${c}</button>`).join("");
  $$("#ex-channels .chip").forEach((b) => b.addEventListener("click", () => {
    const on = b.getAttribute("aria-pressed") !== "true";
    if (!on && EX.chans.size === 1) return;
    b.setAttribute("aria-pressed", String(on)); on ? EX.chans.add(b.dataset.ch) : EX.chans.delete(b.dataset.ch);
    drawExplorer(eeg);
  }));
  $("#ex-all").addEventListener("click", () => { EX.chans = new Set(eeg.channels); $$("#ex-channels .chip").forEach((b) => b.setAttribute("aria-pressed", "true")); drawExplorer(eeg); });
  $("#ex-rec").addEventListener("change", (e) => { EX.key = e.target.value; drawExplorer(eeg); });
  segToggle($("#explorer .seg"), "ver", (v) => { EX.ver = v; drawExplorer(eeg); });
  drawExplorer(eeg);
}
function drawExplorer(eeg) {
  const rec = eeg.recordings[EX.key];
  const chans = eeg.channels.filter((c) => EX.chans.has(c));
  const sig = rec[EX.ver];
  const n = sig[0].length, t = Array.from({ length: n }, (_, i) => eeg.start_s + i / eeg.fs);
  const sds = chans.map((c) => { const s = sig[eeg.channels.indexOf(c)]; const m = s.reduce((a, b) => a + b, 0) / n; return Math.sqrt(s.reduce((a, b) => a + (b - m) ** 2, 0) / n); });
  const spacing = Math.max(40, 5 * median(sds));
  const bad = new Set(rec.bad_channels);
  const traces = chans.map((c, k) => {
    const s = sig[eeg.channels.indexOf(c)], off = -k * spacing, lim = 1.5 * spacing;
    // bad (excluded) channels are drawn dimmed and clipped so they do not hide the others
    const y = bad.has(c) ? s.map((v) => Math.max(-lim, Math.min(lim, v)) + off) : s.map((v) => v + off);
    return { x: t, y, customdata: s, mode: "lines", name: c, line: { color: bad.has(c) ? "#4a5874" : EYE_ADJ.has(c) ? "#b7a9f5" : "#9fd8e6", width: 1 },
      hovertemplate: `${c}${bad.has(c) ? " (excluded, clipped)" : ""}: %{customdata:.0f}<br>t = %{x:.2f} s<extra></extra>` };
  });
  plot("chart-eeg", traces, {
    margin: { l: 52, t: 10 }, hovermode: "closest",
    xaxis: { title: { text: "time in recording (s)" }, showgrid: true },
    yaxis: { tickvals: chans.map((_, k) => -k * spacing), ticktext: chans.map((c) => (bad.has(c) ? `${c} (excluded)` : c)), showgrid: false, zeroline: false, range: [-(chans.length - 0.3) * spacing, spacing * 0.9] },
  });
  $("#ex-meta").textContent = `${pct(rec.rejected_fraction, 0)} of windows rejected` + (rec.bad_channels.length ? ` · excluded channel: ${rec.bad_channels.join(", ")}` : "") +
    ` · trace spacing ${spacing.toFixed(0)} units · purple = eye-adjacent`;
  const ci = eeg.channels.indexOf(chans[0]);
  plot("chart-ex-psd", [
    { x: rec.psd_freqs, y: rec.psd_raw[ci], name: "raw", mode: "lines", line: { color: C.violet, width: 2 }, hovertemplate: "%{x:.2f} Hz: %{y:.2f}<extra>raw</extra>" },
    { x: rec.psd_freqs, y: rec.psd_filtered[ci], name: "filtered", mode: "lines", line: { color: C.aqua, width: 2 }, hovertemplate: "%{x:.2f} Hz: %{y:.2f}<extra>filtered</extra>" },
  ], {
    showlegend: true, xaxis: { title: { text: `frequency (Hz) — channel ${chans[0]}` }, range: [0, 64] }, yaxis: { title: { text: "log₁₀ PSD" } },
    shapes: [vline(45, "45 Hz"), vline(50, "")], annotations: [{ x: 45, y: 1, yref: "paper", text: "analysis limit", showarrow: false, xanchor: "right", font: { size: 10, color: C.text3 } }],
  });
}
function vline(x) { return { type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: C.axis, width: 1 } }; }

/* ------------------------------------------------------------------ processing */
function renderProcessing(v, m) {
  const fr = v.filter_response;
  plot("chart-filter", [{ x: fr.freqs, y: fr.gain_db, mode: "lines", line: { color: C.cyan, width: 2 }, hovertemplate: "%{x:.2f} Hz: %{y:.1f} dB<extra></extra>" }], {
    xaxis: { title: { text: "frequency (Hz)" }, range: [0, 64] }, yaxis: { title: { text: "gain (dB)" }, range: [-80, 6] },
    shapes: [fr.hp_hz, fr.lp_hz, fr.notch_hz].map(vline),
    annotations: [[fr.hp_hz, "1 Hz"], [fr.lp_hz, "45 Hz"], [fr.notch_hz, "50 Hz notch"]].map(([x, t]) => ({ x, y: 1, yref: "paper", text: t, showarrow: false, xanchor: "left", xshift: 3, font: { size: 10, color: C.text3 } })),
  });

  const g = v.group_psd, tr = [];
  for (const c of ["lo", "hi"]) {
    const col = C[COND[c]];
    tr.push({ x: g.freqs, y: g[c].hi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
    tr.push({ x: g.freqs, y: g[c].lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: hexA(col, 0.2), hoverinfo: "skip", showlegend: false });
    tr.push({ x: g.freqs, y: g[c].mean, mode: "lines", name: COND[c], line: { color: col, width: 2 }, hovertemplate: `%{x:.2f} Hz: %{y:.2f}<extra>${COND[c]}</extra>` });
  }
  const bands = Object.entries(v.bands);
  plot("chart-group-psd", tr, {
    showlegend: true, hovermode: "x unified",
    margin: { t: 44 }, legend: { y: 1.2 },
    xaxis: { title: { text: "frequency (Hz)" }, range: [0.5, 46] }, yaxis: { title: { text: "log₁₀ PSD" }, range: [-2.2, 2.4] },
    shapes: bands.map(([, [a, b]], i) => ({ type: "rect", x0: a, x1: b, yref: "paper", y0: 0, y1: 1, fillcolor: i % 2 ? "rgba(147,164,192,0.04)" : "rgba(147,164,192,0.09)", line: { width: 0 }, layer: "below" })),
    annotations: bands.map(([n, [a, b]]) => ({ x: (a + b) / 2, y: 1, yref: "paper", yanchor: "bottom", text: { delta: "δ", theta: "θ", alpha: "α", beta: "β", gamma: "γ" }[n], showarrow: false, font: { size: 13, color: C.text2 } })),
  });

  const bc = v.band_contrast, bandNames = Object.keys(v.bands), chs = [...new Set(bc.map((d) => d.channel))];
  const z = bandNames.map((b) => chs.map((c) => bc.find((d) => d.band === b && d.channel === c).median_log_ratio));
  const zmax = Math.max(...z.flat().map(Math.abs));
  const sig = bc.filter((d) => d.significant);
  plot("chart-contrast", [
    { type: "heatmap", x: chs, y: bandNames, z, zmin: -zmax, zmax, xgap: 2, ygap: 2,
      colorscale: [[0, C.rest], [0.5, "#2a3550"], [1, C.test]],
      colorbar: { title: { text: "log₁₀ ratio", side: "right", font: { size: 11, color: C.text2 } }, tickfont: { color: C.text2, size: 10 }, thickness: 10, outlinewidth: 0 },
      customdata: bandNames.map((b) => chs.map((c) => { const d = bc.find((e) => e.band === b && e.channel === c); return [d.frac_subj_increase, d.p_fdr]; })),
      hovertemplate: "%{y} @ %{x}<br>median log₁₀(test/rest): %{z:.3f}<br>participants with increase: %{customdata[0]:.0%}<br>FDR q = %{customdata[1]:.3g}<extra></extra>" },
    { x: sig.map((d) => d.channel), y: sig.map((d) => d.band), mode: "markers", marker: { color: C.text, size: 5 }, hoverinfo: "skip" },
  ], { margin: { l: 56, t: 10 }, xaxis: { showgrid: false, tickangle: 0, tickfont: { size: 10 } }, yaxis: { showgrid: false, autorange: "reversed" } });

  const sw = v.rejection_sweep;
  plot("chart-sweep", ["rest", "test"].map((c) => ({
    x: sw.map((d) => d.threshold), y: sw.map((d) => d[c]), name: c, mode: "lines+markers", line: { color: C[c], width: 2 }, marker: { size: 8, color: C[c], line: { color: C.surface, width: 2 } },
    hovertemplate: `±%{x}: %{y:.1%} of ${c} windows rejected<extra></extra>`,
  })), {
    showlegend: true, xaxis: { title: { text: "amplitude threshold (± units, ≈ µV)" } }, yaxis: { title: { text: "windows rejected" }, tickformat: ".0%", rangemode: "tozero" },
    shapes: [vline(200)], annotations: [{ x: 200, y: 1, yref: "paper", text: "chosen", showarrow: false, xanchor: "left", xshift: 4, font: { size: 10, color: C.text3 } }],
  });

  const feats = Object.keys(v.feature_distributions);
  fillSelect($("#feat-sel"), feats.map((f) => [f, f]));
  const drawFeat = (f) => {
    const d = v.feature_distributions[f], mids = d.edges.slice(0, -1).map((e, i) => (e + d.edges[i + 1]) / 2);
    plot("chart-feat", ["lo", "hi"].map((c) => ({
      x: mids, y: d[c], name: COND[c], mode: "lines", line: { color: C[COND[c]], width: 2, shape: "hvh" }, fill: "tozeroy", fillcolor: hexA(C[COND[c]], 0.18),
      hovertemplate: `%{x:.3f}: %{y:.1%} of ${COND[c]} windows<extra></extra>`,
    })), { xaxis: { title: { text: f } }, yaxis: { title: { text: "share of windows" }, tickformat: ".0%" }, margin: { t: 10 } });
    $("#feat-desc").textContent = FEATURE_DESC[f] || "";
  };
  $("#feat-sel").addEventListener("change", (e) => drawFeat(e.target.value));
  drawFeat(feats[0]);

  const e = v.variance_decomposition, labels = e.groups.map((x) => GROUP_LABEL[x] || x);
  plot("chart-eta", [
    { y: labels, x: e.subject, name: "participant identity", type: "bar", orientation: "h", marker: { color: C.aqua }, hovertemplate: "%{y}<br>η² participant: %{x:.3f}<extra></extra>" },
    { y: labels, x: e.condition, name: "rest vs test", type: "bar", orientation: "h", marker: { color: C.violet }, hovertemplate: "%{y}<br>η² condition: %{x:.3f}<extra></extra>" },
  ], { showlegend: true, legend: { xref: "container", x: 0.02, y: 0.98, yanchor: "top" }, barmode: "group", bargap: 0.3, bargroupgap: 0.12, margin: { l: 10, t: 48 }, xaxis: { title: { text: "median η² (share of variance)" } }, yaxis: { autorange: "reversed", tickfont: { size: 10.5 } } });
}

/* ------------------------------------------------------------------ machine learning */
const ML_ORDER = ["chance", "classic_index", "logreg", "svm_rbf", "random_forest", "grad_boost"];
function renderML(m, preds) {
  const cl = m.classification, models = ML_ORDER.filter((k) => cl[k]);
  plot("chart-models", [
    { x: models.map((k) => MODEL_LABEL[k]), y: models.map((k) => cl[k].window.roc_auc), name: "window ROC-AUC (95% CI)", mode: "markers", type: "scatter",
      marker: { color: C.aqua, size: 11, line: { color: C.surface, width: 2 } },
      error_y: { type: "data", symmetric: false, array: models.map((k) => cl[k].window.auc_ci[1] - cl[k].window.roc_auc), arrayminus: models.map((k) => cl[k].window.roc_auc - cl[k].window.auc_ci[0]), color: C.aqua, thickness: 1.5, width: 6 },
      hovertemplate: "%{x}<br>window AUC %{y:.3f}<extra></extra>" },
    { x: models.map((k) => MODEL_LABEL[k]), y: models.map((k) => cl[k].recording.accuracy), name: "recording accuracy", mode: "markers", type: "scatter",
      marker: { color: C.violet, size: 11, symbol: "diamond", line: { color: C.surface, width: 2 } }, hovertemplate: "%{x}<br>recording accuracy %{y:.1%}<extra></extra>" },
  ], { showlegend: true, xaxis: { showgrid: false }, yaxis: { range: [0.35, 1.0], title: { text: "score" } },
       shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: C.axis, width: 1 } }],
       annotations: [{ xref: "paper", x: 1, y: 0.5, text: "chance", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: C.text3 } }] });

  const best = models.filter((k) => !["chance", "classic_index"].includes(k)).reduce((a, b) => (cl[a].window.roc_auc >= cl[b].window.roc_auc ? a : b));
  // confusion matrix
  const cmModels = models.filter((k) => k !== "chance");
  fillSelect($("#cm-model"), cmModels.map((k) => [k, MODEL_LABEL[k]]));
  $("#cm-model").value = best;
  let lvl = "window";
  const drawCM = () => {
    const k = $("#cm-model").value, cm = lvl === "window" ? cl[k].window.cm : cl[k].recording.confusion_matrix;
    const rowsum = cm.map((r) => r[0] + r[1]), z = cm.map((r, i) => r.map((v) => v / rowsum[i]));
    plot("chart-cm", [{
      type: "heatmap", z, x: ["pred. rest", "pred. test"], y: ["true rest", "true test"], zmin: 0, zmax: 1, xgap: 3, ygap: 3,
      colorscale: [[0, "#13213d"], [0.5, "#1b64be"], [1, "#9dc8f7"]], showscale: false,
      text: cm.map((r, i) => r.map((v, j) => `${v}<br>${pct(z[i][j], 0)}`)), texttemplate: "%{text}", textfont: { size: 14, color: C.text },
      hovertemplate: "%{y} → %{x}: %{text}<extra></extra>",
    }], { margin: { l: 70, t: 10, b: 30 }, xaxis: { showgrid: false, side: "bottom" }, yaxis: { showgrid: false, autorange: "reversed" } });
  };
  $("#cm-model").addEventListener("change", drawCM);
  segToggle($("#cm-model").closest(".card").querySelector(".seg"), "lvl", (v) => { lvl = v; drawCM(); });
  drawCM();

  // per-subject AUC
  const ps = m.per_subject_auc;
  const psModels = ML_ORDER.filter((k) => ps[k]);
  plot("chart-subj", psModels.map((k) => ({
    type: "box", y: ps[k].auc, x: ps[k].auc.map(() => MODEL_LABEL[k]), name: MODEL_LABEL[k], boxpoints: "all", jitter: 0.5, pointpos: 0,
    marker: { color: C.cyan, size: 5, opacity: 0.7 }, line: { color: C.text2, width: 1 }, fillcolor: "rgba(0,0,0,0)",
    customdata: ps[k].subject, hovertemplate: "participant %{customdata}<br>AUC %{y:.3f}<extra></extra>",
  })), { xaxis: { showgrid: false, tickfont: { size: 10 } }, yaxis: { title: { text: "per-participant AUC" }, range: [0, 1.05] },
         shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: C.axis, width: 1 } }] });

  // importance
  const impModels = Object.keys(m.feature_importance);
  fillSelect($("#imp-model"), impModels.map((k) => [k, MODEL_LABEL[k]]));
  let kind = "group";
  const drawImp = () => {
    const d = m.feature_importance[$("#imp-model").value][kind];
    plot("chart-imp", [{
      type: "bar", orientation: "h", y: d.map((r) => (kind === "group" ? GROUP_LABEL[r.name] || r.name : r.name)), x: d.map((r) => r.mean),
      error_x: { type: "data", array: d.map((r) => r.sd), color: C.text3, thickness: 1, width: 3 },
      marker: { color: d.map((r) => (kind === "channel" && EYE_ADJ.has(r.name) ? C.violet : C.cyan)) },
      hovertemplate: "%{y}<br>AUC drop %{x:.4f}<extra></extra>",
    }], { margin: { l: 10, t: 10 }, yaxis: { autorange: "reversed", tickfont: { size: 10.5 } }, xaxis: { title: { text: "drop in held-out AUC" } },
          annotations: kind === "channel" ? [{ xref: "paper", yref: "paper", x: 1, y: 0, xanchor: "right", yanchor: "bottom", showarrow: false, text: "<span style='color:#9887ef'>■</span> eye-adjacent channel", font: { size: 11, color: C.text2 } }] : [] });
  };
  $("#imp-model").addEventListener("change", drawImp);
  segToggle($("#imp-model").closest(".card").querySelector(".seg"), "kind", (v) => { kind = v; drawImp(); });
  drawImp();

  // prediction timeline
  const subs = preds.subjects;
  fillSelect($("#pred-subj"), subs.map((s) => [s.subject, `Participant ${s.subject}${s.auc != null ? ` · AUC ${fmt(s.auc, 2)}` : ""}`]));
  const medAuc = median(subs.map((s) => s.auc));
  const typical = subs.reduce((a, b) => (Math.abs(b.auc - medAuc) < Math.abs(a.auc - medAuc) ? b : a));
  $("#pred-subj").value = typical.subject;
  const drawPred = () => {
    const s = subs.find((x) => x.subject === +$("#pred-subj").value);
    plot("chart-pred", ["lo", "hi"].map((c) => ({
      x: s.timeline[c].t, y: s.timeline[c].score, name: `${COND[c]} recording`, mode: "markers", type: "scatter",
      marker: { color: C[COND[c]], size: 8, line: { color: C.surface, width: 2 } }, hovertemplate: `${COND[c]} · t=%{x:.0f} s<br>score %{y:.2f}<extra></extra>`,
    })), { showlegend: true, xaxis: { title: { text: "window centre within the recording (s)" }, range: [0, 150] },
           yaxis: { title: { text: preds.score_kind }, range: preds.threshold === 0.5 ? [-0.03, 1.03] : null },
           shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: preds.threshold, y1: preds.threshold, line: { color: C.text3, width: 1 } }],
           annotations: [{ xref: "paper", x: 1, y: preds.threshold, text: "decision threshold", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: C.text3 } }] });
    const ok = (s.rest_pred === 0) + (s.test_pred === 1);
    $("#pred-meta").textContent = `mean score: rest ${fmt(s.rest_score, 2)} · test ${fmt(s.test_score, 2)} → ${ok}/2 recordings correct` +
      (s.rating_test != null ? ` · self-rated workload ${s.rating_rest} → ${s.rating_test}` : " · no rating available");
  };
  $("#pred-subj").addEventListener("change", drawPred);
  drawPred();

  // personalisation
  const pe = m.personalization, keys = pe.results.map((r) => r.approach);
  plot("chart-pers", keys.map((k) => ({
    type: "box", y: m.personalization_per_subject[k], x: m.personalization_per_subject[k].map(() => APPROACH_LABEL[k] || k), name: APPROACH_LABEL[k] || k,
    boxpoints: "all", jitter: 0.5, pointpos: 0, marker: { color: C.cyan, size: 5, opacity: 0.7 }, line: { color: C.text2, width: 1 }, fillcolor: "rgba(0,0,0,0)",
    hovertemplate: "AUC %{y:.3f}<extra></extra>",
  })), { xaxis: { showgrid: false, tickfont: { size: 10.5 } }, yaxis: { title: { text: "per-participant AUC (late windows)" }, range: [0, 1.05] },
         shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: C.axis, width: 1 } }],
         annotations: pe.results.map((r) => ({ x: APPROACH_LABEL[r.approach] || r.approach, y: 1.04, text: `median ${fmt(r.subject_auc_median, 2)}`, showarrow: false, font: { size: 10, color: C.text2 } })) });
}

/* ------------------------------------------------------------------ artifact analyses */
function renderConfounds(m) {
  const rows = m.confounds.table.filter((r) => r.auc != null).slice().reverse();
  const ref = m.confounds.table.find((r) => r.section === "reference").auc;
  plot("chart-confounds", [{
    x: rows.map((r) => r.auc), y: rows.map((r) => r.analysis), type: "scatter", mode: "markers",
    marker: { color: rows.map((r) => (r.section === "reference" ? C.text : r.section.startsWith("d") ? C.violet : C.cyan)), size: 10, line: { color: C.surface, width: 2 } },
    error_x: { type: "data", symmetric: false, array: rows.map((r) => r.auc_hi - r.auc), arrayminus: rows.map((r) => r.auc - r.auc_lo), color: C.text3, thickness: 1.5, width: 4 },
    customdata: rows.map((r) => [r.auc_lo, r.auc_hi, r.delta_auc, r.n_features]),
    hovertemplate: "%{y}<br>AUC %{x:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>Δ vs all features %{customdata[2]:+.3f}<br>%{customdata[3]} features<extra></extra>",
  }], {
    margin: { l: 10, t: 10 }, xaxis: { title: { text: "ROC-AUC on held-out participants (95% CI)" }, range: [0.45, 1] },
    yaxis: { tickfont: { size: 11 }, automargin: true },
    shapes: [ref, 0.5].map((x, i) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: i ? C.axis : C.text3, width: 1 } })),
    annotations: [{ x: 0.5, y: 0, yref: "paper", text: "chance", showarrow: false, xanchor: "left", xshift: 4, yanchor: "bottom", font: { size: 10, color: C.text3 } }],
  });
}

/* ------------------------------------------------------------------ utils */
function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function onVisible(el, fn) {
  if (!("IntersectionObserver" in window)) return fn();
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: "300px" });
  io.observe(el);
}
/* Highlight the nav link of the section in view. (Sections are always visible; an earlier
   fade-in kept them at opacity 0 until observed, which left the page blank after #anchor jumps.) */
function initNavSpy() {
  if (!("IntersectionObserver" in window)) return;
  const links = $$(".nav-links a"), secs = links.map((a) => $(a.getAttribute("href")));
  const spy = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`)); }), { rootMargin: "-45% 0px -50% 0px" });
  secs.forEach((s) => s && spy.observe(s));
}

/* The browser jumps to #anchor before charts and data have finished laying out, so the
   target moves and the jump lands in the wrong place. Re-apply it after each layout change,
   unless the visitor has already started scrolling. */
let userScrolled = false;
["wheel", "touchstart", "keydown", "mousedown"].forEach((ev) => addEventListener(ev, () => { userScrolled = true; }, { once: true, passive: true }));
function settleHash() {
  if (userScrolled || !location.hash) return;
  const t = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (t) t.scrollIntoView({ block: "start" });
}

function safe(fn, id) { try { fn(); } catch (e) { console.error(e); const el = id && document.getElementById(id); if (el) el.innerHTML = `<p class="error small">Could not render: ${e.message}</p>`; } }

async function main() {
  initNavSpy();
  settleHash();
  if (window.renderMathInElement) renderMathInElement($("#docs"), { delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
  try {
    [DATA.metrics, DATA.vis, DATA.preds] = await Promise.all(["metrics", "visualizations", "predictions"].map(loadJSON));
  } catch (e) {
    $$(".chart").forEach((el) => (el.innerHTML = `<p class="error small">Results not found (${e.message}). Run <code>python src/export_results.py</code> and serve the <code>website/</code> folder over HTTP.</p>`));
    return;
  }
  if (!window.Plotly) { $$(".chart").forEach((el) => (el.innerHTML = '<p class="error small">Plotly.js failed to load (offline?).</p>')); }
  onVisible($("#about"), () => { safe(() => renderRatings(DATA.vis), "chart-ratings"); safe(renderMontage, "chart-montage"); });
  onVisible($("#processing"), () => safe(() => renderProcessing(DATA.vis, DATA.metrics), "chart-filter"));
  onVisible($("#ml"), () => safe(() => renderML(DATA.metrics, DATA.preds), "chart-models"));
  onVisible($("#artifacts"), () => safe(() => renderConfounds(DATA.metrics), "chart-confounds"));
  loadJSON("eeg_examples").then((eeg) => {
    DATA.eeg = eeg;
    drawHeroTraces(eeg);
    onVisible($("#explorer"), () => safe(() => initExplorer(eeg), "chart-eeg"));
    settleHash();
  }).catch((e) => { $("#chart-eeg").innerHTML = `<p class="error small">${e.message}</p>`; });
}
document.addEventListener("DOMContentLoaded", main);
addEventListener("load", () => {
  settleHash();
  setTimeout(() => { settleHash(); document.documentElement.classList.add("ready"); }, 800);
});
