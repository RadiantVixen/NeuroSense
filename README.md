# NeuroSense

Classifying resting EEG versus EEG recorded during a multitasking test, in people the classifier was not trained on.
Data: STEW (48 participants, Emotiv EPOC, 14 channels, 128 Hz). Site: https://radiantvixen.github.io/NeuroSense/

The classifier distinguishes rest from an active multitasking task. It does not estimate how much workload a person
experiences: predicting self-rated workload from EEG failed (R² = −0.05, n = 45).

## Results

All numbers below come from `results/` and `website/data/metrics.json`. Evaluation is nested cross-validation grouped by
participant: 8 outer folds of 6 people, hyper-parameters chosen inside each training set. Window = 4 s; recording accuracy
averages the window scores of each 2.5-minute recording. CIs resample participants (1000 draws).

| Model | Window ROC-AUC [95% CI] | Balanced acc. | Recording acc. |
|---|---|---|---|
| Frontal θ / parietal α index (1 feature) | 0.645 [0.59, 0.70] | 0.603 | 0.656 |
| Logistic regression (L2) | 0.881 [0.84, 0.92] | 0.800 | 0.865 |
| SVM (RBF) | 0.877 [0.84, 0.91] | 0.793 | 0.875 |
| Random forest | 0.877 [0.84, 0.91] | 0.801 | 0.885 |
| Gradient boosting | 0.863 [0.81, 0.91] | 0.793 | 0.865 |

The four learned models do not differ (Holm-corrected p = 1.0 against logistic regression). A within-participant label
permutation test gives p = 0.01 (100 permutations).

### Artifact checks (notebook 07)

Logistic regression, same windows and folds; reference AUC 0.881.

| Analysis | Window AUC [95% CI] | Δ vs. reference [95% CI] |
|---|---|---|
| Posterior channels only (O1, O2, P7, P8) | 0.817 [0.771, 0.856] | −0.064 [−0.091, −0.037] |
| Temporal channels only (T7, T8) | 0.725 [0.669, 0.780] | −0.156 [−0.216, −0.096] |
| Without AF3, AF4, F7, F8 | 0.824 [0.763, 0.876] | −0.057 [−0.090, −0.024] |
| Without δ (re-filtered 4–45 Hz) | 0.842 [0.794, 0.886] | −0.039 [−0.065, −0.014] |
| Without γ (re-filtered 1–30 Hz) | 0.880 [0.837, 0.917] | 0.000 [−0.010, +0.010] |
| ICA, eye components removed | 0.827 [0.785, 0.870] | −0.054 [−0.096, −0.010] |
| ICA and without γ | 0.808 [0.754, 0.859] | −0.073 [−0.119, −0.027] |
| Six artifact markers only | 0.765 [0.715, 0.821] | −0.116 [−0.172, −0.056] |
| First half → second half, across participants | 0.872 [0.820, 0.916] | −0.002 [−0.014, +0.010] |
| First half → second half, within each participant | 0.985 [0.973, 0.994] | — |
| Early vs. late within one recording (drift control) | median 0.629 per recording | — |

Removing eye-related signal costs 0.05–0.07 AUC; the corrected models stay near 0.81–0.83. An artifact-only model reaches
0.77, with the horizontal eye-movement proxy (F7 − F8) the strongest single marker. Muscle-dominated γ is not used.
Within-participant accuracy is close to 1, but time segments of a single recording are also partly separable, so that
number partly reflects recognising the recording.

## Limitations

- Rest is not a low-workload version of the task. The recordings differ in visual input, eye movement and posture as well as
  mental effort.
- STEW has no EOG or EMG channels. ICA and the artifact markers rely on proxies built from frontal EEG, so the remaining signal
  cannot be shown to be purely neural.
- One session per person and one continuous recording per condition: no test of stability across days, and within-person
  results are an upper bound.
- The participant (n = 48) is the independent unit; windows from one recording are correlated.
- The dataset documentation does not state units or reference; amplitude thresholds assume µV-like units.
- Zero-phase filtering, whole-recording bad-channel detection and ICA are offline operations.

## Data

STEW: `subXX_lo.txt` is the resting recording and `subXX_hi.txt` the SIMKAP multitasking test (19 200 samples × 14 channels:
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4). `ratings.txt` holds `subject, rating at rest, rating for test`
(1–9); ratings for subjects 5, 24 and 42 are missing. The raw data is not in this repository: download it from IEEE DataPort
and place the files in `data/raw/STEW/`.

- W. L. Lim, O. Sourina, L. Wang, "STEW: Simultaneous Task EEG Workload Data Set," *IEEE Transactions on Neural Systems and
  Rehabilitation Engineering*, 26(11), 2106–2114, 2018. doi:[10.1109/TNSRE.2018.2872924](https://doi.org/10.1109/TNSRE.2018.2872924)
- STEW dataset, IEEE DataPort, 2018. doi:[10.21227/44r8-ya50](https://doi.org/10.21227/44r8-ya50), CC BY 4.0.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions used for the committed results (Python 3.13.1)
pytest tests                             # dataset-structure tests skip when data/raw/STEW is absent

for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
python src/export_results.py             # writes website/data/*.json and the static numbers in website/index.html
python src/site_render.py --check        # fails if index.html no longer matches metrics.json
cd website && python -m http.server 8000
```


**Random seeds.** `modeling.SEED = 0` (shuffled `GroupKFold` folds, random forest, gradient boosting); participant bootstrap
seed 0; label permutations seeds 1000–1099; ICA `random_state` 0 (`confounds.ICA_SEED`); plot jitter seeds are fixed in the
notebooks and do not affect results.

## Layout

```
notebooks/  01 exploration · 02 preprocessing · 03 features · 04 training · 05 evaluation · 06 personalisation · 07 artifact checks
src/        data_loader, quality, preprocessing, feature_extraction, modeling, evaluation,
            personalization, confounds, export_results, site_render
tests/      pytest suite: synthetic-signal checks, dataset structure, website data and static HTML
results/    per-notebook summaries, CSVs and figures
website/    static site (HTML, CSS, JavaScript, Plotly.js); data/*.json written by export_results.py
```

## Results changelog

Each entry lists the commit where a set of numbers first appeared. To trace any figure, check the matching JSON under
`results/` at that commit.

| Commit | Date | Change |
|---|---|---|
| `c7918c3`, `9e9fa94` | 2026-11-25 | Notebooks 01–06 and the website. Logistic regression window AUC 0.881 [0.84, 0.92], recording accuracy 0.865; rating regression R² −0.05. |
| `9a05749` | 2026-11-25 | Notebook 07, artifact and drift analyses (table above). Earlier numbers unchanged. |

Add a row whenever a re-run changes any reported number.

## How to cite

See `CITATION.cff`. Please also cite the STEW paper and dataset above.

## Website deployment

The site is static and reads only `website/data/*.json`; nothing runs on GitHub's side. With Pages set to
"Deploy from branch" (root), the root `index.html` redirects to `website/`. With Pages set to "GitHub Actions",
`.github/workflows/pages.yml` publishes `website/` at the site root instead. `.github/workflows/tests.yml` runs the
tests and the `site_render.py --check` on every push.
