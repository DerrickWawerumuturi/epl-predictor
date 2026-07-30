<img src="logo.svg" alt="XGauges" width="88" align="right" />

# XGauges — Euro 2024 → Premier League

Rating Euro 2024 players, and forecasting how much they actually produced in the 2025–26 Premier League season.

Two components, deliberately kept apart:

1. **Euro Impact Index** — a transparent weighted score that *describes* Euro 2024 performance. No training, no accuracy claim.
2. **2025–26 Forecast** — a supervised model that predicts real Premier League output from Euro form, cross-validated against two baselines.

A React frontend visualises the index. Full methodology and known limitations are in **[MODEL_CARD.md](MODEL_CARD.md)**.

---

## Why the split matters

The first version of this project trained a Random Forest whose target was arithmetic on its own features — the label for forwards was `z(gls_90 − xg_90)` while both `gls_90` and `xg_90` sat in the feature list. It reported strong metrics because it was recovering a subtraction, not learning football. It also claimed to predict the 2025–26 season while containing no 2025–26 data at all.

Rather than paper over that, the descriptive part and the predictive part are now separate programs with separate claims. `MODEL_CARD.md` §3 lists every issue found and what changed.

`backend/models/Models.py` is the original pipeline. It is kept deliberately, unmodified and unused, so the before/after is legible — finding and fixing the leak is the interesting part of this project.

---

## Layout

```
backend/
  data/
    tournaments/euros.py      # FBref → Euro 2024 aggregate (via soccerdata)
    fetch_actuals.py          # FPL API → real 2025-26 EPL output (the labels)
    actuals/                  # generated
  models/
    common.py                 # loading, role mapping, name normalisation
    index_score.py            # Euro Impact Index      (no labels needed)
    forecast.py               # supervised forecast     (labels required)
    Models.py                 # SUPERSEDED — the original leaky pipeline, kept for reference
    outputs/                  # generated CSV + metrics JSON
frontend/                     # React + Vite single-page UI
MODEL_CARD.md
```

---

## Run it

**Install**

```bash
pip install pandas numpy scikit-learn scipy pyarrow soccerdata lxml requests
```

**1. Euro Impact Index** — works offline from the committed parquet:

```bash
cd backend/models
python index_score.py
```

Writes `outputs/euro_impact_index.csv` and `outputs/index_meta.json`.

**2. The forecast** — needs the real labels first:

```bash
python backend/data/fetch_actuals.py     # public FPL API, no key
python backend/models/forecast.py
```

Writes `outputs/forecast_predictions.csv`, `outputs/forecast_metrics.json` and `outputs/feature_importance.csv`.

`forecast.py` deliberately exits with an error if the labels are missing. It will not invent a target.

**3. Frontend**

```bash
cd frontend
npm install
npm run dev
```

`dist/` is a build artefact and is not committed — run `npm run build` to produce it.

---

## Method summary

**Index** (outfield, each component z-scored within position):

| Weight | Component |
|---|---|
| 0.55 | expected goal involvement per 90 (`npxg_xag_90`) |
| 0.25 | actual goals + assists per 90 |
| 0.20 | share of team minutes |

Goalkeepers: availability only. Rating = percentile within position → 1.0–5.0. Minimum 180 Euro minutes.

**Forecast:** Euro 2024 features → actual 2025–26 EPL expected goal involvement per 90. 5-fold CV, MAE / RMSE / R² / Spearman ρ with fold spread, benchmarked against a mean predictor and a minutes-only model, permutation importance for attribution. Minimum 450 Premier League minutes.

---

## Result

Euro 2024 form does carry rank information about 2025–26 Premier League output: **Spearman ρ = +0.54, bootstrap 95% CI [+0.285, +0.715]** (n = 47 outfield players, 5-fold CV), with expected-goal metrics topping permutation importance.

**But no trained model beat a one-feature baseline.** A Ridge regression on Euro *minutes alone* wins on MAE (0.1116 vs 0.1228), RMSE and R². The reported conclusion is therefore negative: on this sample, most of the recoverable signal is selection — good players play more — not anything the per-90 metrics add on top. `forecast.py` prints `WORSE than counting minutes` automatically, because that is the point of having a baseline.

The binding constraint is n = 47, not model choice. Full numbers, the failure analysis, and the survivorship-bias bug that initially made the model *look* like it won are in [MODEL_CARD.md](MODEL_CARD.md).

## Honest limitations

- ~5 Euro matches per player. Per-90 rates over that sample are noisy no matter how the model is built.
- The dataset contains no defensive metrics, so defenders are effectively rated on involvement and availability.
- Cross-role rating comparison is not claimed.
- Forecasting individual output across a competition change is hard; expect modest correlation, and treat a published modest number as more valuable than an impressive one with no validation story.

Not scouting advice. Not betting advice.

---

## Stack

Python · pandas · scikit-learn · scipy · soccerdata (FBref) · Fantasy Premier League API · React 18 · Vite
