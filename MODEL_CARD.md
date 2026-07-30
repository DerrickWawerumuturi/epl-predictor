# Model Card — XG-Predictor

Two separate things live in this repo. Keeping them separate is the point.

| | Euro Impact Index | 2025–26 Forecast |
|---|---|---|
| File | `backend/models/index_score.py` | `backend/models/forecast.py` |
| Kind | Transparent weighted index | Supervised regression |
| Trained? | No | Yes |
| Claim | Describes Euro 2024 | Forecasts Premier League output |
| Validated? | Nothing to validate | 5-fold CV vs. two baselines |

---

## 1. Euro Impact Index

**What it does.** Scores every Euro 2024 player with ≥180 tournament minutes and ranks them *within their own position*.

**Definition.** For outfield players, all components z-scored within role:

```
0.55 · z(npxg_xag_90)     expected goal involvement per 90
0.25 · z(involvement_90)  actual goals + assists per 90
0.20 · z(minutes_share)   share of team minutes
```

Goalkeepers are scored on availability alone — xG-based attacking metrics are meaningless for them — and are flagged in the output.

**Rating scale.** Percentile rank within role, mapped to 1.0–5.0.

**What it is not.** It is not a prediction, and there is no accuracy number to quote. It is a published formula; you can disagree with the weights and recompute.

**Known limits.**

- Cross-role comparison is not claimed. A 4.8 defender and a 4.8 forward are each strong *for their position*; the numbers are not on a common scale.
- ~5 Euro matches per player. Per-90 rates over that sample are noisy, which is why availability is weighted at all.
- Weights are judgement, not fitted.

---

## 2. 2025–26 Forecast

**Task.** `X` = Euro 2024 per-90 metrics, playing time and age (summer 2024). `y` = actual 2025–26 Premier League expected goal involvement per 90 (`epl_xgi_90`, from the public Fantasy Premier League API).

Features and label come from different competitions in different seasons, so target leakage is structurally impossible.

**Protocol.**

- 5-fold shuffled `KFold`, out-of-fold predictions only.
- Reported: MAE, RMSE, R², Spearman ρ, each with fold standard deviation.
- Spearman is the headline metric — the product is a ranking, so rank agreement matters more than squared error.
- Two baselines must be beaten or there is no result: **mean predictor** and **Euro minutes only**.
- Feature attribution via **permutation importance**, not `feature_importances_` (which is biased toward high-cardinality continuous features).
- Goalkeepers are dropped; an attacking label cannot describe them, and leaving them in at ~0 would flatter the error metrics.
- Minimum 450 Premier League minutes for a label to count.

**To reproduce:**

```bash
python backend/data/fetch_actuals.py   # needs internet; writes data/actuals/
python backend/models/forecast.py
```

`forecast.py` refuses to run without real labels rather than substituting a synthetic target.

**Expect modest numbers, and do not be alarmed.** Forecasting individual football output across a competition change from a five-match tournament is genuinely hard. A Spearman ρ in the 0.2–0.4 range with a clear win over both baselines would be a real result. A published ρ of 0.3 is worth more than an unpublished R² of 0.9 that came from leakage.

---

## 3. What was wrong before, and what changed

| Issue | Before | Now |
|---|---|---|
| **Target leakage** | Target for FW was `z(gls_90 − xg_90)` while `gls_90` and `xg_90` were both features. Target for everyone else was `z(starter_rate)` — also a feature. The model recovered an arithmetic identity. | Index makes no predictive claim. Forecast uses a label from a different season and competition. |
| **Unsupported claim** | README promised 2025–26 predictions; no 2025–26 data existed anywhere in the pipeline. | Real 2025–26 labels fetched from the FPL API. |
| **Incoherent target** | Forwards scored on finishing luck, others on minutes, then pooled into one global ranking. | One consistent definition; ranking is within position. |
| **Noise target** | `gls_90 − xg_90` over ~5 games is close to pure noise (finishing overperformance is famously mean-reverting). | Volume/quality signal (`npxg_xag_90`) instead of a luck signal. |
| **Wrong metric** | `rmse = mean_squared_error(...)` — that is MSE, and it was used to select the best model. | `np.sqrt(mean_squared_error(...))`. |
| **Unreliable validation** | Single 80/10/10 split on a small frame → val set of a handful of players. | 5-fold CV with reported spread. |
| **No baseline** | None. | Mean predictor and minutes-only baseline. |
| **Rating compression** | p5–p95 clipping tied the top 7 players at exactly 5.0. | Percentile rank; no ceiling pile-up. |
| **Silent data loss** | `ga`, `xg_xag`, `ga_90`, `xg_xag_90` were 100% null from a broken FBref column mapping. `ga_90` was in the feature list. | Detected and dropped explicitly at load; `involvement_90` recomputed from its parts. |
| **No minutes floor** | Players with 1 minute produced per-90 rates. | ≥180 Euro minutes to be scored; ≥450 PL minutes to be labelled. |
| **Club/age missing** | Output listed national teams, not clubs, despite the README claiming club and age. | Club comes from the FPL join. |

---

## 4. Ethical / practical notes

- Data sources are a public API (FPL) and FBref via `soccerdata`. No scraping against a site's terms.
- Player ratings here are a hobby analysis, not scouting advice, and definitely not betting advice.
- The index will rate a defender highly for being available and involved. That is what it measures — not defensive quality, which is absent from this dataset entirely.
