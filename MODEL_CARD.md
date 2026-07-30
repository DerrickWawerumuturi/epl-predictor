# Model Card — XGauges

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

### Results

`n = 47` outfield players (58 of 256 Euro players matched to a Premier League record, 52 with ≥450 PL minutes, 5 goalkeepers dropped). 5-fold CV, out-of-fold predictions, ± is the fold standard deviation.

| Model | MAE | RMSE | R² | Spearman ρ |
|---|---|---|---|---|
| Mean baseline | 0.1524 ± 0.0375 | 0.1827 ± 0.0361 | −0.076 ± 0.163 | n/a |
| **Euro minutes only** | **0.1116 ± 0.0329** | **0.1448 ± 0.0394** | **+0.324 ± 0.184** | +0.525 |
| Ridge | 0.1179 ± 0.0277 | 0.1485 ± 0.0326 | +0.289 ± 0.196 | +0.494 |
| Random Forest | 0.1228 ± 0.0303 | 0.1533 ± 0.0359 | +0.242 ± 0.254 | +0.540 |
| Gradient Boosting | 0.1406 ± 0.0221 | 0.1804 ± 0.0338 | −0.048 ± 0.550 | +0.438 |

Random Forest Spearman ρ = **+0.540**, bootstrap 95% CI **[+0.285, +0.715]**.

### What this actually says

**Euro 2024 form carries real rank information about 2025–26 Premier League output.** ρ ≈ 0.54 with a bootstrap CI that excludes zero is a genuine signal, and permutation importance puts the expected-goal metrics (`npxg_xag_90`, `xg_90`, `xag_90`, `npxg_90`) at the top — which is what should carry information if the signal is real rather than an artefact.

**But the trained models do not beat a one-feature baseline.** A Ridge regression on Euro *minutes alone* wins on MAE, RMSE and R². Random Forest edges it on Spearman by 0.015, which is meaningless next to a CI 0.43 wide. The honest conclusion is:

> On this sample, none of the trained models is distinguishably better than "how many minutes did he play at the Euros." Most of the recoverable signal appears to be *selection* — good players play more — rather than anything the per-90 metrics add on top.

That is a negative result, and it is reported rather than buried. `forecast.py` prints `WORSE than counting minutes` automatically; the baseline exists precisely so this cannot be missed.

**The binding constraint is `n = 47`,** not model choice. No amount of hyperparameter tuning fixes 47 rows. Tuning against these folds would just overfit the validation set.

### The survivorship-bias bug, and why it mattered

The first run of this backtest pulled labels from the **live** FPL API and produced *better*-looking model numbers: Random Forest MAE 0.1173 vs. a minutes baseline of 0.1267 — i.e. the model appeared to win.

It was wrong. FPL rolls its API over to the next season shortly after the final matchday: relegated clubs are deleted and promoted clubs appear with empty squads. The pull returned **19 clubs**, including Coventry City and Ipswich Town (who never played in 2025–26) and missing Burnley, West Ham and Wolves entirely. Every player at a relegated club was silently absent from the labels — and relegated squads skew toward low output, so the target distribution was truncated at the hard-to-predict end.

Reading the same season from a frozen archive restored all 20 clubs, took the matched sample from 43 to 58, and **reversed the conclusion**: the minutes baseline went from losing to winning.

Two lessons worth more than the model itself:

1. A club count is a one-line sanity check that caught a bias which flattered the result by ~10% MAE. `fetch_actuals.py` now asserts 20 clubs and warns loudly otherwise.
2. Removing a bias made the results *worse*. That is the normal direction, and it is why a result that improves after a data fix deserves suspicion.

### What would actually move this

In rough order of expected value:

1. **More tournaments.** Copa América 2024, Euro 2020, World Cup 2022 — same feature schema, same label join. This is the only change that addresses `n = 47`, and the repo already has a `copa_america.ipynb` stub.
2. **A nested test of incremental value.** Fit minutes-only, then minutes + expected-goal metrics, and test whether the added features improve out-of-fold error beyond noise. That answers the actual question ("do the per-90 metrics add anything?") directly instead of by leaderboard.
3. **Predict a rank-transformed target,** since the product is a ranking and the raw label is right-skewed.
4. **Per-90 uncertainty weighting** — a player with 200 Euro minutes should not carry the same weight as one with 600.

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
