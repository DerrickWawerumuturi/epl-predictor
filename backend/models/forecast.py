"""
The actual supervised task: forecast 2025-26 Premier League output from
Euro 2024 form.

    X = Euro 2024 per-90 metrics, playing time, age   (summer 2024)
    y = actual 2025-26 EPL expected goal involvement per 90   (May 2026)

Features and label come from different competitions in different seasons, so
target leakage is structurally impossible — unlike the original pipeline, where
the label was arithmetic on the features.

What this reports, and why:

  * 5-fold cross-validated MAE / RMSE / R^2 with standard deviations.
    A single 80/10/10 split on ~100 players gives a test set of ~10 people;
    the resulting numbers are noise. CV is the minimum defensible protocol.
  * Spearman rank correlation. The product is a *ranking*, so rank agreement
    matters more than squared error.
  * Two baselines that must be beaten for any result to mean anything:
      - mean predictor (DummyRegressor)
      - Euro minutes only (does "he played a lot" explain everything?)
  * Permutation importance, not RandomForest's `feature_importances_`, which
    is biased toward high-cardinality continuous features.

Run:
    python backend/data/fetch_actuals.py     # once, needs internet
    python backend/models/forecast.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    ACTUALS_CSV,
    MIN_EURO_MINUTES,
    ensure_output_dir,
    load_euro,
    name_keys,
    normalise_name,
    to_five_point,
)

# A player needs a real Premier League sample for the label to be meaningful.
MIN_EPL_MINUTES = 450

LABEL = "epl_xgi_90"

# Euro 2024 inputs only. Nothing here is derived from the label.
NUMERIC_FEATURES = [
    "age",
    "minutes",
    "nineties",
    "mp",
    "starts",
    "minutes_share",
    "starter_rate",
    "gls_90",
    "ast_90",
    "xg_90",
    "xag_90",
    "npxg_90",
    "npxg_xag_90",
]
CATEGORICAL_FEATURES = ["role"]

RANDOM_STATE = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def join_euro_to_actuals(euro: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Match Euro 2024 players to their 2025-26 Premier League records.

    Name matching is the hard part: FBref uses full legal names, FPL uses a mix
    of full and short names. We build a lookup keyed on several normalised
    variants (full name, first+last, surname) and take the first hit, breaking
    ties by minutes played so we prefer the established player over a namesake
    academy prospect.
    """
    actuals = actuals.sort_values("epl_minutes", ascending=False).copy()

    lookup: dict[str, int] = {}
    for idx, row in actuals.iterrows():
        candidates = name_keys(row["player_name"]) | name_keys(row.get("web_name"))
        for key in candidates:
            lookup.setdefault(key, idx)   # first (highest-minutes) wins

    matched_idx: list[int | None] = []
    for name in euro["player_name"]:
        hit = None
        for key in sorted(name_keys(name), key=len, reverse=True):
            if key in lookup:
                hit = lookup[key]
                break
        matched_idx.append(hit)

    euro = euro.copy()
    euro["_match"] = matched_idx
    joined = euro[euro["_match"].notna()].copy()
    joined["_match"] = joined["_match"].astype(int)

    actual_cols = [c for c in actuals.columns if c.startswith("epl_")] + ["club"]
    for col in actual_cols:
        joined[col] = actuals.loc[joined["_match"], col].to_numpy()

    return joined.drop(columns=["_match"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def build_preprocessor(numeric: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, CATEGORICAL_FEATURES)]
    )


def build_models(numeric: list[str]) -> dict[str, Pipeline]:
    pre = lambda: build_preprocessor(numeric)  # noqa: E731 - fresh transformer per model
    return {
        "baseline_mean": Pipeline(
            [("pre", pre()), ("model", DummyRegressor(strategy="mean"))]
        ),
        "baseline_euro_minutes": Pipeline(
            [
                ("pre", build_preprocessor(["minutes"])),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ]
        ),
        "ridge": Pipeline([("pre", pre()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        "random_forest": Pipeline(
            [
                ("pre", pre()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "grad_boost": Pipeline(
            [
                ("pre", pre()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # A constant predictor (the mean baseline) has no rank information, so
    # Spearman is undefined. cross_val_predict still produces tiny fold-to-fold
    # variation in the fitted mean, which yields a spurious non-zero rho.
    # Report NaN rather than a number a reader might take seriously.
    if np.std(y_pred) < 1e-9 or len(np.unique(np.round(y_pred, 9))) <= N_SPLITS:
        rho, p_value = float("nan"), float("nan")
    else:
        rho, p_value = spearmanr(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # NOTE: the original code reported mean_squared_error and called it RMSE.
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho),
        "spearman_p": float(p_value),
    }


def bootstrap_spearman(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for Spearman rho.

    With a modelling set this small, a point estimate of rho is close to
    meaningless on its own. The interval is the honest statement.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 8 or np.std(y_pred) < 1e-9:
        return float("nan"), float("nan")

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y_pred[idx]) < 1e-9 or np.std(y_true[idx]) < 1e-9:
            continue
        rho, _ = spearmanr(y_true[idx], y_pred[idx])
        if np.isfinite(rho):
            stats.append(rho)

    if len(stats) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def cross_validate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Out-of-fold predictions plus per-fold spread."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X, y, cv=kf, n_jobs=1)

    fold_mae, fold_rmse, fold_r2 = [], [], []
    for _, test_idx in kf.split(X):
        yt, yp = y.iloc[test_idx].to_numpy(), oof[test_idx]
        fold_mae.append(mean_absolute_error(yt, yp))
        fold_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        fold_r2.append(r2_score(yt, yp))

    overall = score_predictions(y.to_numpy(), oof)
    overall.update(
        {
            "mae_std": float(np.std(fold_mae)),
            "rmse_std": float(np.std(fold_rmse)),
            "r2_std": float(np.std(fold_r2)),
        }
    )
    return {"metrics": overall, "oof": oof}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("2025-26 Premier League forecast from Euro 2024 form")
    print("=" * 64)

    if not ACTUALS_CSV.exists():
        raise SystemExit(
            f"\nMissing labels: {ACTUALS_CSV}\n"
            "Run `python backend/data/fetch_actuals.py` first (needs internet).\n"
            "Refusing to fabricate a target — that was the original bug."
        )

    euro = load_euro(min_minutes=MIN_EURO_MINUTES)
    actuals = pd.read_csv(ACTUALS_CSV)
    print(f"  {len(actuals)} Premier League players with 2025-26 minutes")

    n_clubs = actuals["club"].nunique()
    if n_clubs != 20:
        print(f"\n  *** WARNING: labels cover {n_clubs} clubs, not 20. ***")
        print("  Re-run backend/data/fetch_actuals.py WITHOUT --live. The live FPL")
        print("  API deletes relegated clubs once the season rolls over, which makes")
        print("  the label set survivorship-biased and the error optimistic.\n")

    joined = join_euro_to_actuals(euro, actuals)
    print(f"  matched {len(joined)} / {len(euro)} Euro players to a PL record")

    unmatched = euro[~euro["player_name"].isin(joined["player_name"])]
    if len(unmatched):
        sample = ", ".join(unmatched.nlargest(8, "minutes")["player_name"])
        print(f"  unmatched (top 8 by Euro minutes): {sample}")
        print("  ^ check these by hand: most are genuinely not in the PL, but any")
        print("    that ARE indicate a name-normalisation gap in common.name_keys().")

    data = joined[joined["epl_minutes"] >= MIN_EPL_MINUTES].copy()
    data = data[data[LABEL].notna()].copy()
    print(f"  {len(data)} with >= {MIN_EPL_MINUTES} PL minutes -> modelling set")

    # Goalkeepers cannot be forecast on an attacking label; drop them explicitly
    # rather than letting them sit at zero and flatter the error metrics.
    n_gk = int((data["role"] == "GK").sum())
    data = data[data["role"] != "GK"].copy()
    print(f"  dropped {n_gk} goalkeeper(s) — attacking label is meaningless for GK")

    if len(data) < 40:
        print(
            f"\nWARNING: only {len(data)} rows. Metrics will be extremely noisy; "
            "treat everything below as indicative, not conclusive."
        )

    numeric = [c for c in NUMERIC_FEATURES if c in data.columns and data[c].notna().any()]
    X = data[numeric + CATEGORICAL_FEATURES]
    y = data[LABEL].astype(float)

    print(f"\n  features: {len(numeric)} numeric + role")
    print(f"  label: {LABEL}   mean={y.mean():.3f}  sd={y.std():.3f}\n")

    results: dict[str, dict] = {}
    print(f"{'model':24s} {'MAE':>16s} {'RMSE':>16s} {'R2':>16s} {'rho':>8s}")
    print("-" * 84)
    for name, pipe in build_models(numeric).items():
        cv = cross_validate(pipe, X, y)
        m = cv["metrics"]
        results[name] = cv
        print(
            f"{name:24s} "
            f"{m['mae']:.4f}±{m['mae_std']:.4f}  "
            f"{m['rmse']:.4f}±{m['rmse_std']:.4f}  "
            f"{m['r2']:+.3f}±{m['r2_std']:.3f}  "
            f"{m['spearman']:+.3f}"
        )

    real = {k: v for k, v in results.items() if not k.startswith("baseline_")}
    best_name = max(real, key=lambda k: real[k]["metrics"]["spearman"])
    best = results[best_name]
    dummy = results["baseline_mean"]["metrics"]
    minutes_only = results["baseline_euro_minutes"]["metrics"]

    print("\n" + "=" * 64)
    print(f"Best by rank correlation: {best_name}")

    lo, hi = bootstrap_spearman(y.to_numpy(), best["oof"])
    ci_note = f"  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else ""
    print(f"  Spearman rho = {best['metrics']['spearman']:+.3f}{ci_note}")
    if np.isfinite(lo) and lo <= 0 <= hi:
        print("  CI spans zero: cannot rule out that the ranking is no better than chance.")

    beat_mae = dummy["mae"] - best["metrics"]["mae"]
    print(f"  MAE vs mean-baseline:    {beat_mae:+.4f} "
          f"({'better' if beat_mae > 0 else 'WORSE — no usable signal'})")

    # The interesting comparison is not against the mean — it is against the
    # cheapest sensible heuristic. If the model cannot clear "he played a lot",
    # it has added nothing worth deploying.
    beat_minutes = minutes_only["mae"] - best["metrics"]["mae"]
    margin_within_noise = abs(beat_minutes) < best["metrics"]["mae_std"]
    print(f"  MAE vs minutes-baseline: {beat_minutes:+.4f} "
          f"({'better' if beat_minutes > 0 else 'WORSE than counting minutes'})")
    if margin_within_noise:
        print(f"    ...but that margin is smaller than the fold spread "
              f"(±{best['metrics']['mae_std']:.4f}), so the two are not "
              "reliably distinguishable.")

    if best["metrics"]["r2"] < 0:
        print("  R2 is negative: the model is worse than predicting the average.")
    if np.isfinite(best["metrics"]["spearman_p"]) and best["metrics"]["spearman_p"] > 0.05:
        print(f"  Spearman p={best['metrics']['spearman_p']:.3f} — rank agreement "
              "is not statistically significant.")

    # --- permutation importance on a full refit -------------------------
    pipe = build_models(numeric)[best_name]
    pipe.fit(X, y)
    perm = permutation_importance(
        pipe, X, y, n_repeats=30, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error"
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nPermutation importance (top 8):")
    print(importance.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- outputs --------------------------------------------------------
    out_dir = ensure_output_dir()

    data = data.reset_index(drop=True)
    data["predicted_xgi_90"] = best["oof"]      # honest out-of-fold predictions
    data["actual_xgi_90"] = y.to_numpy()
    data["abs_error"] = (data["predicted_xgi_90"] - data["actual_xgi_90"]).abs()
    data["rating_5pt"] = to_five_point(data["predicted_xgi_90"])
    data["predicted_rank"] = data["predicted_xgi_90"].rank(ascending=False, method="min").astype(int)
    data["actual_rank"] = data["actual_xgi_90"].rank(ascending=False, method="min").astype(int)
    data["rank_error"] = data["predicted_rank"] - data["actual_rank"]

    keep = [
        "player_name", "team_name", "club", "role", "age",
        "minutes", "npxg_xag_90",
        "epl_minutes", "predicted_xgi_90", "actual_xgi_90", "abs_error",
        "predicted_rank", "actual_rank", "rank_error", "rating_5pt",
    ]
    keep = [c for c in keep if c in data.columns]
    pred_path = out_dir / "forecast_predictions.csv"
    data.sort_values("predicted_xgi_90", ascending=False)[keep].to_csv(pred_path, index=False)

    metrics_blob = {
        "task": "predict 2025-26 EPL expected goal involvement per 90 from Euro 2024 form",
        "label": LABEL,
        "n_players": int(len(data)),
        "min_euro_minutes": MIN_EURO_MINUTES,
        "min_epl_minutes": MIN_EPL_MINUTES,
        "cv": f"{N_SPLITS}-fold KFold (shuffled, seed {RANDOM_STATE})",
        "best_model": best_name,
        "models": {k: v["metrics"] for k, v in results.items()},
        "permutation_importance": importance.to_dict(orient="records"),
        "goalkeepers_excluded": n_gk,
    }
    (out_dir / "forecast_metrics.json").write_text(json.dumps(metrics_blob, indent=2))
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\n  wrote {pred_path}")
    print(f"  wrote {out_dir / 'forecast_metrics.json'}")

    # Label the direction of each miss. The previous version called all of these
    # "over-rated", which was wrong for roughly half of them.
    data["direction"] = np.where(
        data["predicted_xgi_90"] > data["actual_xgi_90"], "over-rated", "under-rated"
    )
    print("\nLargest absolute errors:")
    worst = data.nlargest(5, "abs_error")[
        ["player_name", "role", "predicted_xgi_90", "actual_xgi_90", "abs_error", "direction"]
    ]
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
