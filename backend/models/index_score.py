"""
Euro Impact Index — a transparent, *descriptive* score for Euro 2024 players.

WHY THIS IS NOT A MODEL
-----------------------
The original pipeline trained a regressor whose target was computed from the
same Euro 2024 columns that were fed in as features (target for forwards was
`z(gls_90 - xg_90)` while `gls_90` and `xg_90` were both in the feature list;
target for everyone else was `z(starter_rate)` while `starter_rate` was also a
feature). The model was therefore recovering an arithmetic identity, not
learning anything — classic target leakage, and it inflated every metric.

You cannot fix that by dropping a couple of columns: *any* target derived from
Euro 2024 and predicted from Euro 2024 is circular. So this file stops
pretending. It computes a weighted index with published weights. No training,
no train/test split, no R^2 — because there is nothing to validate.

The genuine supervised task lives in `forecast.py`: Euro 2024 as features,
*actual* 2025-26 Premier League output as the label.

INDEX DEFINITION
----------------
For outfield players (FW / MF / DF), all components z-scored within role:

    0.55 * z(npxg_xag_90)     expected goal involvement per 90 (quality)
    0.25 * z(involvement_90)  actual goals + assists per 90 (output)
    0.20 * z(minutes_share)   share of team minutes played (trust)

Goalkeepers have no meaningful xG-based attacking metrics, so they are scored
on availability alone and flagged as such. Cross-role comparison is NOT
claimed: ratings are percentiles *within* position.
"""

from __future__ import annotations

import json

import pandas as pd

from common import (
    MIN_EURO_MINUTES,
    ensure_output_dir,
    load_euro,
    to_five_point,
    zscore_within,
)

WEIGHTS = {
    "npxg_xag_90": 0.55,
    "involvement_90": 0.25,
    "minutes_share": 0.20,
}


def build_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Actual goal involvement per 90. `ga_90` is all-null in the parquet
    # (broken FBref column mapping), so recompute it from its parts.
    df["involvement_90"] = df["gls_90"].fillna(0) + df["ast_90"].fillna(0)

    # Median-impute the expected-metric columns, which are missing for ~14
    # players, so a single NaN does not knock a player out of the ranking.
    for col in ("npxg_xag_90", "minutes_share"):
        df[col] = df[col].fillna(df[col].median())

    outfield = df["role"].isin(["FW", "MF", "DF"])

    df["index_score"] = 0.0

    # --- outfield: weighted sum of within-role z-scores -------------------
    out = df[outfield].copy()
    if len(out):
        score = 0.0
        for col, weight in WEIGHTS.items():
            out[f"z_{col}"] = zscore_within(out, col, "role")
            score = score + weight * out[f"z_{col}"]
        out["index_score"] = score
        df.loc[out.index, "index_score"] = out["index_score"]
        for col in WEIGHTS:
            df.loc[out.index, f"z_{col}"] = out[f"z_{col}"]

    # --- goalkeepers: availability only ----------------------------------
    gks = df[~outfield].copy()
    if len(gks):
        gks["index_score"] = zscore_within(gks, "minutes_share", "role")
        df.loc[gks.index, "index_score"] = gks["index_score"]

    df["scored_on"] = "attacking + availability"
    df.loc[~outfield, "scored_on"] = "availability only (no xG metrics for GK)"

    # --- rating: percentile WITHIN role, mapped to 1.0-5.0 ---------------
    df["rating_5pt"] = (
        df.groupby("role", group_keys=False)["index_score"].apply(to_five_point)
    )

    df["rank_in_role"] = (
        df.groupby("role")["index_score"].rank(ascending=False, method="min").astype(int)
    )

    return df.sort_values("index_score", ascending=False).reset_index(drop=True)


def main() -> None:
    print("Euro Impact Index")
    print("=" * 60)
    df = load_euro(min_minutes=MIN_EURO_MINUTES)
    ranked = build_index(df)

    out_dir = ensure_output_dir()

    cols = [
        "player_name", "team_name", "role", "age", "minutes", "minutes_share",
        "npxg_xag_90", "involvement_90", "index_score", "rating_5pt",
        "rank_in_role", "scored_on",
    ]
    cols = [c for c in cols if c in ranked.columns]
    csv_path = out_dir / "euro_impact_index.csv"
    ranked[cols].to_csv(csv_path, index=False)

    meta = {
        "kind": "descriptive index (not a trained model)",
        "weights": WEIGHTS,
        "min_euro_minutes": MIN_EURO_MINUTES,
        "n_players": int(len(ranked)),
        "by_role": {k: int(v) for k, v in ranked["role"].value_counts().items()},
        "rating_scale": "percentile rank within role, mapped to 1.0-5.0",
        "caveat": (
            "Cross-role comparison is not claimed. Goalkeepers are scored on "
            "availability only. This index describes Euro 2024; it does not "
            "forecast the Premier League season — see forecast.py for that."
        ),
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  wrote {csv_path}")
    print(f"  {len(ranked)} players  |  by role: {meta['by_role']}\n")

    print("Top 10 by index (within-role percentile shown as rating):")
    show = ranked.head(10)[["player_name", "team_name", "role", "index_score", "rating_5pt"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nTop 3 per role:")
    for role in ["FW", "MF", "DF", "GK"]:
        sub = ranked[ranked["role"] == role].head(3)
        if not len(sub):
            continue
        names = ", ".join(
            f"{r.player_name} ({r.rating_5pt:.1f})" for r in sub.itertuples()
        )
        print(f"  {role}: {names}")


if __name__ == "__main__":
    main()
