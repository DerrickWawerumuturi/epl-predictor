
# # **Euros 2024**

# ## This file main purpose is to get data on players who participated in the 2024 euros

# 1. Install all necessary imports for this project
# Which include:
# 
# * soccerdata → unified access to FBref, ClubElo, FiveThirtyEight SPI, Understat.
# * duckdb → local analytical database (you can query Parquet files fast).
# * polars → faster DataFrame engine for cleaning/aggregation.
# * requests + beautifulsoup4 + lxml → fallback scrapers for tournaments soccerdata doesn’t expose yet.
# * tqdm → progress bars while fetching.
# 

from soccerdata import FBref
from pathlib import Path
import pandas as pd
from typing import Dict
import os


# 2. Storage for the output
print(os.getcwd())

BASE = Path.cwd()
OUT_RAW = BASE / "outputs_euro" / "raw"
OUT_STAGING = BASE / "outputs_euro" / "staging"
OUT_PROCESSED = BASE / "outputs_euro" / "processed"
for p in (OUT_RAW, OUT_STAGING, OUT_PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

# Step 1: Data retrieval

# i.The first step is where do I get the data related to the euros 2024, I chose soccerdata as my source of data, due to its api, connection to other football data
# sites and its simplicity to use


euros_24 = FBref(
    leagues='INT-European Championship',
    seasons='2024'
)


# ii.Since it's raw data,I am not aiming to collect all data for the model, just specific columns that tie in to the player

player_info: Dict[str, str] = \
{
        # General info
        "league": "league",
        "season": "tournament_year",
        "team": "team_name",
        "player": "player_name",
        "nation": "player_country",
        "pos": "pos_raw",
        "age": "age",
        "born": "birth_year",

        # Playing time
        "playing_time_mp": "mp",
        "playing_time_starts": "starts",
        "playing_time_min": "minutes",
        "playing_time_90s": "nineties",

        # Totals (performance/xg blocks)
        "performance_gls": "gls",
        "performance_ast": "ast",
        "performance_gplus_a": "ga",
        "performance_g_pk": "g_pk",

        "expected_xg": "xg",
        "expected_xag": "xag",
        "expected_xgplusxag": "xg_xag",
        "expected_npxg": "npxg",
        "expected_npxgplusxag": "npxg_xag",

        # Per 90 (performance block)
        "per_90_minutes_gls": "gls_90",
        "per_90_minutes_ast": "ast_90",
        "per_90_minutes_gplus_a": "ga_90",
        "per_90_minutes_g_pk": "g_pk_90",
}


# Problem:
# The data is not a typical dataframe, but the columns come as multiindex columns, which basically mean that the have two levels
# one for the column name and the actual stat, that why the columnns in the data frame have:
# *     ('Playing Time', 'Min')
# *     ('Playing Time', 'Starts')
# *     ('Per 90 Minutes', 'Ast')
# 
# So we need to flatten this to sort of join the to be:
# *     'playing_time_min'
# *     'playing_time_starts'
# *     'per_90_minutes_ast'
# 
# Solution:
# We will create a function that does two things
# 1. Joins the two sections\levels.
# 2. Also standardizes the names to Playing Time -> playing_time...

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten season_stats MultiIndex columns to snake_case strings."""
    def norm(x):
        x = (
            str(x)
             .strip()
             .lower()
             .replace(' ', '_')
             .replace('%', 'pct')
             .replace('-', '_')
             .replace('+', 'plus')
             .replace('/', '_'))
        return x
    if isinstance(df.columns, pd.MultiIndex):
         df = df.copy()
         # a(Playing Time, min)
         df.columns = ['_'.join([norm(a) for a in tup if a and str(a) != 'nan']).strip('_')
                       for tup in df.columns.tolist()]
         # df.columns = [playing_time_min,...]
   # no multiindex
    else:
        df = df.copy()
        df.columns = [norm(a) for a in df.columns]

    return df.reset_index()


# iii.Now we need to create a function that goes through the **euros_2024** data  which is raw data from the euros and get the players_info
# 
# What I am thinking is to store the data into a pd Dataframe after retrieval to represent the data a table for easier reading and structure.

# # Helper functions

def _pick_player_pos(pos_raw: str) -> str:
    """
    Picks primary pos for a player, if a player has more than one, the first one is picked
    :param pos_raw: str
    :return: string
    """
    if not isinstance(pos_raw, str) or not pos_raw:
        return pd.NA
    return pos_raw.split(',')[0].strip()



def _safe_div(numer, denom):
    """Elementwise safe division that returns NA when denom==0 or NA."""
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return numer.divide(denom).where(denom != 0)

# ## **Main Pipeline**

def build_player_agg(year: str = "2024") -> pd.DataFrame:
    """
    Build one-row-per-player aggregate for EURO {year}.
    Saves raw -> staging -> processed parquet files.
    """
    # 1) fetch raw standard table
    fb = FBref(leagues=["INT-European Championship"], seasons=[year])
    season_stats = fb.read_player_season_stats(stat_type="standard")

    raw_path = OUT_RAW / f"player_standard_euro_{year}.parquet"
    season_stats.to_pickle(raw_path.with_suffix(".pkl"))

    # 2) flatten & stage
    stats = _flatten_cols(season_stats)
    stats_path = OUT_STAGING / f"player_standard_euro_{year}.parquet"
    stats.to_parquet(stats_path, engine='fastparquet')

    # 3) ensure all mapped columns exist
    missing = [k for k in player_info.keys() if k not in stats.columns]
    for m in missing:
        stats[m] = pd.NA

    # 4) rename to canonical schema
    agg = stats[list(player_info.keys())].rename(columns=player_info).copy()
    agg["tournament"] = "EURO"

    # 5) coerce types for numerics (prevents string math weirdness)
    numeric_cols = [
        "age","birth_year","mp","starts","minutes","nineties",
        "gls","ast","ga","g_pk","xg","xag","xg_xag","npxg","npxg_xag",
        "gls_90","ast_90","ga_90","g_pk_90",
    ]
    for c in numeric_cols:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce")

    # tournament year as int for clean grouping
    agg["tournament_year"] = pd.to_numeric(agg["tournament_year"], errors="coerce").astype("Int64")

    # 6) derivatives
    agg["primary_pos"] = agg["pos_raw"].apply(_pick_player_pos)

    n90 = agg["nineties"]
    for src, dst in [
        ("xg", "xg_90"), ("xag", "xag_90"), ("xg_xag", "xg_xag_90"),
        ("npxg", "npxg_90"), ("npxg_xag", "npxg_xag_90"),
    ]:
        if dst not in agg.columns:
            agg[dst] = _safe_div(agg[src], n90)

    # usage
    team_minutes = agg.groupby(["team_name", "tournament_year"])["minutes"].transform("sum")
    agg["minutes_share"] = _safe_div(agg["minutes"], team_minutes)
    agg["starter_rate"] = _safe_div(agg["starts"], agg["mp"])

    # 7) QA guards
    agg = agg[agg["league"].eq("INT-European Championship")]
    agg = agg.drop_duplicates(subset=["player_name", "player_country", "tournament_year"])

    agg["minutes"] = agg["minutes"].clip(lower=0, upper=720)
    agg["minutes_share"] = agg["minutes_share"].clip(lower=0, upper=1.25)

    # 8) final column order
    cols = [
        "player_name","player_country","team_name","tournament","tournament_year",
        "age","birth_year","pos_raw","primary_pos",
        "mp","starts","minutes","nineties","minutes_share","starter_rate",
        "gls","ast","ga","xg","xag","xg_xag","npxg","npxg_xag",
        "gls_90","ast_90","ga_90","xg_90","xag_90","xg_xag_90","npxg_90","npxg_xag_90",
    ]
    for c in cols:
        if c not in agg.columns:
            agg[c] = pd.NA
    agg = agg[cols]

    # 9) save processed
    out_path = OUT_PROCESSED / f"player_agg_euro_{year}.parquet"
    agg.to_parquet(out_path, engine='fastparquet')

    # tiny sanity prints (optional)
    print(f"Saved raw      → {raw_path}")
    print(f"Saved staging  → {stats_path}")
    print(f"Saved processed→ {out_path} ({len(agg)} rows, {len(agg.columns)} cols)")
    return agg

# # Run the final file


if __name__ == "__main__":
    YEAR = '2024'
    df_out = build_player_agg(YEAR)
    print(f"Saved {len(df_out)} rows → {OUT_PROCESSED / f'player_agg_euro_{YEAR}.parquet'}")