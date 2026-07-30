"""
Shared data loading, role mapping and name-normalisation helpers.

Used by both `index_score.py` (descriptive Euro Impact Index) and
`forecast.py` (the supervised 2025-26 forecast model).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = MODELS_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

EURO_PARQUET = (
    DATA_DIR / "tournaments" / "outputs_euro" / "processed" / "player_agg_euro_2024.parquet"
)
ACTUALS_CSV = DATA_DIR / "actuals" / "epl_2025_26.csv"
OUTPUT_DIR = MODELS_DIR / "outputs"

# Minimum Euro 2024 minutes. Below this, per-90 rates are dominated by
# sampling noise.
#
# These two are deliberately different:
#
#   INDEX (270 = three full matches) is strict, because the index is a public
#   ranking. With a 180-minute floor a two-game cameo could top the list purely
#   on variance — Francisco Conceição ranked #1 forward on 182 minutes and an
#   xGI/90 of 1.25, which is noise, not the best forward at the tournament.
#
#   FORECAST (180) stays permissive, because there the binding constraint is
#   sample size: only 58 of 256 Euro players have a Premier League label at all.
#   Tightening the floor would trade a real statistical problem (n is too small)
#   for a cosmetic one (the ranking looks odd). Noisy features are also less
#   damaging than missing rows, since cross-validation prices them in.
MIN_EURO_MINUTES = 180
MIN_EURO_MINUTES_INDEX = 270

# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "GK": "GK",
    "DF": "DF", "FB": "DF", "WB": "DF", "CB": "DF", "RB": "DF", "LB": "DF",
    "MF": "MF", "DM": "MF", "CM": "MF", "AM": "MF", "WM": "MF",
    "FW": "FW", "CF": "FW", "WF": "FW", "SS": "FW", "ST": "FW", "LW": "FW", "RW": "FW",
}


def to_role(pos: object) -> str:
    """Collapse FBref position strings to one of GK / DF / MF / FW."""
    if not isinstance(pos, str) or not pos.strip():
        return "UNK"
    first = pos.split(",")[0].strip().upper()
    return _ROLE_MAP.get(first, "UNK")


# ---------------------------------------------------------------------------
# Name normalisation (for joining Euro data to EPL actuals)
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def normalise_name(name: object) -> str:
    """
    Lowercase, strip accents and punctuation, collapse whitespace.

    'Jurriën Timber'   -> 'jurrien timber'
    "Rodrigo 'Rodri'"  -> 'rodrigo rodri'
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = _PUNCT.sub(" ", ascii_only.lower())
    return _WS.sub(" ", cleaned).strip()


def name_keys(name: object) -> set[str]:
    """
    Candidate join keys for a player name, most specific first.

    Handles the common mismatch where one source has the full legal name
    ('Gabriel dos Santos Magalhaes') and the other a short name ('Gabriel').
    """
    full = normalise_name(name)
    if not full:
        return set()
    parts = full.split()
    keys = {full}
    if len(parts) >= 2:
        keys.add(f"{parts[0]} {parts[-1]}")   # first + last
        keys.add(parts[-1])                    # surname only
    if len(parts) == 1:
        keys.add(parts[0])
    return keys


# ---------------------------------------------------------------------------
# Euro 2024 feature loading
# ---------------------------------------------------------------------------

# Columns that are entirely NULL in the current parquet because the FBref
# column mapping in `data/tournaments/euros.py` silently failed for them.
KNOWN_EMPTY_COLUMNS = ["ga", "xg_xag", "ga_90", "xg_xag_90"]


def load_euro(min_minutes: int = MIN_EURO_MINUTES, verbose: bool = True) -> pd.DataFrame:
    """
    Load the Euro 2024 aggregate, add `role`, drop all-null columns and
    players with too few minutes to be meaningful.
    """
    if not EURO_PARQUET.exists():
        raise FileNotFoundError(
            f"Euro parquet not found at {EURO_PARQUET}. "
            "Run backend/data/tournaments/euros.py first."
        )

    df = pd.read_parquet(EURO_PARQUET)
    n_raw = len(df)

    # Drop columns that carry no information at all.
    empty = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty:
        if verbose:
            print(f"  dropping {len(empty)} all-null column(s): {', '.join(empty)}")
        df = df.drop(columns=empty)

    df["role"] = df["primary_pos"].apply(to_role)
    df = df[df["role"] != "UNK"].copy()

    # Cast the numeric columns out of pandas' nullable dtypes into plain float,
    # which sklearn handles without complaint.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df[df["minutes"].fillna(0) >= min_minutes].copy()

    if verbose:
        print(f"  loaded {n_raw} rows -> {before} with a known role "
              f"-> {len(df)} with >= {min_minutes} Euro minutes")

    df["join_key"] = df["player_name"].apply(normalise_name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def zscore_within(df: pd.DataFrame, value_col: str, group_col: str = "role") -> pd.Series:
    """Z-score `value_col` separately within each `group_col` bucket."""
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    return df.groupby(group_col, group_keys=False)[value_col].apply(_z)


def to_five_point(scores: pd.Series) -> pd.Series:
    """
    Map arbitrary scores onto a 1.0-5.0 scale using *percentile rank*.

    The original implementation clipped at the 5th/95th percentile, which
    collapsed the entire top decile onto exactly 5.0 (7 players tied at the
    ceiling). Percentile rank spreads players evenly and never ties the top.
    """
    pct = scores.rank(pct=True, method="average")
    return (1.0 + 4.0 * pct).round(2)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
