"""
Fetch ACTUAL 2025-26 Premier League player output — the labels for forecast.py.

TWO SOURCES, AND WHY THE DEFAULT MATTERS
----------------------------------------
`--live` reads the Fantasy Premier League API directly. Do NOT use it for a
finished season. FPL rolls the API over to the next campaign shortly after the
final matchday: relegated clubs are deleted outright and promoted clubs appear
with empty squads. Pulling 2025-26 from the live API in July 2026 returned 19
clubs — including Coventry City and Ipswich Town, who never played in 2025-26,
and missing Burnley, West Ham and Wolves entirely.

That is survivorship bias, not a rounding error: every player at a relegated
club vanishes from the labels, and relegated squads skew toward the low end of
the output distribution. Any model trained on it is fitted to a truncated
sample and its error is optimistic.

The default therefore reads a *frozen* archive of the completed season
(vaastav/Fantasy-Premier-League, the standard public FPL data mirror), which
preserves all 20 clubs.

Run:
    python backend/data/fetch_actuals.py            # frozen archive (correct)
    python backend/data/fetch_actuals.py --live     # in-progress season only

Writes backend/data/actuals/epl_2025_26.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SEASON = "2025-26"
EXPECTED_CLUBS = 20

ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/" + SEASON
)
ARCHIVE_PLAYERS = f"{ARCHIVE_BASE}/players_raw.csv"
ARCHIVE_TEAMS = f"{ARCHIVE_BASE}/teams.csv"
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "actuals"
OUT_CSV = OUT_DIR / f"epl_{SEASON.replace('-', '_')}.csv"

# FPL position ids -> our role buckets. FPL has no separate winger class;
# wingers land in MID, which is close enough for a label join.
POSITION_MAP = {1: "GK", 2: "DF", 3: "MF", 4: "FW", 5: "MF"}

USER_AGENT = "epl-predictor/1.0 (portfolio project; contact via GitHub)"

REQUIRED = [
    "first_name", "second_name", "web_name", "team", "element_type",
    "minutes", "goals_scored", "assists",
]


def _as_float(value: object) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_archive() -> tuple[pd.DataFrame, dict[int, str]]:
    print(f"Reading frozen {SEASON} archive")
    print(f"  {ARCHIVE_PLAYERS}")
    players = pd.read_csv(ARCHIVE_PLAYERS)
    teams = pd.read_csv(ARCHIVE_TEAMS)
    team_names = dict(zip(teams["id"], teams["name"]))
    return players, team_names


def load_live() -> tuple[pd.DataFrame, dict[int, str]]:
    print(f"Reading LIVE FPL API — {FPL_BOOTSTRAP}")
    print("  WARNING: only valid for a season still in progress.")
    req = urllib.request.Request(FPL_BOOTSTRAP, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    players = pd.DataFrame(payload["elements"])
    team_names = {t["id"]: t["name"] for t in payload["teams"]}
    return players, team_names


def build(players: pd.DataFrame, team_names: dict[int, str]) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in players.columns]
    if missing:
        raise SystemExit(
            f"Source is missing required column(s): {', '.join(missing)}.\n"
            "The upstream schema may have changed; inspect the CSV before proceeding."
        )

    has_expected = "expected_goals" in players.columns and "expected_assists" in players.columns
    if not has_expected:
        print("  NOTE: no expected_goals/expected_assists in source; "
              "epl_xgi_90 will fall back to actual goal involvement.")

    rows = []
    for row in players.itertuples(index=False):
        minutes = int(_as_float(getattr(row, "minutes", 0)))
        if minutes <= 0:
            continue  # never appeared; no label to learn from

        nineties = minutes / 90.0
        goals = int(_as_float(getattr(row, "goals_scored", 0)))
        assists = int(_as_float(getattr(row, "assists", 0)))
        involvement_90 = (goals + assists) / nineties

        if has_expected:
            xg = _as_float(getattr(row, "expected_goals", 0.0))
            xa = _as_float(getattr(row, "expected_assists", 0.0))
            xgi_90 = (xg + xa) / nineties
        else:
            xg = xa = float("nan")
            xgi_90 = involvement_90

        first = str(getattr(row, "first_name", "") or "").strip()
        second = str(getattr(row, "second_name", "") or "").strip()

        rows.append(
            {
                "player_name": f"{first} {second}".strip(),
                "web_name": str(getattr(row, "web_name", "") or "").strip(),
                "club": team_names.get(int(_as_float(getattr(row, "team", 0))), ""),
                "epl_role": POSITION_MAP.get(int(_as_float(getattr(row, "element_type", 0))), "UNK"),
                "epl_minutes": minutes,
                "epl_nineties": round(nineties, 3),
                "epl_starts": int(_as_float(getattr(row, "starts", 0))),
                "epl_goals": goals,
                "epl_assists": assists,
                "epl_involvement_90": round(involvement_90, 4),
                "epl_xg": round(xg, 3) if has_expected else None,
                "epl_xa": round(xa, 3) if has_expected else None,
                "epl_xgi_90": round(xgi_90, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("epl_minutes", ascending=False)


def sanity_check(df: pd.DataFrame) -> None:
    clubs = sorted(c for c in df["club"].unique() if c)
    print(f"\n  {len(df)} players with >0 minutes across {len(clubs)} clubs")

    if len(clubs) != EXPECTED_CLUBS:
        print(f"\n  *** WARNING: expected {EXPECTED_CLUBS} clubs, found {len(clubs)} ***")
        print("  A Premier League season has 20 clubs. A different count means the")
        print("  source has rolled over to the next season (relegated clubs deleted,")
        print("  promoted clubs added). Labels built from this are survivorship-biased.")
        print(f"  clubs seen: {', '.join(clubs)}")

    thin = df.groupby("club").size().pipe(lambda s: s[s < 15])
    if len(thin):
        print("\n  *** WARNING: clubs with implausibly few players "
              "(likely wrong-season squads): ***")
        for club, count in thin.items():
            print(f"    {club}: {count}")

    max_minutes = int(df["epl_minutes"].max())
    print(f"\n  median minutes: {df['epl_minutes'].median():.0f}   max: {max_minutes} "
          f"(a full season is 38 x 90 = 3420)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="read the live FPL API instead of the frozen archive "
             "(only correct for a season in progress)",
    )
    args = parser.parse_args()

    try:
        players, team_names = load_live() if args.live else load_archive()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any network/parse failure plainly
        print(f"\nERROR: could not read the source: {exc}", file=sys.stderr)
        print("Check your connection, or pass --live to try the FPL API.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"  {len(players)} player records returned")

    df = build(players, team_names)
    sanity_check(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}")

    print("\n  top 5 by expected goal involvement per 90 (min 900 mins):")
    top = df[df["epl_minutes"] >= 900].nlargest(5, "epl_xgi_90")
    for r in top.itertuples():
        print(f"    {r.player_name:32s} {r.club:22s} {r.epl_xgi_90:.3f}")


if __name__ == "__main__":
    main()
