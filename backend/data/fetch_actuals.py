"""
Fetch ACTUAL 2025-26 Premier League player output — the labels for forecast.py.

Source: the public Fantasy Premier League API (no key, no scraping, no ToS
problem). `bootstrap-static` returns every player who appeared in the league
with season totals including FPL's own expected-goals numbers (Opta-derived).

Run this once from the repo:

    python backend/data/fetch_actuals.py

Writes backend/data/actuals/epl_2025_26.csv

This also solves a second problem: it gives us the real Premier League roster
and club names, which the old FBref `pd.read_html` merge never managed to
attach (the shipped CSV listed national teams, not clubs).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "actuals"
OUT_CSV = OUT_DIR / "epl_2025_26.csv"

# FPL position ids -> our role buckets. FPL lumps wingers into MID and has no
# separate winger class, which is close enough for a label join.
POSITION_MAP = {1: "GK", 2: "DF", 3: "MF", 4: "FW", 5: "MF"}

USER_AGENT = "epl-predictor/1.0 (portfolio project; contact via GitHub)"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(f"Fetching {FPL_BOOTSTRAP}")
    try:
        payload = fetch_json(FPL_BOOTSTRAP)
    except Exception as exc:  # noqa: BLE001 - surface any network failure plainly
        print(f"\nERROR: could not reach the FPL API: {exc}", file=sys.stderr)
        print(
            "If you are behind a proxy or offline, run this on an unrestricted "
            "connection. forecast.py will refuse to run without the labels.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    teams = {t["id"]: t["name"] for t in payload["teams"]}
    elements = payload["elements"]
    print(f"  {len(elements)} player records returned")

    rows = []
    for e in elements:
        minutes = e.get("minutes") or 0
        if minutes <= 0:
            continue  # never appeared; no label to learn from

        nineties = minutes / 90.0
        goals = e.get("goals_scored") or 0
        assists = e.get("assists") or 0

        def as_float(key: str) -> float:
            val = e.get(key)
            try:
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        xg = as_float("expected_goals")
        xa = as_float("expected_assists")

        first = (e.get("first_name") or "").strip()
        second = (e.get("second_name") or "").strip()
        web = (e.get("web_name") or "").strip()

        rows.append(
            {
                "player_name": f"{first} {second}".strip(),
                "web_name": web,
                "club": teams.get(e.get("team"), ""),
                "epl_role": POSITION_MAP.get(e.get("element_type"), "UNK"),
                "epl_minutes": minutes,
                "epl_nineties": round(nineties, 3),
                "epl_starts": e.get("starts") or 0,
                "epl_goals": goals,
                "epl_assists": assists,
                "epl_involvement_90": round((goals + assists) / nineties, 4),
                "epl_xg": round(xg, 3),
                "epl_xa": round(xa, 3),
                "epl_xgi_90": round((xg + xa) / nineties, 4),
            }
        )

    df = pd.DataFrame(rows).sort_values("epl_minutes", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n  wrote {OUT_CSV}")
    print(f"  {len(df)} players with >0 minutes across {df['club'].nunique()} clubs")
    print(f"  median minutes: {df['epl_minutes'].median():.0f}")
    print("\n  top 5 by expected goal involvement per 90 (min 900 mins):")
    top = df[df["epl_minutes"] >= 900].nlargest(5, "epl_xgi_90")
    for r in top.itertuples():
        print(f"    {r.player_name:32s} {r.club:22s} {r.epl_xgi_90:.3f}")


if __name__ == "__main__":
    main()
