"""
Sanity-check the Odds API historical endpoint before burning quota on a
multi-thousand-request backfill.

For 5 random 2024 game dates:
  - Hit /v4/historical/sports/baseball_mlb/odds/ at noon ET that day
  - Verify the response includes Pinnacle h2h moneylines
  - Spot-check vig-free probabilities are sensible (in [0, 1], roughly sum ~1)
  - Spot-check team name mapping works for our games of interest

Reads logs/calibration_2024.csv to pick random games. Pulls API key from .env.
Cost: 5 historical quota.
"""

import os
import csv
import sys
import json
import random
from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

import requests

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


HISTORICAL_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/"
ET_ZONE = ZoneInfo("America/New_York")


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def vig_free_home(home_odds: float, away_odds: float) -> tuple[float, float]:
    h = american_to_prob(home_odds)
    a = american_to_prob(away_odds)
    total = h + a
    return h / total, total  # vig-free home prob, raw vig (total > 1.0)


def main():
    api_key = config.ODDS_API_KEY
    if not api_key:
        print("ERROR: ODDS_API_KEY not set")
        return

    rows = []
    with open("logs/calibration_2024.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Pick 5 random games from 5 different dates
    random.seed(20260501)
    dates_seen = set()
    samples = []
    random.shuffle(rows)
    for r in rows:
        if r["date"] in dates_seen:
            continue
        dates_seen.add(r["date"])
        samples.append(r)
        if len(samples) >= 5:
            break

    print(f"Sampled {len(samples)} games from distinct dates:")
    for s in samples:
        print(f"  {s['date']}  {s['away_team']} @ {s['home_team']}  "
              f"(model_p_home={float(s['predicted_p_home']):.3f}  actual={s['actual_home_win']})")
    print()

    # Hit the API for each
    quota_used = 0
    quota_remaining = None
    matched = 0
    not_found = []
    for s in samples:
        # Query at 18:00 UTC on game day (1pm ET — after pregame markets exist, before most first pitches)
        d = datetime.fromisoformat(s['date']).replace(tzinfo=timezone.utc, hour=18, minute=0)
        ts = d.strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "apiKey":     api_key,
            "date":       ts,
            "regions":    "us,eu",
            "markets":    "h2h",
            "bookmakers": "pinnacle",
            "oddsFormat": "american",
        }
        print(f"--- Query: {ts} ({s['away_team']} @ {s['home_team']}) ---")
        try:
            resp = requests.get(HISTORICAL_URL, params=params, timeout=15)
            quota_remaining = resp.headers.get("x-requests-remaining")
            quota_used      = resp.headers.get("x-requests-used")
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"  REQUEST FAILED: {e}")
            continue

        snap_ts  = payload.get("timestamp")
        next_ts  = payload.get("next_timestamp")
        prev_ts  = payload.get("previous_timestamp")
        games    = payload.get("data", [])
        print(f"  snapshot_ts={snap_ts}  prev={prev_ts}  next={next_ts}  games_in_snapshot={len(games)}")

        # Find the matching game by team names
        # The Odds API uses full team names; calibration_2024 has Kalshi codes.
        # Quick reverse-map for the 5 random teams we sampled:
        from kalshi.markets import TEAM_CODES
        away_full = TEAM_CODES.get(s["away_team"], s["away_team"])
        home_full = TEAM_CODES.get(s["home_team"], s["home_team"])

        match = None
        for g in games:
            if g.get("home_team") == home_full and g.get("away_team") == away_full:
                match = g
                break
        if match is None:
            print(f"  GAME NOT FOUND in snapshot — looked for '{away_full}' @ '{home_full}'")
            print(f"  Available teams in snapshot: {sorted({g.get('home_team', '') for g in games})}")
            not_found.append(s)
            continue
        matched += 1

        commence = match.get("commence_time")
        bookmakers = match.get("bookmakers", [])
        pinnacle = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
        if pinnacle is None:
            print(f"  GAME found but NO PINNACLE LINE  bookmakers_present={[b.get('key') for b in bookmakers]}")
            continue

        h2h_market = next((m for m in pinnacle.get("markets", []) if m.get("key") == "h2h"), None)
        if h2h_market is None:
            print(f"  Pinnacle present but no h2h market")
            continue

        outcomes = {o["name"]: o["price"] for o in h2h_market.get("outcomes", [])}
        if home_full not in outcomes or away_full not in outcomes:
            print(f"  Pinnacle h2h missing teams  outcomes={outcomes}")
            continue

        home_odds = outcomes[home_full]
        away_odds = outcomes[away_full]
        p_home_vf, vig = vig_free_home(home_odds, away_odds)

        print(f"  commence={commence}")
        print(f"  Pinnacle: {away_full}={away_odds:+}  {home_full}={home_odds:+}  vig={(vig-1)*100:.2f}%  p_home_vf={p_home_vf:.3f}")
        print(f"  Model said p_home={float(s['predicted_p_home']):.3f}  Pinnacle says {p_home_vf:.3f}  gap={p_home_vf-float(s['predicted_p_home']):+.3f}")
        print(f"  Actual home_win={s['actual_home_win']}  ({'home won' if s['actual_home_win']=='1' else 'away won'})")
        print()

    print("=== SANITY CHECK SUMMARY ===")
    print(f"  Sampled: {len(samples)}")
    print(f"  Matched in snapshot: {matched}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Quota used (cumulative): {quota_used}")
    print(f"  Quota remaining: {quota_remaining}")


if __name__ == "__main__":
    main()
