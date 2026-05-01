"""
Fill outcomes (home_win) for every game the bot has analyzed, including
games where no bet was placed.

Why this exists: settle.py fills outcomes only for placed bets (it cross-
checks Kalshi settlements against MLB final scores). The bot's daily_stats.csv
records every analyzed game, including no-edge skips and confidence-gate
rejections. To evaluate forward-going experiments like the gap-aware paper-
trade — i.e., "would we have made CLV on the games we passed on?" — we
need realized outcomes for the rejected games too.

This script is idempotent: it reads the existing logs/all_games_outcomes.csv
and only fetches outcomes for (date, away, home) tuples not yet recorded.
Safe to run daily via cron, after settle.py.

Source: MLB Stats API /api/v1/schedule?sportId=1&date=YYYY-MM-DD (free,
unmetered). Match by Kalshi team code → MLB team name via KALSHI_TO_MLB.
"""

import csv
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.odds import _NAME_TO_KALSHI


DAILY_STATS_PATH = Path("logs/daily_stats.csv")
OUTCOMES_PATH    = Path("logs/all_games_outcomes.csv")
SCHEDULE_URL     = "https://statsapi.mlb.com/api/v1/schedule"

CSV_FIELDS = [
    "game_date", "away_code", "home_code",
    "home_win", "home_runs", "away_runs",
    "mlb_game_pk", "game_status", "fetched_at",
]

# MLB Stats API returns full team names; map them to Kalshi codes.
# _NAME_TO_KALSHI from data/odds.py covers all 30 teams.

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
logger = logging.getLogger(__name__)


def _load_existing_keys() -> set[tuple[str, str, str]]:
    if not OUTCOMES_PATH.exists():
        return set()
    keys = set()
    with open(OUTCOMES_PATH, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Only count rows where we successfully filled an outcome
            if r.get("home_win") in ("0", "1"):
                keys.add((r["game_date"], r["away_code"], r["home_code"]))
    return keys


def _load_analyzed_games() -> dict[str, set[tuple[str, str]]]:
    """Return {game_date: {(away_code, home_code), ...}} from daily_stats.csv."""
    if not DAILY_STATS_PATH.exists():
        return {}
    out: dict[str, set[tuple[str, str]]] = {}
    with open(DAILY_STATS_PATH, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r.get("game_date", "")
            a = r.get("away_code", "").strip()
            h = r.get("home_code", "").strip()
            if d and a and h:
                out.setdefault(d, set()).add((a, h))
    return out


def _fetch_date(d: str) -> list[dict]:
    """Fetch all final games for one ET date from MLB Stats API."""
    try:
        resp = requests.get(
            SCHEDULE_URL,
            params={"sportId": 1, "date": d},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"  {d}: schedule fetch failed: {e}")
        return []

    games = []
    for day in data.get("dates", []):
        games.extend(day.get("games", []))
    return games


def main() -> None:
    analyzed = _load_analyzed_games()
    if not analyzed:
        logger.info("No analyzed games in daily_stats.csv. Exiting.")
        return

    existing = _load_existing_keys()
    logger.info(
        f"daily_stats has {sum(len(v) for v in analyzed.values())} games across "
        f"{len(analyzed)} dates. {len(existing)} outcomes already on disk."
    )

    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTCOMES_PATH.exists()
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_et   = date.today().isoformat()

    # Process oldest dates first, skip dates that aren't yet final.
    written = 0
    skipped_today = 0
    skipped_no_match = 0
    with open(OUTCOMES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()

        for d in sorted(analyzed.keys()):
            # Skip today — games likely still in progress
            if d >= today_et:
                skipped_today += len(analyzed[d])
                continue

            needed = {pair for pair in analyzed[d] if (d, *pair) not in existing}
            if not needed:
                continue

            games = _fetch_date(d)
            if not games:
                continue

            # Build lookup: (away_kalshi, home_kalshi) -> mlb game dict
            mlb_lookup: dict[tuple[str, str], dict] = {}
            for g in games:
                teams = g.get("teams", {})
                away_name = teams.get("away", {}).get("team", {}).get("name", "")
                home_name = teams.get("home", {}).get("team", {}).get("name", "")
                away_kalshi = _NAME_TO_KALSHI.get(away_name)
                home_kalshi = _NAME_TO_KALSHI.get(home_name)
                if not away_kalshi or not home_kalshi:
                    continue
                mlb_lookup[(away_kalshi, home_kalshi)] = g

            for pair in needed:
                g = mlb_lookup.get(pair)
                if g is None:
                    # Game was scheduled but maybe postponed/team-mapping failed
                    skipped_no_match += 1
                    continue
                status = g.get("status", {}).get("detailedState", "")
                # Only record final games
                if status not in ("Final", "Game Over", "Completed Early"):
                    continue
                teams = g.get("teams", {})
                home_runs = teams.get("home", {}).get("score")
                away_runs = teams.get("away", {}).get("score")
                if home_runs is None or away_runs is None:
                    continue
                home_win = 1 if int(home_runs) > int(away_runs) else 0
                writer.writerow({
                    "game_date":   d,
                    "away_code":   pair[0],
                    "home_code":   pair[1],
                    "home_win":    home_win,
                    "home_runs":   int(home_runs),
                    "away_runs":   int(away_runs),
                    "mlb_game_pk": g.get("gamePk", ""),
                    "game_status": status,
                    "fetched_at":  fetched_at,
                })
                written += 1

    logger.info(
        f"Outcomes filled: {written} new rows. "
        f"Skipped {skipped_today} today/future games. "
        f"{skipped_no_match} games not found in MLB schedule (postponed?)."
    )


if __name__ == "__main__":
    main()
