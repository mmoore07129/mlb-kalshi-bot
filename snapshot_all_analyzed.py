"""
Snapshot the current sharp-book blend for ALL analyzed games on today's
slate (not just placed bets — clv_snapshot.py handles those).

Purpose: forward-going CLV capture for the gap-aware-threshold paper-trade
evaluation. We need to know, for every game the bot analyzed today:
  - What was Pinnacle's prob at analysis time? (stored in daily_stats.csv)
  - What was Pinnacle's prob near close? (this script — every 5 min until
    each game leaves the feed, latest write wins)
  - Did the home team win? (filled in by settle.py the next morning)

With those three columns plus daily_stats's model_pin_gap, we can later
backtest "would betting the high-gap rejected games have generated CLV?"

Append-only output: logs/all_games_clv.csv. One row per snapshot per game.
The analyzer downstream picks the latest pre-first-pitch row as the close.

Intended cron schedule: same as clv_snapshot.py — every 5 min during the
11 AM – midnight ET window.
"""

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.odds import get_pinnacle_odds


DAILY_STATS_PATH = "logs/daily_stats.csv"
ALL_CLV_PATH     = "logs/all_games_clv.csv"

CSV_FIELDS = [
    "snapshot_time", "game_date", "away_code", "home_code",
    "pin_home_prob", "pin_away_prob", "prob_std", "books_used",
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)


def _today_analyzed_games() -> set[tuple[str, str]]:
    """Return set of (away_code, home_code) for games analyzed today in
    daily_stats.csv. Today defined in ET to match the sharp-book feed."""
    if not os.path.exists(DAILY_STATS_PATH):
        return set()
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    games = set()
    with open(DAILY_STATS_PATH, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("game_date") == today_et:
                away = r.get("away_code", "").strip()
                home = r.get("home_code", "").strip()
                if away and home:
                    games.add((away, home))
    return games


def main() -> None:
    analyzed = _today_analyzed_games()
    if not analyzed:
        logger.debug("No analyzed games for today in daily_stats.csv. Exiting.")
        return

    blend = get_pinnacle_odds()
    if not blend:
        logger.warning("Sharp-book blend unavailable (no ODDS_API_KEY, or fetch failed).")
        return

    out_path = Path(ALL_CLV_PATH)
    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_et      = datetime.now(ZoneInfo("America/New_York")).date().isoformat()

    written = 0
    locked  = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()

        for away, home in sorted(analyzed):
            game = blend.get((away, home))
            if game is None:
                # Game already locked (past first pitch) or not in today's feed
                locked += 1
                continue
            writer.writerow({
                "snapshot_time": snapshot_time,
                "game_date":     today_et,
                "away_code":     away,
                "home_code":     home,
                "pin_home_prob": f"{float(game['home_prob']):.4f}",
                "pin_away_prob": f"{float(game['away_prob']):.4f}",
                "prob_std":      f"{float(game.get('prob_std', 0.0)):.4f}",
                "books_used":    "|".join(game.get('books_used', []) or []),
            })
            written += 1

    logger.info(
        f"all_games_clv: wrote {written} row(s)  locked/missing={locked}  "
        f"(of {len(analyzed)} analyzed today)"
    )


if __name__ == "__main__":
    main()
