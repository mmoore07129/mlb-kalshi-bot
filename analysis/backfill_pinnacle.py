"""
Backfill historical Pinnacle (and LowVig, BetOnline) lines for 2024-2025
MLB regular season games. Output is the input dataset for the model-vs-
Pinnacle CLV backtest.

Strategy:
  - Derive the set of game-days from logs/calibration_2024.csv and
    logs/calibration_2025.csv (so we only query days that have games we
    care about).
  - Per game-day, query 2 snapshots: 16:00 UTC (12 PM ET) and 23:30 UTC
    (7:30 PM ET). The afternoon snapshot catches afternoon games before
    first pitch; the evening snapshot catches evening games 30-90 min
    pre-pitch (much closer to close than 22:00 UTC would be).
  - For each game in the snapshot, store both vig-free probability methods:
    proportional and Shin (1993). If the conclusion of the backtest changes
    between methods, that's a measurement artifact, not edge.

Insurance:
  - Circuit breaker: halt on 3 consecutive non-200s OR if quota drops by
    >50 per request (signals double-charging or other abuse).
  - Raw response saved to logs/raw_odds/YYYY-MM-DD_HHZ.json before any
    parsing, so we can re-derive results without re-querying if the
    parser turns out to have a bug.
  - Resumable: skip (date, snapshot) pairs whose raw JSON already exists.
  - Rate limit: sleep 0.5s between requests (polite to API).

Cost estimate:
  - ~410 game-days × 2 snapshots = 820 requests
  - 10 quota per request = 8,200 quota
"""

import os
import sys
import csv
import json
import time
import logging
from datetime import datetime, date, timezone
from pathlib import Path

import requests

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


HISTORICAL_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/"
SNAPSHOT_HOURS_UTC = (16, 23)              # 16:00 and 23:30 UTC
SNAPSHOT_MINS_UTC  = (0, 30)               # paired with hours above
SNAPSHOT_LABELS    = ("1600Z", "2330Z")    # for filenames

RAW_DIR  = Path("logs/raw_odds")
OUT_CSV  = Path("logs/pinnacle_lines.csv")

QUOTA_BREAKER = 50    # halt if a single request consumes > this much quota
CONSECUTIVE_FAIL_BREAKER = 3
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)


# ── De-vig methods ──────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def proportional_devig(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    """Standard proportional method. Biases toward the underdog when vig is
    asymmetric (Pinnacle juices favorites slightly more on baseball)."""
    total = p1_raw + p2_raw
    if total <= 0:
        return 0.5, 0.5
    return p1_raw / total, p2_raw / total


def shin_devig(p1_raw: float, p2_raw: float, max_iter: int = 60, tol: float = 1e-9) -> tuple[float, float]:
    """Shin (1993) de-vig assuming asymmetric information. Corrects favorite
    bias by inferring the proportion `z` of insider trading from the
    overround structure. Solves for z via bisection on the constraint that
    true probabilities sum to 1.

    Falls back to proportional if no valid z is found in (0, 0.5).
    """
    total = p1_raw + p2_raw
    if total <= 1.0:
        return p1_raw, p2_raw  # no vig, nothing to do

    def shin_pi(z: float, p_raw: float) -> float:
        # Shin's true-prob formula for outcome i:
        #   pi_i = (sqrt(z^2 + 4*(1-z)*p_i_hat^2 / total) - z) / (2*(1-z))
        disc = z * z + 4.0 * (1.0 - z) * p_raw * p_raw / total
        return ((disc ** 0.5) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return shin_pi(z, p1_raw) + shin_pi(z, p2_raw) - 1.0

    lo, hi = 1e-9, 0.5 - 1e-9
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # No sign change in interval — fall back to proportional
        return proportional_devig(p1_raw, p2_raw)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    z = 0.5 * (lo + hi)
    pi_1 = shin_pi(z, p1_raw)
    pi_2 = 1.0 - pi_1  # enforce sum-to-1 exactly
    return pi_1, pi_2


# ── Helpers ─────────────────────────────────────────────────────────────────

def collect_game_days() -> list[date]:
    """Union of game-days from 2024 and 2025 calibration CSVs."""
    days = set()
    for year_file in ("logs/calibration_2024.csv", "logs/calibration_2025.csv"):
        if not os.path.exists(year_file):
            logger.warning(f"Missing {year_file} — did training run produce it?")
            continue
        with open(year_file) as f:
            for r in csv.DictReader(f):
                days.add(date.fromisoformat(r["date"]))
    return sorted(days)


def query_snapshot(api_key: str, ts: datetime) -> tuple[dict | None, dict]:
    """Return (json_payload, headers_dict). json_payload is None on failure."""
    params = {
        "apiKey":     api_key,
        "date":       ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions":    "us",  # Pinnacle confirmed available; us alone halves cost is wrong but doesn't hurt
        "markets":    "h2h",
        "bookmakers": "pinnacle,lowvig,betonlineag",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(HISTORICAL_URL, params=params, timeout=20)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code != 200:
            return None, headers
        return resp.json(), headers
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None, {}


def parse_snapshot(payload: dict) -> list[dict]:
    """Extract per-game lines from a historical snapshot payload."""
    out = []
    snap_ts = payload.get("timestamp", "")
    games   = payload.get("data", [])
    for g in games:
        commence    = g.get("commence_time", "")
        home_full   = g.get("home_team", "")
        away_full   = g.get("away_team", "")

        per_book: dict[str, dict] = {}
        for bm in g.get("bookmakers", []):
            key = bm.get("key", "")
            if key not in ("pinnacle", "lowvig", "betonlineag"):
                continue
            h2h = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
            if h2h is None:
                continue
            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            if home_full not in outcomes or away_full not in outcomes:
                continue
            p_home_raw = american_to_prob(outcomes[home_full])
            p_away_raw = american_to_prob(outcomes[away_full])
            p_home_prop, p_away_prop = proportional_devig(p_home_raw, p_away_raw)
            p_home_shin, p_away_shin = shin_devig(p_home_raw, p_away_raw)
            per_book[key] = {
                "home_odds_amer": outcomes[home_full],
                "away_odds_amer": outcomes[away_full],
                "p_home_prop":    round(p_home_prop, 5),
                "p_home_shin":    round(p_home_shin, 5),
            }

        if not per_book:
            continue
        out.append({
            "snapshot_time": snap_ts,
            "commence_time": commence,
            "home_team":     home_full,
            "away_team":     away_full,
            "per_book":      per_book,
        })
    return out


def main():
    api_key = config.ODDS_API_KEY
    if not api_key:
        logger.error("ODDS_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    days = collect_game_days()
    logger.info(f"Will query {len(days)} game-days × 2 snapshots = {len(days) * 2} requests")
    logger.info(f"Estimated quota: {len(days) * 2 * 10}")

    # If output CSV exists, learn what (date, label) pairs are already done so
    # we don't double-write on resume. Raw JSONs on disk are independent —
    # they're a parse-only resource.
    done_keys: set[tuple[str, str]] = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                done_keys.add((r["game_date"], r["snapshot_label"]))
        logger.info(f"Resume: {len(done_keys)} (date, label) pairs already in CSV — will skip")
    csv_exists = OUT_CSV.exists()
    csv_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow([
            "game_date", "snapshot_label", "snapshot_time", "commence_time",
            "home_team", "away_team", "book",
            "home_odds_amer", "away_odds_amer", "p_home_prop", "p_home_shin",
        ])
        csv_file.flush()

    consecutive_failures = 0
    last_quota_remaining = None
    total_requests = 0
    total_rows = 0

    for d in days:
        for hour, minute, label in zip(SNAPSHOT_HOURS_UTC, SNAPSHOT_MINS_UTC, SNAPSHOT_LABELS):
            if (d.isoformat(), label) in done_keys:
                continue  # already in CSV
            raw_path = RAW_DIR / f"{d.isoformat()}_{label}.json"
            ts = datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc)

            if raw_path.exists():
                # Resume: load existing raw and re-parse (cheap)
                with open(raw_path, "r", encoding="utf-8") as rf:
                    payload = json.load(rf)
            else:
                payload, headers = query_snapshot(api_key, ts)
                total_requests += 1

                # Circuit breakers
                if payload is None:
                    consecutive_failures += 1
                    logger.warning(f"FAIL  {d} {label}  ({consecutive_failures} consecutive)")
                    if consecutive_failures >= CONSECUTIVE_FAIL_BREAKER:
                        logger.error(f"HALT: {CONSECUTIVE_FAIL_BREAKER} consecutive failures — circuit breaker tripped")
                        csv_file.close()
                        sys.exit(2)
                    time.sleep(2 * consecutive_failures)
                    continue
                consecutive_failures = 0

                quota_remaining = headers.get("x-requests-remaining")
                quota_last      = headers.get("x-requests-last")
                if last_quota_remaining is not None and quota_remaining is not None:
                    delta = int(last_quota_remaining) - int(quota_remaining)
                    if delta > QUOTA_BREAKER:
                        logger.error(f"HALT: quota dropped by {delta} on a single request — circuit breaker tripped")
                        csv_file.close()
                        sys.exit(3)
                last_quota_remaining = quota_remaining

                # Save raw before parsing
                with open(raw_path, "w", encoding="utf-8") as rf:
                    json.dump(payload, rf)

                if total_requests % 25 == 0:
                    logger.info(
                        f"[{total_requests:4d} reqs done]  quota_remaining={quota_remaining}  "
                        f"last_request_cost={quota_last}"
                    )

                time.sleep(SLEEP_BETWEEN_REQUESTS)

            games = parse_snapshot(payload)
            for game in games:
                for book, book_data in game["per_book"].items():
                    writer.writerow([
                        d.isoformat(),
                        label,
                        game["snapshot_time"],
                        game["commence_time"],
                        game["home_team"],
                        game["away_team"],
                        book,
                        book_data["home_odds_amer"],
                        book_data["away_odds_amer"],
                        book_data["p_home_prop"],
                        book_data["p_home_shin"],
                    ])
                    total_rows += 1
            csv_file.flush()

    csv_file.close()
    logger.info(f"Done. Total requests: {total_requests}  CSV rows written: {total_rows}")
    logger.info(f"Quota remaining: {last_quota_remaining}")


if __name__ == "__main__":
    main()
