"""
Daily stats — per-game decision diagnostics written to logs/daily_stats.csv.

One row per game the bot analyzes on a given session. Only written when
--dry-run is OFF (live mode) so the stats file reflects real production
behavior, not every debug run.

Purpose: after a week of running, aggregate `reason` counts and `best_ev`
distributions to answer "why is the bot not betting?" without grepping
through per-session bot logs.

Schema designed to be bounded — new fields can be added later and
_ensure_header() will migrate the file in place, same pattern as ledger.py.
"""

import csv
import os
from datetime import datetime

DAILY_STATS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'logs', 'daily_stats.csv',
)

FIELDNAMES = [
    'timestamp', 'mode', 'game_date', 'away_code', 'home_code',
    'away_sp', 'home_sp',
    'model_p_home', 'pin_home_prob', 'pin_src', 'pin_std', 'model_pin_gap',
    'ev_threshold', 'kalshi_home_ask', 'kalshi_away_ask',
    'ev_home_net', 'ev_away_net', 'best_ev',
    'reason',  # placed | no-edge | model-veto-home | model-veto-away |
               # model-pin-gap | confidence-gate | missing-stats |
               # no-orderbook | kelly-zero | session-cap | already-positioned |
               # place-failed
    'bet_contracts', 'bet_price', 'bet_cost', 'order_id',
]


def _ensure_header() -> None:
    """Same migration pattern as ledger.py — tolerate legacy headers."""
    os.makedirs(os.path.dirname(DAILY_STATS_PATH), exist_ok=True)
    if not os.path.exists(DAILY_STATS_PATH) or os.path.getsize(DAILY_STATS_PATH) == 0:
        with open(DAILY_STATS_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        return

    with open(DAILY_STATS_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        if existing_header == FIELDNAMES:
            return
        rows = list(reader)

    with open(DAILY_STATS_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            for k in FIELDNAMES:
                r.setdefault(k, '')
            writer.writerow(r)


def new_game_stats(game_date, away_code: str, home_code: str,
                   away_sp: str = '', home_sp: str = '',
                   mode: str = 'live') -> dict:
    """Factory — returns a stats dict with safe defaults. Caller mutates it as
    the game flows through the decision logic, then passes to write_rows()."""
    return {
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode':             mode,
        'game_date':        str(game_date),
        'away_code':        away_code,
        'home_code':        home_code,
        'away_sp':          away_sp,
        'home_sp':          home_sp,
        'model_p_home':     '',
        'pin_home_prob':    '',
        'pin_src':          '',
        'pin_std':          '',
        'model_pin_gap':    '',
        'ev_threshold':     '',
        'kalshi_home_ask':  '',
        'kalshi_away_ask':  '',
        'ev_home_net':      '',
        'ev_away_net':      '',
        'best_ev':          '',
        'reason':           '',
        'bet_contracts':    '',
        'bet_price':        '',
        'bet_cost':         '',
        'order_id':         '',
    }


def write_rows(rows: list[dict]) -> None:
    """Append all rows to the daily_stats CSV. No-op on empty list."""
    if not rows:
        return
    _ensure_header()
    with open(DAILY_STATS_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        for r in rows:
            writer.writerow(r)
