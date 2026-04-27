"""
Bet/outcome ledger — append-mode CSV at logs/ledger.csv.

record_bet()     — call immediately after a successful order placement
record_clv()     — call from clv_snapshot after fetching closing line
record_outcome() — call on settlement to close the row with result + PnL

prob_src column ('Pinnacle' vs 'model') lets you measure each source's
realized EV independently after enough bets accumulate.

bet_team_code stores the team we bet on as its own column so settle.py
doesn't have to parse bet_label strings (fragile if label format ever changes).

CLV columns are populated separately (later than record_bet) because we need
the closing line from sharp books, which only exists ~5 min before first pitch.
See clv_snapshot.py for the writer. Convention: clv_prob and clv_pct are signed
so positive = market moved toward our pick (we beat the close).
"""

import csv
import os
from datetime import datetime

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'ledger.csv')

FIELDNAMES = [
    'timestamp', 'game_date', 'home_team', 'away_team',
    'bet_label', 'bet_team_code', 'ticker', 'side', 'contracts', 'price',
    'cost', 'ev', 'p_used', 'model_p_home', 'prob_src',
    'order_id', 'outcome', 'pnl',
    # CLV — populated by clv_snapshot.py after the closing line is available
    'close_home_prob', 'close_away_prob', 'close_prob_src',
    'close_snapshot_time', 'clv_prob', 'clv_pct',
]


def _ensure_header() -> None:
    """
    Create the ledger with the current FIELDNAMES header if it doesn't exist.
    If an older ledger exists with a different header (e.g., pre-`bet_team_code`),
    migrate it in place by adding any missing columns as empty strings.
    """
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    if not os.path.exists(LEDGER_PATH) or os.path.getsize(LEDGER_PATH) == 0:
        with open(LEDGER_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        return

    with open(LEDGER_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        if existing_header == FIELDNAMES:
            return
        rows = list(reader)

    # Migrate: rewrite with new FIELDNAMES; missing columns default to ''
    with open(LEDGER_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            for k in FIELDNAMES:
                r.setdefault(k, '')
            writer.writerow(r)


def record_bet(
    game_date,
    home_team: str,
    away_team: str,
    bet_label: str,
    ticker: str,
    side: str,
    contracts: int,
    price,
    cost: float,
    ev: float,
    p_used: float,
    model_p_home: float,
    prob_src: str,
    order_id: str = '',
    bet_team_code: str = '',
) -> None:
    _ensure_header()
    row = {
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'game_date':     str(game_date),
        'home_team':     home_team,
        'away_team':     away_team,
        'bet_label':     bet_label,
        'bet_team_code': bet_team_code,
        'ticker':        ticker,
        'side':          side,
        'contracts':     contracts,
        'price':         round(float(price), 4),
        'cost':          round(float(cost), 2),
        'ev':            round(float(ev), 4),
        'p_used':        round(float(p_used), 4),
        'model_p_home':  round(float(model_p_home), 4),
        'prob_src':      prob_src,
        'order_id':      order_id,
        'outcome':       '',
        'pnl':           '',
    }
    with open(LEDGER_PATH, 'a', newline='') as f:
        # extrasaction='ignore' lets us write rows that have fewer keys than
        # FIELDNAMES without error; missing columns are written as empty.
        csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore').writerow(row)


def record_clv(
    order_id: str,
    close_home_prob: float,
    close_away_prob: float,
    close_prob_src: str,
    clv_prob: float,
    clv_pct: float,
    snapshot_time: str = '',
) -> bool:
    """
    Update the first matching row (by order_id) with CLV data.

    Overwrites any existing CLV — by design. The standalone clv_snapshot.py runs
    every ~5 min during the game window, and we want the LAST pre-lockout snapshot
    (closest to first pitch) to win. Once a game leaves the sharp-book feed (past
    first pitch), no further snapshots fire for it, so the last write naturally
    freezes as the de-facto close.

    Returns True if a row was updated, False if no matching order_id found.
    """
    if not os.path.exists(LEDGER_PATH):
        return False
    rows = []
    updated = False
    snapshot_time = snapshot_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LEDGER_PATH, 'r', newline='') as f:
        for row in csv.DictReader(f):
            if not updated and row.get('order_id') == order_id:
                row['close_home_prob']     = round(float(close_home_prob), 4)
                row['close_away_prob']     = round(float(close_away_prob), 4)
                row['close_prob_src']      = close_prob_src
                row['close_snapshot_time'] = snapshot_time
                row['clv_prob']            = round(float(clv_prob), 4)
                row['clv_pct']             = round(float(clv_pct), 4)
                updated = True
            for k in FIELDNAMES:
                row.setdefault(k, '')
            rows.append(row)
    with open(LEDGER_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return updated


def record_outcome(order_id: str, outcome: str, pnl: float) -> None:
    """
    Update the first unsettled row matching order_id with outcome ('win'/'loss'/'void')
    and realized PnL in dollars.

    Handles legacy ledgers that predate the bet_team_code column: when reading,
    DictReader leaves unknown columns as None; when rewriting, we fill missing
    FIELDNAMES keys with empty strings so the CSV stays schema-consistent.
    """
    if not os.path.exists(LEDGER_PATH):
        return
    rows = []
    updated = False
    with open(LEDGER_PATH, 'r', newline='') as f:
        for row in csv.DictReader(f):
            if not updated and row.get('order_id') == order_id and row.get('outcome', '') == '':
                row['outcome'] = outcome
                row['pnl']     = round(pnl, 2)
                updated = True
            # Ensure every key exists before write-back
            for k in FIELDNAMES:
                row.setdefault(k, '')
            rows.append(row)
    with open(LEDGER_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
