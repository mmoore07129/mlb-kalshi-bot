"""
Paper-trade harness for the gap-aware EV threshold.

The current bot uses a dynamic EV threshold based on sharp-book disagreement
(prob_std). The paper-trade rule additionally rewards model-Pinnacle gap as
a Pinnacle-trust signal: when the model strongly disagrees with the books,
the books are more likely right (per the 2024-2025 CLV backtest), so we
should be more willing to bet at lower EV in that regime.

Formula:
  gap_bonus = max(0, |model_pin_gap| - 0.05) * 0.30
  std_penalty = 1.5 * prob_std
  threshold = clamp(base + std_penalty - gap_bonus, base, ceil)

Notes on the formula choice:
  - 0.05 deadband: gap < 5pp is too noisy to credit (per inversion_check.py
    the H2H signal is essentially flat below 5pp).
  - 0.30 multiplier: at the maximum legitimate gap (~38pp observed), the
    bonus is (0.38-0.05)*0.30 = 0.099, which can take a 0.005 base down by
    nearly 10pp — never below the floor due to clamp.
  - Subtractive: lower threshold = more willing to bet.
  - Same prob_std penalty as current rule, so paper-trade is "current rule
    plus gap-aware adjustment", not a wholly different rule.

This module records the gap-aware decision PER ANALYZED GAME alongside the
real decision (which lives in daily_stats.csv). After 30-60 days of forward
data, paper_trade.csv can be joined with all_games_clv.csv (Pinnacle close)
and outcomes (settle.py) to compute realized CLV under each rule. If the
gap-aware rule's CLV beats the current rule's, deploy it.

NOT a bot behavior change. Logging only.
"""

import csv
import os
from datetime import datetime


PAPER_TRADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'logs', 'paper_trade.csv',
)

FIELDNAMES = [
    'timestamp',
    'mode',
    'game_date',
    'away_code', 'home_code',
    'model_p_home', 'pin_home_prob',
    'model_pin_gap', 'prob_std',
    'kalshi_home_ask', 'kalshi_away_ask',
    'ev_home_net', 'ev_away_net',
    'ev_threshold_current',
    'ev_threshold_gap_aware',
    'decision_current',     # what the live bot did: bet-{TEAM} | no-edge | (other reason)
    'decision_gap_aware',   # what the gap-aware rule would have done: bet-{TEAM} | no-edge
    'gap_aware_changed',    # 1 if the two decisions differ, 0 if same
]


def gap_aware_threshold(base: float, ceil: float, prob_std: float, gap: float) -> float:
    """Gap-aware version of the dynamic EV threshold. See module docstring."""
    gap_bonus   = max(0.0, abs(gap) - 0.05) * 0.30
    std_penalty = 1.5 * prob_std
    raw         = base + std_penalty - gap_bonus
    return max(base, min(ceil, raw))


def _ensure_header() -> None:
    """Same migration pattern as daily_stats / ledger — extras_action='ignore'
    on append, full rebuild if header has drifted."""
    os.makedirs(os.path.dirname(PAPER_TRADE_PATH), exist_ok=True)
    if not os.path.exists(PAPER_TRADE_PATH) or os.path.getsize(PAPER_TRADE_PATH) == 0:
        with open(PAPER_TRADE_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        return

    with open(PAPER_TRADE_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        if existing_header == FIELDNAMES:
            return
        rows = list(reader)

    with open(PAPER_TRADE_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            for k in FIELDNAMES:
                r.setdefault(k, '')
            writer.writerow(r)


def new_paper_row(game_date, away_code: str, home_code: str, mode: str = 'live') -> dict:
    """Factory — returns an empty paper-trade row that the caller fills in."""
    return {
        'timestamp':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode':                   mode,
        'game_date':              str(game_date),
        'away_code':              away_code,
        'home_code':              home_code,
        'model_p_home':           '',
        'pin_home_prob':          '',
        'model_pin_gap':          '',
        'prob_std':               '',
        'kalshi_home_ask':        '',
        'kalshi_away_ask':        '',
        'ev_home_net':            '',
        'ev_away_net':            '',
        'ev_threshold_current':   '',
        'ev_threshold_gap_aware': '',
        'decision_current':       '',
        'decision_gap_aware':     '',
        'gap_aware_changed':      '',
    }


def finalize_row(paper_row: dict, reason: str, bet_team_code: str | None) -> None:
    """Fill in decision_current + gap_aware_changed based on what the live
    bot did this iteration. Mutates in place; called once per row before
    write_rows()."""
    if reason == 'placed' and bet_team_code:
        paper_row['decision_current'] = f'bet-{bet_team_code}'
    elif reason:
        paper_row['decision_current'] = reason
    paper_row['gap_aware_changed'] = (
        '1' if paper_row.get('decision_current') != paper_row.get('decision_gap_aware') else '0'
    )


def write_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_header()
    with open(PAPER_TRADE_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        for r in rows:
            writer.writerow(r)
