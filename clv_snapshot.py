"""
CLV snapshot — scans the ledger for unsettled bets and writes the current
sharp-book blend as their closing line.

Intended to run every ~5 min via cron during the game-day window. Behavior:

1. Read ledger.csv.
2. For each row with a real order_id (not DRY_RUN, not empty) and no settled
   outcome yet, fetch the current sharp-book blend via data.odds.get_pinnacle_odds().
3. Match by (away_team_kalshi_code, home_team_kalshi_code). If the game is still
   in the sharp-book feed (i.e., pre-first-pitch), compute:
     - clv_prob = p_close_fair_our_side − p_used_at_bet
     - clv_pct  = (your_decimal / close_decimal_fair) − 1
   and overwrite the row's CLV columns via ledger.record_clv().
4. Once a game disappears from the sharp-book feed (past first pitch), no further
   updates fire for that row — the last-written value stands as the close.

Sign conventions:
  - Positive clv_prob  → market moved TOWARD our pick (we beat the close).
  - Positive clv_pct   → our implied odds beat the closing fair odds (same meaning).

Model-fallback bets (prob_src == 'model') get CLV relative to the model's own
p_used rather than a book-fair-at-bet reference, which measures "did the market
agree with what my model predicted." Still useful as a model-calibration signal.
"""

import csv
import logging
import os
import sys
from datetime import datetime, date
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mlb_fetcher import KALSHI_TO_MLB
from data.odds import get_pinnacle_odds
from ledger import LEDGER_PATH, record_clv

# Reverse of KALSHI_TO_MLB so ledger rows (which store MLB codes in home_team /
# away_team) can be mapped back to the Kalshi codes used in data.odds.
_MLB_TO_KALSHI = {v: k for k, v in KALSHI_TO_MLB.items()}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)


def _identify_bet_sides(row: dict) -> tuple[str, str, bool] | None:
    """
    Return (away_kalshi, home_kalshi, we_bet_home) for a ledger row, or None if
    we can't confidently identify the matchup.

    Uses ledger's own home_team / away_team columns (MLB-coded) reverse-mapped to
    Kalshi codes, then the ticker's suffix to determine which side we bet. The
    ticker-suffix path works even on legacy rows with empty bet_team_code.

    Team codes vary in length (SD, SF, AZ are 2 chars; most are 3; ATH/CWS are
    3 too) so fixed-offset parsing of the matchup portion of the ticker is
    fragile. This approach sidesteps that entirely.
    """
    away_mlb = row.get('away_team', '') or ''
    home_mlb = row.get('home_team', '') or ''
    if not away_mlb or not home_mlb:
        return None

    away_kalshi = _MLB_TO_KALSHI.get(away_mlb, away_mlb)
    home_kalshi = _MLB_TO_KALSHI.get(home_mlb, home_mlb)

    # The ticker always ends in -<TEAM> where TEAM is the team we said would win
    # (we only buy YES on KXMLBGAME-...-<TEAM> contracts).
    ticker = row.get('ticker', '') or ''
    bet_team = ticker.rsplit('-', 1)[-1] if '-' in ticker else ''

    if bet_team == home_kalshi:
        return (away_kalshi, home_kalshi, True)
    if bet_team == away_kalshi:
        return (away_kalshi, home_kalshi, False)
    return None


def _decimal_from_price(price: float) -> float:
    """Decimal odds from Kalshi fractional price ($/contract). Handles edge cases."""
    if price <= 0 or price >= 1:
        return 0.0
    return 1.0 / price


def snapshot_clv() -> None:
    if not os.path.exists(LEDGER_PATH):
        logger.info("No ledger — nothing to snapshot.")
        return

    with open(LEDGER_PATH, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    # Sharp-book feed only returns today's games, so matching past-dated rows
    # would pull the WRONG game's odds when the same teams play a series. Only
    # snapshot bets whose game_date is today (in ET, since that's the feed's tz).
    today_et = datetime.now(ZoneInfo('America/New_York')).date().isoformat()

    # Rows that need CLV: real order, not dry-run, today's slate, no CLV yet.
    # Past-dated rows without CLV are stuck — closing lines aren't retrievable
    # retroactively from The Odds API free tier. Log them so we know what we
    # missed, but don't try to fill them in.
    pending_today = []
    stale_unfilled = 0
    for r in rows:
        if r.get('order_id') in ('', 'DRY_RUN', None):
            continue
        if r.get('clv_prob'):
            continue
        if r.get('game_date', '') == today_et:
            pending_today.append(r)
        else:
            stale_unfilled += 1

    # Stale past-date rows are permanent — they'll never get CLV. Silent at INFO
    # so the every-5-min cron doesn't flood cron.log; surfaced only when there's
    # real work today (below).
    if not pending_today:
        if stale_unfilled:
            logger.debug(
                f"{stale_unfilled} row(s) missing CLV from past dates — can't backfill, skipping."
            )
        logger.debug("No today-dated rows need CLV. Exiting.")
        return

    if stale_unfilled:
        logger.info(
            f"{stale_unfilled} row(s) missing CLV from past dates — can't backfill, skipping."
        )
    logger.info(f"{len(pending_today)} today-dated row(s) need CLV.")

    # Fetch current sharp-book blend. Returns {} if API key missing or request fails.
    blend = get_pinnacle_odds()
    if not blend:
        logger.warning("Sharp-book blend unavailable (no ODDS_API_KEY, or fetch failed).")
        return

    snapshot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated_n = skipped_no_match = 0

    for row in pending_today:
        order_id = row['order_id']
        sides    = _identify_bet_sides(row)
        if sides is None:
            logger.warning(
                f"Couldn't identify matchup for {order_id} "
                f"(home={row.get('home_team')!r} away={row.get('away_team')!r} "
                f"ticker={row.get('ticker')!r}) — skipping"
            )
            skipped_no_match += 1
            continue
        away_code, home_code, we_bet_home = sides

        game = blend.get((away_code, home_code))
        if game is None:
            # Game already locked (first pitch passed) — nothing to snapshot now.
            skipped_no_match += 1
            continue

        close_home = float(game['home_prob'])
        close_away = float(game['away_prob'])
        src        = 'Blend' if len(game.get('books_used', []) or []) >= 2 else 'Pinnacle'

        bet_team       = home_code if we_bet_home else away_code
        close_our_side = close_home if we_bet_home else close_away

        # p_used is the probability we used at bet time. For Pinnacle/Blend bets
        # this is the sharp-book fair prob then; for model bets it's the model's
        # own estimate. Either way, comparing to close_our_side measures "did the
        # market move toward our pick."
        try:
            p_used = float(row['p_used'])
        except (ValueError, TypeError):
            logger.warning(f"Bad p_used on {order_id} — skipping")
            skipped_no_match += 1
            continue

        bet_price = float(row['price'])
        our_decimal   = _decimal_from_price(bet_price)
        close_decimal = _decimal_from_price(close_our_side)

        clv_prob = close_our_side - p_used
        clv_pct  = (our_decimal / close_decimal) - 1.0 if close_decimal > 0 else 0.0

        if record_clv(
            order_id=order_id,
            close_home_prob=close_home,
            close_away_prob=close_away,
            close_prob_src=src,
            clv_prob=clv_prob,
            clv_pct=clv_pct,
            snapshot_time=snapshot_time,
        ):
            logger.info(
                f"  {away_code}@{home_code} [{bet_team}]  "
                f"close={close_our_side:.1%}  used={p_used:.1%}  "
                f"clv_prob={clv_prob:+.2%}  clv_pct={clv_pct:+.2%}  [{src}]"
            )
            updated_n += 1
        else:
            logger.warning(f"record_clv returned False for {order_id} — no matching row")

    logger.info(
        f"CLV snapshot complete: updated={updated_n}  "
        f"skipped_no_match={skipped_no_match}  "
        f"(games not in feed means already locked)"
    )


if __name__ == '__main__':
    snapshot_clv()
