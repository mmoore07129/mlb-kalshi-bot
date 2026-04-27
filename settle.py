"""
Settler: close out unsettled ledger rows using Kalshi as the source of truth.

Kalshi's /portfolio/settlements gives:
- market_result: 'yes' | 'no' | 'void' | 'scalar'  (void = stake returned)
- fee_cost: actual fees charged (fixed-point dollar string)
- revenue:  net cents received from the settlement
- settled_time: RFC 3339

We use market_result to determine win/loss/void and Kalshi's reported fee_cost
for accurate realized PnL (including rounding quirks the formula can't predict).
If Kalshi hasn't settled yet, we skip and try again on the next run.

We cross-check each settled row against the MLB Stats API result for sanity;
if Kalshi and MLB disagree (rare but possible on disputed/voided games) we
trust Kalshi and log the divergence.

Run at end of day (or next morning) after games finish.
"""

import csv
import logging
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from kalshi.client import KalshiClient
from kalshi.markets import get_settlement_for_ticker
from ledger import LEDGER_PATH, record_outcome
from data.mlb_fetcher import mlb_get, KALSHI_TO_MLB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _to_mlb(team: str) -> str:
    return KALSHI_TO_MLB.get(team, team)


def get_mlb_results(game_date: str) -> dict[tuple[str, str], str]:
    """
    {(away_mlb_abb, home_mlb_abb): winner_mlb_abb} for all Final games on game_date.
    Used only as a cross-check against Kalshi's settlement.
    """
    data = mlb_get('/schedule', {
        'sportId': 1,
        'date':    game_date,
        'hydrate': 'linescore,team',
    })
    results: dict[tuple[str, str], str] = {}
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            if game.get('status', {}).get('abstractGameState', '') != 'Final':
                continue
            teams = game.get('teams', {})
            home  = teams.get('home', {})
            away  = teams.get('away', {})
            home_abb = home.get('team', {}).get('abbreviation', '')
            away_abb = away.get('team', {}).get('abbreviation', '')
            home_score = home.get('score', 0) or 0
            away_score = away.get('score', 0) or 0
            if home_abb and away_abb:
                winner = home_abb if home_score > away_score else away_abb
                results[(away_abb, home_abb)] = winner
    return results


def _compute_pnl_from_kalshi(settlement: dict, row: dict) -> tuple[str, float]:
    """
    Derive (outcome, pnl_dollars) for a single ledger row from Kalshi's settlement.

    Ledger rows from this bot are always YES-side buys, so:
      - market_result 'yes' → our YES won (payout $1/contract)
      - market_result 'no'  → our YES lost (contracts expire worthless)
      - market_result 'void' → refund at cost, pnl = -fee_cost (usually $0)
      - market_result 'scalar' → unexpected for KXMLBGAME; flag and treat conservatively
    """
    market_result = settlement.get('market_result', '')
    fee_cost_str  = settlement.get('fee_cost', '0')
    try:
        fee_cost = float(Decimal(str(fee_cost_str)))
    except Exception:
        fee_cost = 0.0

    contracts = int(row['contracts'])
    price     = float(row['price'])
    side      = row['side']  # always 'yes' for this bot

    if market_result == 'void':
        # Kalshi returns stake; any pre-settlement fees already rebated per spec
        return 'void', round(-fee_cost, 2)

    won = (market_result == side)
    if won:
        # Winning YES contracts pay $1 each; our cost was contracts*price
        gross_pnl = contracts * (1.0 - price)
        return 'win', round(gross_pnl - fee_cost, 2)

    if market_result in ('yes', 'no'):
        # Lost — contracts expire worthless, cost is the loss, no win fee
        return 'loss', round(-contracts * price, 2)

    # Scalar or unrecognized — log and treat as skip
    logger.warning(
        f"Unexpected market_result '{market_result}' for {row['ticker']} — "
        "leaving unsettled for manual review"
    )
    return '', 0.0


def settle_ledger() -> None:
    if not os.path.exists(LEDGER_PATH):
        logger.info("No ledger found.")
        return

    with open(LEDGER_PATH, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    unsettled = [
        r for r in rows
        if r['outcome'] == '' and r['order_id'] not in ('', 'DRY_RUN')
    ]

    if not unsettled:
        logger.info("No unsettled bets to process.")
        print_summary(rows)
        return

    client = KalshiClient(
        key_id=config.KEY_ID,
        private_key_path=config.PRIVATE_KEY_PATH,
        base_url=config.BASE_URL,
    )

    # Prefetch MLB final scores for cross-checking Kalshi's settlement
    dates = sorted(set(r['game_date'] for r in unsettled))
    logger.info(f"Settling {len(unsettled)} bets across dates: {dates}")
    mlb_by_date = {d: get_mlb_results(d) for d in dates}

    settled = skipped = mismatches = 0

    for row in unsettled:
        ticker    = row['ticker']
        order_id  = row['order_id']
        bet_label = row['bet_label']

        settlement = get_settlement_for_ticker(client, ticker)
        if settlement is None:
            logger.info(f"  {bet_label} ({ticker}): no Kalshi settlement yet — skipping")
            skipped += 1
            continue

        outcome, pnl = _compute_pnl_from_kalshi(settlement, row)
        if not outcome:
            skipped += 1
            continue

        # Cross-check: does MLB's final agree with Kalshi's market_result?
        game_date = row['game_date']
        home_mlb  = _to_mlb(row['home_team'])
        away_mlb  = _to_mlb(row['away_team'])
        mlb_winner = mlb_by_date.get(game_date, {}).get((away_mlb, home_mlb))

        if mlb_winner and outcome in ('win', 'loss'):
            # Derive expected outcome from MLB's winner + bet label
            bet_team_code = _extract_bet_team(row)
            expected_win  = (mlb_winner == _to_mlb(bet_team_code)) if bet_team_code else None
            actual_win    = (outcome == 'win')
            if expected_win is not None and expected_win != actual_win:
                logger.warning(
                    f"  MLB/Kalshi divergence for {bet_label} ({ticker}): "
                    f"MLB winner={mlb_winner}, Kalshi result={settlement.get('market_result')}; "
                    "trusting Kalshi"
                )
                mismatches += 1

        record_outcome(order_id, outcome, pnl)
        logger.info(
            f"  {bet_label} ({row['away_team']}@{row['home_team']}): "
            f"{outcome}, PnL ${pnl:+.2f}  "
            f"[kalshi_result={settlement.get('market_result')}]"
        )
        settled += 1

    summary = f"\nSettled {settled} bets, {skipped} still pending"
    if mismatches:
        summary += f", {mismatches} MLB/Kalshi divergences (Kalshi trusted)"
    logger.info(summary + '.')

    # Re-read after updates
    with open(LEDGER_PATH, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    print_summary(rows)


def _extract_bet_team(row: dict) -> str:
    """Return the team code we bet on. Prefer bet_team_code column; fall back to parsing bet_label."""
    code = row.get('bet_team_code', '').strip()
    if code:
        return code
    # Legacy rows: bet_label format "XXX wins"
    return row.get('bet_label', '').replace(' wins', '').strip()


def print_summary(rows: list[dict]) -> None:
    live    = [r for r in rows if r['order_id'] not in ('', 'DRY_RUN')]
    settled = [r for r in live if r['outcome'] in ('win', 'loss', 'void')]

    if not settled:
        logger.info("No settled bets yet.")
        return

    wins   = [r for r in settled if r['outcome'] == 'win']
    losses = [r for r in settled if r['outcome'] == 'loss']
    voids  = [r for r in settled if r['outcome'] == 'void']
    total_pnl  = sum(float(r['pnl']) for r in settled if r['pnl'] not in ('', None))
    resolved   = [r for r in settled if r['outcome'] in ('win', 'loss')]
    total_cost = sum(float(r['cost']) for r in resolved)

    logger.info("\n=== PERFORMANCE SUMMARY ===")
    logger.info(f"Bets:        {len(settled)}  ({len(wins)}W / {len(losses)}L / {len(voids)}V)")
    if resolved:
        logger.info(f"Win rate:    {len(wins)/len(resolved):.1%}  (excluding voids)")
    logger.info(f"Wagered:     ${total_cost:.2f}")
    logger.info(f"Net PnL:     ${total_pnl:+.2f}")
    if total_cost > 0:
        logger.info(f"ROI:         {total_pnl/total_cost:.1%}")

    dates = sorted(set(r['game_date'] for r in settled))
    if len(dates) > 1:
        logger.info("\nBy date:")
        for d in dates:
            day_rows = [r for r in settled if r['game_date'] == d]
            day_pnl  = sum(float(r['pnl']) for r in day_rows if r['pnl'] not in ('', None))
            day_wins = sum(1 for r in day_rows if r['outcome'] == 'win')
            logger.info(f"  {d}:  {day_wins}/{len(day_rows)} wins,  ${day_pnl:+.2f}")

    # CLV breakdown — CLV data is available on any bet that got a snapshot,
    # regardless of whether the game has settled yet. This section works off
    # `live` rather than `settled` so CLV shows up as soon as we've seen a close.
    _print_clv_section(live)


def _print_clv_section(live_rows: list[dict]) -> None:
    """
    CLV breakdown by source, edge bucket, and rolling-100. CLV is the leading
    indicator — positive PnL with flat CLV means you ran hot, not that you're
    good. Track this more closely than win rate.
    """
    with_clv = [r for r in live_rows if r.get('clv_prob') not in ('', None)]
    if not with_clv:
        logger.info("\n=== CLV SUMMARY ===")
        logger.info("No CLV snapshots yet (clv_snapshot.py hasn't written to any row).")
        return

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    all_prob = [float(r['clv_prob']) for r in with_clv]
    all_pct  = [float(r['clv_pct']) for r in with_clv if r.get('clv_pct') not in ('', None)]

    logger.info("\n=== CLV SUMMARY ===")
    logger.info(f"Bets with CLV:  {len(with_clv)}")
    logger.info(f"Mean clv_prob:  {_mean(all_prob):+.2%}  (market fair moved toward our pick on average)")
    logger.info(f"Mean clv_pct:   {_mean(all_pct):+.2%}   (our entry price vs closing fair)")

    # Rolling 100 (most recent by timestamp). Noise targets from your notes:
    # +1% CLV_prob confirms at ~300 bets, +2% at ~150. Use rolling view to watch
    # for drift once enough samples accumulate.
    with_clv_sorted = sorted(with_clv, key=lambda r: r.get('timestamp', ''))
    recent = with_clv_sorted[-100:]
    if len(recent) < len(with_clv_sorted):
        recent_prob = [float(r['clv_prob']) for r in recent]
        logger.info(f"Rolling-100 mean clv_prob: {_mean(recent_prob):+.2%}  (last {len(recent)} bets)")

    # Per prob_src — lets us see if model-fallback CLV diverges from Pinnacle CLV
    logger.info("\nBy prob_src:")
    sources = sorted(set(r.get('prob_src', '') for r in with_clv))
    for src in sources:
        src_rows = [r for r in with_clv if r.get('prob_src', '') == src]
        if not src_rows:
            continue
        vals = [float(r['clv_prob']) for r in src_rows]
        logger.info(f"  {src or '(unknown)':10s} n={len(src_rows):3d}  mean clv_prob={_mean(vals):+.2%}")

    # Per edge bucket — lets us see whether low-edge bets (3-4%) are +CLV at all.
    # If they aren't, raise the threshold for that source.
    logger.info("\nBy edge bucket (net EV at placement):")
    buckets = [
        ('0–3%',  lambda ev: ev < 0.03),
        ('3–4%',  lambda ev: 0.03 <= ev < 0.04),
        ('4–5%',  lambda ev: 0.04 <= ev < 0.05),
        ('5–6%',  lambda ev: 0.05 <= ev < 0.06),
        ('6%+',   lambda ev: ev >= 0.06),
    ]
    for label, pred in buckets:
        bucket_rows = []
        for r in with_clv:
            try:
                ev = float(r.get('ev', 0))
            except (ValueError, TypeError):
                continue
            if pred(ev):
                bucket_rows.append(r)
        if not bucket_rows:
            continue
        vals = [float(r['clv_prob']) for r in bucket_rows]
        logger.info(f"  {label:8s}  n={len(bucket_rows):3d}  mean clv_prob={_mean(vals):+.2%}")


if __name__ == '__main__':
    settle_ledger()
