"""
MLB Kalshi Trading Bot — XGBoost edition
=========================================
Run daily before MLB games start.

Prerequisites: run 'python models/train.py' once to build xgb_model.pkl.

Usage:
  python main.py            # Live mode: place orders and exit
  python main.py --dry-run  # Simulate without placing real orders
  python main.py --watch    # Live + watch: after placement, re-evaluate
                            # resting orders periodically until games start
"""

import os
import sys
import logging
import argparse
from decimal import Decimal
from datetime import datetime, date
from zoneinfo import ZoneInfo

import config
from kalshi.client import KalshiClient
from kalshi.markets import (
    get_mlb_markets,
    get_upcoming_games,
    get_market_prices,
    get_open_positions,
    cancel_resting_mlb_orders,
    place_order,
    get_series_fee_info,
    get_tick_for_price,
)
from data.mlb_fetcher import get_team_id_map, get_probable_pitchers, KALSHI_TO_MLB
from data.game_logs import get_current_season_stats, compute_inference_sp_stats
from data.odds import get_pinnacle_odds
from models.predictor import predict
from risk import calculate_ev, flat_unit_contracts
from ledger import record_bet
from watcher import watch_and_amend
from daily_stats import new_game_stats, write_rows as write_daily_stats
from paper_trade import (
    new_paper_row,
    gap_aware_threshold,
    finalize_row as finalize_paper_row,
    write_rows as write_paper_trade,
)

# Windows cmd defaults to cp1252; reconfigure stdout to UTF-8 so native runs
# don't throw UnicodeEncodeError on our Δ / ─ / ≥ characters. File handler
# is already utf-8 below.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# logging.FileHandler fails if the directory doesn't exist — ensure it first.
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def run(dry_run: bool = False, watch: bool = False) -> None:
    dry_run    = dry_run or config.DRY_RUN
    mode_label = '[DRY RUN] ' if dry_run else ''

    logger.info('=' * 60)
    logger.info(f"{mode_label}MLB Kalshi Bot starting — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info('=' * 60)

    # ── Kalshi connection ────────────────────────────────────────────────────
    client = KalshiClient(
        key_id=config.KEY_ID,
        private_key_path=config.PRIVATE_KEY_PATH,
        base_url=config.BASE_URL,
    )

    logger.info("Checking exchange status...")
    try:
        status = client.get('/exchange/status', auth=False)
        if not status.get('exchange_active'):
            logger.error("Exchange inactive. Exiting.")
            return
        if not status.get('trading_active'):
            logger.warning("Trading not yet active — continuing to queue orders.")
    except Exception as e:
        logger.error(f"Could not reach Kalshi: {e}")
        return

    # Pull the live fee multiplier for KXMLBGAME so EV/Kelly use the real rate,
    # not a hardcoded 0.07 that could drift if Kalshi changes the schedule.
    try:
        _, fee_multiplier = get_series_fee_info(client)
    except Exception as e:
        fee_multiplier = config.FEE_MULTIPLIER_FALLBACK
        logger.warning(f"Could not fetch series fee info — falling back to {fee_multiplier}: {e}")

    cancel_resting_mlb_orders(client)
    open_positions = get_open_positions(client)

    logger.info("Fetching account balance...")
    try:
        bal_data = client.get('/portfolio/balance')
        bankroll = bal_data.get('balance', 0) / 100.0
        logger.info(f"Balance: ${bankroll:.2f}")
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return

    if bankroll < config.MIN_BET_DOLLARS:
        logger.error(f"Bankroll ${bankroll:.2f} below minimum. Exiting.")
        return

    starting_bankroll = bankroll
    session_exposure  = 0.0   # total $ committed to resting orders this session

    # ── Kalshi markets ───────────────────────────────────────────────────────
    logger.info("Fetching KXMLBGAME markets...")
    all_markets    = get_mlb_markets(client)
    upcoming_games = get_upcoming_games(all_markets)

    if not upcoming_games:
        logger.info("No upcoming MLB games today. Exiting.")
        return

    logger.info(f"Upcoming games: {len(upcoming_games)}")
    for g in upcoming_games:
        et = g['start_time_utc'].astimezone(
            ZoneInfo('America/New_York')
        ).strftime('%I:%M %p ET')
        logger.info(f"  {g['away_name']} @ {g['home_name']}  [{et}]")

    # ── Load data (once, shared across all games) ────────────────────────────
    logger.info("Loading season stats and Pinnacle odds...")
    team_id_map  = get_team_id_map()
    season_stats = get_current_season_stats(config.MLB_SEASON, team_id_map, KALSHI_TO_MLB)
    pinnacle_map = get_pinnacle_odds()
    probable_map = get_probable_pitchers(game_date=date.today())

    if pinnacle_map:
        logger.info(f"Blended sharp odds loaded: {len(pinnacle_map)} games")
    else:
        logger.info(f"Sharp odds unavailable — model fallback only (MIN_EV raised to {config.MODEL_MIN_EV:.0%})")

    # ── Evaluate each game ───────────────────────────────────────────────────
    bets_placed   = 0
    total_staked  = 0.0
    active_orders: list[dict] = []    # populated when watch mode is on
    daily_rows:    list[dict] = []    # per-game decision diagnostics
    paper_rows:    list[dict] = []    # gap-aware-threshold paper-trade decisions

    for game in upcoming_games:
        home_code   = game['home_code']
        away_code   = game['away_code']
        home_name   = game['home_name']
        away_name   = game['away_name']
        home_ticker = game['home_ticker']
        away_ticker = game['away_ticker']

        game_date_et = game['start_time_utc'].astimezone(ZoneInfo('America/New_York')).date()
        game_stats = new_game_stats(
            game_date=game_date_et,
            away_code=away_code, home_code=home_code,
            mode=('dry' if dry_run else 'live'),
        )

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Analyzing: {away_name} ({away_code}) @ {home_name} ({home_code})")

        if home_ticker in open_positions or away_ticker in open_positions:
            logger.info(f"Already have a position in this game — skipping")
            game_stats['reason'] = 'already-positioned'
            daily_rows.append(game_stats)
            continue

        home_mlb = KALSHI_TO_MLB.get(home_code, home_code)
        away_mlb = KALSHI_TO_MLB.get(away_code, away_code)

        # Probable starters — names for logging, IDs fed into the model
        pitchers = probable_map.get(
            (away_mlb, home_mlb),
            probable_map.get((away_code, home_code), {}),
        )
        home_sp_info = pitchers.get('home_pitcher') or {}
        away_sp_info = pitchers.get('away_pitcher') or {}
        home_sp_name = home_sp_info.get('name', 'TBD')
        away_sp_name = away_sp_info.get('name', 'TBD')
        home_sp_id   = home_sp_info.get('id')
        away_sp_id   = away_sp_info.get('id')
        logger.info(f"Starters — Home: {home_sp_name}  Away: {away_sp_name}")
        game_stats['home_sp'] = home_sp_name
        game_stats['away_sp'] = away_sp_name

        # Team stats
        home_stats = season_stats.get(home_mlb) or season_stats.get(home_code)
        away_stats = season_stats.get(away_mlb) or season_stats.get(away_code)

        if not home_stats or not away_stats:
            logger.warning(f"Missing stats for {home_code} or {away_code} — skipping")
            game_stats['reason'] = 'missing-stats'
            daily_rows.append(game_stats)
            continue

        # SP rolling stats (L3/L5/YTD + days rest). Prior-fallback when a starter
        # is unknown or has <3 prior starts this season; keeps inference resilient
        # when a rookie or mid-season callup gets the ball.
        home_sp_stats = compute_inference_sp_stats(home_sp_id, config.MLB_SEASON, date.today())
        away_sp_stats = compute_inference_sp_stats(away_sp_id, config.MLB_SEASON, date.today())

        logger.info(
            f"YTD   — Home: {home_stats['wins_ytd']}-{home_stats['losses_ytd']} "
            f"pyth={home_stats['pyth_ytd']:.3f}  "
            f"Away: {away_stats['wins_ytd']}-{away_stats['losses_ytd']} "
            f"pyth={away_stats['pyth_ytd']:.3f}"
        )
        logger.info(
            f"Roll5 — Home: {home_stats['r5_wins']}-{home_stats['r5_losses']} "
            f"pyth={home_stats['r5_pyth']:.3f}  "
            f"Away: {away_stats['r5_wins']}-{away_stats['r5_losses']} "
            f"pyth={away_stats['r5_pyth']:.3f}"
        )

               # ── Model prediction ─────────────────────────────────────────────────
        prediction   = predict(
            home_code, away_code, home_stats, away_stats,
            home_sp_stats=home_sp_stats, away_sp_stats=away_sp_stats,
        )
        model_p_home = prediction['p_home']
        logger.info(f"Model: {home_name}={model_p_home:.1%}  {away_name}={prediction['p_away']:.1%}")
        game_stats['model_p_home'] = round(model_p_home, 4)

        # ── Pinnacle lookup — BEFORE confidence gate ─────────────────────────
        # Primary signal is Pinnacle vs Kalshi spread.
        # Model fires only as fallback when Pinnacle has no line.
        pinnacle = pinnacle_map.get((away_code, home_code))
        if pinnacle:
            p_home    = pinnacle['home_prob']
            p_away    = pinnacle['away_prob']
            prob_std  = pinnacle.get('prob_std', 0.0)
            n_books   = len(pinnacle.get('books_used', []) or [])
            prob_src  = 'Blend' if n_books >= 2 else 'Pinnacle'

            # Dynamic EV threshold: tighter when sharp books disagree.
            # With ≥2 books we can measure disagreement directly; with 1 book we
            # fall back to the conservative fixed PINNACLE_MIN_EV.
            if n_books >= 2:
                ev_threshold = max(
                    config.EV_THRESHOLD_BASE,
                    min(
                        config.EV_THRESHOLD_CEIL,
                        config.EV_THRESHOLD_BASE + config.EV_THRESHOLD_STD_MULT * prob_std,
                    ),
                )
            else:
                ev_threshold = config.PINNACLE_MIN_EV

            model_gap = abs(p_home - model_p_home)
            books_str = ",".join(pinnacle.get('books_used', []) or [])
            logger.info(
                f"Blend ({books_str}): {home_code}={p_home:.1%}  {away_code}={p_away:.1%}  "
                f"std={prob_std:.1%}  threshold={ev_threshold:.1%}  "
                f"(model gap: {model_gap:.1%})"
            )
            game_stats['pin_home_prob'] = round(float(p_home), 4)
            game_stats['pin_src']       = prob_src
            game_stats['pin_std']       = round(float(prob_std), 4)
            game_stats['model_pin_gap'] = round(model_gap, 4)
            game_stats['ev_threshold']  = round(ev_threshold, 4)

            # Sanity check: a large Pinnacle-vs-model gap usually means something
            # the model doesn't know about (late scratch, weather, bullpen news).
            # Safer to sit out than trust either signal in isolation.
            if model_gap > config.MODEL_PINNACLE_GAP_MAX:
                logger.warning(
                    f"Model-Blend gap {model_gap:.1%} exceeds "
                    f"{config.MODEL_PINNACLE_GAP_MAX:.0%} — skipping {away_code} @ {home_code}"
                )
                game_stats['reason'] = 'model-pin-gap'
                daily_rows.append(game_stats)
                continue
        else:
            # No Pinnacle line — fall back to model with stricter confidence + EV gate
            p_home   = model_p_home
            p_away   = prediction['p_away']
            prob_src = 'model'
            ev_threshold = config.MODEL_MIN_EV
            game_stats['pin_src']      = 'model'
            game_stats['ev_threshold'] = round(ev_threshold, 4)

            if max(model_p_home, 1.0 - model_p_home) < config.CONFIDENCE_THRESHOLD:
                logger.info(
                    f"Model p_home={model_p_home:.1%} — stronger side below "
                    f"{config.CONFIDENCE_THRESHOLD:.0%} confidence, no Pinnacle line — skipping"
                )
                game_stats['reason'] = 'confidence-gate'
                daily_rows.append(game_stats)
                continue
            logger.info(f"No Pinnacle line — using model ({home_name}={p_home:.1%}), EV threshold {ev_threshold:.0%}")

        # ── Orderbook ────────────────────────────────────────────────────────
        home_prices = get_market_prices(client, home_ticker)
        away_prices = get_market_prices(client, away_ticker)
        if not home_prices or not away_prices:
            logger.warning(f"No orderbook for {home_ticker}/{away_ticker} — skipping")
            game_stats['reason'] = 'no-orderbook'
            daily_rows.append(game_stats)
            continue

        logger.info(
            f"Kalshi: {home_code} ask={home_prices['yes_ask']} depth={home_prices['yes_ask_depth']}  |  "
            f"{away_code} ask={away_prices['yes_ask']} depth={away_prices['yes_ask_depth']}"
        )
        game_stats['kalshi_home_ask'] = float(home_prices['yes_ask'])
        game_stats['kalshi_away_ask'] = float(away_prices['yes_ask'])

        # ── EV (net of Kalshi taker fee) ─────────────────────────────────────
        ev_home = calculate_ev(p_home, home_prices['yes_ask'], fee_multiplier)
        ev_away = calculate_ev(p_away, away_prices['yes_ask'], fee_multiplier)
        logger.info(
            f"EV — {home_code} YES: {ev_home:+.1%}  "
            f"{away_code} YES: {ev_away:+.1%}  "
            f"(threshold: {ev_threshold:.0%})"
        )
        game_stats['ev_home_net'] = round(ev_home, 4)
        game_stats['ev_away_net'] = round(ev_away, 4)
        game_stats['best_ev']     = round(max(ev_home, ev_away), 4)

        # ── Paper-trade harness: gap-aware EV threshold ──────────────────────
        # Logging only. Runs alongside the live decision so we can compare
        # realized CLV after 30-60 days. NOT a bot behavior change.
        paper_row = None
        if pinnacle:
            ga_thresh = gap_aware_threshold(
                config.EV_THRESHOLD_BASE, config.EV_THRESHOLD_CEIL,
                float(prob_std), float(model_gap),
            )
            if ev_home >= ga_thresh and ev_home >= ev_away:
                ga_decision = f'bet-{home_code}'
            elif ev_away >= ga_thresh:
                ga_decision = f'bet-{away_code}'
            else:
                ga_decision = 'no-edge'
            paper_row = new_paper_row(
                game_date=date.today(), away_code=away_code, home_code=home_code,
                mode='dry-run' if dry_run else 'live',
            )
            paper_row['model_p_home']           = round(float(model_p_home), 4)
            paper_row['pin_home_prob']          = round(float(p_home), 4)
            paper_row['model_pin_gap']          = round(float(model_gap), 4)
            paper_row['prob_std']               = round(float(prob_std), 4)
            paper_row['kalshi_home_ask']        = float(home_prices['yes_ask'])
            paper_row['kalshi_away_ask']        = float(away_prices['yes_ask'])
            paper_row['ev_home_net']            = round(ev_home, 4)
            paper_row['ev_away_net']            = round(ev_away, 4)
            paper_row['ev_threshold_current']   = round(ev_threshold, 4)
            paper_row['ev_threshold_gap_aware'] = round(ga_thresh, 4)
            paper_row['decision_gap_aware']     = ga_decision

        # ── Pick best bet ────────────────────────────────────────────────────
        bet_ticker = bet_side = bet_ev = bet_price = bet_p = bet_label = None
        bet_team_code = None
        is_home_bet   = False
        if ev_home >= ev_threshold and ev_home >= ev_away:
            bet_ticker, bet_side  = home_ticker, 'yes'
            bet_ev, bet_price     = ev_home, home_prices['yes_ask']
            bet_p, bet_label      = p_home, f"{home_code} wins"
            bet_team_code, is_home_bet = home_code, True
        elif ev_away >= ev_threshold:
            bet_ticker, bet_side  = away_ticker, 'yes'
            bet_ev, bet_price     = ev_away, away_prices['yes_ask']
            bet_p, bet_label      = p_away, f"{away_code} wins"
            bet_team_code, is_home_bet = away_code, False

        if bet_ticker is None:
            logger.info(f"No edge — skipping {away_code} @ {home_code}")
            game_stats['reason'] = 'no-edge'
            daily_rows.append(game_stats)
            if paper_row is not None:
                finalize_paper_row(paper_row, 'no-edge', None)
                paper_rows.append(paper_row)
            continue

        # ── Flat-unit sizing ─────────────────────────────────────────────────
        # Every passing bet is FLAT_UNIT_DOLLARS. available_cash gates so we
        # don't oversubscribe the bankroll across multiple games in a session;
        # depth gates so a single bet doesn't walk the book past the quoted ask.
        chosen_prices = home_prices if bet_ticker == home_ticker else away_prices
        depth_cap     = chosen_prices.get('yes_ask_depth', 0)

        available_cash = bankroll - session_exposure
        contracts, cost = flat_unit_contracts(
            ask_price=bet_price,
            available_cash=available_cash,
            unit_dollars=config.FLAT_UNIT_DOLLARS,
            min_bet_dollars=config.MIN_BET_DOLLARS,
            max_depth_contracts=depth_cap,
        )
        if contracts == 0:
            logger.info(f"Insufficient cash for unit (avail=${available_cash:.2f}) — skipping")
            game_stats['reason'] = 'insufficient-cash'
            daily_rows.append(game_stats)
            if paper_row is not None:
                finalize_paper_row(paper_row, 'insufficient-cash', bet_team_code)
                paper_rows.append(paper_row)
            continue

        logger.info(
            f"BET: {bet_label} | {bet_ticker} YES x{contracts} @ ${bet_price} | "
            f"cost=${cost:.2f} | EV={bet_ev:+.1%} | p={bet_p:.1%} [{prob_src}]"
        )

        # ── Place order ──────────────────────────────────────────────────────
        price_structure = game.get('tick_structure', {}).get(bet_ticker, 'linear_cent')
        result = place_order(
            client=client,
            ticker=bet_ticker,
            side=bet_side,
            action='buy',
            price_dollars=bet_price,
            contracts=contracts,
            price_level_structure=price_structure,
            dry_run=dry_run,
        )
        if result:
            # Extract order_id from Kalshi response (live) or dry-run sentinel
            order_id = (
                result.get('order', {}).get('order_id')
                or result.get('order_id', '')
                or ''
            )
            bets_placed      += 1
            total_staked     += cost
            session_exposure += cost
            logger.info(f"Order submitted (id={order_id})  session_exposure=${session_exposure:.2f}")

            # Capture the actually-submitted price (place_order bumps by 1 tick
            # and caps at $0.99). Watch mode compares against this, not the
            # quoted ask, so it doesn't amend away a buffer that's still fine.
            tick = get_tick_for_price(price_structure, bet_price)
            submitted_price = min(bet_price + tick, Decimal('0.99'))

            if watch and order_id and order_id != 'DRY_RUN':
                active_orders.append({
                    'order_id':        order_id,
                    'ticker':          bet_ticker,
                    'side':            bet_side,
                    'action':          'buy',
                    'count':           contracts,
                    'submitted_price': submitted_price,
                    'price_structure': price_structure,
                    'prob_src':        prob_src,
                    'p_used':          bet_p,
                    'ev_threshold':    ev_threshold,  # the dynamic threshold that accepted this bet
                    'start_time_utc':  game['start_time_utc'],
                    'away_code':       away_code,
                    'home_code':       home_code,
                    'is_home_bet':     is_home_bet,
                    'label':           bet_label,
                })

            # ── Write to ledger ──────────────────────────────────────────────
            record_bet(
                game_date     = date.today(),
                home_team     = home_mlb,
                away_team     = away_mlb,
                bet_label     = bet_label,
                bet_team_code = bet_team_code,
                ticker        = bet_ticker,
                side          = bet_side,
                contracts     = contracts,
                price         = bet_price,
                cost          = cost,
                ev            = bet_ev,
                p_used        = bet_p,
                model_p_home  = model_p_home,
                prob_src      = prob_src,
                order_id      = order_id,
            )
            game_stats['reason']        = 'placed'
            game_stats['bet_contracts'] = contracts
            game_stats['bet_price']     = float(bet_price)
            game_stats['bet_cost']      = round(cost, 2)
            game_stats['order_id']      = order_id
            daily_rows.append(game_stats)
            if paper_row is not None:
                finalize_paper_row(paper_row, 'placed', bet_team_code)
                paper_rows.append(paper_row)
        else:
            # place_order returned falsy — API error, dry_run sentinel collision,
            # or Kalshi rejected. Log a stats row so we know this happened.
            game_stats['reason']        = 'place-failed'
            game_stats['bet_contracts'] = contracts
            game_stats['bet_price']     = float(bet_price)
            game_stats['bet_cost']      = round(cost, 2)
            daily_rows.append(game_stats)
            if paper_row is not None:
                finalize_paper_row(paper_row, 'place-failed', bet_team_code)
                paper_rows.append(paper_row)

    # ── Session summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{mode_label}Session complete")
    logger.info(f"  Games analyzed : {len(upcoming_games)}")
    logger.info(f"  Bets placed    : {bets_placed}")
    logger.info(f"  Total staked   : ${total_staked:.2f}")
    logger.info(f"  Session exposure: ${session_exposure:.2f} ({session_exposure/starting_bankroll:.1%} of bankroll)")
    logger.info(f"  Balance (live) : ${bankroll:.2f}  (unchanged — orders are resting)")
    logger.info('=' * 60)

    # Persist per-game diagnostics — skip in dry-run so test runs don't pollute
    # the daily_stats file that feeds the "why isn't it betting?" analysis.
    if not dry_run and daily_rows:
        write_daily_stats(daily_rows)
        logger.info(f"daily_stats: wrote {len(daily_rows)} rows")
    if not dry_run and paper_rows:
        write_paper_trade(paper_rows)
        n_changed = sum(1 for r in paper_rows if r.get('gap_aware_changed') == '1')
        logger.info(f"paper_trade: wrote {len(paper_rows)} rows  ({n_changed} where gap-aware decision differs)")

    # ── Watch mode: re-evaluate resting orders until games start ─────────────
    if watch and active_orders:
        watch_and_amend(
            client                = client,
            active_orders         = active_orders,
            fee_multiplier        = fee_multiplier,
            interval_seconds      = config.WATCH_INTERVAL_SECONDS,
            tick_threshold        = config.WATCH_AMEND_TICK_THRESHOLD,
            ev_threshold_model    = config.MODEL_MIN_EV,
            dry_run               = dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLB Kalshi Trading Bot')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate without placing real orders')
    parser.add_argument('--watch', action='store_true',
                        help='After placement, re-evaluate resting orders until games start')
    args = parser.parse_args()
    run(dry_run=args.dry_run, watch=args.watch)
