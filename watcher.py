"""
Post-placement order watcher — re-evaluates resting orders periodically
and amends / cancels if prices or signals drift before games start.

Solves the "stale price" problem: morning-placed GTC orders sit on the
book for hours as Pinnacle's line and Kalshi's ask both evolve. Without
this, a 10am order that looked +5% EV can become -2% EV by 7pm and still
sit at the same price, waiting to be picked off.

Called from main.py after the initial placement loop when --watch is set.
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal

from kalshi.client import KalshiClient
from kalshi.markets import get_market_prices, amend_order, get_tick_for_price
from data.odds import get_pinnacle_odds
from risk import calculate_ev

logger = logging.getLogger(__name__)

PRICE_MAX = Decimal('0.99')


def _slippage_price(ask: Decimal, structure: str = 'linear_cent') -> Decimal:
    tick = get_tick_for_price(structure, ask)
    return min(ask + tick, PRICE_MAX)


def watch_and_amend(
    client: KalshiClient,
    active_orders: list[dict],
    fee_multiplier: float,
    interval_seconds: int,
    tick_threshold: int,
    ev_threshold_model: float,
    ev_threshold_pinnacle: float | None = None,  # legacy; per-order ev_threshold preferred
    pinnacle_refresh_seconds: int = 600,
    dry_run: bool = False,
) -> None:
    """
    Watch and amend active orders until all their game start_times pass.

    active_orders items: {
        order_id, ticker, side, action, count, submitted_price (Decimal),
        prob_src ('Pinnacle' / 'Blend' / 'model'), p_used, ev_threshold,
        start_time_utc, away_code, home_code, is_home_bet (bool),
        price_structure, label
    }

    Each order carries its own `ev_threshold` (the dynamic threshold that
    accepted it at placement). Watch uses that per-order value so an order
    placed with a permissive threshold doesn't get cancelled by a stricter
    live recomputation, and vice versa.
    """
    if not active_orders:
        logger.info("Watch: no active orders to monitor.")
        return

    logger.info(
        f"Watch mode: tracking {len(active_orders)} order(s); "
        f"poll every {interval_seconds}s; amend on Δ ≥ {tick_threshold}¢"
    )

    pinnacle_map: dict = {}
    last_pinnacle_fetch = 0.0
    # tick_threshold is in cents; convert to a dollar Decimal for ask-movement compare
    tick_threshold_dollars = Decimal(tick_threshold) * Decimal('0.01')

    while True:
        now_utc = datetime.now(timezone.utc)

        # Drop orders whose games have started — Kalshi closes those markets
        active_orders[:] = [o for o in active_orders if o['start_time_utc'] > now_utc]
        if not active_orders:
            logger.info("Watch: all tracked games have started — exiting.")
            return

        if time.time() - last_pinnacle_fetch > pinnacle_refresh_seconds:
            fresh = get_pinnacle_odds()
            if fresh:
                pinnacle_map = fresh
                last_pinnacle_fetch = time.time()
                logger.info(f"Watch: refreshed Pinnacle ({len(pinnacle_map)} games)")

        to_remove: list[dict] = []

        for o in active_orders:
            # Confirm still resting — it may have filled or been cancelled externally
            try:
                status_data = client.get(f"/portfolio/orders/{o['order_id']}")
                order_obj   = status_data.get('order', status_data)
                status      = order_obj.get('status', '')
            except Exception as e:
                logger.warning(f"Watch: status check failed for {o['order_id']}: {e}")
                continue

            if status != 'resting':
                logger.info(f"Watch: {o['ticker']} order {o['order_id']} is {status} — dropping")
                to_remove.append(o)
                continue

            prices = get_market_prices(client, o['ticker'])
            if not prices:
                continue
            new_ask   = prices['yes_ask']
            submitted = o['submitted_price']

            if o['prob_src'] in ('Pinnacle', 'Blend'):
                pair = (o['away_code'], o['home_code'])
                pin  = pinnacle_map.get(pair)
                if pin:
                    p_now = pin['home_prob'] if o['is_home_bet'] else pin['away_prob']
                else:
                    p_now = o['p_used']
            else:
                p_now = o['p_used']

            # Per-order threshold stored at placement (dynamic for blend, fixed for model)
            ev_threshold = o.get('ev_threshold',
                                 ev_threshold_model if o['prob_src'] == 'model'
                                 else (ev_threshold_pinnacle or 0.03))

            new_ev    = calculate_ev(p_now, new_ask, fee_multiplier)
            ask_delta = abs(new_ask - submitted)

            hdr = (
                f"Watch: {o['ticker']} sub=${submitted} ask=${new_ask} "
                f"Δ={ask_delta} p={p_now:.1%} EV={new_ev:+.1%}"
            )

            if new_ev < ev_threshold:
                if dry_run:
                    logger.info(f"{hdr}  would CANCEL (EV < {ev_threshold:.0%})")
                else:
                    try:
                        client.delete(f"/portfolio/orders/{o['order_id']}")
                        logger.info(f"{hdr}  CANCELLED (EV < {ev_threshold:.0%})")
                    except Exception as e:
                        logger.error(f"Watch: cancel failed for {o['order_id']}: {e}")
                to_remove.append(o)
                continue

            if ask_delta >= tick_threshold_dollars:
                new_price = _slippage_price(new_ask, o.get('price_structure', 'linear_cent'))
                if new_price == submitted:
                    continue

                if dry_run:
                    logger.info(f"{hdr}  would AMEND to ${new_price}")
                    o['submitted_price'] = new_price
                    continue

                result = amend_order(
                    client,
                    order_id=o['order_id'],
                    ticker=o['ticker'],
                    side=o['side'],
                    action=o['action'],
                    new_price_dollars=new_price,
                    count=o['count'],
                )
                if result:
                    new_order = result.get('order') or result
                    new_id    = new_order.get('order_id', o['order_id'])
                    o['order_id']        = new_id
                    o['submitted_price'] = new_price
                    logger.info(f"{hdr}  AMENDED to ${new_price} (id={new_id})")

        for o in to_remove:
            if o in active_orders:
                active_orders.remove(o)

        if not active_orders:
            logger.info("Watch: nothing left to track — exiting.")
            return

        time.sleep(interval_seconds)
