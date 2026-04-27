"""
Kalshi MLB market utilities.

KXMLBGAME ticker format:
  KXMLBGAME-{YYMONDD}{HHMM}{AWAY}{HOME}-{SIDE}
  e.g. KXMLBGAME-26APR161235WSHPIT-PIT
       date=2026-04-16, time=12:35 ET, away=WSH, home=PIT, side=PIT (YES if PIT wins)

All times in the ticker are Eastern Time (ET).
April–October = EDT (UTC-4); November–March = EST (UTC-5).
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .client import KalshiClient

logger = logging.getLogger(__name__)

SERIES_TICKER = 'KXMLBGAME'

ET_ZONE = ZoneInfo('America/New_York')

# Kalshi 2-3 char team codes → full team name
# Ordered longest-first so greedy matching works correctly
TEAM_CODES: dict[str, str] = {
    'NYY': 'New York Yankees',
    'NYM': 'New York Mets',
    'LAD': 'Los Angeles Dodgers',
    'LAA': 'Los Angeles Angels',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CWS': 'Chicago White Sox',
    'STL': 'St. Louis Cardinals',
    'HOU': 'Houston Astros',
    'ATL': 'Atlanta Braves',
    'PHI': 'Philadelphia Phillies',
    'MIN': 'Minnesota Twins',
    'MIL': 'Milwaukee Brewers',
    'PIT': 'Pittsburgh Pirates',
    'WSH': 'Washington Nationals',
    'CIN': 'Cincinnati Reds',
    'COL': 'Colorado Rockies',
    'SEA': 'Seattle Mariners',
    'TEX': 'Texas Rangers',
    'BAL': 'Baltimore Orioles',
    'CLE': 'Cleveland Guardians',
    'DET': 'Detroit Tigers',
    'TOR': 'Toronto Blue Jays',
    'MIA': 'Miami Marlins',
    'ATH': 'Oakland Athletics',
    'SF':  'San Francisco Giants',
    'SD':  'San Diego Padres',
    'TB':  'Tampa Bay Rays',
    'KC':  'Kansas City Royals',
    'AZ':  'Arizona Diamondbacks',
}

# Reverse map: full name → Kalshi code
TEAM_NAME_TO_CODE: dict[str, str] = {v: k for k, v in TEAM_CODES.items()}


MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}


def _parse_ticker(ticker: str) -> dict | None:
    """
    Parse a KXMLBGAME ticker into its components.

    Returns dict with:
        away_code, home_code, side_code  — Kalshi team codes
        away_name, home_name, side_name  — full team names
        start_time_utc                   — datetime (UTC, aware)
        is_yes_home                      — True if YES = home team wins
    Returns None if ticker can't be parsed.
    """
    # Strip the side suffix: KXMLBGAME-26APR161235WSHPIT-WSH → base + side
    parts = ticker.split('-')
    if len(parts) != 3 or parts[0] != SERIES_TICKER:
        return None

    date_time_teams = parts[1]  # e.g. 26APR161235WSHPIT
    side_code = parts[2]         # e.g. WSH

    # Parse date: first 7 chars = YYMONDD (e.g. 26APR16)
    m = re.match(r'^(\d{2})([A-Z]{3})(\d{2})(\d{4})(.+)$', date_time_teams)
    if not m:
        return None

    yy, mon_str, dd, hhmm, teams_str = m.groups()
    month = MONTH_MAP.get(mon_str)
    if not month:
        return None

    year = 2000 + int(yy)
    day  = int(dd)
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])

    # Parse team codes from concatenated string (e.g. WSHPIT, SFCIN, LAANYY)
    away_code, home_code = _split_team_codes(teams_str)
    if not away_code or not home_code:
        return None

    # Build start time in ET, then convert to UTC
    try:
        start_et = datetime(year, month, day, hour, minute, tzinfo=ET_ZONE)
        start_utc = start_et.astimezone(timezone.utc)
    except ValueError:
        return None

    away_name = TEAM_CODES.get(away_code, away_code)
    home_name = TEAM_CODES.get(home_code, home_code)
    side_name = TEAM_CODES.get(side_code, side_code)

    return {
        'away_code':      away_code,
        'home_code':      home_code,
        'side_code':      side_code,
        'away_name':      away_name,
        'home_name':      home_name,
        'side_name':      side_name,
        'start_time_utc': start_utc,
        'is_yes_home':    (side_code == home_code),
        'event_ticker':   f"{SERIES_TICKER}-{date_time_teams}",
    }


def _split_team_codes(teams_str: str) -> tuple[str | None, str | None]:
    """
    Split a concatenated pair of Kalshi team codes, e.g.:
      'WSHPIT' → ('WSH', 'PIT')
      'SFCIN'  → ('SF', 'CIN')
      'LAANYY' → ('LAA', 'NYY')

    Tries 3-char then 2-char split for the first team.
    """
    known = set(TEAM_CODES.keys())

    # Try 3-char first team
    if len(teams_str) >= 5:
        c1 = teams_str[:3]
        c2 = teams_str[3:]
        if c1 in known and c2 in known:
            return c1, c2

    # Try 2-char first team
    if len(teams_str) >= 4:
        c1 = teams_str[:2]
        c2 = teams_str[2:]
        if c1 in known and c2 in known:
            return c1, c2

    logger.warning(f"Could not split team codes from '{teams_str}'")
    return None, None


def get_series_fee_info(client: KalshiClient) -> tuple[str, float]:
    """
    Fetch the current fee_type and fee_multiplier for the KXMLBGAME series.

    Returns (fee_type, fee_multiplier). Kalshi can change fee schedules per
    series and runs promo periods, so the EV/Kelly math should be based on
    the live value rather than a hardcoded constant.

    Raises RuntimeError if the series isn't returning a quadratic fee — the
    bot's math assumes `fee_per_contract = multiplier * P * (1-P)`.
    """
    data = client.get(f'/series/{SERIES_TICKER}', auth=False)
    series = data.get('series', data)
    fee_type = series.get('fee_type', '')
    fee_multiplier = float(series.get('fee_multiplier', 0.0))

    if fee_type not in ('quadratic', 'quadratic_with_maker_fees'):
        raise RuntimeError(
            f"{SERIES_TICKER} fee_type is '{fee_type}', but EV/Kelly math "
            "assumes a quadratic fee structure. Update risk.py before trading."
        )
    if fee_multiplier <= 0:
        raise RuntimeError(
            f"{SERIES_TICKER} returned non-positive fee_multiplier={fee_multiplier}"
        )

    logger.info(f"Series fee: type={fee_type}  multiplier={fee_multiplier}")
    return fee_type, fee_multiplier


def get_mlb_markets(client: KalshiClient) -> list[dict]:
    """
    Fetch all open KXMLBGAME markets from Kalshi.
    Returns raw market dicts with an added 'parsed' key containing _parse_ticker() output.
    """
    markets = []
    cursor = None
    skipped_provisional = 0

    while True:
        params = {
            'series_ticker': SERIES_TICKER,
            'status': 'open',
            'mve_filter': 'exclude',
            'limit': 200,
        }
        if cursor:
            params['cursor'] = cursor

        try:
            data = client.get('/markets', params=params, auth=False)
        except Exception as e:
            logger.error(f"Error fetching KXMLBGAME markets: {e}")
            break

        batch = data.get('markets', [])
        for m in batch:
            # Skip provisional markets — Kalshi may change or remove them.
            # Per API docs: never trade is_provisional: true markets.
            if m.get('is_provisional'):
                logger.warning(f"Skipping provisional market {m.get('ticker')}")
                skipped_provisional += 1
                continue
            parsed = _parse_ticker(m['ticker'])
            if parsed:
                m['parsed'] = parsed
                markets.append(m)

        cursor = data.get('cursor')
        if not cursor:
            break

    if skipped_provisional:
        logger.info(f"Skipped {skipped_provisional} provisional market(s)")
    logger.info(f"Fetched {len(markets)} KXMLBGAME markets")
    return markets


def get_upcoming_games(markets: list[dict]) -> list[dict]:
    """
    From all open markets, return one entry per upcoming game
    scheduled for TODAY in Eastern Time and not yet started.
    Groups by event_ticker, returns both the YES-home and YES-away
    market tickers for each game.

    Returns list of dicts:
        event_ticker, away_code, home_code, away_name, home_name,
        start_time_utc, home_ticker, away_ticker
    """
    now = datetime.now(timezone.utc)
    today_et = datetime.now(ET_ZONE).date()
    games: dict[str, dict] = {}

    for m in markets:
        parsed = m.get('parsed', {})
        event = parsed.get('event_ticker')
        if not event:
            continue

        start = parsed.get('start_time_utc')
        if not start or start <= now:
            continue  # already started or in the past

        # Only today's games in Eastern Time
        if start.astimezone(ET_ZONE).date() != today_et:
            continue

        if event not in games:
            games[event] = {
                'event_ticker':   event,
                'away_code':      parsed['away_code'],
                'home_code':      parsed['home_code'],
                'away_name':      parsed['away_name'],
                'home_name':      parsed['home_name'],
                'start_time_utc': start,
                'home_ticker':    None,
                'away_ticker':    None,
                'tick_structure': {},   # ticker -> price_level_structure
            }

        structure = m.get('price_level_structure', 'linear_cent')
        games[event]['tick_structure'][m['ticker']] = structure

        if parsed['is_yes_home']:
            games[event]['home_ticker'] = m['ticker']
        else:
            games[event]['away_ticker'] = m['ticker']

    result = [g for g in games.values() if g['home_ticker'] and g['away_ticker']]
    result.sort(key=lambda g: g['start_time_utc'])
    logger.info(f"Upcoming games: {len(result)}")
    return result


def get_tick_for_price(structure: str, price: Decimal) -> Decimal:
    """
    Return the valid tick size at a given price for a Kalshi price-level structure.

    Kalshi tick structures (from fixed_point_migration docs):
      - linear_cent:         $0.01 everywhere
      - deci_cent:           $0.001 everywhere
      - tapered_deci_cent:   $0.001 in tails [0.00-0.10] and [0.90-1.00], $0.01 in the middle
      - banded_centi_cent:   variable — fall back to $0.01 (rare on sports)

    Unknown structures default to $0.01.
    """
    if structure == 'deci_cent':
        return Decimal('0.001')
    if structure == 'tapered_deci_cent':
        p = Decimal(price)
        if p < Decimal('0.10') or p > Decimal('0.90'):
            return Decimal('0.001')
        return Decimal('0.01')
    # linear_cent, banded_centi_cent, unknown -> safe default
    return Decimal('0.01')


def _ask_depth_within(no_bids: list, max_ask: Decimal) -> int:
    """
    Sum YES contracts available at asks ≤ max_ask.

    Kalshi orderbooks return only bids; the YES ask side is derived from
    `no_bids` via `yes_ask = 1 - no_bid`. A no_bid at $0.56 with 50 contracts
    means 50 YES contracts are available to buy at $0.44. no_bids is sorted
    ascending by price, so we walk from the end (highest no_bid = lowest YES
    ask) backwards, summing sizes while `no_bid_price ≥ 1 - max_ask`.

    Returns integer contract count (floored). Used to cap Kelly sizing so
    a big bet doesn't walk the book and destroy its own edge.
    """
    min_no_bid = Decimal('1.00') - max_ask
    total = Decimal('0')
    for price_str, size_str in reversed(no_bids):
        if Decimal(price_str) < min_no_bid:
            break
        total += Decimal(size_str)
    return int(total)  # floor — partial contract fragments aren't tradeable here


def get_market_prices(client: KalshiClient, ticker: str, depth_cents: int = 2) -> dict | None:
    """
    Fetch orderbook for a market and return best bid/ask prices + depth.

    Returns dict with:
        yes_bid, yes_ask, no_bid, no_ask  — Decimal prices
        implied_yes_prob                   — midpoint of YES bid/ask
        yes_bid_size, no_bid_size          — Decimal quantities at top of book
        yes_ask_depth                      — int, cumulative YES contracts available
                                             at asks within `depth_cents` of top ask
    """
    try:
        data = client.get(f'/markets/{ticker}/orderbook', auth=False)
        ob = data.get('orderbook_fp', {})
        yes_bids = ob.get('yes_dollars', [])
        no_bids  = ob.get('no_dollars', [])

        if not yes_bids and not no_bids:
            logger.warning(f"{ticker}: empty orderbook")
            return None

        best_yes_bid = Decimal(yes_bids[-1][0]) if yes_bids else Decimal('0.01')
        best_no_bid  = Decimal(no_bids[-1][0])  if no_bids  else Decimal('0.01')

        yes_ask = Decimal('1.00') - best_no_bid
        no_ask  = Decimal('1.00') - best_yes_bid

        yes_bid_size = Decimal(yes_bids[-1][1]) if yes_bids else Decimal('0')
        no_bid_size  = Decimal(no_bids[-1][1])  if no_bids  else Decimal('0')

        implied_yes_prob = (best_yes_bid + yes_ask) / 2

        max_ask_for_depth = yes_ask + Decimal(depth_cents) * Decimal('0.01')
        yes_ask_depth = _ask_depth_within(no_bids, max_ask_for_depth)

        return {
            'yes_bid':          best_yes_bid,
            'yes_ask':          yes_ask,
            'no_bid':           best_no_bid,
            'no_ask':           no_ask,
            'implied_yes_prob': implied_yes_prob,
            'yes_bid_size':     yes_bid_size,
            'no_bid_size':      no_bid_size,
            'yes_ask_depth':    yes_ask_depth,
        }
    except Exception as e:
        logger.error(f"Error fetching orderbook for {ticker}: {e}")
        return None


def get_settlement_for_ticker(client: KalshiClient, ticker: str) -> dict | None:
    """
    Fetch Kalshi's settlement record for a single market ticker, if any.

    Kalshi is the source of truth for realized PnL — their `market_result`
    includes 'void' (refunds at cost) which external box-score APIs won't
    surface, and their `fee_cost` reflects actual charged fees including
    rounding. Returns None if the market hasn't settled yet.
    """
    try:
        data = client.get('/portfolio/settlements', params={'ticker': ticker, 'limit': 10})
    except Exception as e:
        logger.warning(f"Settlement lookup failed for {ticker}: {e}")
        return None

    settlements = data.get('settlements', [])
    return settlements[0] if settlements else None


def amend_order(
    client: KalshiClient,
    order_id: str,
    ticker: str,
    side: str,
    action: str,
    new_price_dollars: Decimal,
    count: int,
    client_order_id: str | None = None,
) -> dict | None:
    """
    Amend an existing resting order's price (and/or size). Kalshi requires the
    original ticker/side/action even when only changing the price.

    Returns the API response dict (with old_order + order) on success, None on failure.
    """
    price_key = 'yes_price_dollars' if side == 'yes' else 'no_price_dollars'
    body: dict = {
        'ticker': ticker,
        'side':   side,
        'action': action,
        'count':  count,
        price_key: str(new_price_dollars),
    }
    if client_order_id:
        body['updated_client_order_id'] = client_order_id

    try:
        response = client.post(f'/portfolio/orders/{order_id}/amend', body)
        logger.info(
            f"Order amended: {order_id} | {ticker} {side.upper()} x{count} "
            f"-> ${new_price_dollars}"
        )
        return response
    except Exception as e:
        logger.error(f"Amend failed for {order_id}: {e}")
        return None


def get_open_positions(client: KalshiClient) -> set[str]:
    """
    Returns the set of KXMLBGAME market tickers where we hold a non-zero position.
    Used at startup to skip games we've already bet on (prevents doubling up
    if the bot restarts mid-session or runs twice).
    """
    try:
        data      = client.get('/portfolio/positions')
        positions = data.get('market_positions', data.get('positions', []))
        held = {
            p['ticker']
            for p in positions
            if p.get('position', 0) != 0 and p.get('ticker', '').startswith(SERIES_TICKER + '-')
        }
        if held:
            logger.info(f"Existing positions found: {held}")
        return held
    except Exception as e:
        logger.error(f"Could not fetch positions: {e}")
        return set()


def cancel_resting_mlb_orders(client: KalshiClient) -> int:
    """
    Cancels all resting orders on KXMLBGAME markets left over from previous sessions.
    Returns the number of orders cancelled.
    Called once at startup so we start each day with a clean slate.
    """
    cancelled = 0
    try:
        cursor = None
        while True:
            params: dict = {'status': 'resting', 'limit': 200}
            if cursor:
                params['cursor'] = cursor
            data   = client.get('/portfolio/orders', params=params)
            orders = data.get('orders', [])

            for order in orders:
                if not order.get('ticker', '').startswith(SERIES_TICKER + '-'):
                    continue
                order_id = order.get('order_id', '')
                if not order_id:
                    continue
                try:
                    client.delete(f'/portfolio/orders/{order_id}')
                    logger.info(f"Cancelled stale order {order_id} ({order['ticker']})")
                    cancelled += 1
                except Exception as e:
                    logger.warning(f"Could not cancel order {order_id}: {e}")

            cursor = data.get('cursor')
            if not cursor:
                break

    except Exception as e:
        logger.error(f"Could not fetch resting orders: {e}")

    if cancelled:
        logger.info(f"Cancelled {cancelled} stale KXMLBGAME order(s) from previous sessions")
    return cancelled


def place_order(
    client: KalshiClient,
    ticker: str,
    side: str,
    action: str,
    price_dollars: Decimal,
    contracts: int,
    price_level_structure: str = 'linear_cent',
    dry_run: bool = False,
) -> dict | None:
    """
    Place a limit buy order on Kalshi.

    Price is bumped by 1 tick before submission to absorb small orderbook
    movement between the fetch and the POST (~200ms). The tick size comes
    from the market's price_level_structure (default $0.01 for linear_cent).
    """
    import uuid
    client_order_id = str(uuid.uuid4())

    tick = get_tick_for_price(price_level_structure, price_dollars)
    submitted_price = min(price_dollars + tick, Decimal('0.99'))

    price_key = 'yes_price_dollars' if side == 'yes' else 'no_price_dollars'
    body = {
        'ticker':               ticker,
        'action':               action,
        'side':                 side,
        'count':                contracts,
        price_key:              str(submitted_price),
        'client_order_id':      client_order_id,
        'cancel_order_on_pause': True,
        'time_in_force':        'good_till_canceled',
    }

    if dry_run:
        logger.info(f"[DRY RUN] Would place order: {body}")
        return {'order_id': 'DRY_RUN', 'status': 'resting', **body}

    try:
        response = client.post('/portfolio/orders', body)
        logger.info(
            f"Order placed: {response.get('order', {}).get('order_id')} | "
            f"{ticker} {side.upper()} x{contracts} @ ${submitted_price} "
            f"(quoted ${price_dollars})"
        )
        return response
    except Exception as e:
        logger.error(f"Order failed for {ticker}: {e}")
        return None
