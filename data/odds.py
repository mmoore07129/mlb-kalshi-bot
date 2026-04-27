"""
Sharp-book odds fetcher via The Odds API.

Architecture: blend vig-free probabilities from multiple sharp books to form a
better fair-value prior than any single book. Pinnacle is the gold standard;
LowVig.ag (reduced-juice) and BetOnline track Pinnacle closely and add cheap
redundancy when Pinnacle's feed is stale (they're cited in the Unabated Line
methodology for the same reason).

Weight scheme:
  - pinnacle:    0.55  (sharpest single book)
  - lowvig:      0.30  (reduced-juice, independent quote)
  - betonlineag: 0.15  (tracks Pinnacle, useful when Pinnacle lags)

If a book is missing for a given game, remaining weights are renormalized so
coverage never drops. `prob_std` is returned alongside the blend so `main.py`
can tighten the EV threshold when books disagree.

Configure ODDS_API_KEY in .env (free tier at https://the-odds-api.com/).
When the key is empty this module returns {} silently and model probabilities
are used instead.
"""

import logging
import statistics
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# Sharp books and their weights in the blend. Order matters only for logging.
BOOK_WEIGHTS: dict[str, float] = {
    'pinnacle':    0.55,
    'lowvig':      0.30,
    'betonlineag': 0.15,
}

# The Odds API full team name → Kalshi team code
_NAME_TO_KALSHI: dict[str, str] = {
    'New York Yankees':      'NYY',
    'New York Mets':         'NYM',
    'Los Angeles Dodgers':   'LAD',
    'Los Angeles Angels':    'LAA',
    'Boston Red Sox':        'BOS',
    'Chicago Cubs':          'CHC',
    'Chicago White Sox':     'CWS',
    'St. Louis Cardinals':   'STL',
    'Houston Astros':        'HOU',
    'Atlanta Braves':        'ATL',
    'Philadelphia Phillies': 'PHI',
    'Minnesota Twins':       'MIN',
    'Milwaukee Brewers':     'MIL',
    'Pittsburgh Pirates':    'PIT',
    'Washington Nationals':  'WSH',
    'Cincinnati Reds':       'CIN',
    'Colorado Rockies':      'COL',
    'Seattle Mariners':      'SEA',
    'Texas Rangers':         'TEX',
    'Baltimore Orioles':     'BAL',
    'Cleveland Guardians':   'CLE',
    'Detroit Tigers':        'DET',
    'Toronto Blue Jays':     'TOR',
    'Miami Marlins':         'MIA',
    'Oakland Athletics':     'ATH',
    'San Francisco Giants':  'SF',
    'San Diego Padres':      'SD',
    'Tampa Bay Rays':        'TB',
    'Kansas City Royals':    'KC',
    'Arizona Diamondbacks':  'AZ',
}


def _american_to_prob(odds: int | float) -> float:
    """Convert American moneyline odds to raw implied probability (before vig removal)."""
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _vig_free_home_prob(home_odds: float, away_odds: float) -> float | None:
    """Return vig-free P(home) given both sides' American odds. None on failure."""
    try:
        h = _american_to_prob(home_odds)
        a = _american_to_prob(away_odds)
    except Exception:
        return None
    total = h + a
    if total <= 0:
        return None
    return h / total


def _blend_probs(book_probs: dict[str, float]) -> tuple[float, float]:
    """
    Weighted blend of per-book vig-free home probabilities.

    Returns (home_prob_blend, prob_std). If 0-1 books present, prob_std is 0.0.
    Renormalizes weights across whichever books are present.
    """
    if not book_probs:
        return 0.5, 0.0

    used_weights = {k: BOOK_WEIGHTS[k] for k in book_probs if k in BOOK_WEIGHTS}
    total_w = sum(used_weights.values())
    if total_w <= 0:
        return 0.5, 0.0

    blended = sum(book_probs[k] * (w / total_w) for k, w in used_weights.items())
    values  = list(book_probs.values())
    prob_std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return blended, prob_std


def get_pinnacle_odds() -> dict[tuple[str, str], dict]:
    """
    Fetch today's MLB moneylines from multiple sharp books and blend them.

    Kept under the `get_pinnacle_odds` name for backward compatibility — the
    returned dict shape still works with existing main.py / watcher.py code.

    Returns {(away_kalshi_code, home_kalshi_code): {
        'home_prob':  blended vig-free prob (0-1),
        'away_prob':  1 - home_prob,
        'prob_std':   stdev of per-book home probs (0.0 when <2 books),
        'books_used': list of book keys that contributed,
        'per_book':   {book_key: home_prob} raw per-book home probs (for debugging),
    }}.
    Returns empty dict when ODDS_API_KEY is not set or the request fails.
    """
    api_key = getattr(config, 'ODDS_API_KEY', '')
    if not api_key:
        return {}

    bookmakers_csv = ','.join(BOOK_WEIGHTS.keys())
    try:
        resp = requests.get(
            ODDS_API_URL,
            params={
                'apiKey':     api_key,
                'regions':    'us,eu',   # some of these books are listed under EU
                'markets':    'h2h',
                'bookmakers': bookmakers_csv,
                'dateFormat': 'iso',
                'oddsFormat': 'american',
            },
            timeout=10,
        )
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        logger.warning(f"Sharp odds fetch failed: {e}")
        return {}

    now      = datetime.now(timezone.utc)
    today_et = datetime.now(ZoneInfo('America/New_York')).date()
    result: dict[tuple[str, str], dict] = {}

    for game in games:
        try:
            start = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        except Exception:
            continue
        if start <= now:
            continue
        if start.astimezone(ZoneInfo('America/New_York')).date() != today_et:
            continue

        home_name = game.get('home_team', '')
        away_name = game.get('away_team', '')
        home_code = _NAME_TO_KALSHI.get(home_name)
        away_code = _NAME_TO_KALSHI.get(away_name)
        if not home_code or not away_code:
            logger.debug(f"Sharp odds: unknown team '{away_name}' @ '{home_name}' — skipping")
            continue

        # Collect per-book vig-free home probability
        per_book: dict[str, float] = {}
        for bm in game.get('bookmakers', []):
            key = bm.get('key')
            if key not in BOOK_WEIGHTS:
                continue
            h2h = next((m for m in bm.get('markets', []) if m['key'] == 'h2h'), None)
            if not h2h:
                continue
            outcomes = {o['name']: o['price'] for o in h2h.get('outcomes', [])}
            home_odds_val = outcomes.get(home_name)
            away_odds_val = outcomes.get(away_name)
            if home_odds_val is None or away_odds_val is None:
                continue
            p = _vig_free_home_prob(home_odds_val, away_odds_val)
            if p is not None:
                per_book[key] = p

        if not per_book:
            continue

        home_prob, prob_std = _blend_probs(per_book)
        away_prob           = 1.0 - home_prob

        result[(away_code, home_code)] = {
            'home_prob':  round(home_prob, 4),
            'away_prob':  round(away_prob, 4),
            'prob_std':   round(prob_std, 4),
            'books_used': list(per_book.keys()),
            'per_book':   {k: round(v, 4) for k, v in per_book.items()},
        }

        # One-line summary per game with per-book detail
        per_book_fmt = "  ".join(f"{k}={v:.1%}" for k, v in per_book.items())
        logger.info(
            f"Blend: {away_code} @ {home_code} — "
            f"home={home_prob:.1%}  std={prob_std:.1%}  "
            f"({per_book_fmt})"
        )

    logger.info(f"Sharp odds loaded for {len(result)} games ({len(BOOK_WEIGHTS)} books configured)")
    return result
