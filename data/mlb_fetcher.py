"""
MLB data fetcher: team IDs, schedules, probable pitchers, pitcher game logs.
"""

import json
import logging
import os
import time
import requests
from datetime import date

logger = logging.getLogger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"

# On-disk cache for pitcher game logs — training across 11 seasons would otherwise
# re-fetch the same pitcher-season multiple times (different training runs, feature
# iterations). Cache keyed by (pitcher_id, season) — stats for a completed season
# don't change, so entries are permanent; current-season caches are refreshed by
# the caller when needed (see fetch_pitcher_game_logs).
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'state', 'pitcher_logs',
)

# Kalshi team code → MLB Stats API abbreviation (only where they differ)
KALSHI_TO_MLB: dict[str, str] = {
    'AZ':  'ARI',
    'ATH': 'ATH',
    'SF':  'SF',
    'SD':  'SD',
    'TB':  'TB',
    'KC':  'KC',
    'CWS': 'CWS',
}


def mlb_get(path: str, params: dict = None) -> dict:
    """GET from MLB Stats API with basic retry."""
    for attempt in range(3):
        try:
            resp = requests.get(f"{MLB_API}{path}", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"MLB API {path} attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1.5)
    return {}


def get_team_id_map() -> dict[str, int]:
    """
    Returns {mlb_abbreviation: team_id} for all MLB teams.
    Also adds Kalshi-specific aliases that differ from MLB abbreviations.
    """
    data = mlb_get('/teams', {'sportId': 1})
    result: dict[str, int] = {}
    for team in data.get('teams', []):
        abb = team.get('abbreviation', '')
        tid = team.get('id')
        if abb and tid:
            result[abb] = tid
    for kalshi_code, mlb_abb in KALSHI_TO_MLB.items():
        if mlb_abb in result:
            result[kalshi_code] = result[mlb_abb]
    logger.info(f"Loaded {len(result)} team IDs")
    return result


def get_probable_pitchers(game_date: date = None) -> dict[tuple[str, str], dict]:
    """
    Fetch probable pitchers for all games on game_date (for logging only).
    Returns {(away_mlb_abb, home_mlb_abb): {'home_pitcher': {...}, 'away_pitcher': {...}}}
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime('%Y-%m-%d')
    data = mlb_get('/schedule', {
        'sportId': 1,
        'date':    date_str,
        'hydrate': 'probablePitcher,team',
    })

    result: dict[tuple[str, str], dict] = {}
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            teams    = game.get('teams', {})
            home_abb = teams.get('home', {}).get('team', {}).get('abbreviation', '')
            away_abb = teams.get('away', {}).get('team', {}).get('abbreviation', '')
            if not home_abb or not away_abb:
                continue

            def _pitcher(side: dict) -> dict | None:
                p = side.get('probablePitcher')
                if not p:
                    return None
                return {'id': p.get('id'), 'name': p.get('fullName', 'Unknown')}

            result[(away_abb, home_abb)] = {
                'home_pitcher': _pitcher(teams.get('home', {})),
                'away_pitcher': _pitcher(teams.get('away', {})),
            }

    logger.info(f"Probable pitchers loaded for {len(result)} games on {date_str}")
    return result


# ── Pitcher game logs ───────────────────────────────────────────────────────

def _parse_ip(ip_str) -> float:
    """
    Convert MLB's innings-pitched notation to decimal innings.
    "5.0" → 5.0, "5.1" → 5.333, "5.2" → 5.667.
    The fractional part is outs, not tenths.
    """
    if ip_str in (None, ''):
        return 0.0
    try:
        s = str(ip_str)
        if '.' in s:
            whole, frac = s.split('.')
            return int(whole) + int(frac) / 3.0
        return float(s)
    except Exception:
        return 0.0


def fetch_pitcher_game_logs(
    pitcher_id: int,
    season: int,
    refresh_if_current: bool = False,
) -> list[dict]:
    """
    Fetch per-start game logs for a pitcher in a given season.

    Returns a list of dicts sorted by date:
      {date, gamePk, is_home, is_start, ip, h, r, er, bb, k, hr, bf, pitches}

    Completed prior seasons are cached to disk permanently (stats are frozen).
    Current-season calls can pass refresh_if_current=True to force a re-fetch;
    otherwise they also cache but grow stale between calls.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f'{pitcher_id}_{season}.json')
    current_year = date.today().year

    if os.path.exists(cache_path) and not (refresh_if_current and season == current_year):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass  # corrupt cache — refetch

    data = mlb_get(f'/people/{pitcher_id}/stats', {
        'stats':  'gameLog',
        'season': str(season),
        'group':  'pitching',
    })
    splits = (data.get('stats', [{}]) or [{}])[0].get('splits', []) or []

    games: list[dict] = []
    for sp in splits:
        stat = sp.get('stat', {}) or {}
        game = sp.get('game', {}) or {}
        games.append({
            'date':     sp.get('date', ''),
            'gamePk':   game.get('gamePk'),
            'is_home':  bool(sp.get('isHome')),
            'is_start': int(stat.get('gamesStarted', 0) or 0) > 0,
            'ip':       _parse_ip(stat.get('inningsPitched')),
            'h':        int(stat.get('hits', 0) or 0),
            'r':        int(stat.get('runs', 0) or 0),
            'er':       int(stat.get('earnedRuns', 0) or 0),
            'bb':       int(stat.get('baseOnBalls', 0) or 0),
            'k':        int(stat.get('strikeOuts', 0) or 0),
            'hr':       int(stat.get('homeRuns', 0) or 0),
            'bf':       int(stat.get('battersFaced', 0) or 0),
            'pitches':  int(stat.get('numberOfPitches', 0) or 0),
        })

    games.sort(key=lambda g: g['date'])

    try:
        with open(cache_path, 'w') as f:
            json.dump(games, f)
    except Exception as e:
        logger.warning(f"pitcher log cache write failed for {pitcher_id}/{season}: {e}")

    return games
