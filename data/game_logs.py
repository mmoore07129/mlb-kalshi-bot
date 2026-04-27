"""
Game log data pipeline.

For training:   build_training_dataset(start_year, end_year) → pd.DataFrame
For daily use:  get_current_season_stats(season, team_id_map, kalshi_to_mlb) → dict
For features:   build_game_features(home_stats, away_stats) → dict

Features are engineered with strict data-leakage prevention: all stats for game i
are computed from games 0..i-1 only, never including the current game.

Data source: MLB Stats API (statsapi.mlb.com) — historical game scores + linescore.
No pybaseball required for training data.
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

PYTH_EXP = 1.83


def _pythagorean(rs: float, ra: float) -> float:
    if rs <= 0 or ra <= 0:
        return 0.5
    return (rs ** PYTH_EXP) / (rs ** PYTH_EXP + ra ** PYTH_EXP)


def _log5(p_home: float, p_away: float) -> float:
    denom = p_home + p_away - 2 * p_home * p_away
    if denom == 0:
        return 0.5
    return (p_home - p_home * p_away) / denom


# ── Historical data via MLB Stats API ────────────────────────────────────────

def fetch_season_games(year: int) -> pd.DataFrame:
    """
    Fetch all completed regular-season games for a given year from the MLB Stats API.
    Returns DataFrame: date, home_team, away_team, home_runs, away_runs, home_win,
    home_starter_id, away_starter_id.

    Starter IDs come from /schedule?hydrate=probablePitcher. For completed games this
    field resolves to the actual starter (spot-checked against boxscore; ~93% coverage
    on historical Final games, missing cases mostly 2020-COVID doubleheaders).
    Games missing either starter_id are retained here but dropped at the SP-join step
    so the raw game data is still available for team-rolling features.
    """
    from data.mlb_fetcher import mlb_get

    # Fetch in two halves to stay within any API response limits
    chunks = [
        (f"{year}-03-15", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-11-10"),
    ]

    all_games = []
    for start, end in chunks:
        data = mlb_get('/schedule', {
            'sportId':   1,
            'gameType':  'R',
            'hydrate':   'probablePitcher,linescore,team',
            'startDate': start,
            'endDate':   end,
        })
        for date_entry in data.get('dates', []):
            try:
                game_date = date.fromisoformat(date_entry['date'])
            except Exception:
                continue
            for game in date_entry.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                teams    = game.get('teams', {})
                home_abb = teams.get('home', {}).get('team', {}).get('abbreviation', '')
                away_abb = teams.get('away', {}).get('team', {}).get('abbreviation', '')
                # scores available directly on teams; linescore.teams is a fallback
                h_runs   = teams.get('home', {}).get('score')
                a_runs   = teams.get('away', {}).get('score')
                if not home_abb or not away_abb or h_runs is None or a_runs is None:
                    continue
                home_sp = (teams.get('home', {}).get('probablePitcher') or {}).get('id')
                away_sp = (teams.get('away', {}).get('probablePitcher') or {}).get('id')
                all_games.append({
                    'date':            game_date,
                    'home_team':       home_abb,
                    'away_team':       away_abb,
                    'home_runs':       int(h_runs),
                    'away_runs':       int(a_runs),
                    'home_win':        int(h_runs > a_runs),
                    'home_starter_id': int(home_sp) if home_sp else None,
                    'away_starter_id': int(away_sp) if away_sp else None,
                })
        time.sleep(0.3)

    if not all_games:
        logger.warning(f"  {year}: no games returned from MLB API")
        return pd.DataFrame(columns=[
            'date', 'home_team', 'away_team', 'home_runs', 'away_runs', 'home_win',
            'home_starter_id', 'away_starter_id',
        ])

    df = pd.DataFrame(all_games).drop_duplicates(
        subset=['date', 'home_team', 'away_team']
    ).sort_values('date').reset_index(drop=True)
    with_sp = df[['home_starter_id', 'away_starter_id']].notna().all(axis=1).sum()
    logger.info(
        f"  {year}: {len(df)} completed games fetched  "
        f"({with_sp} with both SPs, {len(df) - with_sp} missing)"
    )
    return df


def _build_team_view(season_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Extract a team's game-by-game view from the season DataFrame.
    Returns: date, is_home, runs, runs_allowed, win (one row per game, chronological).
    """
    home = season_df[season_df['home_team'] == team][
        ['date', 'home_runs', 'away_runs', 'home_win']
    ].copy()
    home['is_home'] = True
    home = home.rename(columns={'home_runs': 'runs', 'away_runs': 'runs_allowed', 'home_win': 'win'})

    away = season_df[season_df['away_team'] == team][
        ['date', 'away_runs', 'home_runs', 'home_win']
    ].copy()
    away['is_home'] = False
    away['win']     = 1 - away['home_win']
    away = away.rename(columns={'away_runs': 'runs', 'home_runs': 'runs_allowed'})
    away = away.drop(columns=['home_win'])

    combined = pd.concat(
        [home[['date', 'is_home', 'runs', 'runs_allowed', 'win']],
         away[['date', 'is_home', 'runs', 'runs_allowed', 'win']]],
        ignore_index=True,
    ).sort_values('date').reset_index(drop=True)
    return combined


def _compute_team_rolling_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative YTD and rolling-5 stats BEFORE each game (leakage-free).
    Adds: wins_ytd, losses_ytd, rs_ytd, ra_ytd, win_pct_ytd, pyth_ytd,
           r5_wins, r5_losses, r5_rs, r5_ra, r5_win_pct, r5_pyth
    """
    df = team_games.reset_index(drop=True)

    wins_ytd_l  = []
    losses_ytd_l = []
    rs_ytd_l    = []
    ra_ytd_l    = []
    r5_wins_l   = []
    r5_losses_l = []
    r5_rs_l     = []
    r5_ra_l     = []

    w_run = l_run = 0
    rs_run = ra_run = 0.0

    for i in range(len(df)):
        window = df.iloc[max(0, i - 5):i]
        r5_wins_l.append(int(window['win'].sum()))
        r5_losses_l.append(len(window) - int(window['win'].sum()))
        r5_rs_l.append(float(window['runs'].sum()))
        r5_ra_l.append(float(window['runs_allowed'].sum()))

        wins_ytd_l.append(w_run)
        losses_ytd_l.append(l_run)
        rs_ytd_l.append(rs_run)
        ra_ytd_l.append(ra_run)

        row = df.iloc[i]
        w_run   += int(row['win'])
        l_run   += int(1 - row['win'])
        rs_run  += float(row['runs'])
        ra_run  += float(row['runs_allowed'])

    df = df.copy()
    df['wins_ytd']    = wins_ytd_l
    df['losses_ytd']  = losses_ytd_l
    df['rs_ytd']      = rs_ytd_l
    df['ra_ytd']      = ra_ytd_l
    df['r5_wins']     = r5_wins_l
    df['r5_losses']   = r5_losses_l
    df['r5_rs']       = r5_rs_l
    df['r5_ra']       = r5_ra_l

    g_ytd             = (df['wins_ytd'] + df['losses_ytd']).clip(lower=1)
    r5_g              = (df['r5_wins']  + df['r5_losses']).clip(lower=1)
    df['win_pct_ytd'] = df['wins_ytd'] / g_ytd
    df['pyth_ytd']    = df.apply(lambda r: _pythagorean(r['rs_ytd'], r['ra_ytd']), axis=1)
    df['r5_win_pct']  = df['r5_wins']  / r5_g
    df['r5_pyth']     = df.apply(lambda r: _pythagorean(r['r5_rs'], r['r5_ra']), axis=1)

    return df


# ── Pitcher rolling stats ───────────────────────────────────────────────────

# Priors for pitchers with insufficient history (rookie debuts, early-season starts).
# These are roughly 2015-2024 league averages for starters; intentionally conservative
# so rookies don't get flagged as aces just because they throw one good game.
_PITCHER_PRIOR = {
    'era':      4.30,
    'whip':     1.30,
    'k_pct':    0.220,
    'bb_pct':   0.082,
    'k_bb_pct': 0.138,
    'hr_per_9': 1.20,
}


def _compute_pitcher_rolling_stats(starts: list[dict]) -> list[dict]:
    """
    Compute leakage-free rolling stats BEFORE each start.

    Only includes starts (is_start=True) in the window — relief appearances are
    excluded because their stats (short stints, leverage-driven) don't describe
    SP quality well.

    For each start, returns:
      date, pitcher_id-less row with:
        sp_l3_era, sp_l3_whip, sp_l3_k_pct, sp_l3_bb_pct, sp_l3_k_bb_pct, sp_l3_hr_per_9
        sp_l5_era, sp_l5_whip, sp_l5_k_pct, sp_l5_bb_pct, sp_l5_k_bb_pct, sp_l5_hr_per_9
        sp_ytd_era, sp_ytd_whip, sp_ytd_k_pct, sp_ytd_bb_pct, sp_ytd_k_bb_pct, sp_ytd_hr_per_9
        sp_days_rest  (days since previous start; NaN on first start)
        sp_starts_ytd (count of prior starts this season)

    Starts with fewer than 3 prior starts fall back to league-average priors for
    the rate stats (so we don't amplify noise on 1-start samples). The
    sp_starts_ytd column lets the model downweight early-season estimates.
    """
    start_games = [g for g in starts if g.get('is_start')]
    start_games.sort(key=lambda g: g['date'])

    out: list[dict] = []

    # running season-to-date totals (BEFORE current start)
    ytd_ip = ytd_er = ytd_bb = ytd_k = ytd_h = ytd_hr = ytd_bf = 0.0
    ytd_n  = 0
    prev_date = None

    for i, g in enumerate(start_games):
        prior_window = start_games[max(0, i - 5):i]  # up to 5 prior starts
        l3 = prior_window[-3:]
        l5 = prior_window

        def _agg(games_subset: list[dict]) -> dict:
            if not games_subset:
                return dict(_PITCHER_PRIOR)  # prior fallback
            ip = sum(x['ip'] for x in games_subset)
            er = sum(x['er'] for x in games_subset)
            bb = sum(x['bb'] for x in games_subset)
            k  = sum(x['k']  for x in games_subset)
            h  = sum(x['h']  for x in games_subset)
            hr = sum(x['hr'] for x in games_subset)
            bf = sum(x['bf'] for x in games_subset)
            if ip <= 0 or bf <= 0:
                return dict(_PITCHER_PRIOR)
            return {
                'era':      er * 9.0 / ip,
                'whip':     (bb + h) / ip,
                'k_pct':    k  / bf,
                'bb_pct':   bb / bf,
                'k_bb_pct': (k - bb) / bf,
                'hr_per_9': hr * 9.0 / ip,
            }

        l3_stats  = _agg(l3)  if len(l3) >= 3 else dict(_PITCHER_PRIOR)
        l5_stats  = _agg(l5)  if len(l5) >= 3 else dict(_PITCHER_PRIOR)
        ytd_games = start_games[:i]
        ytd_stats = _agg(ytd_games) if len(ytd_games) >= 3 else dict(_PITCHER_PRIOR)

        # Flag rate stats as prior-derived when this pitcher has <3 prior starts
        # this season. Lets the model learn to downweight SP features for rookies
        # / early-season callups instead of reacting to the league-average prior
        # as if it were a real measurement.
        is_prior = 1 if len(ytd_games) < 3 else 0

        # Days rest since previous start. None → NaN-equivalent (store 4 as a
        # generic "normal rest" prior; let downstream treat as a feature).
        if prev_date is not None:
            try:
                d_now  = date.fromisoformat(g['date'])
                d_prev = date.fromisoformat(prev_date)
                days_rest = (d_now - d_prev).days
            except Exception:
                days_rest = 4
        else:
            days_rest = 4  # opening-start prior

        row = {
            'date':          g['date'],
            'sp_l3_era':        l3_stats['era'],
            'sp_l3_whip':       l3_stats['whip'],
            'sp_l3_k_pct':      l3_stats['k_pct'],
            'sp_l3_bb_pct':     l3_stats['bb_pct'],
            'sp_l3_k_bb_pct':   l3_stats['k_bb_pct'],
            'sp_l3_hr_per_9':   l3_stats['hr_per_9'],
            'sp_l5_era':        l5_stats['era'],
            'sp_l5_whip':       l5_stats['whip'],
            'sp_l5_k_pct':      l5_stats['k_pct'],
            'sp_l5_bb_pct':     l5_stats['bb_pct'],
            'sp_l5_k_bb_pct':   l5_stats['k_bb_pct'],
            'sp_l5_hr_per_9':   l5_stats['hr_per_9'],
            'sp_ytd_era':       ytd_stats['era'],
            'sp_ytd_whip':      ytd_stats['whip'],
            'sp_ytd_k_pct':     ytd_stats['k_pct'],
            'sp_ytd_bb_pct':    ytd_stats['bb_pct'],
            'sp_ytd_k_bb_pct':  ytd_stats['k_bb_pct'],
            'sp_ytd_hr_per_9':  ytd_stats['hr_per_9'],
            'sp_days_rest':     days_rest,
            'sp_starts_ytd':    i,
            'sp_is_prior':      is_prior,
        }
        out.append(row)

        prev_date = g['date']

    return out


def _build_pitcher_stat_index(season_df: pd.DataFrame, year: int) -> dict:
    """
    For every unique starter in `season_df`, fetch their full season gameLog,
    compute leakage-free rolling stats per start, and index by (pitcher_id, date_str).

    Returns {(pitcher_id, 'YYYY-MM-DD'): rolling_stat_dict}.
    Per-pitcher fetches are disk-cached, so reruns are fast.
    """
    from data.mlb_fetcher import fetch_pitcher_game_logs

    starter_ids = set()
    for col in ('home_starter_id', 'away_starter_id'):
        starter_ids.update(int(x) for x in season_df[col].dropna().unique())
    logger.info(f"  {year}: fetching game logs for {len(starter_ids)} unique starters...")

    index: dict = {}
    for i, pid in enumerate(sorted(starter_ids), 1):
        games   = fetch_pitcher_game_logs(pid, year)
        rolling = _compute_pitcher_rolling_stats(games)
        for r in rolling:
            index[(pid, r['date'])] = r
        if i % 50 == 0:
            logger.info(f"    pitchers processed: {i}/{len(starter_ids)}")

    logger.info(f"  {year}: indexed {len(index)} (pitcher, date) rolling-stat entries")
    return index


# Feature columns emitted by _compute_pitcher_rolling_stats, minus the join key.
_SP_FEATURE_COLS = [
    'sp_l3_era', 'sp_l3_whip', 'sp_l3_k_pct', 'sp_l3_bb_pct', 'sp_l3_k_bb_pct', 'sp_l3_hr_per_9',
    'sp_l5_era', 'sp_l5_whip', 'sp_l5_k_pct', 'sp_l5_bb_pct', 'sp_l5_k_bb_pct', 'sp_l5_hr_per_9',
    'sp_ytd_era', 'sp_ytd_whip', 'sp_ytd_k_pct', 'sp_ytd_bb_pct', 'sp_ytd_k_bb_pct', 'sp_ytd_hr_per_9',
    'sp_days_rest', 'sp_starts_ytd',
    'sp_is_prior',
]


def build_training_dataset(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """
    Build a full labelled training DataFrame (one row per game, home perspective).

    For each season: fetches game scores + starter IDs, computes team rolling stats,
    fetches per-pitcher game logs + rolling stats, then joins team + SP stats into
    training rows. Rows missing either starter_id or rolling-stat lookup are dropped
    (~7% loss vs. pre-SP pipeline, mostly 2020-COVID doubleheaders).
    """
    all_rows = []

    for year in range(start_year, end_year + 1):
        logger.info(f"Processing {year}...")
        season_df = fetch_season_games(year)
        if season_df.empty:
            continue

        # Team rolling stats for this season
        all_teams = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
        team_stats: dict[str, pd.DataFrame] = {}
        for team in all_teams:
            view  = _build_team_view(season_df, team)
            stats = _compute_team_rolling_stats(view)
            team_stats[team] = stats

        # Pitcher rolling stats for this season
        sp_index = _build_pitcher_stat_index(season_df, year)

        # Build training rows: one per game (home perspective)
        games_added = 0
        games_dropped_no_sp = 0
        for _, game in season_df.iterrows():
            home_abb  = game['home_team']
            away_abb  = game['away_team']
            game_date = game['date']

            if home_abb not in team_stats or away_abb not in team_stats:
                continue

            # Find stats for home team before this home game
            h_df   = team_stats[home_abb]
            h_rows = h_df[(h_df['date'] == game_date) & (h_df['is_home'] == True)]
            if h_rows.empty:
                continue
            h = h_rows.iloc[0]

            # Find stats for away team before this away game
            a_df   = team_stats[away_abb]
            a_rows = a_df[(a_df['date'] == game_date) & (a_df['is_home'] == False)]
            if a_rows.empty:
                continue
            a = a_rows.iloc[0]

            home_sp_id = game.get('home_starter_id')
            away_sp_id = game.get('away_starter_id')
            if pd.isna(home_sp_id) or pd.isna(away_sp_id):
                games_dropped_no_sp += 1
                continue

            date_str = game_date.isoformat() if hasattr(game_date, 'isoformat') else str(game_date)
            home_sp = sp_index.get((int(home_sp_id), date_str))
            away_sp = sp_index.get((int(away_sp_id), date_str))
            if home_sp is None or away_sp is None:
                # SP pitched this game but our rolling index doesn't have them
                # on this date — can happen for rescheduled/doubleheader games
                # where /schedule and gameLog dates disagree.
                games_dropped_no_sp += 1
                continue

            row = {
                'date':            game_date,
                'season':          year,
                'home_team':       home_abb,
                'away_team':       away_abb,
                'home_win':        int(game['home_win']),
                # Home YTD
                'home_wins_ytd':   h['wins_ytd'],
                'home_losses_ytd': h['losses_ytd'],
                'home_win_pct':    h['win_pct_ytd'],
                'home_pyth':       h['pyth_ytd'],
                # Home rolling-5
                'home_r5_wins':    h['r5_wins'],
                'home_r5_losses':  h['r5_losses'],
                'home_r5_win_pct': h['r5_win_pct'],
                'home_r5_pyth':    h['r5_pyth'],
                # Away YTD
                'away_wins_ytd':   a['wins_ytd'],
                'away_losses_ytd': a['losses_ytd'],
                'away_win_pct':    a['win_pct_ytd'],
                'away_pyth':       a['pyth_ytd'],
                # Away rolling-5
                'away_r5_wins':    a['r5_wins'],
                'away_r5_losses':  a['r5_losses'],
                'away_r5_win_pct': a['r5_win_pct'],
                'away_r5_pyth':    a['r5_pyth'],
            }
            for k in _SP_FEATURE_COLS:
                row[f'home_{k}'] = home_sp[k]
                row[f'away_{k}'] = away_sp[k]
            all_rows.append(row)
            games_added += 1

        logger.info(
            f"  {year}: {games_added} training rows built, "
            f"{games_dropped_no_sp} dropped for missing SP"
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Team deltas / log5 (original)
    df['pyth_delta']    = df['home_pyth']    - df['away_pyth']
    df['log5']          = df.apply(lambda r: _log5(r['home_pyth'],    r['away_pyth']),    axis=1)
    df['r5_pyth_delta'] = df['home_r5_pyth'] - df['away_r5_pyth']
    df['r5_log5']       = df.apply(lambda r: _log5(r['home_r5_pyth'], r['away_r5_pyth']), axis=1)

    # SP deltas — home minus away. Negative delta on ERA/WHIP/BB%/HR/9 = home SP better;
    # positive on K%/K-BB% = home SP better.
    df['sp_l5_era_delta']      = df['home_sp_l5_era']      - df['away_sp_l5_era']
    df['sp_l5_whip_delta']     = df['home_sp_l5_whip']     - df['away_sp_l5_whip']
    df['sp_l5_k_pct_delta']    = df['home_sp_l5_k_pct']    - df['away_sp_l5_k_pct']
    df['sp_l5_bb_pct_delta']   = df['home_sp_l5_bb_pct']   - df['away_sp_l5_bb_pct']
    df['sp_l5_k_bb_pct_delta'] = df['home_sp_l5_k_bb_pct'] - df['away_sp_l5_k_bb_pct']
    df['sp_ytd_era_delta']     = df['home_sp_ytd_era']     - df['away_sp_ytd_era']
    df['sp_ytd_k_pct_delta']   = df['home_sp_ytd_k_pct']   - df['away_sp_ytd_k_pct']

    logger.info(f"Training dataset complete: {len(df)} games ({start_year}–{end_year})")
    return df


# ── Current Season Stats (for daily prediction) ──────────────────────────────

def get_current_season_stats(season: int, team_id_map: dict, kalshi_to_mlb: dict) -> dict[str, dict]:
    """
    Build current-season stats for all teams: YTD + rolling-5 game stats.

    YTD (wins/losses/RS/RA/Pythagorean) comes from the MLB standings API.
    Rolling-5 comes from the last 5 completed games via the MLB schedule API.

    Returns {team_abbr: {wins_ytd, losses_ytd, win_pct_ytd, pyth_ytd,
                         r5_wins, r5_losses, r5_win_pct, r5_pyth}}
    """
    from data.mlb_fetcher import mlb_get

    # YTD from standings
    standings = mlb_get('/standings', {
        'leagueId': '103,104',
        'season':   str(season),
        'hydrate':  'team',
    })

    ytd: dict[str, dict] = {}
    for record in standings.get('records', []):
        for tr in record.get('teamRecords', []):
            abb    = tr.get('team', {}).get('abbreviation', '')
            wins   = tr.get('wins', 0)
            losses = tr.get('losses', 0)
            rs     = tr.get('runsScored', 0)
            ra     = tr.get('runsAllowed', 0)
            games  = wins + losses
            if not abb or games == 0:
                continue
            ytd[abb] = {
                'wins_ytd':    wins,
                'losses_ytd':  losses,
                'win_pct_ytd': wins / games,
                'pyth_ytd':    _pythagorean(rs, ra),
            }

    # Rolling-5 from schedule (last 10 days)
    today      = date.today()
    start_date = (today - timedelta(days=16)).strftime('%Y-%m-%d')
    end_date   = (today - timedelta(days=1)).strftime('%Y-%m-%d')

    rolling: dict[str, dict] = {}
    for abb, team_id in team_id_map.items():
        if len(abb) > 3 or abb not in ytd:
            continue

        data = mlb_get('/schedule', {
            'sportId':   1,
            'teamId':    team_id,
            'startDate': start_date,
            'endDate':   end_date,
            'hydrate':   'linescore,team',
            'gameType':  'R',
        })

        recent: list[dict] = []
        for date_entry in data.get('dates', []):
            for game in date_entry.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                teams    = game.get('teams', {})
                is_home  = teams.get('home', {}).get('team', {}).get('abbreviation', '') == abb
                side     = 'home' if is_home else 'away'
                opp_side = 'away' if is_home else 'home'
                ls       = game.get('linescore', {}).get('teams', {})
                runs     = ls.get(side, {}).get('runs')
                opp_runs = ls.get(opp_side, {}).get('runs')
                if runs is None or opp_runs is None:
                    continue
                recent.append({
                    'runs': runs, 'opp_runs': opp_runs,
                    'win': int(runs > opp_runs),
                    'date': date_entry.get('date', ''),
                })

        recent.sort(key=lambda g: g['date'])
        last5 = recent[-5:]

        if last5:
            r5_w  = sum(g['win'] for g in last5)
            r5_rs = sum(g['runs'] for g in last5)
            r5_ra = sum(g['opp_runs'] for g in last5)
            rolling[abb] = {
                'r5_wins':    r5_w,
                'r5_losses':  len(last5) - r5_w,
                'r5_win_pct': r5_w / len(last5),
                'r5_pyth':    _pythagorean(r5_rs, r5_ra),
            }
        else:
            rolling[abb] = {'r5_wins': 0, 'r5_losses': 0, 'r5_win_pct': 0.5, 'r5_pyth': 0.5}

        time.sleep(0.1)

    # Merge and add Kalshi aliases
    result: dict[str, dict] = {}
    for abb in ytd:
        r = rolling.get(abb, {'r5_wins': 0, 'r5_losses': 0, 'r5_win_pct': 0.5, 'r5_pyth': 0.5})
        result[abb] = {**ytd[abb], **r}

    for kalshi_code, mlb_abb in kalshi_to_mlb.items():
        if mlb_abb in result:
            result[kalshi_code] = result[mlb_abb]

    logger.info(f"Current season stats loaded for {len(ytd)} teams")
    return result


def compute_inference_sp_stats(
    pitcher_id: int | None,
    season: int,
    today: date,
) -> dict:
    """
    Compute SP features for a starter going into today's game.

    Unlike the training path (which computes stats BEFORE each completed start),
    inference needs stats INCLUDING the pitcher's most recent prior start —
    that's the knowledge set going into today.

    Returns a dict of {sp_l3_*, sp_l5_*, sp_ytd_*, sp_days_rest, sp_starts_ytd}
    matching _SP_FEATURE_COLS. Falls back to league-average priors when:
      - pitcher_id is None / unknown
      - pitcher has <3 prior completed starts this season (early-season / rookie)
      - fetch fails

    Refreshes the pitcher's gameLog from MLB to catch recent starts (cache stale
    for current season by design).
    """
    prior = {f'home_{k}': None for k in _SP_FEATURE_COLS}  # shape-only helper; ignored

    def _fallback() -> dict:
        return {
            'sp_l3_era': _PITCHER_PRIOR['era'], 'sp_l3_whip': _PITCHER_PRIOR['whip'],
            'sp_l3_k_pct': _PITCHER_PRIOR['k_pct'], 'sp_l3_bb_pct': _PITCHER_PRIOR['bb_pct'],
            'sp_l3_k_bb_pct': _PITCHER_PRIOR['k_bb_pct'], 'sp_l3_hr_per_9': _PITCHER_PRIOR['hr_per_9'],
            'sp_l5_era': _PITCHER_PRIOR['era'], 'sp_l5_whip': _PITCHER_PRIOR['whip'],
            'sp_l5_k_pct': _PITCHER_PRIOR['k_pct'], 'sp_l5_bb_pct': _PITCHER_PRIOR['bb_pct'],
            'sp_l5_k_bb_pct': _PITCHER_PRIOR['k_bb_pct'], 'sp_l5_hr_per_9': _PITCHER_PRIOR['hr_per_9'],
            'sp_ytd_era': _PITCHER_PRIOR['era'], 'sp_ytd_whip': _PITCHER_PRIOR['whip'],
            'sp_ytd_k_pct': _PITCHER_PRIOR['k_pct'], 'sp_ytd_bb_pct': _PITCHER_PRIOR['bb_pct'],
            'sp_ytd_k_bb_pct': _PITCHER_PRIOR['k_bb_pct'], 'sp_ytd_hr_per_9': _PITCHER_PRIOR['hr_per_9'],
            'sp_days_rest': 4, 'sp_starts_ytd': 0,
            'sp_is_prior': 1,
        }

    if not pitcher_id:
        return _fallback()

    try:
        from data.mlb_fetcher import fetch_pitcher_game_logs
        games = fetch_pitcher_game_logs(int(pitcher_id), season, refresh_if_current=True)
    except Exception as e:
        logger.warning(f"SP fetch failed for pitcher {pitcher_id}: {e}")
        return _fallback()

    starts = [g for g in games if g.get('is_start') and g.get('date')]
    today_str = today.isoformat() if hasattr(today, 'isoformat') else str(today)
    # Only starts STRICTLY before today
    prior_starts = [g for g in starts if g['date'] < today_str]
    prior_starts.sort(key=lambda g: g['date'])

    if len(prior_starts) < 3:
        # Still return days_rest if at least one prior start; keeps ops signal
        fb = _fallback()
        if prior_starts:
            try:
                last = date.fromisoformat(prior_starts[-1]['date'])
                fb['sp_days_rest'] = (today - last).days
            except Exception:
                pass
            fb['sp_starts_ytd'] = len(prior_starts)
        return fb

    def _agg(subset: list[dict]) -> dict:
        ip = sum(x['ip'] for x in subset)
        er = sum(x['er'] for x in subset)
        bb = sum(x['bb'] for x in subset)
        k  = sum(x['k']  for x in subset)
        h  = sum(x['h']  for x in subset)
        hr = sum(x['hr'] for x in subset)
        bf = sum(x['bf'] for x in subset)
        if ip <= 0 or bf <= 0:
            return dict(_PITCHER_PRIOR)
        return {
            'era':      er * 9.0 / ip,
            'whip':     (bb + h) / ip,
            'k_pct':    k  / bf,
            'bb_pct':   bb / bf,
            'k_bb_pct': (k - bb) / bf,
            'hr_per_9': hr * 9.0 / ip,
        }

    l3  = _agg(prior_starts[-3:])
    l5  = _agg(prior_starts[-5:])
    ytd = _agg(prior_starts)

    try:
        last = date.fromisoformat(prior_starts[-1]['date'])
        days_rest = (today - last).days
    except Exception:
        days_rest = 4

    return {
        'sp_l3_era': l3['era'], 'sp_l3_whip': l3['whip'], 'sp_l3_k_pct': l3['k_pct'],
        'sp_l3_bb_pct': l3['bb_pct'], 'sp_l3_k_bb_pct': l3['k_bb_pct'], 'sp_l3_hr_per_9': l3['hr_per_9'],
        'sp_l5_era': l5['era'], 'sp_l5_whip': l5['whip'], 'sp_l5_k_pct': l5['k_pct'],
        'sp_l5_bb_pct': l5['bb_pct'], 'sp_l5_k_bb_pct': l5['k_bb_pct'], 'sp_l5_hr_per_9': l5['hr_per_9'],
        'sp_ytd_era': ytd['era'], 'sp_ytd_whip': ytd['whip'], 'sp_ytd_k_pct': ytd['k_pct'],
        'sp_ytd_bb_pct': ytd['bb_pct'], 'sp_ytd_k_bb_pct': ytd['k_bb_pct'], 'sp_ytd_hr_per_9': ytd['hr_per_9'],
        'sp_days_rest': days_rest, 'sp_starts_ytd': len(prior_starts),
        'sp_is_prior': 0,
    }


def build_game_features(
    home_stats: dict,
    away_stats: dict,
    home_sp_stats: dict | None = None,
    away_sp_stats: dict | None = None,
) -> dict:
    """
    Convert home/away team + SP stat dicts into the flat feature dict the model expects.

    SP dicts are optional for backward compatibility with the old 20-feature model.
    When either is None, SP features fall back to priors (matches the training-time
    behavior for rookie/early-season starts) and SP deltas compute to zero.
    """
    hp = home_stats.get('pyth_ytd', 0.5)
    ap = away_stats.get('pyth_ytd', 0.5)
    hr = home_stats.get('r5_pyth',  0.5)
    ar = away_stats.get('r5_pyth',  0.5)

    if home_sp_stats is None:
        home_sp_stats = compute_inference_sp_stats(None, date.today().year, date.today())
    if away_sp_stats is None:
        away_sp_stats = compute_inference_sp_stats(None, date.today().year, date.today())

    feat = {
        'home_wins_ytd':   home_stats.get('wins_ytd',    0),
        'home_losses_ytd': home_stats.get('losses_ytd',  0),
        'home_win_pct':    home_stats.get('win_pct_ytd', 0.5),
        'home_pyth':       hp,
        'home_r5_wins':    home_stats.get('r5_wins',     0),
        'home_r5_losses':  home_stats.get('r5_losses',   0),
        'home_r5_win_pct': home_stats.get('r5_win_pct',  0.5),
        'home_r5_pyth':    hr,
        'away_wins_ytd':   away_stats.get('wins_ytd',    0),
        'away_losses_ytd': away_stats.get('losses_ytd',  0),
        'away_win_pct':    away_stats.get('win_pct_ytd', 0.5),
        'away_pyth':       ap,
        'away_r5_wins':    away_stats.get('r5_wins',     0),
        'away_r5_losses':  away_stats.get('r5_losses',   0),
        'away_r5_win_pct': away_stats.get('r5_win_pct',  0.5),
        'away_r5_pyth':    ar,
        'pyth_delta':      hp - ap,
        'log5':            _log5(hp, ap),
        'r5_pyth_delta':   hr - ar,
        'r5_log5':         _log5(hr, ar),
    }
    # SP features prefixed
    for k in _SP_FEATURE_COLS:
        feat[f'home_{k}'] = home_sp_stats[k]
        feat[f'away_{k}'] = away_sp_stats[k]
    # SP deltas matching training
    feat['sp_l5_era_delta']      = home_sp_stats['sp_l5_era']      - away_sp_stats['sp_l5_era']
    feat['sp_l5_whip_delta']     = home_sp_stats['sp_l5_whip']     - away_sp_stats['sp_l5_whip']
    feat['sp_l5_k_pct_delta']    = home_sp_stats['sp_l5_k_pct']    - away_sp_stats['sp_l5_k_pct']
    feat['sp_l5_bb_pct_delta']   = home_sp_stats['sp_l5_bb_pct']   - away_sp_stats['sp_l5_bb_pct']
    feat['sp_l5_k_bb_pct_delta'] = home_sp_stats['sp_l5_k_bb_pct'] - away_sp_stats['sp_l5_k_bb_pct']
    feat['sp_ytd_era_delta']     = home_sp_stats['sp_ytd_era']     - away_sp_stats['sp_ytd_era']
    feat['sp_ytd_k_pct_delta']   = home_sp_stats['sp_ytd_k_pct']   - away_sp_stats['sp_ytd_k_pct']
    return feat
