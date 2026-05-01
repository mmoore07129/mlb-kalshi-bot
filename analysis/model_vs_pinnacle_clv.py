"""
Model-vs-Pinnacle CLV backtest.

Joins:
  - logs/calibration_{YEAR}.csv     model predictions + actual outcomes
  - logs/pinnacle_lines.csv          historical Pinnacle/LowVig/BetOnline lines

Produces threshold-sweep tables answering: on the disagreement subset
(|model_p - blend_p| >= threshold), does the model's directional pick beat
the blend's pick? And does the model's Brier score beat the blend's?

Reports:
  - Threshold sweep table per (de-vig method, comparison target, year subset)
  - Subset base rate alongside accuracies (Web Claude's note: model-acc on a
    subset means nothing without the subset's own home_win baseline)
  - 95% CIs on each directional accuracy

De-vig methods compared: proportional and Shin. If conclusions differ
between methods, that's a measurement artifact, not edge.

Usage:
  python analysis/model_vs_pinnacle_clv.py
"""

import csv
import sys
import os
import math
from collections import defaultdict
from datetime import datetime

# Bot's blend weights (matches data/odds.py)
BOOK_WEIGHTS = {
    'pinnacle':    0.55,
    'lowvig':      0.30,
    'betonlineag': 0.15,
}

CAL_PATHS = {
    2024: "logs/calibration_2024.csv",
    2025: "logs/calibration_2025.csv",
}
LINES_PATH = "logs/pinnacle_lines.csv"

THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15]


# ── Helpers ─────────────────────────────────────────────────────────────────

# Map our calibration team codes (Kalshi-style abbrev) to Odds API full names.
# Reverse of data/odds.py's _NAME_TO_KALSHI:
_KALSHI_TO_NAME = {
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


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval (better than normal approx for small n / extreme p)."""
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    rad = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - rad), min(1.0, center + rad)


def load_calibration(path: str) -> list[dict]:
    """Returns list of {date, home_code, away_code, model_p_home, actual_home_win}."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'date':       r['date'],
                'home_code':  r['home_team'],
                'away_code':  r['away_team'],
                'model_p':    float(r['predicted_p_home']),
                'home_win':   int(r['actual_home_win']),
            })
    return rows


def load_pinnacle_lines(path: str) -> list[dict]:
    """Returns raw rows from pinnacle_lines.csv."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({
                'game_date':      r['game_date'],
                'snapshot_label': r['snapshot_label'],
                'snapshot_time':  r['snapshot_time'],
                'commence_time':  r['commence_time'],
                'home_full':      r['home_team'],
                'away_full':      r['away_team'],
                'book':           r['book'],
                'p_home_prop':    float(r['p_home_prop']),
                'p_home_shin':    float(r['p_home_shin']),
            })
    return out


def index_lines(lines: list[dict]) -> dict:
    """
    Index lines by (date, home_full, away_full) -> list of (snapshot_time,
    book, p_home_prop, p_home_shin, commence_time). Multiple snapshots and
    multiple books per game possible; the joiner picks the best.
    """
    idx: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in lines:
        key = (r['game_date'], r['home_full'], r['away_full'])
        idx[key].append(r)
    return idx


def latest_valid_snapshot(rows: list[dict]) -> str | None:
    """
    Among the snapshots for a game, return the snapshot_time string that's
    latest BEFORE commence_time. None if no valid snapshot.
    """
    valid = []
    for r in rows:
        try:
            snap = datetime.fromisoformat(r['snapshot_time'].replace('Z', '+00:00'))
            comm = datetime.fromisoformat(r['commence_time'].replace('Z', '+00:00'))
            if snap < comm:
                valid.append((snap, r['snapshot_time']))
        except Exception:
            continue
    if not valid:
        return None
    valid.sort()
    return valid[-1][1]  # latest valid


def pick_blend(rows_at_snap: list[dict], devig_field: str) -> tuple[float | None, float | None, list[str]]:
    """
    Among rows at the same snapshot for a single game, compute:
      - pure pinnacle p_home (or None if missing)
      - weighted blend p_home using BOOK_WEIGHTS, renormalized over present books
      - list of books used in blend
    """
    by_book = {r['book']: r for r in rows_at_snap}
    pinnacle_p = by_book['pinnacle'][devig_field] if 'pinnacle' in by_book else None

    used_weights = {b: BOOK_WEIGHTS[b] for b in by_book if b in BOOK_WEIGHTS}
    total_w = sum(used_weights.values())
    blend_p = None
    if total_w > 0:
        blend_p = sum(by_book[b][devig_field] * (w / total_w) for b, w in used_weights.items())
    return pinnacle_p, blend_p, sorted(by_book.keys())


def join_games(calibration: list[dict], lines_idx: dict, devig_field: str) -> list[dict]:
    """
    For each calibration row, pull the best Pinnacle/blend at the latest
    valid snapshot. Returns joined records or skips unmatched.
    """
    matched = []
    unmatched = 0
    for c in calibration:
        home_full = _KALSHI_TO_NAME.get(c['home_code'])
        away_full = _KALSHI_TO_NAME.get(c['away_code'])
        if not home_full or not away_full:
            unmatched += 1
            continue

        candidates = lines_idx.get((c['date'], home_full, away_full), [])
        if not candidates:
            unmatched += 1
            continue

        best_snap = latest_valid_snapshot(candidates)
        if best_snap is None:
            unmatched += 1
            continue

        rows_at_snap = [r for r in candidates if r['snapshot_time'] == best_snap]
        pin_p, blend_p, books = pick_blend(rows_at_snap, devig_field)
        if pin_p is None or blend_p is None:
            unmatched += 1
            continue

        matched.append({
            'date':       c['date'],
            'home_code':  c['home_code'],
            'away_code':  c['away_code'],
            'model_p':    c['model_p'],
            'pinnacle_p': pin_p,
            'blend_p':    blend_p,
            'home_win':   c['home_win'],
            'books':      books,
        })
    return matched, unmatched


def directional_pick_correct(p_home: float, home_win: int) -> int:
    """1 if the prob's directional pick (p_home > 0.5 -> home, else away) is correct."""
    if p_home > 0.5:
        return 1 if home_win == 1 else 0
    elif p_home < 0.5:
        return 1 if home_win == 0 else 0
    else:
        return 0  # exactly 0.5 — no pick


def threshold_sweep(joined: list[dict], compare_to: str, label: str) -> None:
    """
    Print directional + Brier comparison for model vs (pinnacle | blend)
    across thresholds.

    compare_to: 'pinnacle_p' or 'blend_p'
    """
    print(f"\n{'─' * 100}")
    print(f"  {label}   (compare model vs {compare_to})")
    print(f"{'─' * 100}")
    print(f"  {'thresh':<7} {'n':>5} {'subset_p_home':>14} {'model_acc':>11} {'cmp_acc':>9} "
          f"{'Δ(m-c)':>8} {'95% CI Δ':>20} {'Brier_m':>9} {'Brier_c':>9} {'Brier_Δ':>9}")

    for t in THRESHOLDS:
        subset = [r for r in joined if abs(r['model_p'] - r[compare_to]) >= t]
        n = len(subset)
        if n < 10:
            print(f"  ≥{t*100:>4.1f}pp  n={n:>3}  (too small to report)")
            continue

        # subset base rate
        base_rate = sum(r['home_win'] for r in subset) / n

        # directional accuracy
        m_correct = sum(directional_pick_correct(r['model_p'],   r['home_win']) for r in subset)
        c_correct = sum(directional_pick_correct(r[compare_to], r['home_win']) for r in subset)
        m_acc = m_correct / n
        c_acc = c_correct / n
        delta = m_acc - c_acc

        # 95% CI on the DIFFERENCE (paired McNemar-style approximation)
        # SE for two proportions on the same sample: sqrt((p1(1-p1) + p2(1-p2))/n)
        # Reasonable approximation when correlation between m_correct and c_correct
        # isn't extreme. For tighter, we'd use bootstrap.
        se_delta = math.sqrt((m_acc * (1 - m_acc) + c_acc * (1 - c_acc)) / n)
        ci_lo = delta - 1.96 * se_delta
        ci_hi = delta + 1.96 * se_delta

        # Brier
        brier_m = sum((r['model_p'] - r['home_win']) ** 2 for r in subset) / n
        brier_c = sum((r[compare_to] - r['home_win']) ** 2 for r in subset) / n
        brier_delta = brier_m - brier_c  # negative = model better

        ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        print(f"  ≥{t*100:>4.1f}pp  {n:>5}  {base_rate:>14.3f}  {m_acc:>11.3f}  "
              f"{c_acc:>9.3f}  {delta:>+8.3f}  {ci_str:>20}  {brier_m:>9.4f}  "
              f"{brier_c:>9.4f}  {brier_delta:>+9.4f}")


def run_for_devig(devig_label: str, devig_field: str, lines_idx: dict) -> None:
    print(f"\n{'='*100}")
    print(f"  DE-VIG METHOD: {devig_label}")
    print(f"{'='*100}")

    cal_2024 = load_calibration(CAL_PATHS[2024])
    cal_2025 = load_calibration(CAL_PATHS[2025])
    cal_both = cal_2024 + cal_2025

    for year_label, cal in [
        ("2024 only", cal_2024),
        ("2025 only", cal_2025),
        ("2024 + 2025", cal_both),
    ]:
        joined, unmatched = join_games(cal, lines_idx, devig_field)
        print(f"\n[{year_label}] joined {len(joined)} / unmatched {unmatched}")
        threshold_sweep(joined, 'pinnacle_p', f"{year_label}  (vs pure Pinnacle)")
        threshold_sweep(joined, 'blend_p',    f"{year_label}  (vs Pinnacle+LowVig+BetOnline blend)")


def main():
    if not os.path.exists(LINES_PATH):
        print(f"ERROR: {LINES_PATH} does not exist — run backfill_pinnacle.py first")
        sys.exit(1)

    print("Loading historical lines...")
    lines = load_pinnacle_lines(LINES_PATH)
    print(f"  {len(lines)} (game, snapshot, book) rows loaded")

    lines_idx = index_lines(lines)
    print(f"  {len(lines_idx)} unique (date, home, away) game keys")

    run_for_devig("Proportional", "p_home_prop", lines_idx)
    run_for_devig("Shin (1993)",  "p_home_shin", lines_idx)


if __name__ == '__main__':
    main()
