"""
Manually review games where |model_p - blend_p| >= 0.20.

Web Claude's concern: high-gap cases might be feature-pipeline failures
(rookie SPs with 1-3 starts, postponed games where date columns drift,
lineup TBDs at query time, etc.) rather than honest forecaster disagreement.
If they're broken-data, keep a high gap veto as a data-quality sanity check.
If they're real disagreements, drop the veto.

Outputs each high-gap game with:
  - date, teams
  - model_p, blend_p, gap
  - actual outcome
  - which side model picked, which side blend picked, who won
  - Pinnacle's vig (a sanity check — high vig at high gap could indicate
    Pinnacle is also uncertain, which is suspicious)
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_vs_pinnacle_clv import (
    load_calibration, load_pinnacle_lines, index_lines,
    join_games, CAL_PATHS, LINES_PATH,
)

GAP_THRESHOLDS = [0.15, 0.18, 0.20, 0.25]


def main():
    lines = load_pinnacle_lines(LINES_PATH)
    lines_idx = index_lines(lines)

    cal_2024 = load_calibration(CAL_PATHS[2024])
    cal_2025 = load_calibration(CAL_PATHS[2025])
    cal_both = cal_2024 + cal_2025

    joined, _ = join_games(cal_both, lines_idx, "p_home_prop")

    print(f"Total joined games: {len(joined)}")
    print()

    for thresh in GAP_THRESHOLDS:
        high = [r for r in joined if abs(r['model_p'] - r['blend_p']) >= thresh]
        # Sort by gap descending
        high.sort(key=lambda r: -abs(r['model_p'] - r['blend_p']))

        print(f"\n{'=' * 110}")
        print(f"  Games with |model_p - blend_p| >= {thresh*100:.0f}pp   (n={len(high)})")
        print(f"{'=' * 110}")
        if len(high) == 0:
            continue

        print(f"  {'date':<12} {'matchup':<10} {'model_p':>9} {'blend_p':>9} {'gap':>7} "
              f"{'home_win':>9} {'model_pick':>11} {'blend_pick':>11} {'who_was_right':>15}")
        for r in high[:50]:  # cap output
            gap = r['model_p'] - r['blend_p']  # signed: + means model bullish on home
            matchup = f"{r['away_code']}@{r['home_code']}"
            model_pick = 'HOME' if r['model_p'] > 0.5 else 'AWAY' if r['model_p'] < 0.5 else 'TIE'
            blend_pick = 'HOME' if r['blend_p'] > 0.5 else 'AWAY' if r['blend_p'] < 0.5 else 'TIE'

            if model_pick == blend_pick:
                who = 'agree'
            else:
                home_won = (r['home_win'] == 1)
                if (model_pick == 'HOME' and home_won) or (model_pick == 'AWAY' and not home_won):
                    who = 'MODEL'
                elif (blend_pick == 'HOME' and home_won) or (blend_pick == 'AWAY' and not home_won):
                    who = 'BLEND'
                else:
                    who = '?'

            print(f"  {r['date']:<12} {matchup:<10} {r['model_p']:>9.3f} {r['blend_p']:>9.3f} "
                  f"{gap:>+7.3f} {r['home_win']:>9d} {model_pick:>11} {blend_pick:>11} {who:>15}")

    # Summary at most-extreme tier
    print(f"\n{'=' * 110}")
    print("  SUMMARY — head-to-head outcomes by gap bucket (only directional disagreements counted)")
    print(f"{'=' * 110}")
    print(f"  {'gap_bucket':<14} {'n_games':>8} {'n_dir_disagree':>15} {'model_won':>11} {'blend_won':>11} {'blend_win_rate':>16}")
    bucket_edges = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 1.0)]
    for lo, hi in bucket_edges:
        bucket = [r for r in joined if lo <= abs(r['model_p'] - r['blend_p']) < hi]
        n = len(bucket)
        dir_disagree = [r for r in bucket if (r['model_p'] > 0.5) != (r['blend_p'] > 0.5)]
        n_dd = len(dir_disagree)
        if n_dd == 0:
            print(f"  [{lo:.2f}, {hi:.2f})  {n:>8d}  {n_dd:>15d}     n/a")
            continue
        m_won = sum(1 for r in dir_disagree
                    if (r['model_p'] > 0.5 and r['home_win'] == 1) or (r['model_p'] < 0.5 and r['home_win'] == 0))
        b_won = sum(1 for r in dir_disagree
                    if (r['blend_p'] > 0.5 and r['home_win'] == 1) or (r['blend_p'] < 0.5 and r['home_win'] == 0))
        print(f"  [{lo:.2f}, {hi:.2f})  {n:>8d}  {n_dd:>15d}  {m_won:>11d}  {b_won:>11d}     "
              f"{b_won/n_dd:.3f} ({b_won}/{n_dd})")


if __name__ == "__main__":
    main()
