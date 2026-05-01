"""
Inversion-finding due-diligence: is the blend's accuracy climbing with model-
Pinnacle gap a real skill premium, or just an artifact of favorites being
more obvious in high-gap games?

The load-bearing question: at ≥15pp gap, blend gets 64.8% directional
accuracy. But if Pinnacle's mean prediction in that bucket is itself ~65%
(i.e., the favorite is at 65%), then the blend is just calibrated, not
"better at predicting high-gap games." That would mean stripping the model
is correct after all.

Tests run per gap bucket (1pp, 3pp, 5pp, 7pp, 10pp, 12pp, 15pp):
  1. Favorite-pick baseline = mean(max(blend_p, 1 - blend_p)). Expected
     accuracy if blend is calibrated and we always bet the favorite.
  2. Skill premium = actual_blend_acc - favorite_baseline. Positive means
     blend exceeds its own calibration on this subset (real subset skill).
     Zero means calibrated. Negative means overconfident in this subset.
  3. Same for the model.
  4. Direct head-to-head: when model and blend pick DIFFERENT sides, who
     wins more often? This is the cleanest "trust model or blend" question.

Reads logs/calibration_2024.csv, logs/calibration_2025.csv, logs/pinnacle_lines.csv
(same as model_vs_pinnacle_clv.py). Uses both proportional and Shin de-vig.
"""

import csv
import os
import math
from collections import defaultdict
from datetime import datetime

# Reuse helpers from sibling script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_vs_pinnacle_clv import (
    load_calibration, load_pinnacle_lines, index_lines,
    join_games, CAL_PATHS, LINES_PATH, BOOK_WEIGHTS,
)

THRESHOLDS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]


def confidence(p: float) -> float:
    """Forecaster's confidence in its directional pick = max(p, 1-p)."""
    return max(p, 1.0 - p)


def directional_correct(p: float, home_win: int) -> int:
    if p > 0.5:
        return 1 if home_win == 1 else 0
    if p < 0.5:
        return 1 if home_win == 0 else 0
    return 0


def picks_same_side(p1: float, p2: float) -> bool:
    """True iff both probs pick the same directional side (both > 0.5 or both < 0.5)."""
    if p1 == 0.5 or p2 == 0.5:
        return False
    return (p1 > 0.5) == (p2 > 0.5)


def analyze(joined: list[dict], label: str) -> None:
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")
    print(f"  {'thresh':<7} {'n':>5} {'subset_p_home':>13} "
          f"{'blend_acc':>9} {'blend_fav_baseline':>19} {'blend_skill':>12} "
          f"{'model_acc':>9} {'model_fav_baseline':>19} {'model_skill':>12} "
          f"{'n_disagree_dir':>14} {'blend_wins_h2h':>14}")

    for t in THRESHOLDS:
        subset = [r for r in joined if abs(r['model_p'] - r['blend_p']) >= t]
        n = len(subset)
        if n < 10:
            continue

        subset_p_home = sum(r['home_win'] for r in subset) / n

        blend_acc = sum(directional_correct(r['blend_p'], r['home_win']) for r in subset) / n
        model_acc = sum(directional_correct(r['model_p'], r['home_win']) for r in subset) / n

        blend_fav_baseline = sum(confidence(r['blend_p']) for r in subset) / n
        model_fav_baseline = sum(confidence(r['model_p']) for r in subset) / n

        blend_skill = blend_acc - blend_fav_baseline
        model_skill = model_acc - model_fav_baseline

        # Head-to-head: games where they pick DIFFERENT directional sides
        h2h = [r for r in subset if not picks_same_side(r['model_p'], r['blend_p'])]
        n_h2h = len(h2h)
        if n_h2h > 0:
            blend_wins_h2h = sum(directional_correct(r['blend_p'], r['home_win']) for r in h2h) / n_h2h
            blend_wins_str = f"{blend_wins_h2h:.3f} ({n_h2h})"
        else:
            blend_wins_str = "n/a"

        print(f"  ≥{t*100:>4.1f}pp  {n:>5}  {subset_p_home:>13.3f}  "
              f"{blend_acc:>9.3f}  {blend_fav_baseline:>19.3f}  {blend_skill:>+12.3f}  "
              f"{model_acc:>9.3f}  {model_fav_baseline:>19.3f}  {model_skill:>+12.3f}  "
              f"{n_h2h:>14}  {blend_wins_str:>14}")


def main():
    if not os.path.exists(LINES_PATH):
        print(f"ERROR: {LINES_PATH} does not exist")
        sys.exit(1)

    lines = load_pinnacle_lines(LINES_PATH)
    lines_idx = index_lines(lines)

    cal_2024 = load_calibration(CAL_PATHS[2024])
    cal_2025 = load_calibration(CAL_PATHS[2025])
    cal_both = cal_2024 + cal_2025

    for devig_label, devig_field in [("Proportional", "p_home_prop"), ("Shin (1993)", "p_home_shin")]:
        print(f"\n\n{'#'*120}")
        print(f"#  DE-VIG: {devig_label}")
        print(f"{'#'*120}")
        for year_label, cal in [("2024 only", cal_2024), ("2025 only", cal_2025), ("2024+2025", cal_both)]:
            joined, _ = join_games(cal, lines_idx, devig_field)
            analyze(joined, f"{year_label}  ({devig_label})")


if __name__ == "__main__":
    main()
