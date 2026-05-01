"""
Investigate the 2025 [0.40, 0.45) calibration anomaly.

The model was underconfident in this bin: mean_pred=0.432, win_rate=0.515,
residual=+0.083 on n=132. That's ~1.9-sigma. We split by year-half to
distinguish:
  - Early-season effect (rookie SP / small-sample pitcher artifact)
  - Persistent pocket where the model has signal Pinnacle might not

Reads logs/calibration_2025.csv (produced by models/train.py).
"""

import csv
import sys
from datetime import date


def main(csv_path: str = "logs/calibration_2025.csv") -> None:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'date': date.fromisoformat(r['date']),
                'p':    float(r['predicted_p_home']),
                'y':    int(r['actual_home_win']),
            })

    print(f"Total 2025 rows: {len(rows)}")
    print(f"Date range: {min(r['date'] for r in rows)} to {max(r['date'] for r in rows)}")
    print()

    bin_rows = [r for r in rows if 0.40 <= r['p'] < 0.45]
    n = len(bin_rows)
    mp = sum(r['p'] for r in bin_rows) / n
    wr = sum(r['y'] for r in bin_rows) / n
    print(f"=== [0.40, 0.45) bin overall ===")
    print(f"  n={n}  mean_pred={mp:.3f}  win_rate={wr:.3f}  residual={wr-mp:+.3f}")
    print()

    def report(rows_in, label):
        n_ = len(rows_in)
        if n_ == 0:
            print(f"  {label:20s}  n=0")
            return
        mp_ = sum(r['p'] for r in rows_in) / n_
        wr_ = sum(r['y'] for r in rows_in) / n_
        se = (wr_ * (1 - wr_) / n_) ** 0.5
        ci_lo = wr_ - 1.96 * se
        ci_hi = wr_ + 1.96 * se
        print(f"  {label:20s}  n={n_:3d}  mean_pred={mp_:.3f}  win_rate={wr_:.3f}  "
              f"residual={wr_-mp_:+.3f}  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    print("=== Split by year-half ===")
    apr_jun  = [r for r in bin_rows if 4 <= r['date'].month <= 6]
    jul_sep  = [r for r in bin_rows if 7 <= r['date'].month <= 9]
    print(f"  (per Web Claude's hint about rookie-SP / small-sample early-season)")
    report(apr_jun, "Apr-Jun")
    report(jul_sep, "Jul-Sep")
    print()

    print("=== Finer split (per month) ===")
    months_present = sorted(set(r['date'].month for r in bin_rows))
    for m in months_present:
        sub = [r for r in bin_rows if r['date'].month == m]
        report(sub, f"Month {m:02d}")
    print()

    print("=== Sanity: home_win_rate across all 2025 games by half ===")
    all_apr_jun = [r for r in rows if 4 <= r['date'].month <= 6]
    all_jul_sep = [r for r in rows if 7 <= r['date'].month <= 9]
    print(f"  Apr-Jun: n={len(all_apr_jun)}  home_win_rate={sum(r['y'] for r in all_apr_jun)/len(all_apr_jun):.3f}")
    print(f"  Jul-Sep: n={len(all_jul_sep)}  home_win_rate={sum(r['y'] for r in all_jul_sep)/len(all_jul_sep):.3f}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'logs/calibration_2025.csv'
    main(path)
