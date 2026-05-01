"""
XGBoost training pipeline for MLB game outcome prediction.

Run once (or at the start of each season) to train and save the model:
    python models/train.py
    python models/train.py --start 2015 --end 2025

The trained model is saved to models/xgb_model.pkl.
After training, run main.py normally for daily predictions.
"""

import os
import sys
import logging
import pickle

# Allow running from the project root or from models/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from data.game_logs import build_training_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)

FEATURES = [
    # Team — YTD
    'home_wins_ytd',   'home_losses_ytd', 'home_win_pct',    'home_pyth',
    'away_wins_ytd',   'away_losses_ytd', 'away_win_pct',    'away_pyth',
    # Team — rolling 5
    'home_r5_wins',    'home_r5_losses',  'home_r5_win_pct', 'home_r5_pyth',
    'away_r5_wins',    'away_r5_losses',  'away_r5_win_pct', 'away_r5_pyth',
    # Team — combined
    'pyth_delta',      'log5',            'r5_pyth_delta',   'r5_log5',
    # SP — home last-5 starts
    'home_sp_l5_era', 'home_sp_l5_whip', 'home_sp_l5_k_pct',
    'home_sp_l5_bb_pct', 'home_sp_l5_k_bb_pct', 'home_sp_l5_hr_per_9',
    # SP — away last-5 starts
    'away_sp_l5_era', 'away_sp_l5_whip', 'away_sp_l5_k_pct',
    'away_sp_l5_bb_pct', 'away_sp_l5_k_bb_pct', 'away_sp_l5_hr_per_9',
    # SP — YTD (stabilizer for low-sample L5)
    'home_sp_ytd_era', 'home_sp_ytd_whip', 'home_sp_ytd_k_pct', 'home_sp_ytd_bb_pct',
    'away_sp_ytd_era', 'away_sp_ytd_whip', 'away_sp_ytd_k_pct', 'away_sp_ytd_bb_pct',
    # SP — context
    'home_sp_days_rest', 'away_sp_days_rest',
    'home_sp_starts_ytd', 'away_sp_starts_ytd',
    # SP — fallback-prior flag (1 = rookie/early-season, rate stats are league-avg priors)
    'home_sp_is_prior', 'away_sp_is_prior',
    # SP — matchup deltas (home minus away; let the model use the interaction directly)
    'sp_l5_era_delta', 'sp_l5_whip_delta', 'sp_l5_k_pct_delta',
    'sp_l5_bb_pct_delta', 'sp_l5_k_bb_pct_delta',
    'sp_ytd_era_delta', 'sp_ytd_k_pct_delta',
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_model.pkl')

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.5,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def _make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**XGB_PARAMS)


def train(start_year: int = 2015, end_year: int = 2025) -> None:
    logger.info(f"Building training dataset ({start_year}–{end_year})...")
    df = build_training_dataset(start_year, end_year)

    if df.empty:
        logger.error("No training data — aborting.")
        return

    # Drop early-season rows where sample size is too small to be informative
    min_games = 5
    df = df[
        (df['home_wins_ytd'] + df['home_losses_ytd'] >= min_games) &
        (df['away_wins_ytd'] + df['away_losses_ytd'] >= min_games)
    ].copy()
    logger.info(f"After {min_games}-game minimum filter: {len(df)} games")

    X = df[FEATURES].astype(float).values
    y = df['home_win'].values

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_lls = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        m = _make_model()
        m.fit(X[tr_idx], y[tr_idx])
        probs = m.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        cv_accs.append(accuracy_score(y[val_idx], preds))
        cv_lls.append(log_loss(y[val_idx], probs))
        logger.info(f"  Fold {fold+1}: acc={cv_accs[-1]:.3f}  logloss={cv_lls[-1]:.4f}")

    logger.info(f"CV mean accuracy : {np.mean(cv_accs):.3f} ± {np.std(cv_accs):.3f}")
    logger.info(f"CV mean log-loss : {np.mean(cv_lls):.4f}")

    # ── Walk-forward test: train on all-but-last-year, test on last year ─────
    logger.info(f"\nWalk-forward test (held-out year: {end_year})...")
    tr_mask   = df['date'].apply(lambda d: d.year < end_year)
    test_mask = ~tr_mask

    wf_model = _make_model()
    wf_model.fit(X[tr_mask], y[tr_mask])
    test_probs = wf_model.predict_proba(X[test_mask])[:, 1]

    logger.info(f"Walk-forward games: {test_mask.sum()}")
    logger.info("Accuracy by confidence threshold:")
    for thr in [0.50, 0.53, 0.55, 0.57, 0.60, 0.63, 0.66]:
        mask = (test_probs >= thr) | (test_probs <= 1 - thr)
        if mask.sum() < 10:
            continue
        acc = accuracy_score(y[test_mask][mask], (test_probs[mask] >= 0.5).astype(int))
        logger.info(f"  ≥{thr:.0%}: n={mask.sum():4d}  acc={acc:.3f}")

    # ── Calibration on holdouts (post-isotonic, leakage-free) ────────────────
    # The walk-forward model above is RAW XGBoost. Production uses isotonic-
    # calibrated probabilities, so to evaluate the production-equivalent model
    # we re-train with CalibratedClassifierCV on pre-holdout data and predict
    # on the holdout. This tells us "when the model says 65%, do they actually
    # win 65% of the time?"
    #
    # We do this for two holdout years (end_year - 1, end_year) to check
    # whether calibration is stable across years. If 2024 and 2025 calibrate
    # similarly, the diagnostic is robust. If they diverge meaningfully, the
    # model's behavior is year-dependent — itself an important finding.
    cal_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(cal_dir, exist_ok=True)
    bins = [(round(lo, 2), round(lo + 0.05, 2)) for lo in np.arange(0.0, 1.0, 0.05)]

    for holdout_year in [end_year - 1, end_year]:
        if holdout_year < start_year + 1:
            continue  # need at least 1 prior year to train on
        logger.info(f"\nCalibration on {holdout_year} holdout (post-isotonic, leakage-free)...")
        h_tr_mask   = df['date'].apply(lambda d, y=holdout_year: d.year <  y)
        h_test_mask = df['date'].apply(lambda d, y=holdout_year: d.year == y)
        if h_test_mask.sum() < 100:
            logger.warning(f"  Skipping {holdout_year} — only {h_test_mask.sum()} games in holdout")
            continue

        wf_cal = CalibratedClassifierCV(_make_model(), method='isotonic', cv=5)
        wf_cal.fit(X[h_tr_mask], y[h_tr_mask])
        cal_probs = wf_cal.predict_proba(X[h_test_mask])[:, 1]
        y_test    = y[h_test_mask]

        logger.info(f"  Train n={int(h_tr_mask.sum())}  Holdout n={int(h_test_mask.sum())}")
        logger.info(f"  {'bin':<14}  {'n':>5}  {'mean_pred':>10}  {'win_rate':>10}  {'residual':>10}")
        for lo, hi in bins:
            mask = (cal_probs >= lo) & (cal_probs < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            mean_pred = float(cal_probs[mask].mean())
            win_rate  = float(y_test[mask].mean())
            residual  = win_rate - mean_pred
            logger.info(
                f"  [{lo:.2f}, {hi:.2f})  {n:>5d}  {mean_pred:>10.3f}  "
                f"{win_rate:>10.3f}  {residual:>+10.3f}"
            )

        brier = float(np.mean((cal_probs - y_test) ** 2))
        logger.info(f"  Brier score: {brier:.4f}  (0.25 = always-50% baseline)")

        cal_path = os.path.join(cal_dir, f'calibration_{holdout_year}.csv')
        df_holdout = df[h_test_mask].reset_index(drop=True)
        with open(cal_path, 'w', encoding='utf-8') as f:
            f.write('date,home_team,away_team,predicted_p_home,actual_home_win\n')
            for i in range(len(cal_probs)):
                row = df_holdout.iloc[i]
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                home = row.get('home_team', row.get('home_code', ''))
                away = row.get('away_team', row.get('away_code', ''))
                f.write(f"{date_str},{home},{away},{cal_probs[i]:.6f},{int(y_test[i])}\n")
        logger.info(f"  Raw predictions saved → {cal_path}")

    # ── Train final model on ALL data ────────────────────────────────────────
    logger.info("\nTraining final model on full dataset...")
    base_model = _make_model()
    # Isotonic calibration via 5-fold CV so probabilities reflect true frequencies
    final_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    final_model.fit(X, y)

    # Save
    payload = {'model': final_model, 'features': FEATURES}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved → {MODEL_PATH}")

    # Feature importance (from the underlying base estimator of the first calibrated fold)
    logger.info("Feature importances (gain):")
    base = final_model.calibrated_classifiers_[0].estimator
    importances = base.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = '█' * int(imp * 200)
        logger.info(f"  {feat:25s} {imp:.4f}  {bar}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train MLB XGBoost model')
    parser.add_argument('--start', type=int, default=2015, help='First season to include')
    parser.add_argument('--end',   type=int, default=2025, help='Last season to include')
    args = parser.parse_args()
    train(args.start, args.end)
