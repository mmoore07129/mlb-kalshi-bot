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
