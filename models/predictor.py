"""
XGBoost inference for MLB game outcome prediction.

Loads the pre-trained model from xgb_model.pkl (produced by models/train.py).
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_model.pkl')

_model    = None
_features = None


def _load():
    global _model, _features
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run 'python models/train.py' first."
            )
        with open(MODEL_PATH, 'rb') as f:
            payload = pickle.load(f)
        _model    = payload['model']
        _features = payload['features']
        logger.info(f"XGBoost model loaded ({len(_features)} features)")


def predict(
    home_code: str,
    away_code: str,
    home_stats: dict,
    away_stats: dict,
    home_sp_stats: dict | None = None,
    away_sp_stats: dict | None = None,
) -> dict:
    """
    Predict home team win probability.

    SP stat dicts are optional — when omitted, build_game_features falls back to
    priors. This keeps backward-compatibility with old model.pkl files that only
    use the 20 team features. New SP-aware models need both dicts supplied.

    Returns:
        p_home      — probability home team wins [0.05, 0.95]
        p_away      — 1 - p_home
        confidence  — distance from 50% = abs(p_home - 0.5), range [0, 0.5]
        features    — feature dict used
        details     — human-readable summary string
    """
    from data.game_logs import build_game_features
    _load()

    features = build_game_features(home_stats, away_stats, home_sp_stats, away_sp_stats)
    X        = [[features[f] for f in _features]]
    p_home   = float(_model.predict_proba(X)[0][1])
    p_home   = max(0.05, min(0.95, p_home))

    # Include SP summary in log line when available — useful to spot feature plumbing
    # issues early (e.g., missing pitcher data falling through to priors silently).
    sp_summary = ''
    if 'home_sp_l5_era' in features:
        # Mark SP stats that came from the league-average prior (rookie / early-season)
        # with a '*' so rookie-driven predictions are easy to spot in logs.
        h_prior = features.get('home_sp_is_prior', 0)
        a_prior = features.get('away_sp_is_prior', 0)
        h_mark  = '*' if h_prior else ''
        a_mark  = '*' if a_prior else ''
        sp_summary = (
            f"  SP L5 ERA: H={features['home_sp_l5_era']:.2f}{h_mark}/A={features['away_sp_l5_era']:.2f}{a_mark}  "
            f"K%: H={features['home_sp_l5_k_pct']:.0%}/A={features['away_sp_l5_k_pct']:.0%}  "
            f"rest: H={int(features['home_sp_days_rest'])}/A={int(features['away_sp_days_rest'])}"
        )
    details = (
        f"pyth={features['home_pyth']:.3f} vs {features['away_pyth']:.3f}  "
        f"log5={features['log5']:.3f}  "
        f"r5_pyth={features['home_r5_pyth']:.3f} vs {features['away_r5_pyth']:.3f}  "
        f"r5_log5={features['r5_log5']:.3f}  "
        f"Δpyth={features['pyth_delta']:+.3f}"
        f"{sp_summary}  "
        f"[XGB: {p_home:.1%}]"
    )
    logger.info(f"{home_code} vs {away_code}: {details}")

    return {
        'p_home':     p_home,
        'p_away':     1.0 - p_home,
        'confidence': abs(p_home - 0.5),
        'features':   features,
        'details':    details,
    }
