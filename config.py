import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# Kalshi credentials
KEY_ID = os.environ.get("KALSHI_KEY_ID", "").strip()
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(__file__), "credentials", "kalshi_private_key.txt")
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

if not KEY_ID:
    raise RuntimeError(
        "KALSHI_KEY_ID is not set. Copy .env.example to .env and fill in your credentials."
    )

# MLB
MLB_SEASON = 2026

# Trading parameters — flat-unit sizing.
# Every bet that passes the EV gate is sized to FLAT_UNIT_DOLLARS, regardless of
# edge size or bankroll. Simpler than Kelly and easier to reason about for a
# small-bankroll data-collection phase. available_cash still gates (you can't
# spend money you don't have), and depth still caps (don't walk the book).
FLAT_UNIT_DOLLARS = 20.00  # Dollar size of every bet that passes the gate
MIN_BET_DOLLARS   = 1.00   # Sanity floor — orders below this never go through

# EV thresholds (net of fee — use net EV, not gross)
PINNACLE_MIN_EV = 0.005   # Single-book path (only 1 sharp book quoted the game)
MODEL_MIN_EV    = 0.08    # Model fallback: wider margin for noisier signal

# Dynamic EV threshold when 2+ sharp books quoted the game:
#   threshold = clamp(BASE + STD_MULT × prob_std, BASE, CEIL)
# When books all agree (std≈0), threshold ≈ BASE (most permissive).
# When books disagree (std high), threshold rises up to CEIL (more selective).
EV_THRESHOLD_BASE     = 0.005
EV_THRESHOLD_STD_MULT = 1.5
EV_THRESHOLD_CEIL     = 0.08

# Model confidence gate (only applies when Pinnacle unavailable).
CONFIDENCE_THRESHOLD = 0.55

# Circuit breaker for catastrophic model failure (e.g., feature pipeline bug
# producing wild predictions). Set far outside the legitimate forecast-
# disagreement range; legitimate model-vs-book disagreements should be
# allowed through to bet selection. Not a forecast-quality gate.
MODEL_PINNACLE_GAP_MAX = 0.50

# Watch mode: after placing orders, re-check orderbook + Pinnacle periodically
# and amend resting orders if prices move materially.
WATCH_INTERVAL_SECONDS = 180          # 3 minutes between polls
WATCH_AMEND_TICK_THRESHOLD = 2        # amend when ask moves by >= this many cents

# Set True to simulate without placing real orders
DRY_RUN = False

# The Odds API key for Pinnacle moneylines (https://the-odds-api.com/ — free tier: 500 req/month).
# When set, Pinnacle's vig-free probability is the primary signal; model is fallback only.
# Leave empty to use model probabilities only.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()

# Kalshi fee structure. fee_type is 'quadratic' / 'quadratic_with_maker_fees' / 'flat'
# and fee_multiplier is a double (default 0.07 for quadratic MLB markets). Both are
# fetched at runtime from GET /series/{SERIES_TICKER} — these are only fallbacks
# in case that request fails.
FEE_MULTIPLIER_FALLBACK = 0.07
FEE_TYPE_FALLBACK       = 'quadratic'
