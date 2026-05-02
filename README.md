mlb-kalshi-bot
Automated MLB betting bot trading event contracts on Kalshi's KXMLBGAME moneyline market. Sources sharp probabilities from Pinnacle (via The Odds API), runs an XGBoost model as a fallback signal, scores net-of-fee expected value against Kalshi's live orderbook, and places flat-unit orders when edge clears a dynamic threshold. Includes a watcher that amends or cancels resting orders as prices drift, a CLV tracker for measuring real edge vs. lucky streaks, and a settlement loop that reconciles outcomes against the MLB Stats API.
Built and operated solo. ~2,500 lines of Python. Live on a DigitalOcean droplet under cron.

Why this project exists
Sports betting markets are noisy, prediction-market venues like Kalshi are new enough that pricing inefficiencies still show up, and the whole problem reduces to a clean stack: estimate a probability, score it against a price, account for fees, and size the bet. Every layer is testable. That made it a good vehicle for combining statistical modeling, API integration, cost accounting, and live operations into one project.
The headline design decision is Pinnacle-primary, model-fallback-only: Pinnacle's vig-free blend is the primary signal because its features are a strict superset of anything the model sees. As of 2026-05-01, the XGBoost model has no veto power on Pinnacle-sourced bets — a 2024–2025 CLV backtest showed the model is anti-calibrated specifically on disagreement games (when model and blend pick opposite sides at ≥20pp gap, the blend wins ~77% of head-to-head). The model now serves only as a fallback when Pinnacle has no line, plus a >50pp gap circuit-breaker for catastrophic data-quality failures. This bounds upside (the model can't beat Pinnacle's close) but maximizes robustness.

How it works
Daily placement (9:00 AM ET cron):

Fetch vig-free probabilities from Pinnacle, LowVig, and BetOnline via The Odds API. Blend them (0.55 / 0.30 / 0.15).
For each game, run the XGBoost model on team and starting-pitcher rolling stats (used as fallback only).
Pull Kalshi's orderbook for the corresponding KXMLBGAME ticker.
Compute net-of-fee EV on both sides at the current ask using Kalshi's quadratic fee formula: fee = 0.07 × multiplier × P × (1 − P).
Apply the gate stack — model-Pinnacle gap circuit breaker (>50pp), dynamic EV threshold scaled by sharp-book disagreement, available cash, depth cap.
Place flat-unit YES orders ($20 per qualifying bet) for any side that survives all gates. Log the decision (placed or skipped, with reason) to daily_stats.csv.

Watch mode (every 3 min until first pitch):

Amend resting order prices when the ask moves ≥ 2¢.
Cancel orders if EV drops below the per-order threshold that originally admitted them.

CLV snapshot (every 5 min, 11 AM – midnight ET):

For each unsettled bet placed today, snapshot the current sharp-book fair price.
Last write before the game leaves the feed becomes the de-facto closing line.
Two CLV metrics computed: probability-space (p_close − p_used) and ROI-space (your_decimal / close_decimal − 1).
A parallel job (snapshot_all_analyzed.py) captures Pinnacle close for every analyzed game, not just placed bets — foundation for the gap-aware paper-trade evaluation.

Settlement (2:00 AM ET cron):

Pull Kalshi's /portfolio/settlements, cross-check against MLB Stats API final scores.
Use Kalshi's reported fee_cost (actual, not computed) for precise PnL.
Print summary: win rate, ROI, by-date breakdown, CLV summary by source and edge bucket.


Staking & gates
FLAT-UNIT SIZING ($20/bet, raised from $10 on 2026-05-01). Kelly was removed on 2026-04-27: at low bankroll, 1/8 Kelly on a 0.5% edge rounded below the min-bet floor, so the bot never actually bet. Trade-off: per-bet exposure is now ~33% of bankroll at small bankroll levels, to be revisited as bankroll grows.

DYNAMIC EV THRESHOLD: threshold = clamp(0.005 + 1.5 × prob_std, 0.005, 0.08). Floor lowered from 2.5% to 0.5% on 2026-04-27 to start collecting placed-bet data on thin edges (data-collection phase, not a high-confidence growth strategy). Single-sharp-book path uses a static 0.5% min EV; model-only fallback keeps a strict 8% min EV and 55% confidence gate.


The model
XGBoost binary classifier predicting P(home team wins), calibrated via isotonic regression on 5-fold stratified CV.

Training data: 24,686 MLB regular-season games, 2015–2025, from MLB Stats API.
Features (53): team rolling Pythagorean, Log5, win percentages (YTD and last 5); starting-pitcher rolling ERA, WHIP, K%, BB%, K-BB%, HR/9 (YTD and L5); SP context (days rest, starts YTD, rookie/early-season flag); SP matchup deltas.
Validation: 5-fold CV → walk-forward held-out final year → final fit on full data.
Leakage discipline: all rolling stats for game i derive only from games 0..i−1. League-average fallback for pitchers with fewer than 3 prior starts, flagged via sp_is_prior.
Performance (held-out 2025): 56.0% ± 0.5% accuracy overall; 65.9% at the ≥66% confidence band (n=323).


Architecture
main.py                  Entry point. Daily placement loop.
config.py                Parameters: thresholds, sizing, MLB season, Kalshi URLs.
risk.py                  Net-of-fee EV calc, flat-unit sizing, fee formula.
watcher.py               Resting-order watcher. Amend/cancel based on price drift and EV decay.
settle.py                End-of-day settlement + performance summary.
clv_snapshot.py          Standalone cron job. Captures closing-line value for placed bets.
snapshot_all_analyzed.py Parallel cron job. Captures Pinnacle close for all analyzed games.
fill_all_outcomes.py     Daily cron. Fills home_win for every analyzed game (paper-trade foundation).
paper_trade.py           Gap-aware threshold harness. Logging only; does not change behavior.
ledger.py                24-column CSV ledger with auto-migrating schema.
daily_stats.py           Per-game decision diagnostics with rejection-reason taxonomy.

kalshi/                  RSA-signed Kalshi API client and market helpers.
data/                    MLB Stats API, sharp-book odds, pitcher rolling stats.
models/                  XGBoost training pipeline + inference wrapper.
tests/                   Unit tests.
Two deployment locations stay in sync: a Windows local for development and a Ubuntu droplet for production. Crontab on the droplet runs placement, CLV capture, settlement, and outcome fills on schedule.

What's instrumented
The single highest-leverage decision was building diagnostics before scaling capital. Every game the bot considers gets a row in daily_stats.csv with the EV computation, the gate that rejected it (or placed), the model and Pinnacle probabilities, and the Kalshi quote. Every placed bet flows into a 24-column ledger with CLV columns that get back-filled by a separate cron until the game locks. A second cron snapshots Pinnacle close for every analyzed game (not just bets placed), so the gap-aware paper-trade hypothesis can be evaluated forward-going. Settlement uses Kalshi's reported fees, not computed fees, so PnL is exact.
This makes the bot self-debugging: "why didn't it bet on game X" is a one-row CSV lookup, not a grep through session logs.

Setup
bash# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Then edit .env with your KALSHI_KEY_ID and ODDS_API_KEY.
# Place your Kalshi RSA private key at credentials/kalshi_private_key.txt (chmod 600).

# Train the model (first run: ~30 min, builds pitcher cache)
python models/train.py

# Dry-run today's slate (no orders placed)
python main.py --dry-run

# Live mode + watch
python main.py --watch

# Settle yesterday's bets
python settle.py
Windows users can use the included .bat wrappers (run_dry.bat, train.bat, settle.bat).

Stack
Python 3.12 · XGBoost · scikit-learn · pandas · requests · cryptography (RSA-PSS for Kalshi auth) · The Odds API · MLB Stats API · cron · DigitalOcean

Status
Live, with a small bankroll, in a data-collection phase under a 0.5% net-EV gate and $20 flat-unit sizing. Currently iterating on park factors, bullpen state, and a small-sample-pitcher flag, plus collecting placed-bet data to validate the gap↔EV correlation hypothesis that motivated the gate drop. See OVERVIEW.txt for the full design doc, including the changelog of gate adjustments and the rationale behind each decision.

Disclaimer
This is a personal project for educational and operational-engineering purposes. Sports betting and event-contract trading involve substantial risk of loss. Nothing here is financial advice. Don't deploy capital you can't afford to lose.
