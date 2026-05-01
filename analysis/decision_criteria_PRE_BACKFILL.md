# Pre-registered decision criteria for model-vs-Pinnacle CLV backtest

Locked 2026-05-01, BEFORE backfill data was visible.

The point of pre-registration: prevent post-hoc rationalization. If the
backfill results are ambiguous, the temptation is to find a story that
keeps the model alive. These criteria are the contract.

## Path 3 (model as primary trigger on Kalshi MLB ML) — pursue IFF all:

1. On the ≥5pp disagreement subset (|model_p_home - blend_p_home| ≥ 0.05):
   - `model_directional_accuracy > blend_directional_accuracy`
   - 95% CI lower bound on `model_directional_accuracy > blend's accuracy`

2. `Brier(model) < Brier(blend)` on the same ≥5pp subset.

3. Robustness: both proportional and Shin de-vig give the same conclusion.

4. Stability: the signal direction is consistent across 2024 AND 2025
   (magnitudes can differ; sign cannot flip).

## Path 2 (confirm dominated, pivot off Kalshi MLB ML) — IFF:

- `model_acc ≤ blend_acc` across ALL thresholds in {1pp, 3pp, 5pp, 7pp,
  10pp, 12pp, 15pp}.
- `Brier(model) ≥ Brier(blend)` across all thresholds.

## Ambiguous (meta-model territory) — IF:

- Mixed signals (e.g., Brier favors model on some thresholds but directional
  doesn't, or 2024 signal disappears in 2025).
- In this case: do NOT change bot config based on this backtest. Either
  collect more data going forward or design a meta-model with model_p,
  blend_p, gap, and game features as inputs.

## Reading discipline

- Always look at the threshold sweep TABLE before any single number.
- Always check both de-vig methods before deciding.
- Always include subset base rate in the comparison (model_acc vs blend_acc
  on THIS subset, not vs all-game home_win_rate).

Signed: Claude Code (Opus 4.7 max) on behalf of mmoore07129.
