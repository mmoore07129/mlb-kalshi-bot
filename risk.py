"""
Risk management — EV calculation and flat-unit position sizing.

EV is net of Kalshi taker fee so the gate reflects what you actually keep,
not gross payout. Sizing is flat-unit: every passing bet is sized to a fixed
dollar amount regardless of edge size.

Kalshi's quadratic fee formula is:
    per-contract fee = QUADRATIC_BASE × fee_multiplier × P × (1 - P)

Where QUADRATIC_BASE is 0.07 (the industry-standard rate) and fee_multiplier
is a per-series scaling factor returned by GET /series/{ticker}:
  - 1.0 = standard fee rate (most markets, including sports)
  - 0.5 = half-rate markets (S&P 500, Nasdaq-100)
  - other values possible for special events

Empirically verified: back-computing fees from real KXMLBGAME settlements
matches 0.07 × N × P × (1-P) to the rounding cent.

As a fraction of winnings (gross = 1-P per contract) the effective fee rate is
QUADRATIC_BASE × fee_multiplier × P — i.e., linear in price.
"""

from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

# Kalshi's quadratic fee base coefficient. The per-series fee_multiplier
# scales this, so e.g. a multiplier of 1.0 means the standard 0.07 rate.
QUADRATIC_BASE = 0.07


def _fee_rate_on_winnings(price: float, fee_multiplier: float) -> float:
    """
    Effective fee as a fraction of gross winnings per contract.
      per-contract fee = QUADRATIC_BASE × fee_multiplier × P × (1-P)
      gross winnings   = (1 - P)
      fee / winnings   = QUADRATIC_BASE × fee_multiplier × P   (the 1-P cancels)
    """
    return QUADRATIC_BASE * fee_multiplier * price


def calculate_ev(p_model: float, ask_price: Decimal, fee_multiplier: float) -> float:
    """
    Net expected value per $1 spent, after Kalshi taker fee on winnings.

    Gross odds:  b = (1 - price) / price
    Net odds:    b_net = b × (1 - fee_rate)
    Net EV = P × b_net - (1 - P)
    """
    price = float(ask_price)
    if price <= 0 or price >= 1:
        return -99.0
    fee_rate = _fee_rate_on_winnings(price, fee_multiplier)
    b_net = ((1.0 - price) / price) * (1.0 - fee_rate)
    return p_model * b_net - (1.0 - p_model)


def flat_unit_contracts(
    ask_price: Decimal,
    available_cash: float,
    unit_dollars: float,
    min_bet_dollars: float = 1.00,
    max_depth_contracts: int | None = None,
) -> tuple[int, float]:
    """
    Returns (num_contracts, dollar_cost) for a flat-unit bet.

    Targets `unit_dollars` regardless of edge size. Trims down only when:
      - available_cash < unit_dollars (can't spend what we don't have)
      - max_depth_contracts would force walking the book past the quoted ask
    """
    price = float(ask_price)
    if price <= 0 or price >= 1:
        return 0, 0.0

    target_dollars = min(unit_dollars, available_cash)
    if target_dollars < min_bet_dollars:
        return 0, 0.0

    contracts = max(1, int(target_dollars / price))

    if max_depth_contracts is not None and max_depth_contracts >= 0:
        if contracts > max_depth_contracts:
            logger.info(
                f"Flat unit wanted {contracts} contracts but depth only "
                f"{max_depth_contracts} near top-of-book; trimming to depth"
            )
            contracts = max_depth_contracts

    if contracts <= 0:
        return 0, 0.0

    actual_cost = contracts * price
    if actual_cost < min_bet_dollars:
        return 0, 0.0

    return contracts, round(actual_cost, 2)
