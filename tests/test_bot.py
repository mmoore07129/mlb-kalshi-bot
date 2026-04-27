"""
Unit tests for the math and parsing the bot depends on.

Run from the project root:
    python -m tests.test_bot

No pytest needed — plain asserts + a simple runner. These cover the EV / unit
sizing math and the Kalshi ticker parser — the two places a silent refactor
bug would most likely put real money at risk.
"""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk import calculate_ev, flat_unit_contracts
from kalshi.markets import (
    _parse_ticker,
    _split_team_codes,
    get_tick_for_price,
    _ask_depth_within,
)
from data.odds import _american_to_prob, _blend_probs, _vig_free_home_prob


PASS = 0
FAIL = 0


def check(cond: bool, label: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ── EV ────────────────────────────────────────────────────────────────────────
print("\n[calculate_ev]")

# At p=ask (fair price, no fee), gross EV = 0; net EV should be negative (fee takes the edge)
ev = calculate_ev(0.50, Decimal("0.50"), 0.07)
check(ev < 0, "fair-priced bet has negative EV after fee")

# Hand calc: p=0.60, ask=0.50, fee_multiplier=1.0 (standard Kalshi sports rate)
#   fee_rate_on_winnings = QUADRATIC_BASE * 1.0 * 0.50 = 0.07 * 0.50 = 0.035
#   b_net = (0.50/0.50) * (1 - 0.035) = 0.965
#   EV = 0.60 * 0.965 - 0.40 = 0.179
ev = calculate_ev(0.60, Decimal("0.50"), 1.0)
check(approx(ev, 0.179), f"p=0.6 ask=0.5 m=1.0 (std) -> EV=0.179 (got {ev:.4f})")

# Half-rate markets (S&P 500, Nasdaq): multiplier 0.5 -> smaller fee, higher EV
ev_half = calculate_ev(0.60, Decimal("0.50"), 0.5)
check(ev_half > ev, "half-rate markets have higher net EV than standard-rate")

# EV decreases as fee multiplier rises
ev_lo = calculate_ev(0.60, Decimal("0.50"), 0.0)
ev_hi = calculate_ev(0.60, Decimal("0.50"), 2.0)
check(ev_lo > ev > ev_hi, "EV monotone in fee multiplier")

# Edge: price at 0 or 1 -> sentinel
check(calculate_ev(0.5, Decimal("0"),    1.0) == -99.0, "ask=0 -> sentinel")
check(calculate_ev(0.5, Decimal("1"),    1.0) == -99.0, "ask=1 -> sentinel")
check(calculate_ev(0.5, Decimal("-0.1"), 1.0) == -99.0, "ask<0 -> sentinel")


# ── Flat-unit sizing ──────────────────────────────────────────────────────────
print("\n[flat_unit_contracts]")

# Healthy cash: $10 unit at $0.50 ask -> 20 contracts, $10 cost
n, cost = flat_unit_contracts(Decimal("0.50"), available_cash=1000.0, unit_dollars=10.0)
check(n == 20 and cost == 10.0, f"$10 @ $0.50 -> 20 contracts (got {n} @ ${cost})")

# Cash below unit -> trim to whatever cash supports
n, cost = flat_unit_contracts(Decimal("0.50"), available_cash=4.0, unit_dollars=10.0)
check(n == 8 and cost == 4.0, f"avail=$4 @ $0.50 -> 8 contracts (got {n} @ ${cost})")

# Cash below min floor -> skip
n, cost = flat_unit_contracts(Decimal("0.50"), available_cash=0.50, unit_dollars=10.0, min_bet_dollars=1.0)
check(n == 0 and cost == 0.0, f"avail=$0.50 below min -> skip (got {n} @ ${cost})")

# Edge prices -> sentinel
n, _ = flat_unit_contracts(Decimal("0"), available_cash=100.0, unit_dollars=10.0)
check(n == 0, "ask=0 -> zero")
n, _ = flat_unit_contracts(Decimal("1"), available_cash=100.0, unit_dollars=10.0)
check(n == 0, "ask=1 -> zero")


# ── Ticker parsing ────────────────────────────────────────────────────────────
print("\n[_parse_ticker]")

# Happy path, 3-char teams
p = _parse_ticker("KXMLBGAME-26APR161235WSHPIT-PIT")
check(p is not None and p['away_code'] == 'WSH' and p['home_code'] == 'PIT', "WSH@PIT parsed")
check(p['side_code'] == 'PIT' and p['is_yes_home'] is True, "YES=home detected (side=home)")

# 2-char team
p = _parse_ticker("KXMLBGAME-26JUN152210SFCIN-SF")
check(p is not None and p['away_code'] == 'SF' and p['home_code'] == 'CIN', "SF@CIN parsed (2-char away)")
check(p['is_yes_home'] is False, "YES=away detected")

# Mixed 3+2 char (LAA + NYY)
p = _parse_ticker("KXMLBGAME-26JUL041905LAANYY-NYY")
check(p is not None and p['away_code'] == 'LAA' and p['home_code'] == 'NYY', "LAA@NYY parsed")

# Invalid ticker shapes
check(_parse_ticker("NOT_KXMLB") is None, "non-KXMLBGAME ticker -> None")
check(_parse_ticker("KXMLBGAME-26XYZ161235WSHPIT-PIT") is None, "bogus month abbrev -> None")
check(_parse_ticker("KXMLBGAME-bad-bad") is None, "unparseable body -> None")


# ── Team code splitter ────────────────────────────────────────────────────────
print("\n[_split_team_codes]")
check(_split_team_codes('WSHPIT') == ('WSH', 'PIT'), "WSHPIT")
check(_split_team_codes('SFCIN')  == ('SF',  'CIN'), "SFCIN (2+3)")
check(_split_team_codes('LAANYY') == ('LAA', 'NYY'), "LAANYY (3+3)")
check(_split_team_codes('TBKC')   == ('TB',  'KC'),  "TBKC (2+2)")
check(_split_team_codes('ZZZZZZ') == (None, None),   "unknown codes -> None")


# ── Tick size lookup ──────────────────────────────────────────────────────────
print("\n[get_tick_for_price]")
check(get_tick_for_price('linear_cent',       Decimal('0.50')) == Decimal('0.01'),  "linear_cent -> 0.01")
check(get_tick_for_price('deci_cent',         Decimal('0.50')) == Decimal('0.001'), "deci_cent -> 0.001")
check(get_tick_for_price('tapered_deci_cent', Decimal('0.05')) == Decimal('0.001'), "tapered low tail -> 0.001")
check(get_tick_for_price('tapered_deci_cent', Decimal('0.50')) == Decimal('0.01'),  "tapered middle -> 0.01")
check(get_tick_for_price('tapered_deci_cent', Decimal('0.95')) == Decimal('0.001'), "tapered high tail -> 0.001")
check(get_tick_for_price('unknown_structure', Decimal('0.50')) == Decimal('0.01'),  "unknown -> default 0.01")


# ── American odds -> implied prob (Pinnacle conversion) ───────────────────────
print("\n[_american_to_prob]")
# -200: favorite; implied = 200 / (200+100) = 0.6667
check(approx(_american_to_prob(-200), 0.6667, 1e-3), "-200 -> ~66.67%")
# +150: underdog; implied = 100 / (150+100) = 0.40
check(approx(_american_to_prob(150),  0.40,   1e-3), "+150 -> ~40%")
# +100 (pickem): 0.50
check(approx(_american_to_prob(100),  0.50,   1e-3), "+100 -> 50%")

# Vig removal: two negatives sum to > 1 before normalization
home_raw = _american_to_prob(-150)
away_raw = _american_to_prob(-110)
total = home_raw + away_raw
check(total > 1.0, "two-favorite market has total implied > 1 (vig present)")
home_fair = home_raw / total
away_fair = away_raw / total
check(approx(home_fair + away_fair, 1.0), "vig-free probs sum to 1.0")


# ── Orderbook depth ───────────────────────────────────────────────────────────
print("\n[_ask_depth_within]")
# YES ask is derived from NO bid: ask = 1 - no_bid. Ascending no_bids.
no_bids = [
    ["0.52", "500"],   # → YES ask 0.48 / 500
    ["0.53", "300"],   # → YES ask 0.47 / 300
    ["0.54", "200"],   # → YES ask 0.46 / 200
    ["0.55", "100"],   # → YES ask 0.45 / 100
    ["0.56", "50"],    # top ask: 0.44 / 50
]
check(_ask_depth_within(no_bids, Decimal("0.44")) == 50,   "depth at top-of-book = 50")
check(_ask_depth_within(no_bids, Decimal("0.45")) == 150,  "depth within 1¢ = 150")
check(_ask_depth_within(no_bids, Decimal("0.46")) == 350,  "depth within 2¢ = 350")
check(_ask_depth_within([],      Decimal("0.50")) == 0,    "empty book -> 0")


# ── Depth-capped flat unit ────────────────────────────────────────────────────
print("\n[flat_unit_contracts depth cap]")
n_nocap, _ = flat_unit_contracts(Decimal("0.50"), 1000.0, 10.0)
n_cap,   _ = flat_unit_contracts(Decimal("0.50"), 1000.0, 10.0, max_depth_contracts=5)
check(n_cap == 5 and n_nocap == 20, f"depth cap clips unit (got {n_cap} vs uncapped {n_nocap})")

n_big, _ = flat_unit_contracts(Decimal("0.50"), 1000.0, 10.0, max_depth_contracts=10_000_000)
check(n_big == n_nocap, "very large cap is no-op")

n_zero, _ = flat_unit_contracts(Decimal("0.50"), 1000.0, 10.0, max_depth_contracts=0)
check(n_zero == 0, "zero-depth cap -> no bet")


# ── Multi-book blend ──────────────────────────────────────────────────────────
print("\n[_blend_probs / _vig_free_home_prob]")
p = _vig_free_home_prob(-200, +170)
check(approx(p, 0.667 / (0.667 + 0.370), 1e-3), f"vig-free P(home) for -200/+170 = {p:.4f}")

blend, std = _blend_probs({'pinnacle': 0.60, 'lowvig': 0.60, 'betonlineag': 0.60})
check(approx(blend, 0.60) and approx(std, 0.0), "perfect book agreement -> blend=0.60 std=0")

blend, std = _blend_probs({'pinnacle': 0.58, 'lowvig': 0.60, 'betonlineag': 0.62})
# weighted 0.58*0.55 + 0.60*0.30 + 0.62*0.15 = 0.592
check(approx(blend, 0.592, 1e-3), f"weighted blend = {blend:.4f}")
check(std > 0.0, "divergent books -> positive std")

blend, std = _blend_probs({'pinnacle': 0.55})
check(approx(blend, 0.55) and approx(std, 0.0), "single book -> blend=prob std=0")

blend, std = _blend_probs({'pinnacle': 0.50, 'lowvig': 0.60})
# renormalized weights: 0.55/0.85=0.647, 0.30/0.85=0.353
# blend = 0.50*0.647 + 0.60*0.353 = 0.5353
check(approx(blend, 0.5353, 1e-3), f"two-book renormalized blend = {blend:.4f}")

blend, std = _blend_probs({})
check(approx(blend, 0.5), "empty input falls back to 0.5")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
print('='*50)
if FAIL:
    sys.exit(1)
