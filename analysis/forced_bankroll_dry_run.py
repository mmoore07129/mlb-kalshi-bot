"""
Forced-bankroll dry-run: smoke test the new bet-selection logic (no veto,
gap veto raised to 50pp, paper-trade harness) on today's actual slate
without needing real funds in the Kalshi account.

Monkey-patches KalshiClient.get to return a fake $100 balance for
/portfolio/balance, then calls main.run(dry_run=True). All other API
calls (markets, prices, settlements) still hit real Kalshi.

Use this once after a code change to confirm the new logic actually
exercises end-to-end without surprises.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import kalshi.client as kalshi_client_mod

_original_get = kalshi_client_mod.KalshiClient.get
FAKE_BALANCE_CENTS = 10000  # $100.00


def _patched_get(self, endpoint, params=None, auth=True):
    if endpoint == '/portfolio/balance':
        return {'balance': FAKE_BALANCE_CENTS, 'portfolio_value': 0}
    return _original_get(self, endpoint, params=params, auth=auth)


kalshi_client_mod.KalshiClient.get = _patched_get


import main  # noqa: E402

if __name__ == '__main__':
    main.run(dry_run=True, watch=False)
