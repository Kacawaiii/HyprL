"""Read-only application API: payload contracts and hard limits.

The frontend is never the source of truth. Every business value it displays --
a direction, a strength, an exposure, a rank correlation -- is computed in
Python and shipped as data. The browser formats and lays out; it does not
decide. That rule is what keeps a cockpit from slowly growing a second,
divergent implementation of the trading logic in TypeScript.

The limits below are infrastructure boundaries, not financial parameters. They
exist so that no view can ever ask for an unbounded slice of history: a page
that fetches "everything" works fine on a year of hourly candles and dies on
five, and the failure arrives long after the design decision that caused it.
"""

from __future__ import annotations

APP_API_VERSION = "trading-lab.app-api.v1"

# Bounded from day one. A request above the maximum is refused rather than
# silently clamped -- a caller that asked for 50 000 rows and received 1 000
# without being told will build a UI on a false assumption.
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 1_000

# A chart has a finite number of pixels; shipping more points than this is
# bandwidth spent on detail nobody can see.
DEFAULT_CHART_POINTS = 500
MAX_CHART_POINTS = 2_000

SUPPORTED_PRODUCTS = ("BTC-USD", "ETH-USD")
SUPPORTED_TIMEFRAME = "1h"

# Every capability the UI may branch on, stated once. False here means the
# feature genuinely does not exist -- the UI must show that, not simulate it.
CAPABILITIES = {
    "market_history": True,
    "signal_engine": True,
    "position_target": True,
    "economic_backtest": False,
    "paper_trading": False,
    "live_trading": False,
    "realtime_stream": False,
}


class AppApiError(RuntimeError):
    """Raised when a request violates the API contract."""

    status = 400


class NotFoundError(AppApiError):
    status = 404
