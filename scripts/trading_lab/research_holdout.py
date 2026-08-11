"""The registered confirmatory holdout, defined once, with no dependencies.

This lives in its own module for one reason: the guard that enforces the window
must work in an environment that has no ML stack installed. Reading the
definition out of `real_benchmark_v2` — which imports the model classes to hash
their specs — would make a *safety* component depend on scikit-learn being
present. A safety check that silently cannot load is worse than none.

`real_benchmark_v2` imports the same dictionary, so there is still exactly one
definition and the V2 benchmark identity is unchanged.
"""

from __future__ import annotations

CONFIRMATORY_HOLDOUT_V2 = {
    "holdout_id": "coinbase_confirmatory_2026q4",
    "provider": "coinbase_exchange_rest",
    "products": ["BTC-USD", "ETH-USD"],
    "timeframe": "1h",
    "range_start": "2026-09-01T00:00:00Z",
    "range_end": "2026-11-30T23:00:00Z",
    "role": "confirmatory",
    "captured": False,
    "single_use": True,
    "note": (
        "One evaluation only, of the V2 contract exactly as registered here. Once "
        "observed this window is spent too, and any later hypothesis needs either a "
        "fresh holdout or an explicit exploratory label. If V2 changes before this "
        "window is evaluated, the holdout must point at the new version explicitly."
    ),
}
