from __future__ import annotations

import pandas as pd

from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.autorank import AutorankConstraints, apply_autorank_filters


def test_meta_diag_filters_reduce_frame_and_are_deterministic() -> None:
    df = pd.DataFrame(
        [
            {
                "portfolio_profit_factor": 1.6,
                "portfolio_sharpe": 1.1,
                "portfolio_max_drawdown_pct": 15.0,
                "n_trades": 60,
                "correlation_max": 0.4,
                "tickers": "AAA",
            },
            {
                "portfolio_profit_factor": 1.0,
                "portfolio_sharpe": 0.2,
                "portfolio_max_drawdown_pct": 55.0,
                "n_trades": 15,
                "correlation_max": 0.95,
                "tickers": "BBB",
            },
        ]
    )
    diag_a = build_meta_diag_frame(df, meta_weight=0.5, model=None)
    diag_b = build_meta_diag_frame(df, meta_weight=0.5, model=None)
    pd.testing.assert_frame_equal(
        diag_a.drop(columns=["timestamp_evaluated"]),
        diag_b.drop(columns=["timestamp_evaluated"]),
        check_exact=False,
    )
    constraints = AutorankConstraints(min_pf=1.2, min_sharpe=0.5, max_dd=0.3, min_trades=20)
    filtered, stats = apply_autorank_filters(diag_a, constraints)
    assert len(filtered) == 1
    assert stats["survivors"] == 1
