from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.eth_paper.backtest import backtest_series, build_target_series, summarize


def prices() -> pd.Series:
    return pd.Series(
        np.geomspace(500.0, 4_000.0, 500),
        index=pd.date_range("2024-01-01", periods=500, freq="D", tz="UTC"),
    )


def test_backtest_lags_target_by_one_full_bar() -> None:
    targets = build_target_series(prices())
    frame = backtest_series(prices(), cost_bps=0)
    assert frame["position"].equals(targets.shift(1).fillna(0.0))


def test_costs_only_reduce_final_equity_and_metrics_are_finite() -> None:
    free = backtest_series(prices(), cost_bps=0)
    costly = backtest_series(prices(), cost_bps=50)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]

    metrics = summarize(costly)
    assert np.isfinite(metrics.cagr)
    assert np.isfinite(metrics.sharpe)
    assert metrics.max_drawdown <= 0
    assert 0 <= metrics.average_exposure <= 0.75
