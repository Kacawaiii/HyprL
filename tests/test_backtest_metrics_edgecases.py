from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hyprl.backtest.runner import summarize_oos_metrics


def _mk_trade(pnl: float, ts: str) -> SimpleNamespace:
    ts_obj = pd.Timestamp(ts)
    return SimpleNamespace(pnl=float(pnl), exit_time=ts_obj, exit_timestamp=ts_obj)


def test_metrics_pf_clip_and_no_trades_handling() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="H")
    equity = pd.Series(np.linspace(1000, 1100, len(idx)), index=idx)
    trades = [_mk_trade(5, "2024-01-02"), _mk_trade(7, "2024-01-04")]
    metrics = summarize_oos_metrics(equity, trades, bars_per_year=24 * 365)
    assert metrics["pf"] == 10.0

    metrics_no_trades = summarize_oos_metrics(equity, [], bars_per_year=24 * 365)
    assert "cvar95_m" in metrics_no_trades
    assert "pf" in metrics_no_trades


def test_metrics_domains_and_types() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="H")
    equity = pd.Series(np.linspace(1000, 1200, len(idx)), index=idx)
    trades = [_mk_trade(10, "2024-01-10"), _mk_trade(-4, "2024-01-16"), _mk_trade(8, "2024-02-01")]
    metrics = summarize_oos_metrics(equity, trades)
    assert "sharpe" in metrics and "calmar" in metrics
    assert metrics["pf"] >= 0.0
    assert 0.0 <= metrics["maxdd"] <= 1.0 or math.isnan(metrics["maxdd"])
