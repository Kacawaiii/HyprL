from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.analysis.trade_aggregator import (
    compute_metrics,
    compute_rolling,
)
from hyprl.analysis.trade_gates import check_gate1, check_gate2


def _make_trades(pnl_values: list[float], strategy_id: str) -> pd.DataFrame:
    n = len(pnl_values)
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    entry_ts = [base_time + pd.Timedelta(minutes=i) for i in range(n)]
    exit_ts = [ts + pd.Timedelta(minutes=1) for ts in entry_ts]
    signs = np.sign(pnl_values)
    sides = ["LONG" if s >= 0 else "SHORT" for s in signs]
    return pd.DataFrame(
        {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "symbol": ["NVDA"] * n,
            "side": sides,
            "qty": [1.0] * n,
            "entry_price": [100.0] * n,
            "exit_price": [100.0 + v for v in pnl_values],
            "pnl": pnl_values,
            "pnl_pct": [v / 100.0 for v in pnl_values],
            "exit_reason": ["signal"] * n,
            "source_type": ["backtest"] * n,
            "session_id": ["sess"] * n,
            "strategy_label": ["strat"] * n,
            "strategy_id": [strategy_id] * n,
        }
    )


def test_gate1_passes_good_strategy() -> None:
    pnl = [1.0] * 320 + [-0.2] * 10  # PF high, mild drawdown
    trades = _make_trades(pnl, "good_gate1")
    metrics = compute_metrics(trades, initial_equity=1.0)
    rolling = compute_rolling(trades, window=100, initial_equity=1.0)
    gate1 = check_gate1(metrics, rolling)
    gate2 = check_gate2(metrics, rolling)
    assert gate1.passed is True
    assert gate2.passed is False  # not enough trades


def test_gate2_requires_trades() -> None:
    pnl = [1.0 if i % 2 == 0 else 0.8 for i in range(900)]  # PF -> inf, DD ~0, Sharpe finite
    trades = _make_trades(pnl, "almost_gate2")
    metrics = compute_metrics(trades, initial_equity=1.0)
    rolling = compute_rolling(trades, window=100, initial_equity=1.0)
    gate1 = check_gate1(metrics, rolling)
    gate2 = check_gate2(metrics, rolling)
    assert gate1.passed is True
    assert gate2.passed is False  # trades < 1000


def test_gate2_passes_good_strategy() -> None:
    pnl = [1.0 if i % 2 == 0 else 0.8 for i in range(1000)]  # PF -> inf, DD ~0, Sharpe finite
    trades = _make_trades(pnl, "good_gate2")
    metrics = compute_metrics(trades, initial_equity=1.0)
    rolling = compute_rolling(trades, window=100, initial_equity=1.0)
    gate1 = check_gate1(metrics, rolling)
    gate2 = check_gate2(metrics, rolling)
    assert gate1.passed is True
    assert gate2.passed is True


def test_bad_strategy_fails_gate1() -> None:
    pnl = [1.0] * 150 + [-2.0] * 170  # PF < 1, DD large
    trades = _make_trades(pnl, "bad_gate1")
    metrics = compute_metrics(trades, initial_equity=1.0)
    rolling = compute_rolling(trades, window=100, initial_equity=1.0)
    gate1 = check_gate1(metrics, rolling)
    assert gate1.passed is False
