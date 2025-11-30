from __future__ import annotations

import math

import pandas as pd

from hyprl.portfolio.core import (
    build_portfolio_equity,
    compute_correlation_matrix,
    compute_portfolio_stats,
)


def test_build_portfolio_equity_equal_weights() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    series_a = pd.Series([100.0, 110.0, 120.0], index=idx)
    series_b = pd.Series([100.0, 95.0, 97.0], index=idx)
    portfolio = build_portfolio_equity({"A": series_a, "B": series_b}, total_capital=200.0)
    assert math.isclose(portfolio.iloc[0], 200.0, rel_tol=1e-9)
    assert math.isclose(portfolio.iloc[-1], 217.0, rel_tol=1e-9)


def test_compute_portfolio_stats_outputs_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([100.0, 102.0, 101.0, 104.0, 107.0], index=idx)
    stats = compute_portfolio_stats(equity, initial_balance=100.0, seed=1, bootstrap_runs=32)
    assert stats["final_balance"] == 107.0
    assert stats["profit_factor"] >= 0.0
    assert stats["max_drawdown_pct"] >= 0.0
    assert 0.0 <= stats["risk_of_ruin"] <= 1.0


def test_compute_correlation_matrix_values() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    returns_a = pd.Series([0.01, 0.02, -0.01, 0.0], index=idx)
    returns_b = pd.Series([0.01, 0.02, -0.01, 0.0], index=idx)
    returns_c = pd.Series([0.0, -0.01, 0.02, 0.01], index=idx)
    matrix = compute_correlation_matrix({"A": returns_a, "B": returns_b, "C": returns_c})
    assert matrix.shape == (3, 3)
    assert math.isclose(matrix.loc["A", "B"], 1.0, rel_tol=1e-9)
    assert matrix.loc["A", "C"] <= 1.0
