from __future__ import annotations

import math

import numpy as np

from hyprl.risk.metrics import (
    bootstrap_equity_drawdowns,
    compute_basic_stats,
    compute_risk_of_ruin,
)


def test_basic_stats_computes_expectancy_and_ratio() -> None:
    returns = [0.02, -0.01, 0.015, -0.02, 0.03]
    stats = compute_basic_stats(returns)
    assert math.isclose(stats["expectancy"], np.mean(returns))
    assert stats["win_rate"] == 0.6
    assert stats["avg_win"] > 0
    assert stats["avg_loss"] < 0
    assert stats["avg_win_loss_ratio"] > 0


def test_risk_of_ruin_bounds_and_monotonicity() -> None:
    winning = [0.03, 0.02, 0.01, -0.01, 0.04, 0.02]
    losing = [-0.03, -0.02, -0.01, 0.005, -0.04, -0.02]
    low_ror = compute_risk_of_ruin(winning, initial_capital=100_000, risk_per_trade=1_000)
    high_ror = compute_risk_of_ruin(losing, initial_capital=100_000, risk_per_trade=1_000)
    assert 0.0 <= low_ror <= 1.0
    assert 0.0 <= high_ror <= 1.0
    assert high_ror > low_ror


def test_bootstrap_equity_drawdowns_shapes() -> None:
    returns = [0.01, -0.005, 0.015, -0.002, 0.008, -0.01]
    maxdds, pnl, summary = bootstrap_equity_drawdowns(returns, n_runs=128, seed=42)
    assert len(maxdds) == 128
    assert len(pnl) == 128
    assert 0.0 <= summary.maxdd_p95 <= 1.0
    assert summary.pnl_p05 <= summary.pnl_p50 <= summary.pnl_p95
