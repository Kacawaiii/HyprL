from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import pytest

from hyprl.adaptive.engine import AdaptiveConfig, AdaptiveRegime, AdaptiveState, evaluate_window, update_state
from hyprl.backtest.runner import TradeRecord


def _make_trade(pnl: float, equity_after: float, return_pct: float) -> TradeRecord:
    timestamp = pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
    return TradeRecord(
        entry_timestamp=timestamp,
        exit_timestamp=timestamp,
        direction="long",
        probability_up=0.7,
        threshold=0.6,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        position_size=1,
        pnl=pnl,
        return_pct=return_pct,
        equity_after=equity_after,
        risk_amount=50.0,
        expected_pnl=pnl,
        risk_profile="normal",
        effective_long_threshold=0.6,
        effective_short_threshold=0.4,
        regime_name="normal",
    )


def test_evaluate_window_reports_metrics():
    trades = [
        _make_trade(120.0, 10_120.0, 0.012),
        _make_trade(-60.0, 10_060.0, -0.006),
        _make_trade(80.0, 10_140.0, 0.008),
    ]

    metrics = evaluate_window(trades)

    assert pytest.approx(metrics["win_rate"], rel=1e-3) == 2 / 3
    assert metrics["profit_factor"] > 1.0
    assert metrics["drawdown_pct"] >= 0.0
    assert "sharpe" in metrics


def test_update_state_switches_regime_on_drawdown():
    config = AdaptiveConfig(
        enable=True,
        lookback_trades=3,
        default_regime="normal",
        regimes={
            "normal": AdaptiveRegime(
                name="normal",
                min_equity_drawdown=-0.01,
                max_equity_drawdown=0.08,
                min_profit_factor=0.8,
                min_sharpe=-0.5,
            ),
            "safe": AdaptiveRegime(
                name="safe",
                min_equity_drawdown=0.08,
                max_equity_drawdown=1.0,
                min_profit_factor=0.0,
                min_sharpe=-1.0,
            ),
        },
    )
    state = AdaptiveState(regime_name="normal")
    poor_metrics = {
        "drawdown_pct": 0.2,
        "profit_factor": 0.5,
        "expectancy": -20.0,
        "sharpe": -1.2,
        "win_rate": 0.2,
    }

    updated = update_state(config, state, poor_metrics, trade_index=4)

    assert updated.regime_name == "safe"
    assert updated.regime_changes == 1

    recovery_metrics = {
        "drawdown_pct": 0.01,
        "profit_factor": 1.4,
        "expectancy": 15.0,
        "sharpe": 1.0,
        "win_rate": 0.7,
    }

    updated = update_state(config, updated, recovery_metrics, trade_index=8)

    assert updated.regime_name == "normal"
