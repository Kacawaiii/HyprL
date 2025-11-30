from __future__ import annotations

from datetime import timedelta

import math
import pandas as pd

from hyprl.backtest.runner import TradeRecord, summarize_oos_metrics


def _trade(ts: pd.Timestamp, pnl: float, equity_after: float) -> TradeRecord:
    return TradeRecord(
        entry_timestamp=ts - timedelta(hours=1),
        exit_timestamp=ts,
        direction="long",
        probability_up=0.6,
        threshold=0.55,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        position_size=1,
        pnl=pnl,
        return_pct=pnl / 1000.0,
        equity_after=equity_after,
        risk_amount=abs(pnl) if pnl != 0 else 1.0,
        expected_pnl=pnl * 0.5,
        risk_profile="normal",
        effective_long_threshold=0.55,
        effective_short_threshold=0.45,
        regime_name="normal",
    )


def test_summarize_oos_metrics_contract() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="H")
    equity = pd.Series([10_000, 10_200, 10_000, 10_400, 10_300], index=idx)
    trades = [
        _trade(idx[1], 200.0, 10_200),
        _trade(idx[2], -200.0, 10_000),
        _trade(idx[3], 400.0, 10_400),
        _trade(idx[4], -100.0, 10_300),
    ]
    metrics = summarize_oos_metrics(equity, trades)
    expected_keys = {"pf", "sharpe", "calmar", "maxdd", "cvar95_m", "dsr", "pboc"}
    assert expected_keys == set(metrics.keys())
    assert 0.0 <= metrics["maxdd"] <= 1.0
    assert metrics["pf"] >= 0.0
    assert math.isnan(metrics["pboc"]) or 0.0 <= metrics["pboc"] <= 1.0
