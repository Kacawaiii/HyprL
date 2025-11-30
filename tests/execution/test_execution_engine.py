from __future__ import annotations

import math

import pandas as pd

from hyprl.backtest.runner import TradeRecord
from hyprl.execution.broker import PaperBroker
from hyprl.execution.engine import replay_trades
from hyprl.execution.logging import LiveLogger


def _make_trade(direction: str, entry_price: float, exit_price: float, qty: int) -> TradeRecord:
    return TradeRecord(
        entry_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        exit_timestamp=pd.Timestamp("2024-01-02T00:00:00Z"),
        direction=direction,
        probability_up=0.6,
        threshold=0.55,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=qty,
        pnl=(exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty,
        return_pct=0.0,
        equity_after=0.0,
        risk_amount=0.0,
        expected_pnl=0.0,
        risk_profile=None,
        effective_long_threshold=None,
        effective_short_threshold=None,
        regime_name=None,
    )


def test_replay_trades_updates_equity(tmp_path):
    trades = {"AAA": [_make_trade("long", 100.0, 110.0, 5)]}
    broker = PaperBroker(initial_balance=10_000.0)
    logger = LiveLogger("test_session", base_dir=tmp_path)
    equity_series = replay_trades(trades, broker, logger)
    assert not equity_series.empty
    assert math.isclose(equity_series.iloc[-1], 10_000.0 + 50.0)
    assert (tmp_path / "test_session" / "trades.csv").exists()
    assert (tmp_path / "test_session" / "equity.csv").exists()
