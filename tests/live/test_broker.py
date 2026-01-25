from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.types import Bar, TradeSignal


def _make_bar(price: float) -> Bar:
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000,
    )


def _make_signal(bar: Bar, **overrides) -> TradeSignal:
    payload = {
        "symbol": bar.symbol,
        "side": "long",
        "size": 1.0,
        "reason": "strategy:long:0.60",
        "timestamp": bar.timestamp,
        "probability_up": 0.7,
        "threshold": 0.6,
        "entry_price": bar.close,
        "expected_pnl": 5.0,
        "risk_amount": 2.5,
        "long_threshold": 0.6,
        "short_threshold": 0.4,
        "stop_price": bar.close - 2.0,
        "take_profit_price": bar.close + 2.0,
        "trailing_stop_activation_price": None,
        "trailing_stop_distance_price": None,
        "risk_profile": "normal",
        "regime_name": "normal",
    }
    payload.update(overrides)
    return TradeSignal(**payload)


def test_paper_broker_logs_trade_csv(tmp_path) -> None:
    trade_path = tmp_path / "live_trades.csv"
    broker = PaperBrokerImpl(
        cash=10_000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
        trade_log_path=trade_path,
    )
    entry_bar = _make_bar(100.0)
    signal = _make_signal(entry_bar)
    broker.submit_signal(signal, entry_bar)

    exit_bar = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 11, 0, 0),
        open=102.0,
        high=102.0,
        low=102.0,
        close=102.0,
        volume=1_000,
    )
    broker.close_position(symbol="AAPL", reason="manual", bar=exit_bar)

    assert trade_path.exists()
    df = pd.read_csv(trade_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["entry_price"] == 100.0
    assert row["exit_price"] == 102.0
    assert row["probability_up"] == 0.7
    assert row["threshold"] == 0.6
    assert row["position_size"] == 1.0
    assert row["risk_amount"] == 2.5
    assert row["expected_pnl"] == 5.0
    assert row["pnl"] == 2.0
    assert row["exit_reason"] == "manual"
    assert row["risk_profile"] == "normal"
    assert row["effective_long_threshold"] == 0.6
    assert row["effective_short_threshold"] == 0.4
    assert row["stop_price"] == 98.0
    assert row["take_profit_price"] == 102.0


def test_auto_exit_hits_take_profit() -> None:
    broker = PaperBrokerImpl(cash=10_000.0, commission_pct=0.0, slippage_pct=0.0)
    entry_bar = _make_bar(100.0)
    signal = _make_signal(entry_bar, take_profit_price=101.0)
    broker.submit_signal(signal, entry_bar)

    exit_bar = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 10, 5, 0),
        open=100.2,
        high=101.2,
        low=99.8,
        close=100.5,
        volume=1_000,
    )
    broker.mark_to_market(exit_bar)

    assert "AAPL" not in broker.positions
    trade = broker.trades[-1]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == 101.0


def test_auto_exit_trailing_stop() -> None:
    broker = PaperBrokerImpl(cash=10_000.0, commission_pct=0.0, slippage_pct=0.0)
    entry_bar = _make_bar(100.0)
    signal = _make_signal(
        entry_bar,
        stop_price=98.0,
        take_profit_price=105.0,
        trailing_stop_activation_price=101.0,
        trailing_stop_distance_price=0.5,
    )
    broker.submit_signal(signal, entry_bar)

    activation_bar = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 10, 5, 0),
        open=100.5,
        high=102.0,
        low=100.4,
        close=101.8,
        volume=1_000,
    )
    broker.mark_to_market(activation_bar)

    exit_bar = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 10, 10, 0),
        open=101.7,
        high=101.9,
        low=101.3,
        close=101.4,
        volume=1_000,
    )
    broker.mark_to_market(exit_bar)

    assert "AAPL" not in broker.positions
    trade = broker.trades[-1]
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_price == pytest.approx(101.5)
