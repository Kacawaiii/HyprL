from __future__ import annotations

from datetime import date, datetime

from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.risk import LiveRiskConfig, LiveRiskManager
from hyprl.live.types import Position, TradeSignal


class _Clock:
    def __init__(self, current: date) -> None:
        self._current = current

    def set(self, current: date) -> None:
        self._current = current

    def __call__(self) -> date:
        return self._current


def _make_signal(size: float = 1.0) -> TradeSignal:
    now = datetime(2024, 1, 1, 10, 0, 0)
    return TradeSignal(
        symbol="AAPL",
        side="long",
        size=size,
        reason="strategy:long:0.60",
        timestamp=now,
        probability_up=0.7,
        threshold=0.6,
        entry_price=100.0,
        expected_pnl=5.0,
        risk_amount=2.5,
        long_threshold=0.6,
        short_threshold=0.4,
        stop_price=98.0,
        take_profit_price=102.0,
        trailing_stop_activation_price=None,
        trailing_stop_distance_price=None,
        risk_profile="normal",
        regime_name="normal",
    )


def test_risk_manager_blocks_notional_limit() -> None:
    broker = PaperBrokerImpl(cash=1_000.0)
    manager = LiveRiskManager(
        LiveRiskConfig(max_daily_loss_pct=0.5, max_position_notional_pct=0.05, max_gross_exposure_pct=1.0),
        broker=broker,
    )
    signal = _make_signal(size=1.0)
    decision = manager.allow_trade(signal, price=100.0)
    assert not decision.allowed
    assert decision.reason == "position_notional_exceeded"


def test_risk_manager_engages_daily_loss_kill_switch() -> None:
    broker = PaperBrokerImpl(cash=1_000.0)
    manager = LiveRiskManager(
        LiveRiskConfig(max_daily_loss_pct=0.05, max_position_notional_pct=1.0, max_gross_exposure_pct=5.0),
        broker=broker,
    )
    signal = _make_signal(size=0.1)
    first = manager.allow_trade(signal, price=10.0)
    assert first.allowed

    broker.cash = 900.0  # Simulate 10% loss relative to start of day
    blocked = manager.allow_trade(signal, price=10.0)
    assert not blocked.allowed
    assert blocked.reason == "daily_loss_kill_switch"

    broker.cash = 2_000.0  # Even if equity recovers intra-day, kill switch stays latched
    still_blocked = manager.allow_trade(signal, price=10.0)
    assert not still_blocked.allowed
    assert still_blocked.reason == "daily_loss_kill_switch"


def test_risk_manager_resets_with_custom_clock() -> None:
    broker = PaperBrokerImpl(cash=1_000.0)
    clock = _Clock(date(2024, 1, 1))
    manager = LiveRiskManager(
        LiveRiskConfig(max_daily_loss_pct=0.05, max_position_notional_pct=1.0, max_gross_exposure_pct=5.0),
        broker=broker,
        clock=clock,
    )
    signal = _make_signal(size=0.1)
    first = manager.allow_trade(signal, price=10.0)
    assert first.allowed

    broker.cash = 400.0
    blocked = manager.allow_trade(signal, price=10.0)
    assert not blocked.allowed
    assert blocked.reason == "daily_loss_kill_switch"

    clock.set(date(2024, 1, 2))
    broker.cash = 1_000.0
    reset = manager.allow_trade(signal, price=10.0)
    assert reset.allowed


def test_risk_manager_parity_mode_bypasses_notional_and_gross() -> None:
    broker = PaperBrokerImpl(cash=1_000.0)
    config = LiveRiskConfig(max_daily_loss_pct=0.5, max_position_notional_pct=0.05, max_gross_exposure_pct=0.2)
    manager = LiveRiskManager(config, broker=broker, parity_mode=True)

    decision = manager.allow_trade(_make_signal(size=1.0), price=100.0)
    assert decision.allowed
    assert decision.bypassed is True
    assert decision.reason == "position_notional_exceeded"

    # Raise the notional cap so we can trip the gross exposure guard instead
    config.max_position_notional_pct = 1.0
    broker.positions["MSFT"] = Position(symbol="MSFT", side="long", size=3.0, avg_price=100.0)
    gross_decision = manager.allow_trade(_make_signal(size=1.0), price=100.0)
    assert gross_decision.allowed
    assert gross_decision.bypassed is True
    assert gross_decision.reason == "gross_exposure_exceeded"

