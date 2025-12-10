from __future__ import annotations

from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.risk import LiveRiskConfig, LiveRiskManager
from hyprl.live.types import TradeSignal


def make_signal(size: float, price: float = 10.0) -> TradeSignal:
    return TradeSignal(
        symbol="NVDA",
        side="long",
        size=size,
        reason="test",
        timestamp=None,
        probability_up=0.6,
        threshold=0.55,
        entry_price=price,
        expected_pnl=0.0,
        risk_amount=0.0,
        long_threshold=0.55,
        short_threshold=0.40,
        stop_price=price * 0.9,
        take_profit_price=price * 1.1,
        trailing_stop_activation_price=None,
        trailing_stop_distance_price=None,
        risk_profile=None,
        regime_name=None,
    )


def test_live_risk_allows_when_clamp_compliant() -> None:
    broker = PaperBrokerImpl(cash=10_000.0)
    cfg = LiveRiskConfig(max_position_notional_pct=0.2, max_notional_per_trade=3_000.0)
    manager = LiveRiskManager(cfg, broker, parity_mode=False)
    # Notional = 1_000, within both pct (20% of 10k) and absolute (3k) caps
    signal = make_signal(size=100.0, price=10.0)
    decision = manager.allow_trade(signal, price=10.0)
    assert decision.allowed is True
    assert decision.reason is None


def test_live_risk_blocks_when_caps_exceeded() -> None:
    broker = PaperBrokerImpl(cash=10_000.0)
    cfg = LiveRiskConfig(max_position_notional_pct=0.1, max_notional_per_trade=500.0)
    manager = LiveRiskManager(cfg, broker, parity_mode=False)
    # Notional = 1_000 exceeds both 10% of equity (1k cap) and 500 absolute cap -> block
    signal = make_signal(size=100.0, price=10.0)
    decision = manager.allow_trade(signal, price=10.0)
    assert decision.allowed is False
    assert decision.reason == "position_notional_exceeded"


def test_live_risk_parity_mode_skips_caps() -> None:
    broker = PaperBrokerImpl(cash=10_000.0)
    cfg = LiveRiskConfig(max_position_notional_pct=0.05, max_notional_per_trade=100.0)
    manager = LiveRiskManager(cfg, broker, parity_mode=True)
    # Would exceed both caps, but parity_mode should allow (bypassed caps are skipped in parity).
    signal = make_signal(size=500.0, price=10.0)  # notional=5_000
    decision = manager.allow_trade(signal, price=10.0)
    assert decision.allowed is True
    assert decision.reason is None
