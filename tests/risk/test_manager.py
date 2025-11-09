from __future__ import annotations

import pytest

from hyprl.risk.manager import RiskConfig, plan_trade


def test_plan_trade_long_direction():
    config = RiskConfig(
        balance=10_000.0,
        risk_pct=0.01,
        atr_multiplier=1.5,
        reward_multiple=2.5,
        min_position_size=1,
    )
    result = plan_trade(entry_price=100.0, atr=2.0, direction="long", config=config)
    assert result.position_size == 33
    assert pytest.approx(result.stop_price, rel=1e-3) == 97.0
    assert pytest.approx(result.take_profit_price, rel=1e-3) == 107.5
    assert pytest.approx(result.risk_amount, rel=1e-3) == 99.0
    assert result.rr_multiple == pytest.approx(config.reward_multiple)


def test_plan_trade_short_direction():
    config = RiskConfig(
        balance=50_000.0,
        risk_pct=0.02,
        atr_multiplier=1.0,
        reward_multiple=2.0,
        min_position_size=10,
    )
    result = plan_trade(entry_price=250.0, atr=5.0, direction="short", config=config)
    assert result.position_size == 200
    assert pytest.approx(result.stop_price, rel=1e-3) == 255.0
    assert pytest.approx(result.take_profit_price, rel=1e-3) == 240.0
    assert pytest.approx(result.risk_amount, rel=1e-3) == 1_000.0
    assert result.rr_multiple == pytest.approx(config.reward_multiple)


def test_plan_trade_handles_zero_atr_and_invalid_direction():
    config = RiskConfig()
    zero_atr = plan_trade(entry_price=150.0, atr=0.0, direction="long", config=config)
    assert zero_atr.position_size == 0
    assert zero_atr.risk_amount == 0.0
    assert zero_atr.stop_price == zero_atr.entry_price
    with pytest.raises(ValueError):
        plan_trade(entry_price=150.0, atr=1.0, direction="flat", config=config)


def test_position_size_returns_zero_when_budget_too_small():
    config = RiskConfig(balance=1_000.0, risk_pct=0.0001, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=1)
    result = plan_trade(entry_price=200.0, atr=5.0, direction="long", config=config)
    assert result.position_size == 0
    assert result.risk_amount == 0.0
