import math

import pytest

from hyprl.risk.manager import RiskConfig, plan_trade
from hyprl.strategy.core import expected_trade_pnl


def test_plan_trade_expected_value_alignment() -> None:
    config = RiskConfig(balance=10_000.0, risk_pct=0.02, atr_multiplier=2.0, reward_multiple=2.0)
    outcome = plan_trade(entry_price=500.0, atr=4.0, direction="long", config=config)
    assert math.isclose(outcome.position_size, 25.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(outcome.risk_amount, 200.0, rel_tol=0.0, abs_tol=1e-6)
    ev = expected_trade_pnl(outcome, probability_up=0.6)
    assert ev == pytest.approx(160.0, abs=1e-6)


def test_plan_trade_zero_when_min_size_not_met() -> None:
    config = RiskConfig(balance=1_000.0, risk_pct=0.005, atr_multiplier=3.0, reward_multiple=1.5, min_position_size=1.0)
    outcome = plan_trade(entry_price=100.0, atr=20.0, direction="short", config=config)
    assert outcome.position_size == 0.0
    assert outcome.risk_amount == 0.0
    ev = expected_trade_pnl(outcome, probability_up=0.4)
    assert ev == 0.0
