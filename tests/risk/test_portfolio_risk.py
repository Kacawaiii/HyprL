from __future__ import annotations

import pytest

from hyprl.risk.portfolio import PortfolioRiskLimits, PortfolioRiskManager


class FakePosition:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()


class FakeBroker:
    def __init__(self, open_risk: dict[str, float] | None = None, positions: list[FakePosition] | None = None) -> None:
        self._open_risk = open_risk or {}
        self._positions = positions or []

    def get_open_risk_amounts(self) -> dict[str, float]:
        return dict(self._open_risk)

    def get_positions(self) -> list[FakePosition]:
        return list(self._positions)


def test_allows_within_caps() -> None:
    limits = PortfolioRiskLimits(max_total_risk_pct=0.05, max_ticker_risk_pct=0.03, max_group_risk_pct=0.04)
    manager = PortfolioRiskManager(limits, {"NVDA": "semis"})
    broker = FakeBroker()

    decision = manager.evaluate(
        "NVDA",
        proposed_position_size=100.0,
        risk_amount=200.0,
        equity=10_000.0,
        broker=broker,
        min_position_size=1.0,
    )

    assert decision.allowed is True
    assert decision.adjusted is False
    assert decision.position_size == pytest.approx(100.0)
    assert decision.risk_amount == pytest.approx(200.0)


def test_scales_to_group_cap() -> None:
    limits = PortfolioRiskLimits(max_total_risk_pct=0.05, max_ticker_risk_pct=0.03, max_group_risk_pct=0.03)
    manager = PortfolioRiskManager(limits, {"NVDA": "semis", "AMD": "semis"})
    broker = FakeBroker(open_risk={"AMD": 250.0})

    decision = manager.evaluate(
        "NVDA",
        proposed_position_size=100.0,
        risk_amount=200.0,
        equity=10_000.0,
        broker=broker,
        min_position_size=1.0,
    )

    assert decision.allowed is True
    assert decision.adjusted is True
    assert decision.risk_amount == pytest.approx(50.0)  # 3% group cap minus 250 already used
    assert decision.position_size == pytest.approx(25.0)


def test_rejects_on_total_cap() -> None:
    limits = PortfolioRiskLimits(max_total_risk_pct=0.05, max_ticker_risk_pct=0.03, max_group_risk_pct=0.03)
    manager = PortfolioRiskManager(limits, {"NVDA": "semis"})
    broker = FakeBroker(open_risk={"MSFT": 500.0})

    decision = manager.evaluate(
        "NVDA",
        proposed_position_size=10.0,
        risk_amount=10.0,
        equity=10_000.0,
        broker=broker,
        min_position_size=1.0,
    )

    assert decision.allowed is False
    assert decision.reason == "total_risk_cap"


def test_respects_max_positions() -> None:
    limits = PortfolioRiskLimits(max_total_risk_pct=0.05, max_ticker_risk_pct=0.03, max_group_risk_pct=0.03, max_positions=1)
    manager = PortfolioRiskManager(limits, {"NVDA": "semis"})
    broker = FakeBroker(open_risk={"MSFT": 100.0}, positions=[FakePosition("MSFT")])

    decision = manager.evaluate(
        "NVDA",
        proposed_position_size=10.0,
        risk_amount=10.0,
        equity=10_000.0,
        broker=broker,
        min_position_size=1.0,
    )

    assert decision.allowed is False
    assert decision.reason == "max_positions"
