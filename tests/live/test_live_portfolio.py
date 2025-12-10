from __future__ import annotations

from pathlib import Path

import pytest

from hyprl.live.portfolio import PortfolioRunStats, build_portfolio_summary


class DummyBroker:
    def __init__(self, balance: float, open_risk: dict[str, float]) -> None:
        self._balance = balance
        self._open_risk = {k.upper(): float(v) for k, v in open_risk.items()}

    def get_balance(self) -> float:
        return self._balance

    def get_open_risk_amounts(self) -> dict[str, float]:
        return dict(self._open_risk)


def test_build_portfolio_summary(tmp_path: Path) -> None:
    broker = DummyBroker(balance=10_000.0, open_risk={"NVDA": 300.0, "META": 200.0})
    stats = PortfolioRunStats()
    stats.record_blocked(symbol="AMD", reason="total_risk_cap", risk_amount=150.0)
    stats.record_scaled(symbol="NVDA", scale=0.5, risk_before=200.0, risk_after=100.0)
    stats.record_executed(
        symbol="META",
        expected_pnl=50.0,
        probability_up=0.65,
        direction="long",
        threshold=0.55,
    )

    summary_path = tmp_path / "2025-01-01" / "summary.json"
    payload = build_portfolio_summary(
        broker=broker,
        tickers=["NVDA", "META", "AMD"],
        stats=stats,
        summary_path=summary_path,
    )

    assert summary_path.exists()
    assert payload["risk_used_pct_total"] == 0.05
    assert payload["counts"]["blocked"] == 1
    assert payload["counts"]["scaled"] == 1
    assert payload["counts"]["executed"] == 1
    assert payload["per_ticker_skew"]["META"] == pytest.approx(0.10)
