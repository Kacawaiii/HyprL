from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 110, rows)
    noise = np.sin(np.linspace(0, 8 * np.pi, rows)) * 0.8
    close = base + noise
    data = {
        "open": close - 0.1,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "adj_close": close,
        "volume": np.linspace(1_000, 2_000, rows),
    }
    return pd.DataFrame(data, index=index)


def _mock_prices(monkeypatch, frame: pd.DataFrame) -> None:
    def _fake_get_prices(self, **kwargs):
        return frame.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)


def _base_config() -> runner.BacktestConfig:
    return runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=50_000.0,
        long_threshold=0.65,
        short_threshold=0.35,
        random_state=321,
        risk=RiskConfig(
            balance=50_000.0,
            risk_pct=0.05,
            atr_multiplier=1.0,
            reward_multiple=1.5,
            min_position_size=10,
        ),
    )


def test_transaction_costs_reduce_pnl(monkeypatch):
    prices = _synthetic_price_frame()
    _mock_prices(monkeypatch, prices)

    cfg_no_costs = replace(_base_config(), commission_pct=0.0, slippage_pct=0.0)
    cfg_costs = replace(_base_config(), commission_pct=0.001, slippage_pct=0.001)

    result_no_costs = runner.run_backtest(cfg_no_costs)
    result_costs = runner.run_backtest(cfg_costs)

    assert result_no_costs.n_trades == result_costs.n_trades >= 1
    assert result_costs.final_balance < result_no_costs.final_balance


def test_benchmark_positive_on_rising_prices(monkeypatch):
    prices = _synthetic_price_frame()
    _mock_prices(monkeypatch, prices)

    result = runner.run_backtest(_base_config())

    assert result.benchmark_final_balance > _base_config().initial_balance
    assert result.benchmark_return > 0
