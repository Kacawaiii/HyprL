from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest.runner import BacktestConfig, sweep_thresholds
from hyprl.risk.manager import RiskConfig
from hyprl.backtest import runner


def _synthetic_price_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 110, rows)
    noise = np.sin(np.linspace(0, 4 * np.pi, rows)) * 0.5
    close = base + noise
    data = {
        "open": close - 0.1,
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "adj_close": close,
        "volume": np.linspace(1_000, 1_500, rows),
    }
    return pd.DataFrame(data, index=index)


def test_sweep_thresholds_returns_ordered_summaries(monkeypatch):
    price_df = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):
        return price_df.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)
    base_cfg = BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=20_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        random_state=321,
        risk=RiskConfig(balance=20_000.0, risk_pct=0.03, atr_multiplier=1.0, reward_multiple=1.4, min_position_size=5),
    )
    thresholds = [0.55, 0.6, 0.65]

    summaries = sweep_thresholds(base_cfg, thresholds)

    assert len(summaries) == len(thresholds)
    assert [s.threshold for s in summaries] == thresholds
    assert all(np.isfinite(summary.total_return) for summary in summaries)
    for summary in summaries:
        assert np.isfinite(summary.benchmark_return)
        assert np.isfinite(summary.alpha_return)
        assert summary.alpha_return == pytest.approx(summary.total_return - summary.benchmark_return)
        assert summary.expectancy == pytest.approx(summary.expectancy)
    final_balances = {summary.final_balance for summary in summaries}
    assert len(final_balances) >= 1
