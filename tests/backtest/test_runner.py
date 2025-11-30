from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 110, rows)
    noise = np.sin(np.linspace(0, 8 * np.pi, rows)) * 0.8
    close = base + noise
    open_ = close - 0.1
    high = close + 0.6
    low = close - 0.6
    volume = np.linspace(1_000, 2_000, rows)
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }
    return pd.DataFrame(data, index=index)


def test_run_backtest_generates_trades(monkeypatch):
    price_df = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):
        return price_df.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=50_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=50_000.0, risk_pct=0.05, atr_multiplier=1.0, reward_multiple=1.5, min_position_size=10),
    )

    result = runner.run_backtest(cfg)

    assert result.n_trades > 0
    assert len(result.equity_curve) > 0
    assert result.final_balance > 0
    assert 0.0 <= result.win_rate <= 1.0


def test_run_backtest_supports_explicit_dates(monkeypatch):
    price_df = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):
        return price_df.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period=None,
        start="2024-01-01",
        end="2024-05-01",
        initial_balance=25_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=25_000.0, risk_pct=0.02, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=5),
    )

    result = runner.run_backtest(cfg)

    assert result.final_balance >= 0
    assert len(result.equity_curve) >= 1
    assert result.max_drawdown >= 0
    assert result.sharpe_ratio is None or isinstance(result.sharpe_ratio, float)


def test_backtest_is_deterministic_with_seed(monkeypatch):
    price_df = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):
        return price_df.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=15_000.0,
        long_threshold=0.65,
        short_threshold=0.35,
        random_state=123,
        risk=RiskConfig(balance=15_000.0, risk_pct=0.03, atr_multiplier=1.0, reward_multiple=1.2, min_position_size=5),
    )

    result1 = runner.run_backtest(cfg)
    result2 = runner.run_backtest(cfg)

    assert result1.final_balance == result2.final_balance
    assert result1.equity_curve == result2.equity_curve
    assert result1.n_trades == result2.n_trades
