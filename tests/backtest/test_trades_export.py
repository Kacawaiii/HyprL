from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    trend = np.linspace(100, 115, rows)
    oscillation = np.sin(np.linspace(0, 10 * np.pi, rows)) * 1.5
    close = trend + oscillation
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "adj_close": close,
            "volume": np.linspace(1_000, 1_800, rows),
        },
        index=index,
    )


def _mock_prices(monkeypatch, frame: pd.DataFrame) -> None:
    def _fake_get_prices(self, **kwargs):
        return frame.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)


def _base_config() -> runner.BacktestConfig:
    return runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=20_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        random_state=7,
        risk=RiskConfig(
            balance=20_000.0,
            risk_pct=0.03,
            atr_multiplier=1.0,
            reward_multiple=1.5,
            min_position_size=5,
        ),
    )


def test_run_backtest_emits_trade_records(monkeypatch):
    price_df = _synthetic_price_frame()
    _mock_prices(monkeypatch, price_df)

    result = runner.run_backtest(_base_config())

    assert result.trades, "Expected at least one trade record."
    for trade in result.trades:
        assert trade.direction in {"long", "short"}
        assert pd.notna(trade.entry_timestamp)
        assert pd.notna(trade.exit_timestamp)
        assert np.isfinite(trade.entry_price)
        assert np.isfinite(trade.exit_price)
        assert np.isfinite(trade.pnl)
        assert np.isfinite(trade.equity_after)


def test_trade_records_can_be_exported_to_csv(tmp_path: Path, monkeypatch):
    price_df = _synthetic_price_frame()
    _mock_prices(monkeypatch, price_df)
    result = runner.run_backtest(_base_config())
    export_path = tmp_path / "trades.csv"

    records = [asdict(trade) for trade in result.trades]
    df = pd.DataFrame.from_records(records)
    df.to_csv(export_path, index=False)

    assert export_path.exists()
    exported = pd.read_csv(export_path)
    assert len(exported) == len(result.trades)
