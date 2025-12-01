from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from hyprl.backtest.runner import BacktestConfig, BacktestResult, TradeRecord
from hyprl.risk.manager import RiskConfig
from hyprl.snapshots import save_snapshot, make_backup_zip


def _dummy_result() -> BacktestResult:
    trade = TradeRecord(
        entry_timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        exit_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
        direction="long",
        probability_up=0.6,
        threshold=0.55,
        entry_price=100.0,
        exit_price=102.0,
        exit_reason="take_profit",
        position_size=10,
        pnl=20.0,
        return_pct=0.002,
        equity_after=10020.0,
        risk_amount=200.0,
        expected_pnl=5.0,
        risk_profile="normal",
        effective_long_threshold=0.6,
        effective_short_threshold=0.4,
        regime_name="normal",
    )
    return BacktestResult(
        final_balance=10020.0,
        equity_curve=[10000.0, 10020.0],
        n_trades=1,
        win_rate=1.0,
        max_drawdown=0.0,
        sharpe_ratio=1.0,
        trades=[trade],
        benchmark_final_balance=10100.0,
        benchmark_return=1.0,
        annualized_return=None,
        annualized_benchmark_return=None,
        annualized_volatility=None,
        sortino_ratio=None,
        profit_factor=None,
        expectancy=20.0,
        avg_r_multiple=0.1,
        avg_expected_pnl=5.0,
        long_trades=1,
        short_trades=0,
        long_win_rate=1.0,
        short_win_rate=0.0,
        long_total_pnl=20.0,
        short_total_pnl=0.0,
        brier_score=0.2,
        log_loss=0.5,
        final_risk_profile="normal",
        final_long_threshold=0.6,
        final_short_threshold=0.4,
        adaptive_profile_changes=1,
        regime_usage={"normal": 1},
        regime_transitions=[{"trade": 1, "regime": "safe"}],
    )


def test_snapshot_creation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BacktestConfig(
        ticker="TEST",
        period="5d",
        initial_balance=10_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(),
    )
    result = _dummy_result()
    snapshot_dir = save_snapshot(config, result, trades_path=None)
    assert snapshot_dir.exists()
    config_json = snapshot_dir / "config.json"
    result_json = snapshot_dir / "result.json"
    assert config_json.exists()
    assert result_json.exists()
    loaded = json.loads(result_json.read_text())
    assert loaded["n_trades"] == 1
    assert "trades_summary" in loaded

    zip_path = make_backup_zip(snapshot_dir)
    assert zip_path.exists()
