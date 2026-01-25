from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pandas as pd
import pytest

from hyprl.backtest.runner import StrategyStats, SupercalcDataset
from hyprl.search import optimizer as search_optimizer
from hyprl.search.optimizer import SearchConfig, run_search, save_results_csv


def _make_stats(final_balance: float, seed: int) -> StrategyStats:
    timestamps = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=seed),
        pd.Timestamp("2024-02-01") + pd.Timedelta(days=seed),
    ]
    equity_history = list(zip(timestamps, [10_000.0, final_balance]))
    return StrategyStats(
        final_balance=final_balance,
        profit_factor=1.5,
        sharpe_ratio=1.1,
        max_drawdown_pct=10.0,
        expectancy=150.0,
        n_trades=40,
        win_rate=0.55,
        trades_in_fear=3,
        trades_in_greed=2,
        trade_returns=[0.02, -0.01] * 20,
        equity_history=equity_history,
    )


@pytest.fixture(autouse=True)
def stub_supercalc(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = SupercalcDataset(
        rows=[],
        prices=pd.DataFrame({"close": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2, freq="D")),
        benchmark_return_pct=5.0,
        initial_balance=10_000.0,
    )

    def _fake_prepare(config):  # noqa: ANN001
        return dataset

    def _fake_evaluate(dataset, configs, engine, require_trade_returns):  # noqa: ANN001
        stats = []
        for cfg in configs:
            if cfg.ticker == "AAA":
                stats.append(_make_stats(11_000.0, 1))
            else:
                stats.append(_make_stats(10_500.0, 2))
        return stats

    monkeypatch.setattr(search_optimizer, "prepare_supercalc_dataset", _fake_prepare)
    monkeypatch.setattr(search_optimizer, "evaluate_candidates", _fake_evaluate)
    monkeypatch.setattr(
        search_optimizer,
        "_passes_hard_constraints",
        lambda cfg, result, tracker=None, **_: True,
    )


def test_run_search_portfolio_outputs_metrics() -> None:
    config = SearchConfig(
        ticker="AAA",
        tickers=["AAA", "BBB"],
        period="90d",
        interval="1h",
        initial_balance=20_000.0,
        seed=7,
        use_presets=False,
        long_thresholds=[0.55],
        short_thresholds=[0.45],
        risk_pcts=[0.02],
        min_trades=0,
        min_profit_factor=0.0,
        min_sharpe=-10.0,
        max_drawdown_pct=10.0,
        max_risk_of_ruin=1.0,
        min_expectancy=-10.0,
        bootstrap_runs=16,
        min_portfolio_profit_factor=0.0,
        min_portfolio_sharpe=-10.0,
        max_portfolio_drawdown_pct=10.0,
        max_portfolio_risk_of_ruin=1.0,
        max_correlation=1.0,
    )
    results = run_search(config)
    assert results, "portfolio search should yield results"
    res = results[0]
    assert set(res.per_ticker_details.keys()) == {"AAA", "BBB"}
    assert res.portfolio_profit_factor >= 1.0
    assert 0.0 <= res.portfolio_risk_of_ruin <= 1.0


def test_save_results_csv_embeds_portfolio_weights(tmp_path: Path) -> None:
    config = SearchConfig(
        ticker="AAA",
        tickers=["AAA", "BBB"],
        period="90d",
        interval="1h",
        initial_balance=20_000.0,
        seed=7,
        use_presets=False,
        long_thresholds=[0.55],
        short_thresholds=[0.45],
        risk_pcts=[0.02],
        min_trades=0,
        min_profit_factor=0.0,
        min_sharpe=-10.0,
        max_drawdown_pct=10.0,
        max_risk_of_ruin=1.0,
        min_expectancy=-10.0,
        bootstrap_runs=16,
        min_portfolio_profit_factor=0.0,
        min_portfolio_sharpe=-10.0,
        max_portfolio_drawdown_pct=10.0,
        max_portfolio_risk_of_ruin=1.0,
        max_correlation=1.0,
        weighting_scheme="inv_vol",
    )
    results = run_search(config)
    assert results
    output = tmp_path / "weights.csv"
    save_results_csv(results, output, metadata=None)
    df = pd.read_csv(output)
    assert "portfolio_weights" in df.columns
    weights = json.loads(df.loc[0, "portfolio_weights"])
    assert set(weights.keys()) == {"AAA", "BBB"}
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
