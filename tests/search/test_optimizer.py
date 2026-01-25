from __future__ import annotations

from pathlib import Path
from typing import Callable

import math
import pandas as pd
import pytest
import numpy as np

from hyprl.backtest import runner as backtest_runner
import importlib.util

from hyprl.search import optimizer as search_optimizer
from hyprl.search.optimizer import SearchConfig, run_search, save_results_csv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_SUPERSEARCH_SPEC = importlib.util.spec_from_file_location(
    "run_supersearch",
    _REPO_ROOT / "scripts" / "run_supersearch.py",
)
if _RUN_SUPERSEARCH_SPEC is None or _RUN_SUPERSEARCH_SPEC.loader is None:
    raise RuntimeError("Unable to load run_supersearch.py spec")
run_supersearch = importlib.util.module_from_spec(_RUN_SUPERSEARCH_SPEC)
_RUN_SUPERSEARCH_SPEC.loader.exec_module(run_supersearch)


@pytest.fixture(autouse=True)
def stub_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = 320
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    base = pd.Series(np.sin(np.linspace(0.0, 12.0, periods)), index=index)
    trend = 100.0 + pd.Series(np.linspace(0.0, 2.0, periods), index=index)
    price = trend + base
    prices = pd.DataFrame(
        {
            "open": price + 0.05,
            "high": price + 0.75,
            "low": price - 0.75,
            "close": price + 0.2,
            "volume": 1_000_000.0,
        },
        index=index,
    )

    def _fake_get_prices(self, interval, period, start, end):  # noqa: ANN001
        return prices

    monkeypatch.setattr(backtest_runner.MarketDataFetcher, "get_prices", _fake_get_prices)


def test_run_search_sorts_by_profit_factor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_optimizer, "_passes_hard_constraints", lambda cfg, result, **_: True)
    config = SearchConfig(
        ticker="TEST",
        period="90d",
        interval="1h",
        initial_balance=50_000.0,
        seed=7,
        use_presets=False,
        long_thresholds=[0.55, 0.6],
        short_thresholds=[0.35, 0.4],
        risk_pcts=[0.01, 0.015],
        min_ev_multiples=[0.0, 0.1],
        trend_filter_flags=[False, True],
        min_trades=0,
        min_profit_factor=0.0,
        min_sharpe=-10.0,
        max_drawdown_pct=2.0,
        max_risk_of_ruin=1.0,
        min_expectancy=-10.0,
        bootstrap_runs=16,
    )
    results = run_search(config)
    assert results, "expected at least one search result"
    sort_key: Callable = search_optimizer._score_tuple
    assert results == sorted(results, key=sort_key)
    for res in results:
        assert math.isfinite(res.strategy_return_pct)
        assert math.isfinite(res.alpha_pct)
        assert math.isfinite(res.max_drawdown_pct)
        assert res.n_trades >= 0
        assert 0.0 <= res.config.short_threshold <= res.config.long_threshold <= 1.0
        assert 0.0 <= res.risk_of_ruin <= 1.0
        assert res.maxdd_p95 >= 0.0

    out_path = tmp_path / "search.csv"
    save_results_csv(results, out_path)
    df = pd.read_csv(out_path)
    assert len(df) == len(results)
    assert {"long_threshold", "risk_of_ruin", "maxdd_p95"}.issubset(df.columns)


def test_hard_constraints_logic() -> None:
    cfg = SearchConfig(ticker="TEST", period="6mo")
    candidate = search_optimizer.CandidateConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        risk_pct=0.02,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-0.5,
        sentiment_max=0.5,
        sentiment_regime="off",
    )
    good = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=12.0,
        benchmark_return_pct=5.0,
        alpha_pct=7.0,
        profit_factor=1.5,
        sharpe=1.1,
        max_drawdown_pct=12.0,
        expectancy=150.0,
        n_trades=100,
        win_rate_pct=55.0,
        sentiment_stats={"trades_in_fear": 5, "trades_in_greed": 4},
        expectancy_per_trade=0.01,
        risk_of_ruin=0.05,
        maxdd_p95=18.0,
        pnl_p05=-0.02,
    )
    assert search_optimizer._passes_hard_constraints(cfg, good)

    bad = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=5.0,
        benchmark_return_pct=4.0,
        alpha_pct=1.0,
        profit_factor=1.05,
        sharpe=0.3,
        max_drawdown_pct=60.0,
        expectancy=-10.0,
        n_trades=5,
        win_rate_pct=90.0,
        sentiment_stats={"trades_in_fear": 0, "trades_in_greed": 0},
        expectancy_per_trade=-0.02,
        risk_of_ruin=0.8,
        maxdd_p95=70.0,
        pnl_p05=-0.5,
    )
    assert not search_optimizer._passes_hard_constraints(cfg, bad)


def test_hard_constraints_sharpe_disabled_when_min_zero() -> None:
    cfg = SearchConfig(
        ticker="TEST",
        period="6mo",
        min_trades=1,
        min_sharpe=0.0,
        min_profit_factor=0.0,
        max_drawdown_pct=10.0,
    )
    candidate = search_optimizer.CandidateConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        risk_pct=0.02,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-0.5,
        sentiment_max=0.5,
        sentiment_regime="off",
    )
    sharpe_nan = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=2.0,
        benchmark_return_pct=1.0,
        alpha_pct=1.0,
        profit_factor=1.1,
        sharpe=float("nan"),
        max_drawdown_pct=5.0,
        expectancy=0.1,
        n_trades=10,
        win_rate_pct=55.0,
        expectancy_per_trade=0.01,
        risk_of_ruin=0.05,
        maxdd_p95=6.0,
        pnl_p05=-0.01,
    )
    assert search_optimizer._passes_hard_constraints(cfg, sharpe_nan)


def test_hard_constraints_robustness_threshold() -> None:
    cfg = SearchConfig(
        ticker="TEST",
        period="6mo",
        min_trades=1,
        min_sharpe=0.0,
        min_profit_factor=0.0,
        max_drawdown_pct=10.0,
        min_robustness_score=0.6,
    )
    candidate = search_optimizer.CandidateConfig(
        long_threshold=0.55,
        short_threshold=0.45,
        risk_pct=0.01,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-0.5,
        sentiment_max=0.5,
        sentiment_regime="off",
    )
    poor = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=5.0,
        benchmark_return_pct=3.0,
        alpha_pct=2.0,
        profit_factor=1.5,
        sharpe=1.1,
        max_drawdown_pct=5.0,
        expectancy=0.1,
        n_trades=50,
        win_rate_pct=55.0,
        expectancy_per_trade=0.01,
        risk_of_ruin=0.05,
        maxdd_p95=6.0,
        pnl_p05=-0.01,
        robustness_score=0.4,
    )
    assert not search_optimizer._passes_hard_constraints(cfg, poor)

    solid = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=7.0,
        benchmark_return_pct=4.0,
        alpha_pct=3.0,
        profit_factor=1.6,
        sharpe=1.2,
        max_drawdown_pct=5.0,
        expectancy=0.2,
        n_trades=50,
        win_rate_pct=55.0,
        expectancy_per_trade=0.02,
        risk_of_ruin=0.04,
        maxdd_p95=5.0,
        pnl_p05=-0.01,
        robustness_score=0.75,
    )
    assert search_optimizer._passes_hard_constraints(cfg, solid)


def test_hard_constraints_portfolio_sharpe_disabled_when_min_zero() -> None:
    cfg = SearchConfig(
        ticker="TEST",
        period="6mo",
        min_trades=1,
        min_sharpe=0.0,
        min_profit_factor=0.0,
        max_drawdown_pct=10.0,
        min_portfolio_sharpe=0.0,
    )
    candidate = search_optimizer.CandidateConfig(
        long_threshold=0.55,
        short_threshold=0.45,
        risk_pct=0.01,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-0.5,
        sentiment_max=0.5,
        sentiment_regime="off",
    )
    portfolio_negative = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=1.0,
        benchmark_return_pct=0.5,
        alpha_pct=0.5,
        profit_factor=1.05,
        sharpe=0.1,
        max_drawdown_pct=5.0,
        expectancy=0.01,
        n_trades=10,
        win_rate_pct=55.0,
        expectancy_per_trade=0.001,
        risk_of_ruin=0.05,
        maxdd_p95=6.0,
        portfolio_return_pct=1.0,
        portfolio_profit_factor=1.05,
        portfolio_sharpe=-0.2,
        portfolio_max_drawdown_pct=5.0,
        portfolio_risk_of_ruin=0.05,
        portfolio_maxdd_p95=6.0,
    )
    assert search_optimizer._passes_hard_constraints(cfg, portfolio_negative)


def test_supersearch_cli_limits_follow_global_flags(tmp_path: Path) -> None:
    out_path = tmp_path / "out.csv"
    args = run_supersearch.parse_args(
        [
            "--ticker",
            "TEST",
            "--period",
            "90d",
            "--interval",
            "1h",
            "--max-dd",
            "0.9",
            "--max-ror",
            "0.8",
            "--output",
            str(out_path),
            "--no-use-presets",
        ]
    )
    cfg = run_supersearch._build_config(args)
    candidate = search_optimizer.CandidateConfig(
        long_threshold=0.55,
        short_threshold=0.4,
        risk_pct=0.02,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-0.5,
        sentiment_max=0.5,
        sentiment_regime="off",
    )
    result = search_optimizer.SearchResult(
        config=candidate,
        strategy_return_pct=8.0,
        benchmark_return_pct=1.0,
        alpha_pct=7.0,
        profit_factor=1.3,
        sharpe=1.0,
        max_drawdown_pct=80.0,
        expectancy=5.0,
        n_trades=120,
        win_rate_pct=55.0,
        expectancy_per_trade=0.05,
        risk_of_ruin=0.05,
        maxdd_p95=85.0,
        portfolio_return_pct=6.0,
        portfolio_profit_factor=1.15,
        portfolio_sharpe=0.3,
        portfolio_max_drawdown_pct=45.0,
        portfolio_risk_of_ruin=0.7,
        portfolio_maxdd_p95=55.0,
    )
    assert search_optimizer._passes_hard_constraints(cfg, result)
