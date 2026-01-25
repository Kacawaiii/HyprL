import pandas as pd
import pytest

from hyprl.backtest.runner import BacktestConfig, RiskConfig, StrategyStats, SupercalcDataset
from hyprl.labels.amplitude import LabelConfig
from hyprl.search.optimizer import (
    CandidateConfig,
    SearchConfig,
    _precompute_native_stats,
    run_search,
)


def _make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 101.0],
            "volume": [1_000.0, 1_100.0, 1_050.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )


def _make_candidate(risk_pct: float) -> tuple[CandidateConfig, BacktestConfig]:
    candidate = CandidateConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        risk_pct=risk_pct,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-1.0,
        sentiment_max=1.0,
        sentiment_regime="off",
    )
    risk_cfg = RiskConfig(balance=10_000.0, risk_pct=risk_pct, atr_multiplier=2.0, reward_multiple=3.0, min_position_size=1)
    bt_config = BacktestConfig(
        ticker="AAPL",
        period="1y",
        interval="1h",
        initial_balance=10_000.0,
        long_threshold=candidate.long_threshold,
        short_threshold=candidate.short_threshold,
        risk=risk_cfg,
        commission_pct=0.0,
        slippage_pct=0.0,
        label=LabelConfig(),
    )
    return candidate, bt_config


def test_precompute_native_stats(monkeypatch):
    prices = _make_prices()
    dataset = SupercalcDataset(rows=[], prices=prices, benchmark_return_pct=0.0, initial_balance=10_000.0)
    entries = [_make_candidate(0.01), _make_candidate(0.02)]

    monkeypatch.setattr("hyprl.search.optimizer.native_engine_available", lambda: True)
    monkeypatch.setattr("hyprl.search.optimizer._build_signal_series", lambda *_: ([1.0] * len(prices), 0, 0))

    def _fake_native_batch(df, signal, cfgs, constraints, top_k, labels=None):
        assert labels == ["0", "1"]
        assert top_k == 2
        report_a = {
            "config": {"label": labels[0]},
            "metrics": {
                "profit_factor": 1.5,
                "sharpe": 1.2,
                "sortino": 1.1,
                "max_drawdown": -0.1,
                "expectancy": 0.02,
                "win_rate": 0.55,
                "risk_of_ruin": 0.3,
                "maxdd_p05": 0.05,
                "maxdd_p95": 0.25,
                "pnl_p05": -0.1,
                "pnl_p50": 0.01,
                "pnl_p95": 0.3,
                "robustness_score": 0.42,
            },
            "equity_curve": [
                {"ts": 1_700_000_000_000, "equity": 1.0},
                {"ts": 1_700_000_100_000, "equity": 1.1},
            ],
            "n_trades": 12,
        }
        report_b = {
            "config": {"label": labels[1]},
            "metrics": {
                "profit_factor": 1.8,
                "sharpe": 1.5,
                "sortino": 1.4,
                "max_drawdown": -0.05,
                "expectancy": 0.03,
                "win_rate": 0.6,
                "risk_of_ruin": 0.25,
                "maxdd_p05": 0.04,
                "maxdd_p95": 0.2,
                "pnl_p05": -0.08,
                "pnl_p50": 0.015,
                "pnl_p95": 0.35,
                "robustness_score": 0.5,
            },
            "equity_curve": [
                {"ts": 1_700_000_000_000, "equity": 1.0},
                {"ts": 1_700_000_100_000, "equity": 1.2},
            ],
            "n_trades": 15,
        }
        return [report_a, report_b]

    monkeypatch.setattr("hyprl.search.optimizer.run_native_search_batch", _fake_native_batch)

    stats_map, metrics_map = _precompute_native_stats([
        "AAPL"
    ], {"AAPL": dataset}, entries, per_ticker_capital=5_000.0)

    assert stats_map
    key0 = (0, "AAPL")
    key1 = (1, "AAPL")
    assert key0 in stats_map and key1 in stats_map
    assert stats_map[key0].final_balance == pytest.approx(5_500.0)
    assert stats_map[key1].final_balance == pytest.approx(6_000.0)
    assert stats_map[key0].risk_of_ruin == pytest.approx(0.3)
    assert stats_map[key0].sortino_ratio == pytest.approx(1.1)
    assert stats_map[key0].pnl_p95 == pytest.approx(0.3)
    assert stats_map[key0].robustness_score == pytest.approx(0.42)
    assert metrics_map[key1]["risk_of_ruin"] == pytest.approx(0.25)


def test_run_search_uses_native_prefetch(monkeypatch):
    prices = _make_prices()
    dataset = SupercalcDataset(rows=[], prices=prices, benchmark_return_pct=1.5, initial_balance=10_000.0)
    stats = StrategyStats(
        final_balance=11_000.0,
        profit_factor=1.4,
        sharpe_ratio=1.1,
        max_drawdown_pct=8.0,
        expectancy=100.0,
        n_trades=20,
        win_rate=0.6,
        trades_in_fear=1,
        trades_in_greed=2,
        trade_returns=[],
        equity_history=[
            (pd.Timestamp("2024-01-01"), 10_000.0),
                (pd.Timestamp("2024-01-15"), 9_000.0),
                (pd.Timestamp("2024-01-31"), 11_000.0),
        ],
        sortino_ratio=0.9,
        risk_of_ruin=0.18,
        maxdd_p95=0.12,
        pnl_p05=-0.05,
        pnl_p50=0.01,
        pnl_p95=0.08,
        robustness_score=0.6,
    )
    metrics = {"risk_of_ruin": 0.18, "profit_factor": 1.4, "sharpe": 1.1, "max_drawdown": -0.08}

    monkeypatch.setattr("hyprl.search.optimizer.prepare_supercalc_dataset", lambda cfg: dataset)
    monkeypatch.setattr(
        "hyprl.search.optimizer._precompute_native_stats",
        lambda tickers, datasets, entries, capital: ({(0, "AAPL"): stats}, {(0, "AAPL"): metrics}),
    )

    def _fail_eval(*_args, **_kwargs):
        raise AssertionError("evaluate_candidates should not be called when native prefetch succeeds")

    monkeypatch.setattr("hyprl.search.optimizer.evaluate_candidates", _fail_eval)

    search_cfg = SearchConfig(
        ticker="AAPL",
        period="1y",
        interval="1h",
        initial_balance=10_000.0,
        seed=42,
        use_presets=False,
        long_thresholds=[0.6],
        short_thresholds=[0.4],
        risk_pcts=[0.01],
        min_trades=10,
        min_profit_factor=1.0,
        min_sharpe=0.0,
        max_risk_of_ruin=0.5,
        min_expectancy=-1.0,
        debug=True,
        engine="native",
    )

    results = run_search(search_cfg)
    assert results
    detail = results[0].per_ticker_details.get("AAPL")
    assert detail is not None
    assert detail["risk_of_ruin"] == pytest.approx(0.18)
    assert detail["sortino"] == pytest.approx(0.9)
    assert detail["maxdd_p95"] == pytest.approx(0.12)
    assert detail["pnl_p05"] == pytest.approx(-0.05)
    assert detail["pnl_p50"] == pytest.approx(0.01)
    assert detail["pnl_p95"] == pytest.approx(0.08)
    assert detail["robustness_score"] == pytest.approx(0.6)
    assert results[0].config.risk_pct == pytest.approx(0.01)