from __future__ import annotations

from pathlib import Path

import pandas as pd

from hyprl.analysis.phase1 import (
    build_meta_robustness_dataset,
    compute_robustness_score,
)


def test_compute_robustness_score_behaviour() -> None:
    high = compute_robustness_score(
        pf_ratio=1.1,
        sharpe_ratio=1.0,
        dd_ratio=0.9,
        equity_vol_ratio=0.95,
        winrate_delta=0.05,
    )
    low = compute_robustness_score(
        pf_ratio=0.5,
        sharpe_ratio=0.4,
        dd_ratio=1.8,
        equity_vol_ratio=1.7,
        winrate_delta=-0.15,
    )
    assert 0.0 <= low < high <= 1.0


def test_build_meta_robustness_dataset(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    results_path = tmp_path / "results.csv"
    csv_path = tmp_path / "supersearch.csv"

    panel_df = pd.DataFrame(
        [
            {
                "strat_id": "STRAT_01",
                "source_csv": str(csv_path),
                "config_index": 0,
                "tickers": "AAA,BBB",
            }
        ]
    )
    panel_df.to_csv(panel_path, index=False)

    results_df = pd.DataFrame(
        [
            {
                "strat_id": "STRAT_01",
                "source_csv": str(csv_path),
                "config_index": 0,
                "pf_backtest": 1.3,
                "pf_live": 1.1,
                "pf_ratio": 0.85,
                "sharpe_backtest": 0.9,
                "sharpe_live": 0.8,
                "sharpe_ratio": 0.89,
                "maxdd_backtest": 18.0,
                "maxdd_live": 22.0,
                "dd_ratio": 1.22,
                "winrate_backtest": 0.52,
                "winrate_live": 0.5,
                "winrate_delta": -0.02,
                "equity_vol_backtest": 0.15,
                "equity_vol_live": 0.18,
                "equity_vol_ratio": 1.2,
                "robustness_score": 0.6,
            }
        ]
    )
    results_df.to_csv(results_path, index=False)

    config_df = pd.DataFrame(
        [
            {
                "long_threshold": 0.6,
                "short_threshold": 0.4,
                "risk_pct": 0.015,
                "min_ev_multiple": 0.1,
                "trend_filter": True,
                "sentiment_min": -0.4,
                "sentiment_max": 0.5,
                "sentiment_regime": "off",
                "portfolio_profit_factor": 1.3,
                "portfolio_sharpe": 0.9,
                "portfolio_max_drawdown_pct": 18.0,
                "win_rate_pct": 52.0,
                "n_trades": 60,
                "correlation_mean": 0.25,
                "correlation_max": 0.5,
                "search_weighting_scheme": "equal",
            }
        ]
    )
    config_df.to_csv(csv_path, index=False)

    dataset = build_meta_robustness_dataset(panel_path, results_path)
    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["long_threshold"] == 0.6
    assert row["pf_ratio"] == 0.85
    assert row["robustness_score"] == 0.6
