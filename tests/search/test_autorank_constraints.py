from __future__ import annotations

import json

import pandas as pd

from hyprl.meta.autorank import AutorankConstraints, apply_autorank_filters, build_phase1_shortlist


def test_autorank_constraints_filtering_and_panel() -> None:
    diag = pd.DataFrame(
        [
            {
                "tickers": "AAA",
                "config_index": 0,
                "portfolio_pf": 1.6,
                "portfolio_sharpe": 1.2,
                "portfolio_dd": 12.0,
                "trades_backtest": 80,
                "corr_max": 0.4,
                "portfolio_weights": json.dumps({"AAA": 0.55, "BBB": 0.45}),
                "final_score": 0.92,
                "base_score_normalized": 0.8,
            },
            {
                "tickers": "BBB",
                "config_index": 1,
                "portfolio_pf": 1.0,
                "portfolio_sharpe": 0.3,
                "portfolio_dd": 45.0,
                "trades_backtest": 20,
                "corr_max": 0.95,
                "portfolio_weights": json.dumps({"BBB": 0.85, "CCC": 0.15}),
                "final_score": 0.40,
                "base_score_normalized": 0.35,
            },
        ]
    )
    constraints = AutorankConstraints(
        min_pf=1.2,
        min_sharpe=0.5,
        max_dd=0.2,
        max_corr=0.8,
        min_trades=50,
        min_weight=0.2,
        max_weight=0.7,
    )
    filtered, stats = apply_autorank_filters(diag, constraints)
    assert len(filtered) == 1
    assert stats["filtered_by_pf"] == 1 or stats["filtered_by_sharpe"] == 1
    assert stats["filtered_by_weight_max"] == 1
    panel = build_phase1_shortlist(filtered, max_strategies=2)
    assert list(panel.columns) == [
        "strat_id",
        "source_csv",
        "config_index",
        "tickers",
        "interval",
        "period",
        "pf_backtest",
        "sharpe_backtest",
        "maxdd_backtest",
        "expectancy_backtest",
        "trades_backtest",
        "portfolio_risk_of_ruin",
        "correlation_max",
    ]
    assert panel.iloc[0]["tickers"] == "AAA"
