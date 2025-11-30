from __future__ import annotations

from pathlib import Path

import math
import pandas as pd

from hyprl.analysis.phase1 import (
    Phase1Filters,
    build_phase1_panel,
    compute_phase1_comparison,
)


def test_compute_phase1_comparison_basic() -> None:
    backtest = {"pf": 2.0, "sharpe": 1.5, "maxdd": 20.0, "win_rate": 0.55, "equity_vol": 0.12}
    live = {"pf": 1.6, "sharpe": 1.2, "maxdd": 25.0, "win_rate": 0.48, "equity_vol": 0.18}
    result = compute_phase1_comparison(backtest, live)
    assert math.isclose(result["pf_ratio"], 0.8)
    assert math.isclose(result["sharpe_ratio"], 0.8)
    assert math.isclose(result["dd_ratio"], 1.25)
    assert math.isclose(result["equity_vol_ratio"], 1.5)
    assert math.isclose(result["winrate_delta"], -0.07)
    assert 0.0 <= result["robustness_score"] <= 1.0


def test_panel_builder_filters_and_limits(tmp_path: Path) -> None:
    csv_path = tmp_path / "supersearch.csv"
    df = pd.DataFrame(
        [
            {
                "profit_factor": 1.8,
                "portfolio_profit_factor": 1.8,
                "portfolio_sharpe": 1.3,
                "portfolio_max_drawdown_pct": 20.0,
                "portfolio_risk_of_ruin": 0.05,
                "n_trades": 150,
                "correlation_max": 0.4,
                "tickers": "AAA,BBB",
                "search_interval": "1h",
                "search_period": "1y",
                "expectancy_per_trade": 45.0,
            },
            {
                "profit_factor": 1.1,
                "portfolio_profit_factor": 1.1,
                "portfolio_sharpe": 0.5,
                "portfolio_max_drawdown_pct": 45.0,
                "portfolio_risk_of_ruin": 0.2,
                "n_trades": 20,
                "correlation_max": 0.9,
                "tickers": "CCC",
                "search_interval": "1h",
                "search_period": "1y",
                "expectancy_per_trade": 10.0,
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    filters = Phase1Filters()
    diagnostics: dict[str, int] = {}
    panel = build_phase1_panel([csv_path], filters, max_strategies=2, diagnostics=diagnostics)
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["tickers"] == "AAA,BBB"
    assert row["pf_backtest"] == 1.8
    assert row["trades_backtest"] == 150
    assert row["strat_id"].startswith("STRAT_")
    assert diagnostics["total"] == 2
    assert diagnostics["survivors"] == 1
    assert diagnostics["filtered_dd"] == 1


def test_panel_builder_logs_when_empty(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "supersearch_empty.csv"
    df = pd.DataFrame(
        [
            {
                "portfolio_profit_factor": 1.0,
                "portfolio_sharpe": 0.2,
                "portfolio_max_drawdown_pct": 80.0,
                "portfolio_risk_of_ruin": 0.8,
                "n_trades": 5,
                "correlation_max": 0.95,
            }
        ]
    )
    df.to_csv(csv_path, index=False)
    filters = Phase1Filters()
    diagnostics: dict[str, int] = {}
    panel = build_phase1_panel([csv_path], filters, max_strategies=2, diagnostics=diagnostics)
    captured = capsys.readouterr()
    assert panel.empty
    assert "Phase 1 panel vide" in captured.out
    assert diagnostics["total"] == 1
    assert diagnostics["survivors"] == 0
    assert diagnostics["filtered_pf"] == 1


def test_panel_builder_handles_column_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "supersearch_alias.csv"
    df = pd.DataFrame(
        [
            {
                "profit_factor": 1.2,
                "sharpe": 0.7,
                "max_drawdown_pct": 25.0,
                "risk_of_ruin": 0.05,
                "trades": 40,
                "corr_max": 0.4,
                "tickers": "XYZ",
                "expectancy": 12.0,
            }
        ]
    )
    df.to_csv(csv_path, index=False)
    filters = Phase1Filters()
    panel = build_phase1_panel([csv_path], filters, max_strategies=1)
    assert len(panel) == 1
    assert panel.iloc[0]["tickers"] == "XYZ"
