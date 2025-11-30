from __future__ import annotations

import pandas as pd

from hyprl.analysis.phase1 import (
    compute_phase1_comparison,
    load_backtest_metrics,
    load_live_metrics,
)


def build_phase1_results(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Phase 1 results with detailed robustness metrics.
    
    This function compares backtest vs live performance for each paper trading session
    and exposes ALL individual ratios (not just the aggregated robustness_score).
    
    Output Columns:
    ---------------
    - session_id, strat_id, source_csv, config_index, tickers, interval, period, initial_balance
    
    Backtest metrics:
    - pf_backtest, sharpe_backtest, maxdd_backtest, expectancy_backtest
    - trades_backtest, winrate_backtest, equity_vol_backtest
    
    Live metrics:
    - pf_live, sharpe_live, maxdd_live, expectancy_live
    - trades_live, winrate_live, equity_vol_live
    
    Ratios (live / backtest):
    - pf_ratio: PF degradation (< 1.0 = worse live, > 1.0 = better live)
    - sharpe_ratio: Sharpe degradation
    - dd_ratio: DD change (< 1.0 = improved DD live, > 1.0 = worse DD live)
    - expectancy_ratio: Expectancy degradation
    - equity_vol_ratio: Volatility change
    - winrate_delta: Win rate change (absolute difference, not ratio)
    
    Legacy score:
    - robustness_score: Weighted composite (for ranking/filtering)
    
    Usage Example:
    --------------
    ```python
    df = pd.read_csv('phase1_results.csv')
    
    # Find strategies with robust PF and Sharpe (>80% of backtest)
    robust = df[(df['pf_ratio'] > 0.8) & (df['sharpe_ratio'] > 0.8)]
    
    # Find strategies where DD improved live (DD_ratio < 1.0)
    improved_dd = df[df['dd_ratio'] < 1.0]
    
    # Find strategies with degraded expectancy
    degraded = df[df['expectancy_ratio'] < 0.9]
    
    # Sort by composite score
    leaderboard = df.sort_values('robustness_score', ascending=False)
    ```
    
    Args:
        sessions_df: DataFrame with columns {session_id, source_csv, config_index, log_dir, ...}
    
    Returns:
        DataFrame with detailed backtest/live comparison
    """
    if sessions_df.empty:
        raise ValueError("Aucune session Phase 1 fournie.")
    rows: list[dict[str, object]] = []
    for _, row in sessions_df.iterrows():
        backtest = load_backtest_metrics(row["source_csv"], int(row["config_index"]))
        live = load_live_metrics(row["log_dir"])
        comparison = compute_phase1_comparison(backtest, live)
        rows.append(
            {
                "session_id": row["session_id"],
                "strat_id": row.get("strat_id", ""),
                "source_csv": row["source_csv"],
                "config_index": int(row["config_index"]),
                "tickers": row.get("tickers", ""),
                "interval": row.get("interval", ""),
                "period": row.get("period", ""),
                "initial_balance": float(row.get("initial_balance", 0.0)),
                # Backtest metrics
                "pf_backtest": backtest["pf"],
                "sharpe_backtest": backtest["sharpe"],
                "maxdd_backtest": backtest["maxdd"],
                "expectancy_backtest": backtest["expectancy"],
                "trades_backtest": backtest["trades"],
                "winrate_backtest": backtest["win_rate"],
                "equity_vol_backtest": backtest["equity_vol"],
                # Live metrics
                "pf_live": live["pf"],
                "sharpe_live": live["sharpe"],
                "maxdd_live": live["maxdd"],
                "expectancy_live": live["expectancy"],
                "trades_live": live["trades"],
                "winrate_live": live["win_rate"],
                "equity_vol_live": live["equity_vol"],
                # Ratios
                "pf_ratio": comparison["pf_ratio"],
                "sharpe_ratio": comparison["sharpe_ratio"],
                "dd_ratio": comparison["dd_ratio"],
                "expectancy_ratio": comparison["expectancy_ratio"],
                "equity_vol_ratio": comparison["equity_vol_ratio"],
                "winrate_delta": comparison["winrate_delta"],
                # Legacy composite score
                "robustness_score": comparison["robustness_score"],
            }
        )
    return pd.DataFrame(rows)
