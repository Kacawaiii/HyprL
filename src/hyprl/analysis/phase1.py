from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import math
import numpy as np
import pandas as pd

from hyprl.portfolio.core import compute_portfolio_stats


@dataclass(slots=True)
class Phase1Filters:
    min_pf: float = 1.05
    min_sharpe: float = 0.5
    max_dd: float = 0.40  # ratio
    max_ror: float = 0.20
    min_trades: int = 30
    max_correlation: float = 0.90


def _coalesce(row: Mapping[str, object], keys: Iterable[str], default: float) -> float:
    for key in keys:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _normalize_tickers(value: str | None) -> str:
    if not value or not isinstance(value, str):
        return ""
    tokens = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if token:
            tokens.append(token.upper())
    return ",".join(tokens)


def _candidate_score(pf: float, sharpe: float, max_dd_pct: float, risk_of_ruin: float, corr_max: float) -> float:
    dd_ratio = max(max_dd_pct / 100.0, 0.01)
    core = (max(pf, 0.0) * max(sharpe, 0.0)) / dd_ratio
    penalty = (risk_of_ruin * 10.0) + max(corr_max - 0.5, 0.0)
    return core - penalty


def _equity_volatility(series: pd.Series) -> float:
    returns = series.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1))


def _estimate_backtest_equity_vol(row: Mapping[str, object]) -> float:
    vol = _coalesce(row, ("portfolio_equity_vol", "equity_volatility"), math.nan)
    if math.isfinite(vol) and vol > 0:
        return float(vol)
    total_return = _coalesce(row, ("portfolio_return_pct", "strategy_return_pct"), math.nan)
    sharpe = _coalesce(row, ("portfolio_sharpe", "sharpe"), math.nan)
    if math.isfinite(total_return) and math.isfinite(sharpe) and abs(sharpe) > 1e-6:
        approx = abs(total_return / max(sharpe, 1e-6))
        return float(approx)
    return 0.0


def _init_stats() -> dict[str, int]:
    return {
        "total": 0,
        "survivors": 0,
        "selected": 0,
        "filtered_pf": 0,
        "filtered_sharpe": 0,
        "filtered_dd": 0,
        "filtered_ror": 0,
        "filtered_trades": 0,
        "filtered_correlation": 0,
    }


def _log_empty_panel(filters: Phase1Filters, stats: Mapping[str, int]) -> None:
    total = stats.get("total", 0)
    print(
        "[WARN] Phase 1 panel vide "
        f"(candidats={total}, pf<{filters.min_pf}: {stats.get('filtered_pf', 0)}, "
        f"sharpe<{filters.min_sharpe}: {stats.get('filtered_sharpe', 0)}, "
        f"dd>{filters.max_dd:.2f}: {stats.get('filtered_dd', 0)}, "
        f"ror>{filters.max_ror:.2f}: {stats.get('filtered_ror', 0)}, "
        f"trades<{filters.min_trades}: {stats.get('filtered_trades', 0)}, "
        f"corr>{filters.max_correlation:.2f}: {stats.get('filtered_correlation', 0)})."
    )


def build_phase1_panel(
    csv_paths: list[Path],
    filters: Phase1Filters,
    max_strategies: int,
    diagnostics: MutableMapping[str, int] | None = None,
) -> pd.DataFrame:
    stats = _init_stats()
    records: list[dict[str, object]] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "config_index"})
        for _, row in df.iterrows():
            row_map = row.to_dict()
            stats["total"] += 1
            pf = _coalesce(row_map, ("portfolio_profit_factor", "profit_factor"), 0.0)
            sharpe = _coalesce(row_map, ("portfolio_sharpe", "sharpe"), 0.0)
            max_dd_pct = _coalesce(
                row_map,
                ("portfolio_max_drawdown_pct", "max_drawdown_pct"),
                100.0,
            )
            ror = _coalesce(row_map, ("portfolio_risk_of_ruin", "risk_of_ruin"), 1.0)
            trades = int(_coalesce(row_map, ("n_trades", "trades"), 0.0))
            corr_max = _coalesce(row_map, ("correlation_max", "corr_max"), 0.0)
            if pf < filters.min_pf:
                stats["filtered_pf"] += 1
                continue
            if sharpe < filters.min_sharpe:
                stats["filtered_sharpe"] += 1
                continue
            if max_dd_pct > filters.max_dd * 100.0:
                stats["filtered_dd"] += 1
                continue
            if ror > filters.max_ror:
                stats["filtered_ror"] += 1
                continue
            if trades < filters.min_trades:
                stats["filtered_trades"] += 1
                continue
            if corr_max > filters.max_correlation:
                stats["filtered_correlation"] += 1
                continue
            tickers = _normalize_tickers(
                str(row_map.get("tickers") or row_map.get("search_tickers") or "")
            )
            interval = str(row_map.get("search_interval") or row_map.get("interval") or "1h")
            period = str(row_map.get("search_period") or row_map.get("period") or "")
            expectancy = _coalesce(row_map, ("expectancy_per_trade", "expectancy"), 0.0)
            score = _candidate_score(pf, sharpe, max_dd_pct, ror, corr_max)
            records.append(
                {
                    "source_csv": str(csv_path),
                    "config_index": int(row_map["config_index"]),
                    "tickers": tickers,
                    "interval": interval,
                    "period": period,
                    "pf_backtest": pf,
                    "sharpe_backtest": sharpe,
                    "maxdd_backtest": max_dd_pct,
                    "expectancy_backtest": expectancy,
                    "trades_backtest": trades,
                    "portfolio_risk_of_ruin": ror,
                    "correlation_max": corr_max,
                    "score": score,
                }
            )
    stats["survivors"] = len(records)
    if not records:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(stats)
        _log_empty_panel(filters, stats)
        return pd.DataFrame(
            columns=[
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
        )
    panel = pd.DataFrame.from_records(records)
    panel = panel.sort_values("score", ascending=False).head(max_strategies).reset_index(drop=True)
    stats["selected"] = len(panel)
    panel["strat_id"] = [f"STRAT_{idx:02d}" for idx in range(1, len(panel) + 1)]
    cols = [
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
    result = panel[cols]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stats)
    return result


def load_backtest_metrics(csv_path: str | Path, config_index: int) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if config_index < 0 or config_index >= len(df):
        raise IndexError("config_index out of range for backtest CSV")
    row = df.iloc[config_index].to_dict()
    return {
        "pf": _coalesce(row, ("portfolio_profit_factor", "profit_factor"), 0.0),
        "sharpe": _coalesce(row, ("portfolio_sharpe", "sharpe"), 0.0),
        "maxdd": _coalesce(row, ("portfolio_max_drawdown_pct", "max_drawdown_pct"), 0.0),
        "expectancy": _coalesce(row, ("expectancy_per_trade", "expectancy"), 0.0),
        "trades": int(_coalesce(row, ("n_trades", "trades"), 0.0)),
        "win_rate": _coalesce(row, ("win_rate_pct", "win_rate"), 0.0) / 100.0,
        "equity_vol": _estimate_backtest_equity_vol(row),
    }


def _extract_trade_pnls(trades_path: Path) -> list[float]:
    if not trades_path.exists():
        return []
    df = pd.read_csv(trades_path)
    if df.empty:
        return []
    trade_pnls: list[float] = []
    positions: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        side = str(row.get("side", "")).lower()
        qty = float(row.get("quantity", 0.0))
        delta = qty if side == "buy" else -qty
        prev_qty = positions.get(ticker, 0.0)
        new_qty = prev_qty + delta
        positions[ticker] = new_qty
        realized = float(row.get("realized_pnl", 0.0))
        if abs(new_qty) < 1e-9 and abs(prev_qty) > 0.0:
            trade_pnls.append(realized)
        elif abs(new_qty) < 1e-9 and realized != 0.0:
            trade_pnls.append(realized)
    return trade_pnls


def load_live_metrics(log_dir: Path | str) -> dict[str, float]:
    base = Path(log_dir)
    equity_path = base / "equity.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing equity log: {equity_path}")
    equity_df = pd.read_csv(equity_path, parse_dates=["timestamp"])
    if equity_df.empty:
        return {
            "pf": 0.0,
            "sharpe": 0.0,
            "maxdd": 0.0,
            "expectancy": 0.0,
            "trades": 0,
        }
    equity_series = pd.Series(equity_df["equity"].to_numpy(), index=equity_df["timestamp"])
    initial_balance = float(equity_series.iloc[0])
    stats = compute_portfolio_stats(
        portfolio_equity=equity_series,
        initial_balance=initial_balance,
        seed=None,
        bootstrap_runs=128,
    )
    trade_pnls = _extract_trade_pnls(base / "trades.csv")
    trades = len(trade_pnls)
    expectancy = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    positive = sum(pnl for pnl in trade_pnls if pnl > 0)
    negative = sum(pnl for pnl in trade_pnls if pnl < 0)
    if negative < 0:
        pf = positive / abs(negative)
    elif positive > 0:
        pf = float("inf")
    else:
        pf = stats["profit_factor"]
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    win_rate = float(wins / trades) if trades else 0.0
    equity_vol = _equity_volatility(equity_series)
    return {
        "pf": float(pf),
        "sharpe": float(stats["sharpe"]),
        "maxdd": float(stats["max_drawdown_pct"]),
        "expectancy": expectancy,
        "trades": trades,
        "win_rate": win_rate,
        "equity_vol": equity_vol,
    }


def compute_robustness_score(
    pf_ratio: float,
    sharpe_ratio: float,
    dd_ratio: float,
    equity_vol_ratio: float,
    winrate_delta: float,
) -> float:
    pf_component = np.clip(pf_ratio, 0.0, 2.0) / 2.0 if math.isfinite(pf_ratio) else 0.5
    sharpe_component = np.clip(sharpe_ratio, 0.0, 2.0) / 2.0 if math.isfinite(sharpe_ratio) else 0.5
    dd_inverse = (
        np.clip(2.0 - np.clip(dd_ratio, 0.0, 2.0), 0.0, 1.0) if math.isfinite(dd_ratio) else 0.5
    )
    vol_inverse = (
        np.clip(2.0 - np.clip(equity_vol_ratio, 0.0, 2.0), 0.0, 1.0)
        if math.isfinite(equity_vol_ratio)
        else 0.5
    )
    # Winrate delta expected in [-0.2, 0.2]; center on 0.0 and scale
    win_component = np.clip((winrate_delta + 0.2) / 0.4, 0.0, 1.0) if math.isfinite(winrate_delta) else 0.5
    score = (
        0.3 * pf_component
        + 0.3 * sharpe_component
        + 0.2 * dd_inverse
        + 0.1 * vol_inverse
        + 0.1 * win_component
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_phase1_comparison(
    backtest: Mapping[str, float],
    live: Mapping[str, float],
) -> dict[str, float]:
    """
    Compute robustness ratios between backtest and live performance.
    
    Robustness Formula:
    -------------------
    For each metric M:
        M_ratio = M_live / M_backtest
    
    Interpretation:
    ---------------
    - M_ratio ≈ 1.0: Strategy performs as expected (robust)
    - M_ratio > 1.0: Strategy performs BETTER live (rare, could indicate luck or overfitting detection)
    - M_ratio < 0.8: Strategy degrades significantly (fragile)
    
    Special Cases:
    --------------
    - max_drawdown: Lower is better, so DD_ratio < 1.0 is GOOD (improved DD)
    - risk_of_ruin: Lower is better, so RoR_ratio < 1.0 is GOOD (reduced tail risk)
    
    Legacy robustness_score:
    ------------------------
    The original `robustness_score` is a simplified composite:
        robustness_score = weighted_avg(pf_ratio, sharpe_ratio, 1-dd_ratio, vol_inverse, win_component)
    
    This compresses all metrics into a single number, which:
    - Is useful for quick ranking/filtering
    - Loses granularity (can't see which metric degraded)
    - Has arbitrary weights (0.3 PF, 0.3 Sharpe, 0.2 DD, 0.1 vol, 0.1 winrate)
    
    Recommendation:
    ---------------
    Use individual ratios (pf_ratio, sharpe_ratio, dd_ratio, etc.) for detailed analysis.
    Use robustness_score for quick leaderboard ranking.
    
    Args:
        backtest: Dict with keys {pf, sharpe, maxdd, expectancy, trades, win_rate, equity_vol}
        live: Same structure as backtest
    
    Returns:
        dict: All individual ratios plus legacy robustness_score
    """
    def _ratio(num: float, denom: float) -> float:
        if denom <= 0 or not math.isfinite(denom):
            return math.nan
        return float(num / denom)

    pf_ratio = _ratio(live.get("pf", math.nan), backtest.get("pf", math.nan))
    sharpe_ratio = _ratio(live.get("sharpe", math.nan), backtest.get("sharpe", math.nan))
    dd_ratio = _ratio(live.get("maxdd", math.nan), backtest.get("maxdd", math.nan))
    equity_vol_ratio = _ratio(live.get("equity_vol", math.nan), backtest.get("equity_vol", math.nan))
    expectancy_ratio = _ratio(live.get("expectancy", math.nan), backtest.get("expectancy", math.nan))
    
    # Winrate is already in [0, 1], so delta is more meaningful than ratio
    winrate_delta = float(live.get("win_rate", math.nan)) - float(backtest.get("win_rate", math.nan))

    robustness = compute_robustness_score(
        pf_ratio=pf_ratio,
        sharpe_ratio=sharpe_ratio,
        dd_ratio=dd_ratio,
        equity_vol_ratio=equity_vol_ratio,
        winrate_delta=winrate_delta if math.isfinite(winrate_delta) else 0.0,
    )

    return {
        "pf_ratio": pf_ratio,
        "sharpe_ratio": sharpe_ratio,
        "dd_ratio": dd_ratio,
        "equity_vol_ratio": equity_vol_ratio,
        "expectancy_ratio": expectancy_ratio,
        "winrate_delta": winrate_delta,
        "robustness_score": robustness,
    }


def build_meta_robustness_dataset(panel_path: Path | str, results_path: Path | str) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    results = pd.read_csv(results_path)
    if panel.empty or results.empty:
        return pd.DataFrame()
    merge_keys = ["strat_id", "source_csv", "config_index"]
    merged = results.merge(panel, on=merge_keys, suffixes=("_res", "_panel"))
    cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        csv_path = row["source_csv"]
        cfg_index = int(row["config_index"])
        if csv_path not in cache:
            cache[csv_path] = pd.read_csv(csv_path)
        df = cache[csv_path]
        if cfg_index < 0 or cfg_index >= len(df):
            continue
        cfg_row = df.iloc[cfg_index].to_dict()
        record = {
            "strat_id": row["strat_id"],
            "source_csv": csv_path,
            "config_index": cfg_index,
            "long_threshold": cfg_row.get("long_threshold"),
            "short_threshold": cfg_row.get("short_threshold"),
            "risk_pct": cfg_row.get("risk_pct"),
            "min_ev_multiple": cfg_row.get("min_ev_multiple"),
            "trend_filter": cfg_row.get("trend_filter"),
            "sentiment_min": cfg_row.get("sentiment_min"),
            "sentiment_max": cfg_row.get("sentiment_max"),
            "sentiment_regime": cfg_row.get("sentiment_regime"),
            "weighting_scheme": cfg_row.get("search_weighting_scheme", "equal"),
            "portfolio_profit_factor": cfg_row.get("portfolio_profit_factor"),
            "portfolio_sharpe": cfg_row.get("portfolio_sharpe"),
            "portfolio_max_drawdown_pct": cfg_row.get("portfolio_max_drawdown_pct"),
            "portfolio_equity_vol": cfg_row.get("portfolio_equity_vol"),
            "win_rate_pct": cfg_row.get("win_rate_pct"),
            "trades_backtest": cfg_row.get("n_trades"),
            "correlation_mean": cfg_row.get("correlation_mean"),
            "correlation_max": cfg_row.get("correlation_max"),
            "pf_backtest": row["pf_backtest"],
            "pf_live": row["pf_live"],
            "sharpe_backtest": row["sharpe_backtest"],
            "sharpe_live": row["sharpe_live"],
            "maxdd_backtest": row["maxdd_backtest"],
            "maxdd_live": row["maxdd_live"],
            "winrate_backtest": row.get("winrate_backtest"),
            "winrate_live": row.get("winrate_live"),
            "equity_vol_backtest": row.get("equity_vol_backtest"),
            "equity_vol_live": row.get("equity_vol_live"),
            "pf_ratio": row["pf_ratio"],
            "sharpe_ratio": row["sharpe_ratio"],
            "dd_ratio": row["dd_ratio"],
            "winrate_delta": row.get("winrate_delta"),
            "equity_vol_ratio": row.get("equity_vol_ratio"),
            "robustness_score": row["robustness_score"],
        }
        records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)
