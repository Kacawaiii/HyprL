from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_meta_info(model_path: Path) -> dict[str, str]:
    meta = {
        "meta_model_type": "",
        "meta_dataset_hash": "",
        "meta_trained_at": "",
    }
    meta_path = model_path.with_name("meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        meta["meta_model_type"] = raw.get("model_type", "")
        meta["meta_dataset_hash"] = raw.get("dataset_hash", "")
        meta["meta_trained_at"] = raw.get("timestamp", "")
    return meta


@dataclass(slots=True)
class AutorankConstraints:
    min_pf: float | None = None
    min_sharpe: float | None = None
    max_dd: float | None = None  # ratio (0-1)
    max_corr: float | None = None
    min_trades: int | None = None
    min_weight: float | None = None
    max_weight: float | None = None


def _parse_weights(payload: Any) -> dict[str, float]:
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}
    if isinstance(payload, str) and payload.strip():
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
    return {}


def apply_autorank_filters(
    diag: pd.DataFrame,
    constraints: AutorankConstraints,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "total": int(len(diag)),
        "survivors": 0,
        "filtered_by_pf": 0,
        "filtered_by_sharpe": 0,
        "filtered_by_dd": 0,
        "filtered_by_corr": 0,
        "filtered_by_trades": 0,
        "filtered_by_weight_min": 0,
        "filtered_by_weight_max": 0,
    }
    allowed_indices: list[int] = []
    for idx, row in diag.iterrows():
        pf = float(row.get("portfolio_pf", row.get("pf_backtest", row.get("portfolio_profit_factor", 0.0))))
        sharpe = float(row.get("portfolio_sharpe", row.get("sharpe_backtest", row.get("sharpe", 0.0))))
        max_dd_pct = float(row.get("portfolio_dd", row.get("maxdd_backtest", row.get("portfolio_max_drawdown_pct", 0.0))))
        corr = float(row.get("corr_max", row.get("correlation_max", 0.0)))
        trades = int(row.get("trades_backtest", row.get("n_trades", 0)))
        weights = _parse_weights(row.get("portfolio_weights"))
        failed = False
        if constraints.min_pf is not None and pf < constraints.min_pf:
            stats["filtered_by_pf"] += 1
            failed = True
        if constraints.min_sharpe is not None and sharpe < constraints.min_sharpe:
            stats["filtered_by_sharpe"] += 1
            failed = True
        if constraints.max_dd is not None and max_dd_pct > constraints.max_dd * 100.0:
            stats["filtered_by_dd"] += 1
            failed = True
        if constraints.max_corr is not None and corr > constraints.max_corr:
            stats["filtered_by_corr"] += 1
            failed = True
        if constraints.min_trades is not None and trades < constraints.min_trades:
            stats["filtered_by_trades"] += 1
            failed = True
        if constraints.min_weight is not None and weights:
            w_min = min(weights.values())
            if w_min < constraints.min_weight:
                stats["filtered_by_weight_min"] += 1
                failed = True
        if constraints.max_weight is not None and weights:
            w_max = max(weights.values())
            if w_max > constraints.max_weight:
                stats["filtered_by_weight_max"] += 1
                failed = True
        if not failed:
            allowed_indices.append(idx)
    filtered = diag.loc[allowed_indices].reset_index(drop=True)
    stats["survivors"] = len(filtered)
    return filtered, stats


def build_phase1_shortlist(diag: pd.DataFrame, max_strategies: int) -> pd.DataFrame:
    columns = [
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
    if diag.empty:
        return pd.DataFrame(columns=columns)
    ranked = diag.sort_values("final_score", ascending=False).head(max_strategies)
    records: list[dict[str, Any]] = []
    for offset, (_, row) in enumerate(ranked.iterrows(), start=1):
        records.append(
            {
                "strat_id": f"STRAT_{offset:02d}",
                "source_csv": row.get("source_csv", ""),
                "config_index": int(row.get("config_index", offset - 1)),
                "tickers": row.get("tickers", ""),
                "interval": row.get("search_interval", row.get("interval", "")),
                "period": row.get("search_period", row.get("period", "")),
                "pf_backtest": float(row.get("portfolio_pf", row.get("pf_backtest", 0.0))),
                "sharpe_backtest": float(row.get("portfolio_sharpe", row.get("sharpe_backtest", 0.0))),
                "maxdd_backtest": float(row.get("portfolio_dd", row.get("maxdd_backtest", 0.0))),
                "expectancy_backtest": float(row.get("expectancy_per_trade", row.get("expectancy", 0.0))),
                "trades_backtest": int(row.get("trades_backtest", row.get("n_trades", 0))),
                "portfolio_risk_of_ruin": float(row.get("portfolio_risk_of_ruin", row.get("risk_of_ruin", 0.0))),
                "correlation_max": float(row.get("corr_max", row.get("correlation_max", 0.0))),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def write_summary(
    summary_path: Path,
    model_path: Path | None,
    meta_info: dict[str, str],
    df: pd.DataFrame,
    meta_weight: float,
    seed: int,
    topk: int,
    filters: dict[str, int] | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    header = [
        f"Meta model: {model_path or 'N/A'}",
        f"Meta info: type={meta_info.get('meta_model_type','')}, hash={meta_info.get('meta_dataset_hash','')}, trained_at={meta_info.get('meta_trained_at','')}",
        f"Meta weight: {meta_weight:.2f}, seed={seed}",
        f"Candidates: {len(df)}",
        f"Generated: {timestamp}",
        "",
        "Top strategies:",
    ]
    lines = header
    if filters:
        lines.extend(
            [
                "",
                "Filters:",
                f"  pf: {filters.get('filtered_by_pf', 0)}",
                f"  sharpe: {filters.get('filtered_by_sharpe', 0)}",
                f"  dd: {filters.get('filtered_by_dd', 0)}",
                f"  corr: {filters.get('filtered_by_corr', 0)}",
                f"  trades: {filters.get('filtered_by_trades', 0)}",
                f"  w_min: {filters.get('filtered_by_weight_min', 0)}",
                f"  w_max: {filters.get('filtered_by_weight_max', 0)}",
            ]
        )
        lines.append("")
    top_rows = df.head(topk)
    for idx, row in enumerate(top_rows.itertuples(), start=1):
        lines.append(
            f"{idx:02d}. tickers={row.tickers} cfg={getattr(row, 'config_index', '')} "
            f"thr=({getattr(row, 'long_threshold', float('nan')):.2f}/{getattr(row, 'short_threshold', float('nan')):.2f}) "
            f"risk={getattr(row, 'risk_pct', float('nan')):.3f} base={row.base_score_normalized:.3f} "
            f"meta={row.meta_pred:.3f} final={row.final_score:.3f}"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
