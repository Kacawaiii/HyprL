from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TABLE_BEGIN = "<!-- BEGIN_UNIVERSE_V1_2 -->"
TABLE_END = "<!-- END_UNIVERSE_V1_2 -->"

TABLE_COLUMNS = [
    "ticker",
    "tradable",
    "short_threshold",
    "best_long_threshold",
    "strategy_return_pct",
    "strategy_ann_pct",
    "benchmark_return_pct",
    "alpha_pct",
    "profit_factor",
    "sharpe",
    "max_drawdown_pct",
    "n_trades",
    "win_rate_pct",
    "expectancy",
    "score",
    "best_regime",
]

TABLE_HEADER = (
    "Ticker | Tradable | Short | Best Long | Strat % | Ann % | Bench % | Alpha % | PF | Sharpe | "
    "Max DD % | Trades | Win % | Exp | Score | Best Regime"
)
TABLE_SEPARATOR = (
    "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"
)


def _format_bool(flag: bool) -> str:
    return "YES" if bool(flag) else "NO"


def _fmt(value: float | int | str, decimals: int = 2) -> str:
    if isinstance(value, str):
        return value or ""
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{decimals}f}"


def generate_universe_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    missing = [col for col in TABLE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Universe CSV {csv_path} missing columns: {missing}")
    lines: list[str] = [TABLE_HEADER, TABLE_SEPARATOR]
    for row in df.itertuples(index=False):
        tradable = _format_bool(getattr(row, "tradable", False))
        entries = [
            getattr(row, "ticker", ""),
            tradable,
            _fmt(getattr(row, "short_threshold", 0.0)),
            _fmt(getattr(row, "best_long_threshold", 0.0)),
            _fmt(getattr(row, "strategy_return_pct", 0.0)),
            _fmt(getattr(row, "strategy_ann_pct", 0.0)),
            _fmt(getattr(row, "benchmark_return_pct", 0.0)),
            _fmt(getattr(row, "alpha_pct", 0.0)),
            _fmt(getattr(row, "profit_factor", 0.0)),
            _fmt(getattr(row, "sharpe", 0.0)),
            _fmt(getattr(row, "max_drawdown_pct", 0.0)),
            f"{int(getattr(row, 'n_trades', 0))}",
            _fmt(getattr(row, "win_rate_pct", 0.0)),
            _fmt(getattr(row, "expectancy", 0.0)),
            _fmt(getattr(row, "score", 0.0)),
            getattr(row, "best_regime", "") or "",
        ]
        lines.append(" | ".join(entries))
    return "\n".join(lines)


def extract_table_block(text: str) -> str:
    if TABLE_BEGIN not in text or TABLE_END not in text:
        raise ValueError("Universe table markers not found.")
    start = text.index(TABLE_BEGIN) + len(TABLE_BEGIN)
    end = text.index(TABLE_END, start)
    return text[start:end].strip()
