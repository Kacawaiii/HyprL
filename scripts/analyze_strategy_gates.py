#!/usr/bin/env python3
"""Analyze trade logs and evaluate Gate1/Gate2 status per strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from hyprl.analysis.trade_aggregator import (
    RollingMetrics,
    StrategyMetrics,
    compute_metrics,
    compute_rolling,
    group_trades_by_strategy,
    load_trades,
)
from hyprl.analysis.trade_gates import GateDecision, check_gate1, check_gate2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gate1/Gate2 for trade logs.")
    parser.add_argument(
        "--trade-logs",
        nargs="+",
        required=True,
        help="One or more CSV paths of trades to aggregate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/strategy_gates_status.csv"),
        help="Output CSV path for per-strategy status.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Optional directory to write per-strategy markdown reports.",
    )
    parser.add_argument(
        "--report-strategy",
        help="Optional specific strategy_id to emit a markdown report for.",
    )
    parser.add_argument(
        "--default-symbol",
        help="Fallback symbol to use when missing from logs.",
    )
    parser.add_argument(
        "--default-source",
        help="Fallback source_type to use when missing from logs.",
    )
    parser.add_argument(
        "--default-session",
        help="Fallback session_id to use when missing from logs.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="Initial equity used for drawdown/returns (default: 10000).",
    )
    return parser.parse_args()


def _status_label(metrics: StrategyMetrics, gate1: GateDecision, gate2: GateDecision) -> str:
    if gate2.passed:
        return "Gate2"
    if gate1.passed:
        return "Gate1"
    if metrics.n_trades < 300:
        return "<300"
    return "partial"


def _maybe_write_report(
    reports_dir: Path,
    strategy_id: str,
    metrics: StrategyMetrics,
    rolling: RollingMetrics,
    gate1: GateDecision,
    gate2: GateDecision,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{strategy_id}.md"
    lines = [
        f"# Strategy {strategy_id}",
        "",
        "## Metrics",
        f"- Trades: {metrics.n_trades}",
        f"- PF: {metrics.pf:.4f}",
        f"- Sharpe: {metrics.sharpe:.4f}",
        f"- MaxDD: {metrics.maxdd:.4f}",
        f"- Expectancy: {metrics.expectancy:.4f}",
        f"- Winrate: {metrics.winrate:.4f}",
        f"- Source mix: {metrics.source_mix}",
        f"- Last exit: {metrics.last_exit_ts}",
        "",
        "## Rolling (100 trades)",
        f"- Min PF: {rolling.pf_min_100 if rolling.pf_min_100 is not None else 'n/a'}",
        f"- Max DD: {rolling.maxdd_max_100 if rolling.maxdd_max_100 is not None else 'n/a'}",
        "",
        "## Gates",
        f"- Gate1: {'PASS' if gate1.passed else 'FAIL'} ({gate1.reasons})",
        f"- Gate2: {'PASS' if gate2.passed else 'FAIL'} ({gate2.reasons})",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trades = load_trades(
        args.trade_logs,
        default_source=args.default_source,
        default_symbol=args.default_symbol,
        default_session=args.default_session,
    )
    grouped = group_trades_by_strategy(trades)
    rows: list[dict[str, Any]] = []
    reports_dir = args.reports_dir
    focus_id = args.report_strategy

    for strategy_id, df in grouped.items():
        metrics = compute_metrics(df, initial_equity=args.initial_equity)
        rolling = compute_rolling(df, window=100, initial_equity=args.initial_equity)
        gate1 = check_gate1(metrics, rolling, strategy_id=strategy_id)
        gate2 = check_gate2(metrics, rolling)
        status = _status_label(metrics, gate1, gate2)
        row = {
            "strategy_id": strategy_id,
            "trades": metrics.n_trades,
            "pf": metrics.pf,
            "sharpe": metrics.sharpe,
            "maxdd": metrics.maxdd,
            "expectancy": metrics.expectancy,
            "winrate": metrics.winrate,
            "gate1_passed": gate1.passed,
            "gate2_passed": gate2.passed,
            "status": status,
            "last_update": metrics.last_exit_ts,
            "source_mix": metrics.source_mix,
        }
        rows.append(row)
        if reports_dir and (focus_id is None or focus_id == strategy_id):
            _maybe_write_report(reports_dir, strategy_id, metrics, rolling, gate1, gate2)

    status_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
