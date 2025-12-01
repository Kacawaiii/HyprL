#!/usr/bin/env python3
"""
Wrapper to run portfolio health monitor for Ascendant v2 live-lite logs.

Example:
    python scripts/ops/run_portfolio_monitor_live.py \
      --log-root live/logs \
      --summary-out live/logs/portfolio_live/health_asc_v2.json

    python scripts/ops/run_portfolio_monitor_live.py \
      --log-root live/logs \
      --output-dir live/logs/portfolio_live \
      --weights NVDA=0.30,MSFT=0.27,AMD=0.27

    python scripts/ops/run_portfolio_monitor_live.py \
      --log-root live/logs \
      --weights NVDA=0.30,MSFT=0.27,AMD=0.27 \
      --summary-out live/logs/portfolio_live/health_custom.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ASC_V2_WEIGHTS = "NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio health monitor for Ascendant v2 live-lite logs.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Path to write portfolio health summary JSON. If omitted, uses --output-dir/health_asc_v2.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("live/logs/portfolio_live"),
        help="If provided, summary file defaults to <output-dir>/health_asc_v2.json unless --summary-out is set.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10000.0,
        help="Initial equity used for portfolio monitor (default: 10000).",
    )
    parser.add_argument(
        "--annualization",
        type=float,
        default=1638.0,
        help="Annualization factor for Sharpe (default: 1638).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=ASC_V2_WEIGHTS,
        help=f"Weights mapping (default: Ascendant v2 -> {ASC_V2_WEIGHTS}).",
    )
    return parser.parse_args()


def parse_weights(weights_arg: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in weights_arg.split(","):
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid weight entry: {item}")
        ticker, val = item.split("=", 1)
        weights[ticker.strip().upper()] = float(val)
    if not weights:
        raise SystemExit("No weights parsed.")
    return weights


def expected_trade_logs(weights: dict[str, float], log_root: Path) -> list[str]:
    logs: list[str] = []
    for ticker in sorted(weights):
        weight = weights[ticker]
        if weight == 0.0:
            continue
        path = log_root / f"live_{ticker.lower()}" / f"trades_{ticker}_live_all.csv"
        if not path.is_file():
            raise SystemExit(f"Missing trade log for {ticker} at {path}. Run concat before monitoring.")
        logs.append(str(path))
    if not logs:
        raise SystemExit("No trade logs resolved for non-zero weights.")
    return logs


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)
    trade_logs = expected_trade_logs(weights, args.log_root)
    summary_path = args.summary_out or (args.output_dir / "health_asc_v2.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/monitor/monitor_portfolio_health.py",
        "--trade-logs",
        *trade_logs,
        "--weights",
        args.weights,
        "--initial-equity",
        str(args.initial_equity),
        "--annualization",
        str(args.annualization),
        "--summary-out",
        str(summary_path),
    ]
    print(f"[OPS] Running monitor: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = summary.get("status", "UNKNOWN")
            print(f"[OPS] Monitor status={status} -> {summary_path}")
        except Exception:  # pragma: no cover - defensive
            print(f"[OPS] Health written to {summary_path} (status not parsed)")
    else:
        print(f"[OPS] Health written to {summary_path}")


if __name__ == "__main__":
    main()
