#!/usr/bin/env python3
"""
Generate monthly track-record reports for Palier 1 (NVDA) and Palier 2 (Asc v2 portfolio).

Outputs Markdown/HTML reports and a JSON summary with PF/MaxDD/Sharpe/trades/as_of.

Examples:
    python scripts/ops/run_monthly_track_record.py \
      --live-root live/logs \
      --output-root reports \
      --month 2025-01
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scripts.reporting.export_report import compute_metrics

ASC_V2_WEIGHTS = {
    "NVDA": 0.30,
    "MSFT": 0.27,
    "AMD": 0.27,
    "META": 0.09,
    "QQQ": 0.07,
    "SPY": 0.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monthly track-record reports (Palier 1 NVDA + Palier 2 Asc v2).")
    parser.add_argument("--live-root", type=Path, default=Path("live/logs"), help="Root directory for live logs (default: live/logs).")
    parser.add_argument("--output-root", type=Path, default=Path("reports"), help="Root output directory for reports (default: reports).")
    parser.add_argument("--month", help="Target month in YYYY-MM (default: current month).")
    parser.add_argument("--annualization", type=int, default=1638, help="Annualization factor for Sharpe (default: 1638).")
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="Initial equity used for metrics (default: 10000).")
    parser.add_argument("--formats", default="md,html", help="Comma-separated formats: md,html (default: md,html).")
    return parser.parse_args()


def target_month(args_month: str | None) -> str:
    if args_month:
        return args_month
    return datetime.utcnow().strftime("%Y-%m")


def month_dir(output_root: Path, palier: str, month: str) -> Path:
    return output_root / palier / month


def run_export_report(trade_logs: List[Path], weights: Dict[str, float], initial_equity: float, annualization: int, output_path: Path, fmt: str) -> None:
    cmd = [
        "python",
        "scripts/reporting/export_report.py",
        "--trade-logs",
        *[str(p) for p in trade_logs],
        "--weights",
        ",".join(f"{k}={v}" for k, v in weights.items()),
        "--initial-equity",
        str(initial_equity),
        "--annualization",
        str(annualization),
        "--output",
        str(output_path),
        "--format",
        fmt,
    ]
    subprocess.run(cmd, check=True)


def compute_summary(trade_logs: List[Path], weights: Dict[str, float], annualization: int, initial_equity: float) -> Dict[str, float]:
    merged = []
    for path in trade_logs:
        df = pd.read_csv(path)
        if df.empty:
            continue
        stem = path.stem
        ticker = stem.split("_")[1].upper() if "_" in stem else stem.upper()
        w = weights.get(ticker, 0.0)
        df["pnl"] = df["pnl"].astype(float) * w
        merged.append(df)
    if not merged:
        return {}
    combo = pd.concat(merged, ignore_index=True)
    stats = compute_metrics(combo, annualization=annualization, initial_equity=initial_equity)
    return {
        "pf": stats.pf,
        "maxdd": stats.maxdd,
        "sharpe": stats.sharpe,
        "trades": stats.trades,
        "equity_end": stats.equity_end,
        "equity_start": stats.equity_start,
        "start_ts": stats.start_ts.isoformat() if stats.start_ts else None,
        "end_ts": stats.end_ts.isoformat() if stats.end_ts else None,
    }


def palier1_nvda(live_root: Path) -> Path:
    return live_root / "live_nvda" / "trades_NVDA_live_all.csv"


def palier2_paths(live_root: Path, weights: Dict[str, float]) -> List[Path]:
    paths: List[Path] = []
    for ticker, weight in weights.items():
        if weight == 0.0:
            continue
        sym_lower = ticker.lower()
        paths.append(live_root / f"live_{sym_lower}" / f"trades_{ticker}_live_all.csv")
    return paths


def write_summary_json(summary: Dict[str, float], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    month = target_month(args.month)
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]

    # Palier 1 NVDA
    nvda_trade = palier1_nvda(args.live_root)
    p1_dir = month_dir(args.output_root, "palier1_nvda", month)
    p1_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = p1_dir / f"nvda_{month}.{fmt}"
        run_export_report([nvda_trade], {"NVDA": 1.0}, args.initial_equity, args.annualization, out, fmt)
    p1_summary = compute_summary([nvda_trade], {"NVDA": 1.0}, args.annualization, args.initial_equity)
    write_summary_json(p1_summary, p1_dir / f"nvda_{month}.json")

    # Palier 2 Asc v2
    p2_trade_logs = palier2_paths(args.live_root, ASC_V2_WEIGHTS)
    p2_dir = month_dir(args.output_root, "palier2_asc_v2", month)
    p2_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = p2_dir / f"asc_v2_{month}.{fmt}"
        run_export_report(p2_trade_logs, ASC_V2_WEIGHTS, args.initial_equity, args.annualization, out, fmt)
    p2_summary = compute_summary(p2_trade_logs, ASC_V2_WEIGHTS, args.annualization, args.initial_equity)
    write_summary_json(p2_summary, p2_dir / f"asc_v2_{month}.json")

    print(f"[TRACK] Palier1 NVDA reports in {p1_dir}")
    print(f"[TRACK] Palier2 Asc v2 reports in {p2_dir}")


if __name__ == "__main__":
    main()
