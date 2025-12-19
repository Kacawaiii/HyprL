#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ensure we can import run_live_replay when executed as a script
sys.path.append(str(Path(__file__).resolve().parent))
from run_live_replay import main as run_live_replay_main  # type: ignore


def run_replays(configs: List[str], start: str, end: str, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_logs: List[Path] = []
    for cfg in configs:
        cfg_path = Path(cfg)
        stem = cfg_path.stem
        trade_csv = out_dir / f"trades_{stem}.csv"
        args = [
            "--config",
            str(cfg_path),
            "--start",
            start,
            "--end",
            end,
            "--trade-log",
            str(trade_csv),
        ]
        run_live_replay_main(args=args)  # type: ignore
        trade_logs.append(trade_csv)
    return trade_logs


def aggregate_portfolio(trade_logs: List[Path], weights: List[float]) -> dict:
    frames = []
    for path, w in zip(trade_logs, weights):
        df = pd.read_csv(path, parse_dates=["exit_timestamp"])
        df = df.sort_values("exit_timestamp")
        pnl = df["pnl"].astype(float)
        returns = df["return_pct"].astype(float)
        equity = 10000.0 + pnl.cumsum()
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": df["exit_timestamp"].values,
                    "pnl": pnl * w,
                    "return_pct": returns * w,
                    "equity": equity * w,
                }
            )
        )
    merged = pd.concat(frames, axis=0).sort_values("timestamp")
    merged["portfolio_pnl"] = merged["pnl"]
    merged["portfolio_equity"] = 10000.0 + merged["portfolio_pnl"].cumsum()
    gross_profit = merged["portfolio_pnl"][merged["portfolio_pnl"] > 0].sum()
    gross_loss = -merged["portfolio_pnl"][merged["portfolio_pnl"] < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    equity = merged["portfolio_equity"]
    dd = (equity.cummax() - equity) / equity.cummax()
    maxdd = float(dd.max() * 100) if not dd.empty else 0.0
    return {
        "pf": float(pf),
        "maxdd": maxdd,
        "trades": int(len(merged)),
        "equity_end": float(equity.iloc[-1]) if not equity.empty else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio replay via sequential per-ticker replays.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config paths (order must match weights).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        type=float,
        help="Portfolio weights aligned with configs.",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", default="data/portfolio_replay", type=Path)
    parser.add_argument("--summary-out", type=Path, help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.weights):
        raise SystemExit("configs and weights must have same length")
    trade_logs = run_replays(args.configs, args.start, args.end, Path(args.out_dir))
    summary = aggregate_portfolio(trade_logs, args.weights)
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
