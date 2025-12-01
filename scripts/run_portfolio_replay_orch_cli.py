#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate per-ticker run_live_replay.py calls via CLI and emit trade logs."
    )
    parser.add_argument("--configs", nargs="+", required=True, help="YAML config paths (one per ticker).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("live/logs/portfolio/orch_cli"),
        help="Output directory for per-ticker trade logs.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON summary file listing produced trade logs.",
    )
    return parser.parse_args()


def ticker_from_config(path: Path) -> str:
    # Heuristic: use stem up to first '-' (e.g., NVDA-1h_v2.yaml -> NVDA)
    stem = path.stem
    return stem.split("-")[0].upper()


def run_replay_cli(cfg_path: Path, start: str, end: str, trade_log: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_live_replay.py",
        "--config",
        str(cfg_path),
        "--start",
        start,
        "--end",
        end,
        "--trade-log",
        str(trade_log),
    ]
    print(f"[ORCH] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Tuple[str, str]] = []
    for cfg in args.configs:
        cfg_path = Path(cfg)
        ticker = ticker_from_config(cfg_path)
        trade_log = out_dir / f"trades_{ticker}_orch.csv"
        run_replay_cli(cfg_path, args.start, args.end, trade_log)
        produced.append((ticker, str(trade_log)))

    print("[ORCH] Completed. Produced trade logs:")
    for ticker, path in produced:
        print(f"  {ticker}: {path}")

    if args.summary_out:
        payload = {"start": args.start, "end": args.end, "logs": produced}
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
