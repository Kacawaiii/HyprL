#!/usr/bin/env python3
"""
Sequential launcher for multiple tickers using run_live_single_ticker_hourly.py.

Example:
    python scripts/ops/run_live_multi_ticker_hourly.py \
      --symbols NVDA,MSFT,AMD \
      --log-root live/logs \
      --config-template configs/{symbol}-1h_v2.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple tickers hourly via run_live_single_ticker_hourly.py.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--symbols",
        default="NVDA,MSFT,AMD",
        help="Comma-separated symbols (default: NVDA,MSFT,AMD). Include META/QQQ via override.",
    )
    group.add_argument(
        "--configs",
        nargs="+",
        help="Explicit config paths; symbol inferred from filename prefix before first '-'.",
    )
    parser.add_argument(
        "--config-template",
        default="configs/{symbol}-1h_v2.yaml",
        help="Template for config paths when using --symbols; {symbol} placeholder replaced with uppercase ticker.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument("--backfill", action="store_true", help="Pass --backfill to child runs.")
    parser.add_argument("--start", help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Optional end date (YYYY-MM-DD).")
    return parser.parse_args()


def config_for(symbol: str, template: str) -> Path:
    return Path(template.format(symbol=symbol.upper(), symbol_lower=symbol.lower()))


def infer_symbol_from_config(config_path: Path) -> str:
    stem = config_path.stem
    if "-" in stem:
        return stem.split("-", 1)[0].upper()
    return stem.upper()


def run_symbol(symbol: str, config_path: Path, log_root: Path, backfill: bool, start: str | None, end: str | None) -> None:
    cmd: List[str] = [
        sys.executable,
        "scripts/ops/run_live_single_ticker_hourly.py",
        "--config",
        str(config_path),
        "--symbol",
        symbol.upper(),
        "--log-root",
        str(log_root),
    ]
    if backfill:
        cmd.append("--backfill")
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    print(f"[OPS] Launching {symbol.upper()} -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.configs:
        configs = [Path(c) for c in args.configs]
        if not configs:
            raise SystemExit("No configs provided.")
        for cfg in configs:
            if not cfg.is_file():
                raise SystemExit(f"Config not found: {cfg}")
            sym = infer_symbol_from_config(cfg)
            run_symbol(sym, cfg, args.log_root, args.backfill, args.start, args.end)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            raise SystemExit("No symbols provided.")
        for symbol in symbols:
            cfg = config_for(symbol, args.config_template)
            if not cfg.is_file():
                raise SystemExit(f"Config not found for {symbol}: {cfg}")
            run_symbol(symbol, cfg, args.log_root, args.backfill, args.start, args.end)


if __name__ == "__main__":
    main()
