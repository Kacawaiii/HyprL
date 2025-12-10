#!/usr/bin/env python3
"""
Thin wrapper to launch a single-ticker live/paper run of run_live_hour.py with dated log paths.

Examples:
    python scripts/ops/run_live_single_ticker_hourly.py \
      --config configs/NVDA-1h_v2.yaml \
      --symbol NVDA \
      --log-root live/logs

    # Symbol can be inferred from config basename (prefix before first '-')
    python scripts/ops/run_live_single_ticker_hourly.py \
      --config configs/MSFT-1h_v2.yaml \
      --log-root live/logs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single ticker live/paper session with dated logs.")
    parser.add_argument("--config", required=True, help="Path to ticker YAML config (e.g., configs/NVDA-1h_v2.yaml).")
    parser.add_argument("--symbol", help="Ticker symbol (e.g., NVDA). If omitted, inferred from config filename.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument("--backfill", action="store_true", help="Optional backfill mode (passes --backfill).")
    parser.add_argument("--start", help="Optional start date (YYYY-MM-DD) for backfill.")
    parser.add_argument("--end", help="Optional end date (YYYY-MM-DD) for backfill.")
    return parser.parse_args()


def infer_symbol(config_path: Path) -> str:
    stem = config_path.stem  # e.g., NVDA-1h_v2 -> NVDA
    if "-" in stem:
        return stem.split("-", 1)[0].upper()
    return stem.upper()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    symbol_upper = (args.symbol or infer_symbol(config_path)).upper()
    symbol_lower = symbol_upper.lower()
    today = datetime.now(timezone.utc).date().isoformat()

    day_dir = args.log_root / f"live_{symbol_lower}" / today
    day_dir.mkdir(parents=True, exist_ok=True)
    trade_log = day_dir / f"trades_{symbol_upper}_live.csv"
    summary_file = day_dir / f"summary_{symbol_upper}_live.json"
    heartbeat_file = day_dir / "heartbeat.json"

    cmd = [
        sys.executable,
        "scripts/run_live_hour.py",
        "--config",
        str(args.config),
        "--trade-log",
        str(trade_log),
        "--summary-file",
        str(summary_file),
    ]
    if args.backfill:
        cmd.append("--backfill")
    if args.start:
        cmd.extend(["--start", args.start])
    if args.end:
        cmd.extend(["--end", args.end])

    mode = "backfill" if args.backfill else "live/paper"
    print(f"[OPS] [{symbol_upper}] mode={mode} log_root={args.log_root} trade_log={trade_log} summary={summary_file}")
    try:
        subprocess.run(cmd, check=True)
        heartbeat_payload = {
            "ticker": symbol_upper,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }
        heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_file.write_text(json.dumps(heartbeat_payload), encoding="utf-8")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        print(f"[ERR] run_live_hour failed with exit code {exc.returncode}")
        raise
    print(f"[OPS] [{symbol_upper}] completed. Trades -> {trade_log}, Summary -> {summary_file}")


if __name__ == "__main__":
    main()
