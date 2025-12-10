#!/usr/bin/env python3
"""One-shot ops maintenance helper for HyprL Ascendant v2.

Runs (optionally) the per-ticker hourly runners, always runs daily ops checks
and palier status, and (optionally) the broker dry-run bridge.

Example:
    python scripts/ops/run_ops_maintenance_once.py --refresh-runners --run-broker
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]

TICKER_CONFIGS = {
    "NVDA": "configs/NVDA-1h_v2.yaml",
    "MSFT": "configs/MSFT-1h_v2.yaml",
    "AMD": "configs/AMD-1h_v2.yaml",
    "META": "configs/META-1h_v2.yaml",
    "QQQ": "configs/QQQ-1h_v2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ops maintenance tasks (runners, daily checks, broker dry-run).")
    parser.add_argument(
        "--tickers",
        default="NVDA,MSFT,AMD,META,QQQ",
        help="Comma-separated tickers to process (default: NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--refresh-runners",
        action="store_true",
        help="If set, run per-ticker hourly runners once.",
    )
    parser.add_argument(
        "--run-broker",
        action="store_true",
        help="If set, run the broker dry-run bridge once.",
    )
    parser.add_argument(
        "--log-root",
        default="live/logs",
        help="Log root (default: live/logs).",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return proc


def refresh_runners(tickers: Iterable[str]) -> None:
    print("[STEP] Refresh runners")
    for ticker in tickers:
        cfg = TICKER_CONFIGS.get(ticker.upper())
        if not cfg:
            print(f"[RUNNER][{ticker}] skipped (no config mapping)")
            continue
        cmd = [sys.executable, "scripts/ops/run_live_single_ticker_hourly.py", "--config", cfg]
        proc = subprocess.run(cmd, cwd=ROOT)
        print(f"[RUNNER][{ticker}] exit code={proc.returncode}")
        if proc.returncode != 0:
            print(f"[RUNNER][{ticker}][WARN] runner failed", file=sys.stderr)


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    failure = False

    if args.refresh_runners:
        refresh_runners(tickers)
    else:
        print("[STEP] Refresh runners skipped")

    print("[STEP] Daily ops checks")
    daily_cmd = [sys.executable, "scripts/ops/run_daily_ops_checks.py"]
    daily_proc = run_cmd(daily_cmd, cwd=ROOT)
    if daily_proc.returncode != 0:
        failure = True
    print("[STEP] Palier status")
    palier_cmd = [sys.executable, "scripts/ops/check_palier_status.py", "--live-root", args.log_root]
    palier_proc = run_cmd(palier_cmd, cwd=ROOT)
    if palier_proc.returncode != 0:
        failure = True

    broker_status = "SKIPPED"
    if args.run_broker:
        api_base = os.getenv("HYPRL_API_BASE", "http://127.0.0.1:8000")
        broker_cmd = [
            sys.executable,
            "scripts/ops/run_broker_dryrun.py",
            "--tickers",
            ",".join(tickers),
            "--api-base",
            api_base,
            "--state-file",
            f"{args.log_root}/broker_state_dryrun.json",
            "--audit-file",
            f"{args.log_root}/audit_trades_dryrun.jsonl",
        ]
        print(f"[STEP] Broker dry-run (api-base={api_base})")
        broker_proc = run_cmd(broker_cmd, cwd=ROOT)
        if broker_proc.returncode != 0:
            broker_status = "FAIL"
            failure = True
            print("[BROKER][WARN] run_broker_dryrun failed (possible API down)", file=sys.stderr)
        else:
            broker_status = "OK"
    else:
        print("[STEP] Broker dry-run skipped")

    print("[SUMMARY]")
    print(f"  Runners refreshed: {'yes' if args.refresh_runners else 'no'}")
    print(f"  Daily ops checks: {'FAIL' if daily_proc.returncode != 0 else 'OK'} (exit={daily_proc.returncode})")
    print(f"  Palier status: {'FAIL' if palier_proc.returncode != 0 else 'OK'} (exit={palier_proc.returncode})")
    print(f"  Broker dry-run: {broker_status}")

    sys.exit(1 if failure else 0)


if __name__ == "__main__":
    main()
