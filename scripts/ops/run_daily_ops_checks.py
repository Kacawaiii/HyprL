#!/usr/bin/env python3
"""Run daily Palier checks (status, health alerts, heartbeats).

Usage:
    python scripts/ops/run_daily_ops_checks.py \
      --log-root live/logs \
      --health live/logs/portfolio_live/health_asc_v2.json \
      --symbols NVDA,MSFT,AMD,META,QQQ

Notes:
- Designed for cron/systemd; logs to stdout/stderr (redirect in scheduler).
- Exit code non-zero if any check fails (palier status, health alert, or heartbeat).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily Palier status + health + heartbeat checks.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument(
        "--health",
        type=Path,
        default=Path("live/logs/portfolio_live/health_asc_v2.json"),
        help="Portfolio health JSON path (default: live/logs/portfolio_live/health_asc_v2.json).",
    )
    parser.add_argument(
        "--symbols",
        default="NVDA,MSFT,AMD,META,QQQ",
        help="Comma-separated symbols to check heartbeats (default: NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--pf-alert",
        type=float,
        default=1.3,
        help="PF alert threshold (default: 1.3).",
    )
    parser.add_argument(
        "--dd-alert",
        type=float,
        default=20.0,
        help="DD alert threshold percent (default: 20).",
    )
    parser.add_argument(
        "--sharpe-alert",
        type=float,
        default=1.5,
        help="Sharpe alert threshold (default: 1.5).",
    )
    parser.add_argument(
        "--max-age-min",
        type=int,
        default=90,
        help="Max heartbeat age minutes (default: 90).",
    )
    return parser.parse_args()


def run_cmd(cmd: Sequence[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.returncode


def main() -> None:
    args = parse_args()
    cmds = [
        [
            "python",
            "scripts/ops/check_palier_status.py",
            "--live-root",
            str(args.log_root),
        ],
        [
            "python",
            "scripts/ops/alert_portfolio_health.py",
            "--health",
            str(args.health),
            "--pf-alert",
            str(args.pf_alert),
            "--dd-alert",
            str(args.dd_alert),
            "--sharpe-alert",
            str(args.sharpe_alert),
        ],
        [
            "python",
            "scripts/ops/check_heartbeat.py",
            "--root",
            str(args.log_root),
            "--max-age-min",
            str(args.max_age_min),
            "--symbols",
            args.symbols,
        ],
    ]

    failures = 0
    for cmd in cmds:
        rc = run_cmd(cmd)
        if rc != 0:
            failures += 1

    if failures:
        print(f"[FAIL] {failures} check(s) failed", file=sys.stderr)
        sys.exit(1)
    print("[OK] All checks passed")


if __name__ == "__main__":
    main()
