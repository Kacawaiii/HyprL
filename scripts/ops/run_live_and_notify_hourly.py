#!/usr/bin/env python3
"""
One-shot hourly pipeline: live runs -> concat trades -> portfolio monitor -> Discord alerts.

Usage:
  python scripts/ops/run_live_and_notify_hourly.py \
    --session friends-live \
    --symbols NVDA,MSFT,AMD,META,QQQ \
    --config-template "configs/{symbol}-1h_v2.yaml" \
    --log-root live/logs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ASC_V2_WEIGHTS = "NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live multi-ticker + concat + monitor + Discord poster.")
    parser.add_argument(
        "--symbols",
        default="NVDA,MSFT,AMD,META,QQQ",
        help="Comma-separated symbols for live runs (default: NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--config-template",
        default="configs/{symbol}-1h_v2.yaml",
        help="Template for config paths (default: configs/{symbol}-1h_v2.yaml).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory for live logs (default: live/logs).",
    )
    parser.add_argument(
        "--weights",
        default=ASC_V2_WEIGHTS,
        help=f"Portfolio weights for monitor (default: {ASC_V2_WEIGHTS}).",
    )
    parser.add_argument("--session", required=True, help="Discord session slug (see live/logs/discord_sessions.json).")
    parser.add_argument(
        "--discord-tickers",
        help="Comma-separated tickers to post to Discord (default: --symbols).",
    )
    parser.add_argument("--backfill", action="store_true", help="Pass --backfill to live runner.")
    parser.add_argument("--start", help="Optional start date (YYYY-MM-DD) for backfill.")
    parser.add_argument("--end", help="Optional end date (YYYY-MM-DD) for backfill.")
    parser.add_argument("--skip-concat", action="store_true", help="Skip concat step.")
    parser.add_argument("--skip-monitor", action="store_true", help="Skip portfolio monitor step.")
    parser.add_argument("--skip-discord", action="store_true", help="Skip Discord posting.")
    parser.add_argument("--dry-run-discord", action="store_true", help="Send Discord payloads to stdout only.")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(f"[OPS] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")
    discord_tickers = (
        [s.strip().upper() for s in args.discord_tickers.split(",") if s.strip()]
        if args.discord_tickers
        else symbols
    )

    # 1) Live runs
    live_cmd = [
        sys.executable,
        "scripts/ops/run_live_multi_ticker_hourly.py",
        "--symbols",
        ",".join(symbols),
        "--log-root",
        str(args.log_root),
        "--config-template",
        args.config_template,
    ]
    if args.backfill:
        live_cmd.append("--backfill")
    if args.start:
        live_cmd.extend(["--start", args.start])
    if args.end:
        live_cmd.extend(["--end", args.end])
    run_cmd(live_cmd)

    # 2) Concat trades
    if not args.skip_concat:
        concat_cmd = [
            sys.executable,
            "scripts/ops/concat_live_trades.py",
            "--symbols",
            ",".join(symbols),
            "--log-root",
            str(args.log_root),
        ]
        run_cmd(concat_cmd)

    # 3) Portfolio monitor
    if not args.skip_monitor:
        monitor_cmd = [
            sys.executable,
            "scripts/ops/run_portfolio_monitor_live.py",
            "--log-root",
            str(args.log_root),
            "--weights",
            args.weights,
        ]
        run_cmd(monitor_cmd)

    # 4) Discord poster
    if not args.skip_discord:
        if not os.getenv("DISCORD_BOT_TOKEN") and not args.dry_run_discord:
            raise SystemExit("DISCORD_BOT_TOKEN is required for Discord posting (or use --dry-run-discord).")
        poster_cmd = [
            sys.executable,
            "scripts/ops/post_trade_events_to_discord.py",
            "--session",
            args.session,
            "--tickers",
            ",".join(discord_tickers),
            "--log-root",
            str(args.log_root),
        ]
        if args.dry_run_discord:
            poster_cmd.append("--dry-run")
        run_cmd(poster_cmd)


if __name__ == "__main__":
    main()
