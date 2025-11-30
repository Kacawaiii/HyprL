#!/usr/bin/env python3
"""CLI orchestrator: Autorank CSVs and launch realtime sessions."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import repo
from api.autorank_manager import autorank_manager
from api import db
from api.db import session_scope
from api.schemas import AutorankSessionConfig, AutorankStartRequest


def _parse_csv_args(values: List[str]) -> List[str]:
    csvs: List[str] = []
    for val in values:
        for token in val.replace(";", ",").split(","):
            token = token.strip()
            if token:
                csvs.append(token)
    if not csvs:
        raise SystemExit("Provide at least one --csv path")
    return csvs


def _constraints_from_args(args: argparse.Namespace) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if args.min_pf is not None:
        mapping["min_pf"] = args.min_pf
    if args.min_sharpe is not None:
        mapping["min_sharpe"] = args.min_sharpe
    if args.max_dd is not None:
        mapping["max_dd"] = args.max_dd
    if args.max_corr is not None:
        mapping["max_corr"] = args.max_corr
    if args.min_trades is not None:
        mapping["min_trades"] = args.min_trades
    if args.min_weight is not None:
        mapping["min_weight"] = args.min_weight
    if args.max_weight is not None:
        mapping["max_weight"] = args.max_weight
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autorank Supersearch CSVs then launch sessions via SessionManager.")
    parser.add_argument("--csv", dest="csv_paths", action="append", required=True, help="CSV path(s) under data/experiments or docs/experiments.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of strategies to launch.")
    parser.add_argument("--meta-model", help="Optional Meta-ML model path (model.joblib).")
    parser.add_argument("--meta-weight", type=float, default=0.4, help="Meta weight [0-1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pf", type=float)
    parser.add_argument("--min-sharpe", type=float)
    parser.add_argument("--max-dd", type=float)
    parser.add_argument("--max-corr", type=float)
    parser.add_argument("--min-trades", type=int)
    parser.add_argument("--min-weight", type=float)
    parser.add_argument("--max-weight", type=float)
    parser.add_argument("--session-interval", default="1m", choices=["1m", "5m", "15m", "1h"], help="Realtime interval.")
    parser.add_argument("--session-threshold", type=float, default=0.6)
    parser.add_argument("--session-risk-pct", type=float, default=0.1)
    parser.add_argument("--kill-switch-dd", type=float, default=0.30)
    parser.add_argument("--enable-paper", action="store_true", help="Submit paper orders (default off).")
    parser.add_argument("--account-id", default="acc_autorank_cli")
    parser.add_argument("--token-id", default="tok_autorank_cli")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    csv_paths = _parse_csv_args(args.csv_paths)
    constraints = _constraints_from_args(args)
    session_cfg = AutorankSessionConfig(
        interval=args.session_interval,
        threshold=args.session_threshold,
        risk_pct=args.session_risk_pct,
        kill_switch_dd=args.kill_switch_dd,
        enable_paper=args.enable_paper,
    )
    request = AutorankStartRequest(
        csv_paths=csv_paths,
        top_k=args.top_k,
        meta_model=args.meta_model,
        meta_weight=args.meta_weight,
        constraints=constraints or None,
        session=session_cfg,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    with session_scope() as db:
        repo.ensure_account(db, args.account_id, credits_total=10**6)
        job = await autorank_manager.start_autorank(
            request,
            account_id=args.account_id,
            token_id=args.token_id,
            db=db,
        )
    print(f"[AUTORANK] job={job.job_id} status={job.status} debited={job.debited_credits}")
    print(f"  csv={job.autoranked_csv}")
    print(f"  summary={job.summary_path}")
    for record in job.sessions:
        print(
            f"  session rank={record.rank} session_id={record.session_id} "
            f"source_csv={record.source_csv} cfg={record.config_index}"
        )


def main() -> None:
    args = parse_args()
    db.configure_engine(os.getenv("HYPRL_DB_URL", db.DATABASE_URL), force=True)
    db.init_db()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
