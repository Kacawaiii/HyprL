#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import time
import uuid

from hyprl.connectors.alpaca import AlpacaPaperBroker, AlpacaSource
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger
from hyprl.rt.tuner import Tuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyprL realtime paper MVP (Alpaca).")
    parser.add_argument("--provider", choices=["alpaca"], default="alpaca")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. AAPL,MSFT)")
    parser.add_argument("--interval", default="1m", choices=["1m", "5m", "15m", "1h"])
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--risk-pct", type=float, default=0.25)
    parser.add_argument("--weighting-scheme", choices=["equal", "inv_vol"], default="equal")
    parser.add_argument("--warmup-bars", type=int, default=60)
    parser.add_argument("--session-id", default="auto")
    parser.add_argument(
        "--resume-session",
        dest="resume_session",
        type=str,
        help="Resume an existing session under data/live/sessions/<ID> (append, no duplicates).",
    )
    parser.add_argument("--enable-paper", action="store_true", help="Submit paper orders (default dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run even if enable-paper was true.")
    parser.add_argument("--meta-registry", help="KEY@Stage pour le modèle Meta-ML.")
    parser.add_argument("--meta-robustness", help="Chemin direct vers model.joblib.")
    parser.add_argument("--meta-calibration-registry", help="KEY@Stage pour le calibrateur Meta-ML.")
    parser.add_argument("--meta-calibration", help="Chemin direct vers calibrator.joblib.")
    parser.add_argument("--max-orders-per-min", type=int, default=10)
    parser.add_argument("--per-symbol-cap", type=int, default=3)
    parser.add_argument("--min-qty", type=int, default=1)
    parser.add_argument("--max-qty", type=int)
    parser.add_argument("--kill-switch-dd", type=float, default=0.30, help="Drawdown kill-switch (ratio 0-1).")
    parser.add_argument("--log-root", type=Path, default=Path("data/live/sessions"))
    parser.add_argument("--tuner-enable", action="store_true", help="Enable realtime tuner adjustments.")
    parser.add_argument("--tuner-thr-min", type=float, default=0.45)
    parser.add_argument("--tuner-thr-max", type=float, default=0.65)
    parser.add_argument("--tuner-thr-step", type=float, default=0.02)
    parser.add_argument("--tuner-risk-min", type=float, default=0.05)
    parser.add_argument("--tuner-risk-max", type=float, default=0.50)
    parser.add_argument("--tuner-risk-step", type=float, default=0.01)
    parser.add_argument("--tuner-cooldown", type=int, default=20)
    return parser.parse_args()


def _symbols(payload: str) -> list[str]:
    return [token.strip().upper() for token in payload.replace(";", ",").split(",") if token.strip()]


def _gen_session_id() -> str:
    return f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"


async def _main_async(args: argparse.Namespace) -> None:
    if args.provider != "alpaca":
        raise SystemExit("Only alpaca provider supported in MVP")
    source = AlpacaSource()
    broker = AlpacaPaperBroker()
    if args.resume_session:
        session_id = args.resume_session
    elif args.session_id != "auto":
        session_id = args.session_id
    else:
        session_id = _gen_session_id()

    session_dir = args.log_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    live_cfg = LiveConfig(
        symbols=_symbols(args.symbols),
        interval=args.interval,
        threshold=args.threshold,
        risk_pct=args.risk_pct,
        weighting_scheme=args.weighting_scheme,
        warmup_bars=args.warmup_bars,
        enable_paper=args.enable_paper and not args.dry_run,
        session_id=session_id,
        resume_session=args.resume_session,
        meta_model_path=args.meta_robustness,
        meta_calibration_path=args.meta_calibration,
        max_orders_per_min=args.max_orders_per_min,
        per_symbol_cap=args.per_symbol_cap,
        min_qty=args.min_qty,
        max_qty=args.max_qty,
        kill_switch_dd=args.kill_switch_dd,
    )
    if args.tuner_enable:
        live_cfg.tuner = Tuner(
            thr_min=args.tuner_thr_min,
            thr_max=args.tuner_thr_max,
            thr_step=args.tuner_thr_step,
            risk_min=args.tuner_risk_min,
            risk_max=args.tuner_risk_max,
            risk_step=args.tuner_risk_step,
            cooldown_bars=args.tuner_cooldown,
            thr=live_cfg.threshold,
            risk=live_cfg.risk_pct,
        )
    logger = LiveLogger(root=args.log_root, session_id=live_cfg.session_id, config=vars(args))
    dry_flag = "no" if live_cfg.enable_paper else "yes"
    print(f"[RT] Provider={args.provider}  Interval={args.interval}  DryRun={dry_flag}")
    if args.resume_session:
        print(f"[RT] RESUME: {args.resume_session}")
    try:
        await run_realtime_paper(
            source,
            broker,
            live_cfg,
            logger,
            meta_registry=args.meta_registry,
            meta_cal_registry=args.meta_calibration_registry,
        )
    finally:
        logger.close()


def main() -> None:
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
