#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hyprl.phase1.paper import Phase1ExecutionConfig, run_phase1_sessions
from hyprl.utils.provenance import save_dataframe_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestre les sessions Phase 1 en mode paper trading.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/experiments/PHASE1_PANEL.csv"),
        help="Panel généré via build_phase1_panel.py.",
    )
    parser.add_argument("--period", default="1y", help="Fenêtre à rejouer si le panel n'a pas de période.")
    parser.add_argument("--interval", default="1h", help="Intervalle fallback si le panel ne précise rien.")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--engine", choices=["auto", "python", "native"], default="auto")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--calibration", default="none")
    parser.add_argument("--default-long-threshold", type=float, default=0.6)
    parser.add_argument("--default-short-threshold", type=float, default=0.4)
    parser.add_argument("--session-prefix", default="phase1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/PHASE1_SESSIONS.csv"),
        help="Fichier CSV listant les sessions exécutées.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        raise SystemExit(f"Panel absent: {args.panel}")
    panel_df = pd.read_csv(args.panel)
    if panel_df.empty:
        raise SystemExit("Panel vide, rien à lancer.")
    exec_cfg = Phase1ExecutionConfig(
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        initial_balance=args.initial_balance,
        engine=args.engine,
        model_type=args.model_type,
        calibration=args.calibration,
        default_long_threshold=args.default_long_threshold,
        default_short_threshold=args.default_short_threshold,
        session_prefix=args.session_prefix,
    )
    sessions_df = run_phase1_sessions(panel_df, exec_cfg)
    save_dataframe_with_meta(sessions_df, args.output, key_columns=["session_id"])
    print(f"[OK] Sessions Phase 1 consignées dans {args.output}")


if __name__ == "__main__":
    main()
