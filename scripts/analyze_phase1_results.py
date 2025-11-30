#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hyprl.phase1.results import build_phase1_results
from hyprl.utils.provenance import save_dataframe_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse les sessions Phase 1 (paper vs backtest).")
    parser.add_argument(
        "--sessions",
        type=Path,
        default=Path("docs/experiments/PHASE1_SESSIONS.csv"),
        help="CSV produit par run_phase1_experiments.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/phase1_results.csv"),
        help="Synthèse agrégée des sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sessions.exists():
        raise SystemExit(f"Sessions CSV introuvable: {args.sessions}")
    sessions = pd.read_csv(args.sessions)
    if sessions.empty:
        raise SystemExit("Aucune session enregistrée.")
    result_df = build_phase1_results(sessions)
    save_dataframe_with_meta(result_df, args.output, key_columns=["session_id"])
    leaderboard = result_df.sort_values("robustness_score", ascending=False).head(5)
    print("[INFO] Top sessions Phase 1 (robustness):")
    for _, row in leaderboard.iterrows():
        print(
            f"  {row['session_id']}: robustness={row['robustness_score']:.3f} "
            f"PF_live={row['pf_live']:.2f} Sharpe_live={row['sharpe_live']:.2f}"
        )
    print(f"[OK] Résultats sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
