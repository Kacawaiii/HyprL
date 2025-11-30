#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hyprl.analysis.phase1 import build_meta_robustness_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporte le dataset Meta-ML pour la robustesse Phase 1.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/experiments/PHASE1_PANEL.csv"),
        help="Panel Phase 1 (sortie build_phase1_panel.py).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/experiments/phase1_results.csv"),
        help="Résultats analytiques (sortie analyze_phase1_results.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/meta_robustness_dataset.csv"),
        help="Chemin du dataset Meta-ML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        raise SystemExit(f"Panel introuvable: {args.panel}")
    if not args.results.exists():
        raise SystemExit(f"Résultats Phase 1 introuvables: {args.results}")
    dataset = build_meta_robustness_dataset(args.panel, args.results)
    if dataset.empty:
        print("[WARN] Dataset Meta-ML vide (vérifier panel/results).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"[OK] Dataset Meta-robustesse exporté ({len(dataset)} lignes) → {args.output}")


if __name__ == "__main__":
    main()
