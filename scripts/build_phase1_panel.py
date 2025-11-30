#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hyprl.analysis.phase1 import Phase1Filters, build_phase1_panel


def _parse_csv_paths(payload: str) -> list[Path]:
    tokens = []
    for part in payload.replace(";", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(Path(part))
    if not tokens:
        raise ValueError("Aucun chemin CSV fourni.")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit le panel Phase 1 à partir de CSV Supersearch.")
    parser.add_argument(
        "--csv-paths",
        required=True,
        help="Liste de CSV (séparateur virgule) contenant les résultats Supersearch.",
    )
    parser.add_argument("--max-strategies", type=int, default=5)
    parser.add_argument(
        "--min-pf",
        "--min-portfolio-pf",
        dest="min_pf",
        type=float,
        default=Phase1Filters.min_pf,
        help="PF portefeuille minimal (alias --min-portfolio-pf).",
    )
    parser.add_argument(
        "--min-sharpe",
        "--min-portfolio-sharpe",
        dest="min_sharpe",
        type=float,
        default=Phase1Filters.min_sharpe,
        help="Sharpe portefeuille minimal.",
    )
    parser.add_argument(
        "--max-dd",
        "--max-portfolio-dd",
        dest="max_dd",
        type=float,
        default=Phase1Filters.max_dd,
        help="Drawdown max portefeuille (ratio).",
    )
    parser.add_argument(
        "--max-ror",
        "--max-portfolio-ror",
        dest="max_ror",
        type=float,
        default=Phase1Filters.max_ror,
        help="Risque de ruine max portefeuille.",
    )
    parser.add_argument(
        "--min-trades",
        "--min-portfolio-trades",
        dest="min_trades",
        type=int,
        default=Phase1Filters.min_trades,
        help="Nombre de trades minimum.",
    )
    parser.add_argument(
        "--max-correlation",
        "--max-corr",
        dest="max_correlation",
        type=float,
        default=Phase1Filters.max_correlation,
        help="Corrélation maximale autorisée.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/PHASE1_PANEL.csv"),
        help="Fichier panel en sortie.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = _parse_csv_paths(args.csv_paths)
    filters = Phase1Filters(
        min_pf=args.min_pf,
        min_sharpe=args.min_sharpe,
        max_dd=args.max_dd,
        max_ror=args.max_ror,
        min_trades=args.min_trades,
        max_correlation=args.max_correlation,
    )
    diagnostics: dict[str, int] = {}
    panel = build_phase1_panel(csv_paths, filters, args.max_strategies, diagnostics=diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    total = diagnostics.get("total", 0)
    survivors = diagnostics.get("survivors", len(panel))
    selected = diagnostics.get("selected", len(panel))
    print(
        "[INFO] Phase 1 stats "
        f"(candidats={total}, survivants={survivors}, retenus={selected}, "
        f"PF<{filters.min_pf}: {diagnostics.get('filtered_pf', 0)}, "
        f"Sharpe<{filters.min_sharpe}: {diagnostics.get('filtered_sharpe', 0)}, "
        f"DD>{filters.max_dd:.2f}: {diagnostics.get('filtered_dd', 0)}, "
        f"RoR>{filters.max_ror:.2f}: {diagnostics.get('filtered_ror', 0)}, "
        f"Trades<{filters.min_trades}: {diagnostics.get('filtered_trades', 0)}, "
        f"Corr>{filters.max_correlation:.2f}: {diagnostics.get('filtered_correlation', 0)} )"
    )
    print(f"[OK] Panel Phase 1 construit ({len(panel)} stratégies) → {args.output}")


if __name__ == "__main__":
    main()
