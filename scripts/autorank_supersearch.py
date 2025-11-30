#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.autorank import (
    AutorankConstraints,
    apply_autorank_filters,
    build_phase1_shortlist,
    load_meta_info,
    write_summary,
)
from hyprl.meta.model import MetaRobustnessModel
from hyprl.meta.registry import resolve_model


def _parse_csvs(payload: str) -> list[Path]:
    parts = [item.strip() for item in payload.replace(";", ",").split(",") if item.strip()]
    if not parts:
        raise ValueError("Aucun CSV fourni.")
    return [Path(part) for part in parts]


def _default_output_path(csv_paths: list[Path]) -> tuple[Path, Path]:
    first = csv_paths[0]
    if len(csv_paths) == 1:
        base = first.with_suffix("")
        csv_out = base.with_name(base.name + "_autoranked.csv")
    else:
        csv_out = first.parent / "autoranked_combined.csv"
    summary_out = csv_out.with_suffix(".SUMMARY.txt")
    return csv_out, summary_out


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["tickers"] = frame.get("tickers", "").fillna("")
    frame["config_index"] = frame.get("config_index", frame.index)
    return frame


def _resolve_artifact(path: Optional[str], registry_key: Optional[str]) -> Optional[Path]:
    candidate = path.strip() if path else None
    if candidate:
        resolved = Path(candidate)
        if resolved.exists():
            return resolved
        raise SystemExit(f"Artifact introuvable: {candidate}")
    if registry_key:
        return resolve_model(registry_key)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autorank Supersearch CSVs with Meta-ML + constraints.")
    parser.add_argument("--csv", required=True, help="Chemins CSV supersearch (séparés par virgule).")
    parser.add_argument("--meta-robustness", help="Modèle Meta-ML (model.joblib).")
    parser.add_argument("--meta-registry", help="Clé@Stage ou alias du registry si --meta-robustness absent.")
    parser.add_argument("--meta-calibration", help="Calibrateur joblib optionnel.")
    parser.add_argument("--meta-calibration-registry", help="Clé@Stage ou alias pour le calibrateur.")
    parser.add_argument("--meta-weight", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", help="CSV autoranké (défaut: suffixe _autoranked).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-portfolio-pf", type=float, default=0.0)
    parser.add_argument("--min-portfolio-sharpe", type=float, default=-10.0)
    parser.add_argument("--max-portfolio-dd", type=float, default=1.0, help="Drawdown max (ratio, ex 0.35).")
    parser.add_argument("--max-corr", type=float, default=1.0)
    parser.add_argument("--min-trades", type=int, default=0)
    parser.add_argument("--min-weight-per-ticker", type=float)
    parser.add_argument("--max-weight-per-ticker", type=float)
    parser.add_argument("--emit-phase1-panel", type=Path, help="Chemin de sortie PHASE1_PANEL.csv auto-généré.")
    parser.add_argument("--max-strategies", type=int, default=5, help="Nb max stratégies dans le panel Phase 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = _parse_csvs(args.csv)
    model_path = _resolve_artifact(args.meta_robustness, args.meta_registry)
    if model_path is None:
        raise SystemExit("Spécifiez --meta-robustness ou --meta-registry.")
    model = MetaRobustnessModel.load(model_path)
    meta_info = load_meta_info(model_path)

    calibrator: object | None = None
    calibrator_path = _resolve_artifact(args.meta_calibration, args.meta_calibration_registry)
    if calibrator_path:
        calibrator = joblib.load(calibrator_path)

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise SystemExit(f"CSV introuvable: {csv_path}")
        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)
        frames.append(df)
    if not frames:
        raise SystemExit("Aucun CSV valide fourni.")
    full_df = pd.concat(frames, ignore_index=True)
    full_df = _prepare_frame(full_df)

    diag = build_meta_diag_frame(
        full_df,
        meta_weight=float(args.meta_weight),
        model=model,
        calibrator=calibrator,
    )
    diag["meta_model_type"] = meta_info.get("meta_model_type", "")
    diag["meta_dataset_hash"] = meta_info.get("meta_dataset_hash", "")
    diag["meta_trained_at"] = meta_info.get("meta_trained_at", "")
    diag["meta_calibrated"] = bool(calibrator is not None)

    constraints = AutorankConstraints(
        min_pf=args.min_portfolio_pf,
        min_sharpe=args.min_portfolio_sharpe,
        max_dd=args.max_portfolio_dd,
        max_corr=args.max_corr,
        min_trades=args.min_trades,
        min_weight=args.min_weight_per_ticker,
        max_weight=args.max_weight_per_ticker,
    )
    filtered_diag, stats = apply_autorank_filters(diag, constraints)

    rng = np.random.default_rng(args.seed)
    filtered_diag["_tie"] = rng.random(len(filtered_diag))
    filtered_diag = filtered_diag.sort_values(
        ["final_score", "base_score_normalized", "_tie"],
        ascending=[False, False, True],
        kind="mergesort",
    ).drop(columns=["_tie"])

    out_csv, summary_path = (
        _default_output_path(csv_paths)
        if not args.out
        else (Path(args.out), Path(args.out).with_suffix(".SUMMARY.txt"))
    )
    filtered_diag.to_csv(out_csv, index=False)
    write_summary(
        summary_path,
        model_path,
        meta_info,
        filtered_diag,
        float(args.meta_weight),
        args.seed,
        args.topk,
        filters=stats,
    )

    top_rows = filtered_diag.head(args.topk)
    print(
        f"[INFO] Autoranking terminé (candidats={stats.get('total', len(diag))}, "
        f"survivants={len(filtered_diag)}) → {out_csv}"
    )
    pf_limit = constraints.min_pf if constraints.min_pf is not None else 0.0
    sharpe_limit = constraints.min_sharpe if constraints.min_sharpe is not None else -10.0
    dd_limit = constraints.max_dd if constraints.max_dd is not None else 1.0
    corr_limit = constraints.max_corr if constraints.max_corr is not None else 1.0
    trade_limit = constraints.min_trades if constraints.min_trades is not None else 0
    print(
        "[INFO] Filtres: "
        f"pf<{pf_limit}: {stats.get('filtered_by_pf', 0)}, "
        f"sharpe<{sharpe_limit}: {stats.get('filtered_by_sharpe', 0)}, "
        f"dd>{dd_limit:.2f}: {stats.get('filtered_by_dd', 0)}, "
        f"corr>{corr_limit:.2f}: {stats.get('filtered_by_corr', 0)}, "
        f"trades<{trade_limit}: {stats.get('filtered_by_trades', 0)}, "
        f"w<min: {stats.get('filtered_by_weight_min', 0)}, "
        f"w>max: {stats.get('filtered_by_weight_max', 0)}"
    )
    for idx, row in enumerate(top_rows.itertuples(), start=1):
        print(
            f"{idx:02d}. {row.tickers} cfg={row.config_index} "
            f"base={row.base_score_normalized:.3f} meta={row.meta_pred:.3f} final={row.final_score:.3f}"
        )
    print(f"[INFO] Résumé écrit dans {summary_path}")

    if args.emit_phase1_panel:
        panel = build_phase1_shortlist(filtered_diag, max_strategies=args.max_strategies)
        args.emit_phase1_panel.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.emit_phase1_panel, index=False)
        print(
            f"[INFO] Panel Phase 1 ({len(panel)} strat.) → {args.emit_phase1_panel}"
        )


if __name__ == "__main__":
    main()
