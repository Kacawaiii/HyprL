#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.autorank import (
    AutorankConstraints,
    apply_autorank_filters,
    build_phase1_shortlist,
    load_meta_info,
)
from hyprl.meta.model import MetaRobustnessModel
from hyprl.meta.registry import get_registry_snapshot, resolve_model
from hyprl.phase1.paper import Phase1ExecutionConfig, run_phase1_sessions
from hyprl.phase1.results import build_phase1_results
from hyprl.utils.provenance import hash_file, save_dataframe_with_meta, stamp_provenance


def _parse_csvs(payload: str | None) -> list[Path]:
    if not payload:
        return []
    tokens = [item.strip() for item in payload.replace(";", ",").split(",") if item.strip()]
    return [Path(token) for token in tokens]


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["tickers"] = frame.get("tickers", "").fillna("")
    frame["config_index"] = frame.get("config_index", frame.index)
    return frame


def _resolve_artifact(path_arg: str | None, registry_arg: str | None) -> tuple[Path | None, str | None]:
    if path_arg:
        candidate = Path(path_arg)
        if not candidate.exists():
            raise SystemExit(f"Artifact introuvable: {candidate}")
        return candidate, None
    if registry_arg:
        return Path(resolve_model(registry_arg)), registry_arg
    return None, None


def _load_meta_payload(model_path: Path | None, registry_ref: str | None) -> tuple[MetaRobustnessModel | None, dict | None]:
    if not model_path:
        return None, None
    model = MetaRobustnessModel.load(model_path)
    meta = load_meta_info(model_path)
    meta["path"] = str(model_path)
    if registry_ref:
        key, selector = registry_ref.split("@", 1)
        meta["registry_key"] = key
        meta["selector"] = selector
    return model, meta


def _load_calibrator_payload(path: Path | None, registry_ref: str | None) -> tuple[object | None, dict | None]:
    if not path:
        return None, None
    calibrator = joblib.load(path)
    meta_path = path.with_name("calibration_meta.json")
    meta = {"path": str(path)}
    if meta_path.exists():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    if registry_ref:
        key, selector = registry_ref.split("@", 1)
        meta["registry_key"] = key
        meta["selector"] = selector
    return calibrator, meta


def _default_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"docs/experiments/RUN_{ts}")


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - git absent
        return None
    return result.stdout.strip()


def _collect_dataset_hashes(paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.exists():
            hashes[str(path)] = hash_file(path)
    return hashes


def run_workflow(args: argparse.Namespace) -> dict[str, Path]:
    if not args.autoranked and not args.csv:
        raise SystemExit("Spécifiez --autoranked ou --csv.")
    outdir = Path(args.outdir) if args.outdir else _default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    model_path, model_ref = _resolve_artifact(args.meta_robustness, args.meta_registry)
    meta_model, meta_context = _load_meta_payload(model_path, model_ref)
    calibrator_path, calibrator_ref = _resolve_artifact(args.meta_calibration, args.meta_calibration_registry)
    calibrator, calibrator_meta = _load_calibrator_payload(calibrator_path, calibrator_ref)
    if calibrator is None:
        print("[WARN] Meta calibrator absent – workflow uncalibrated.")

    data_inputs: list[Path] = []
    if args.autoranked:
        autoranked_path = Path(args.autoranked)
        if not autoranked_path.exists():
            raise SystemExit(f"CSV autoranked introuvable: {autoranked_path}")
        diag_raw = pd.read_csv(autoranked_path)
        diag = _prepare_frame(diag_raw)
        if "base_score_normalized" not in diag.columns:
            if "final_score" in diag.columns:
                diag["base_score_normalized"] = pd.to_numeric(diag["final_score"], errors="coerce").fillna(0.0)
            else:
                diag["base_score_normalized"] = 1.0
        data_inputs.append(autoranked_path)
    else:
        csv_paths = _parse_csvs(args.csv)
        if not csv_paths:
            raise SystemExit("Aucun CSV fourni via --csv.")
        frames: list[pd.DataFrame] = []
        for path in csv_paths:
            if not path.exists():
                raise SystemExit(f"CSV introuvable: {path}")
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            frames.append(df)
            data_inputs.append(path)
        merged = pd.concat(frames, ignore_index=True)
        diag = build_meta_diag_frame(merged, meta_weight=args.meta_weight, model=meta_model, calibrator=calibrator)
    diag = _prepare_frame(diag)

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
    if filtered_diag.empty:
        raise SystemExit("Aucune stratégie ne satisfait les contraintes Phase 1.")

    panel = build_phase1_shortlist(filtered_diag, max_strategies=args.max_strategies)
    panel_path = outdir / "PHASE1_PANEL.csv"
    save_dataframe_with_meta(panel, panel_path, key_columns=["strat_id"])

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
    sessions_df = run_phase1_sessions(panel, exec_cfg)
    sessions_path = outdir / "PHASE1_SESSIONS.csv"
    save_dataframe_with_meta(sessions_df, sessions_path, key_columns=["session_id"])

    results_df = build_phase1_results(sessions_df)
    results_path = outdir / "phase1_results.csv"
    save_dataframe_with_meta(results_df, results_path, key_columns=["session_id"])

    summary_path = outdir / "SUMMARY.txt"
    top_result = results_df.sort_values("robustness_score", ascending=False).head(1)
    best_line = "N/A"
    if not top_result.empty:
        row = top_result.iloc[0]
        best_line = f"best={row['session_id']} robustness={row['robustness_score']:.3f} PF_live={row['pf_live']:.2f}"
    summary_lines = [
        f"Input rows={len(diag)}",
        f"Survivors={len(filtered_diag)} filters={stats}",
        f"Panel strategies={len(panel)}",
        f"Sessions={len(sessions_df)}",
        f"Results rows={len(results_df)} {best_line}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    dataset_hashes = _collect_dataset_hashes(set(data_inputs))
    dataset_hashes[str(panel_path)] = hash_file(panel_path)
    dataset_hashes[str(sessions_path)] = hash_file(sessions_path)
    dataset_hashes[str(results_path)] = hash_file(results_path)

    registry_keys = []
    if model_ref:
        registry_keys.append(model_ref.split("@", 1)[0])
    if calibrator_ref:
        registry_keys.append(calibrator_ref.split("@", 1)[0])
    registry_snapshot = get_registry_snapshot(registry_keys) if registry_keys else None
    provenance_path = stamp_provenance(
        outdir,
        meta_model=meta_context,
        calibrator_meta=calibrator_meta,
        registry_snapshot=registry_snapshot,
        cli_args=vars(args),
        dataset_hashes=dataset_hashes,
        git_hash=_git_hash(),
    )

    return {
        "outdir": outdir,
        "panel": panel_path,
        "sessions": sessions_path,
        "results": results_path,
        "summary": summary_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrateur Phase 1 depuis un autorank ou CSV Supersearch.")
    parser.add_argument("--autoranked", help="CSV autoranked existant.")
    parser.add_argument("--csv", help="CSV Supersearch (séparés par virgule) si pas d'autoranked.")
    parser.add_argument("--meta-robustness", help="Modèle Meta-ML (.joblib).")
    parser.add_argument("--meta-registry", help="Clé@Stage ou alias dans le registry Meta-ML.")
    parser.add_argument("--meta-calibration", help="Calibrateur joblib.")
    parser.add_argument("--meta-calibration-registry", help="Clé@Stage ou alias pour le calibrateur.")
    parser.add_argument("--meta-weight", type=float, default=0.4)
    parser.add_argument("--min-portfolio-pf", type=float, default=0.0)
    parser.add_argument("--min-portfolio-sharpe", type=float, default=-10.0)
    parser.add_argument("--max-portfolio-dd", type=float, default=1.0)
    parser.add_argument("--max-corr", type=float, default=1.0)
    parser.add_argument("--min-trades", type=int, default=0)
    parser.add_argument("--min-weight-per-ticker", type=float)
    parser.add_argument("--max-weight-per-ticker", type=float)
    parser.add_argument("--max-strategies", type=int, default=5)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--engine", choices=["auto", "python", "native"], default="auto")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--calibration", default="none")
    parser.add_argument("--default-long-threshold", type=float, default=0.6)
    parser.add_argument("--default-short-threshold", type=float, default=0.4)
    parser.add_argument("--session-prefix", default="phase1")
    parser.add_argument("--outdir", help="Répertoire de sortie (défaut docs/experiments/RUN_<ts>).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(args)


if __name__ == "__main__":
    main()
