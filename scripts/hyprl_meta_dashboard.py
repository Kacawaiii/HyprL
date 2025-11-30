#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.autorank import (
    AutorankConstraints,
    apply_autorank_filters,
    build_phase1_shortlist,
    load_meta_info,
    write_summary,
)
from hyprl.meta.model import MetaRobustnessModel


def _list_csv_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.glob("*.csv") if path.is_file()])


def _load_csv_paths(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    st.set_page_config(page_title="HyprL Meta Ranking Dashboard", layout="wide")
    st.title("HyprL Meta-Ranking Dashboard")

    data_mode = st.sidebar.radio("Source CSV", ["Répertoire", "Upload"], index=0)
    data_frames: list[pd.DataFrame] = []
    if data_mode == "Répertoire":
        directory = st.sidebar.text_input("Répertoire data", "data")
        available = _list_csv_files(Path(directory))
        if available:
            selected = st.sidebar.multiselect(
                "Fichiers Supersearch",
                options=[str(p) for p in available],
                default=[str(available[0])],
            )
            data_frames.append(_load_csv_paths([Path(p) for p in selected]))
    else:
        uploads = st.sidebar.file_uploader("Uploader CSV Supersearch", accept_multiple_files=True, type=["csv"])
        if uploads:
            for uploaded in uploads:
                df = pd.read_csv(uploaded)
                df["source_csv"] = uploaded.name
                data_frames.append(df)

    if not data_frames:
        st.info("Sélectionnez au moins un CSV Supersearch.")
        return
    raw_df = pd.concat(data_frames, ignore_index=True)

    meta_model_path = st.sidebar.text_input(
        "Meta model (.joblib)",
        "artifacts/meta_ml/robustness_v0/model.joblib",
    )
    meta_weight = st.sidebar.slider("Meta weight", 0.0, 1.0, 0.4, 0.05)
    meta_model = None
    meta_loaded = False
    if meta_model_path:
        path = Path(meta_model_path)
        if path.exists():
            try:
                meta_model = MetaRobustnessModel.load(path)
                meta_loaded = True
                st.sidebar.success("Modèle Meta chargé.")
            except Exception as exc:  # pragma: no cover
                st.sidebar.error(f"Échec chargement modèle: {exc}")
        else:
            st.sidebar.warning("Modèle introuvable.")

    calibrator_path = st.sidebar.text_input("Calibrator (.joblib)", "")
    calibrator = None
    if calibrator_path:
        cal_path = Path(calibrator_path)
        if cal_path.exists():
            try:
                calibrator = joblib.load(cal_path)
                st.sidebar.success("Calibrateur chargé.")
            except Exception as exc:  # pragma: no cover
                st.sidebar.error(f"Échec chargement calibrateur: {exc}")
        else:
            st.sidebar.warning("Calibrateur introuvable.")

    diag = build_meta_diag_frame(raw_df, meta_weight=meta_weight, model=meta_model, calibrator=calibrator)
    diag["meta_calibrated"] = bool(calibrator is not None)

    st.subheader("Filtres")
    min_trades = st.number_input("Trades minimum", min_value=0, value=30)
    min_pf = st.number_input("PF min", min_value=0.0, value=1.0)
    min_sharpe = st.number_input("Sharpe min", min_value=-2.0, value=0.3)
    max_dd = st.number_input("Max drawdown (%)", min_value=0.0, value=40.0)
    max_corr = st.number_input("Max corr", min_value=0.0, value=0.9)
    min_weight = st.number_input("Min weight par ticker", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    max_weight = st.number_input("Max weight par ticker", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    constraints = AutorankConstraints(
        min_pf=min_pf,
        min_sharpe=min_sharpe,
        max_dd=max_dd / 100.0,
        max_corr=max_corr,
        min_trades=min_trades,
        min_weight=min_weight if min_weight > 0.0 else None,
        max_weight=max_weight if max_weight < 1.0 else None,
    )
    filtered, stats = apply_autorank_filters(diag, constraints)

    if calibrator is not None:
        st.markdown("**Meta predictions** · :green[Calibrated]")
    else:
        st.markdown("**Meta predictions** · :gray[Raw]")

    st.caption(
        "Filtrage: "
        f"pf<{constraints.min_pf}: {stats.get('filtered_by_pf', 0)} · "
        f"sharpe<{constraints.min_sharpe}: {stats.get('filtered_by_sharpe', 0)} · "
        f"dd>{constraints.max_dd or 0:.2f}: {stats.get('filtered_by_dd', 0)} · "
        f"corr>{constraints.max_corr or 0:.2f}: {stats.get('filtered_by_corr', 0)} · "
        f"trades<{constraints.min_trades or 0}: {stats.get('filtered_by_trades', 0)}"
    )

    st.subheader("Scores Meta vs Base")
    scatter = (
        alt.Chart(filtered)
        .mark_circle()
        .encode(
            x=alt.X("base_score_normalized", title="Base score (normalisé)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("meta_pred", title="Meta prediction", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("weighting_scheme", title="Poids"),
            size=alt.Size("trades_backtest", title="Trades", scale=alt.Scale(range=[50, 500])),
            tooltip=[
                "tickers",
                "config_index",
                "long_threshold",
                "short_threshold",
                "risk_pct",
                "base_score_normalized",
                "meta_pred",
                "final_score",
            ],
        )
    )
    st.altair_chart(scatter, use_container_width=True)

    topk = st.slider("Top K", 5, 30, 10)
    top_df = filtered.sort_values("final_score", ascending=False).head(topk)
    bar = alt.Chart(top_df).mark_bar().encode(
        x=alt.X("final_score", title="Final score"),
        y=alt.Y("tickers", sort="-x"),
        color="weighting_scheme",
        tooltip=["tickers", "config_index", "final_score", "base_score_normalized", "meta_pred"],
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("Tableau des stratégies")
    display_cols = [
        "tickers",
        "config_index",
        "long_threshold",
        "short_threshold",
        "risk_pct",
        "weighting_scheme",
        "trades_backtest",
        "portfolio_pf",
        "portfolio_sharpe",
        "portfolio_dd",
        "corr_max",
        "base_score_normalized",
        "meta_pred",
        "final_score",
        "score_delta",
        "delta_flag",
    ]
    st.dataframe(top_df[display_cols])

    export_path = st.text_input("Export autoranked (.csv)", "data/meta_autoranked.csv")
    if st.button("Exporter autoranked") and not filtered.empty:
        export_csv = Path(export_path)
        export_df = filtered.sort_values("final_score", ascending=False)
        export_df.to_csv(export_csv, index=False)
        meta_info = load_meta_info(Path(meta_model_path)) if meta_loaded else {}
        summary_path = export_csv.with_suffix(".SUMMARY.txt")
        write_summary(
            summary_path,
            Path(meta_model_path) if meta_loaded else None,
            meta_info,
            export_df,
            meta_weight,
            seed=0,
            topk=10,
            filters=stats,
        )
        st.success(f"Exporté sous {export_csv} et {summary_path}.")

    st.subheader("Phase 1 Auto-Shortlist")
    panel_path = st.text_input("Chemin PHASE1_PANEL.csv", "docs/experiments/PHASE1_PANEL_AUTO.csv")
    max_strats = st.slider("Stratégies max", 1, 20, 5)
    if st.button("Générer panel Phase 1") and not filtered.empty:
        panel = build_phase1_shortlist(filtered, max_strats)
        panel_csv = Path(panel_path)
        panel_csv.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(panel_csv, index=False)
        st.success(f"Panel Phase 1 ({len(panel)} strat.) → {panel_csv}")


if __name__ == "__main__":
    main()
