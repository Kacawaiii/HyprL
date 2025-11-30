"""Reusable Streamlit layout helpers for the portal."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import streamlit as st

from portal.hyprl_client import HyprlAPIError

try:  # Streamlit distributions include pandas, but keep failure-safe fallback.
    import pandas as pd
except Exception:  # pragma: no cover - optional import
    pd = None


def show_usage_card(usage: dict) -> None:
    st.subheader("Crédits")
    col1, col2 = st.columns(2)
    col1.metric("Total", usage.get("credits_total", "—"))
    col2.metric("Restants", usage.get("credits_remaining", "—"))
    by_endpoint = _normalize_rows(usage.get("by_endpoint"))
    if by_endpoint:
        st.markdown("#### Consommation par endpoint")
        st.dataframe(by_endpoint, hide_index=True, use_container_width=True)


def show_session_status(status: dict) -> None:
    st.markdown("#### Statut de session")
    st.json(status)


def show_session_report(report: dict) -> None:
    st.markdown("#### Rapport détaillé")
    st.json(report)


def show_error(error: Exception | str) -> None:
    if isinstance(error, HyprlAPIError):
        payload = error.payload
        if isinstance(payload, (list, dict)):
            st.error({"status_code": error.status_code, "detail": payload})
        else:
            st.error(f"[{error.status_code}] {payload}")
    else:
        st.error(str(error))


def show_predict_table(history: list[dict]) -> None:
    if not history:
        st.info("Aucune prédiction enregistrée pour cette session.")
        return
    if pd is not None:
        df = pd.DataFrame(history)
        df = df.sort_values("ts", ascending=False)
        df = df.assign(timestamp=pd.to_datetime(df["ts"], unit="s"))
        columns = [
            col
            for col in [
                "timestamp",
                "symbol",
                "prob_up",
                "direction",
                "threshold",
                "risk_pct",
                "tp",
                "sl",
                "closed",
                "outcome",
                "pnl",
            ]
            if col in df.columns
        ]
        st.dataframe(df[columns], hide_index=True, use_container_width=True)
    else:  # pragma: no cover - pandas expected at runtime
        st.table(history)


def show_predict_charts(history: list[dict]) -> None:
    if not history:
        st.write("Aucune donnée pour les graphiques.")
        return
    if pd is not None:
        df = pd.DataFrame(history)
        df = df.assign(timestamp=pd.to_datetime(df["ts"], unit="s")).sort_values("timestamp")
        st.line_chart(df.set_index("timestamp")["prob_up"], height=200)
        direction_counts = df["direction"].str.upper().value_counts().rename_axis("direction").to_frame("count")
        st.bar_chart(direction_counts, height=200)
        winrate = float((df["direction"].str.upper() == "UP").mean())
        real_winrate: float | None = None
        pnl_chart = None
        if "pnl" in df.columns:
            pnl_chart = df[df["pnl"].notna()].copy()
        if pnl_chart is not None and not pnl_chart.empty:
            pnl_chart = pnl_chart.sort_values("timestamp")
            pnl_chart = pnl_chart.assign(pnl_cum=pnl_chart["pnl"].astype(float).cumsum())
            st.line_chart(pnl_chart.set_index("timestamp")["pnl_cum"], height=200)
            if "closed" in pnl_chart.columns:
                closed_df = pnl_chart[pnl_chart["closed"].fillna(False)]
            else:
                closed_df = pnl_chart
            if not closed_df.empty and "outcome" in closed_df.columns:
                real_winrate = float((closed_df["outcome"].str.upper() == "WIN").mean())
    else:  # pragma: no cover
        counts = Counter(item.get("direction", "?").upper() for item in history)
        st.write("Répartition directions:", dict(counts))
        total = sum(counts.values())
        winrate = (counts.get("UP", 0) / total) if total else 0.0
        real_winrate = None
    col1, col2 = st.columns(2)
    col1.metric("Winrate local (proxy)", f"{winrate:.1%}")
    if real_winrate is not None:
        col2.metric("Winrate réel (fermé)", f"{real_winrate:.1%}")
    else:
        col2.metric("Winrate réel (fermé)", "N/A")


def show_predict_summary(summary: dict | None) -> None:
    if not summary:
        st.info("Aucun résumé disponible pour le moment.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Prédictions totales", summary.get("total_predictions", "—"))
    col2.metric("Prédictions fermées", summary.get("closed_predictions", "—"))
    winrate = summary.get("winrate_real")
    if winrate is not None:
        col3.metric("Winrate réel", f"{winrate:.1%}")
    else:
        col3.metric("Winrate réel", "N/A")
    pnl_cols = st.columns(2)
    pnl_cols[0].metric("PnL total", f"{summary.get('pnl_total', 0.0):.2f}")
    avg = summary.get("avg_pnl")
    pnl_cols[1].metric("PnL moyen", f"{avg:.2f}" if avg is not None else "N/A")


def _normalize_rows(by_endpoint: Any) -> list[dict]:
    rows: list[dict] = []
    if isinstance(by_endpoint, dict):
        for endpoint, payload in by_endpoint.items():
            row = {"endpoint": endpoint}
            if isinstance(payload, dict):
                row.update(payload)
            else:
                row["value"] = payload
            rows.append(row)
    elif isinstance(by_endpoint, Iterable) and not isinstance(by_endpoint, (str, bytes)):
        for entry in by_endpoint:
            if isinstance(entry, dict):
                rows.append(entry)
    return rows


__all__ = [
    "show_error",
    "show_predict_charts",
    "show_predict_summary",
    "show_predict_table",
    "show_session_report",
    "show_session_status",
    "show_usage_card",
]
