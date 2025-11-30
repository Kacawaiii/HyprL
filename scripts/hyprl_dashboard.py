#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hyprl.portfolio.core import compute_portfolio_stats

SESSIONS_DIR = Path("data/live/sessions")


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def main() -> None:
    st.set_page_config(page_title="HyprL Paper Trading Dashboard", layout="wide")
    st.title("HyprL Paper Trading Dashboard")
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_ids = sorted([p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()])
    if not session_ids:
        st.warning("Aucune session enregistrée pour le moment.")
        return
    session_id = st.sidebar.selectbox("Session", session_ids, index=len(session_ids) - 1)
    session_dir = SESSIONS_DIR / session_id
    trades = _load_csv(session_dir / "trades.csv")
    equity = _load_csv(session_dir / "equity.csv")

    if equity is not None and not equity.empty:
        equity = equity.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        st.subheader("Courbe d'equity")
        st.line_chart(equity["equity"], use_container_width=True)
        stats = compute_portfolio_stats(equity["equity"], initial_balance=float(equity["equity"].iloc[0]))
        cols = st.columns(4)
        cols[0].metric("PnL %", f"{stats['return_pct']:.2f}%")
        cols[1].metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        cols[2].metric("Sharpe", f"{stats['sharpe']:.2f}")
        cols[3].metric("Max DD %", f"{stats['max_drawdown_pct']:.2f}%")
    else:
        st.info("Equity log vide pour cette session.")

    if trades is not None and not trades.empty:
        st.subheader("Trades récents")
        st.dataframe(trades.tail(20), height=400)
    else:
        st.info("Aucun trade enregistré.")

    st.sidebar.write(f"Logs: {session_dir}")


if __name__ == "__main__":
    main()
