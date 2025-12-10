#!/usr/bin/env python3
"""
Lightweight dashboard for Palier 2 live-lite.

Reads:
    - live/logs/portfolio_live/health_asc_v2.json (PF/DD/Sharpe/status)
    - live/logs/live_<symbol>/heartbeat.json per ticker
    - (optional) live/logs/portfolio_live/equity_portfolio.csv for equity curve

Usage:
    streamlit run scripts/dashboard/palier2_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_heartbeats(root: Path) -> List[Dict[str, Any]]:
    hb_paths = sorted(root.glob("live_*/heartbeat.json"))
    out: List[Dict[str, Any]] = []
    for path in hb_paths:
        data = _load_json(path)
        ticker = path.parent.name.replace("live_", "").upper()
        if not data:
            out.append({"ticker": ticker, "timestamp": None, "age_min": None})
            continue
        ts_str = data.get("ts_iso") or data.get("timestamp") or data.get("ts")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else None
        except Exception:
            ts = None
        age_min = None
        if ts:
            age_min = (datetime.utcnow() - ts.replace(tzinfo=None)).total_seconds() / 60.0
        out.append({"ticker": ticker, "timestamp": ts_str, "age_min": age_min})
    return out


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        raise SystemExit("Streamlit is required. Install with `pip install streamlit`.")

    st.set_page_config(page_title="HyprL Asc v2 – Palier 2 Dashboard", layout="wide")

    log_root = Path("live/logs")
    health_path = log_root / "portfolio_live" / "health_asc_v2.json"
    health = _load_json(health_path) or {}

    st.header("Ascendant v2 – Portfolio Health")
    cols = st.columns(4)
    cols[0].metric("Status", health.get("status", "UNKNOWN"))
    cols[1].metric("PF", f"{health.get('pf'):.3f}" if health.get("pf") is not None else "N/A")
    cols[2].metric("MaxDD%", f"{health.get('maxdd'):.2f}%" if health.get("maxdd") is not None else "N/A")
    cols[3].metric("Sharpe", f"{health.get('sharpe'):.2f}" if health.get("sharpe") is not None else "N/A")
    st.write(f"Trades: {health.get('trades', 'N/A')}  |  As of: {health.get('as_of', 'N/A')}")

    st.subheader("Heartbeats")
    heartbeats = _load_heartbeats(log_root)
    st.table(
        [
            {
                "Ticker": hb["ticker"],
                "Timestamp": hb["timestamp"],
                "Age (min)": f"{hb['age_min']:.1f}" if hb["age_min"] is not None else "N/A",
            }
            for hb in heartbeats
        ]
    )

    equity_csv = log_root / "portfolio_live" / "equity_portfolio.csv"
    if equity_csv.is_file():
        import pandas as pd

        try:
            df = pd.read_csv(equity_csv, parse_dates=["timestamp"])
            st.subheader("Equity Curve (Portfolio)")
            st.line_chart(df.set_index("timestamp")[["equity_portfolio"]])
        except Exception as exc:
            st.warning(f"Failed to load equity curve: {exc}")
    else:
        st.info("No equity_portfolio.csv found; skipping equity chart.")


if __name__ == "__main__":
    main()
