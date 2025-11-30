"""Sessions management page."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client
from portal.hyprl_client import HyprlAPIError
from portal.layout import show_error, show_session_report, show_session_status

client = get_portal_client()

st.title("Sessions Paper")

with st.form("start_session_form"):
    st.subheader("Démarrer une session")
    symbols_input = st.text_input("Symbols (comma-separated)", value="AAPL,MSFT")
    interval = st.selectbox("Interval", options=["1m", "5m", "15m"], index=2)
    threshold = st.slider("Threshold", min_value=0.40, max_value=0.70, value=0.55, step=0.01)
    risk_pct = st.slider("Risk %", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
    kill_switch_dd = st.slider("Kill switch DD", min_value=0.10, max_value=0.50, value=0.30, step=0.01)
    enable_paper = st.checkbox("Paper trading mode", value=False)
    submitted = st.form_submit_button("Start session")
    if submitted:
        symbols = [sym.strip().upper() for sym in symbols_input.split(",") if sym.strip()]
        if not symbols:
            st.warning("Merci de renseigner au moins un symbole.")
        else:
            payload = {
                "symbols": symbols,
                "interval": interval,
                "threshold": threshold,
                "risk_pct": risk_pct,
                "kill_switch_dd": kill_switch_dd,
                "enable_paper": enable_paper,
            }
            try:
                response = client.start_session(payload)
                session_id = response.get("session_id", "inconnu")
                st.success(f"Session {session_id} démarrée.")
            except HyprlAPIError as exc:
                if exc.status_code == 402:
                    st.error("Crédits insuffisants pour démarrer une nouvelle session.")
                elif exc.status_code == 429:
                    st.error("Trop de requêtes. Merci de réessayer dans quelques secondes.")
                else:
                    show_error(exc)

st.subheader("Consulter une session")
session_id = st.text_input("Session ID", value="", help="Identifiant retourné par /v2/sessions")
col_status, col_report = st.columns(2)

if col_status.button("Status"):
    if not session_id.strip():
        st.warning("Indiquez un Session ID.")
    else:
        try:
            status = client.get_session_status(session_id.strip())
            show_session_status(status)
        except HyprlAPIError as exc:
            show_error(exc)

if col_report.button("Report"):
    if not session_id.strip():
        st.warning("Indiquez un Session ID.")
    else:
        try:
            report = client.get_session_report(session_id.strip())
            show_session_report(report)
        except HyprlAPIError as exc:
            show_error(exc)
