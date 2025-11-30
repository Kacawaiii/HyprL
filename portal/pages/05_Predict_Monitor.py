"""Predict Monitor page driving /v2/predict from the portal."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client, get_portal_settings
from portal.hyprl_client import HyprlAPIError
from portal.layout import show_error, show_predict_charts, show_predict_summary, show_predict_table
from portal.predict_monitor import append_predict_history, parse_optional_float, parse_symbols


st.title("Predict Monitor")
settings = get_portal_settings()
st.caption(f"API base: {settings.api_base}")

client = get_portal_client()
history: list[dict] = st.session_state.setdefault("predict_history", [])
summary_state = st.session_state.setdefault("predict_summary", None)


def _refresh_summary() -> None:
    try:
        summary = client.get_predict_summary()
        st.session_state["predict_summary"] = summary
    except HyprlAPIError as exc:
        show_error(exc)


if summary_state is None:
    try:
        summary_state = client.get_predict_summary()
        st.session_state["predict_summary"] = summary_state
    except HyprlAPIError:
        summary_state = None

with st.form("predict_form"):
    st.subheader("Nouvelle requête /v2/predict")
    symbols_input = st.text_input("Symboles (séparés par des virgules)", value="AAPL,MSFT")
    col1, col2 = st.columns(2)
    threshold_input = col1.text_input("Threshold (optionnel)", value="0.55")
    risk_pct_input = col2.text_input("Risk % (optionnel)", value="0.10")
    submitted = st.form_submit_button("Lancer Predict")

if submitted:
    try:
        symbols = parse_symbols(symbols_input)
        if not symbols:
            raise ValueError("Fournissez au moins un symbole.")
        threshold = parse_optional_float(threshold_input)
        risk_pct = parse_optional_float(risk_pct_input)
        response = client.predict(symbols, threshold=threshold, risk_pct=risk_pct)
        st.success("Requête /v2/predict envoyée.")
        history = append_predict_history(history, response)
        st.session_state["predict_history"] = history
        _refresh_summary()
    except (ValueError, HyprlAPIError) as exc:
        show_error(exc)

st.divider()
st.subheader("Résumé global des prédictions")
if st.button("Actualiser le résumé", type="secondary"):
    _refresh_summary()
summary_state = st.session_state.get("predict_summary")
show_predict_summary(summary_state)
st.divider()
st.subheader("Historique des prédictions")
show_predict_table(history)
st.divider()
show_predict_charts(history)
