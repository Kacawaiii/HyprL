"""Autorank management page."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client
from portal.hyprl_client import HyprlAPIError
from portal.layout import show_error

client = get_portal_client()

st.title("Autorank + Sessions")

with st.form("autorank_start_form"):
    st.subheader("Lancer un Autorank")
    csv_paths_input = st.text_input(
        "CSV paths",
        value="data/experiments/supersearch_portfolio_AAPL_MSFT_1y.csv",
        help="Liste séparée par des virgules.",
    )
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)
    dry_run = st.checkbox("Dry run", value=False)
    meta_weight = st.slider("Meta weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    constraints_col = st.container()
    with constraints_col:
        st.caption("Contraintes minimales")
        min_pf = st.number_input("Min PF", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        max_dd = st.number_input("Max DD", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        min_trades = st.number_input("Min trades", min_value=0, max_value=1000, value=0, step=10)
    autorank_submitted = st.form_submit_button("Start autorank")
    if autorank_submitted:
        csv_paths = [path.strip() for path in csv_paths_input.split(",") if path.strip()]
        if not csv_paths:
            st.warning("Merci de renseigner au moins un CSV.")
        else:
            payload = {
                "csv_paths": csv_paths,
                "top_k": int(top_k),
                "dry_run": dry_run,
                "meta_weight": meta_weight,
                "constraints": {
                    "min_pf": min_pf,
                    "max_dd": max_dd,
                    "min_trades": int(min_trades),
                },
            }
            try:
                response = client.start_autorank(payload)
                autorank_id = response.get("autorank_id", "inconnu")
                sessions = response.get("sessions", [])
                st.success(f"Autorank {autorank_id} lancé ({len(sessions)} sessions).")
                if sessions:
                    st.dataframe(sessions, use_container_width=True)
                artifacts = response.get("artifacts", [])
                if artifacts:
                    st.write("Artefacts:", artifacts)
            except HyprlAPIError as exc:
                show_error(exc)

st.subheader("Consulter un Autorank")
autorank_id_input = st.text_input("Autorank ID", value="")
if st.button("Fetch autorank status"):
    autorank_id = autorank_id_input.strip()
    if not autorank_id:
        st.warning("Indiquez un Autorank ID.")
    else:
        try:
            status = client.get_autorank_status(autorank_id)
            st.json(status)
            sessions = status.get("sessions") or []
            if sessions:
                st.dataframe(sessions, use_container_width=True)
        except HyprlAPIError as exc:
            show_error(exc)
