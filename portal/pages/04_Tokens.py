"""Admin token management page."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client
from portal.hyprl_client import HyprlAPIError
from portal.layout import show_error

client = get_portal_client()

st.title("Tokens API")

with st.form("create_token_form"):
    st.subheader("Créer un token")
    account_id = st.text_input("Account ID")
    scopes = st.multiselect(
        "Scopes",
        options=["read:predict", "read:usage", "write:session", "admin:*"],
        default=["read:usage"],
    )
    credits_total = st.number_input("Credits total", min_value=0, max_value=1_000_000, value=1000, step=100)
    label = st.text_input("Label (optionnel)", value="")
    create_submitted = st.form_submit_button("Create token")
    if create_submitted:
        if not account_id:
            st.warning("Account ID requis.")
        elif not scopes:
            st.warning("Sélectionnez au moins un scope.")
        else:
            payload = {
                "account_id": account_id,
                "scopes": scopes,
                "credits_total": int(credits_total),
            }
            if label.strip():
                payload["label"] = label.strip()
            try:
                response = client.create_token(payload)
                token_id = response.get("token_id", "inconnu")
                token_plain = response.get("token_plain")
                st.success(f"Token {token_id} créé.")
                if token_plain:
                    st.info("Copiez ce token maintenant, il ne sera plus affiché.")
                    st.code(token_plain)
            except HyprlAPIError as exc:
                show_error(exc)

st.subheader("Révoquer un token")
token_id_revoke = st.text_input("Token ID à révoquer", key="revoke_token_id")
if st.button("Revoke token"):
    if not token_id_revoke.strip():
        st.warning("Indiquez un token ID.")
    else:
        try:
            client.revoke_token(token_id_revoke.strip())
            st.success(f"Token {token_id_revoke.strip()} révoqué.")
        except HyprlAPIError as exc:
            show_error(exc)
