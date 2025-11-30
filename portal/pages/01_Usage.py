"""Usage / Crédits page."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client, get_portal_settings
from portal.hyprl_client import HyprlAPIError
from portal.layout import show_error, show_usage_card

st.title("Usage / Crédits")
st.caption(f"API base: {get_portal_settings().api_base}")

client = get_portal_client()

try:
    usage = client.get_usage()
    show_usage_card(usage)
except HyprlAPIError as exc:
    show_error(exc)
