"""Streamlit entrypoint for the HyprL portal."""

from __future__ import annotations

import streamlit as st

from portal import get_portal_client, get_portal_settings


def main() -> None:
    settings = get_portal_settings()
    _ = get_portal_client()
    st.set_page_config(page_title=settings.title, layout="wide")
    st.sidebar.title(settings.title)
    st.sidebar.write("HyprL V2 Portal")
    st.write("Cette app se lance avec `streamlit run portal/main.py`.")
    st.info("Utilisez le menu latéral Streamlit pour accéder aux pages Usage, Sessions, Autorank et Tokens.")


if __name__ == "__main__":
    main()
