from typing import Optional
import streamlit as st
from pathlib import Path
from apps.track_record.lib.constants import ROOT_DIR

ENTITLEMENTS_FILE = ROOT_DIR / "apps/track_record/entitlements_allowlist.txt"

def check_access(api_key: Optional[str]) -> bool:
    """Checks if the provided API key is valid for private access."""
    if not api_key:
        return False
    
    if not ENTITLEMENTS_FILE.exists():
        # If no entitlements file, default to no private access (or could be all public)
        return False
    
    allowed_keys = [line.strip() for line in ENTITLEMENTS_FILE.read_text().splitlines() if line.strip()]
    return api_key in allowed_keys

def get_current_user_level() -> str:
    """Returns 'public' or 'private' based on session state."""
    # Check query params first
    query_params = st.query_params
    key = query_params.get("key", None)
    
    if check_access(key):
        return "private"
    return "public"
