"""Configuration helpers for the HyprL Streamlit portal."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class PortalSettings:
    api_base: str
    api_token: str
    title: str

    @classmethod
    def from_env(cls) -> "PortalSettings":
        api_base = os.getenv("HYPRL_API_BASE", "http://localhost:8000").rstrip("/")
        api_token = os.getenv("HYPRL_API_TOKEN")
        if not api_token:
            raise RuntimeError("HYPRL_API_TOKEN is required to start the portal")
        title = os.getenv("HYPRL_PORTAL_TITLE", "HyprL Control Panel")
        return cls(api_base=api_base, api_token=api_token, title=title)


__all__ = ["PortalSettings"]
