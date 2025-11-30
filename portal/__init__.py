"""Portal helpers for sharing settings and client instances."""

from __future__ import annotations

from portal.config import PortalSettings
from portal.hyprl_client import PortalHyprlClient

_SETTINGS_CACHE: PortalSettings | None = None
_CLIENT_CACHE: PortalHyprlClient | None = None


def get_portal_settings() -> PortalSettings:
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = PortalSettings.from_env()
    return _SETTINGS_CACHE


def get_portal_client() -> PortalHyprlClient:
    global _CLIENT_CACHE
    if _CLIENT_CACHE is None:
        _CLIENT_CACHE = PortalHyprlClient(get_portal_settings())
    return _CLIENT_CACHE


__all__ = ["get_portal_settings", "get_portal_client"]
