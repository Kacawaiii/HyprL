from __future__ import annotations

import os

import pytest

from portal.config import PortalSettings


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPRL_API_BASE", "http://testserver")
    monkeypatch.setenv("HYPRL_API_TOKEN", "secret-token")
    monkeypatch.setenv("HYPRL_PORTAL_TITLE", "Test Portal")
    settings = PortalSettings.from_env()
    assert settings.api_base == "http://testserver"
    assert settings.api_token == "secret-token"
    assert settings.title == "Test Portal"


def test_settings_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPRL_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        PortalSettings.from_env()
