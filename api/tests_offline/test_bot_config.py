from __future__ import annotations

import pytest

from bot.config import BotSettings


def test_bot_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord_test_token")
    monkeypatch.setenv("HYPRL_API_TOKEN", "hyprl_test_token")
    monkeypatch.delenv("HYPRL_API_BASE", raising=False)
    settings = BotSettings.from_env()
    assert settings.discord_token == "discord_test_token"
    assert settings.hyprl_api_token == "hyprl_test_token"
    assert settings.hyprl_api_base == "http://localhost:8000"


def test_bot_settings_missing_discord_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.setenv("HYPRL_API_TOKEN", "hyprl_test_token")
    with pytest.raises(RuntimeError) as exc:
        BotSettings.from_env()
    assert "DISCORD_BOT_TOKEN" in str(exc.value)


def test_bot_settings_missing_hyprl_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord_test_token")
    monkeypatch.delenv("HYPRL_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as exc:
        BotSettings.from_env()
    assert "HYPRL_API_TOKEN" in str(exc.value)
