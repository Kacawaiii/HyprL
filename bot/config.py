"""Configuration helpers for the HyprL Discord bot."""

from __future__ import annotations

import os
from dataclasses import dataclass

from portal.config import PortalSettings

@dataclass(slots=True)
class BotSettings:
    discord_token: str
    hyprl_api_base: str
    hyprl_api_token: str

    @classmethod
    def from_env(cls) -> "BotSettings":
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_token:
            raise RuntimeError("DISCORD_BOT_TOKEN is required to start the bot")
        portal_settings = PortalSettings.from_env()
        return cls(
            discord_token=discord_token,
            hyprl_api_base=portal_settings.api_base,
            hyprl_api_token=portal_settings.api_token,
        )


__all__ = ["BotSettings"]
