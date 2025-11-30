"""Entry point for the HyprL Discord bot."""

from __future__ import annotations

import asyncio

import nextcord
from nextcord.ext import commands

from bot.config import BotSettings
from bot.hyprl_client import HyprlClient
from bot.commands import autorank, basic, predict, sessions, usage


async def create_bot() -> commands.Bot:
    intents = nextcord.Intents.default()
    bot = commands.Bot(intents=intents)
    settings = BotSettings.from_env()
    bot.settings = settings  # type: ignore[attr-defined]
    bot.hyprl_client = HyprlClient(settings)  # type: ignore[attr-defined]

    @bot.event
    async def on_ready():
        print(f"HyprL Discord bot connected as {bot.user}")

    # Register cogs
    basic.setup(bot)
    usage.setup(bot)
    predict.setup(bot)
    sessions.setup(bot)
    autorank.setup(bot)
    return bot


async def main() -> None:
    bot = await create_bot()
    try:
        await bot.start(bot.settings.discord_token)  # type: ignore[attr-defined]
    finally:
        await bot.hyprl_client.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
