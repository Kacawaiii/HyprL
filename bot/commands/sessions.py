"""Realtime API session commands + Discord channel session management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import nextcord
from nextcord.abc import Messageable
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError
from bot.session_store import (
    SESSION_STORE_PATH,
    get_session_channels,
    get_sessions,
    resolve_session,
    save_sessions,
    slugify,
)

logger = logging.getLogger(__name__)
bot_ref: commands.Bot | None = None


def _split_symbols(raw: str) -> list[str]:
    return [token.strip().upper() for token in raw.replace(";", ",").split(",") if token.strip()]


def _format_session_summary(slug: str, session: dict[str, Any]) -> str:
    channels = get_session_channels(session)
    name = session.get("name", slug)
    cat = session.get("category_id", "n/a")
    return (
        f"{name} ({slug})\n"
        f"  category: {cat}\n"
        f"  overview: {channels.get('overview', 'n/a')}\n"
        f"  alerts:   {channels.get('alerts', 'n/a')}\n"
        f"  trades:   {channels.get('trades', 'n/a')}"
    )


async def _load_sessions() -> dict[str, dict[str, Any]]:
    return await asyncio.to_thread(get_sessions, SESSION_STORE_PATH)


async def _persist_sessions(data: dict[str, dict[str, Any]]) -> None:
    await asyncio.to_thread(save_sessions, SESSION_STORE_PATH, data)


class SessionChannelsCog(commands.Cog):
    """Manage Discord categories/channels per HyprL session (prefix commands)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _ensure_category(self, guild: nextcord.Guild, name: str) -> nextcord.CategoryChannel:
        for cat in guild.categories:
            if cat.name == name:
                return cat
        return await guild.create_category(name)

    async def _ensure_text_channel(
        self, guild: nextcord.Guild, category: nextcord.CategoryChannel, name: str
    ) -> nextcord.TextChannel:
        for channel in category.text_channels:
            if channel.name == name:
                return channel
        return await guild.create_text_channel(name, category=category)

    @commands.command(name="create_session")
    async def create_session_command(self, ctx: commands.Context, *, name: str) -> None:
        slug = slugify(name)
        if not slug:
            await ctx.send("Invalid session name. Use letters/numbers.", reference=None)
            return
        sessions = await _load_sessions()
        if slug in sessions:
            await ctx.send(f"Session `{slug}` already exists:\n{_format_session_summary(slug, sessions[slug])}")
            return
        if ctx.guild is None:
            await ctx.send("This command must be used in a guild/server.", reference=None)
            return

        category_name = f"HyprL - {name}"
        category = await self._ensure_category(ctx.guild, category_name)
        overview = await self._ensure_text_channel(ctx.guild, category, f"{slug}-overview")
        alerts = await self._ensure_text_channel(ctx.guild, category, f"{slug}-alerts")
        trades = await self._ensure_text_channel(ctx.guild, category, f"{slug}-trades")

        sessions[slug] = {
            "name": name,
            "slug": slug,
            "category_id": category.id,
            "channels": {
                "overview": overview.id,
                "alerts": alerts.id,
                "trades": trades.id,
            },
        }
        await _persist_sessions(sessions)
        await ctx.send(
            f"Session `{slug}` created.\n{_format_session_summary(slug, sessions[slug])}",
            reference=None,
        )

    @commands.command(name="list_sessions")
    async def list_sessions_command(self, ctx: commands.Context) -> None:
        sessions = await _load_sessions()
        if not sessions:
            await ctx.send("No sessions yet.", reference=None)
            return
        lines = [_format_session_summary(slug, data) for slug, data in sorted(sessions.items())]
        await ctx.send("\n\n".join(lines))

    @commands.command(name="session_info")
    async def session_info_command(self, ctx: commands.Context, *, name_or_slug: str) -> None:
        sessions = await _load_sessions()
        session = resolve_session(name_or_slug, sessions)
        if not session:
            await ctx.send("Unknown session.", reference=None)
            return
        slug = session.get("slug") or slugify(name_or_slug)
        await ctx.send(_format_session_summary(slug, session), reference=None)


class RealtimeSessionsCog(commands.Cog):
    """Slash commands against HyprL realtime API."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Démarrer une session realtime HyprL")
    async def start_session(self, interaction: nextcord.Interaction, symbols: str, interval: str = "1m") -> None:
        parsed = _split_symbols(symbols)
        if not parsed:
            await interaction.response.send_message("Merci de fournir au moins un symbole.", ephemeral=True)
            return
        client = self.bot.hyprl_client
        try:
            payload = await client.start_session(
                symbols=parsed,
                interval=interval,
                threshold=0.6,
                risk_pct=0.1,
                kill_switch_dd=0.30,
                enable_paper=False,
            )
        except HyprlAPIError as exc:
            await interaction.response.send_message(f"Erreur session ({exc.status_code}): {exc.payload}", ephemeral=True)
            return
        session_id = payload.get("session_id")
        embed = nextcord.Embed(title="Session démarrée", color=0xFFAA00)
        embed.add_field(name="Session ID", value=session_id or "?", inline=False)
        embed.add_field(name="Impl", value=str(payload.get("impl", "?")), inline=True)
        embed.add_field(name="Log dir", value=str(payload.get("log_dir", "n/a")), inline=False)
        await interaction.response.send_message(embed=embed)

    @nextcord.slash_command(description="Consulter l'état d'une session")
    async def session_status(self, interaction: nextcord.Interaction, session_id: str) -> None:
        client = self.bot.hyprl_client
        try:
            payload = await client.get_session_status(session_id)
        except HyprlAPIError as exc:
            await interaction.response.send_message(f"Erreur session ({exc.status_code}): {exc.payload}", ephemeral=True)
            return
        counters = payload.get("counters", {})
        embed = nextcord.Embed(title=f"Session {session_id}", color=0x3366FF)
        embed.add_field(name="Status", value=str(payload.get("status")), inline=True)
        embed.add_field(name="Kill switch", value=str(payload.get("kill_switch_triggered")), inline=True)
        embed.add_field(
            name="Counters",
            value=f"bars={counters.get('bars', 0)} / preds={counters.get('predictions', 0)} / fills={counters.get('fills', 0)}",
            inline=False,
        )
        await interaction.response.send_message(embed=embed)

    @nextcord.slash_command(description="Rapport métriques session")
    async def session_report(self, interaction: nextcord.Interaction, session_id: str) -> None:
        client = self.bot.hyprl_client
        try:
            payload = await client.get_session_report(session_id)
        except HyprlAPIError as exc:
            await interaction.response.send_message(f"Erreur rapport ({exc.status_code}): {exc.payload}", ephemeral=True)
            return
        metrics = payload.get("metrics", {})
        embed = nextcord.Embed(title=f"Report {session_id}", color=0x8844FF)
        embed.add_field(name="PF", value=f"{metrics.get('pf', 'n/a')}", inline=True)
        embed.add_field(name="Sharpe", value=f"{metrics.get('sharpe', 'n/a')}", inline=True)
        embed.add_field(name="Drawdown", value=f"{metrics.get('dd', 'n/a')}", inline=True)
        embed.add_field(name="Winrate", value=f"{metrics.get('winrate', 'n/a')}", inline=True)
        embed.add_field(name="Exposure", value=f"{metrics.get('exposure', 'n/a')}", inline=True)
        embed.add_field(name="Avg hold", value=f"{metrics.get('avg_hold_bars', 'n/a')}", inline=True)
        await interaction.response.send_message(embed=embed)

    @nextcord.slash_command(description="Lister des sessions existantes")
    async def sessions_list(self, interaction: nextcord.Interaction, session_ids: str) -> None:
        ids = [token.strip() for token in session_ids.replace(";", ",").split(",") if token.strip()]
        if not ids:
            await interaction.response.send_message("Fournis des IDs séparés par des virgules.", ephemeral=True)
            return
        client = self.bot.hyprl_client
        sessions = await client.list_sessions(ids)
        lines = []
        for session in sessions:
            sid = session.get("session_id", "?")
            status = session.get("status", session.get("error", "unknown"))
            lines.append(f"{sid}: {status}")
        await interaction.response.send_message("\n".join(lines))


async def send_to_session_channel(
    session_slug: str,
    channel_type: str,
    content: str | None = None,
    embed: nextcord.Embed | None = None,
) -> None:
    """Send a message to a session-scoped channel by slug and type."""
    if bot_ref is None:
        raise RuntimeError("Bot not initialized; call setup first.")
    if channel_type not in {"overview", "alerts", "trades"}:
        raise ValueError(f"Invalid channel_type: {channel_type}")
    if content is None and embed is None:
        raise ValueError("content or embed must be provided")
    sessions = await _load_sessions()
    session = resolve_session(session_slug, sessions)
    if not session:
        raise ValueError(f"Unknown session slug: {session_slug}")
    channels = get_session_channels(session)
    channel_id = channels.get(channel_type)
    if channel_id is None:
        raise ValueError(f"Channel {channel_type} not configured for session {session_slug}")
    channel = bot_ref.get_channel(int(channel_id))
    if channel is None:
        channel = await bot_ref.fetch_channel(int(channel_id))
    if not isinstance(channel, Messageable):
        raise RuntimeError(f"Channel {channel_id} is not messageable")
    await channel.send(content=content, embed=embed)


def setup(bot: commands.Bot) -> None:
    global bot_ref
    bot_ref = bot
    bot.add_cog(RealtimeSessionsCog(bot))
    bot.add_cog(SessionChannelsCog(bot))
