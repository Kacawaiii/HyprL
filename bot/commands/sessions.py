"""Realtime session commands."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


def _split_symbols(raw: str) -> list[str]:
    return [token.strip().upper() for token in raw.replace(";", ",").split(",") if token.strip()]


class SessionsCog(commands.Cog):
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


def setup(bot: commands.Bot) -> None:
    bot.add_cog(SessionsCog(bot))
