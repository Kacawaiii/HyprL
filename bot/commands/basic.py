"""Basic utility slash commands."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


class BasicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Test connectivity with the HyprL API")
    async def ping(self, interaction: nextcord.Interaction) -> None:
        client = self.bot.hyprl_client
        try:
            health = await client.get_health()
            if health.get("ok"):
                await interaction.response.send_message("HyprL V2 API: OK")
            else:
                await interaction.response.send_message("HyprL V2 API: unexpected response", ephemeral=True)
        except HyprlAPIError as exc:
            await interaction.response.send_message(
                f"HyprL V2 API error ({exc.status_code}): {exc.payload}", ephemeral=True
            )

    @nextcord.slash_command(name="hyprl_ping", description="Vérifie le token HyprL utilisé par le bot")
    async def hyprl_ping(self, interaction: nextcord.Interaction) -> None:
        await hyprl_ping_handler(interaction, self.bot.hyprl_client)

    @nextcord.slash_command(name="hyprl_predict_stats", description="Affiche le winrate et PnL des prédictions")
    async def hyprl_predict_stats(self, interaction: nextcord.Interaction) -> None:
        await hyprl_predict_stats_handler(interaction, self.bot.hyprl_client)

    @nextcord.slash_command(name="help_hyprl", description="List bot commands")
    async def help_hyprl(self, interaction: nextcord.Interaction) -> None:
        commands_list = """
/ping – Vérifie /health
/hyprl_ping – Statut token via /v2/usage
/usage – Affiche les crédits restants
/predict <symbole> – Appelle /v2/predict
/start_session – Lance une session realtime
/session_status – Affiche l'état d'une session
/session_report – Rapporte les métriques d'une session
/autorank_start – Lance un job autorank + sessions
""".strip()
        await interaction.response.send_message(commands_list, ephemeral=True)


def build_hyprl_ping_embed(api_base: str, payload: dict) -> nextcord.Embed:
    embed = nextcord.Embed(title="HyprL Ping", color=0x10B981)
    embed.add_field(name="Account", value=payload.get("account_id", "n/a"), inline=True)
    embed.add_field(
        name="Credits Remaining",
        value=str(payload.get("credits_remaining", "n/a")),
        inline=True,
    )
    embed.add_field(name="API Base", value=api_base, inline=False)
    by_endpoint = payload.get("by_endpoint") or {}
    if by_endpoint:
        lines = [f"{endpoint}: {data}" for endpoint, data in sorted(by_endpoint.items())]
        embed.add_field(name="Endpoints", value="\n".join(lines[:6]), inline=False)
    return embed


async def hyprl_ping_handler(interaction: nextcord.Interaction, client) -> None:
    try:
        usage = await client.get_usage()
    except HyprlAPIError as exc:
        await interaction.response.send_message(
            f"HyprL usage indisponible ({exc.status_code}): {exc.payload}",
            ephemeral=True,
        )
        return
    embed = build_hyprl_ping_embed(client.api_base, usage)
    await interaction.response.send_message(embed=embed, ephemeral=True)


def build_predict_stats_embed(payload: dict) -> nextcord.Embed:
    embed = nextcord.Embed(title="HyprL Predict Stats", color=0x3B82F6)
    embed.add_field(
        name="Total Predictions",
        value=str(payload.get("total_predictions", "n/a")),
        inline=True,
    )
    embed.add_field(
        name="Closed",
        value=str(payload.get("closed_predictions", "n/a")),
        inline=True,
    )
    winrate = payload.get("winrate_real")
    embed.add_field(
        name="Real Winrate",
        value=f"{winrate:.1%}" if isinstance(winrate, float) else "n/a",
        inline=True,
    )
    embed.add_field(
        name="Total PnL",
        value=f"{payload.get('pnl_total', 0.0):.2f}",
        inline=True,
    )
    avg = payload.get("avg_pnl")
    embed.add_field(
        name="Avg PnL",
        value=f"{avg:.2f}" if isinstance(avg, (float, int)) else "n/a",
        inline=True,
    )
    return embed


async def hyprl_predict_stats_handler(interaction: nextcord.Interaction, client) -> None:
    try:
        summary = await client.get_predict_summary()
    except HyprlAPIError as exc:
        await interaction.response.send_message(
            f"Résumé predict indisponible ({exc.status_code}): {exc.payload}",
            ephemeral=True,
        )
        return
    embed = build_predict_stats_embed(summary)
    await interaction.response.send_message(embed=embed, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(BasicCog(bot))
