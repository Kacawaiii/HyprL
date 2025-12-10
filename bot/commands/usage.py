"""Usage reporting command."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


def build_usage_embed(payload: dict) -> nextcord.Embed:
    embed = nextcord.Embed(title="HyprL Usage", color=0x00AAFF)
    embed.add_field(
        name="Credits",
        value=f"{payload.get('credits_remaining', 'n/a')} / {payload.get('credits_total', 'n/a')}",
        inline=False,
    )
    by_endpoint = payload.get("by_endpoint", {}) or {}
    if by_endpoint:
        lines = [f"{endpoint}: {cost}" for endpoint, cost in sorted(by_endpoint.items())]
        embed.add_field(name="Par endpoint", value="\n".join(lines), inline=False)
    return embed


def format_usage_error(status_code: int, payload: str | dict | None) -> str:
    if status_code <= 0:
        return "Usage indisponible (connexion HyprL impossible). Réessaie plus tard."
    return f"Usage indisponible ({status_code}): {payload}"


class UsageCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Afficher les crédits restants HyprL")
    async def usage(self, interaction: nextcord.Interaction) -> None:
        client = self.bot.hyprl_client
        try:
            payload = await client.get_usage()
        except HyprlAPIError as exc:
            await interaction.response.send_message(format_usage_error(exc.status_code, exc.payload), ephemeral=True)
            return
        await interaction.response.send_message(embed=build_usage_embed(payload))

    @commands.command(name="usage")
    async def usage_prefix(self, ctx: commands.Context) -> None:
        client = self.bot.hyprl_client
        try:
            payload = await client.get_usage()
        except HyprlAPIError as exc:
            await ctx.send(format_usage_error(exc.status_code, exc.payload))
            return
        await ctx.send(embed=build_usage_embed(payload))


def setup(bot: commands.Bot) -> None:
    bot.add_cog(UsageCog(bot))
