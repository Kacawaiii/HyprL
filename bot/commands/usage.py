"""Usage reporting command."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


class UsageCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Afficher les crédits restants HyprL")
    async def usage(self, interaction: nextcord.Interaction) -> None:
        client = self.bot.hyprl_client
        try:
            payload = await client.get_usage()
        except HyprlAPIError as exc:
            await interaction.response.send_message(
                f"Usage indisponible ({exc.status_code}): {exc.payload}", ephemeral=True
            )
            return
        embed = nextcord.Embed(title="HyprL Usage", color=0x00AAFF)
        embed.add_field(
            name="Credits", value=f"{payload.get('credits_remaining', 'n/a')} / {payload.get('credits_total', 'n/a')}", inline=False
        )
        by_endpoint = payload.get("by_endpoint", {}) or {}
        if by_endpoint:
            lines = [f"{endpoint}: {cost}" for endpoint, cost in sorted(by_endpoint.items())]
            embed.add_field(name="Par endpoint", value="\n".join(lines), inline=False)
        await interaction.response.send_message(embed=embed)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(UsageCog(bot))
