"""Predict command hooking into /v2/predict."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


class PredictCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Lancer une prédiction HyprL pour un symbole")
    async def predict(self, interaction: nextcord.Interaction, symbol: str) -> None:
        cleaned = symbol.upper().strip()
        if not cleaned:
            await interaction.response.send_message("Merci de spécifier un symbole.", ephemeral=True)
            return
        client = self.bot.hyprl_client
        try:
            payload = await client.post_predict([cleaned])
        except HyprlAPIError as exc:
            message = {
                401: "Token invalide pour l'API HyprL.",
                402: "Crédits insuffisants pour /predict.",
                403: "Scope manquant pour /predict.",
                429: "Limite de débit atteinte, réessaie plus tard.",
            }.get(exc.status_code, f"Erreur HyprL ({exc.status_code}): {exc.payload}")
            await interaction.response.send_message(message, ephemeral=True)
            return
        results = payload.get("results", [])
        if not results:
            await interaction.response.send_message("Aucun résultat reçu.", ephemeral=True)
            return
        result = results[0]
        embed = nextcord.Embed(title=f"Predict {cleaned}", color=0x00CC66)
        embed.add_field(name="Probabilité", value=f"{result.get('prob_up', 0.0)*100:.2f}%", inline=True)
        embed.add_field(name="Direction", value=str(result.get("direction", "?")), inline=True)
        embed.add_field(name="Threshold", value=str(result.get("threshold", "n/a")), inline=True)
        embed.add_field(name="TP", value=str(result.get("tp", "n/a")), inline=True)
        embed.add_field(name="SL", value=str(result.get("sl", "n/a")), inline=True)
        await interaction.response.send_message(embed=embed)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(PredictCog(bot))
