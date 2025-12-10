"""Predict command hooking into /v2/predict."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError


def build_predict_embed(symbol: str, payload: dict) -> nextcord.Embed:
    results = payload.get("results") or []
    if not results:
        return nextcord.Embed(
            title=f"Predict {symbol.upper()}",
            description="Aucune prédiction retournée par HyprL.",
            color=nextcord.Color.dark_grey(),
        )

    pred = results[0]
    meta = payload.get("meta") or {}

    direction = (pred.get("direction") or "UNKNOWN").upper()
    prob_up = pred.get("prob_up")
    threshold = pred.get("threshold")
    risk_pct = pred.get("risk_pct")
    tp = pred.get("tp")
    sl = pred.get("sl")
    outcome = pred.get("outcome")
    pnl = pred.get("pnl")
    closed = bool(pred.get("closed"))
    interval = pred.get("interval") or "1h"

    model = meta.get("model", "prob_bridge_v2")
    version = meta.get("version")

    prob_dir = None
    if isinstance(prob_up, (int, float)):
        if direction == "UP":
            prob_dir = prob_up
        elif direction == "DOWN":
            prob_dir = 1.0 - prob_up
        else:
            prob_dir = prob_up

    prob_pct = prob_dir * 100 if prob_dir is not None else None
    if isinstance(risk_pct, (int, float)) and risk_pct <= 1:
        risk_pct_disp = risk_pct * 100
    else:
        risk_pct_disp = risk_pct
    tp_disp = tp
    sl_disp = sl

    if direction == "UP":
        color = nextcord.Color.green()
        direction_label = "UP (long)"
    elif direction == "DOWN":
        color = nextcord.Color.red()
        direction_label = "DOWN (short)"
    else:
        color = nextcord.Color.dark_grey()
        direction_label = direction

    embed = nextcord.Embed(
        title=f"Predict {symbol.upper()} ({interval})",
        description="Signal instantané HyprL v2",
        color=color,
    )

    signal_lines = [f"**Direction :** {direction_label}"]
    if prob_pct is not None:
        signal_lines.append(f"**Probabilité (direction) :** {prob_pct:.2f}%")
    if threshold is not None:
        signal_lines.append(f"**Seuil (threshold) :** {float(threshold):.2f}")
    if risk_pct_disp is not None:
        signal_lines.append(f"**Risque par trade :** {float(risk_pct_disp):.2f}%")

    embed.add_field(name="Signal", value="\n".join(signal_lines), inline=False)

    trade_lines: list[str] = []
    if tp_disp is not None:
        trade_lines.append(f"**TP :** {float(tp_disp):.2f}%")
    if sl_disp is not None:
        trade_lines.append(f"**SL :** {float(sl_disp):.2f}%")
    if tp_disp not in (None, 0) and sl_disp not in (None, 0):
        try:
            rr = abs(float(tp_disp) / float(sl_disp))
        except (TypeError, ValueError, ZeroDivisionError):
            rr = None
        if rr is not None:
            trade_lines.append(f"**Ratio R/R approx :** {rr:.2f}")
    if trade_lines:
        embed.add_field(name="Trade idea", value="\n".join(trade_lines), inline=False)

    status_lines = []
    if closed:
        status = outcome or "N/A"
        status_lines.append(f"**Statut :** CLOSED ({status})")
        if pnl is not None:
            status_lines.append(f"**PnL simulé :** {float(pnl):.2f}")
    else:
        status_lines.append("**Statut :** LIVE / en cours")
    embed.add_field(name="Backtest / outcome", value="\n".join(status_lines), inline=False)

    footer = f"Model: {model}"
    if version is not None:
        footer += f" (v{version})"
    footer += " • Endpoint: /v2/predict"
    embed.set_footer(text=footer)
    return embed


def format_predict_error(status_code: int, payload: str | dict | None) -> str:
    if status_code <= 0:
        return "Connexion HyprL impossible (réseau ou host indisponible). Réessaie plus tard."
    return {
        401: "Token invalide pour l'API HyprL.",
        402: "Crédits insuffisants pour /predict.",
        403: "Scope manquant pour /predict.",
        429: "Limite de débit atteinte, réessaie plus tard.",
    }.get(status_code, f"Erreur HyprL ({status_code}): {payload}")


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
            await interaction.response.send_message(
                format_predict_error(exc.status_code, exc.payload), ephemeral=True
            )
            return
        results = payload.get("results", [])
        if not results:
            await interaction.response.send_message("Aucun résultat reçu.", ephemeral=True)
            return
        await interaction.response.send_message(embed=build_predict_embed(cleaned, payload))

    @commands.command(name="predict")
    async def predict_prefix(self, ctx: commands.Context, symbol: str | None = None) -> None:
        if not symbol:
            await ctx.send("Usage: !predict <symbole>")
            return
        cleaned = symbol.upper().strip()
        client = self.bot.hyprl_client
        try:
            payload = await client.post_predict([cleaned])
        except HyprlAPIError as exc:
            await ctx.send(format_predict_error(exc.status_code, exc.payload))
            return
        results = payload.get("results", [])
        if not results:
            await ctx.send("Aucun résultat reçu.")
            return
        await ctx.send(embed=build_predict_embed(cleaned, payload))


def setup(bot: commands.Bot) -> None:
    bot.add_cog(PredictCog(bot))
