"""Autorank orchestration command."""

from __future__ import annotations

import nextcord
from nextcord.ext import commands

from bot.hyprl_client import HyprlAPIError

DEFAULT_AUTORANK_CSV = "data/experiments/supersearch_portfolio_AAPL_MSFT_1y.csv"


class AutorankCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(description="Lancer un autorank + sessions")
    async def autorank_start(
        self,
        interaction: nextcord.Interaction,
        top_k: int = nextcord.SlashOption(int, "Nombre de stratégies", default=3),
        dry_run: bool = nextcord.SlashOption(bool, "Simulation uniquement", default=False),
    ) -> None:
        client = self.bot.hyprl_client
        payload = {
            "csv_paths": [DEFAULT_AUTORANK_CSV],
            "top_k": max(1, top_k),
            "meta_model": None,
            "meta_weight": 0.4,
            "constraints": {"min_pf": 0.0, "max_dd": 0.5, "min_trades": 10},
            "session": {
                "interval": "1m",
                "threshold": 0.6,
                "risk_pct": 0.1,
                "kill_switch_dd": 0.30,
                "enable_paper": False,
            },
            "seed": 42,
            "dry_run": dry_run,
        }
        try:
            job = await client.start_autorank(payload)
        except HyprlAPIError as exc:
            await interaction.response.send_message(f"Autorank KO ({exc.status_code}): {exc.payload}", ephemeral=True)
            return
        sessions = job.get("sessions", []) or []
        lines = [f"{item.get('rank')}: {item.get('session_id', 'n/a')}" for item in sessions]
        text = "\n".join(lines) or "Aucune session (dry_run?)."
        await interaction.response.send_message(
            f"Autorank {job.get('autorank_id')} lancé. Sessions: {len(sessions)}\n{text}"
        )


def setup(bot: commands.Bot) -> None:
    bot.add_cog(AutorankCog(bot))
