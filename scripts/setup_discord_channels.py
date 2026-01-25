#!/usr/bin/env python3
"""
Setup Discord Channels for HyprL
=================================
Crée automatiquement les catégories et salons Discord pour chaque compte paper.

Usage:
    python scripts/setup_discord_channels.py

Requires:
    - DISCORD_BOT_TOKEN dans .env.ops
    - DISCORD_GUILD_ID (ID du serveur Discord)
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.ops")

try:
    import discord
    from discord import Permissions
except ImportError:
    print("❌ discord.py non installé")
    print("   pip install discord.py")
    sys.exit(1)


@dataclass
class ChannelSetup:
    """Configuration d'un salon à créer."""
    name: str
    topic: str
    category: str


# Structure des salons à créer
CHANNEL_STRUCTURE = {
    "HyprL Normal": [
        ChannelSetup("trades-normal", "Entrées et sorties de trades - Compte Normal", "HyprL Normal"),
        ChannelSetup("alerts-normal", "Alertes et warnings - Compte Normal", "HyprL Normal"),
        ChannelSetup("summary-normal", "Résumés horaires et journaliers - Compte Normal", "HyprL Normal"),
    ],
    "HyprL Aggressive": [
        ChannelSetup("trades-aggressive", "Entrées et sorties de trades - Compte Aggressive", "HyprL Aggressive"),
        ChannelSetup("alerts-aggressive", "Alertes et warnings - Compte Aggressive", "HyprL Aggressive"),
        ChannelSetup("summary-aggressive", "Résumés horaires et journaliers - Compte Aggressive", "HyprL Aggressive"),
    ],
    "HyprL Mix": [
        ChannelSetup("trades-mix", "Entrées et sorties de trades - Compte Mix", "HyprL Mix"),
        ChannelSetup("alerts-mix", "Alertes et warnings - Compte Mix", "HyprL Mix"),
        ChannelSetup("summary-mix", "Résumés horaires et journaliers - Compte Mix", "HyprL Mix"),
    ],
    "HyprL Global": [
        ChannelSetup("global-summary", "Résumé global tous comptes", "HyprL Global"),
        ChannelSetup("risk-alerts", "Alertes critiques et circuit breakers", "HyprL Global"),
        ChannelSetup("heartbeat", "Heartbeat et status système", "HyprL Global"),
    ],
}


class DiscordSetup(discord.Client):
    """Client Discord pour setup des salons."""

    def __init__(self, guild_id: int):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.guild_id = guild_id
        self.webhooks_created: Dict[str, str] = {}
        self.setup_complete = asyncio.Event()

    async def on_ready(self):
        """Appelé quand le bot est connecté."""
        print(f"✅ Connecté en tant que {self.user}")
        print(f"   Serveurs: {len(self.guilds)}")

        # Trouver le serveur
        guild = self.get_guild(self.guild_id)
        if not guild:
            print(f"❌ Serveur {self.guild_id} non trouvé")
            print("   Serveurs disponibles:")
            for g in self.guilds:
                print(f"   - {g.name} (ID: {g.id})")
            await self.close()
            return

        print(f"\n📍 Serveur: {guild.name}")
        print("=" * 60)

        try:
            await self.setup_channels(guild)
        except Exception as e:
            print(f"❌ Erreur: {e}")
        finally:
            self.setup_complete.set()
            await self.close()

    async def setup_channels(self, guild: discord.Guild):
        """Crée les catégories et salons."""
        print("\n🔧 Création des catégories et salons...\n")

        for category_name, channels in CHANNEL_STRUCTURE.items():
            print(f"📁 {category_name}")

            # Chercher ou créer la catégorie
            category = discord.utils.get(guild.categories, name=category_name)
            if not category:
                print(f"   ➕ Création catégorie...")
                category = await guild.create_category(
                    name=category_name,
                    reason="HyprL Trading Bot Setup"
                )
            else:
                print(f"   ✓ Catégorie existe déjà")

            # Créer les salons dans cette catégorie
            for channel_setup in channels:
                existing = discord.utils.get(guild.text_channels, name=channel_setup.name)

                if existing:
                    print(f"   ├── #{channel_setup.name} (existe)")
                    channel = existing
                else:
                    print(f"   ├── #{channel_setup.name} (création...)")
                    channel = await guild.create_text_channel(
                        name=channel_setup.name,
                        category=category,
                        topic=channel_setup.topic,
                        reason="HyprL Trading Bot Setup"
                    )

                # Créer un webhook pour ce salon
                webhooks = await channel.webhooks()
                hyprl_webhook = None
                for wh in webhooks:
                    if wh.name == "HyprL Bot":
                        hyprl_webhook = wh
                        break

                if not hyprl_webhook:
                    hyprl_webhook = await channel.create_webhook(
                        name="HyprL Bot",
                        reason="HyprL Trading Bot Notifications"
                    )
                    print(f"   │   └── Webhook créé")
                else:
                    print(f"   │   └── Webhook existe")

                # Stocker l'URL du webhook
                webhook_key = self._get_webhook_key(channel_setup.name)
                self.webhooks_created[webhook_key] = hyprl_webhook.url

            print()

        # Générer le fichier .env avec les webhooks
        self._generate_env_file()

        # Envoyer un message de bienvenue dans global-summary
        await self._send_welcome_message(guild)

    def _get_webhook_key(self, channel_name: str) -> str:
        """Convertit le nom du salon en clé webhook."""
        # trades-normal -> DISCORD_WEBHOOK_NORMAL_TRADES
        parts = channel_name.split("-")
        if len(parts) == 2:
            channel_type, account = parts
            return f"DISCORD_WEBHOOK_{account.upper()}_{channel_type.upper()}"
        elif channel_name == "global-summary":
            return "DISCORD_WEBHOOK_GLOBAL_SUMMARY"
        elif channel_name == "risk-alerts":
            return "DISCORD_WEBHOOK_GLOBAL_ALERTS"
        elif channel_name == "heartbeat":
            return "DISCORD_WEBHOOK_GLOBAL_HEARTBEAT"
        return f"DISCORD_WEBHOOK_{channel_name.upper().replace('-', '_')}"

    def _generate_env_file(self):
        """Génère le fichier .env.discord avec les webhooks."""
        env_path = Path(__file__).parent.parent / "configs" / "runtime" / ".env.discord"

        lines = [
            "# HyprL Discord Webhooks",
            "# Generated automatically by setup_discord_channels.py",
            "# " + "=" * 50,
            "",
            "# Normal Account",
        ]

        # Grouper par compte
        accounts = ["NORMAL", "AGGRESSIVE", "MIX", "GLOBAL"]

        for account in accounts:
            if account != "NORMAL":
                lines.append(f"\n# {account.capitalize()} Account")

            for key, url in sorted(self.webhooks_created.items()):
                if account in key:
                    lines.append(f"{key}={url}")

        env_content = "\n".join(lines) + "\n"

        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(env_content)

        print("=" * 60)
        print(f"✅ Fichier .env généré: {env_path}")
        print("=" * 60)
        print("\n📋 Webhooks créés:\n")
        for key, url in sorted(self.webhooks_created.items()):
            print(f"   {key}")
            print(f"   └── {url[:60]}...")
            print()

    async def _send_welcome_message(self, guild: discord.Guild):
        """Envoie un message de bienvenue."""
        channel = discord.utils.get(guild.text_channels, name="global-summary")
        if not channel:
            return

        embed = discord.Embed(
            title="🤖 HyprL Trading Bot - Setup Complete",
            description="Les salons Discord ont été configurés avec succès!",
            color=0x00FF00
        )

        embed.add_field(
            name="📊 Structure",
            value="""
**HyprL Normal** - Compte équilibré
**HyprL Aggressive** - Compte high-risk
**HyprL Mix** - 70% Normal + 30% Aggressive
**HyprL Global** - Résumés et alertes critiques
            """,
            inline=False
        )

        embed.add_field(
            name="🚀 Prochaines étapes",
            value="""
1. Source le fichier .env.discord
2. Lancer le bot de production
3. Les trades apparaîtront automatiquement
            """,
            inline=False
        )

        embed.set_footer(text="HyprL v1.1 | Paper Trading")

        await channel.send(embed=embed)
        print("✅ Message de bienvenue envoyé dans #global-summary")


async def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("HYPRL DISCORD SETUP")
    print("=" * 60)

    # Récupérer le token
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("❌ DISCORD_BOT_TOKEN non trouvé dans .env.ops")
        print("   Ajoute: DISCORD_BOT_TOKEN=ton_token")
        sys.exit(1)

    # Récupérer l'ID du serveur
    guild_id = os.environ.get("DISCORD_GUILD_ID")
    if not guild_id:
        print("\n⚠️  DISCORD_GUILD_ID non défini")
        print("   Je vais lister les serveurs disponibles...\n")

        # Lancer temporairement pour lister les serveurs
        class ListGuilds(discord.Client):
            async def on_ready(self):
                print("Serveurs disponibles:")
                for g in self.guilds:
                    print(f"  - {g.name} (ID: {g.id})")
                print("\nAjoute dans .env.ops:")
                print(f"  DISCORD_GUILD_ID={self.guilds[0].id if self.guilds else 'TON_ID'}")
                await self.close()

        client = ListGuilds(intents=discord.Intents.default())
        await client.start(token)
        sys.exit(0)

    guild_id = int(guild_id)
    print(f"🎯 Guild ID: {guild_id}")

    # Lancer le setup
    client = DiscordSetup(guild_id)
    await client.start(token)


if __name__ == "__main__":
    asyncio.run(main())
