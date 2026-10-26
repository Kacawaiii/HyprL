"""Optional Discord gateway adapter. Installing it never edits an existing bot."""

from __future__ import annotations

import asyncio
import hashlib
import time

import discord
from discord import app_commands

from .destinations import CHANNELS
from .model import KINDS, Alert, AlertError, synthetic_alert
from .security import SafeURL, resolve


class Access:
    def __init__(self, store, *, guild_id, role_id):
        self.store, self.guild_id, self.role_id = store, guild_id, role_id

    def customer(self, interaction):
        if (
            interaction.guild_id != self.guild_id
            or not interaction.guild
            or not any(role.id == self.role_id for role in getattr(interaction.user, "roles", []))
        ):
            raise AlertError("ACCESS_DENIED")
        customer = f"{self.guild_id}/{interaction.user.id}"
        self.store.customer(customer)  # explicit operator allowlist, staff only initially
        return customer


async def reply(interaction, text, **kwargs):
    if interaction.response.is_done():
        await interaction.followup.send(text, ephemeral=True, **kwargs)
    else:
        await interaction.response.send_message(text, ephemeral=True, **kwargs)


class PauseView(discord.ui.View):
    def __init__(self, access, destination):
        super().__init__(timeout=None)
        button = discord.ui.Button(
            label="Mettre en pause",
            custom_id=f"hyprl-alert:pause:{destination}",
            style=discord.ButtonStyle.secondary,
        )

        async def callback(interaction):
            try:
                customer = access.customer(interaction)
                access.store.pause(customer, destination, True)
                await reply(
                    interaction, "Destination en pause. Reprise : /alertes pause active:False."
                )
            except AlertError:
                await reply(interaction, "Action refusée ou destination indisponible.")

        button.callback = callback
        self.add_item(button)


class DestinationModal(discord.ui.Modal, title="Ajouter une destination privée"):
    endpoint = discord.ui.TextInput(label="URL du webhook ou identifiant Telegram", max_length=2048)
    signing_key = discord.ui.TextInput(
        label="Clé HMAC (webhook personnalisé seulement)", required=False, max_length=128
    )

    def __init__(self, access, channel, kinds, watchlist):
        super().__init__()
        self.access, self.channel, self.kinds, self.watchlist = access, channel, kinds, watchlist

    async def on_submit(self, interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            customer = self.access.customer(
                interaction
            )  # role may have been removed since /ajouter
            if self.channel == "telegram":
                config = {"chat_id": self.endpoint.value.strip()}
            else:
                url = self.endpoint.value.strip()
                endpoint = SafeURL.parse(url)
                endpoint.validate_addresses(
                    await asyncio.wait_for(resolve(endpoint.host), timeout=5)
                )
                config = {"url": url}
                if self.channel == "webhook":
                    config["signing_key"] = self.signing_key.value
            destination = self.access.store.add(
                customer, self.channel, config, kinds=self.kinds, watchlist=self.watchlist
            )
            await reply(
                interaction,
                f"Destination {destination} ajoutée ({self.channel}).",
                view=PauseView(self.access, destination),
            )
        except Exception:
            await reply(
                interaction,
                "Destination refusée. Vérifiez le canal, les champs et les autorisations staff.",
            )

    async def on_error(self, interaction, error):
        await reply(interaction, "Ajout indisponible.")


class AlertCommands(app_commands.Group):
    def __init__(self, access, *, clock=time.time):
        super().__init__(
            name="alertes",
            description="Alertes d'information HyprL — accès anticipé",
            guild_only=True,
        )
        self.access, self.clock = access, clock

    async def interaction_check(self, interaction):
        try:
            self.access.customer(interaction)
            return True
        except AlertError:
            await reply(
                interaction, "Accès anticipé requis ; compte autorisé par l'équipe uniquement."
            )
            return False

    async def on_error(self, interaction, error):
        await reply(
            interaction,
            "Action refusée ou indisponible. Aucune configuration privée n'est affichée.",
        )

    @app_commands.command(description="Connecter un canal via un formulaire privé")
    @app_commands.choices(canal=[app_commands.Choice(name=c, value=c) for c in sorted(CHANNELS)])
    async def ajouter(
        self, interaction: discord.Interaction, canal: str, symboles: str = "", types: str = ""
    ):
        self.access.customer(interaction)
        if canal in ("email", "api"):
            await reply(
                interaction,
                "Canal prévu ; e-mail en attente du choix de l'expéditeur, API non exposée.",
            )
            return
        kinds = [x.strip() for x in types.split(",") if x.strip()] if types else sorted(KINDS)
        watchlist = [x.strip().upper() for x in symboles.split(",") if x.strip()]
        from .destinations import subscriptions

        kinds, watchlist = subscriptions(kinds, watchlist)
        await interaction.response.send_modal(
            DestinationModal(self.access, canal, kinds, watchlist)
        )

    @app_commands.command(description="Lister vos destinations et leurs boutons de pause")
    async def liste(self, interaction: discord.Interaction):
        customer = self.access.customer(interaction)
        rows = self.access.store.list(customer)
        await reply(interaction, f"{len(rows)} destination(s). Les URL restent privées.")
        for row in rows:
            state = "en pause" if row["paused"] else "active"
            await reply(
                interaction,
                f"{row['id']} · {row['channel']} · {state}",
                view=PauseView(self.access, row["id"]),
            )

    @app_commands.command(description="Mettre un test synthétique en file pour une destination")
    async def test(self, interaction: discord.Interaction, destination: str):
        customer = self.access.customer(interaction)
        event = synthetic_alert()
        now = self.clock()
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")
        event.update(
            occurred_at=stamp,
            available_at=stamp,
            event_id=hashlib.sha256(f"test:{interaction.id}".encode()).hexdigest(),
        )
        count = self.access.store.enqueue(
            customer, Alert.parse(event), now=now, destination=destination
        )
        await reply(
            interaction,
            f"Test synthétique : {count} envoi(s) en file. /alertes liste pour mettre en pause.",
        )

    @app_commands.command(description="Mettre en pause ou reprendre une destination")
    async def pause(self, interaction: discord.Interaction, destination: str, active: bool = True):
        self.access.store.pause(self.access.customer(interaction), destination, active)
        await reply(interaction, "Destination en pause." if active else "Destination reprise.")

    @app_commands.command(description="Supprimer une destination et ses envois en attente")
    async def supprimer(self, interaction: discord.Interaction, destination: str):
        self.access.store.remove(self.access.customer(interaction), destination)
        await reply(interaction, "Destination supprimée.")


def install(client, store, *, guild_id, role_id):
    """Attach only /alertes to a Bot/Client; caller controls registration/lifecycle."""
    tree = getattr(client, "tree", None)
    if tree is None:
        tree = app_commands.CommandTree(client)
    access = Access(store, guild_id=guild_id, role_id=role_id)
    group = AlertCommands(access)
    tree.add_command(group, guild=discord.Object(id=guild_id))
    for customer in store.allowed:
        for row in store.list(customer):
            client.add_view(PauseView(access, row["id"]))
    return tree, group


def run_gateway(store, *, token, guild_id, role_id, register=False):
    class Gateway(discord.Client):
        async def setup_hook(self):
            tree, group = install(self, store, guild_id=guild_id, role_id=role_id)
            if register:
                # Upsert this group only; never bulk sync/delete the bot's other commands.
                await self.http.upsert_guild_command(
                    self.application_id, guild_id, group.to_dict(tree)
                )

    # Guild metadata populates the role-ID cache used by Member.roles.
    # Message content and member-list privileged intents remain disabled.
    client = Gateway(intents=discord.Intents(guilds=True))
    # discord.py exceptions may contain token-bearing URLs. No default log handler.
    client.run(token, log_handler=None)
