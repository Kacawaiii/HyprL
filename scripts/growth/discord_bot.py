"""Dedicated HyprL referral client. Never imported by another bot or trading engine."""

import asyncio
import fcntl
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from discord import app_commands

from .referrals import Ledger, attributed


class ReferralBot(discord.Client):
    def __init__(self, config, ledger):
        intents = discord.Intents.none()
        intents.guilds = True
        intents.members = True
        intents.invites = True
        super().__init__(intents=intents, max_messages=None)
        self.config, self.ledger = config, ledger
        self.tree = app_commands.CommandTree(self)
        self.lock = asyncio.Lock()
        self.before = None
        self.ready_for_joins = False
        self.job = None

        @self.tree.error
        async def command_error(interaction, error):
            logging.error('Referral command failed; details withheld')
            if interaction.response.is_done():
                await interaction.followup.send('Le parrainage est temporairement indisponible. Réessayez plus tard.', ephemeral=True)
            else:
                await interaction.response.send_message('Le parrainage est temporairement indisponible. Réessayez plus tard.', ephemeral=True)

        @self.tree.command(name='parrainage', description='Votre invitation personnelle et vos parrainages validés')
        @app_commands.guild_only()
        async def referral(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            if interaction.guild_id != self.config['guild'] or interaction.user.bot or not self.ready_for_joins:
                await interaction.followup.send('Le parrainage est temporairement indisponible ici.', ephemeral=True)
                return
            async with self.lock:
                self.ledger.purge(datetime.now(timezone.utc).timestamp())
                code = self.ledger.invite(interaction.user.id)
                invites = await interaction.guild.invites()
                known = {i.code: i for i in invites}
                if code and code not in known:
                    # A deleted permanent link is never silently replaced by another invitation.
                    await interaction.followup.send('Votre invitation a été supprimée. Contactez l’équipe pour examiner votre registre.', ephemeral=True)
                    return
                if not code:
                    channel = interaction.guild.get_channel(self.config['invitation_channel'])
                    invite = await channel.create_invite(max_age=0, max_uses=0, unique=True, reason='Programme de parrainage HyprL')
                    try:
                        self.ledger.register(interaction.user.id, invite.code, datetime.now(timezone.utc).timestamp())
                    except Exception:
                        await invite.delete(reason='Invitation non enregistrée')
                        raise
                    code = invite.code
                    # Do not reset other counters: concurrent joins retain their deltas.
                    if self.before is not None:
                        self.before[code] = 0
                n = self.ledger.counts(interaction.user.id)
                await interaction.followup.send(f"Votre lien permanent : https://discord.gg/{code}\nValidés : {n.get('valid', 0)}. En attente des sept jours : {n.get('pending', 0)}.\nClassement : {self.ledger.alias(interaction.user.id)}.\n3 validés : un crédit d’un mois Premium, utilisable lorsque Stripe sera disponible. 10 : une place Fondateur. Chaque palier est attribué une fois.\nLes comptes de moins de sept jours à l’arrivée et les départs avant sept jours sont exclus. Règles et données : {self.config['public_url']}/parrainage", ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    async def setup_hook(self):
        guild = discord.Object(id=self.config['guild'])
        # This is the existing HyprL application. Bulk tree.sync would delete its
        # unrelated guild commands. Upsert only our one named command instead.
        command = self.tree.get_command('parrainage')
        await self.http.upsert_guild_command(self.application_id, guild.id,
                                             payload=command.to_dict(self.tree))

    async def on_ready(self):
        if not self.job or self.job.done():
            self.job = asyncio.create_task(self.reconcile_loop())
        async with self.lock:
            self.ready_for_joins = False
            guild = self.get_guild(self.config['guild'])
            if not guild:
                raise RuntimeError('Configured guild unavailable')
            self.before = {i.code: i.uses or 0 for i in await guild.invites()}
            self.ready_for_joins = True

    async def on_disconnect(self):
        self.ready_for_joins = False
        self.before = None

    async def on_resumed(self):
        # Missed events cannot be attributed safely from an old counter snapshot.
        await self.on_ready()

    async def on_member_join(self, member):
        if member.guild.id != self.config['guild']:
            return
        async with self.lock:
            if not self.ready_for_joins:
                return
            try:
                after = {i.code: i.uses or 0 for i in await member.guild.invites()}
            except discord.HTTPException:
                self.before = None
                # Keep an unattributed arrival so a later rejoin cannot gain credit.
                self.ledger.join(member.id, None, member.created_at.timestamp(), member.joined_at.timestamp(), bot=member.bot)
                return
            code = attributed(self.before, after)
            self.before = after
            self.ledger.join(member.id, code, member.created_at.timestamp(), member.joined_at.timestamp(), bot=member.bot)

    async def on_member_remove(self, member):
        if member.guild.id == self.config['guild']:
            async with self.lock:
                self.ledger.leave(member.id, datetime.now(timezone.utc).timestamp())

    async def reconcile(self):
        guild = self.get_guild(self.config['guild'])
        now = datetime.now(timezone.utc)
        self.ledger.purge(now.timestamp())
        present = {}
        for row in self.ledger.pending():
            if now.timestamp() < row['joined'] + 7 * 86400:
                continue
            try:
                member = await guild.fetch_member(int(row['member']))
            except discord.NotFound:
                continue
            # Other HTTP failures abort reconciliation; never convert an outage to a departure.
            if member.joined_at:
                present[row['member']] = member.joined_at.timestamp()
        self.ledger.mature(now.timestamp(), present)
        for reward in self.ledger.rewards():
            if not self.config.get('apply_rewards', False):
                continue
            if reward['state'] == 'role_granted':
                continue
            try:
                member = await guild.fetch_member(int(reward['sponsor']))
            except discord.NotFound:
                continue
            role = guild.get_role(self.config['reward_roles'][reward['threshold']])
            if not role or role.managed or role >= guild.me.top_role:
                raise RuntimeError('Reward role configuration invalid')
            # add_roles is idempotent. Credit remains pending for a future Stripe adapter.
            await member.add_roles(role, reason='Palier de parrainage HyprL validé')
            self.ledger.mark_role(reward['sponsor'], reward['threshold'])
        monday = (now - timedelta(days=now.weekday())).replace(hour=9, minute=0, second=0, microsecond=0)
        if self.config.get('publish_leaderboard', False) and now >= monday:
            week = monday.date().isoformat()
            board = self.ledger.leaderboard()
            if board and self.ledger.reserve_post(week):
                channel = guild.get_channel(self.config['leaderboard_channel'])
                text = 'Parrainages validés · classement hebdomadaire\n' + '\n'.join(f'{i}. {alias} : {n}' for i, (alias, n) in enumerate(board, 1)) + '\nValidation après sept jours. Votre invitation et votre compteur : /parrainage.'
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                self.ledger.mark_post(week)

    async def reconcile_loop(self):
        while not self.is_closed():
            try:
                if self.is_ready():
                    async with self.lock:
                        if not self.ready_for_joins:
                            guild = self.get_guild(self.config['guild'])
                            self.before = {i.code: i.uses or 0 for i in await guild.invites()}
                            self.ready_for_joins = True
                        await self.reconcile()
            except Exception:
                logging.error('Referral reconciliation failed; private operator review required')
            await asyncio.sleep(300)

    async def on_error(self, event, *args, **kwargs):
        logging.error('Referral event failed; details withheld')


def configuration(env):
    # The operator explicitly chose the existing HyprL application. Its identity
    # file is loaded by systemd, never rewritten or copied by this process.
    required = ['DISCORD_BOT_TOKEN', 'DISCORD_GUILD_ID', 'REFERRAL_INVITE_CHANNEL_ID', 'REFERRAL_LEDGER', 'REFERRAL_ALIAS_KEY']
    rewards = env.get('REFERRAL_APPLY_REWARDS') == 'true'
    leaderboard = env.get('REFERRAL_PUBLISH_LEADERBOARD') == 'true'
    if rewards:
        required += ['REFERRAL_PREMIUM_CREDIT_ROLE_ID', 'REFERRAL_FONDATEUR_ROLE_ID']
    if leaderboard:
        required += ['REFERRAL_LEADERBOARD_CHANNEL_ID']
    if any(not env.get(k) for k in required) or len(env['REFERRAL_ALIAS_KEY']) < 32:
        raise SystemExit('HyprL referral configuration is incomplete')
    numeric = ['DISCORD_GUILD_ID', 'REFERRAL_INVITE_CHANNEL_ID']
    numeric += [k for k in required if k.endswith('_ROLE_ID') or k == 'REFERRAL_LEADERBOARD_CHANNEL_ID']
    if any(not env[k].isdigit() or int(env[k]) <= 0 for k in numeric):
        raise SystemExit('HyprL referral identifiers must be positive integers')
    roles = {3: int(env['REFERRAL_PREMIUM_CREDIT_ROLE_ID']), 10: int(env['REFERRAL_FONDATEUR_ROLE_ID'])} if rewards else {}
    if rewards and len(set(roles.values())) != 2:
        raise SystemExit('Distinct reward marker roles required')
    return {'guild': int(env['DISCORD_GUILD_ID']), 'invitation_channel': int(env['REFERRAL_INVITE_CHANNEL_ID']),
            'leaderboard_channel': int(env['REFERRAL_LEADERBOARD_CHANNEL_ID']) if leaderboard else None,
            'reward_roles': roles, 'apply_rewards': rewards, 'publish_leaderboard': leaderboard,
            'public_url': env.get('PUBLIC_URL', 'https://hyprlcore.com')}


def main():
    os.umask(0o077)
    env = os.environ
    if env.get('REFERRAL_ENABLED') != 'true':
        raise SystemExit('Referral networking disabled; operator activation required')
    config = configuration(env)
    file = Path(env['REFERRAL_LEDGER'])
    if not file.is_absolute():
        raise SystemExit('Absolute private ledger path required')
    file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(str(file) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Ledger(file, env['REFERRAL_ALIAS_KEY'])
        try:
            ReferralBot(config, ledger).run(env['DISCORD_BOT_TOKEN'], log_handler=None)
        except Exception:
            raise SystemExit('Referral service failed; details withheld') from None
        finally:
            ledger.close()


if __name__ == '__main__':
    main()
