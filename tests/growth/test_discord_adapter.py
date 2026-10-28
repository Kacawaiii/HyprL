"""Exercise discord.py callbacks with synthetic objects, never login or sync remotely."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from scripts.growth.discord_bot import ReferralBot
from scripts.growth.referrals import Ledger


@pytest.fixture
def config():
    return {'guild': 800000000000000001, 'invitation_channel': 800000000000000002, 'leaderboard_channel': 800000000000000003, 'reward_roles': {3: 700000000000000001, 10: 700000000000000002}, 'public_url': 'https://example.test'}


def test_command_creates_one_permanent_invite_and_responds_privately(tmp_path, config):
    async def run():
        ledger = Ledger(tmp_path / 'ledger.sqlite', 'synthetic-alias-key')
        bot = ReferralBot(config, ledger)
        invite = SimpleNamespace(code='synthetic-permanent-invite', delete=AsyncMock())
        channel = SimpleNamespace(create_invite=AsyncMock(return_value=invite))
        guild = SimpleNamespace(invites=AsyncMock(side_effect=[[], [invite]]), get_channel=lambda _id: channel)
        response = SimpleNamespace(defer=AsyncMock())
        followup = SimpleNamespace(send=AsyncMock())
        interaction = SimpleNamespace(guild_id=config['guild'], user=SimpleNamespace(id=900000000000000001, bot=False), guild=guild, response=response, followup=followup)
        bot.before = {}; bot.ready_for_joins = True
        command = bot.tree.get_command('parrainage')
        await command.callback(interaction)
        await command.callback(interaction)
        channel.create_invite.assert_awaited_once_with(max_age=0, max_uses=0, unique=True, reason='Programme de parrainage HyprL')
        assert response.defer.await_args.kwargs['ephemeral'] is True
        assert all(call.kwargs['ephemeral'] for call in followup.send.await_args_list)
        assert 'synthetic-permanent-invite' in followup.send.await_args.args[0]
        assert ledger.invite(interaction.user.id) == invite.code
        assert bot.intents.members and bot.intents.invites
        assert not bot.intents.message_content
        await bot.close(); ledger.close()
    asyncio.run(run())


def test_join_and_remove_callbacks_use_all_invites_and_disqualify_early_leaving(tmp_path, config):
    async def run():
        ledger = Ledger(tmp_path / 'ledger.sqlite', 'synthetic-alias-key')
        ledger.register('100', 'personal', 1)
        bot = ReferralBot(config, ledger)
        now = datetime.now(timezone.utc)
        guild = SimpleNamespace(id=config['guild'], invites=AsyncMock(return_value=[SimpleNamespace(code='personal', uses=1), SimpleNamespace(code='ordinary', uses=4)]))
        member = SimpleNamespace(id=200, guild=guild, joined_at=now, created_at=now - timedelta(days=10), bot=False)
        bot.before = {'personal': 0, 'ordinary': 4}; bot.ready_for_joins = True
        await bot.on_member_join(member)
        assert ledger.counts('100') == {'pending': 1}
        await bot.on_member_remove(member)
        assert ledger.counts('100') == {'rejected': 1}
        await bot.close(); ledger.close()
    asyncio.run(run())


def test_reconciliation_fetches_current_members_and_role_grant_is_retry_safe(tmp_path, config):
    async def run():
        ledger = Ledger(tmp_path / 'ledger.sqlite', 'synthetic-alias-key')
        now = datetime.now(timezone.utc).timestamp()
        ledger.register('100', 'personal', now - 86400 * 9)
        for i in range(3):
            ledger.join(str(200 + i), 'personal', now - 86400 * 30, now - 86400 * 8)
        bot = ReferralBot(config, ledger)
        sponsor = SimpleNamespace(add_roles=AsyncMock())
        async def fetch(mid):
            return sponsor if mid == 100 else SimpleNamespace(joined_at=datetime.fromtimestamp(now - 86400 * 8, timezone.utc))
        class Role:
            managed = False
            def __ge__(self, _other): return False
        role = Role()
        channel = SimpleNamespace(send=AsyncMock())
        guild = SimpleNamespace(fetch_member=AsyncMock(side_effect=fetch), get_role=lambda _id: role, get_channel=lambda _id: channel, me=SimpleNamespace(top_role=object()))
        bot.get_guild = lambda _id: guild
        await bot.reconcile(); await bot.reconcile()
        sponsor.add_roles.assert_awaited_once()
        assert ledger.rewards()[0]['state'] == 'role_granted'
        assert ledger.db.execute('SELECT state FROM credits').fetchone()[0] == 'pending_stripe'
        assert channel.send.await_count <= 1
        await bot.close(); ledger.close()
    asyncio.run(run())


def test_api_outage_never_matures_or_rejects_members_and_does_not_grant_roles(tmp_path, config):
    async def run():
        ledger = Ledger(tmp_path / 'ledger.sqlite', 'synthetic-alias-key')
        now = datetime.now(timezone.utc).timestamp()
        ledger.register('100', 'personal', now)
        ledger.join('200', 'personal', now - 86400 * 30, now - 86400 * 8)
        bot = ReferralBot(config, ledger)
        guild = SimpleNamespace(fetch_member=AsyncMock(side_effect=RuntimeError('synthetic outage')))
        bot.get_guild = lambda _id: guild
        with pytest.raises(RuntimeError, match='synthetic outage'):
            await bot.reconcile()
        assert ledger.counts('100') == {'pending': 1} and ledger.rewards() == []
        await bot.close(); ledger.close()
    asyncio.run(run())
