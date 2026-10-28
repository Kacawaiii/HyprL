"""Synthetic identities only. No Discord connection or billing operation."""
import os
import json
import subprocess
import sys

import pytest

from scripts.growth.referrals import Ledger, WEEK, attributed, expires

NOW = 1793188800


@pytest.fixture
def ledger(tmp_path):
    store = Ledger(tmp_path / 'ledger.sqlite', 'synthetic-alias-key')
    store.register('100', 'synthetic-personal-invite', NOW)
    yield store
    store.close()


@pytest.mark.parametrize('before,after,want', [
    (None, {'a': 1}, None), ({'a': 0}, {'a': 1}, 'a'),
    ({'a': 1}, {'a': 1}, None), ({'a': 0}, {'a': 2}, None),
    ({'a': 0, 'b': 0}, {'a': 1, 'b': 1}, None),
    ({'a': 0}, {'a': 1, 'unknown': 1}, None),
    ({'a': 1}, {'a': 0, 'b': 1}, None),
    ({'a': 0, 'deleted': 2}, {'a': 1}, None),
    ({'a': 0, 'ordinary': 4}, {'a': 0, 'ordinary': 5}, 'ordinary'),
])
def test_attribution_never_guesses(before, after, want):
    assert attributed(before, after) == want


def test_age_boundary_and_complete_seven_day_wait(ledger):
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.join('201', 'synthetic-personal-invite', NOW - WEEK + 1, NOW)
    ledger.mature(NOW + WEEK - 1, {'200': NOW, '201': NOW})
    assert ledger.counts('100') == {'pending': 1, 'rejected': 1}
    ledger.mature(NOW + WEEK, {'200': NOW, '201': NOW})
    assert ledger.counts('100') == {'valid': 1, 'rejected': 1}


def test_early_leave_rejoin_bot_self_and_unattributed_never_count(ledger):
    for member, code, bot in [('200', 'synthetic-personal-invite', False), ('100', 'synthetic-personal-invite', False), ('201', 'synthetic-personal-invite', True), ('202', None, False)]:
        ledger.join(member, code, NOW - WEEK, NOW, bot=bot)
    ledger.leave('200', NOW + WEEK - 1)
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW + WEEK)
    ledger.join('202', 'synthetic-personal-invite', NOW - WEEK, NOW + WEEK)
    ledger.mature(NOW + 2 * WEEK, {str(i): NOW for i in [100, 200, 201, 202]})
    assert ledger.counts('100') == {'rejected': 3}
    assert ledger.rewards() == []


def test_departure_at_exact_boundary_and_missing_member(ledger):
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.join('201', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.leave('200', NOW + WEEK)
    ledger.mature(NOW + WEEK, {})
    assert ledger.counts('100') == {'valid': 1, 'rejected': 1}


def test_offline_leave_rejoin_is_detected_from_current_join_time(ledger):
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.mature(NOW + WEEK, {'200': NOW + 100})
    assert ledger.counts('100') == {'rejected': 1}


def test_rewards_durable_once_and_credit_never_claims_payment(ledger):
    present = {}
    for i in range(10):
        member = str(200 + i)
        ledger.join(member, 'synthetic-personal-invite', NOW - WEEK, NOW)
        present[member] = NOW
        ledger.mature(NOW + WEEK, present)
        expected = int(i >= 2) + int(i >= 9)
        assert len(ledger.rewards()) == expected
    ledger.mark_role('100', 3)
    ledger.mature(NOW + WEEK, present)
    assert [r['state'] for r in ledger.rewards()] == ['role_granted', 'pending']
    assert tuple(ledger.db.execute('SELECT months,state FROM credits').fetchone()) == (1, 'pending_stripe')
    again = Ledger(ledger.file, 'synthetic-alias-key')
    try:
        again.mature(NOW + WEEK, present)
        assert len(again.rewards()) == 2
        assert again.invite('100') == 'synthetic-personal-invite'
    finally:
        again.close()


def test_private_files_weekly_reservation_and_anonymous_output(ledger):
    assert os.stat(ledger.file).st_mode & 0o777 == 0o600
    assert ledger.reserve_post('2026-10-26')
    assert not ledger.reserve_post('2026-10-26')
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.mature(NOW + WEEK, {'200': NOW})
    [(alias, n)] = ledger.leaderboard()
    assert alias.startswith('Parrain-') and n == 1
    assert '100' not in alias and 'synthetic-personal-invite' not in alias


def test_retention_and_erasure_do_not_resurrect_records(ledger):
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    ledger.mature(NOW + WEEK, {'200': NOW})
    ledger.purge(expires(NOW) - 1)
    assert ledger.invite('100')
    ledger.purge(expires(NOW))
    assert ledger.invite('100') is None and ledger.leaderboard() == []
    ledger.register('300', 'another-synthetic-invite', NOW)
    ledger.erase('300')
    assert ledger.invite('300') is None


def test_unsafe_directory_refused(tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match='0700'):
        Ledger(tmp_path / 'ledger.sqlite', 'synthetic-key')


def test_rights_export_is_individual_private_and_erasure_keeps_other_members(tmp_path, ledger):
    ledger.join('200', 'synthetic-personal-invite', NOW - WEEK, NOW)
    output = tmp_path / 'individual.json'
    result = subprocess.run([sys.executable, '-m', 'scripts.growth.admin', '--ledger', str(ledger.file), '--member', '100', '--export-to', str(output)], check=True, capture_output=True, text=True)
    personal = json.loads(output.read_text())
    assert personal['sponsors'][0]['member'] == '100'
    assert personal['arrivals'] == []
    assert 'synthetic-personal-invite' not in result.stdout
    assert output.stat().st_mode & 0o777 == 0o600
    subprocess.run([sys.executable, '-m', 'scripts.growth.admin', '--ledger', str(ledger.file), '--member', '100', '--erase'], check=True, capture_output=True)
    assert ledger.invite('100') is None
    row = ledger.db.execute('SELECT * FROM arrivals WHERE member=?', ('200',)).fetchone()
    assert row['sponsor'] is None  # No unrelated person's data is deleted or exported.
