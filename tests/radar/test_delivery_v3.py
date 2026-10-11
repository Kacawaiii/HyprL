"""Operator-only POST boundary, durable dedupe and French digest behavior."""
from datetime import timedelta
import json

import pytest

from scripts.radar.core import RadarError
from scripts.radar.service import radar
from scripts.radar.synthetic import seed
from tests.radar.test_radar import state, story
from tests.radar.test_official_v3 import official


@pytest.fixture
def telegram_credentials(tmp_path):
    path = tmp_path / 'synthetic-telegram.env'
    path.write_text('TELEGRAM_BOT_TOKEN=123456:synthetic_token\nADMIN_TELEGRAM_ID=12345678\n')
    return path


def test_french_digest_keeps_top_five_mechanism_priced_in_anomalies_and_full_path(state):
    from scripts.radar.telegram import short_digest
    store, _, _, _ = state
    seed(store)
    report = radar(store)
    message = short_digest(report, '/synthetic/reports/radar-full.md')
    assert len(message.encode('utf-16-le')) // 2 <= 3500
    assert 'Régime:' in message and 'Mécanisme:' in message and 'Déjà intégré:' in message
    assert 'Anomalies:' in message and 'causalité' in message
    assert '/synthetic/reports/radar-full.md' in message
    # Long/untrusted feed text cannot crowd out required sections or inject links.
    report['events'][0]['headline'] = '😀' * 4000
    report['events'][0]['transmission_hypotheses'][0]['mechanism'] = 'É' * 6000
    bounded = short_digest(report, '/synthetic/reports/radar-full.md')
    assert len(bounded.encode('utf-16-le')) // 2 <= 3500 and bounded.endswith('radar-full.md')


def test_digest_sent_only_to_operator_once_after_restart(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    from scripts.radar.core import Store
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    def post(url, payload):
        calls.append((url, payload))
        return 200, b'{"ok":true,"result":{"message_id":42}}'
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials, post=post)
    assert sender.send('digest-day-morning', 'Radar synthétique')['status'] == 'SENT'
    restarted = TelegramDelivery(Store(store.root, clock=clock), grant, credentials=telegram_credentials, post=post)
    assert restarted.send('digest-day-morning', 'Radar synthétique')['status'] == 'DEDUPLICATED'
    assert len(calls) == 1 and calls[0][1]['chat_id'] == '12345678'
    assert 'parse_mode' not in calls[0][1]
    assert calls[0][1]['disable_web_page_preview'] is True
    assert 'synthetic_token' not in json.dumps(store.rows('delivery'))
    assert store.counts() == {'telegram': 1}


def test_failed_or_uncertain_delivery_consumes_budget_and_never_blindly_retries(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    store, _, _, _ = state
    grant, _ = official
    calls = []
    def post(*args):
        calls.append(args)
        raise RuntimeError('https://api.telegram.org/bot123456:synthetic_token/sendMessage')
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials, post=post)
    assert sender.send('once', 'Synthétique')['status'] == 'UNCONFIRMED'
    assert sender.send('once', 'Synthétique')['status'] == 'DEDUPLICATED'
    assert len(calls) == 1 and store.counts()['telegram'] == 1
    assert 'synthetic_token' not in json.dumps(store.rows('delivery'))


def test_total_messages_never_exceed_twelve_and_reset_next_utc_day(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    store, _, clock, _ = state
    grant, _ = official
    sent = []
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials,
                              post=lambda *args: (sent.append(args) or 200, b'{"ok":true,"result":{"message_id":1}}'))
    for i in range(12):
        assert sender.send(str(i), 'Synthétique')['status'] == 'SENT'
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        sender.send('over', 'Synthétique')
    assert len(sent) == 12
    clock.at += timedelta(days=1)
    assert sender.send('tomorrow', 'Synthétique')['status'] == 'SENT'


def test_alerts_require_eighty_importance_and_recent_publisher_time(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    from scripts.radar.analysis import cluster
    store, _, clock, _ = state
    grant, _ = official
    events = cluster([story('high', headline='Federal Reserve interest rate policy', primary=True),
                      story('low', headline='Synthetic small company update'),
                      story('old', headline='Federal Reserve inflation policy earlier', primary=True, published=clock.at-timedelta(days=2))])
    events[0]['importance'] = 80
    calls = []
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials,
                              post=lambda *args: (calls.append(args) or 200, b'{"ok":true,"result":{"message_id":1}}'))
    sender.alerts(events, '/synthetic/reports/radar-full.md')
    sender.alerts(events, '/synthetic/reports/radar-full.md')
    assert len(calls) == 1 and 'ALERTE' in calls[0][1]['text']


def test_alert_burst_reserves_room_for_morning_and_evening_digests(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    store, _, _, _ = state
    grant, _ = official
    calls = []
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials,
                              post=lambda *args: (calls.append(args) or 200, b'{"ok":true,"result":{"message_id":1}}'))
    for i in range(10):
        sender.send('alert:' + str(i), 'Synthétique', reserve_digests=True)
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        sender.send('alert:over', 'Synthétique', reserve_digests=True)
    sender.send('morning', 'Synthétique', digest_slot='morning')
    sender.send('evening', 'Synthétique', digest_slot='evening')
    assert len(calls) == 12


def test_delivery_expiry_and_recipient_scope_are_checked_before_post(state, official, telegram_credentials):
    from scripts.radar.telegram import TelegramDelivery
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    sender = TelegramDelivery(store, grant, credentials=telegram_credentials, post=lambda *args: calls.append(args))
    grant.payload['scope']['https://api.telegram.org']['recipient'] = 'someone else'
    with pytest.raises(RadarError, match='OUTSIDE_GRANT'):
        sender.send('wrong', 'Synthétique')
    clock.at += timedelta(days=3)
    with pytest.raises(RadarError, match='EXPIRED'):
        sender.send('expired', 'Synthétique')
    assert not calls and store.counts() == {}


def test_official_deployment_enables_only_radar_units_and_delivery():
    from scripts.radar.deploy import unit_texts
    units = unit_texts('/synthetic/bin/python', '/synthetic/worktree', official=True)
    assert len(units) == 8 and all(name.startswith('hyprl-radar-') for name in units)
    assert '*:02/5:00 UTC' in units['hyprl-radar-official.timer']
    assert 'scripts.radar.service official' in units['hyprl-radar-official.service']
    assert ' --telegram' in units['hyprl-radar-morning.service']
    assert ' --telegram' in units['hyprl-radar-evening.service']


def test_cached_radar_uses_receipts_without_any_http_dispatch(state):
    from scripts.radar.core import Client, iso
    store, grant, clock, _ = state
    seed(store)
    store.append('health', 'synthetic', {'source': 'synthetic', 'status': 'LIVE', 'items': 1, 'checked_at': iso(clock.at)})
    grant.payload['budgets']['llm'] = 0
    calls = []
    report = radar(store, grant, live=True, cached=True, client=Client(store, grant, send=lambda *args: calls.append(args)))
    assert report['schema'] == 'news-radar-v3' and report['status'] == 'PARTIAL'
    assert report['llm']['status'] == 'DAILY_BUDGET'
    assert any('Relecture' in line for line in report['limitations'])
    assert calls == []
