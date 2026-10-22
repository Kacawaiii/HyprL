"""Exercise live transport boundaries with in-memory wire replies, never live providers."""
from datetime import timedelta
from hashlib import sha256
import json

import pytest

from scripts.trading_lab.agent_v2_quality.contracts import Blocked, digest, instant, iso
from scripts.trading_lab.agent_v2_quality.deploy import activate
from scripts.trading_lab.agent_v2_quality.ledger import Journal
from scripts.trading_lab.agent_v2_quality.models import Models
from scripts.trading_lab.agent_v2_quality.producer import Client, Producer, radar_headlines
from scripts.trading_lab.agent_v2_quality.service import Service
from scripts.trading_lab.agent_v2_quality.spec import PREREG_HASH, SPEC_HASH
from scripts.trading_lab.agent_v2_quality.synthetic import Clock, SyntheticModels, bundle


class Grant:
    def check(self):
        return {'models': {'claude_opus_calls_per_day': 3, 'gpt_calls_per_day': 3}}


@pytest.fixture
def forward(tmp_path):
    clock = Clock(instant('2026-10-23T12:45:00Z'))
    journal = Journal.initialize(tmp_path / 'forward', 'SHADOW_FORWARD', 'c' * 40, clock())
    yield journal, clock
    journal.close()


def client(forward, tmp_path, send):
    journal, clock = forward
    credentials = tmp_path / 'synthetic-credentials'
    credentials.write_text('APCA_API_KEY_ID=synthetic-key\nAPCA_API_SECRET_KEY=synthetic-value\n')
    sec = tmp_path / 'synthetic-sec-ua.json'
    sec.write_text(json.dumps({'granted_at': '2026-01-01T00:00:00Z', 'not_after': '2027-01-01T00:00:00Z',
                              'scope': {'https://www.sec.gov': {'user_agent': 'Synthetic Test Agent'}}}))
    def advance(seconds):
        clock.at += timedelta(seconds=seconds)
    return Client(journal, Grant(), credentials, sec, clock=clock, send=send, pause=advance)


def test_http_reserved_before_dispatch_received_after_full_response_and_halted(forward, tmp_path):
    journal, clock = forward
    count = []
    def send(url, headers):
        count.append(url)
        assert len(journal.records('http_reservation')) == len(count)
        with journal.owned():  # network does not own the writer lock
            journal.verify()
        clock.at += timedelta(seconds=2)
        return 200 if len(count) == 1 else 429, b'{"quotes":{}}'
    value = client(forward, tmp_path, send)
    url = 'https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=ACME&feed=iex'
    raw, receipt = value.get(url)
    assert receipt['received_at'] == '2026-10-23T12:45:02Z'
    assert receipt['sha256'] == sha256(raw).hexdigest()
    with pytest.raises(Blocked, match='HTTP_429'):
        value.get(url)
    with pytest.raises(Blocked, match='HTTP_HALTED'):
        value.get(url)
    assert len(count) == 2


def test_sec_pacing_covers_submissions_and_document_same_budget(forward, tmp_path):
    journal, clock = forward
    value = client(forward, tmp_path, lambda url, headers: (200, b'{}'))
    value.get('https://data.sec.gov/submissions/CIK0000000001.json')
    value.get('https://www.sec.gov/Archives/edgar/data/1/000000000126000001/a.htm')
    rows = journal.records('http_reservation')
    assert rows[0]['at'] == '2026-10-23T12:45:00Z'
    assert rows[1]['at'] == '2026-10-23T12:45:01Z'
    assert {r['body']['group'] for r in rows} == {'sec'}


def test_live_runtime_rejects_inactive_and_synthetic_runner(forward):
    journal, clock = forward
    with pytest.raises(Blocked, match='SYNTHETIC_RUNNER_FORBIDDEN'):
        Service(journal, SyntheticModels(), clock=clock)
    with pytest.raises(Blocked, match='RUNTIME_NOT_ACTIVATED'):
        Service(journal, Models(journal, Grant(), clock=clock), clock=clock)


def test_activation_requires_grant_binding_review_ci_and_recent_producer(forward):
    journal, clock = forward
    clock.at = instant('2026-10-23T18:00:00Z')
    proof = {'schema': 'agent-v2-quality-deploy-proof-v2', 'sha': 'c' * 40,
             'spec_hash': SPEC_HASH, 'preregistration_hash': PREREG_HASH,
             'authorization_sha256': digest(Grant().check()), 'ci_conclusion': 'success',
             'local_checks': 'PASS', 'single_owner_reviewed': True, 'replay_waived_by_grant': True,
             'ci_url': 'https://github.com/example/synthetic/actions/runs/1', 'producer_snapshot_hash': 'd' * 64}
    with pytest.raises(Blocked, match='RECENT_PRODUCER_PROOF_REQUIRED'):
        activate(journal, Grant(), proof, clock=clock)
    with journal.owned():
        journal.append([('producer_snapshot', {'hash': 'd' * 64})], clock())
    for field, value in [('single_owner_reviewed', False), ('authorization_sha256', 'e' * 64),
                         ('ci_conclusion', 'pending'), ('sha', 'e' * 40), ('local_checks', 'BLOCKED')]:
        with pytest.raises(Blocked, match='DEPLOY_EVIDENCE_REQUIRED'):
            activate(journal, Grant(), proof | {field: value}, clock=clock)
    assert not journal.records('activation')
    assert activate(journal, Grant(), proof, clock=clock)['status'] == 'ACTIVATED_SHADOW_ONLY'
    assert journal.records('activation')[0]['body']['authorization_sha256'] == digest(Grant().check())
    with pytest.raises(Blocked, match='ALREADY_ACTIVATED'):
        activate(journal, Grant(), proof, clock=clock)


def test_models_reserve_durably_and_release_writer_before_cli(forward, monkeypatch):
    journal, clock = forward
    invocations = []
    class Process:
        returncode = 0
        def __init__(self, argv, **kwargs):
            invocations.append(argv)
            assert len(journal.records('model_reservation')) == len(invocations)
            self.output = kwargs['stdout']
        def communicate(self, *args, **kwargs):
            with journal.owned():
                journal.verify()
            self.output.write(json.dumps({'type': 'result', 'modelUsage': {'claude-opus-synthetic': {}},
                                          'structured_output': {'facts': []}}))
    monkeypatch.setattr('subprocess.Popen', Process)
    runner = Models(journal, Grant(), clock=clock)
    for _ in range(3):
        assert runner.call('extraction', {'sources': []}) == {'facts': []}
    with pytest.raises(Blocked, match='MODEL_BUDGET'):
        runner.call('extraction', {'sources': []})
    assert len(invocations) == 3


def test_usage_limit_stops_all_remaining_model_calls(forward, monkeypatch):
    journal, clock = forward
    class Process:
        returncode = 1
        def __init__(self, argv, **kwargs):
            self.err = kwargs['stderr']
        def communicate(self, *args, **kwargs):
            self.err.write('Usage limit reached')
    monkeypatch.setattr('subprocess.Popen', Process)
    runner = Models(journal, Grant(), clock=clock)
    with pytest.raises(Blocked, match='MODEL_USAGE_LIMIT'):
        runner.call('extraction', {})
    with pytest.raises(Blocked, match='MODEL_USAGE_LIMIT'):
        runner.call('contradictor', {})
    assert len(journal.records('model_reservation')) == 1


def test_radar_prices_and_crypto_do_not_enter_model_context():
    at = instant('2026-10-23T12:45:00Z')
    story = {'id': 'a', 'headline': 'Synthetic issuer guidance', 'url': 'https://example.test/a',
             'published_at': '2026-10-23T11:00:00Z', 'received_at': '2026-10-23T11:01:00Z'}
    radar = {'schema': 'news-radar-v3', 'live': True, 'status': 'PARTIAL', 'cutoff': iso(at),
             'regime': {'protected_price': 'DO_NOT_USE'}, 'anomalies': [{'price': 'DO_NOT_USE'}],
             'events': [{'entities': {'symbols': ['ACME']}, 'stories': [story]},
                        {'entities': {'symbols': ['BTC/USD', 'ACME']}, 'stories': [story | {'id': 'b'}]}]}
    value = radar_headlines(radar, ['ACME'], at)
    assert len(value['headlines']) == 1
    assert 'DO_NOT_USE' not in json.dumps(value)


def test_producer_consumes_observed_envelopes_and_all_pages(tmp_path):
    clock = Clock(instant('2026-10-23T12:45:00Z'))
    data = bundle(clock())
    paths = {}
    for key, value in [('registry', data['registry']), ('calendar', data['calendar']),
                       ('hunter', data['snapshot']['hunter']),
                       ('radar', {'schema': 'news-radar-v3', 'status': 'PARTIAL', 'live': True,
                                  'cutoff': iso(clock()), 'events': []})]:
        paths[key] = tmp_path / (key + '.json')
        paths[key].write_text(json.dumps(value))
    class Wire:
        calls = 0
        def json(self, url):
            self.calls += 1
            assert 'feed=iex' in url and 'adjustment=raw' in url
            selected = data['snapshot']['instruments'][:1] if self.calls == 1 else data['snapshot']['instruments'][1:]
            assert ('page_token=next' in url) == (self.calls == 2)
            return {'bars': {r['symbol']: [{k: b[k] for k in ('t', 'c', 'h', 'l', 'v')}
                                           for b in r['bars']] for r in selected},
                    'next_page_token': 'next' if self.calls == 1 else None}, {'received_at': iso(clock())}
    wire = Wire()
    result = Producer(wire, clock=clock).snapshot(*(paths[k] for k in ('registry', 'calendar', 'hunter', 'radar')))
    assert wire.calls == 2
    assert result['snapshot']['mode'] == 'SHADOW_FORWARD'
    assert len(result['snapshot']['instruments']) == 3
    assert all(b['available_at'] == iso(clock()) for r in result['snapshot']['instruments'] for b in r['bars'])
