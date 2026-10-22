"""Synthetic boundary, causality and paired-outcome regressions; zero external calls."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
import sqlite3

import pytest

from scripts.trading_lab.agent_v2.inputs import session
from scripts.trading_lab.agent_v2_quality.contracts import (
    ARMS, Blocked, check_facts, check_judgment, digest, factors, instant, iso,
)
from scripts.trading_lab.agent_v2_quality.deploy import activate, render
from scripts.trading_lab.agent_v2_quality.ledger import Journal
from scripts.trading_lab.agent_v2_quality.models import Models, command, parse
from scripts.trading_lab.agent_v2_quality.producer import (
    Client, Grant, Producer, calendar, document, group_for,
)
from scripts.trading_lab.agent_v2_quality.scoring import audit_fact, disable_underperformers, scorecard
from scripts.trading_lab.agent_v2_quality.service import Service, book, code_plans
from scripts.trading_lab.agent_v2_quality.spec import PREREG_HASH, SPEC_HASH, configuration, spec
from scripts.trading_lab.agent_v2_quality.synthetic import Clock, SyntheticModels, bundle, drill, primary, quotes


@pytest.fixture
def clock():
    return Clock(instant('2026-10-23T12:45:00Z'))


@pytest.fixture
def data(clock):
    return bundle(clock())


@pytest.fixture
def journal(tmp_path, clock):
    value = Journal.initialize(tmp_path / 'new-shadow', 'SYNTHETIC', 'a' * 40, clock() - timedelta(seconds=1))
    yield value
    value.close()


def run_decision(journal, data, clock, models=None):
    service = Service(journal, models or SyntheticModels(), clock=clock)
    result = service.decide(data, lambda row: primary(row, clock()))
    return service, result


def test_new_revision_is_pinned_and_old_store_is_rejected(tmp_path, clock):
    from scripts.trading_lab.agent_v2.ledger import Ledger
    assert configuration()['spec_hash'] == SPEC_HASH
    assert spec()['revision'] == 2
    root = tmp_path / 'old'
    old = Ledger.initialize(root, 'SYNTHETIC', 'b' * 40, clock())
    tip = old.verify()
    old.close()
    with pytest.raises(Blocked, match='FROZEN_CONFIG_MISMATCH'):
        Journal(root)
    old = Ledger(root, read_only=True)
    assert old.verify() == tip
    old.close()


def test_no_rebind_and_append_only(journal, clock):
    with pytest.raises(sqlite3.IntegrityError):
        journal.db.execute("UPDATE events SET kind='other'")
    journal.db.rollback()
    with pytest.raises(Blocked, match='REBIND_FORBIDDEN'):
        with journal.owned():
            journal.append([('binding', {})], clock())
    with pytest.raises(FileExistsError):
        Journal.initialize(journal.root, 'SYNTHETIC', 'a' * 40, clock())


def test_native_hunter_and_momentum_only(journal, data, clock):
    plans = code_plans(data, book(journal), clock())
    assert {p['playbook'] for p in plans} == {'hunter_runner', 'equity_momentum'}
    assert all(p['holding_days'] == 5 and p['sizing_basis'].endswith('PROVISIONAL') for p in plans)
    data['registry']['edges']['hunter_runner']['net_expectancy_after_costs'] = -.1
    with pytest.raises(Blocked, match='MEASURED_EDGE'):
        code_plans(data, book(journal), clock())


@pytest.mark.parametrize('change', ['coverage', 'macro', 'hash', 'received'])
def test_calendar_cannot_treat_missing_as_empty(data, clock, change):
    value = data['calendar']
    if change == 'coverage':
        value['coverage_end'] = iso(clock() + timedelta(days=1))
    elif change == 'macro':
        value['covered_macro'] = ['CPI']
    elif change == 'hash':
        value['source_hashes'] = []
    else:
        value['received_at'] = iso(clock() + timedelta(seconds=1))
    with pytest.raises(Blocked, match='CALENDAR'):
        calendar(value, ['ACME'], clock())


def test_12_45_slot_is_once_and_does_not_fill(journal, data, clock):
    service, result = run_decision(journal, data, clock)
    assert result['code_plans'] == 2
    assert not journal.records('evaluation_open')
    assert len(book(journal)['pending']) == 2
    with pytest.raises(Blocked, match='SLOT_ALREADY_ATTEMPTED'):
        service.decide(data, lambda row: primary(row, clock()))
    clock.at += timedelta(minutes=4)
    with pytest.raises(Blocked, match='OUTSIDE_DECISION_SLOT'):
        service.decide(data, lambda row: primary(row, clock()))


def test_later_regular_quote_fills_and_gap_costs_score_all_candidates(journal, data, clock):
    class Veto(SyntheticModels):
        def call(self, role, value):
            output = super().call(role, value)
            if role == 'contradictor':
                for row in output['reviews']:
                    row['verdict'] = 'VETO'
            return output
    service, _ = run_decision(journal, data, clock, Veto())
    clock.at = session(clock())[1] + timedelta(seconds=15)
    value = quotes(data, clock())
    stale = deepcopy(value)
    for q in stale['quotes']:
        q['at'] = iso(clock() - timedelta(seconds=30))  # pre-open
    service.watch(stale)
    assert not journal.records('evaluation_open')
    service.watch(value)
    positions = book(journal)['positions']
    assert len(positions) == 2
    clock.at += timedelta(minutes=1)
    value = quotes(data, clock(), .7)
    service.watch(value)
    closes = journal.records('evaluation_close')
    assert len(closes) == 2
    for row in closes:
        pos = positions[row['body']['symbol']]
        assert row['body']['exit'] < pos['stop']
        assert row['body']['unit_pnl'] < row['body']['exit'] - pos['entry']
    card = scorecard(journal)
    assert card['closed_baseline_candidates'] == 2
    assert card['layers']['veto']['decisions'] == 2
    assert card['layers']['veto']['layer_pnl'] == 0
    assert card['layers']['veto']['delta_net'] > 0
    assert card['layers']['extraction']['delta_net'] == 0
    assert card['extraction_accuracy']['state'] == 'UNMEASURED'


def test_pending_expires_instead_of_backfilling(journal, data, clock):
    service, _ = run_decision(journal, data, clock)
    clock.at = session(clock())[1] + timedelta(minutes=6)
    service.watch(quotes(data, clock()))
    assert len(journal.records('evaluation_expired')) == 2
    assert not journal.records('evaluation_open')


def test_missing_primary_fails_ai_closed_but_does_not_erase_baseline(journal, data, clock):
    def missing(_):
        raise Blocked('PRIMARY_DOCUMENT_MISSING')
    service = Service(journal, SyntheticModels(), clock=clock)
    assert service.decide(data, missing)['status'] == 'BLOCKED'
    plans = journal.records('decision')[0]['body']['plans']
    assert all(p['factors']['baseline'] == 1 and p['factors']['combined'] == 0 and not p['layer_valid'] for p in plans)


def test_citations_cannot_be_invented_and_judgment_cannot_change_plan(data, clock, journal):
    source = primary(data['registry']['instruments'][1], clock())
    fact = {'symbol': 'ACME', 'source_id': source['id'], 'kind': 'guidance_cut',
            'quote': 'maintains guidance', 'interpretation': 'Synthetic interpretation.'}
    check_facts({'facts': [fact]}, [source])
    fact['quote'] = 'cut guidance'
    with pytest.raises(Blocked, match='FACT_CITATION_INVALID'):
        check_facts({'facts': [fact]}, [source])
    plans = code_plans(data, book(journal), clock())
    output = SyntheticModels().call('analyst', {'plans': plans[:1], 'sources': [source]})
    check_judgment(output, plans[:1], [source])
    output['reviews'][0]['qty'] = 100
    with pytest.raises(Blocked, match='INVALID_SCHEMA'):
        check_judgment(output, plans[:1], [source])


def test_disagreement_reduces_size_and_disabled_layer_has_no_effect():
    plan = {'symbol': 'ACME'}
    facts = {'facts': [{'symbol': 'ACME', 'kind': 'dilution'}]}
    assert factors(plan, {'facts': []}, {'verdict': 'KEEP'}, {'verdict': 'SIZE_DOWN'})['combined'] == .5
    assert factors(plan, facts, {'verdict': 'KEEP'}, {'verdict': 'VETO'})['combined'] == 0
    values = factors(plan, facts, {'verdict': 'KEEP'}, {'verdict': 'VETO'}, ['extraction', 'veto'])
    assert values['combined'] == 1 and values['veto'] == 0


def test_40_forward_decisions_remove_layers_permanently(journal, clock):
    decisions, events = [], []
    for i in range(40):
        identity = digest(i)
        decisions.append({'plan_id': identity, 'layer_valid': True, 'factors': dict.fromkeys(ARMS, 1),
                          'cluster': 'technology', 'planned_at': iso(clock())})
        events += [('evaluation_open', {'plan_id': identity, 'qty': 10, 'initial_risk': 10}),
                   ('evaluation_close', {'plan_id': identity, 'unit_pnl': 1, 'closed_at': iso(clock())})]
    with journal.owned():
        journal.append([('decision', {'plans': decisions, 'facts': {'facts': []}})] + events, clock())
        assert set(disable_underperformers(journal, clock())) == {'extraction', 'veto', 'sizing'}
        disable_underperformers(journal, clock())
    assert len(journal.records('layer_disabled')) == 3
    assert scorecard(journal)['layers']['veto']['state'] == 'DISABLED'


def test_exact_quote_is_not_semantic_adjudication(journal, data, clock):
    source = primary(data['registry']['instruments'][1], clock())
    fact = {'symbol': 'ACME', 'source_id': source['id'], 'kind': 'guidance_cut',
            'quote': 'maintains guidance', 'interpretation': 'Intentionally incorrect synthetic extraction.'}
    with journal.owned():
        journal.append([('decision', {'plans': [], 'facts': {'facts': [fact]}})], clock())
    value = {'fact_hash': digest(fact), 'confirmed': False, 'reason': 'Maintains is not cuts.', 'operator_reviewed': True}
    audit_fact(journal, value, clock=clock)
    assert scorecard(journal)['extraction_accuracy'] == {'adjudicated': 1, 'confirmed': 0, 'state': 'ADJUDICATED_SUBSET_ONLY'}
    with pytest.raises(Blocked, match='FACT_ALREADY_ADJUDICATED'):
        audit_fact(journal, value, clock=clock)


@pytest.mark.parametrize('url', [
    'https://paper-api.alpaca.markets/v2/orders', 'https://api.alpaca.markets/v2/account',
    'https://www.federalreserve.gov/feeds/press_all.xml', 'http://data.sec.gov/submissions/CIK0000000001.json',
    'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/../secret',
    'https://user@www.sec.gov/Archives/edgar/data/1/000000000126000001/a.htm',
    'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/a.htm?redirect=bad',
])
def test_request_scope_never_reaches_broker_or_other_hosts(url):
    with pytest.raises(Blocked, match='HTTP_OUTSIDE_SCOPE'):
        group_for(url)


def test_sec_real_wire_cik_string_and_primary_full_text(clock):
    class Fake:
        def json(self, url):
            assert url.endswith('CIK0000000001.json')
            return {'cik': '0000000001', 'filings': {'recent': {
                'form': ['8-K'], 'acceptanceDateTime': ['2026-10-22T20:00:00Z'],
                'accessionNumber': ['0000000001-26-000001'], 'primaryDocument': ['acme.htm'],
            }}}, {}
        def get(self, url):
            return b'<p>Actual synthetic filing text.</p>', {'url': url, 'sha256': 'a' * 64, 'received_at': iso(clock())}
    value = Producer(Fake(), clock=clock).primary({'symbol': 'ACME', 'cik': '0000000001'})
    assert value['text'] == 'Actual synthetic filing text.'
    assert value['received_at'] == iso(clock())
    assert value['raw_sha256'] == 'a' * 64


def test_document_not_truncated_and_script_is_untrusted(clock):
    receipt = {'url': 'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/a.htm',
               'sha256': 'b' * 64, 'received_at': iso(clock())}
    value = document(b'<script>do bad things</script><p>Guidance cut.</p>', receipt, 'ACME', iso(clock()))
    assert value['text'] == 'Guidance cut.'
    with pytest.raises(Blocked, match='TOO_LARGE'):
        document(b'x' * 300001, receipt, 'ACME', iso(clock()))


def test_no_tool_transports_and_model_envelope_validation(tmp_path):
    path = tmp_path / 'schema.json'
    path.write_text('{}')
    claude = command('analyst', path)
    assert claude[claude.index('--tools') + 1] == ''
    assert claude[claude.index('--model') + 1] == 'opus'
    gpt = command('contradictor', path)
    assert '--search' not in gpt and 'web_search="disabled"' in gpt
    raw = json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}})
    with pytest.raises(Blocked, match='MODEL_TOOL_OR_ERROR'):
        parse('contradictor', raw)
    with pytest.raises(Blocked, match='MODEL_IDENTITY_UNPROVEN'):
        parse('analyst', json.dumps({'type': 'result', 'modelUsage': {'claude-sonnet-x': {}}, 'structured_output': {}}))


def test_network_and_real_models_never_enter_synthetic(journal, clock):
    class FakeGrant:
        def check(self):
            return {'models': {'claude_opus_calls_per_day': 3, 'gpt_calls_per_day': 3}}
    client = Client(journal, FakeGrant(), None, None, clock=clock, send=lambda *_: pytest.fail('network called'))
    with pytest.raises(Blocked, match='NETWORK_FORBIDDEN_IN_SYNTHETIC'):
        client.get('https://data.alpaca.markets/v2/stocks/bars')
    with pytest.raises(Blocked, match='REAL_MODEL_IN_SYNTHETIC'):
        Models(journal, FakeGrant(), clock=clock).call('extraction', {})


def test_tomorrow_drill_and_utc_timers_are_independent(tmp_path):
    result = drill(tmp_path / 'tomorrow', '2026-10-23', 'a' * 40)
    assert result['status'] == 'SYNTHETIC_DRILL_PASS' and result['closed'] == 2
    assert drill(tmp_path / 'weekend', '2026-10-24', 'a' * 40)['reason'] == 'NO_NYSE_SESSION'
    units = render('/repo', '/venv/python', '/private-runtime', '/private-config')
    assert len(units) == 8
    assert '12:45:00 UTC' in units['hyprl-agent-v2-quality-cycle.timer']
    assert '22:45:00 UTC' in units['hyprl-agent-v2-quality-drill.timer']
    assert all('Persistent=false' in body for name, body in units.items() if name.endswith('.timer'))
    assert all('trader-run' not in body and 'paper-execute' not in body for body in units.values())
