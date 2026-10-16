from copy import deepcopy
from datetime import timedelta
import math
import hashlib
import sqlite3

import pytest

from scripts.trading_lab.agent_v2.contracts import Blocked, NEWS, digest, iso, strict_json, validate
from scripts.trading_lab.agent_v2.core import candidates, momentum_score
from scripts.trading_lab.agent_v2.deploy import deploy_window, render
from scripts.trading_lab.agent_v2.inputs import classify, validate_snapshot
from scripts.trading_lab.agent_v2.ledger import Ledger
from scripts.trading_lab.agent_v2.models import Claude, SyntheticClaude
from scripts.trading_lab.agent_v2.reporting import reconcile, summary
from scripts.trading_lab.agent_v2.risk import governor, size
from scripts.trading_lab.agent_v2.service import Service
from scripts.trading_lab.agent_v2.spec import configuration, spec
from .conftest import quotes


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(Blocked):
        strict_json(raw)


def test_hashes_and_schema():
    assert configuration()['spec_hash']
    validate({'events':[]}, NEWS)
    with pytest.raises(Blocked):
        validate({'events':[], 'decision':'BUY'}, NEWS)


def test_momentum_analytic_parity():
    # Exponential prices give R2=1; formula matches the original book's OLS90 x R2.
    assert momentum_score([100 * math.exp(.001*i) for i in range(90)]) == pytest.approx(math.exp(.25)-1)
    assert momentum_score([100]*90) == 0


def test_real_shape_and_point_in_time(snapshot, clock):
    instruments = validate_snapshot(snapshot, clock(), 'SYNTHETIC')
    assert len(instruments) == 4
    snapshot['instruments'][0]['bars'][-1]['available_at'] = iso(clock() + timedelta(seconds=1))
    with pytest.raises(Blocked, match='NONCAUSAL'):
        validate_snapshot(snapshot, clock(), 'SYNTHETIC')


@pytest.mark.parametrize('field', ['metadata_at','calendar_at','at'])
def test_missing_or_stale_inputs(snapshot, clock, field):
    snapshot[field] = iso(clock() - timedelta(days=10))
    with pytest.raises(Blocked, match='STALE'):
        validate_snapshot(snapshot, clock(), 'SYNTHETIC')


def test_source_anchor_and_primary(snapshot, clock):
    text = 'Synthetic issuer reports an earnings miss and cuts guidance.'
    source = {'id':'source-1','symbol':'NEWS','kind':'8-K','url':'https://www.sec.gov/Archives/edgar/data/123/456/a.htm',
              'published_at':iso(clock()-timedelta(hours=1)), 'received_at':iso(clock()), 'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    snapshot['sources'] = [source]
    output = {'events':[{'source_id':'source-1','symbol':'NEWS','kind':'8-K','sentiment':'negative',
                         'code':'guidance_cut','evidence':text}]}
    result = classify(output, snapshot, clock())
    assert result[0]['primary'] and result[0]['liquid_small_mid']
    source['url'] = 'https://www.sec.gov.evil.test/Archives/edgar/data/123/a.htm'
    assert not classify(output, snapshot, clock())[0]['primary']
    output['events'][0]['evidence'] = 'invented'
    with pytest.raises(Blocked, match='NEWS_EVIDENCE_INVALID'):
        classify(output, snapshot, clock())


def test_core_uses_only_two_layer_hunter(snapshot, clock):
    instruments = validate_snapshot(snapshot, clock(), 'SYNTHETIC')
    result, _ = candidates(snapshot, instruments, [], clock(), spec())
    assert {r['playbook'] for r in result} == {'hunter_runner','equity_momentum','etf_trend'}
    snapshot['hunter']['candidates'][0]['two_layer'] = False
    result, _ = candidates(snapshot, instruments, [], clock(), spec())
    assert 'hunter_runner' not in {r['playbook'] for r in result}


def test_decision_then_later_quote_fill_and_idempotence(service, ledger, snapshot, clock):
    service.decide(snapshot)
    assert not ledger.state()['positions']
    assert len(ledger.state()['pending']) == 3
    before = quotes(snapshot, clock)
    for q in before['quotes']:
        q['at'] = iso(clock()-timedelta(seconds=1))
    service.watch(before)
    assert not ledger.state()['positions']
    clock.at += timedelta(seconds=10)
    service.watch(quotes(snapshot, clock))
    state = ledger.state()
    assert len(state['positions']) == 3 and not state['pending']
    fill = state['positions']['ACME']
    q = next(q for q in snapshot['quotes'] if q['symbol'] == 'ACME')
    assert fill['entry'] == pytest.approx(q['ask'] * 1.0002)
    assert fill['entry_fee'] == pytest.approx(fill['entry'] * fill['qty'] * .001)
    service.watch(quotes(snapshot, clock))
    assert len(ledger.records('entry_fill')) == 3
    with pytest.raises(Blocked, match='SLOT_ALREADY_ATTEMPTED'):
        service.decide(snapshot)


def test_gap_stop_losses_and_cash_reconcile(service, ledger, snapshot, clock):
    service.decide(snapshot)
    clock.at += timedelta(seconds=10)
    service.watch(quotes(snapshot, clock))
    clock.at += timedelta(minutes=1)
    service.watch(quotes(snapshot, clock, bid=40, ask=40.05))
    state = ledger.state()
    assert not state['positions'] and len(state['closed']) == 3
    assert all(p['pnl'] < 0 and p['reason'] == 'STOP' for p in state['closed'])
    assert state['cash'] == pytest.approx(100000 + sum(p['pnl'] for p in state['closed']))
    report = reconcile(ledger, clock=clock)
    assert report['status'] == 'PASS'
    assert sum(p['closed'] for p in report['layers'].values()) == 3


def test_stale_reconciliation_is_blocked(ledger, clock):
    assert reconcile(ledger, clock=clock)['status'] == 'BLOCKED'


def test_supervisor_coded_veto(service, ledger, snapshot):
    class Veto(SyntheticClaude):
        def call(self, role, value):
            out = super().call(role, value)
            if role == 'supervisor':
                out['reviews'][0].update(verdict='VETO', reason='NO_EVIDENCE')
            return out
    service.runner = Veto()
    service.decide(snapshot)
    view = summary(ledger, clock=service.clock)
    assert view['supervisor_vetoes']['count'] == 1
    assert len(ledger.state()['pending']) == 2
    assert view['supervisor_vetoes']['counterfactual'] == 'UNMEASURED'


@pytest.mark.parametrize('bad', ['missing','invent','resize','wrong_reason','failure'])
def test_supervisor_cannot_modify_or_invent(service, ledger, snapshot, bad):
    class Invalid(SyntheticClaude):
        def call(self, role, value):
            out = super().call(role, value)
            if role == 'supervisor':
                if bad == 'failure':
                    raise Blocked('MODEL_USAGE_LIMIT')
                if bad == 'missing': out['reviews'].pop()
                if bad == 'invent': out['reviews'][0]['plan_id'] = 'f'*64
                if bad == 'resize': out['reviews'][0]['qty'] = 99999
                if bad == 'wrong_reason': out['reviews'][0]['reason'] = 'CAP'
            return out
    service.runner = Invalid()
    with pytest.raises(Blocked):
        service.decide(snapshot)
    assert not ledger.state()['pending']
    assert len(ledger.state()['vetoes']) >= 3


def test_heat_clusters_pending_and_no_shorts(snapshot, ledger, clock):
    state = ledger.state()
    q = {q['symbol']: q for q in snapshot['quotes']}
    candidate = {'symbol':'NEWS','cluster':'health','side':'BUY','playbook':'primary_news',
                 'layer':'news','stop':80,'atr':5,'holding_days':5,'risk_fraction':.01}
    settings = spec()
    plan, reason = size(candidate, state, q, snapshot, [], settings, clock())
    assert not reason and plan['initial_risk'] <= 1000
    state['pending']['x'] = {**plan, 'symbol':'OTHER','initial_risk':1499,'qty':1}
    assert size(candidate, state, q, snapshot, [], settings, clock())[1] == 'RISK_OR_CAP'
    state['pending'].clear()
    state['pending']['x'] = {**plan,'cluster':'energy','symbol':'OTHER','initial_risk':4000,'qty':1}
    assert size(candidate, state, q, snapshot, [], settings, clock())[1] == 'RISK_OR_CAP'
    candidate['side'] = 'SELL'
    assert size(candidate, state, q, snapshot, [], settings, clock())[1] == 'SHORT_FORBIDDEN'


def test_event_and_negative_news_hard_veto(snapshot, ledger, clock):
    state, settings = ledger.state(), spec()
    q = {q['symbol']:q for q in snapshot['quotes']}
    candidate = {'symbol':'NEWS','cluster':'health','side':'BUY','playbook':'primary_news',
                 'layer':'news','stop':80,'atr':5,'holding_days':5,'risk_fraction':.0025}
    news = [{'symbol':'NEWS','primary':True,'timely':True,'sentiment':'negative'}]
    assert size(candidate, state, q, snapshot, news, settings, clock())[1] == 'NEGATIVE_PRIMARY_NEWS'
    snapshot['events'] = [{'kind':'FOMC','symbol':None,'at':iso(clock()+timedelta(hours=5))}]
    assert size(candidate, state, q, snapshot, [], settings, clock())[1] == 'EVENT_RISK'


def test_governor(clock):
    assert governor({'marks':[]}, 96999, clock())[0] == .5
    assert governor({'marks':[]}, 93999, clock())[0] == .25
    assert governor({'marks':[]}, 91999, clock())[0] == 0


def test_append_only_readonly_and_no_rebind(ledger, clock):
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute('DELETE FROM events')
    ledger.db.rollback()
    read = Ledger(ledger.root, read_only=True)
    try:
        with pytest.raises(Blocked, match='READ_ONLY'):
            read.append([('test',{})], clock())
    finally:
        read.close()
    with pytest.raises(Blocked, match='REBIND_FORBIDDEN'):
        ledger.append([('binding',{})], clock())
    with pytest.raises(FileExistsError):
        Ledger.initialize(ledger.root, 'SYNTHETIC', 'b'*40, clock())


def test_tamper_detected(ledger):
    ledger.db.execute('DROP TRIGGER no_update')
    ledger.db.execute("UPDATE events SET body='{}'")
    ledger.db.commit()
    with pytest.raises(Blocked, match='LEDGER_INTEGRITY'):
        ledger.verify()


def test_no_forward_fake_runner(tmp_path, clock):
    ledger = Ledger.initialize(tmp_path/'forward', 'SHADOW_FORWARD', 'a'*40, clock())
    try:
        with pytest.raises(Blocked, match='SYNTHETIC_RUNNER_FORBIDDEN'):
            Service(ledger, SyntheticClaude(), clock=clock)
    finally:
        ledger.close()


def test_no_implicit_authorization(ledger, tmp_path, clock):
    path = tmp_path/'wrong-grant.json'
    path.write_text('{"authorization":"replay-bench-v1"}')
    with pytest.raises(Blocked, match='V2_AUTHORIZATION_REQUIRED'):
        Claude(ledger,path,clock=clock).grant()
    assert not ledger.records('model_reservation')


@pytest.mark.parametrize('at', ['2026-10-16T11:30:00Z','2026-10-16T13:49:59Z','2026-10-16T15:00:00Z',
                               '2026-10-16T16:29:59Z','2026-10-16T19:45:00Z','2026-10-16T20:09:59Z'])
def test_deploy_blackouts(at):
    from scripts.trading_lab.agent_v2.contracts import instant
    with pytest.raises(Blocked, match='PROTECTED_DEPLOY_WINDOW'):
        deploy_window(instant(at))


def test_units_are_isolated_and_dst_aware():
    texts=render('/repo','/venv/python','/runtime-v2','/grant.json')
    assert len(texts)==6
    assert all('hyprl-agent-v2-shadow-' in name for name in texts)
    assert all('America/New_York' in body and 'Persistent=false' in body
               for name,body in texts.items() if name.endswith('.timer'))


def test_forward_requires_activation_even_with_runner(tmp_path, clock):
    ledger = Ledger.initialize(tmp_path/'inert', 'SHADOW_FORWARD', 'a'*40, clock())
    try:
        with pytest.raises(Blocked, match='RUNTIME_NOT_ACTIVATED'):
            Service(ledger, object(), clock=clock)
    finally:
        ledger.close()


def test_expiry_rechecks_risk_and_never_fills_old_plans(service, ledger, snapshot, clock):
    service.decide(snapshot)
    clock.at += timedelta(minutes=6)
    service.watch(quotes(snapshot, clock))
    assert len(ledger.records('expired')) == 3
    assert not ledger.state()['positions'] and not ledger.state()['pending']


def test_news_five_session_exit(service, ledger, snapshot, clock):
    from scripts.trading_lab.agent_v2.contracts import instant
    text = 'Synthetic issuer beats earnings estimates and raises guidance.'
    snapshot['sources'] = [{'id':'earnings-1','symbol':'NEWS','kind':'earnings',
        'url':'https://ir.example.test/release', 'published_at':iso(clock()-timedelta(hours=1)),
        'received_at':iso(clock()), 'text':text, 'sha256':hashlib.sha256(text.encode()).hexdigest()}]
    class News(SyntheticClaude):
        def call(self, role, value):
            if role == 'news':
                return {'events':[{'source_id':'earnings-1','symbol':'NEWS','kind':'earnings',
                    'sentiment':'positive','code':'beat_and_raise','evidence':text}]}
            return super().call(role, value)
    service.runner = News()
    service.decide(snapshot)
    clock.at += timedelta(seconds=10)
    service.watch(quotes(snapshot, clock))
    assert ledger.state()['positions']['NEWS']['holding_days'] == 5
    clock.at = instant('2026-10-19T13:40:00Z')
    service.watch(quotes(snapshot, clock))
    assert 'NEWS' in ledger.state()['positions']  # held overnight and across weekend
    clock.at = instant('2026-10-23T13:40:00Z')
    service.watch(quotes(snapshot, clock))
    news = [p for p in ledger.state()['closed'] if p['symbol'] == 'NEWS']
    assert len(news) == 1 and news[0]['reason'] == 'TIME_STOP' and news[0]['layer'] == 'news'


def test_early_close_calendar_and_dst(clock):
    from scripts.trading_lab.agent_v2.inputs import session
    from scripts.trading_lab.agent_v2.contracts import instant
    assert iso(session(instant('2026-11-27T17:40:00Z'))[2]) == '2026-11-27T18:00:00Z'
    assert iso(session(instant('2026-11-02T14:40:00Z'))[1]) == '2026-11-02T14:30:00Z'
