from datetime import timedelta
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.replay_bench import run as parent
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.replay_seal_forward import commitment, experiment as ex, forward as fw, models, spec
from scripts.trading_lab.replay_seal_forward import score as scoring
from scripts.trading_lab.sources.canonical import sha256_canonical

T = instant('2026-10-20T13:05:00Z')


def sealed():
    a = {'schema': 'replay-context-v1', 'universe': ['AAPL'], 'headlines': {'AAPL': {'items': []}},
         'prices': {}, 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD'}
    c = {'schema': 'replay-news-context-v1', 'candidates': [], 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD'}
    value = {'contexts': {'ab': a, 'cd': c}, 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD',
             'manifests': {'ab': {'borrow_flags': {}}, 'cd': {'borrow_flags': {}}}, 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash()}
    return dict(value, snapshot_hash=sha256_canonical(value))


def output():
    return {'analyst': {'regime': ['synthetic'], 'views': [dict(asset='AAPL', horizon=h, view='ABSTAIN', p_outperform=.7,
        confidence_reason='synthetic', catalysts=[], priced_in_assessment='', second_order='', counter_thesis='',
        falsifier='', event_risk=[]) for h in ('1d', '5d')]}, 'news': {'views': [], 'forecasts': []}}


@pytest.fixture
def runtime(tmp_path, monkeypatch, ledger):
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    monkeypatch.setattr(ex, 'committed', lambda: None)
    monkeypatch.setattr(ex, 'head', lambda: 'sha')
    file = tmp_path / 'freeze.json'
    file.write_text(json.dumps({'forward_config_hashes': {f: fw.forward_config_hash(f) for f in ex.FAMILIES},
                                'forward_first_session': '2026-10-20', 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash()}))
    monkeypatch.setattr(ex, 'FREEZE', file)
    monkeypatch.setattr(ex, 'frozen_payload', lambda root: json.loads(file.read_text()))
    monkeypatch.setattr(fw, 'snapshot', lambda root, at: sealed())
    ledger.clock = lambda: T
    calls = []
    def runner(args, **kwargs):
        calls.append((args, kwargs['input']))
        value = output()
        if args[0] == 'claude':
            raw = json.dumps({'subtype': 'success', 'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': value})
        else:
            raw = '\n'.join(json.dumps(e) for e in [{'type': 'thread.started'},
                {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(value)}},
                {'type': 'turn.completed'}])
        return subprocess.CompletedProcess(args, 0, raw, '')
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda family: 150)
    client = models.Models(grant, fw.ForwardBudget(tmp_path), tmp_path / 'transcripts', runner=runner,
                           clock=lambda: T, window=fw.window, scope='forward')
    return client, calls


def test_exchange_window_dst_holidays_and_no_backfill():
    assert fw.decision_time('2026-10-20') == T
    assert fw.decision_time('2026-11-02') == instant('2026-11-02T14:05:00Z')
    assert fw.window(T) == 1200
    for at in (T-timedelta(microseconds=1), T+timedelta(minutes=20)):
        with pytest.raises(ReplayError, match='OUTSIDE_FORWARD_WINDOW'):
            fw.window(at)
    with pytest.raises(ReplayError, match='NOT_FORWARD_SESSION'):
        fw.window(fw.decision_time('2026-11-26'))
    with pytest.raises(ReplayError, match='NOT_FORWARD_SESSION'):
        fw.window(fw.decision_time('2026-10-18'))


def test_one_call_each_identical_inputs_published_before_open_and_resume(tmp_path, runtime, ledger):
    client, calls = runtime
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert result['status'] == 'DONE' and len(calls) == 2
    assert calls[0][1] == calls[1][1]
    assert 'LIVE_FORWARD' in calls[0][1]
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert len(calls) == 2
    hashes = {'snapshot': result['snapshot_hash']}
    for family in ex.FAMILIES:
        value = json.loads((fw.directory(tmp_path, '2026-10-20') / (family+'.json')).read_text())
        assert value['provenance'] == 'LIVE_FORWARD' and value['decision_time'] == iso(T)
        assert value['snapshot_hash'] == result['snapshot_hash']
        assert value['output']['analyst']['views'][0]['p_outperform'] == .7
        hashes[family] = value['decision_hash']
    with pytest.raises(ReplayError, match='FORWARD_CALL_ALREADY_RESERVED'):
        client.budget.reserve_model('opus', 150, T)
    # Audit item 3: hash-only line pushed on the ledger branch and posted to #journal-des-ordres before 09:30.
    receipt = commitment.published(tmp_path, 'forward', '2026-10-20')
    assert receipt['line']['hashes'] == hashes and instant(receipt['published_at']) < fw.publication_deadline('2026-10-20')
    pushed = subprocess.run(['git', '--git-dir', str(ledger.remote), 'show', commitment.BRANCH + ':' + commitment.LEDGER_FILE],
                            capture_output=True, text=True, check=True).stdout
    assert [json.loads(l)['hashes'] for l in pushed.splitlines()] == [hashes]
    assert result['snapshot_hash'] in ledger.log.read_text() and 'AAPL' not in pushed + ledger.log.read_text()


def test_call_order_alternates_by_session():
    # Audit item 10.
    assert fw.order('2026-10-20') == ('opus', 'gpt')
    assert fw.order('2026-10-21') == ('gpt', 'opus')
    assert fw.order('2026-10-23') == ('gpt', 'opus')
    assert fw.order('2026-10-26') == ('opus', 'gpt')


def test_timeout_never_retries_and_other_family_completes(tmp_path, runtime, ledger):
    client, calls = runtime
    original = client.runner
    attempts = []
    def runner(args, **kwargs):
        attempts.append(args[0])
        if args[0] == 'claude':
            raise subprocess.TimeoutExpired(args, 1)
        return original(args, **kwargs)
    client.runner = runner
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert result['status'] == 'BLOCKED'
    assert result['models']['opus'] == 'CALL_OUTCOME_UNKNOWN'
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert sorted(attempts) == ['claude', 'codex']
    # The failure itself is committed publicly, never hidden.
    receipt = commitment.published(tmp_path, 'forward', '2026-10-20')
    assert receipt['line']['hashes']['opus'] == sha256_canonical(json.loads((fw.directory(tmp_path, '2026-10-20') / 'opus-failure.json').read_text()))


def test_tampered_saved_decision_refused(tmp_path, runtime, ledger):
    client, _ = runtime
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    file = fw.directory(tmp_path, '2026-10-20') / 'opus.json'
    value = json.loads(file.read_text()); value['output']['analyst']['views'][0]['p_outperform'] = .1
    atomic_json(file, value)
    with pytest.raises(ReplayError, match='FORWARD_DECISION_BINDING_MISMATCH'):
        fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)


class FakeOutcomes:
    seen = []
    def __init__(self, *args, **kwargs):
        self.decision_time = kwargs['decision_time']
    def get(self, group, day, asset, h):
        FakeOutcomes.seen.append((asset, h))
        return {'rel': .01, 'cost': .003}


def scored_setup(tmp_path, runtime, ledger, monkeypatch):
    client, _ = runtime
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    parent.write_json(fw.directory(tmp_path, '2026-10-20') / 'snapshot.json.gz', sealed())
    FakeOutcomes.seen = []
    monkeypatch.setattr(scoring, 'Outcomes', FakeOutcomes)


def test_mature_labels_only_failures_reported_and_weekly_at_most_once(tmp_path, runtime, ledger, monkeypatch):
    scored_setup(tmp_path, runtime, ledger, monkeypatch)
    values = fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    assert len(values['rows']) == 2 and FakeOutcomes.seen == [('AAPL', 1), ('AAPL', 1)]
    assert all(r['horizon'] == '1d' and r['brier'] == pytest.approx(.09) for r in values['rows'])
    assert values['evidence_class'] == 'LIVE_FORWARD' and values['primary_endpoint']['verdict'] == 'NO_WINNER'
    assert values['failures'] == {f: {'failed_sessions': [], 'count': 0} for f in ex.FAMILIES}
    friday = instant('2026-10-23T21:15:00Z')
    values['scored_at'] = iso(friday)
    atomic_json(tmp_path/fw.STATE_DIR/'scores.json', values)
    deliveries = []
    def deliver(args, **kwargs):
        deliveries.append((args, kwargs['input']))
        return subprocess.CompletedProcess(args, 0, 'discord\n', '')
    fw.weekly(tmp_path, notify=Path('/synthetic/notify'), clock=lambda: friday, runner=deliver)
    assert deliveries[0][0] == ['/synthetic/notify', 'brouillons']
    text = ''.join(d[1] for d in deliveries)
    assert all(len(d[1]) <= 1800 for d in deliveries)
    assert '"n": 1' in text and '"n": 0' in text and 'jours en échec=0' in text and 'No winner' in text
    sent = list(deliveries)
    with pytest.raises(ReplayError, match='ALREADY_ATTEMPTED'):
        fw.weekly(tmp_path, notify=Path('/synthetic/notify'), clock=lambda: friday, runner=deliver)
    assert deliveries == sent


def test_failed_day_counted_not_dropped(tmp_path, runtime, ledger, monkeypatch):
    client, _ = runtime
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, 'not json', '')
    client.runner = runner
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    parent.write_json(fw.directory(tmp_path, '2026-10-20') / 'snapshot.json.gz', sealed())
    monkeypatch.setattr(scoring, 'Outcomes', FakeOutcomes)
    values = fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    assert values['rows'] == []
    assert values['failures']['opus']['count'] == 1 and values['failures']['gpt']['count'] == 1
    assert values['failures']['opus']['failed_sessions'][0]['session'] == '2026-10-20'


def test_scorer_checks_frozen_forward_config_first_session_and_publication(tmp_path, runtime, ledger, monkeypatch):
    # Audit item 9: forward.score enforces the freeze itself, not only the CLI.
    scored_setup(tmp_path, runtime, ledger, monkeypatch)
    frozen = json.loads(ex.FREEZE.read_text())
    ex.FREEZE.write_text(json.dumps(dict(frozen, forward_config_hashes=dict(frozen['forward_config_hashes'], gpt='0'*64))))
    with pytest.raises(ReplayError, match='FORWARD_CONFIG_NOT_FROZEN'):
        fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    ex.FREEZE.write_text(json.dumps(dict(frozen, forward_first_session='2026-10-21')))
    with pytest.raises(ReplayError, match='FORWARD_CONFIG_NOT_FROZEN'):
        fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    ex.FREEZE.write_text(json.dumps(frozen))
    receipt = commitment.receipt_path(tmp_path, 'forward', '2026-10-20')
    value = json.loads(receipt.read_text())
    late = dict(value, published_at=iso(instant('2026-10-20T13:30:00Z')))
    atomic_json(receipt, late)
    with pytest.raises(ReplayError, match='PUBLISHED_TOO_LATE'):
        fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    receipt.unlink()
    with pytest.raises(ReplayError, match='DECISION_HASH_NOT_PUBLISHED'):
        fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))


def test_publication_retry_only_before_open(tmp_path, runtime, ledger):
    client, _ = runtime
    ledger.notify.write_text('#!/bin/sh\necho telegram-fallback\n')
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert result['status'] == 'BLOCKED' and result['publication'] == {'error': 'COMMITMENT_DISCORD_UNCONFIRMED'}
    parent.write_json(fw.directory(tmp_path, '2026-10-20') / 'snapshot.json.gz', sealed())
    ledger.notify.write_text('#!/bin/sh\ncat > /dev/null\necho discord\n')
    with pytest.raises(ReplayError, match='DEADLINE_PASSED'):
        fw.publish_pending(tmp_path, ledger=ledger, clock=lambda: instant('2026-10-20T13:30:00Z'))
    ledger.clock = lambda: T + timedelta(minutes=15)
    done = fw.publish_pending(tmp_path, ledger=ledger, clock=lambda: T + timedelta(minutes=15))
    assert done['status'] == 'PUBLISHED'
    pushed = subprocess.run(['git', '--git-dir', str(ledger.remote), 'show', commitment.BRANCH + ':' + commitment.LEDGER_FILE],
                            capture_output=True, text=True, check=True).stdout
    assert len(pushed.splitlines()) == 1


def test_first_session_gate_and_whole_missing_day_are_enforced_by_scorer(tmp_path, runtime, ledger, monkeypatch):
    values = fw.score(tmp_path, None, clock=lambda: T.replace(hour=21), write=False)
    assert all(v['count'] == 1 and v['failed_sessions'][0]['error'] == 'SNAPSHOT_MISSING' for v in values['failures'].values())
    assert all(r['n'] == 0 for p, fs in values['full_table'].items() if p != 'cd:trades' for r in fs.values())
    scored_setup(tmp_path, runtime, ledger, monkeypatch)
    fw.directory(tmp_path, '2026-10-20').rename(fw.directory(tmp_path, '2026-10-19'))
    with pytest.raises(ReplayError, match='FORWARD_BEFORE_FIRST_SESSION'):
        fw.score(tmp_path, None, clock=lambda: T.replace(hour=21), write=False)


def test_shared_data_quality_day_exclusion_publishes_hashes_without_model_calls(tmp_path, runtime, ledger, monkeypatch):
    client, calls = runtime
    value = sealed()
    value.pop('snapshot_hash')
    value['contexts']['ab']['universe'] = []
    for manifest in value['manifests'].values():
        manifest['data_quality'] = {'excluded_day': True, 'counts_by_reason': {'NEWS_ARCHIVE_COVERAGE_MISSING': 1}}
    value['snapshot_hash'] = sha256_canonical(value)
    monkeypatch.setattr(fw, 'snapshot', lambda root, at: value)
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T, ledger=ledger)
    assert result['status'] == 'DONE'
    assert result['models'] == {'opus': 'DATA_QUALITY_EXCLUDED', 'gpt': 'DATA_QUALITY_EXCLUDED'}
    assert calls == []
    receipt = commitment.published(tmp_path, 'forward', '2026-10-20')
    assert set(receipt['line']['hashes']) == {'snapshot', 'opus', 'gpt'}
    parent.write_json(fw.directory(tmp_path, '2026-10-20') / 'snapshot.json.gz', value)
    scored = fw.score(tmp_path, SimpleNamespace(), clock=lambda: T + timedelta(hours=8))
    assert all(row['count'] == 0 for row in scored['failures'].values())
    assert scored['trade_exclusions']['opus:input:NEWS_ARCHIVE_COVERAGE_MISSING'] == 1
    assert scored['trade_exclusions']['gpt:input:NEWS_ARCHIVE_COVERAGE_MISSING'] == 1


@pytest.mark.parametrize('late', [False, True])
def test_intraday_records_completion_clock_and_requires_publication_before_30_seconds(tmp_path, runtime, ledger, monkeypatch, late):
    from scripts.trading_lab.replay_seal_forward import intraday, collect_plan
    client, calls = runtime
    at = [instant('2026-10-20T13:40:00Z')]
    client.clock = lambda: at[0]
    client.window = models.continuous_remaining
    ledger.clock = lambda: at[0] + (timedelta(seconds=30) if late else timedelta())
    monkeypatch.setattr(collect_plan, 'live', lambda *a, **k: {})
    def snapshot(root, when, *, cutoff):
        value = sealed()
        value.pop('snapshot_hash')
        value.update(decision_time=iso(cutoff), input_cutoff=iso(cutoff), mode='INTRADAY')
        for context in value['contexts'].values():
            context['decision_time'] = iso(cutoff)
        value['snapshot_hash'] = sha256_canonical(value)
        parent.write_json(fw.directory(root, '2026-10-20')/'snapshot.json.gz', value)
        return value
    monkeypatch.setattr(fw, 'snapshot', snapshot)
    def sleep(seconds):
        at[0] += timedelta(seconds=seconds)
    result = intraday.run(tmp_path, client, expected_commit='sha', grants=None, alpaca_env=None,
                          clock=lambda: at[0], ledger=ledger, sleep=sleep)
    assert len(calls) == 2
    assert result['decision_time'] == iso(at[0])
    assert instant(result['entry_after']) == at[0] + timedelta(seconds=30)
    assert result['status'] == ('BLOCKED' if late else 'DONE')
    if late:
        assert result['publication']['error'] == 'DECISION_HASH_PUBLISHED_TOO_LATE'
    else:
        saved = parent.read_json(fw.directory(tmp_path, '2026-10-20')/'snapshot.json.gz')
        record = json.loads((fw.directory(tmp_path, '2026-10-20')/'opus.json').read_text())
        fw.check_record(tmp_path, '2026-10-20', 'opus', record, saved, json.loads(ex.FREEZE.read_text()))
    with pytest.raises(ReplayError, match='ALREADY_ATTEMPTED'):
        intraday.run(tmp_path, client, expected_commit='sha', grants=None, alpaca_env=None,
                     clock=lambda: at[0], ledger=ledger, sleep=sleep)
    assert len(calls) == 2


def test_intraday_first_trade_is_strict_including_provider_nanoseconds():
    from scripts.trading_lab.replay_seal_forward import intraday
    after = instant('2026-10-20T13:45:30.123456Z')
    trades = [{'t': '2026-10-20T13:45:30.123456000Z', 'p': 100},
              {'t': '2026-10-20T13:45:30.123456002Z', 'p': 102},
              {'t': '2026-10-20T13:45:30.123456001Z', 'p': 101}]
    transport = SimpleNamespace(get=lambda *a: ({'trades': {'AAPL': trades}}, {}))
    assert intraday.first_trades(transport, ['AAPL'], after)['AAPL']['p'] == 101


def test_intraday_labels_use_actual_entry_and_preserve_fixed_1d_exit():
    from scripts.trading_lab.replay_seal_forward import intraday
    from scripts.trading_lab.replay_bench import marketdata as md
    at = instant('2026-10-20T14:00:00Z')
    sessions = md.sessions('2026-09-01', '2026-10-26')
    bars = {symbol:[{'t':s.session_date+'T04:00:00Z','o':price,'h':price,'l':price,'c':price,'v':1e6}
                    for s in sessions] for symbol,price in [('AAPL',220),('SPY',102)]}
    def get(origin, path, params):
        if 'trades' in path:
            if '19:55:00' in params['start']:
                values = {'AAPL':[{'t':params['start'],'p':210}], 'SPY':[{'t':params['start'],'p':101}]}
            else:
                values = {s:[{'t':'2026-10-20T14:00:30.000000000Z','p':999},
                             {'t':'2026-10-20T14:00:30.000000001Z','p':p}]
                          for s,p in [('AAPL',200),('SPY',100)]}
            return {'trades':values}, {}
        return {'bars':{} if 'crypto' in path else bars}, {}
    outcomes = intraday.Outcomes(SimpleNamespace(get=get), '2026-10-20', '2026-10-26', spec.spec()['costs'],
                                decision_time=lambda day:at, live=True)
    one = outcomes.get('ab','2026-10-20','AAPL',1)
    five = outcomes.get('ab','2026-10-20','AAPL',5)
    assert one['rel'] == pytest.approx(.04)
    assert five['rel'] == pytest.approx(.08)
    assert one['entry_at'] == '2026-10-20T14:00:30.000000001Z'
    assert outcomes.get('ab','2026-10-20','SOL-USD',1) is None
    assert outcomes.exclusion_reasons[('ab','2026-10-20','SOL-USD',1)] == 'INTRADAY_CRYPTO_TRADE_NOT_GRANTED'


def test_regular_workers_overlap_and_keep_full_timeout(tmp_path, runtime, ledger):
    from threading import Barrier
    client, _ = runtime
    original=client.runner
    barrier=Barrier(2,timeout=5)
    seen=[]
    def runner(args,**kwargs):
        seen.append(kwargs['timeout'])
        barrier.wait()
        return original(args,**kwargs)
    client.runner=runner
    result=fw.run(tmp_path,client,expected_commit='sha',clock=lambda:T,ledger=ledger)
    assert result['status']=='DONE' and seen==[900,900]
    assert fw.crypto_entry(T)==fw.publication_deadline('2026-10-20')


def test_late_regular_dispatch_reserves_no_shortened_call(tmp_path,runtime,ledger):
    client,calls=runtime
    client.clock=lambda:T+timedelta(minutes=6)
    result=fw.run(tmp_path,client,expected_commit='sha',clock=client.clock,ledger=ledger)
    assert result['status']=='BLOCKED' and not calls
    assert set(result['models'].values())=={'INSUFFICIENT_CALL_WINDOW'}
    assert client.budget.used('forward:model:opus',T)==0
    assert client.budget.used('forward:model:gpt',T)==0


def test_legacy_forward_registry_preserves_original_binding():
    previous=json.loads((spec.ROOT/'docs/artifacts/replay_duel_test_freeze_v9.json').read_text())
    sealed={'spec_hash':previous['spec_hash'],'code_hash':previous['code_hash']}
    assert fw.configuration_for(sealed,{})==previous
    with pytest.raises(ReplayError,match='FORWARD_CONFIG_NOT_FROZEN'):
        fw.configuration_for(dict(sealed,code_hash='0'*64),{})
    with pytest.raises(ReplayError,match='FORWARD_CONFIG_NOT_FROZEN'):
        fw.configuration_for(dict(sealed,spec_hash='0'*64),{})
