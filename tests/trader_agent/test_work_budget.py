"""Offline admission and duplex permission checks; never launch a real model."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from scripts.trading_lab.trader_agent.config import Authorization, TraderError, instant
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.runners import ModelRunner, command, prompt
from scripts.trading_lab.trader_agent.schemas import validate_analyst, validate_reviewer
from scripts.trading_lab.trader_agent.synthetic import SyntheticRunner
from scripts.trading_lab.trader_agent import work_budget as work
from test_pm_slot import pm_grant
from test_news_ab import news_service, radar


def approved(grant):
    return replace(grant, work_budget_approval={
        'schema': 'trader-agent-work-budget-amendment-v2', 'status': 'APPROVED',
        'spec_hash': work.SPEC_HASH, 'parent_preregistration_hash': work.spec()['base_preregistration_hash'],
        'base_grant_hash': grant.base_identity or grant.identity,
        'approved_at': '2026-10-20T00:00:00Z', 'effective_from': '2026-10-21T00:00:00Z',
        'not_after': '2026-11-01T00:00:00Z'})


@pytest.fixture
def bounded(ledger, clock):
    clock.at = instant('2026-10-21T12:00:00Z')
    ledger.grant = approved(pm_grant(ledger.grant))
    return ledger


def fake_cli(monkeypatch, tmp_path, events):
    """The child performs a synthetic web operation only after reading host ALLOW."""
    script = tmp_path / 'synthetic_cli.py'
    executed = tmp_path / 'executed.jsonl'
    script.write_text('''import json,sys,os
def send(value):
 print(json.dumps(value),flush=True)
init=json.loads(sys.stdin.readline())
send({'type':'system','subtype':'ui_invalidate','event':'ui.render'})
send({'type':'control_response','response':{'subtype':'success','request_id':init['request_id'],'response':{}}})
user=json.loads(sys.stdin.readline())
assert user['type']=='user'
for event in json.loads(sys.argv[1]):
 send(event)
 if event.get('type')=='control_request':
  line=sys.stdin.readline()
  if not line:sys.exit(0)
  response=json.loads(line)['response']['response']
  if response['behavior']!='allow':sys.exit(0)
  with open(sys.argv[2],'a') as stream:stream.write(json.dumps(event['request'])+'\\n')
''')
    monkeypatch.setattr('scripts.trading_lab.trader_agent.runners.command',
        lambda *a, **kw: [sys.executable, str(script), json.dumps(events), str(executed)])
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic-cli-v1'))
    return executed


def result(output=None, *, new=50000, cached=200000, turns=8, helper=10000):
    output_tokens = min(1000, new)
    creation = min(10000, new - output_tokens)
    return {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': turns,
        'structured_output': output if output is not None else {'regime': [], 'views': []},
        # Aggregate duplicates the main entry and must not be added again.
        'usage': {'input_tokens': new - 1000, 'output_tokens': 1000},
        'modelUsage': {
            'synthetic-main': {'inputTokens': new - creation - output_tokens, 'outputTokens': output_tokens,
                'cacheCreationInputTokens': creation, 'cacheReadInputTokens': cached},
            'synthetic-web-helper': {'inputTokens': helper, 'outputTokens': 0,
                'cacheCreationInputTokens': 0, 'cacheReadInputTokens': 0}}}


def request(index, name='WebSearch', **inputs):
    return {'type': 'control_request', 'request_id': 'permission-' + str(index), 'request': {
        'subtype': 'can_use_tool', 'tool_name': name, 'tool_use_id': 'tool-' + str(index),
        'input': inputs or {'query': 'Synthetic publication'}}}


def test_admission_preserves_both_attempts_and_reviewer_across_runtime_restart(bounded, tmp_path, monkeypatch):
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    first = bounded.reserve('analyst_claude')
    bounded.record_work(first, 5000, 900000, 'SCHEMA_INVALID')
    other = Ledger(tmp_path / 'other-runtime', bounded.grant, clock=bounded.clock, budget_root=bounded.budget_root)
    other.reserve('analyst_claude')
    with pytest.raises(TraderError, match='WORK_BUDGET_EXHAUSTED'):
        other.reserve('analyst_claude')
    for role in ('reviewer', 'reviewer'):
        other.reserve(role)
    other.arm = 'news'
    for role in ('analyst_claude', 'analyst_claude', 'reviewer', 'reviewer'):
        other.reserve(role)
    assert [(r['arm'], r['reserved']) for r in other.work_counts()] == [('base', 4000000), ('news', 4000000)]
    assert other.work_counts()[0]['observed_new'] == 5000
    with other.connect() as db:
        for table in ('work_reservations', 'work_receipts'):
            with pytest.raises(sqlite3.IntegrityError, match='immutable'):
                db.execute('DELETE FROM ' + table)



def test_concurrent_admission_is_atomic(bounded):
    def reserve(_):
        try:
            return bounded.reserve('analyst_claude')
        except TraderError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(reserve, range(3)))
    assert sum(type(v) is int for v in outcomes) == 2
    assert outcomes.count('WORK_BUDGET_EXHAUSTED') == 1
    assert bounded.counts() == {'analyst_claude': 2}



@pytest.mark.parametrize('change', [{'status': 'DRAFT'}, {'base_grant_hash': '0' * 64},
    {'spec_hash': '0' * 64}, {'not_after': '2026-10-21T12:00:00Z'},
    {'effective_from': '2026-10-22T00:00:00Z'}, {'approved_at': '2026-10-22T00:00:00Z'},
    {'effective_from': None}])
def test_invalid_approval_cannot_launch_even_cli_version(bounded, monkeypatch, change):
    bounded.grant = replace(bounded.grant, work_budget_approval={**bounded.grant.work_budget_approval, **change})
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Unapproved CLI launch'))
    with pytest.raises(TraderError, match='WORK_BUDGET_APPROVAL_INVALID'):
        ModelRunner(bounded, clock=bounded.clock).once('analyst_claude', 'Synthetic')
    assert bounded.counts() == {}


def test_cutover_requires_operator_record_and_does_not_edit_grant(ledger, clock, monkeypatch):
    clock.at = instant(work.REQUIRES_APPROVAL_FROM)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Missing approval launched CLI'))
    with pytest.raises(TraderError, match='WORK_BUDGET_APPROVAL_REQUIRED'):
        ModelRunner(ledger, clock=clock).once('analyst_claude', 'Synthetic')
    assert ledger.counts() == {}


def test_command_retains_isolation_and_forces_host_permissions(bounded):
    args = command('analyst_claude', bounded.grant, limits=work.spec())
    assert '--max-turns' not in args
    assert args[args.index('--permission-prompt-tool') + 1] == 'stdio'
    settings = json.loads(args[args.index('--settings') + 1])
    assert settings == {'disableAllHooks': True, 'permissions': {'ask': ['WebSearch', 'WebFetch']}}
    assert '--safe-mode' in args and '--strict-mcp-config' in args
    assert args[args.index('--mcp-config') + 1] == '{"mcpServers":{}}'
    assert args[args.index('--tools') + 1] == 'WebSearch,WebFetch'


def test_duplex_success_counts_helpers_once_and_preserves_full_output(bounded, monkeypatch, tmp_path):
    executed = fake_cli(monkeypatch, tmp_path, [request(1), request(2, 'WebFetch', url='https://example.invalid/news'), result()])
    runner = ModelRunner(bounded, clock=bounded.clock)
    assert runner.infer('analyst_claude', 'Synthetic') == {'regime': [], 'views': []}
    assert len(executed.read_text().splitlines()) == 2
    assert runner.web_searches == {'analyst_claude': 1}
    assert bounded.work_counts() == [{'slot': 'am', 'arm': 'base', 'reserved': 1000000,
                                     'observed_new': 60000, 'observed_cache_read': 200000, 'incomplete_usage': 0}]






@pytest.mark.parametrize('url', ['https://www.sec.gov/news', 'https://data.sec.gov/submissions/',
    'https://federalreserve.gov./news', 'https://www.federalreserve.gov/news',
    'http://example.invalid/news', 'https://synthetic:secret@example.invalid/news'])
def test_forbidden_fetch_cannot_receive_allow(bounded, monkeypatch, tmp_path, url):
    executed = fake_cli(monkeypatch, tmp_path, [request(1, 'WebFetch', url=url), result()])
    with pytest.raises(TraderError, match='TAINTED_RUN'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic')
    assert not executed.exists()
    assert bounded.counts() == {'analyst_claude': 1}






def test_reported_overrun_is_evidence_and_cannot_poison_reviewer(bounded, monkeypatch, tmp_path):
    fake_cli(monkeypatch, tmp_path, [result(new=1000001, helper=2000)])
    assert ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic')['views'] == []
    bounded.reserve('reviewer')
    assert bounded.work_counts()[0]['observed_new'] == 1002001



def test_interrupted_call_retains_envelope_and_allows_recovery_and_review(bounded, tmp_path):
    bounded.reserve('analyst_claude')
    restarted = Ledger(tmp_path / 'restarted', bounded.grant, clock=bounded.clock,
                       budget_root=bounded.budget_root)
    restarted.reserve('analyst_claude')
    restarted.reserve('reviewer')
    assert restarted.work_counts()[0]['reserved'] == 3000000
    assert restarted.work_counts()[0]['incomplete_usage'] == 3



def test_activation_cannot_reset_legacy_work_on_the_same_day(bounded):
    original = bounded.grant
    bounded.grant = replace(original, work_budget_approval=None)
    bounded.reserve('analyst_claude')
    bounded.grant = original
    with pytest.raises(TraderError, match='WORK_BUDGET_UNACCOUNTED'):
        bounded.reserve('reviewer')
    assert bounded.work_counts() == []


def test_quota_failure_never_retries_or_counts_unknown_usage_as_zero(bounded, monkeypatch, tmp_path):
    failure = {'type': 'result', 'is_error': True, 'num_turns': 0,
        'subtype': 'error_during_execution', 'errors': ['HTTP 429 usage limit']}
    fake_cli(monkeypatch, tmp_path, [failure])
    with pytest.raises(TraderError, match='SKIPPED_QUOTA'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic')
    assert bounded.counts() == {'analyst_claude': 1}
    assert bounded.work_counts()[0]['incomplete_usage'] == 1
    bounded.reserve('reviewer')


def test_oversized_prompt_is_refused_before_any_cli_or_reservation(bounded, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Oversized prompt launched CLI'))
    with pytest.raises(TraderError, match='WORK_BUDGET_PROMPT_TOO_LARGE'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'é' * 260001)
    assert bounded.counts() == {}


def test_stalled_cli_input_obeys_the_model_deadline(bounded, monkeypatch, tmp_path):
    script = tmp_path / 'stalled.py'
    script.write_text("import json,sys,time\n"
        "init=json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'control_response','response':{'subtype':'success',"
        "'request_id':init['request_id'],'response':{}}}),flush=True)\n"
        "time.sleep(5)\n")
    monkeypatch.setattr('scripts.trading_lab.trader_agent.runners.command',
                        lambda *a, **kw: [sys.executable, str(script)])
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic'))
    runner = ModelRunner(bounded, clock=bounded.clock)
    runner.deadline = bounded.clock() + timedelta(seconds=.2)
    with pytest.raises(TraderError, match='MODEL_TIMEOUT'):
        runner.once('analyst_claude', 'x' * 250000)
    assert bounded.work_counts()[0]['incomplete_usage'] == 1


def test_expiring_work_approval_cannot_authorize_a_later_web_call(bounded, monkeypatch, tmp_path):
    marker = tmp_path / 'clock-advanced'
    script = tmp_path / 'approval_expiry.py'
    script.write_text("import json,sys\nfrom pathlib import Path\n"
        "init=json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'control_response','response':{'subtype':'success',"
        "'request_id':init['request_id'],'response':{}}}),flush=True)\n"
        "sys.stdin.readline()\nPath(sys.argv[1]).touch()\n"
        "print(sys.argv[2],flush=True)\nsys.stdin.readline()\n")
    monkeypatch.setattr('scripts.trading_lab.trader_agent.runners.command',
        lambda *a, **kw: [sys.executable, str(script), str(marker), json.dumps(request(1))])
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic'))
    original_clock = bounded.clock
    bounded.clock = lambda: instant('2026-11-01T00:00:00Z') if marker.exists() else original_clock()
    with pytest.raises(TraderError, match='WORK_BUDGET_APPROVAL_INVALID'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic')
    assert bounded.work_counts('2026-10-21')[0]['incomplete_usage'] == 1


def test_reviewer_coverage_is_still_required_under_the_work_cap(bounded, monkeypatch, tmp_path):
    fake_cli(monkeypatch, tmp_path, [result({'verdicts': []})])
    with pytest.raises(TraderError, match='REVIEW_COVERAGE_INVALID'):
        ModelRunner(bounded, clock=bounded.clock).infer('reviewer', 'Synthetic',
            validator=lambda output: validate_reviewer(output, {'analyst_claude': {
                'views': [{'asset': 'AAPL', 'horizon': '1d'}]}}))
    alerts = [json.loads(line)['code'] for line in (bounded.root / 'alerts.jsonl').read_text().splitlines()]
    assert alerts == ['REVIEW_COVERAGE_INVALID', 'REVIEW_COVERAGE_INVALID']


@pytest.mark.parametrize('slot', ['am', 'pm'])
def test_synthetic_service_keeps_method_binding_and_all_decision_contracts(news_service, clock, slot):
    service = news_service
    clock.at = instant('2026-10-21T' + ('12' if slot == 'am' else '16') + ':00:00Z')
    service.ledger.slot = slot
    service.ledger.grant = approved(pm_grant(service.ledger.grant))
    base = service.run()
    assert base['status'] == 'COMPLETE', base
    news = service.run_news()
    assert news['status'] == 'COMPLETE', news
    for result in (base, news):
        binding = service.store.records('replay-summary', object_id=result['run_id'] + ':work-budget')
        assert len(binding) == 1
        assert binding[0]['payload']['work_budget_spec_hash'] == work.SPEC_HASH
        assert binding[0]['payload']['context_hash'] == result['decision']['context_hash']
        assert result['decision']['preregistration_hash'] == work.spec()['base_preregistration_hash']
    assert service.ledger.work_counts() == [
        {'slot': slot, 'arm': arm, 'reserved': 2000000, 'observed_new': 0,
         'observed_cache_read': 0, 'incomplete_usage': 0} for arm in ('base', 'news')]
    assert service.store.verify()['verified']


def test_probe_uses_its_own_nonrefundable_pool_and_no_web(bounded, monkeypatch, tmp_path):
    fake_cli(monkeypatch, tmp_path, [result(new=3000, helper=0, turns=1)])
    runner = ModelRunner(bounded, clock=bounded.clock)
    for _ in range(2):
        for role in ('analyst_claude', 'reviewer'):
            runner.once(role, 'Synthetic probe', probe=True)
    assert bounded.work_counts()[0]['arm'] == 'monitor'
    assert bounded.work_counts()[0]['reserved'] == 100000
    bounded.record_work(bounded.reserve('analyst_claude'), 0, 0, 'COMPLETE')
    bounded.reserve('reviewer')


@pytest.mark.parametrize('failure,code', [('future', 'FUTURE_SOURCE'), ('citation', 'UNSUPPORTED_VIEW'),
                                        ('schema', 'SCHEMA_INVALID'), ('url', None)])
def test_bounded_inference_keeps_existing_analyst_validations(bounded, clock, monkeypatch, tmp_path, failure, code):
    # Generate native-shaped synthetic output in a separate bank, then send it through the real runner.
    synthetic = Ledger(tmp_path / 'shape', replace(bounded.grant, work_budget_approval=None), clock=clock)
    context = {'universe': ['AAPL'], 'decision_time': '2026-10-21T12:00:00Z'}
    output = SyntheticRunner(synthetic, clock=clock).once('analyst_claude', prompt(context, 'Synthetic skill'))
    if failure == 'future':
        output['views'][0]['catalysts'][0]['published_at'] = '2026-10-21T12:00:01Z'
    elif failure == 'citation':
        output['views'][0]['catalysts'] = []
    elif failure == 'schema':
        output['unexpected'] = True
    else:
        output['views'][0]['catalysts'][0]['url'] = 'http://example.invalid/synthetic'
    fake_cli(monkeypatch, tmp_path, [result(output)])
    runner = ModelRunner(bounded, clock=clock)
    validator = lambda value: validate_analyst(value, ['AAPL'], clock())
    if code:
        # Both original grant-authorized attempts reject the same invalid result.
        with pytest.raises(TraderError, match=code):
            runner.infer('analyst_claude', 'Synthetic', validator=validator)
        alerts = [json.loads(line)['code'] for line in (bounded.root / 'alerts.jsonl').read_text().splitlines()]
        assert alerts == [code, code]
    else:
        accepted = runner.infer('analyst_claude', 'Synthetic', validator=validator)
        assert accepted['views'][0]['view'] == 'ABSTAIN'  # existing URL-drop behavior, never repaired here
    assert bounded.counts() == {'analyst_claude': 2 if code else 1}
