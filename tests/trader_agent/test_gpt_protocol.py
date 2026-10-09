"""Synthetic copies of the Codex web event envelope observed on October 9."""
import json
from types import SimpleNamespace

import pytest

from scripts.trading_lab.trader_agent.config import TraderError
from scripts.trading_lab.trader_agent.runners import ModelRunner, parse_gpt
from scripts.trading_lab.trader_agent.schemas import validate_analyst


WEB_STARTED = ('{"type":"item.started","item":{"id":"item_2","type":"web_search",'
               '"id":"exec-synthetic","query":"","action":{"type":"other"}}}')
WEB_COMPLETED = ('{"type":"item.completed","item":{"id":"item_2","type":"web_search",'
                 '"id":"exec-synthetic","query":"synthetic query",'
                 '"action":{"type":"search","queries":["synthetic query"]},'
                 '"results":[{"type":"text_result","domain":"example.invalid",'
                 '"ref_id":"synthetic0","snippet":"Synthetic evidence",'
                 '"title":"Synthetic result","url":"https://example.invalid/"}]}}')


def stream(output, *events):
    return '\n'.join([json.dumps({'type': 'thread.started', 'thread_id': 'synthetic'}),
        json.dumps({'type': 'item.completed', 'item': {'id': 'item_0', 'type': 'error',
                    'message': 'Under-development features enabled: synthetic warning'}}),
        json.dumps({'type': 'turn.started'}),
        json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Synthetic progress'}}),
        *events,
        json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(output)}}),
        json.dumps({'type': 'turn.completed'})])


def cli(monkeypatch, raw, *, returncode=0, stderr=''):
    calls = []
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic-cli'))

    class Process:
        pid = 12345

        def __init__(self, args, **kw):
            self.returncode = returncode
            self.out, self.err = kw['stdout'], kw['stderr']
            self.args = args

        def communicate(self, prompt, **kw):
            calls.append({'args': self.args, 'prompt': prompt, **kw})
            self.out.write(raw)
            self.err.write(stderr)
            self.out.flush()
            self.err.flush()

    monkeypatch.setattr('subprocess.Popen', Process)
    return calls


def test_valid_seventy_views_survive_duplicate_web_transport_ids(ledger, monkeypatch):
    assets = [f'SYN{i}' for i in range(35)]
    output = {'regime': ['Synthetic transport test'], 'views': [
        {'asset': asset, 'horizon': horizon, 'view': 'ABSTAIN', 'p_outperform': .5,
         'confidence_reason': 'Synthetic', 'catalysts': [], 'priced_in_assessment': 'Synthetic',
         'second_order': 'Synthetic', 'counter_thesis': 'Synthetic', 'falsifier': 'Synthetic', 'event_risk': []}
        for asset in assets for horizon in ('1d', '5d')]}
    calls = cli(monkeypatch, stream(output, WEB_STARTED, WEB_COMPLETED))
    result = ModelRunner(ledger, clock=ledger.clock).infer('analyst_gpt', 'Synthetic protocol test',
        validator=lambda value: validate_analyst(value, assets, ledger.clock()))
    assert result == output and len(result['views']) == 70
    assert len(calls) == ledger.counts()['analyst_gpt'] == 1


@pytest.mark.parametrize('event', [
    '{"type":"item.completed","type":"item.started","item":{}}',
    '{"type":"item.completed","item":{"type":"web_search","type":"command_execution","id":"a","id":"b"}}',
    '{"type":"item.completed","item":{"type":"web_search","id":"a","id":"b","id":"c"}}',
    '{"type":"item.completed","item":{"type":"web_search","id":"a","id":7}}',
    '{"type":"item.completed","item":{"type":"agent_message","id":"a","id":"b","text":"{}"}}',
    '{"type":"item.completed","item":{"type":"web_search","id":"a","id":"b","action":{"type":"search","type":"other"}}}',
])
def test_transport_exception_never_accepts_other_ambiguous_fields(event):
    with pytest.raises(TraderError, match='MODEL_JSON_INVALID'):
        parse_gpt(stream({'regime': [], 'views': []}, event))


def test_duplicate_answer_fields_stay_rejected_after_successful_web_search():
    raw = stream({'regime': [], 'views': []}, WEB_COMPLETED).replace(
        '\\"views\\": []', '\\"views\\": [], \\"views\\": []')
    with pytest.raises(TraderError, match='MODEL_JSON_INVALID'):
        parse_gpt(raw)


def test_duplicate_web_ids_do_not_hide_taint_even_with_failed_cli_and_quota(ledger, monkeypatch):
    raw = stream({'regime': [], 'views': []}, WEB_COMPLETED,
                 '{"type":"item.completed","item":{"type":"command_execution"}}',
                 '{"type":"error","message":"rate limit"}')
    cli(monkeypatch, raw, returncode=1)
    with pytest.raises(TraderError, match='TAINTED_RUN'):
        ModelRunner(ledger, clock=ledger.clock).once('analyst_gpt', 'Synthetic')


def test_preflight_proves_schema_and_web_with_one_separate_call_and_no_view(ledger, monkeypatch, clock):
    from scripts.trading_lab.trader_agent.preflight import preflight
    from scripts.trading_lab.trader_agent.config import instant
    clock.at = instant('2026-10-06T10:30:00Z')
    calls = cli(monkeypatch, stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_STARTED, WEB_COMPLETED))
    result = preflight(ledger, ModelRunner(ledger, clock=clock))
    assert result['state'] == 'GREEN' and result['web_searches'] == 1
    assert ledger.counts() == {'gpt_preflight': 1}
    assert calls[0]['timeout'] <= 120 and len(calls[0]['prompt']) < 1000
    assert 'CONTEXT_DATA' not in calls[0]['prompt'] and 'TRADER_SKILL' not in calls[0]['prompt']
    assert not (ledger.root / 'evidence').exists()
    again = preflight(ledger, ModelRunner(ledger, clock=clock))
    assert again['state'] == 'ALREADY_ATTEMPTED' and len(calls) == 1


@pytest.mark.parametrize('raw,stderr,error', [
    (stream({'regime': ['PREFLIGHT_OK'], 'views': []}), '', 'MODEL_WEB_UNPROVEN'),
    (stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_COMPLETED.replace('text_result', 'error')), '', 'MODEL_WEB_UNPROVEN'),
    (stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_COMPLETED), 'code-mode host is disabled', 'MODEL_WEB_UNAVAILABLE'),
    (stream({'regime': ['PREFLIGHT_OK'], 'views': []}, '{"type":"error","message":"quota exhausted"}'), '', 'SKIPPED_QUOTA'),
    (stream({'regime': ['PREFLIGHT_OK'], 'views': []}, '{"type":"item.completed","item":{"type":"command_execution"}}'), '', 'TAINTED_RUN'),
    (stream({'regime': ['PREFLIGHT_OK'], 'views': [], 'extra': True}, WEB_COMPLETED), '', 'SCHEMA_INVALID'),
    (stream({'regime': ['Synthetic unexpected output'], 'views': []}, WEB_COMPLETED), '', 'PREFLIGHT_OUTPUT_INVALID'),
    ('bad-json', '', 'MODEL_JSON_INVALID'),
])
def test_failed_preflight_alerts_without_retry_or_consuming_analyst_calls(ledger, monkeypatch, raw, stderr, error):
    from scripts.trading_lab.trader_agent.preflight import preflight
    calls = cli(monkeypatch, raw, stderr=stderr)
    result = preflight(ledger, ModelRunner(ledger, clock=ledger.clock))
    assert result['state'] == 'BLOCKED' and result['error'] == error
    assert json.loads((ledger.root / 'alert.json').read_text())['code'] == 'GPT_PREFLIGHT_' + error
    assert json.loads((ledger.root / 'gpt-preflight.json').read_text()) == result
    assert ledger.counts() == {'gpt_preflight': 1} and len(calls) == 1
    assert preflight(ledger, ModelRunner(ledger, clock=ledger.clock))['state'] == 'ALREADY_ATTEMPTED'
    assert len(calls) == 1


def test_preflight_budget_survives_failure_and_runtime_change(ledger, tmp_path, monkeypatch, clock):
    from scripts.trading_lab.trader_agent.preflight import preflight
    from scripts.trading_lab.trader_agent.ledger import Ledger
    from datetime import timedelta
    calls = cli(monkeypatch, 'bad-json')
    assert preflight(ledger, ModelRunner(ledger, clock=clock))['state'] == 'BLOCKED'
    other = Ledger(tmp_path / 'other-runtime', ledger.grant, clock=clock, budget_root=ledger.budget_root)
    assert preflight(other, ModelRunner(other, clock=clock))['state'] == 'ALREADY_ATTEMPTED'
    assert len(calls) == 1
    clock.at += timedelta(days=1)
    assert preflight(other, ModelRunner(other, clock=clock))['state'] == 'BLOCKED'
    assert len(calls) == 2 and other.counts() == {'gpt_preflight': 1}
    assert json.loads((other.root / 'alert.json').read_text())['code'] == 'GPT_PREFLIGHT_MODEL_JSON_INVALID'


@pytest.mark.parametrize('blocked', ['paused', 'expired', 'owner'])
def test_preflight_never_bypasses_global_controls(ledger, monkeypatch, blocked, clock):
    from contextlib import nullcontext
    from scripts.trading_lab.trader_agent.preflight import preflight
    from scripts.trading_lab.trader_agent.config import instant
    calls = cli(monkeypatch, stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_COMPLETED))
    if blocked == 'paused':
        (ledger.budget_root / 'PAUSED').touch()
    elif blocked == 'expired':
        clock.at = instant(ledger.grant.payload['not_after'])
    with ledger.owner() if blocked == 'owner' else nullcontext():
        assert preflight(ledger, ModelRunner(ledger, clock=clock))['state'] == 'BLOCKED'
    assert not calls and not ledger.counts()


def test_preflight_does_not_reduce_the_six_real_model_call_budget(ledger, monkeypatch):
    from scripts.trading_lab.trader_agent.preflight import preflight
    cli(monkeypatch, stream({'regime': ['PREFLIGHT_OK'], 'views': []}, WEB_COMPLETED))
    assert preflight(ledger, ModelRunner(ledger, clock=ledger.clock))['state'] == 'GREEN'
    for role in ('analyst_claude', 'analyst_gpt', 'reviewer'):
        ledger.reserve(role)
        ledger.reserve(role)
    assert sum(ledger.counts().values()) == 7
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        ledger.reserve('analyst_gpt')
