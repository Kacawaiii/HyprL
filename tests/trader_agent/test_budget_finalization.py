"""Budget pressure must complete through the model boundary without another web tool."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.trading_lab.trader_agent.runners import ModelRunner
from scripts.trading_lab.trader_agent import work_budget as work
from test_work_budget import bounded, request, result


def replay_cli(monkeypatch, tmp_path, research, final):
    script = tmp_path / 'recorded_cli.py'
    trace = tmp_path / 'transport.jsonl'
    script.write_text('''import json,sys
mode=sys.argv[1]
init=json.loads(sys.stdin.readline())
print(json.dumps({'type':'control_response','response':{'subtype':'success','request_id':init['request_id']}}),flush=True)
user=json.loads(sys.stdin.readline())
with open(sys.argv[3],'a') as f:f.write(json.dumps({'mode':mode,'prompt':user})+'\\n')
for event in json.loads(sys.argv[2]):
 print(json.dumps(event),flush=True)
 if event.get('type')=='control_request':
  reply=sys.stdin.readline()
  if not reply:sys.exit(0)
  with open(sys.argv[3],'a') as f:f.write(reply)
''')
    def command(*args, final_only=False, **kwargs):
        return [sys.executable, str(script), 'final' if final_only else 'research',
                json.dumps(final if final_only else research), str(trace)]
    monkeypatch.setattr('scripts.trading_lab.trader_agent.runners.command', command)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='recorded-cli'))
    return trace


@pytest.mark.parametrize('role,tool', [('analyst_claude','WebSearch'), ('analyst_claude','WebFetch'),
                                       ('reviewer','WebSearch'), ('reviewer','WebFetch')])
def test_tool_ceiling_produces_validated_final_without_more_tools(bounded, monkeypatch, tmp_path, role, tool):
    count = work.spec()['tools'][role][tool]
    output = {'verdicts': []} if role == 'reviewer' else {'regime': [], 'views': []}
    requests = [request(i, tool, **({'url': 'https://example.invalid/news'} if tool == 'WebFetch' else {}))
                for i in range(count + 1)]
    trace = replay_cli(monkeypatch, tmp_path, requests, [result(output)])
    runner = ModelRunner(bounded, clock=bounded.clock)
    assert runner.infer(role, 'Frozen synthetic context') == output
    events = [json.loads(l) for l in trace.read_text().splitlines()]
    assert [e['mode'] for e in events if 'mode' in e] == ['research', 'final']
    assert len([e for e in events if e.get('type') == 'control_response']) == count
    assert bounded.counts() == {role: 1}
    assert runner.work_metadata[role]['finalization_reason'] == 'TOOL_LIMIT'


def test_turn_ceiling_keeps_observed_data_for_tool_free_answer(bounded, monkeypatch, tmp_path):
    events = [{'type': 'assistant', 'message': {'id': str(i), 'content': [
        {'type': 'text', 'text': 'Synthetic observation'}]}} for i in range(work.spec()['max_turns'])]
    trace = replay_cli(monkeypatch, tmp_path, events, [result()])
    runner = ModelRunner(bounded, clock=bounded.clock)
    assert runner.infer('analyst_claude', 'Original frozen context')['views'] == []
    final = next(json.loads(l) for l in trace.read_text().splitlines() if json.loads(l).get('mode') == 'final')
    assert 'Original frozen context' in final['prompt']['message']['content']
    assert 'Synthetic observation' in final['prompt']['message']['content']
    assert runner.work_metadata['analyst_claude']['finalization_reason'] == 'TURN_LIMIT'
    # Unknown helper usage is explicit and cannot prevent the reserved reviewer.
    assert bounded.work_counts()[0]['incomplete_usage'] == 1
    bounded.reserve('reviewer')


def test_token_ceiling_finalizes_and_recovery_capacity_remains(bounded, monkeypatch, tmp_path):
    event = {'type': 'assistant', 'message': {'id': 'large-research', 'content': [],
             'usage': {'input_tokens': 750000}}}
    replay_cli(monkeypatch, tmp_path, [event], [result()])
    runner = ModelRunner(bounded, clock=bounded.clock)
    assert runner.infer('analyst_claude', 'Synthetic')['views'] == []
    assert runner.work_metadata['analyst_claude']['finalization_reason'] == 'TOKEN_LIMIT'
    bounded.reserve('analyst_claude')
    bounded.reserve('reviewer')
    assert bounded.counts() == {'analyst_claude': 2, 'reviewer': 1}


def test_final_transport_cannot_request_web(bounded, monkeypatch, tmp_path):
    import pytest
    from scripts.trading_lab.trader_agent.config import TraderError
    from scripts.trading_lab.trader_agent.runners import command
    args = command('analyst_claude', bounded.grant, limits=work.spec(), final_only=True)
    assert args[args.index('--tools') + 1] == ''
    assert args[args.index('--allowedTools') + 1] == ''
    assert '--max-turns' not in args  # the native CLI must not terminate a structured completion
    replay_cli(monkeypatch, tmp_path, [{'type': 'result', 'subtype': 'error_max_turns'}], [request(999)])
    with pytest.raises(TraderError, match='TAINTED_RUN'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic')
    assert bounded.counts() == {'analyst_claude': 1}


def test_rehearsal_uses_same_budget_completion_with_zero_web(bounded, monkeypatch, tmp_path):
    trace = replay_cli(monkeypatch, tmp_path, [], [result(turns=2)])
    runner = ModelRunner(bounded, clock=bounded.clock)
    assert runner.once('analyst_claude', 'Synthetic probe', probe=True)['views'] == []
    assert [json.loads(l)['mode'] for l in trace.read_text().splitlines()] == ['final']
    assert bounded.counts() == {'rehearsal_analyst_claude': 1}
    assert runner.work_metadata['analyst_claude']['finalization_reason'] == 'TOOL_LIMIT'
    assert bounded.work_counts()[0]['incomplete_usage'] == 0


def test_rehearsal_receipt_binds_budget_path_and_reuses_only_same_policy(bounded, monkeypatch, tmp_path):
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks, MonitorState
    expected = {'regime': ['PREFLIGHT_OK'], 'views': []}
    replay_cli(monkeypatch, tmp_path, [], [result(expected, turns=2)])
    checks = LiveChecks(SimpleNamespace(), MonitorState(tmp_path / 'monitor'), clock=bounded.clock)
    checks.ready, checks.grant, checks.ledger = True, bounded.grant, bounded
    proof = checks.model('analyst_claude')
    assert proof['work_budget']['finalization_reason'] == 'TOOL_LIMIT'
    assert proof['work_budget']['final_tools'] == 0
    assert checks.model('analyst_claude', retry=True)['cached']
    assert bounded.counts() == {'rehearsal_analyst_claude': 1}


def test_old_reservation_hash_cannot_be_rebound(bounded):
    import pytest
    from scripts.trading_lab.trader_agent.config import TraderError
    from dataclasses import replace
    original = bounded.grant
    bounded.grant = replace(original, work_budget_approval=None)
    seq = bounded.reserve('analyst_claude')
    bounded.grant = original
    # Seed an old synthetic reservation without altering any immutable row.
    with bounded.connect() as db:
        db.execute('INSERT INTO work_reservations VALUES(?,?,?,?,?,?,?,?)',
                   (seq, '2026-10-21', 'am', 'base', 'analyst_claude', 95000,
                    'b55b99d515efe0e9598b866683ffeb00a019a94a89d45124bd61dcabcf4307ff', '0' * 64))
    with pytest.raises(TraderError, match='WORK_BUDGET_SPEC_HASH_MISMATCH'):
        bounded.reserve('reviewer')


def test_budget_completion_still_rejects_future_sources(bounded, monkeypatch, tmp_path):
    from dataclasses import replace
    from scripts.trading_lab.trader_agent.config import TraderError
    from scripts.trading_lab.trader_agent.ledger import Ledger
    from scripts.trading_lab.trader_agent.synthetic import SyntheticRunner
    from scripts.trading_lab.trader_agent.runners import prompt
    from scripts.trading_lab.trader_agent.schemas import validate_analyst
    shape = Ledger(tmp_path / 'shape', replace(bounded.grant, work_budget_approval=None), clock=bounded.clock)
    context = {'universe': ['AAPL'], 'decision_time': '2026-10-21T12:00:00Z'}
    output = SyntheticRunner(shape, clock=bounded.clock).once('analyst_claude', prompt(context, 'Synthetic'))
    output['views'][0]['catalysts'][0]['published_at'] = '2026-10-21T12:00:01Z'
    replay_cli(monkeypatch, tmp_path, [{'type': 'result', 'subtype': 'error_max_turns'}], [result(output)])
    with pytest.raises(TraderError, match='FUTURE_SOURCE'):
        ModelRunner(bounded, clock=bounded.clock).infer('analyst_claude', 'Synthetic',
            validator=lambda value: validate_analyst(value, ['AAPL'], bounded.clock()))
    assert bounded.counts() == {'analyst_claude': 2}


def test_missing_probe_usage_remains_incomplete(bounded, monkeypatch, tmp_path):
    from scripts.trading_lab.trader_agent.config import TraderError
    replay_cli(monkeypatch, tmp_path, [], [{**result(), 'modelUsage': {}}])
    with pytest.raises(TraderError, match='MODEL_USAGE_INVALID'):
        ModelRunner(bounded, clock=bounded.clock).once('analyst_claude', 'Synthetic probe', probe=True)
    assert bounded.work_counts()[0]['incomplete_usage'] == 1
