"""Opus protocol changes, synthetic probes and dispatches only."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

from scripts.trading_lab.replay_bench import audit, cli, experiment, grant as gr, models, run, spec
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.sources.canonical import sha256_canonical
from test_night_safety import AT, client
from test_replay_bench import daily_bars, item, universe_bars


def test_sonnet_refused_before_budget_and_dispatch(tmp_path):
    c = client(tmp_path, lambda *a, **kw: pytest.fail('Sonnet dispatched'))
    with pytest.raises(gr.ReplayError, match='MODEL_NOT_ALLOWED'):
        c.call('sonnet', 'q', {}, tag='x')
    assert not c.budget.path.exists()


def test_eligible_opus_dates_still_require_input_proof_before_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'evidence', lambda root: {'floor': {'opus': '2026-07-01'}})
    monkeypatch.setattr(experiment, 'committed', lambda: None)
    with pytest.raises(gr.ReplayError, match='POINT_IN_TIME_INPUT_PROOF_MISSING'):
        experiment.run_group(tmp_path, 'dev', 'ab', object(), object())
    assert not (tmp_path / 'v4/transcripts').exists()


def test_60th_call_allowed_61st_refused_and_next_night_resumes(tmp_path):
    c = client(tmp_path)
    c.grant.model_limit = lambda family: 150
    atomic_json(c.budget.path, {'2026-10-19': {'model:opus': 59}})
    c.call('opus', 'q', {}, tag='60')
    assert c.budget.used('model:opus', AT) == 60
    with pytest.raises(gr.ReplayError, match='BUDGET_EXHAUSTED'):
        c.call('opus', 'q', {}, tag='61')
    c.clock = lambda: AT + timedelta(days=1)
    c.call('opus', 'q', {}, tag='61')
    assert c.budget.used('model:opus', c.clock()) == 1


def test_recovered_quota_response_stops_other_tags(tmp_path):
    c = client(tmp_path)
    c.call('opus', 'q', {}, tag='x')
    path = next(c.transcripts.glob('*.json'))
    row = json.loads(path.read_text())
    row.update(raw='', err='usage limit', returncode=1)
    atomic_json(path, row)
    with pytest.raises(gr.ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call('opus', 'q', {}, tag='x')
    with pytest.raises(gr.ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call('opus', 'different', {}, tag='y')
    assert c.budget.used('model:opus', AT) == 1


def test_all_decision_stages_dispatch_opus(tmp_path, monkeypatch):
    calls = []
    def validated(claude, family, text, schema, check, tag):
        calls.append((family, tag))
        return {'views': []}, [], []
    monkeypatch.setattr(run, 'call_validated', validated)
    monkeypatch.setattr(run, 'views_with_review', lambda *a, **kw: [])
    universe = type('Universe', (), {'bars': universe_bars(), 'news': []})()
    run.decide_ab('2026-07-15', universe, object(), tmp_path / 'named', {})
    run.decide_ab('2026-07-15', universe, object(), tmp_path / 'anon', {}, anonymized=True)
    monkeypatch.setattr(run.md, 'news', lambda *a: ([item(1, '2026-07-15T10:00:00Z', ['SMLL'], 'Guidance cut')], []))
    monkeypatch.setattr(run.md, 'stock_bars', lambda *a: ({'SMLL': daily_bars(price=20, volume=2e6)}, []))
    run.decide_c('2026-07-15', object(), object(), tmp_path / 'news', {}, ['cut'])
    assert len(calls) == 4 and all(family == 'opus' for family, _ in calls)
    assert sum('reviewer' in tag for _, tag in calls) == 1


def synthetic_probes(tmp_path, monkeypatch):
    questions = [{'id': f'D{m}-{i}', 'month': f'2026-{m:02}', 'symbol': 'SYN', 'up': i < 7}
                 for m in range(5, 10) for i in range(12)]
    eps = [{'id': f'Q{i}', 'month': '2026-07', 'symbol': 'SYN', 'eps': 1.23} for i in range(35)]
    values = {name: {'questions': qs, 'rule': 'synthetic pre-call rule'} for name, qs in (
        ('probe-preregistration.json', eps), ('probe2-preregistration.json', questions),
        ('probe3-preregistration.json', questions))}
    for family in ('opus', 'sonnet'):
        for prefix, qs in (('', eps), ('probe2-', questions), ('probe3-', questions)):
            answers = {q['id']: 'unknown' for q in qs}
            if prefix == 'probe3-':
                answers = {}
                for month in range(5, 10):
                    correct = (10 if month == 8 else 9 if month == 9 else 7) if family == 'sonnet' else 7
                    for i, q in enumerate(q for q in qs if q['month'] == f'2026-{month:02}'):
                        up = q['up'] if i < correct else not q['up']
                        answers[q['id']] = 'up' if up else 'down'
            values[prefix + 'answers-' + family + '.json'] = {'raw': json.dumps(answers), 'answers': answers}
    for name, value in values.items():
        atomic_json(tmp_path / name, value)
    value = deepcopy(spec.spec())
    value['probe']['files'] = {name: spec.file_hash(tmp_path / name) for name in values}
    monkeypatch.setattr(spec, 'spec', lambda: value)
    monkeypatch.setenv('REPLAY_PROBE_ROOT', str(tmp_path))
    return value


def test_operator_probes_regraded_and_digest_bound(tmp_path, monkeypatch):
    synthetic_probes(tmp_path, monkeypatch)
    result = audit.evidence(tmp_path)
    assert result['floor'] == {'opus': '2026-07-01'} and result['excluded_families'] == ['sonnet']
    assert result['tables']['opus']['forced_direction']['pooled_correct'] == 21
    assert result['tables']['sonnet']['forced_direction']['pooled_correct'] == 26
    assert result['tables']['opus']['exact_eps'] == {'correct': 0, 'n': 35}
    file = tmp_path / 'probe3-answers-opus.json'
    file.write_text(file.read_text() + ' ')
    with pytest.raises(gr.ReplayError, match='OPERATOR_PROBE_HASH_MISMATCH'):
        audit.evidence(tmp_path)


def test_missing_probe_and_inconsistent_raw_response_block(tmp_path, monkeypatch):
    value = synthetic_probes(tmp_path, monkeypatch)
    file = tmp_path / 'answers-opus.json'
    response = json.loads(file.read_text())
    response['answers'].pop('Q0')
    atomic_json(file, response)
    value['probe']['files'][file.name] = spec.file_hash(file)
    with pytest.raises(gr.ReplayError, match='OPERATOR_PROBE_RESPONSE_INVALID'):
        audit.evidence(tmp_path)
    file.unlink()
    with pytest.raises(gr.ReplayError, match='OPERATOR_PROBE_MISSING'):
        audit.evidence(tmp_path)


def test_forced_rule_boundary_and_base_rate():
    rule = spec.spec()['probe']
    qs = [{'id': str(i), 'month': f'2026-{7+i//12:02}', 'up': i % 12 < 7} for i in range(36)]
    def answer_for(counts):
        return {q['id']: ('up' if q['up'] else 'down') if i % 12 < counts[i // 12]
                else ('down' if q['up'] else 'up') for i, q in enumerate(qs)}
    assert not audit.forced_summary(qs, answer_for([7, 8, 8]), rule)['flagged']
    assert audit.forced_summary(qs, answer_for([8, 8, 8]), rule)['flagged']
    assert audit.forced_summary(qs, answer_for([10, 0, 0]), rule)['flagged']
    assert audit.forced_summary(qs, answer_for([7, 7, 7]), rule)['always_up_pooled'] == 21


def test_only_model_is_added_to_previous_frozen_test_configs():
    previous = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_test_freeze_v2.json').read_text())
    for group in ('ab', 'cd'):
        before = run.config_payload('test', group, amended=False)
        after = run.config_payload('test', group)
        assert sha256_canonical(before) == previous['config_hashes'][group]
        assert after.pop('model') == 'opus' and after == before
    old = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_spec_v3.json').read_text())
    current = spec.spec()
    expected = deepcopy(old['variants'])
    expected['a_v1_as_live'].update(analyst='opus', reviewer='opus')
    expected['b_v1_no_reviewer']['analyst'] = 'same Opus call as a_v1_as_live'
    expected['c_news_small_mid']['analyst'] = 'opus'
    assert current['variants'] == expected
    for key in ('periods', 'fills', 'costs', 'labels', 'baselines', 'information_snapshot', 'decision_time_utc'):
        assert current[key] == old[key]


def test_input_seal_is_required_and_freeze_is_not_overwritten(tmp_path, monkeypatch):
    frozen = {'config_hashes': {'ab': 'a', 'cd': 'b'}, 'test_days': {'ab': ['d'], 'cd': ['d']}}
    file = tmp_path / 'freeze.json'
    atomic_json(file, frozen)
    monkeypatch.setattr(experiment, 'FREEZE', file)
    monkeypatch.setattr(experiment, 'INPUT_SEAL', tmp_path / 'missing.json')
    monkeypatch.setattr(experiment, 'committed', lambda: None)
    monkeypatch.setattr(experiment, 'frozen_payload', lambda root: frozen)
    monkeypatch.setattr(experiment, 'input_proof', lambda root: 'synthetic-proof-hash')
    original = file.read_bytes()
    with pytest.raises(gr.ReplayError, match='TEST_INPUTS_NOT_SEALED'):
        experiment.check_freeze(tmp_path)
    assert file.read_bytes() == original


def test_progress_report_cannot_claim_done_without_scored_test(tmp_path, monkeypatch):
    report = tmp_path / 'report.md'
    report.write_text('STATUS: BLOCKED\n\nOperator summary.\n')
    monkeypatch.setenv('REPLAY_REPORT', str(report))
    cli.progress_report(tmp_path, {'status': 'DONE'})
    assert report.read_text().startswith('STATUS: BLOCKED\n')
    assert 'Operator summary.' in report.read_text()
    atomic_json(tmp_path / 'v4/results/test.json', {'status': 'SCORED', 'spec_hash': spec.SPEC_HASH,
                                                 'code_hash': run.code_hash()})
    cli.progress_report(tmp_path, {'status': 'DONE'})
    assert report.read_text().startswith('STATUS: DONE\n')
    assert report.read_text().count('<!-- replay-opus-progress -->') == 1


def timer_installer_fixture(tmp_path, monkeypatch):
    import importlib.util
    import os
    import subprocess
    path = spec.ROOT / 'deploy/research/install_replay_timer.py'
    module_spec = importlib.util.spec_from_file_location('opus_timer_install', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    target = tmp_path / '.config/systemd/user'
    target.mkdir(parents=True)
    service = target / 'hyprl-replay-bench.service'
    timer = target / 'hyprl-replay-bench.timer'
    service.write_text('ExecStart=python -m scripts.trading_lab.replay_bench.cli night --expected-commit old\n')
    timer.write_text('Unit=hyprl-replay-bench.service\n')
    env = tmp_path / 'research.env'
    env.write_text('REPLAY_DATA_ROOT=synthetic\n')
    os.chmod(env, 0o600)
    binary = tmp_path / 'claude'
    binary.write_text('synthetic executable')
    os.chmod(binary, 0o700)
    def read(args, **kwargs):
        return 'inactive\n' if args[0] == 'systemctl' else ('synthetic-sha\n' if 'rev-parse' in args else '')
    monkeypatch.setattr(subprocess, 'check_output', read)
    args = ['--environment', str(env), '--claude', str(binary), '--replace', '--backup-dir', str(tmp_path / 'backup')]
    return module, args, service, timer


def test_timer_verification_failure_preserves_existing_units(tmp_path, monkeypatch):
    import subprocess
    installer, args, service, timer = timer_installer_fixture(tmp_path, monkeypatch)
    original = service.read_bytes(), timer.read_bytes()
    def fail(args, **kwargs):
        assert args[:3] == ['systemd-analyze', '--user', 'verify']
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        installer.main(args)
    assert (service.read_bytes(), timer.read_bytes()) == original
    assert not (tmp_path / 'backup').exists()


def test_replacement_refuses_foreign_service(tmp_path, monkeypatch):
    installer, args, service, timer = timer_installer_fixture(tmp_path, monkeypatch)
    service.write_text('ExecStart=unrelated-service\n')
    with pytest.raises(SystemExit, match='not the expected research replay units'):
        installer.main(args)
    assert service.read_text() == 'ExecStart=unrelated-service\n'
