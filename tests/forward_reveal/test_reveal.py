import hashlib
import json
import subprocess

import pytest

from scripts import forward_reveal as reveal
from scripts.trading_lab.replay_bench.grant import instant
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.replay_seal_forward import commitment, forward
from scripts.trading_lab.sources.canonical import sha256_canonical


def decision(ledger, day='2026-10-22', published_at=None, reason='Décision synthétique'):
    values = {}
    for family in ('opus', 'gpt'):
        value = {'schema': 'live-forward-decision-v2', 'provenance': 'LIVE_FORWARD',
                 'family': family, 'model': 'synthetic-model-' + family,
                 'snapshot_hash': 'a' * 64, 'decision_time': day + 'T13:05:00Z',
                 'output': {'reason': reason}, 'call': {'transcript': 'synthetic.json'}}
        value['decision_hash'] = sha256_canonical(value)
        atomic_json(forward.directory(ledger.root, day) / (family + '.json'), value)
        values[family] = value
    ledger.clock = lambda: instant(published_at or day + 'T13:16:00Z')
    receipt = ledger.publish('forward', day, {**{f: v['decision_hash'] for f, v in values.items()},
                                            'snapshot': 'a' * 64})
    return values, receipt


def test_revealed_files_are_the_original_hash_preimages(ledger):
    values, receipt = decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    publisher = reveal.Publisher(ledger)
    result = publisher.run(publish=True)
    assert result['sessions']['2026-10-22']['status'] == 'DELIVERED'
    for family, label in [('opus', 'HypR-A1'), ('gpt', 'HypR-E1')]:
        path = ledger.worktree / 'reveals' / '2026-10-22' / (label + '.json')
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == values[family]['decision_hash']
        assert not raw.endswith(b'\n')
        assert json.loads(raw)['model'] == 'synthetic-model-' + family
        assert b'\\u00e9' in raw
        pushed = publisher.git('show', commitment.BRANCH + ':' + str(path.relative_to(ledger.worktree)))
        assert pushed.encode() == raw
    readme = (path.parent / 'README.md').read_text()
    assert receipt['ledger_commit'] in readme
    assert 'sha256sum' in readme and 'HypR-A1.json' in readme and 'HypR-E1.json' in readme
    assert 'identités réelles des modèles' in readme
    assert 'Révélations' in ledger.log.read_text()
    before = ledger.log.read_bytes()
    assert publisher.run(publish=True)['sessions']['2026-10-22']['status'] == 'DELIVERED'
    assert ledger.log.read_bytes() == before


@pytest.mark.parametrize('day,before,after', [
    ('2026-10-22', '2026-10-22T20:14:59Z', '2026-10-22T20:15:00Z'),
    ('2026-11-02', '2026-11-02T21:14:59Z', '2026-11-02T21:15:00Z'),
    ('2026-11-27', '2026-11-27T18:14:59Z', '2026-11-27T18:15:00Z'),
])
def test_close_gate_uses_dst_and_short_sessions(ledger, day, before, after):
    decision(ledger, day, day + ('T13:16:00Z' if day < '2026-11' else 'T14:16:00Z'))
    publisher = reveal.Publisher(ledger)
    ledger.clock = lambda: instant(before)
    # Pending sessions do not even open the decision files.
    path = forward.directory(ledger.root, day) / 'opus.json'
    saved = path.read_bytes()
    path.write_text('invalid')
    assert publisher.run(publish=True)['sessions'][day]['status'] == 'WITHHELD'
    assert not (ledger.worktree / 'reveals').exists()
    path.write_bytes(saved)
    ledger.clock = lambda: instant(after)
    assert publisher.run()['sessions'][day]['status'] == 'READY'
    assert not (ledger.worktree / 'reveals').exists()
    assert not (ledger.root / 'reveals').exists()


@pytest.mark.parametrize('published_at', ['2026-10-22T13:30:00Z', '2026-10-22T14:22:00Z'])
def test_late_commitments_only_publish_unpublished_readme(ledger, published_at):
    decision(ledger, published_at=published_at)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result = reveal.Publisher(ledger).run(publish=True)
    row = result['sessions']['2026-10-22']
    assert row['reason'] == 'HASH_NOT_PUBLISHED_BEFORE_US_OPEN'
    assert {r['status'] for r in row['agents'].values()} == {'UNPUBLISHED'}
    assert list((ledger.worktree / 'reveals/2026-10-22').glob('*.json')) == []


def test_intraday_is_not_reclassified_as_preopen(ledger):
    decision(ledger, published_at='2026-10-22T14:22:00Z')
    path = commitment.receipt_path(ledger.root, 'forward', '2026-10-22')
    receipt = json.loads(path.read_text())
    path.unlink()
    ledger.publish('forward', '2026-10-22-intraday', receipt['line']['hashes'])
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result = reveal.Publisher(ledger).run(publish=True)
    assert result['sessions']['2026-10-22-intraday']['reason'] == 'HASH_NOT_PUBLISHED_BEFORE_US_OPEN'
    assert not list((ledger.worktree / 'reveals').rglob('*.json'))


def test_missing_receipt_marks_unpublished_without_disclosing(ledger):
    decision(ledger)
    commitment.receipt_path(ledger.root, 'forward', '2026-10-22').unlink()
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result, files = reveal.Publisher(ledger).plan('2026-10-22')
    assert result['status'] == 'UNPUBLISHED'
    assert list(files) == ['reveals/2026-10-22/README.md']


def test_both_agent_hashes_must_be_committed_before_revealing_session(ledger):
    values, _ = decision(ledger)
    day = '2026-10-23'
    for family, value in values.items():
        atomic_json(forward.directory(ledger.root, day) / (family + '.json'), value)
    ledger.clock = lambda: instant(day + 'T13:16:00Z')
    ledger.publish('forward', day, {'snapshot': 'a' * 64, 'opus': values['opus']['decision_hash']})
    ledger.clock = lambda: instant(day + 'T20:15:00Z')
    result, files = reveal.Publisher(ledger).plan(day)
    assert result['reason'] == 'INCOMPLETE_SESSION_COMMITMENT'
    assert list(files) == ['reveals/2026-10-23/README.md']


def test_failure_hash_is_not_published_as_decision(ledger):
    day = '2026-10-22'
    ledger.clock = lambda: instant(day + 'T13:16:00Z')
    failure = {'error': 'MODEL_USAGE_LIMIT', 'at': day + 'T13:10:00Z', 'snapshot_hash': 'a' * 64}
    for family in ('opus', 'gpt'):
        atomic_json(forward.directory(ledger.root, day) / (family + '-failure.json'), failure)
    ledger.publish('forward', day, {'snapshot': 'a' * 64, **{f: sha256_canonical(failure) for f in ('opus', 'gpt')}})
    ledger.clock = lambda: instant(day + 'T20:15:00Z')
    publisher = reveal.Publisher(ledger)
    result, files = publisher.plan(day)
    assert {r['status'] for r in result['agents'].values()} == {'NO_DECISION'}
    assert list(files) == ['reveals/2026-10-22/README.md']


@pytest.mark.parametrize('mutation', ['output', 'snapshot', 'family', 'hash'])
def test_corrupt_records_never_reach_git(ledger, mutation):
    values, _ = decision(ledger)
    value = values['opus']
    if mutation == 'output':
        value['output']['reason'] = 'changed'
    else:
        value[{'snapshot': 'snapshot_hash', 'family': 'family', 'hash': 'decision_hash'}[mutation]] = 'changed'
    atomic_json(forward.directory(ledger.root, '2026-10-22') / 'opus.json', value)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result = reveal.Publisher(ledger).run(publish=True)
    assert result['sessions']['2026-10-22']['error'] == 'REVEAL_DECISION_HASH_MISMATCH'
    assert not (ledger.worktree / 'reveals').exists()


def test_receipt_must_reference_pushed_ledger_bytes(ledger):
    _, receipt = decision(ledger)
    receipt['line']['hashes']['opus'] = 'b' * 64
    receipt['line_sha256'] = sha256_canonical(receipt['line'])
    atomic_json(commitment.receipt_path(ledger.root, 'forward', '2026-10-22'), receipt)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result = reveal.Publisher(ledger).run(publish=True)
    assert result['sessions']['2026-10-22']['error'] == 'COMMITMENT_LEDGER_MISMATCH'
    assert not (ledger.worktree / 'reveals').exists()


def test_late_ledger_commit_cannot_use_a_preopen_receipt_timestamp(ledger):
    _, receipt = decision(ledger)
    publisher = reveal.Publisher(ledger)
    ledger.clock = lambda: instant('2026-10-22T14:00:00Z')
    publisher.git('commit', '--allow-empty', '-qm', 'late pointer')
    publisher.git('push', '-q', 'origin', commitment.BRANCH)
    receipt['ledger_commit'] = publisher.git('rev-parse', 'HEAD')
    atomic_json(commitment.receipt_path(ledger.root, 'forward', '2026-10-22'), receipt)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    result, files = publisher.plan('2026-10-22')
    assert result['reason'] == 'LEDGER_COMMIT_NOT_BEFORE_US_OPEN'
    assert list(files) == ['reveals/2026-10-22/README.md']


def test_push_failure_resumes_same_commit_before_notifying(ledger):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    old_log = ledger.log.read_bytes()
    original = ledger.runner
    ledger.runner = lambda args, **kw: subprocess.CompletedProcess(args, 1, '', '') if 'push' in args else original(args, **kw)
    publisher = reveal.Publisher(ledger)
    assert publisher.run(publish=True)['status'] == 'BLOCKED'
    head = publisher.git('rev-parse', 'HEAD')
    assert ledger.log.read_bytes() == old_log
    ledger.runner = original
    assert publisher.run(publish=True)['sessions']['2026-10-22']['reveal_commit'] == head


def test_unknown_notice_delivery_is_never_automatically_resent(ledger):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    original, notices = ledger.runner, []
    def unconfirmed(args, **kwargs):
        if args == [str(ledger.notify), 'brouillons']:
            notices.append(kwargs['input'])
            return subprocess.CompletedProcess(args, 0, 'fallback', '')
        return original(args, **kwargs)
    ledger.runner = unconfirmed
    publisher = reveal.Publisher(ledger)
    for _ in range(2):
        assert publisher.run(publish=True)['sessions']['2026-10-22']['error'] == 'REVEAL_NOTICE_DELIVERY_UNKNOWN'
    assert len(notices) == 1


def test_changed_existing_reveal_is_never_overwritten(ledger):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    path = ledger.worktree / 'reveals/2026-10-22/HypR-A1.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'preexisting')
    result = reveal.Publisher(ledger).run(publish=True)
    assert result['sessions']['2026-10-22']['error'] == 'REVEAL_EXISTING_BYTES_DIFFER'
    assert path.read_bytes() == b'preexisting'


def test_private_preimage_is_blocked_without_redaction(ledger):
    decision(ledger, reason='synthetic-contact@example.invalid')
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    with pytest.raises(reveal.ReplayError, match='REVEAL_PRIVATE_CONTENT'):
        reveal.Publisher(ledger).plan('2026-10-22')
    assert not (ledger.worktree / 'reveals').exists()


def test_unrelated_dirty_files_and_commits_block_publication(ledger):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    publisher = reveal.Publisher(ledger)
    path = ledger.worktree / 'unrelated.txt'
    path.write_text('synthetic unrelated change')
    assert publisher.run(publish=True)['sessions']['2026-10-22']['error'] == 'LEDGER_UNRELATED_CHANGES'
    publisher.git('add', 'unrelated.txt')
    publisher.git('commit', '-qm', 'unrelated')
    assert publisher.run(publish=True)['sessions']['2026-10-22']['error'] == 'LEDGER_UNRELATED_COMMITS'
    assert not (ledger.worktree / 'reveals').exists()


def test_clock_is_rechecked_before_any_write(ledger):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    publisher = reveal.Publisher(ledger)
    result, files = publisher.plan('2026-10-22')
    ledger.clock = lambda: instant('2026-10-22T20:14:59Z')
    with pytest.raises(reveal.ReplayError, match='REVEAL_BEFORE_CLOSE'):
        publisher.publish(result, files)
    assert not (ledger.worktree / 'reveals').exists()


def test_symlink_target_is_rejected(ledger, tmp_path):
    decision(ledger)
    ledger.clock = lambda: instant('2026-10-22T20:15:00Z')
    other = tmp_path / 'outside'
    other.mkdir()
    (ledger.worktree / 'reveals').symlink_to(other, target_is_directory=True)
    # Unrelated dirty paths are rejected even before any symlink can be followed.
    result = reveal.Publisher(ledger).run(publish=True)
    assert result['status'] == 'BLOCKED'
    assert not list(other.iterdir())
