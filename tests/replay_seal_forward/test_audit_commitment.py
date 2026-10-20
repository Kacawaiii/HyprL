import json
import subprocess
from datetime import timedelta

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.replay_seal_forward import commitment
from scripts.trading_lab.replay_seal_forward import experiment as ex, spec
from scripts.trading_lab.sources.canonical import sha256_canonical


def test_all_nightly_hashes_reach_discord_and_the_pushed_ledger(ledger, tmp_path):
    hashes = {f'opus:dev:ab:2026-07-{i:03d}': sha256_canonical(i) for i in range(150)}
    result = ledger.publish('replay-night', '2026-10-20', hashes)
    assert result['messages'] > 1
    text = ledger.log.read_text()
    assert all(value in text for value in hashes.values())
    assert all(len(message) < 2000 for message in text.split('--\n'))
    pushed = subprocess.check_output(['git', '--git-dir', str(ledger.remote), 'show',
                                     commitment.BRANCH + ':' + commitment.LEDGER_FILE], text=True)
    assert json.loads(pushed)['hashes'] == hashes
    assert ledger.publish('replay-night', '2026-10-20', hashes) == result
    assert commitment.require(tmp_path, 'replay-night', '2026-10-20', hashes,
                              instant(result['published_at']) + timedelta(seconds=1)) == result
    with pytest.raises(ReplayError, match='PUBLISHED_TOO_LATE'):
        commitment.require(tmp_path, 'replay-night', '2026-10-20', hashes, instant(result['published_at']))
    with pytest.raises(ReplayError, match='NOT_HASH_ONLY'):
        ledger.publish('forward', '2026-10-20', {'opus': 'FAILED:SENSITIVE_ERROR'})


def test_wrong_channel_or_forged_ledger_line_cannot_be_used_for_scoring(ledger, tmp_path):
    hashes = {'snapshot': 'a'*64, 'opus': 'b'*64, 'gpt': 'c'*64}
    result = ledger.publish('forward', '2026-10-20', hashes)
    path = commitment.receipt_path(tmp_path, 'forward', '2026-10-20')
    atomic_json(path, dict(result, channel='wrong'))
    assert commitment.published(tmp_path, 'forward', '2026-10-20') is None
    altered = dict(result['line'], hashes=dict(hashes, opus='d'*64))
    atomic_json(path, dict(result, line=altered, line_sha256=sha256_canonical(altered)))
    with pytest.raises(ReplayError, match='LEDGER_MISMATCH'):
        commitment.published(tmp_path, 'forward', '2026-10-20')


def test_failed_push_or_discord_never_creates_a_publication_receipt(ledger, tmp_path):
    original = ledger.runner
    events = []
    def runner(args, **kwargs):
        events.append(args)
        if args[0] == 'git' and 'push' in args:
            return subprocess.CompletedProcess(args, 1, '', '')
        return original(args, **kwargs)
    ledger.runner = runner
    with pytest.raises(ReplayError, match='LEDGER_GIT_FAILED'):
        ledger.publish('forward', '2026-10-20', {'opus': 'a'*64})
    assert not any(a[0] == str(ledger.notify) for a in events)
    assert commitment.published(tmp_path, 'forward', '2026-10-20') is None
    def bad_route(args, **kwargs):
        if args[0] == str(ledger.notify):
            return subprocess.CompletedProcess(args, 0, 'telegram', '')
        return original(args, **kwargs)
    ledger.runner = bad_route
    with pytest.raises(ReplayError, match='DISCORD_UNCONFIRMED'):
        ledger.publish('forward', '2026-10-20', {'opus': 'a'*64})
    assert commitment.published(tmp_path, 'forward', '2026-10-20') is None
    ledger.runner = original
    result = ledger.publish('forward', '2026-10-20', {'opus': 'a'*64})
    assert result['state'] == 'PUBLISHED'
    assert len((ledger.worktree / commitment.LEDGER_FILE).read_text().splitlines()) == 1


def test_push_failure_explicitly_names_remote_branch_and_alerts(ledger, monkeypatch):
    original, events = ledger.runner, []
    monkeypatch.setenv('REPLAY_NOTIFY', str(ledger.notify))
    def runner(args, **kwargs):
        events.append(args)
        if args[0] == 'git' and 'push' in args:
            assert args[-2:] == ['origin', 'ledger/forward-duel']
            return subprocess.CompletedProcess(args, 1, '', '')
        return original(args, **kwargs)
    ledger.runner = runner
    with pytest.raises(ReplayError, match='LEDGER_GIT_FAILED'):
        ledger.publish('forward', '2026-10-20', {'opus':'a'*64})
    assert [str(ledger.notify), 'alertes'] in events


def test_replay_start_marker_is_published_before_dispatch_and_partial_responses_are_committed(ledger, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, 'ready_test', lambda _: 'a'*64)
    monkeypatch.setattr(ex, 'seal_payload', lambda _: {'seal': 'b'*64})
    monkeypatch.setattr(ex, 'head', lambda: 'c'*40)
    freeze = tmp_path/'freeze.json'
    freeze.write_text('{}')
    monkeypatch.setattr(ex, 'FREEZE', freeze)
    with pytest.raises(ReplayError, match='TEST_START_NOT_PUBLISHED'):
        ex.claim_test(tmp_path, 'opus')
    ex.claim_test(tmp_path, 'opus', ledger)
    marker = json.loads((tmp_path/spec.STATE_DIR/'test_started_opus.json').read_text())
    receipt = commitment.published(tmp_path, 'replay-test-start', 'opus')
    assert receipt['line']['hashes'] == {'test_start_marker': sha256_canonical(marker)}
    transcripts = tmp_path/spec.STATE_DIR/'transcripts'
    transcripts.mkdir()
    raw = {'state': 'USAGE_LIMIT', 'raw': 'private synthetic response', 'err': '', 'returncode': 1}
    atomic_json(transcripts/('d'*64 + '.json'), raw)
    published = ex.publish_pending(tmp_path, ledger, '2026-10-20')
    expected = sha256_canonical({k: raw[k] for k in ('raw', 'err', 'returncode')})
    assert published['line']['hashes'] == {'response:' + 'd'*64: expected}
    assert 'private synthetic response' not in ledger.log.read_text()


def test_push_timeout_is_blocked_and_alerted(ledger,monkeypatch):
    original=ledger.runner;events=[]
    monkeypatch.setenv('REPLAY_NOTIFY',str(ledger.notify))
    def runner(args,**kwargs):
        events.append(args)
        if args[0]=='git' and 'push' in args:raise subprocess.TimeoutExpired(args,120)
        return original(args,**kwargs)
    ledger.runner=runner
    with pytest.raises(ReplayError,match='LEDGER_GIT_FAILED'):
        ledger.publish('forward','2026-10-20',{'opus':'b'*64})
    assert [str(ledger.notify),'alertes'] in events
