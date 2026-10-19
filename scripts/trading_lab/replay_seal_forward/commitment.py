"""External commitment of hashes before outcomes exist: a hash-only Git ledger and Discord.

Each publication appends one JSON line (kind, key, hashes, time, protocol hash: no data, no
secrets) to `forward-duel-ledger.jsonl` on the dedicated branch `ledger/forward-duel`, pushes it
fast-forward, then posts the same hashes to #journal-des-ordres through the operator notifier.
A durable private receipt records both. Scorers refuse any decision whose hash has no receipt
published before the deadline (forward: the session open; replay: before scoring).
"""
import json
import os
from pathlib import Path
import subprocess

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json
from . import spec

BRANCH = 'ledger/forward-duel'
LEDGER_FILE = 'forward-duel-ledger.jsonl'
CHANNEL = 'ordres'
HEX = set('0123456789abcdef')


def receipts_dir(root):
    return Path(root) / 'commitments'


def receipt_path(root, kind, key):
    if not kind.replace('-', '').isalnum() or not key.replace('-', '').replace(':', '').isalnum():
        raise ReplayError('COMMITMENT_KEY_INVALID')
    return receipts_dir(root) / kind / (key.replace(':', '_') + '.json')


def _hash_only(hashes):
    for name, value in hashes.items():
        ok = isinstance(value, str) and (len(value) == 64 and set(value) <= HEX or value.startswith('FAILED:'))
        if not ok or len(name) > 120:
            raise ReplayError('COMMITMENT_NOT_HASH_ONLY', name)


class Ledger:
    def __init__(self, root, worktree, notify, *, runner=subprocess.run, clock=now):
        self.root, self.worktree, self.notify = Path(root), Path(worktree), Path(notify)
        self.runner, self.clock = runner, clock

    def _git(self, *args):
        result = self.runner(['git', '-C', str(self.worktree), *args], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ReplayError('LEDGER_GIT_FAILED', args[0])
        return (result.stdout or '').strip()

    def publish(self, kind, key, hashes, *, summary=None):
        """Idempotent: an already published receipt with the same hashes is returned unchanged."""
        _hash_only(hashes)
        path = receipt_path(self.root, kind, key)
        line = {'kind': kind, 'key': key, 'hashes': dict(sorted(hashes.items())), 'protocol_spec_hash': spec.SPEC_HASH}
        old = json.loads(path.read_text()) if path.is_file() else None
        if old:
            if old['line']['hashes'] != line['hashes']:
                raise ReplayError('COMMITMENT_CONFLICT', kind + ':' + key)
            if old['state'] == 'PUBLISHED':
                return old
        if self._git('rev-parse', '--abbrev-ref', 'HEAD') != BRANCH:
            raise ReplayError('LEDGER_WRONG_BRANCH')
        if old:
            # Resume an interrupted publication with the very same line: never a second entry.
            line = old['line']
        else:
            line['committed_at'] = iso(self.clock())
            atomic_json(path, {'state': 'INTENT', 'line': line})
        ledger = self.worktree / LEDGER_FILE
        text = ledger.read_text() if ledger.is_file() else ''
        entry = json.dumps(line, sort_keys=True, separators=(',', ':'))
        if entry not in text.splitlines():
            with ledger.open('a') as stream:
                stream.write(entry + '\n')
        if self._git('status', '--porcelain', '--', LEDGER_FILE):
            self._git('add', LEDGER_FILE)
            self._git('commit', '-q', '-m', f'ledger: {kind} {key}', '-m', 'Hash-only commitment; no data.')
        commit = self._git('rev-parse', 'HEAD')
        self._git('push', '-q', 'origin', BRANCH)
        message = '\n'.join([f'Engagement HyprL {kind} {key} (hash uniquement)'] +
                            [f'{name}: {value}' for name, value in sorted(hashes.items())][:12] +
                            ([summary] if summary else []) +
                            [f'ledger {BRANCH} @ {commit[:12]}, {len(hashes)} hash(es), ligne sha256 {sha256_canonical(line)[:16]}'])
        result = self.runner([str(self.notify), CHANNEL], input=message, capture_output=True, text=True, timeout=60)
        if result.returncode or (result.stdout or '').strip() != 'discord':
            raise ReplayError('COMMITMENT_DISCORD_UNCONFIRMED')
        receipt = {'state': 'PUBLISHED', 'line': line, 'line_sha256': sha256_canonical(line), 'ledger_commit': commit,
                   'route': 'discord', 'channel': CHANNEL, 'published_at': iso(self.clock())}
        atomic_json(path, receipt)
        return receipt


def published(root, kind, key):
    path = receipt_path(root, kind, key)
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get('state') != 'PUBLISHED' or value.get('line_sha256') != sha256_canonical(value['line']):
        return None
    return value


def require(root, kind, key, hashes, deadline):
    """The scorer gate: every given hash was published, before the deadline."""
    receipt = published(root, kind, key)
    if receipt is None:
        raise ReplayError('DECISION_HASH_NOT_PUBLISHED', kind + ':' + key)
    if not instant(receipt['published_at']) < deadline:
        raise ReplayError('DECISION_HASH_PUBLISHED_TOO_LATE', kind + ':' + key)
    for name, value in hashes.items():
        if receipt['line']['hashes'].get(name) != value:
            raise ReplayError('DECISION_HASH_NOT_PUBLISHED', kind + ':' + key + ':' + name)
    return receipt


def published_hashes(root, kind, before):
    """All hashes of one kind published strictly before `before`, for many-record replay checks."""
    out = {}
    directory = receipts_dir(root) / kind
    for file in sorted(directory.glob('*.json')) if directory.is_dir() else []:
        value = json.loads(file.read_text())
        if (value.get('state') == 'PUBLISHED' and value.get('line_sha256') == sha256_canonical(value['line'])
                and instant(value['published_at']) < before):
            out.update(value['line']['hashes'])
    return out


def from_environment(root, *, runner=subprocess.run, clock=now):
    worktree, notify = os.environ.get('REPLAY_LEDGER_WORKTREE'), os.environ.get('REPLAY_NOTIFY')
    if not worktree or not notify:
        raise ReplayError('COMMITMENT_NOT_CONFIGURED')
    return Ledger(root, worktree, notify, runner=runner, clock=clock)
