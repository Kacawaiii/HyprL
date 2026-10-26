"""Post-close forward decision disclosures, outside frozen research execution.

Default is a read-only local audit. --publish is for the authorized ledger
operator: it pushes only ledger/forward-duel and notifies #brouillons.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.public_agent_names import PUBLIC_AGENT_NAMES
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from scripts.trading_lab.replay_seal_forward import commitment, experiment as ex, forward, spec
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical

REPOSITORY = 'https://github.com/Kacawaiii/HyprL'
REMOTE = 'refs/remotes/origin/' + commitment.BRANCH
SESSION = re.compile(r'\d{4}-\d{2}-\d{2}(?:-intraday)?\Z')
FAMILIES = ('opus', 'gpt')
PRIVATE_CONTENT = re.compile(r'/home/|/srv/|/tmp/|[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|'
                             r'-----BEGIN .*PRIVATE KEY-----|\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_\-]{16,}')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Publisher:
    def __init__(self, ledger):
        if tuple(PUBLIC_AGENT_NAMES.get(f) for f in FAMILIES) != ('HypR-A1', 'HypR-E1'):
            raise ReplayError('REVEAL_PUBLIC_NAMES_INVALID')
        self.ledger = ledger
        self.git = ledger._git

    def sessions(self):
        keys = {p.stem for p in (self.ledger.root / 'commitments/forward').glob('*.json')}
        for path in (self.ledger.root / forward.STATE_DIR).glob('????-??-??'):
            if path.is_dir():
                keys.add(path.name + ('-intraday' if (path / 'decision-clock.json').exists() else ''))
        if any(SESSION.fullmatch(key) is None for key in keys):
            raise ReplayError('REVEAL_SESSION_INVALID')
        return sorted(keys)

    def plan(self, key):
        """Return bytes only after the close and a verified pre-open receipt."""
        if SESSION.fullmatch(key) is None:
            raise ReplayError('REVEAL_SESSION_INVALID')
        day = key[:10]
        sessions = forward.md.sessions(day, day)
        if len(sessions) != 1:
            raise ReplayError('REVEAL_NOT_EXCHANGE_SESSION')
        session = sessions[0]
        after = session.close_at + timedelta(minutes=15)
        if self.ledger.clock() < after:
            return {'session': key, 'status': 'WITHHELD', 'reveal_after': iso(after)}, {}
        path = commitment.receipt_path(self.ledger.root, 'forward', key)
        raw = json.loads(path.read_text()) if path.is_file() else {}
        receipt = commitment.published(self.ledger.root, 'forward', key,
                                       protocol_hash=raw.get('line', {}).get('protocol_spec_hash'))
        state, reason = 'READY', None
        if receipt is None:
            state, reason = 'UNPUBLISHED', 'NO_CONFIRMED_PUBLICATION'
        elif (instant(receipt['published_at']) >= session.open_at or
              instant(receipt['line']['committed_at']) >= session.open_at):
            state, reason = 'UNPUBLISHED', 'HASH_NOT_PUBLISHED_BEFORE_US_OPEN'
        elif instant(self.git('show', '-s', '--format=%cI', receipt['ledger_commit'])) >= session.open_at:
            state, reason = 'UNPUBLISHED', 'LEDGER_COMMIT_NOT_BEFORE_US_OPEN'
        elif not {'snapshot', *FAMILIES} <= receipt['line']['hashes'].keys():
            state, reason = 'UNPUBLISHED', 'INCOMPLETE_SESSION_COMMITMENT'
        rows, files = {}, {}
        prefix = 'reveals/' + key + '/'
        for family in FAMILIES:
            label = PUBLIC_AGENT_NAMES[family]
            rows[label] = {'status': state if state == 'UNPUBLISHED' else 'NO_DECISION'}
            if state == 'UNPUBLISHED':
                continue
            expected = receipt['line']['hashes'].get(family)
            if expected is None:
                rows[label] = {'status': 'UNPUBLISHED'}
                continue
            source = forward.directory(self.ledger.root, day) / (family + '.json')
            # The intraday run has its own receipt but shares the historical
            # directory with the failed regular run. Never bind one to the other.
            if key == day and (source.parent / 'decision-clock.json').exists():
                continue
            if not source.is_file():
                failure = source.with_name(family + '-failure.json')
                if not failure.is_file() or sha256_canonical(json.loads(failure.read_text())) != expected:
                    raise ReplayError('REVEAL_DECISION_MISSING')
                rows[label]['committed_sha256'] = expected
                continue
            record = json.loads(source.read_text())
            saved = record.pop('decision_hash')
            # This is the exact function used by sha256_canonical at decision
            # creation. No pretty printing, aliases, newlines, or new envelope.
            content = canonical_bytes(record)
            if (digest(content) != expected or saved != expected or record.get('family') != family or
                    record.get('provenance') != forward.PROVENANCE or
                    record.get('snapshot_hash') != receipt['line']['hashes'].get('snapshot')):
                raise ReplayError('REVEAL_DECISION_HASH_MISMATCH')
            # Fail closed rather than redact an immutable preimage. The operator
            # must review any such record under the repository disclosure rule.
            if PRIVATE_CONTENT.search(content.decode()):
                raise ReplayError('REVEAL_PRIVATE_CONTENT')
            files[prefix + label + '.json'] = content
            rows[label] = {'status': 'REVEALED', 'sha256': expected, 'bytes': len(content)}
        commit = receipt['ledger_commit'] if receipt else None
        result = {'session': key, 'status': state, 'agents': rows, 'ledger_commit': commit}
        if reason:
            result['reason'] = reason
        disclosure = state if state == 'UNPUBLISHED' else ('REVEALED' if len(files) == 2 else 'PARTIAL' if files else 'NO_DECISION')
        result['disclosure'] = disclosure
        lines = ['# Révélations — ' + key, '', 'État : ' + disclosure + ((' — ' + reason) if reason else '') + '.', '',
                 'Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.', '']
        if commit:
            link = REPOSITORY + '/blob/' + commit + '/' + commitment.LEDGER_FILE
            lines += ['Engagement antérieur : [registre à ce commit](' + link + ').', '',
                      'Publication confirmée : ' + receipt['published_at'] + '.', '']
        lines += ['| Agent | État | SHA-256 |', '| --- | --- | --- |']
        for label, row in rows.items():
            lines.append('| ' + label + ' | ' + row['status'] + ' | ' + row.get('sha256', row.get('committed_sha256', '—')) + ' |')
        lines += ['', 'Vérification des fichiers REVEALED, après téléchargement brut :', '', '```sh']
        for label, row in rows.items():
            if row['status'] == 'REVEALED':
                lines.append('sha256sum ' + label + '.json')
        lines += ['```', '', "Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).",
                  'Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.',
                  'Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.',
                  'UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.',
                  'NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.', '']
        files[prefix + 'README.md'] = '\n'.join(lines).encode()
        return result, files

    def run(self, *, publish=False):
        if publish:
            directory = commitment.receipts_dir(self.ledger.root)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (directory / '.publisher.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.git('fetch', '-q', 'origin', commitment.BRANCH)
                return self._run(publish=True)
        return self._run(publish=False)

    def _run(self, *, publish):
        results = {}
        for key in self.sessions():
            try:
                result, files = self.plan(key)
                if publish and files:
                    result = self.publish(result, files)
            except (ReplayError, OSError, ValueError, KeyError, TypeError) as error:
                result = {'session': key, 'status': 'BLOCKED', 'error': error.code if isinstance(error, ReplayError)
                          else 'INVALID_OR_MISSING_INPUT'}
            results[key] = result
        return {'status': 'BLOCKED' if any(r['status'] == 'BLOCKED' for r in results.values()) else 'AUDITED' if not publish else 'COMPLETE',
                'sessions': results}

    def publish(self, result, files):
        key = result['session']
        if self.git('rev-parse', '--abbrev-ref', 'HEAD') != commitment.BRANCH:
            raise ReplayError('LEDGER_WRONG_BRANCH')
        # The usual push rejects divergence; never reset, merge, or force push.
        self.git('merge-base', '--is-ancestor', REMOTE, 'HEAD')
        if set(self.git('diff', '--name-only', REMOTE, 'HEAD').splitlines()) - files.keys():
            raise ReplayError('LEDGER_UNRELATED_COMMITS')
        changes = self.git('status', '--porcelain', '--untracked-files=all').splitlines()
        if any(line[3:] not in files for line in changes):
            raise ReplayError('LEDGER_UNRELATED_CHANGES')
        marker = self.ledger.root / 'reveals' / (key + '.json')
        manifest = {name: digest(raw) for name, raw in files.items()}
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        if previous and previous['files'] != manifest:
            raise ReplayError('REVEAL_INTENT_CONFLICT')
        for name, content in files.items():
            target = self.ledger.worktree / name
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise ReplayError('REVEAL_SYMLINK')
            if target.exists() and target.read_bytes() != content:
                raise ReplayError('REVEAL_EXISTING_BYTES_DIFFER')
        if not previous:
            atomic_json(marker, {'state': 'INTENT', 'files': manifest})
        # Check the clock again immediately before writing any disclosed bytes.
        if self.ledger.clock() < forward.md.sessions(key[:10], key[:10])[0].close_at + timedelta(minutes=15):
            raise ReplayError('REVEAL_BEFORE_CLOSE')
        for name, content in files.items():
            target = self.ledger.worktree / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
        self.git('add', '--', *files)
        if self.git('diff', '--cached', '--name-only'):
            self.git('commit', '-q', '-m', 'ledger: reveal ' + key, '-m',
                     'Disclose verified original decision preimages after the US close.\n\nCo-Authored-By: Codex <codex@openai.com>')
        commit = self.git('log', '-1', '--format=%H', '--', 'reveals/' + key + '/README.md')
        for name in files:
            if self.git('rev-parse', commit + ':' + name) != self.git('hash-object', '--no-filters', '--', name):
                raise ReplayError('REVEAL_COMMIT_BYTES_DIFFER')
        if self.ledger.clock() < forward.md.sessions(key[:10], key[:10])[0].close_at + timedelta(minutes=15):
            raise ReplayError('REVEAL_BEFORE_CLOSE')
        self.git('push', '-q', 'origin', commitment.BRANCH)
        self.git('merge-base', '--is-ancestor', commit, REMOTE)
        if previous.get('state') == 'DELIVERED':
            return dict(result, status='DELIVERED', reveal_commit=commit)
        if previous.get('state') == 'NOTICE_INTENT':
            raise ReplayError('REVEAL_NOTICE_DELIVERY_UNKNOWN')
        notice = ('Révélations HypR-A1 / HypR-E1 — ' + key + ' (' + result['disclosure'] + ') : ' +
                  REPOSITORY + '/commit/' + commit + '\n')
        atomic_json(marker, {'state': 'NOTICE_INTENT', 'files': manifest, 'reveal_commit': commit})
        try:
            sent = self.ledger.runner([str(self.ledger.notify), 'brouillons'], input=notice,
                                      capture_output=True, text=True, timeout=40)
        except (OSError, subprocess.TimeoutExpired):
            raise ReplayError('REVEAL_NOTICE_DELIVERY_UNKNOWN') from None
        if sent.returncode or (sent.stdout or '').strip() != 'discord':
            raise ReplayError('REVEAL_NOTICE_DELIVERY_UNKNOWN')
        atomic_json(marker, {'state': 'DELIVERED', 'files': manifest, 'reveal_commit': commit,
                             'delivered_at': iso(self.ledger.clock())})
        return dict(result, status='DELIVERED', reveal_commit=commit)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    try:
        root = Path(args.data_root).expanduser().resolve()
        if root == spec.ROOT or spec.ROOT in root.parents:
            raise ReplayError('DATA_ROOT_MUST_BE_OUTSIDE_GIT')
        ex.committed()
        if ex.head() != args.expected_commit:
            raise ReplayError('INSTALLED_SHA_MISMATCH')
        ledger = commitment.from_environment(root)
        if args.publish:
            with owned(root):
                result = Publisher(ledger).run(publish=True)
                atomic_json(root / 'reveals/status.json', result)
        else:
            result = Publisher(ledger).run()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2 if result['status'] == 'BLOCKED' else 0
    except (ReplayError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'BLOCKED', 'error': error.code if isinstance(error, ReplayError)
                          else 'INVALID_OR_MISSING_INPUT'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
