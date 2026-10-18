"""Offline research CLI; status and audit consume no model quota."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.trading_lab.replay_bench.grant import Grant, ReplayError, Transport, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import audit, experiment as ex, spec
from .models import Models, DuelBudget


def progress_report(root, result):
    target = os.environ.get('REPLAY_DUEL_REPORT')
    if not target:
        return
    target = Path(target).expanduser().resolve()
    if target == spec.ROOT or spec.ROOT in target.parents:
        raise ReplayError('REPORT_MUST_BE_OUTSIDE_GIT')
    path = ex.result_path(root, 'test')
    value = json.loads(path.read_text()) if path.is_file() else {}
    done = (result.get('status') == 'DONE' and value.get('status') == 'SCORED' and
            value.get('spec_hash') == spec.SPEC_HASH and value.get('code_hash') == spec.code_hash() and
            set(value.get('models', {})) == set(ex.FAMILIES) and bool(value.get('paired_tests')))
    marker = '<!-- replay-duel-progress -->'
    old = target.read_text() if target.is_file() else 'STATUS: BLOCKED\n\nRésumé FR — Duel non terminé.\n'
    body = old.split(marker)[0].split('\n', 1)[-1]
    text = ('STATUS: ' + ('DONE' if done else 'BLOCKED') + '\n' + body.rstrip() + '\n\n' + marker + '\n' +
            'Mise à jour FR : ' + ('les deux tests sont notés.' if done else 'non terminé ; voir le blocage ci-dessous.') +
            '\n\nNight batch at ' + iso(now()) + '\n\n```json\n' + json.dumps(result, sort_keys=True, indent=2) + '\n```\n')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_duel')
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'), required='REPLAY_AUTHORIZATION' not in os.environ)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'audit', 'freeze', 'seal-inputs'):
        sub.add_parser(name)
    night = sub.add_parser('night')
    night.add_argument('--expected-commit', required=True)
    score = sub.add_parser('score')
    score.add_argument('period', choices=('dev', 'test'))
    args = parser.parse_args(argv)
    root = None
    try:
        root = Path(args.data_root).expanduser().resolve()
        if root == spec.ROOT or spec.ROOT in root.parents:
            root = None
            raise ReplayError('DATA_ROOT_MUST_BE_OUTSIDE_GIT')
        grant = Grant(args.authorization)
        if sha256_canonical(grant.payload) != spec.spec()['authorization']['payload_sha256']:
            raise ReplayError('GRANT_HASH_MISMATCH')
        budget = DuelBudget(root)
        # The common ownership lock also excludes the previous research timer.
        with owned(root):
            if args.command == 'status':
                result = ex.status(root, budget)
            elif args.command == 'audit':
                result = audit.evidence(root)
            elif args.command in ('freeze', 'seal-inputs'):
                target = ex.FREEZE if args.command == 'freeze' else ex.INPUT_SEAL
                if target.exists():
                    raise ReplayError('FREEZE_OR_SEAL_ALREADY_EXISTS')
                result = ex.frozen_payload(root) if args.command == 'freeze' else ex.seal_payload(root)
                atomic_json(target, result)
            else:
                transport = Transport(grant, budget, root / 'cache', offline=True)
                if args.command == 'score':
                    from .score import score
                    result = score(root, args.period, transport)
                else:
                    client = Models(grant, budget, root / spec.STATE_DIR / 'transcripts')
                    result = ex.night(root, budget, transport, client, expected_commit=args.expected_commit)
            atomic_json(root / spec.STATE_DIR / 'status.json', result)
        if args.command == 'night':
            progress_report(root, result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (ReplayError, FileNotFoundError, ValueError, KeyError) as error:
        result = {'status': 'BLOCKED', 'error': error.code if isinstance(error, ReplayError) else 'INVALID_OR_MISSING_INPUT'}
        if root is not None:
            atomic_json(root / spec.STATE_DIR / 'status.json', result)
            if args.command == 'night':
                progress_report(root, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
