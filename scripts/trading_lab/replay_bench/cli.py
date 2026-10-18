"""Research-only replay CLI. No model or network activity for status/audit/freeze."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.trading_lab.sources.canonical import sha256_canonical
from . import audit, experiment, spec
from .grant import Budget, Grant, ReplayError, Transport, iso, now
from .models import Claude
from .storage import atomic_json, owned


def context(args):
    root = Path(args.data_root).expanduser().resolve()
    if root == spec.ROOT or spec.ROOT in root.parents:
        raise ReplayError('DATA_ROOT_MUST_BE_OUTSIDE_GIT')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    grant = Grant(args.authorization)
    if sha256_canonical(grant.payload) != spec.spec()['authorization']['payload_sha256']:
        raise ReplayError('GRANT_HASH_MISMATCH')
    return root, grant, Budget(root)


def progress_report(root, result):
    """Update only an explicitly configured private research report after a night batch."""
    target = os.environ.get('REPLAY_REPORT')
    if not target:
        return
    target = Path(target).expanduser().resolve()
    if target == spec.ROOT or spec.ROOT in target.parents:
        raise ReplayError('REPORT_MUST_BE_OUTSIDE_GIT')
    score_path = root / experiment.STATE_DIR / 'results/test.json'
    scored = json.loads(score_path.read_text()) if score_path.is_file() else {}
    done = (result.get('status') == 'DONE' and scored.get('status') == 'SCORED' and
            scored.get('spec_hash') == spec.SPEC_HASH and scored.get('code_hash') == experiment.run.code_hash())
    marker = '<!-- replay-opus-progress -->'
    existing = target.read_text() if target.exists() else 'STATUS: BLOCKED\n\nReplay research progress.\n'
    body = existing.split(marker)[0].split('\n', 1)[-1]
    text = ('STATUS: ' + ('DONE' if done else 'BLOCKED') + '\n' + body.rstrip() + '\n\n' + marker + '\n' +
            'Night batch update: ' + iso(now()) + '.\n\n' +
            ('Test period scored.\n' if done else 'Not done: test period not scored to completion by this batch.\n') +
            '```json\n' + json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n```\n')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(text)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_bench')
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'), required='REPLAY_AUTHORIZATION' not in os.environ)
    parser.add_argument('--alpaca-env', default=os.environ.get('REPLAY_ALPACA_ENV'))
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'audit', 'freeze', 'seal-inputs'):
        sub.add_parser(name)
    night = sub.add_parser('night')
    night.add_argument('--expected-commit', required=True)
    run = sub.add_parser('run')
    run.add_argument('period', choices=('dev', 'test'))
    run.add_argument('group', choices=tuple(experiment.GROUP_MODELS))
    score = sub.add_parser('score')
    score.add_argument('period', choices=('dev', 'test'))
    args = parser.parse_args(argv)
    try:
        root, grant, budget = context(args)
        with owned(root):
            if args.command == 'status':
                result = experiment.status(root, budget)
                atomic_json(root / experiment.STATE_DIR / 'status.json', result)
            elif args.command == 'audit':
                result = audit.evidence(root)
                atomic_json(root / experiment.STATE_DIR / 'probe_audit.json', result)
            elif args.command == 'seal-inputs':
                result = experiment.input_seal_payload(root)
                if experiment.INPUT_SEAL.exists():
                    raise ReplayError('INPUT_SEAL_ALREADY_EXISTS')
                atomic_json(experiment.INPUT_SEAL, result)
            elif args.command == 'freeze':
                result = experiment.frozen_payload(root)
                if not all(result['test_days'].values()):
                    raise ReplayError('NO_POST_CUTOFF_TEST_DATES')
                if experiment.FREEZE.exists():
                    raise ReplayError('FREEZE_ALREADY_EXISTS')
                atomic_json(experiment.FREEZE, result)
            else:
                transport = Transport(grant, budget, root / 'cache', alpaca_env=args.alpaca_env,
                                      sec_user_agent=os.environ.get('REPLAY_SEC_USER_AGENT'))
                if args.command == 'score':
                    from .score import score
                    result = score(root, args.period, transport, spec.spec())
                else:
                    claude = Claude(grant, budget, root / experiment.STATE_DIR / 'transcripts')
                    if args.command == 'night':
                        result = experiment.night(root, budget, transport, claude, expected_commit=args.expected_commit)
                    else:
                        result = experiment.run_group(root, args.period, args.group, transport, claude)
                atomic_json(root / experiment.STATE_DIR / 'status.json', result)
        if args.command == 'night':
            progress_report(root, result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result.get('status') != 'BLOCKED' else 2
    except (ReplayError, FileNotFoundError, ValueError, KeyError) as error:
        code = error.code if isinstance(error, ReplayError) else 'INVALID_OR_MISSING_INPUT'
        result = {'status': 'BLOCKED', 'error': code}
        if 'root' in locals():
            atomic_json(root / experiment.STATE_DIR / 'status.json', result)
            if args.command == 'night':
                progress_report(root, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
