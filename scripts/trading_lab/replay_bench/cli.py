"""Research-only replay CLI. No model or network activity for status/audit/freeze."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.trading_lab.sources.canonical import sha256_canonical
from . import audit, experiment, spec
from .grant import Budget, Grant, ReplayError, Transport
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


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_bench')
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'), required='REPLAY_AUTHORIZATION' not in os.environ)
    parser.add_argument('--alpaca-env', default=os.environ.get('REPLAY_ALPACA_ENV'))
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'audit', 'freeze'):
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
                atomic_json(root / 'v3/status.json', result)
            elif args.command == 'audit':
                result = audit.evidence(root)
                atomic_json(root / 'v3/probe_audit.json', result)
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
                    claude = Claude(grant, budget, root / 'v3/transcripts')
                    if args.command == 'night':
                        result = experiment.night(root, budget, transport, claude, expected_commit=args.expected_commit)
                    else:
                        result = experiment.run_group(root, args.period, args.group, transport, claude)
                atomic_json(root / 'v3/status.json', result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result.get('status') != 'BLOCKED' else 2
    except (ReplayError, FileNotFoundError, ValueError, KeyError) as error:
        code = error.code if isinstance(error, ReplayError) else 'INVALID_OR_MISSING_INPUT'
        result = {'status': 'BLOCKED', 'error': code}
        if 'root' in locals():
            atomic_json(root / 'v3/status.json', result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
