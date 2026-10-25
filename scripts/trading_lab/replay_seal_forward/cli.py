"""Research CLI: no HTTP source client is ever enabled."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.trading_lab.replay_bench.grant import Grant, ReplayError, Transport, instant, now
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import audit, experiment as ex, forward, spec
from .models import Models, DuelBudget, api_billing


def progress_report(root, result):
    target = os.environ.get('REPLAY_SEAL_FORWARD_REPORT')
    if not target:
        return
    target = Path(target).expanduser().resolve()
    if target == spec.ROOT or spec.ROOT in target.parents:
        raise ReplayError('REPORT_MUST_BE_OUTSIDE_GIT')
    done = False
    try:
        ex.check_freeze(root)
        frozen = json.loads(ex.FREEZE.read_text())
        from scripts.trading_lab.replay_bench import run as parent
        for path in sorted((root / forward.STATE_DIR).glob('????-??-??/snapshot.json.gz')):
            sealed = parent.read_json(path)
            check = dict(sealed)
            if (check.pop('snapshot_hash') != sha256_canonical(check) or
                    sealed['provenance'] != forward.PROVENANCE or path.parent.name < spec.spec()['forward']['first_session']):
                continue
            valid = True
            for family in ex.FAMILIES:
                file = path.parent / (family + '.json')
                if not file.is_file():
                    valid = False
                    break
                value = json.loads(file.read_text())
                check = dict(value)
                if (check.pop('decision_hash') != sha256_canonical(check) or value['snapshot_hash'] != sealed['snapshot_hash'] or
                        value['configuration_hash'] != forward.configuration_for(sealed, frozen)['forward_config_hashes'][family] or
                        value['provenance'] != forward.PROVENANCE):
                    valid = False
                    break
                forward.check_record(root, path.parent.name, family, value, sealed, frozen)
            if valid:
                done = True
                break
    except (ReplayError, OSError, ValueError, KeyError):
        pass
    marker = '<!-- replay-seal-forward-progress -->'
    old = target.read_text() if target.is_file() else 'STATUS: BLOCKED\n\nRésumé FR — non terminé.\n'
    body = old.split(marker)[0].split('\n', 1)[-1].rstrip()
    text = ('STATUS: ' + ('DONE' if done else 'BLOCKED') + '\n' + body + '\n\n' + marker + '\n\n' +
            ('Scellement commité et premier duel forward complet vérifiés.' if done else 'Non terminé : les deux conditions de livraison ne sont pas réunies.') +
            '\n\n```json\n' + json.dumps(result, sort_keys=True, indent=2) + '\n```\n')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = target.with_suffix('.tmp')
    temp.write_text(text)
    os.chmod(temp, 0o600)
    os.replace(temp, target)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_seal_forward')
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required='REPLAY_DATA_ROOT' not in os.environ)
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'), required='REPLAY_AUTHORIZATION' not in os.environ)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'audit', 'freeze', 'build-inputs', 'refresh-inputs', 'seal-inputs', 'reparse-stops'):
        sub.add_parser(name)
    for name in ('night', 'forward', 'forward-intraday', 'forward-publish', 'forward-score', 'weekly'):
        command = sub.add_parser(name)
        command.add_argument('--expected-commit', required=True)
        if name == 'forward-score':
            command.add_argument('--as-of', type=instant)
        if name == 'weekly':
            command.add_argument('--correction-week')
            command.add_argument('--score-run')
    dry = sub.add_parser('forward-dry-run')
    dry.add_argument('--expected-commit', required=True)
    dry.add_argument('--session', required=True)
    sub.add_parser('score').add_argument('period', choices=('dev', 'test'))
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
        # The continuous replay owns its coordinator, leaving the independently
        # budgeted forward and collection jobs free to run on their own schedule.
        with owned(root / spec.STATE_DIR / 'worker' if args.command == 'night' else root):
            if hasattr(args, 'expected_commit'):
                ex.committed()
                if ex.head() != args.expected_commit:
                    raise ReplayError('INSTALLED_SHA_MISMATCH')
            if args.command == 'status':
                result = ex.status(root, budget)
            elif args.command == 'audit':
                result = audit.evidence(root)
            elif args.command == 'reparse-stops':
                paths = list(root.glob('*/transcripts/**/*.json')) + list((root / 'transcripts').rglob('*.json'))
                result = dict(budget.reparse_stops(paths), status='REPARSED')
            elif args.command == 'forward-dry-run':
                from .collect import Grants
                Grants(os.environ['REPLAY_COLLECTION_GRANT'], args.authorization,
                       clock=lambda: forward.decision_time(args.session))
                result = forward.dry_run(root, args.session, grant, expected_commit=args.expected_commit)
            elif args.command in ('freeze', 'seal-inputs'):
                target = ex.FREEZE if args.command == 'freeze' else ex.INPUT_SEAL
                if target.exists():
                    raise ReplayError('FREEZE_OR_SEAL_ALREADY_EXISTS')
                result = ex.frozen_payload(root) if args.command == 'freeze' else ex.seal_payload(root)
                atomic_json(target, result)
            elif args.command == 'build-inputs':
                result = ex.build_inputs(root)
            elif args.command == 'refresh-inputs':
                result = ex.refresh_inputs(root)
            elif args.command == 'forward':
                budget = forward.ForwardBudget(root)
                client = Models(grant, budget, root / forward.STATE_DIR / 'transcripts', window=forward.window,
                                billing=api_billing(root), scope='forward')
                result = forward.run(root, client, expected_commit=args.expected_commit)
            elif args.command == 'forward-intraday':
                from . import intraday
                from .collect import Grants
                budget = forward.ForwardBudget(root)
                client = Models(grant, budget, root / forward.STATE_DIR / 'transcripts',
                                billing=api_billing(root), scope='forward')
                grants = Grants(os.environ['REPLAY_COLLECTION_GRANT'], args.authorization)
                result = intraday.run(root, client, expected_commit=args.expected_commit, grants=grants,
                                      alpaca_env=os.environ['REPLAY_ALPACA_ENV'])
            elif args.command == 'forward-publish':
                from . import commitment
                result = forward.publish_pending(root, ledger=commitment.from_environment(root))
            elif args.command == 'weekly':
                notify = os.environ.get('REPLAY_NOTIFY')
                if not notify:
                    raise ReplayError('NOTIFY_EXECUTABLE_MISSING')
                result = forward.weekly(root, notify=Path(notify), correction_week=args.correction_week,
                                        score_run=args.score_run)
            else:
                transport = Transport(grant, budget, root / 'cache', offline=True)
                if args.command == 'night':
                    client = Models(grant, budget, root / spec.STATE_DIR / 'transcripts', billing=api_billing(root))
                    result = ex.night(root, budget, transport, client, expected_commit=args.expected_commit)
                elif args.command == 'score':
                    from .score import score
                    result = score(root, args.period, transport)
                else:
                    index = root / 'live-archive' / 'outcome-cache' / 'archive_index.json'
                    if not index.is_file():
                        raise ReplayError('LIVE_OUTCOME_REQUEST_INDEX_MISSING')
                    from .archive import Archive
                    archive = Archive(root, index, before=now())
                    transport.cache = archive.cache
                    transport.allowed_cache = archive.objects
                    result = forward.score(root, transport, outcome_proof=archive.proof(), as_of=args.as_of)
            atomic_json(root / spec.STATE_DIR / (args.command + '-status.json'), result)
        if args.command in ('night', 'forward', 'forward-intraday', 'forward-publish', 'forward-score', 'weekly'):
            progress_report(root, result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (ReplayError, FileNotFoundError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'BLOCKED', 'error': error.code if isinstance(error, ReplayError) else 'INVALID_OR_MISSING_INPUT'}
        if isinstance(error, ReplayError) and error.code == 'PIPELINE_DEFECT':
            from .quality import alert
            result['alert_confirmed'] = alert(error.code)
        if root is not None:
            result['failures'] = ex.failure_summary(root)
            atomic_json(root / spec.STATE_DIR / (args.command + '-status.json'), result)
            if args.command in ('night', 'forward', 'forward-intraday', 'forward-publish', 'forward-score', 'weekly'):
                progress_report(root, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
