"""Offline initialization, local inbox decisions/fills and independent reconciliation."""
import argparse
from pathlib import Path
import os
import subprocess

from .contracts import Blocked, canonical, strict_json
from .ledger import Ledger
from .models import Claude, SyntheticClaude
from .reporting import reconcile, summary, write_report
from .service import Service


def code_sha():
    repo = Path(__file__).resolve().parents[3]
    return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'activate', 'decide', 'watch', 'reconcile', 'status'))
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--inbox', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--activation-evidence', type=Path)
    args = parser.parse_args(argv)
    ledger = None
    try:
        if args.action == 'init':
            ledger = Ledger.initialize(args.runtime, 'SYNTHETIC' if args.synthetic else 'SHADOW_FORWARD', code_sha())
            print(canonical({'status': 'INITIALIZED', 'binding_hash': ledger.verify(), 'broker_orders': 0}))
            return 0
        ledger = Ledger(args.runtime, read_only=args.action == 'status')
        if args.synthetic != (ledger.binding['mode'] == 'SYNTHETIC'):
            raise Blocked('MODE_MISMATCH')
        if args.action == 'status':
            print(canonical(summary(ledger)))
            return 0
        if ledger.binding['mode'] == 'SHADOW_FORWARD' and code_sha() != ledger.binding['code_sha']:
            raise Blocked('CODE_SHA_MISMATCH')
        if args.action == 'activate':
            from .deploy import check_activation
            from .contracts import digest, utc
            if args.synthetic or not args.authorization or not args.activation_evidence:
                raise Blocked('DEPLOY_EVIDENCE_REQUIRED')
            with ledger.owned():
                if ledger.records('activation'):
                    raise Blocked('ALREADY_ACTIVATED')
                proof = check_activation(ledger, args.authorization, args.activation_evidence, utc())
                ledger.append([('activation', {'proof_hash': digest(proof), 'code_sha': code_sha()})], utc())
            result = {'status': 'ACTIVATED_SHADOW_ONLY', 'broker_orders': 0}
        elif args.action == 'reconcile':
            result = reconcile(ledger)
            result['report_hash'] = write_report(args.runtime / 'reports', result)
        else:
            if not args.inbox or not args.inbox.is_file() or args.inbox.stat().st_size > 16 * 1024 * 1024:
                raise Blocked('LOCAL_INPUT_REQUIRED')
            if args.synthetic:
                runner = SyntheticClaude()
            elif not args.authorization:
                raise Blocked('V2_AUTHORIZATION_REQUIRED')
            else:
                runner = Claude(ledger, args.authorization)
                runner.grant()  # No input/model use without a current v2 grant.
            service = Service(ledger, runner)
            data = strict_json(args.inbox.read_text())
            result = service.decide(data) if args.action == 'decide' else service.watch(data)
        print(canonical(result))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (Blocked, OSError, KeyError) as error:
        if str(error) in ('MARKET_CLOSED', 'OUTSIDE_DECISION_SLOT'):
            print(canonical({'status': 'SKIPPED', 'reason': str(error)}))
            return 0
        print(canonical({'status': 'BLOCKED', 'reason': str(error) if isinstance(error, Blocked) else 'LOCAL_RUNTIME_UNAVAILABLE'}))
        return 2
    finally:
        if ledger:
            ledger.close()


if __name__ == '__main__':
    raise SystemExit(main())
