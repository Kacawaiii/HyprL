"""Explicit SHADOW-only commands. Deployment and external writes stay operator-controlled."""
import argparse
from datetime import timedelta
from pathlib import Path
import os
import subprocess
import tempfile

from scripts.trading_lab.agent_v2.inputs import session
from .contracts import Blocked, canonical, digest, iso, utc
from .deploy import activate, write_units
from .ledger import Journal
from .models import Models
from .producer import Client, Grant, Producer, atomic_json, read_local
from .scoring import audit_fact, scorecard, weekly_text
from .service import Service, book
from .synthetic import drill


def code_sha():
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[3]),
                                    'rev-parse', 'HEAD'], text=True).strip()


def clean_checkout():
    if subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[3]),
                                'status', '--porcelain'], text=True).strip():
        raise Blocked('CLEAN_COMMITTED_CHECKOUT_REQUIRED')


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'produce', 'activate', 'cycle', 'watch', 'status',
                                         'weekly', 'drill', 'render', 'audit-fact'))
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args(argv)
    journal = None
    try:
        if args.action == 'drill':
            # Independent fresh synthetic runtime, even when installed beside a forward runtime.
            root = args.runtime / 'drills'
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory = Path(tempfile.mkdtemp(prefix='tomorrow-', dir=root)) / 'runtime'
            result = drill(directory, (utc() + timedelta(days=1)).date().isoformat(), code_sha())
            atomic_json(directory.parent / 'result.json', result)
        elif args.action == 'render':
            if not args.output or not args.config:
                raise Blocked('CONFIG_AND_OUTPUT_REQUIRED')
            result = write_units(args.output, Path(__file__).resolve().parents[3],
                                 args.runtime.resolve(), args.config.resolve())
        elif args.action == 'init':
            if not args.synthetic:
                clean_checkout()
            journal = Journal.initialize(args.runtime, 'SYNTHETIC' if args.synthetic else 'SHADOW_FORWARD', code_sha())
            result = {'status': 'INITIALIZED', 'binding_hash': journal.verify(), 'broker_orders': 0}
        else:
            journal = Journal(args.runtime, read_only=args.action == 'status')
            if args.synthetic != (journal.binding['mode'] == 'SYNTHETIC'):
                raise Blocked('MODE_MISMATCH')
            if not args.synthetic and args.action != 'status':
                clean_checkout()
            if code_sha() != journal.binding['code_sha']:
                raise Blocked('CODE_SHA_MISMATCH')
            if args.action == 'status':
                result = scorecard(journal)
            elif args.action == 'audit-fact':
                audit_fact(journal, read_local(args.evidence))
                result = {'status': 'FACT_ADJUDICATED'}
            else:
                if not args.config:
                    raise Blocked('PRIVATE_CONFIG_REQUIRED')
                config = read_local(args.config)
                grant = Grant(config['authorization'])
                grant.check()
                if args.action == 'activate':
                    result = activate(journal, grant, read_local(args.evidence))
                elif args.action == 'weekly':
                    # Explicit task authorization: only this channel and reviewed notifier.
                    if not journal.records('activation') or journal.binding['mode'] != 'SHADOW_FORWARD':
                        raise Blocked('FORWARD_ACTIVATION_REQUIRED')
                    week = utc().strftime('%G-W%V')
                    with journal.owned():
                        if any(r['body']['week'] == week for r in journal.records('weekly_reservation')):
                            raise Blocked('WEEKLY_ALREADY_ATTEMPTED')
                        journal.append([('weekly_reservation', {'week': week})], utc())
                    delivery = subprocess.run([str(Path.home() / 'bin/hyprl-notify'), 'brouillons'],
                                              input=weekly_text(journal), text=True, capture_output=True, timeout=45)
                    if delivery.returncode:
                        raise Blocked('WEEKLY_DELIVERY_FAILED')
                    result = {'status': 'WEEKLY_SENT', 'scorecard': scorecard(journal)}
                else:
                    at = utc()
                    if args.action in ('cycle', 'watch'):
                        day, opened, closed = session(at)
                        if args.action == 'cycle' and not (at.hour == 12 and 45 <= at.minute <= 47):
                            raise Blocked('OUTSIDE_DECISION_SLOT')
                        if args.action == 'watch' and not opened <= at <= closed:
                            raise Blocked('MARKET_CLOSED')
                    if args.action == 'cycle':
                        # A cycle cannot acquire data before activation or repeat a failed acquisition slot.
                        service = Service(journal, Models(journal, grant))
                        with journal.owned():
                            if any(r['body']['day'] == day for r in journal.records('cycle_acquisition')):
                                raise Blocked('SLOT_ALREADY_ATTEMPTED')
                            journal.append([('cycle_acquisition', {'day': day})], utc())
                    client = Client(journal, grant, config['credentials'], config['sec_authorization'])
                    producer = Producer(client)
                    if args.action in ('produce', 'cycle'):
                        bundle = producer.snapshot(config['registry'], config['calendar'], config['hunter'], config['radar'])
                        with journal.owned():
                            journal.append([('producer_snapshot', {'hash': digest(bundle),
                                             'input_hashes': bundle['input_hashes']})], utc())
                        atomic_json(args.runtime / 'inbox' / 'quality-snapshot.json', bundle)
                        result = {'status': 'PRODUCED', 'snapshot_hash': digest(bundle)}
                        if args.action == 'cycle':
                            result = service.decide(bundle, producer.primary)
                    else:
                        service = Service(journal, Models(journal, grant))
                        state = book(journal)
                        symbols = set(state['positions']) | {p['symbol'] for p in state['pending'].values()}
                        if symbols:
                            value = producer.quotes(symbols)
                            with journal.owned():
                                journal.append([('producer_quotes', {'hash': digest(value)})], utc())
                            atomic_json(args.runtime / 'inbox' / 'quality-quotes.json', value)
                            result = service.watch(value)
                        else:
                            result = {'status': 'NO_PENDING_OR_OPEN_CANDIDATES', 'broker_orders': 0}
        print(canonical(result))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (Blocked, OSError, KeyError, TypeError, UnicodeError, ValueError) as error:
        reason = str(error) if isinstance(error, Blocked) else 'LOCAL_INPUT_OR_RUNTIME_INVALID'
        print(canonical({'status': 'SKIPPED' if reason in ('MARKET_CLOSED', 'OUTSIDE_DECISION_SLOT') else 'BLOCKED',
                         'reason': reason, 'broker_orders': 0}))
        return 0 if reason in ('MARKET_CLOSED', 'OUTSIDE_DECISION_SLOT') else 2
    finally:
        if journal:
            journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
