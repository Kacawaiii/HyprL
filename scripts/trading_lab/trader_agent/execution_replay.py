"""Counterfactual flat-account sizing from immutable recorded decisions; no HTTP."""
from pathlib import Path
from datetime import timedelta
from tempfile import TemporaryDirectory

from .alpaca_paper import PaperExecutor, PaperLedger, PrivateQuotes, decimal
from .config import TraderError, instant, iso
from .scoring import rows


def replay(research, grant, day, *, equity='100000'):
    if decimal(equity) <= 0:
        raise TraderError('PAPER_INVALID_EQUITY')
    chain = research.verify()
    runs = [r['payload'] for r in rows(research, 'replay-summary')
            if r['payload'].get('schema') == 'trader-run-v1' and not r['payload'].get('synthetic', True)
            and r['payload'].get('status') in {'COMPLETE', 'DEGRADED'}
            and r['payload']['at'][:10] == day]
    if not runs:
        raise TraderError('PAPER_REPLAY_NO_RECORDED_DECISION')
    run = runs[-1]
    # Recording completes just after decision_at. Reproduce the first instant when
    # the full durable prediction set was available, never bypass availability.
    recorded = [instant(r['recorded_at']) for r in rows(research, 'prediction')
                if r['payload']['signal']['run_id'] == run['run_id']]
    available_at = max([instant(run['at']), *recorded]) + timedelta(microseconds=1)
    clock = lambda: available_at
    def forbidden(*args, **kwargs):
        raise TraderError('PAPER_REPLAY_NETWORK_REFUSED')
    with TemporaryDirectory(prefix='hyprl-execution-replay-') as root:
        ledger = PaperLedger(Path(root), grant, clock=clock)
        if any(v.get('asset') in grant.assets('ia_crypto') for v in run['decision']['views']):
            from .weekend import registration_target
            # Virtual registration is confined to this disposable counterfactual ledger.
            ledger.append('weekend_registration', None, **registration_target(grant))
        executor = PaperExecutor(ledger, research, clock=clock, client_factory=forbidden, quotes=PrivateQuotes())
        executor.replay_prior_registration = True
        accounts = []
        from .data import calendar_session
        for account in ('ia_actions', 'ia_crypto'):
            if account == 'ia_actions' and calendar_session(day) is None:
                continue
            lots = executor.entries(account, decimal(equity), [])
            plans = executor.plans(account, lots, 'entry')
            accounts.append({'account': account,
                'orders': [{k: v for k, v in p['order'].items() if k != 'client_order_id'} for p in plans],
                'lots': [{k: l[k] for k in ('asset', 'horizon', 'qty', 'variant', 'population', 'prediction_hash', 'size_multiplier')} for l in lots]})
        ledger.store.close()
    if research.verify() != chain:
        raise TraderError('PAPER_REPLAY_RESEARCH_CHANGED')
    return {'state': 'COMPLETE', 'counterfactual': True, 'no_order': True, 'network_requests': 0,
            'at': iso(clock()), 'run_id': run['run_id'], 'run_status': run['status'],
            'assumption': 'Flat account, fixed equity; original decision inputs only; v3 rule replay, no performance claim.',
            'equity_per_account': str(decimal(equity)), 'research_head_hash': chain['head_hash'], 'accounts': accounts}


if __name__ == '__main__':
    import argparse
    import json
    import os
    from scripts.trading_lab.research.store import ResearchStore
    from .alpaca_paper import PaperAuthorization
    from .config import Authorization
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--paper-authorization', required=True)
    parser.add_argument('--day', required=True)
    args = parser.parse_args()
    grant = PaperAuthorization.load(args.paper_authorization, Authorization.load(args.authorization))
    research = ResearchStore(args.research_root, read_only=True)
    print(json.dumps(replay(research, grant, args.day), indent=2))
