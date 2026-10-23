"""Native-shape synthetic tomorrow drill; never reads production state or calls a model."""
from datetime import datetime, time, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from scripts.trading_lab.agent_v2.inputs import sessions
from .contracts import digest, instant, iso
from .ledger import Journal
from .scoring import scorecard
from .service import Service


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        return self.at


class SyntheticModels:
    synthetic = True

    def call(self, role, value):
        if role == 'extraction':
            return {'facts': []}
        return {'reviews': [{'plan_id': p['plan_id'], 'verdict': 'KEEP',
                             'reason': 'Synthetic primary evidence only.',
                             'falsifier': 'Synthetic guidance would be cut.',
                             'source_ids': [s['id'] for s in value['sources'] if s['symbol'] == p['symbol']]}
                            for p in value['plans']]}


def bundle(at):
    schedule = sessions((at - timedelta(days=500)).date().isoformat(), (at - timedelta(days=1)).date().isoformat())
    instruments, registrations = [], []
    for symbol, asset, cluster in [('SPY', 'etf', 'broad_index'), ('ACME', 'equity', 'technology'),
                                   ('HUNT', 'equity', 'industrials')]:
        history = []
        for i, (_, row) in enumerate(schedule.iloc[-210:].iterrows()):
            close = 60 * 1.002**i
            history.append({'t': iso(datetime.combine(row.market_open.date(), time(), ZoneInfo('America/New_York'))),
                            'available_at': iso(row.market_close.to_pydatetime() + timedelta(minutes=5)),
                            'c': close, 'h': close + 1, 'l': close - 1, 'v': 1_000_000})
        instrument = {'symbol': symbol, 'asset': asset, 'cluster': cluster, 'market_cap': 2e9,
                      'dollar_volume': 20e6, 'primary_domains': ['example.test'],
                      'bars': history, 'momentum_member': symbol == 'ACME'}
        instruments.append(instrument)
        registrations.append({k: instrument[k] for k in ('symbol', 'asset', 'cluster', 'market_cap',
                                                          'primary_domains', 'momentum_member')} |
                             {'cik': '0000000001'})
    reg = {'schema': 'agent-v2-quality-registry-v2', 'received_at': iso(at), 'instruments': registrations,
           'edges': {name: {'decisions': 100, 'net_expectancy_after_costs': .01, 'evidence_sha256': 'a' * 64}
                     for name in ('equity_momentum', 'hunter_runner')}}
    cal = {'schema': 'agent-v2-quality-calendar-v2', 'received_at': iso(at), 'source_hashes': ['b' * 64],
           'covered_macro': ['CPI', 'FOMC', 'NFP'],
           'coverage_start': iso(at - timedelta(days=1)), 'coverage_end': iso(at + timedelta(days=110)),
           'covered_symbols': [r['symbol'] for r in instruments], 'events': []}
    hunter = {'asof': instruments[0]['bars'][-1]['t'][:10], 'generated': iso(at),
              'candidates': [{'symbol': 'HUNT', 'setup': 'S4+F3', 'two_layer': True,
                              'rev_accel_pts': 15, 'signal_close': history[-1]['c']}]}
    snapshot = {'schema': 'agent-v2-quality-snapshot-v3', 'mode': 'SYNTHETIC', 'at': iso(at),
                'metadata_at': iso(at), 'calendar_at': iso(at), 'instruments': instruments,
                'sources': [], 'quotes': [], 'hunter': hunter, 'events': []}
    return {'snapshot': snapshot, 'registry': reg, 'calendar': cal,
            'radar': {'status': 'PASS', 'cutoff': iso(at), 'headlines': []},
            'input_hashes': {'registry': digest(reg), 'calendar': digest(cal), 'hunter': digest(hunter)}}


def primary(row, at):
    text = 'Synthetic issuer maintains guidance.'
    url = 'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/' + row['symbol'] + '.htm'
    return {'id': digest([row['symbol'], url, 'c' * 64]), 'symbol': row['symbol'], 'url': url,
            'text': text, 'published_at': iso(at - timedelta(days=1)), 'received_at': iso(at),
            'sha256': sha256(text.encode()).hexdigest(), 'raw_sha256': 'c' * 64}


def quotes(data, at, price_factor=1):
    return {'schema': 'agent-v2-quotes-v1', 'mode': 'SYNTHETIC', 'at': iso(at),
            'quotes': [{'symbol': r['symbol'], 'at': iso(at), 'received_at': iso(at),
                        'bid': r['bars'][-1]['c'] * price_factor,
                        'ask': r['bars'][-1]['c'] * price_factor + .01,
                        'bid_size': 10000, 'ask_size': 10000, 'source': 'synthetic'}
                       for r in data['snapshot']['instruments']]}


def drill(root, day, code_sha):
    at = instant(day + 'T12:45:00Z')
    schedule = sessions(day, day)
    if schedule.empty:
        return {'status': 'SKIPPED', 'reason': 'NO_NYSE_SESSION', 'day': day, 'broker_orders': 0}
    clock = Clock(at)
    journal = Journal.initialize(root, 'SYNTHETIC', code_sha, at - timedelta(seconds=1))
    try:
        data = bundle(at)
        service = Service(journal, SyntheticModels(), clock=clock)
        decision = service.decide(data, lambda row: primary(row, clock()))
        clock.at = schedule.iloc[0].market_open.to_pydatetime() + timedelta(seconds=15)
        service.watch(quotes(data, clock()))
        # A later synthetic gap crosses stops; fills must use the observed bid, with costs.
        clock.at += timedelta(minutes=1)
        service.watch(quotes(data, clock(), .7))
        card = scorecard(journal)
        if decision['code_plans'] == 0 or card['closed_baseline_candidates'] != decision['code_plans']:
            from .contracts import Blocked
            raise Blocked('DRILL_DID_NOT_EXERCISE_FILLS')
        return {'status': 'SYNTHETIC_DRILL_PASS', 'day': day, 'code_plans': decision['code_plans'],
                'closed': card['closed_baseline_candidates'], 'ledger_hash': journal.verify(), 'broker_orders': 0}
    finally:
        journal.close()
