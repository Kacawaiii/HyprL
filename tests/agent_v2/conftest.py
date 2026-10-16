from copy import deepcopy
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from scripts.trading_lab.agent_v2.contracts import digest, instant, iso
from scripts.trading_lab.agent_v2.inputs import sessions
from scripts.trading_lab.agent_v2.ledger import Ledger
from scripts.trading_lab.agent_v2.models import SyntheticClaude
from scripts.trading_lab.agent_v2.service import Service


@pytest.fixture
def clock():
    class Clock:
        at = instant('2026-10-16T13:40:00Z')
        def __call__(self):
            return self.at
    return Clock()


@pytest.fixture
def snapshot(clock):
    schedule = sessions('2025-11-01', '2026-10-15').iloc[-210:]
    instruments, quotes = [], []
    for symbol, asset, cluster in [('SPY','etf','broad_index'), ('ACME','equity','technology'),
                                   ('HUNT','equity','industrials'), ('NEWS','equity','health')]:
        bars = []
        for i, (_, row) in enumerate(schedule.iterrows()):
            close = 60 * 1.002 ** i
            bars.append({'t': iso(datetime.combine(row.market_open.date(), time(), ZoneInfo('America/New_York'))),
                         'available_at': iso(row.market_close.to_pydatetime() + timedelta(minutes=5)),
                         'c': close, 'h': close + 1, 'l': close - 1, 'v': 1000000})
        instruments.append({'symbol':symbol, 'asset':asset, 'cluster':cluster, 'market_cap':2e9,
                            'dollar_volume':20e6, 'primary_domains':['ir.example.test'], 'bars':bars,
                            'momentum_member': symbol == 'ACME'})
        quotes.append({'symbol':symbol,'at':iso(clock()), 'received_at':iso(clock()),'bid':bars[-1]['c'],
                       'ask':bars[-1]['c'] + .05,'bid_size':10000,'ask_size':10000,'source':'synthetic'})
    return {'schema':'agent-v2-snapshot-v1','mode':'SYNTHETIC','at':iso(clock()),
            'metadata_at':iso(clock()),'calendar_at':iso(clock()),'instruments':instruments,
            'sources':[], 'events':[], 'quotes':quotes,
            'hunter':{'asof':'2026-10-15','generated':iso(clock()),'candidates':[
                {'symbol':'HUNT','setup':'S4','two_layer':True,'rev_accel_pts':15,'signal_close':bars[-1]['c']}]}}


@pytest.fixture
def ledger(tmp_path, clock):
    ledger = Ledger.initialize(tmp_path / 'shadow', 'SYNTHETIC', 'a'*40, clock())
    yield ledger
    ledger.close()


@pytest.fixture
def service(ledger, clock):
    return Service(ledger, SyntheticClaude(), clock=clock)


def quotes(snapshot, clock, **changes):
    rows = deepcopy(snapshot['quotes'])
    for row in rows:
        row.update({'at':iso(clock()), 'received_at':iso(clock())})
        row.update(changes)
    return {'schema':'agent-v2-quotes-v1','mode':'SYNTHETIC','at':iso(clock()),'quotes':rows}
