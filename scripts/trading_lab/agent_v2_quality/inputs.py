"""Revision-three snapshot boundary; revision-one contracts stay frozen."""
import hashlib

from scripts.trading_lab.agent_v2.inputs import fresh, previous_session, quote_map, sessions
from .contracts import Blocked, SNAPSHOT, instant, validate


def validate_snapshot(value, at, mode):
    validate(value, SNAPSHOT)
    if value['mode'] != mode or not fresh(value['at'], at, 300):
        raise Blocked('SNAPSHOT_STALE')
    if not fresh(value['metadata_at'], at, 4 * 86400) or not fresh(value['calendar_at'], at, 18 * 3600):
        raise Blocked('METADATA_OR_CALENDAR_STALE')
    previous = previous_session(at)
    instruments = {}
    for row in value['instruments']:
        symbol = row['symbol']
        if row['market_cap'] is None and not (
                symbol == 'SPY' and row['asset'] == 'etf' and row['cluster'] == 'broad_index'):
            raise Blocked('MARKET_CAP_REQUIRED')
        if symbol in instruments:
            raise Blocked('DUPLICATE_SYMBOL')
        bars = row['bars']
        if not bars or bars[-1]['t'][:10] != previous:
            raise Blocked('BARS_STALE')
        last = None
        history = sessions(bars[0]['t'][:10], bars[-1]['t'][:10])
        closes = {str(index.date()): row.market_close.to_pydatetime() for index, row in history.iterrows()}
        for bar in bars:
            t, available = instant(bar['t']), instant(bar['available_at'])
            market_close = closes.get(t.date().isoformat())
            if (market_close is None
                    or (last is not None and t.date() <= last.date()) or not t <= available <= instant(value['at']) <= at
                    or available < market_close
                    or not bar['l'] <= bar['c'] <= bar['h']):
                raise Blocked('NONCAUSAL_OR_INVALID_BAR')
            last = t
        instruments[symbol] = row
    if 'SPY' not in instruments or len(instruments['SPY']['bars']) < 200:
        raise Blocked('REGIME_DATA_MISSING')
    source_ids = set()
    for source in value['sources']:
        if source['id'] in source_ids or source['symbol'] not in instruments:
            raise Blocked('SOURCE_ID_INVALID')
        source_ids.add(source['id'])
        if (not instant(source['published_at']) <= instant(source['received_at']) <= instant(value['at']) <= at
                or hashlib.sha256(source['text'].encode()).hexdigest() != source['sha256']):
            raise Blocked('SOURCE_NONCAUSAL_OR_DIGEST')
    for event in value['events']:
        instant(event['at'])
        if event['kind'] == 'EARNINGS' and event['symbol'] not in instruments:
            raise Blocked('EVENT_SYMBOL_INVALID')
    quote_map(value, at, mode)
    return instruments

