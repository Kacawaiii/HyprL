"""Causal local input boundary. No fetching, broker imports or key loading."""
from datetime import timedelta
import hashlib
from functools import lru_cache
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars

from .contracts import Blocked, QUOTES, SNAPSHOT, digest, instant, validate


@lru_cache(maxsize=64)
def sessions(start, end):
    return calendars.get_calendar('NYSE').schedule(start_date=start, end_date=end)


def session(at):
    day = at.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    schedule = sessions(day, day)
    if schedule.empty:
        raise Blocked('MARKET_CLOSED')
    row = schedule.iloc[0]
    return day, row.market_open.to_pydatetime(), row.market_close.to_pydatetime()


def previous_session(at):
    schedule = sessions((at - timedelta(days=12)).date().isoformat(), at.date().isoformat())
    rows = schedule[schedule.market_close < at]
    return rows.index[-1].date().isoformat()


def holding_days(opened, at):
    return max(0, len(sessions(opened[:10], at.date().isoformat())) - 1)


def fresh(value, at, seconds):
    age = (at - instant(value)).total_seconds()
    return 0 <= age <= seconds


def quote_map(value, at, mode, max_age=60):
    if value['mode'] != mode or not fresh(value['at'], at, max_age):
        raise Blocked('QUOTE_SNAPSHOT_STALE')
    rows = {}
    for q in value['quotes']:
        if q['symbol'] in rows:
            raise Blocked('DUPLICATE_SYMBOL')
        if (q['bid'] > q['ask'] or not fresh(q['at'], at, max_age)
                or not fresh(q['received_at'], at, max_age)
                or instant(q['at']) > instant(q['received_at'])
                or instant(q['received_at']) > instant(value['at'])
                or (mode == 'SYNTHETIC') != (q['source'] == 'synthetic')):
            raise Blocked('QUOTE_INVALID')
        rows[q['symbol']] = q
    return rows


def validate_quotes(value, at, mode):
    validate(value, QUOTES)
    return quote_map(value, at, mode)


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


def primary(source, instrument):
    try:
        url = urlsplit(source['url'])
        port = url.port
    except ValueError:
        return False
    if url.scheme != 'https' or url.username or url.password or port not in (None, 443):
        return False
    if source['kind'] == '8-K':
        return url.hostname == 'www.sec.gov' and url.path.startswith('/Archives/edgar/data/')
    return (source['kind'] == 'earnings' and url.hostname in instrument['primary_domains']
            and url.hostname not in ('', None))


def classify(output, snapshot, at):
    from .contracts import NEWS
    validate(output, NEWS)
    sources = {s['id']: s for s in snapshot['sources']}
    instruments = {r['symbol']: r for r in snapshot['instruments']}
    out, seen = [], set()
    for row in output['events']:
        source = sources.get(row['source_id'])
        if (not source or row['source_id'] in seen or row['symbol'] != source['symbol']
                or row['kind'] != source['kind'] or row['evidence'] not in source['text']):
            raise Blocked('NEWS_EVIDENCE_INVALID')
        seen.add(row['source_id'])
        if (row['code'] == 'beat_and_raise' and row['sentiment'] != 'positive'
                or row['code'] in ('guidance_cut', 'earnings_miss', 'financing_distress', 'fraud_regulatory')
                and row['sentiment'] != 'negative'):
            raise Blocked('NEWS_CLASSIFICATION_CONFLICT')
        instrument = instruments[row['symbol']]
        liquid_small_mid = 300e6 <= instrument['market_cap'] <= 20e9 and instrument['dollar_volume'] > 10e6
        out.append({**row, 'primary': primary(source, instrument), 'liquid_small_mid': liquid_small_mid,
                    'timely': fresh(source['published_at'], at, 3 * 86400), 'published_at': source['published_at'], 'source_hash': source['sha256']})
    return out


def event_block(snapshot, symbol, at, days=1):
    horizon = at + timedelta(days=days)
    # Conservative full-day blackout for scheduled macro releases; earnings through holding horizon.
    return any((event['kind'] != 'EARNINGS' and instant(event['at']).date() == at.date()
                or event['kind'] == 'EARNINGS' and event['symbol'] == symbol
                and at - timedelta(hours=1) <= instant(event['at']) <= horizon)
               for event in snapshot['events'])
