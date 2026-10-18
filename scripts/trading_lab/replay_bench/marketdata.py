"""Alpaca news, bars and SEC filings for the replay. Data may extend past T; snapshots alone decide what T sees."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from scripts.trading_lab.trader_agent.data import calendar_session, equity_calendar

from .grant import ReplayError, UTC, instant, iso

ALPACA = 'https://data.alpaca.markets'
NEW_YORK = ZoneInfo('America/New_York')
PROTECTED = {'BTC', 'ETH', 'BTCUSD', 'ETHUSD', 'BTC/USD', 'ETH/USD', 'BTC-USD', 'ETH-USD'}
PROTECTED_WINDOW = ('2026-09-01T00:00:00Z', '2026-12-01T00:00:00Z')


def guard(symbols, start, end):
    """BTC/ETH are never requested while their window overlaps the protected holdout."""
    touches = instant(start) < instant(PROTECTED_WINDOW[1]) and instant(end) > instant(PROTECTED_WINDOW[0])
    if touches and PROTECTED.intersection(symbols):
        raise ReplayError('PROTECTED_REQUEST', sorted(PROTECTED.intersection(symbols)))


def crypto_pair(asset):
    return asset.replace('-USD', '/USD')


def paginate(transport, path, params, key):
    token, pages, seen = None, [], set()
    while True:
        query = dict(params, **({'page_token': token} if token else {}))
        body, meta = transport.get(ALPACA, path, query)
        pages.append((body, meta))
        token = body.get('next_page_token')
        if not token:
            return pages
        if token in seen or len(pages) > 10000:
            raise ReplayError('PAGINATION_LOOP')
        seen.add(token)


def stock_trades(transport, symbols, at):
    """First trade in a bounded one-second fill window. Never substitute a future-minute VWAP."""
    symbols = sorted(symbols)
    start, end = iso(at), iso(at + timedelta(seconds=1))
    guard(symbols, start, end)
    out = {}
    for body, _ in paginate(transport, '/v2/stocks/trades', {
            'symbols': ','.join(symbols), 'start': start, 'end': end, 'feed': 'sip', 'limit': 10000}, 'trades'):
        for symbol, trades in (body.get('trades') or {}).items():
            for trade in trades:
                if symbol in symbols and at <= instant(trade['t']) < at + timedelta(seconds=1) and trade['p'] > 0:
                    if symbol not in out or instant(trade['t']) < instant(out[symbol]['t']):
                        out[symbol] = trade
    return out


def stock_bars(transport, symbols, start, end, timeframe='1Day'):
    guard(symbols, start, end)
    out, metas = {}, []
    for i in range(0, len(symbols), 100):
        chunk = sorted(symbols)[i:i + 100]
        for body, meta in paginate(transport, '/v2/stocks/bars', {
                'symbols': ','.join(chunk), 'timeframe': timeframe, 'start': start, 'end': end,
                'feed': 'sip', 'adjustment': 'raw', 'limit': 10000}, 'bars'):
            metas.append(meta)
            for symbol, bars in (body.get('bars') or {}).items():
                out.setdefault(symbol, []).extend(bars)
    return out, metas


def crypto_bars(transport, assets, start, end, timeframe='1Day'):
    pairs = [crypto_pair(a) for a in assets]
    guard(pairs + list(assets), start, end)
    out, metas = {}, []
    for body, meta in paginate(transport, '/v1beta3/crypto/us/bars', {
            'symbols': ','.join(sorted(pairs)), 'timeframe': timeframe, 'start': start, 'end': end, 'limit': 10000}, 'bars'):
        metas.append(meta)
        for pair, bars in (body.get('bars') or {}).items():
            out.setdefault(pair.replace('/USD', '-USD'), []).extend(bars)
    return out, metas


def news(transport, start, end, symbols=None):
    """Every news item created in [start, end), oldest first, without article bodies."""
    if symbols:
        guard(symbols, start, end)
    params = {'start': start, 'end': end, 'limit': 50, 'sort': 'asc', 'include_content': 'false'}
    if symbols:
        params['symbols'] = ','.join(sorted(symbols))
    items, metas = {}, []
    for body, meta in paginate(transport, '/v1beta1/news', params, 'news'):
        metas.append(meta)
        for item in body.get('news') or []:
            items[item['id']] = {k: item.get(k) for k in ('id', 'headline', 'summary', 'source', 'symbols', 'url',
                                                         'created_at', 'updated_at', 'author')}
    return sorted(items.values(), key=lambda n: (n['created_at'], n['id'])), metas


def session_rows(bars):
    """Alpaca daily equity bars -> trader_agent price rows keyed by the calendar's session open."""
    rows = []
    for bar in bars:
        day = bar['t'][:10]
        session = calendar_session(day)
        if session is None:
            raise ReplayError('PRICE_OFF_CALENDAR', day)
        rows.append({'bar_open_at': session.open_at, 'close_at': session.close_at, 'open': bar['o'], 'high': bar['h'],
                     'low': bar['l'], 'close': bar['c'], 'volume': bar['v']})
    return rows


def crypto_rows(bars):
    rows = []
    for bar in bars:
        opening = instant(bar['t'])
        if opening.hour or opening.minute:
            raise ReplayError('CRYPTO_BAR_NOT_UTC_DAY')
        rows.append({'bar_open_at': opening, 'close_at': opening + timedelta(days=1), 'open': bar['o'],
                     'high': bar['h'], 'low': bar['l'], 'close': bar['c'], 'volume': bar['v']})
    return rows


def sessions(first, last):
    start = instant(first + 'T00:00:00Z')
    end = instant(last + 'T23:59:59Z')
    calendar = equity_calendar(start.year)
    return [s for s in calendar.sessions_between(start, end) if first <= s.session_date <= last]


def previous_close(at):
    calendar = equity_calendar(at.year)
    prior = [s for s in calendar.sessions_between(at - timedelta(days=10), at) if s.close_at <= at]
    return prior[-1].close_at


def acceptance_upper_bound(text):
    """The 'Z' suffix of acceptanceDateTime does not prove its zone (EDGAR registry UV2). Reading the wall time as
    New York local gives the later of the two possible instants, so a filing is never admitted before T too early."""
    wall = datetime.strptime(text[:19], '%Y-%m-%dT%H:%M:%S')
    return max(wall.replace(tzinfo=UTC), wall.replace(tzinfo=NEW_YORK).astimezone(UTC))


class SecFilings:
    """8-K acceptance times from data.sec.gov submissions; one request per CIK, cached."""

    def __init__(self, transport):
        self.transport = transport
        self._tickers = None

    def cik(self, symbol):
        if self._tickers is None:
            body, _ = self.transport.get('https://www.sec.gov', '/files/company_tickers.json')
            self._tickers = {row['ticker'].upper(): int(row['cik_str']) for row in body.values()}
        return self._tickers.get(symbol.upper().replace('.', '-'))

    def eight_ks(self, symbol):
        cik = self.cik(symbol)
        if cik is None:
            return None
        body, meta = self.transport.get('https://data.sec.gov', f'/submissions/CIK{cik:010d}.json')
        if body.get('cik') != f'{cik:010d}':
            raise ReplayError('SEC_CIK_IDENTITY_MISMATCH')
        recent = body.get('filings', {}).get('recent', {})
        if len({len(recent.get(k, [])) for k in ('form', 'acceptanceDateTime', 'accessionNumber')}) != 1:
            raise ReplayError('SEC_PARALLEL_ARRAY_LENGTH_MISMATCH')
        out = []
        for form, accepted, accession in zip(recent.get('form', []), recent.get('acceptanceDateTime', []),
                                             recent.get('accessionNumber', [])):
            if form == '8-K':
                out.append({'accepted_at': iso(acceptance_upper_bound(accepted)), 'accession': accession,
                            'acceptance_text': accepted})
        return out
