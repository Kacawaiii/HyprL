"""GET-only, allowlisted acquisition with durable reservations and receipt times."""
from datetime import timedelta
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts.trading_lab.agent_v2.inputs import fresh, previous_session, sessions
from .inputs import validate_snapshot
from .contracts import Blocked, canonical, digest, instant, iso, strict_json, utc
from .spec import spec


class Grant:
    def __init__(self, path, clock=utc):
        self.path, self.clock = Path(path), clock

    def check(self):
        value = strict_json(self.path.read_text())
        if (digest(value) != spec()['authorization_sha256']
                or value.get('id') != 'trader-agent-v2-shadow'
                or self.clock() >= instant(value['expires'] + 'T00:00:00Z') + timedelta(days=1)
                or self.clock() < instant('2026-10-22T00:00:00Z')):
            raise Blocked('QUALITY_AUTHORIZATION_REQUIRED')
        return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def transport(url, headers):
    try:
        with build_opener(NoRedirect()).open(Request(url, headers=headers, method='GET'), timeout=25) as response:
            raw = response.read(12_000_001)
            if len(raw) > 12_000_000:
                raise Blocked('HTTP_RESPONSE_TOO_LARGE')
            return response.status, raw
    except HTTPError as error:
        return error.code, b''
    except (OSError, URLError):
        raise Blocked('HTTP_UNAVAILABLE') from None


def group_for(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment
            or '%' in parsed.path or '..' in parsed.path or '\\' in parsed.path):
        raise Blocked('HTTP_OUTSIDE_SCOPE')
    if parsed.hostname == 'data.alpaca.markets' and parsed.path in (
            '/v2/stocks/bars', '/v2/stocks/quotes/latest'):
        return 'alpaca'
    if parsed.query:
        raise Blocked('HTTP_OUTSIDE_SCOPE')
    if parsed.hostname == 'data.sec.gov' and re.fullmatch(r'/submissions/CIK\d{10}\.json', parsed.path):
        return 'sec'
    if parsed.hostname == 'www.sec.gov' and re.fullmatch(
            r'/Archives/edgar/data/\d{1,10}/\d{18}/[A-Za-z0-9_.-]+', parsed.path):
        return 'sec'
    raise Blocked('HTTP_OUTSIDE_SCOPE')


class Client:
    def __init__(self, journal, grant, credentials, sec_authorization, *, clock=utc, send=transport, pause=time.sleep):
        self.journal, self.grant = journal, grant
        self.credentials, self.sec_authorization = credentials, sec_authorization
        self.clock, self.send, self.pause = clock, send, pause

    def headers(self, group):
        if group == 'sec':
            # Only the approved UA is reused; this grant does not borrow the radar's endpoint scope.
            value = strict_json(Path(self.sec_authorization).read_text())
            if not instant(value['granted_at']) <= self.clock() < instant(value['not_after']):
                raise Blocked('SEC_USER_AGENT_AUTHORIZATION_EXPIRED')
            ua = value['scope']['https://www.sec.gov']['user_agent']
            if not isinstance(ua, str) or not ua.strip() or any(ord(c) < 32 for c in ua):
                raise Blocked('SEC_USER_AGENT_REQUIRED')
            return {'User-Agent': ua, 'Accept': 'application/json,text/html'}
        values = {}
        for line in Path(self.credentials).read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.removeprefix('export ').split('=', 1)
                values[key.strip()] = value.strip().strip('\"\'')
        key = values.get('APCA_API_KEY_ID') or values.get('ALPACA_API_KEY')
        secret = values.get('APCA_API_SECRET_KEY') or values.get('ALPACA_SECRET_KEY')
        if not key or not secret:
            raise Blocked('DATA_CREDENTIALS_REQUIRED')
        return {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret, 'Accept': 'application/json'}

    def get(self, url):
        grant = self.grant.check()
        if self.journal.binding['mode'] != 'SHADOW_FORWARD':
            raise Blocked('NETWORK_FORBIDDEN_IN_SYNTHETIC')
        group = group_for(url)
        headers = self.headers(group)
        # Bound acquisition spacing without holding the writer lock; another owner is checked again below.
        previous = [r for r in self.journal.records('http_reservation') if r['body']['group'] == 'sec']
        if group == 'sec' and previous:
            remaining = 1 - (self.clock() - instant(previous[-1]['at'])).total_seconds()
            if remaining > 0:
                self.pause(min(1, remaining))
        now = self.clock()
        with self.journal.owned():
            rows = [r for r in self.journal.records('http_reservation')
                    if r['at'][:10] == iso(now)[:10] and r['body']['group'] == group]
            if len(rows) >= spec()['data_budgets'][group]:
                raise Blocked('HTTP_BUDGET')
            if rows and group == 'sec' and (now - instant(rows[-1]['at'])).total_seconds() < 1:
                raise Blocked('SEC_SPACING')  # <=1 req/s, stricter than grant; no retry loop
            if any(r['at'][:10] == iso(now)[:10] and r['body']['group'] == group
                   for r in self.journal.records('http_halt')):
                raise Blocked('HTTP_HALTED')
            self.journal.append([('http_reservation', {'group': group, 'url': url,
                                  'authorization_hash': digest(grant)})], now)
        status, raw = self.send(url, headers)
        received = self.clock()  # after the entire response, never the publication timestamp
        receipt = {'group': group, 'url': url, 'status': status,
                   'received_at': iso(received), 'sha256': sha256(raw).hexdigest()}
        with self.journal.owned():
            events = [('http_receipt', receipt)]
            if status in (403, 429):
                events.append(('http_halt', {'group': group, 'status': status}))
            self.journal.append(events, received)
        if status != 200:
            raise Blocked('HTTP_' + str(status))
        return raw, receipt

    def json(self, url):
        raw, receipt = self.get(url)
        return strict_json(raw.decode('utf-8')), receipt


class FilingText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag in ('p', 'div', 'br', 'tr', 'td'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def document(raw, receipt, symbol, published_at):
    parser = FilingText()
    parser.feed(raw.decode('utf-8', errors='strict'))
    text = ' '.join(''.join(parser.parts).split())
    if not text or len(text) > 300_000:
        raise Blocked('PRIMARY_DOCUMENT_EMPTY_OR_TOO_LARGE')  # never silently truncate a filing
    if instant(published_at) > instant(receipt['received_at']):
        raise Blocked('PRIMARY_DOCUMENT_NONCAUSAL')
    return {'id': digest([symbol, receipt['url'], receipt['sha256']]), 'symbol': symbol,
            'url': receipt['url'], 'published_at': published_at,
            'received_at': receipt['received_at'], 'text': text,
            'sha256': sha256(text.encode()).hexdigest(), 'raw_sha256': receipt['sha256']}


def registry(value, at):
    if (value.get('schema') != 'agent-v2-quality-registry-v2'
            or not fresh(value.get('received_at'), at, 4 * 86400)):
        raise Blocked('REGISTRY_MISSING_OR_STALE')
    rows = value.get('instruments', [])
    if not 1 <= len(rows) <= 200 or len({r['symbol'] for r in rows}) != len(rows):
        raise Blocked('REGISTRY_INVALID')
    for row in rows:
        if (not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', row['symbol'])
                or row['symbol'] in ('BTC', 'ETH', 'BTC-USD', 'ETH-USD')
                or row['asset'] not in ('equity', 'etf')
                or row['cluster'] not in spec()['risk']['clusters']
                or row['asset'] == 'equity' and not re.fullmatch(r'\d{10}', row.get('cik', ''))):
            raise Blocked('REGISTRY_INVALID')
    return value


def evaluation_end(at):
    schedule = sessions(at.date().isoformat(), (at + timedelta(days=21)).date().isoformat())
    if len(schedule) < 6:
        raise Blocked('CALENDAR_SESSION_COVERAGE_REQUIRED')
    return schedule.iloc[5].market_close.to_pydatetime()


def calendar(value, symbols, at):
    if (value.get('schema') != 'agent-v2-quality-calendar-v2'
            or not fresh(value.get('received_at'), at, 18 * 3600)
            or not value.get('source_hashes')
            or any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in value['source_hashes'])
            or not set(symbols) <= set(value.get('covered_symbols', []))
            or set(value.get('covered_macro', [])) != {'CPI', 'FOMC', 'NFP'}
            or instant(value['coverage_start']) > at
            or instant(value['coverage_end']) < evaluation_end(at)):
        raise Blocked('CALENDAR_COVERAGE_REQUIRED')
    return value


def read_local(path, maximum=16_000_000):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise Blocked('LOCAL_INPUT_TOO_LARGE')
    return strict_json(path.read_text())


def radar_headlines(value, symbols, at):
    """Only candidate-related stock headlines; never radar prices, crypto, or model opinions."""
    if (value.get('schema') != 'news-radar-v3' or value.get('live') is not True
            or value.get('status') not in ('READY', 'PARTIAL')
            or not fresh(value.get('cutoff'), at, 18 * 3600)):
        raise Blocked('RADAR_MISSING_OR_STALE')
    headlines = []
    for event in value.get('events', []):
        names = set(event.get('entities', {}).get('symbols', []))
        if not names & set(symbols) or any('/' in name or name in ('BTC', 'ETH', 'BTC-USD', 'ETH-USD') for name in names):
            continue
        for story in event.get('stories', []):
            if not story.get('published_at') or not (
                    instant(story['published_at']) <= instant(story['received_at']) <= instant(value['cutoff']) <= at):
                raise Blocked('RADAR_HEADLINE_NONCAUSAL')
            headlines.append({k: story[k] for k in ('id', 'headline', 'url', 'published_at', 'received_at')})
    return {'cutoff': value['cutoff'], 'status': value['status'], 'headlines': headlines[:100],
            'upstream_sha256': digest(value)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp-' + str(os.getpid()))
    try:
        with temporary.open('x') as stream:
            os.chmod(temporary, 0o600)
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Producer:
    def __init__(self, client, *, clock=utc):
        self.client, self.clock = client, clock

    def snapshot(self, registry_path, calendar_path, hunter_path, radar_path):
        at = self.clock()
        reg = registry(read_local(registry_path), at)
        cal = calendar(read_local(calendar_path), [r['symbol'] for r in reg['instruments']], at)
        hunter = read_local(hunter_path)
        # Keep the observed hunter-live envelope and only the adapter's consumed fields.
        hunter = {k: hunter[k] for k in ('asof', 'generated', 'candidates')}
        admitted = {r['symbol'] for r in reg['instruments']}
        hunter['candidates'] = [r for r in hunter['candidates'] if r.get('symbol') in admitted]
        if not fresh(hunter['generated'], at, 4 * 86400):
            raise Blocked('HUNTER_STALE')
        radar = radar_headlines(read_local(radar_path), [r['symbol'] for r in reg['instruments']], at)
        previous = previous_session(at)
        schedule = sessions(previous, previous)
        end = schedule.iloc[0].market_close.to_pydatetime() + timedelta(seconds=1)
        params = {'symbols': ','.join(r['symbol'] for r in reg['instruments']), 'timeframe': '1Day',
                  'start': iso(at - timedelta(days=520)), 'end': iso(end), 'feed': 'iex',
                  'adjustment': 'raw', 'limit': 10000, 'sort': 'asc'}
        bars, seen = {}, set()
        for _ in range(20):
            payload, receipt = self.client.json('https://data.alpaca.markets/v2/stocks/bars?' + urlencode(params))
            for symbol, rows in payload['bars'].items():
                if symbol not in params['symbols'].split(','):
                    raise Blocked('BARS_SYMBOL_INVALID')
                bars.setdefault(symbol, []).extend({k: b[k] for k in ('t', 'c', 'h', 'l', 'v')} |
                                                   {'available_at': receipt['received_at']} for b in rows)
            token = payload.get('next_page_token')
            if not token:
                break
            if token in seen:
                raise Blocked('BARS_PAGINATION_LOOP')
            seen.add(token)
            params['page_token'] = token
        else:
            raise Blocked('BARS_PAGINATION_INCOMPLETE')
        instruments = []
        for row in reg['instruments']:
            history = bars.get(row['symbol'], [])[-300:]
            if len(history) < 200:
                raise Blocked('BARS_HISTORY_INCOMPLETE')
            instruments.append({k: row[k] for k in ('symbol', 'asset', 'cluster', 'market_cap',
                                                   'primary_domains', 'momentum_member')} |
                               {'bars': history, 'dollar_volume': sum(b['c'] * b['v'] for b in history[-20:]) / 20})
        snapshot = {'schema': 'agent-v2-quality-snapshot-v3', 'mode': 'SHADOW_FORWARD', 'at': iso(self.clock()),
                    'metadata_at': reg['received_at'], 'calendar_at': cal['received_at'],
                    'instruments': instruments, 'sources': [], 'events': cal['events'], 'quotes': [],
                    'hunter': hunter}
        validate_snapshot(snapshot, self.clock(), 'SHADOW_FORWARD')
        return {'snapshot': snapshot, 'registry': reg, 'calendar': cal, 'radar': radar,
                'input_hashes': {k: digest(v) for k, v in [('registry', reg), ('calendar', cal),
                                                        ('hunter', hunter), ('radar', radar)]}}

    def primary(self, instrument):
        cik = instrument['cik']
        value, _ = self.client.json('https://data.sec.gov/submissions/CIK' + cik + '.json')
        if value.get('cik') != cik:  # submissions wire identity is a zero-padded string
            raise Blocked('SEC_CIK_MISMATCH')
        recent = value['filings']['recent']
        lengths = {len(recent[k]) for k in ('form', 'acceptanceDateTime', 'accessionNumber', 'primaryDocument')}
        if len(lengths) != 1:
            raise Blocked('SEC_RECENT_SHAPE_INVALID')
        selected = next((i for i, form in enumerate(recent['form']) if form in ('8-K', '10-Q', '10-K')
                         and fresh(recent['acceptanceDateTime'][i], self.clock(), 120 * 86400)), None)
        if selected is None:
            raise Blocked('PRIMARY_DOCUMENT_MISSING')
        accession = recent['accessionNumber'][selected].replace('-', '')
        name = recent['primaryDocument'][selected]
        if not re.fullmatch(r'\d{18}', accession) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise Blocked('SEC_DOCUMENT_PATH_INVALID')
        url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{name}'
        raw, receipt = self.client.get(url)
        return document(raw, receipt, instrument['symbol'], recent['acceptanceDateTime'][selected])

    def quotes(self, symbols):
        value, receipt = self.client.json('https://data.alpaca.markets/v2/stocks/quotes/latest?' +
                                          urlencode({'symbols': ','.join(sorted(symbols)), 'feed': 'iex'}))
        if set(value['quotes']) != set(symbols):
            raise Blocked('QUOTE_COVERAGE_INCOMPLETE')
        return {'schema': 'agent-v2-quotes-v1', 'mode': 'SHADOW_FORWARD', 'at': receipt['received_at'],
                'quotes': [{'symbol': s, 'at': q['t'], 'received_at': receipt['received_at'],
                            'bid': q['bp'], 'ask': q['ap'], 'bid_size': q['bs'], 'ask_size': q['as'],
                            'source': 'alpaca_iex'} for s, q in value['quotes'].items()]}
