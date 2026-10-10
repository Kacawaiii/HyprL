"""Separate signed grant for bounded SEC/Fed metadata and operator delivery."""
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
from datetime import timedelta
import json
import math
import re
import xml.etree.ElementTree as ET

from .core import Authorization, RadarError, canonical_url, digest, instant, iso
from .sources import Collector, item, parse_feed, publication, text

SEC_FEED = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom&count=100'
TICKERS = 'https://www.sec.gov/files/company_tickers.json'
FED_FEEDS = {'press': 'https://www.federalreserve.gov/feeds/press_all.xml',
             'speeches': 'https://www.federalreserve.gov/feeds/speeches.xml'}
FEED_SCOPE = '/cgi-bin/browse-edgar (action=getcurrent, type=8-K, output=atom)'


class OfficialAuthorization(Authorization):
    def __init__(self, path):
        try:
            p = json.loads(Path(path).expanduser().read_text())
            if (p['authorization'] != 'news-radar-sec-fed-telegram-v1'
                    or p['extends'] != 'news-radar-v1' or p['operator_signed'] is not True
                    or instant(p['not_after']) <= instant(p['granted_at'])):
                raise ValueError()
            scope = p['scope']
            budgets = {}
            for host, group, cap, method in [('www.sec.gov', 'sec', 300, 'GET'),
                    ('www.federalreserve.gov', 'fed', 48, 'GET'), ('api.telegram.org', 'telegram', 12, 'POST')]:
                rule = scope['https://' + host]
                maximum = rule.get('max_requests_per_day') if method == 'GET' else rule.get('max_messages_per_day')
                if (rule['methods'] != [method] or type(maximum) is not int or not 0 < maximum <= cap
                        or not isinstance(rule['paths'], list) or any(not isinstance(x, str) for x in rule['paths'])):
                    raise ValueError()
                budgets[group] = maximum
            rate = scope['https://www.sec.gov']['max_requests_per_second']
            ua = scope['https://www.sec.gov']['user_agent']
            if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 2
                    or not isinstance(ua, str) or not ua.strip() or any(ord(c) < 32 for c in ua)):
                raise ValueError()
            self.identity = digest(p)  # digest original signed document, not derived settings
            self.payload = {**p, 'budgets': budgets, 'spacing_seconds': {'sec': 1 / rate, 'fed': 0}}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise RadarError('AUTHORIZATION_MISSING_OR_INVALID') from None

    def group_for(self, url):
        return {'www.sec.gov': 'sec', 'www.federalreserve.gov': 'fed'}.get(urlsplit(url).hostname)

    def permit(self, url):
        canonical_url(url)  # validate transport safety without dropping query keys
        p = urlsplit(url)
        rule = self.payload['scope'].get('https://' + p.netloc)
        if not rule or rule['methods'] != ['GET'] or self.group_for(url) is None:
            raise RadarError('SOURCE_OUTSIDE_GRANT')
        if p.hostname == 'www.sec.gov' and p.path == '/cgi-bin/browse-edgar':
            pairs = parse_qsl(p.query, keep_blank_values=True)
            query = dict(pairs)
            if (FEED_SCOPE not in rule['paths'] or len(pairs) != len(query)
                    or set(query) - {'action', 'type', 'output', 'count'}
                    or any(query.get(k) != v for k, v in {'action': 'getcurrent', 'type': '8-K', 'output': 'atom'}.items())
                    or query.get('count', '100') not in ('10', '20', '40', '80', '100')):
                raise RadarError('PATH_OUTSIDE_GRANT')
        elif p.query or p.path not in rule['paths'] or url.split('?')[0] not in (TICKERS, *FED_FEEDS.values()):
            raise RadarError('PATH_OUTSIDE_GRANT')

    def permit_delivery(self):
        rule = self.payload['scope']['https://api.telegram.org']
        if (rule['paths'] != ['/bot<token>/sendMessage']
                or rule.get('recipient') != "the operator's own chat (ADMIN_TELEGRAM_ID) only"):
            raise RadarError('TELEGRAM_OUTSIDE_GRANT')


ITEM_TYPES = {'1.01': ('material_agreement', 'material agreements', 'material_agreement'),
              '2.02': ('results', 'results of operations', 'earnings'),
              '5.02': ('executive_change', 'executive changes', 'executive_change'),
              '8.01': ('other', 'other events', 'other')}


def parse_sec(raw, received, mapping):
    if re.search(br'<!\s*(DOCTYPE|ENTITY)', raw, re.I):
        raise RadarError('UNSAFE_XML')
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise RadarError('INVALID_XML') from None
    ns = '{http://www.w3.org/2005/Atom}'
    if root.tag != ns + 'feed':
        raise RadarError('NOT_A_FEED')
    result = []
    for node in root.findall(ns + 'entry'):
        title = node.findtext(ns + 'title', '')
        issuer = re.fullmatch(r'(8-K(?:/A)?)\s+-\s+(.+?)\s+\((\d{1,10})\)\s+\(Filer\)', title.strip())
        link = next((c.get('href', '') for c in node.findall(ns + 'link') if c.get('rel') == 'alternate'), '')
        location = re.fullmatch(r'https://www\.sec\.gov/Archives/edgar/data/(\d+)/(\d{18})/(\d{10}-\d{2}-\d{6})-index\.(?:htm|html)', link)
        if not issuer or not location or int(issuer[3]) != int(location[1]) or location[2] != location[3].replace('-', ''):
            continue
        cik = issuer[3].zfill(10)
        summary = text(node.findtext(ns + 'summary', ''), 2000)
        # Current feed supplies literal "Item 2.02: ..." lines. Never infer an
        # item from a company's name, size, accession digits or generic prose.
        items = sorted(set(re.findall(r'\bItem\s+(\d+\.\d{2})\s*:', summary, re.I)))
        classified = [ITEM_TYPES[k] for k in items if k in ITEM_TYPES]
        labels = ', '.join(c[1] for c in classified) or ('other disclosed items' if items else 'items unknown')
        published = node.findtext(ns + 'published')
        story = item('sec:8k', 'sec', link, issuer[2] + ' — ' + issuer[1] + ': ' + labels,
                     summary, published, received, primary=True, symbols=mapping.get(cik, []))
        story['themes'] = sorted({c[2] for c in classified})
        story['filing'] = {'form': issuer[1], 'cik': cik, 'accession': location[3],
                           'items': items, 'classifications': [c[0] for c in classified],
                           'items_status': 'DISCLOSED_IN_FEED' if items else 'UNKNOWN_FEED_METADATA',
                           'updated_at': publication(node.findtext(ns + 'updated'), received)}
        # A correction becomes a new observation of the same filing event.
        story['id'] = digest([story['id'], story['filing'], story['symbols']])
        result.append(story)
    return result


class OfficialCollector(Collector):
    def __init__(self, client, *, sleep=None):
        super().__init__(client, channels=[])
        self.sleep = sleep
        self.mapping = {}

    def tickers(self):
        self.client.grant.check(self.store.clock())
        self.client.grant.permit(TICKERS)
        cached = self.store.latest('ticker_map', 'sec')
        if cached and self.store.clock() - instant(cached['received_at']) < timedelta(days=1):
            self.mapping = cached['mapping']
            return len(self.mapping)
        headers = {'If-None-Match': cached['etag']} if cached and cached.get('etag') else {}
        status, response, raw, received = self.client.get(TICKERS, 'sec', 'sec:tickers', headers=headers)
        if status == 304 and cached:
            self.mapping = cached['mapping']
        elif status == 200:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not payload:
                raise RadarError('INVALID_PROVIDER_SHAPE')
            mapping = {}
            for row in payload.values():
                if (not isinstance(row, dict) or type(row.get('cik_str')) is not int or not 0 < row['cik_str'] < 10**10
                        or not isinstance(row.get('title'), str) or not isinstance(row.get('ticker'), str)
                        or not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,14}', row['ticker'])):
                    raise RadarError('INVALID_PROVIDER_SHAPE')
                mapping.setdefault(str(row['cik_str']).zfill(10), []).append(row['ticker'])
            self.mapping = {cik: sorted(set(symbols)) for cik, symbols in mapping.items()}
        else:
            raise RadarError('UNVERIFIED_304' if status == 304 else 'HTTP_' + str(status))
        normalized = {k.lower(): v for k, v in response.items()}
        self.store.append('ticker_map', 'sec', {'mapping': self.mapping, 'received_at': iso(received),
                          'etag': normalized.get('etag') or (cached or {}).get('etag')})
        return len(self.mapping)

    def official_feed(self, source, url, group):
        previous = self.store.latest('feed_cache', source) or {}
        headers = {k: previous[v] for k, v in [('If-None-Match', 'etag'), ('If-Modified-Since', 'last_modified')] if previous.get(v)}
        status, response, raw, received = self.client.get(url, group, source, headers=headers)
        if status == 304:
            if not previous.get('verified'):
                raise RadarError('UNVERIFIED_304')
            if group == 'sec' and self.mapping:
                # A previously unmapped filing can gain a ticker only after a
                # later authorized mapping receipt; preserve the old observation.
                filings = {s['url']: s for s in self.store.rows('news') if s['source'] == 'sec:8k'}
                for story in filings.values():
                    symbols = self.mapping.get(story['filing']['cik'], [])
                    if symbols != story['symbols']:
                        enriched = {**story, 'symbols': symbols, 'received_at': iso(received),
                                    'mapping_received_at': (self.store.latest('ticker_map', 'sec') or {}).get('received_at')}
                        enriched['id'] = digest([story['id'], symbols])
                        self.add([enriched])
            return 0
        if status != 200:
            raise RadarError('HTTP_' + str(status))
        if group == 'sec':
            stories = parse_sec(raw, received, self.mapping)
        else:
            stories = parse_feed(raw, source, 'federal_reserve', received, primary=True)
            stories = [s for s in stories if urlsplit(s['url']).hostname == 'www.federalreserve.gov']
            for story in stories:
                story['themes'] = ['macro']
                story['macro_kind'] = source.split(':')[1]
        normalized = {k.lower(): v for k, v in response.items()}
        self.store.append('feed_cache', source, {'etag': normalized.get('etag'), 'last_modified': normalized.get('last-modified'), 'verified': True})
        return self.add(stories)

    def collect(self, *, sec=True, fed=True):
        if sec:
            self.attempt('sec:tickers', self.tickers)
            if self.sleep:
                self.sleep(self.client.grant.payload['spacing_seconds']['sec'])
            self.poll('sec:8k', SEC_FEED, 'sec', 300)
        if fed:
            for kind, url in FED_FEEDS.items():
                self.poll('fed:' + kind, url, 'fed', 3600)
        return self.health

    def poll(self, source, url, group, interval):
        last = self.store.last_dispatch(group, source)
        if last and self.store.clock() - last < timedelta(seconds=interval):
            previous = self.store.latest('health', source)
            self.health.append(previous or {'source': source, 'status': 'BLOCKED', 'reason': 'POLL_SPACING', 'items': 0, 'checked_at': iso(last)})
            return
        self.attempt(source, lambda: self.official_feed(source, url, group))
