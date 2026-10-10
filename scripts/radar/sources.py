"""Bounded, headline-only RSS/Atom, Alpaca and GDELT adapters."""
from datetime import timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from urllib.parse import urlencode, quote, urlsplit
import html
import re
import xml.etree.ElementTree as ET

from .core import RadarError, UTC, canonical_url, digest, instant, iso
from .registry import FEEDS, CHANNELS, THEMES, CRYPTO_NAMES, universe


def text(value, limit=600):
    value = re.sub(r'<[^>]*>', ' ', str(value or ''))
    return ' '.join(html.unescape(value).split())[:limit]


def publication(value, received):
    if not value:
        return None
    try:
        at = instant(value)
    except (ValueError, AttributeError):
        try:
            at = parsedate_to_datetime(value)
            if at.tzinfo is None:
                return None
            at = at.astimezone(UTC)
        except (ValueError, TypeError, IndexError):
            return None
    return iso(at) if at <= received else None


def item(source, publisher, url, headline, summary, published_at, received_at, *, symbols=(), retail=False, primary=False, discovery_at=None):
    if (not isinstance(headline, str) or summary is not None and not isinstance(summary, str)
            or not isinstance(symbols, (list, tuple)) or any(not isinstance(s, str) for s in symbols)):
        raise RadarError('INVALID_PROVIDER_SHAPE')
    publisher = publisher if isinstance(publisher, str) and publisher else 'unknown'
    source_url = url
    url = canonical_url(url, allow_http=True)
    if not text(headline):
        raise RadarError('EMPTY_HEADLINE')
    # Only headline + short summary, never content:encoded / full article bodies.
    return {'id': digest([source, url, headline, summary, published_at]),
            'source': source, 'publisher': publisher, 'url': url, 'source_url': source_url,
            'published_at': publication(published_at, received_at), 'received_at': iso(received_at),
            'discovery_at': discovery_at, 'headline': text(headline, 300), 'summary': text(summary),
            'headline_hash': sha256(str(headline).encode()).hexdigest(),
            'summary_hash': sha256(str(summary).encode()).hexdigest(),
            'symbols': list(symbols), 'retail': retail, 'primary': primary}


def parse_feed(raw, source, publisher, received, *, primary=False, retail=False):
    if re.search(br'<!\s*(DOCTYPE|ENTITY)', raw, re.I):
        raise RadarError('UNSAFE_XML')
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise RadarError('INVALID_XML') from None
    name = lambda node: node.tag.rsplit('}', 1)[-1]
    if name(root) not in ('rss', 'feed', 'RDF'):
        raise RadarError('NOT_A_FEED')
    result = []
    for node in root.iter():
        if name(node) not in ('item', 'entry'):
            continue
        fields = {name(c): c for c in node}
        def field(key):
            element = fields.get(key)
            return ''.join(element.itertext()) if element is not None else ''
        link = field('link')
        for child in node:
            if name(child) == 'link' and child.get('rel', 'alternate') == 'alternate' and child.get('href'):
                link = child.get('href')
                break
        published = field('pubDate') or field('published') or field('date')
        # Atom 'updated' is an update time, not an original publication time.
        summary = field('description') or field('summary')
        if retail:
            summary = next((''.join(c.itertext()) for c in node.iter() if name(c) == 'description'), '')
        try:
            result.append(item(source, publisher, link, field('title'), summary, published, received,
                               primary=primary and urlsplit(link).hostname == 'www.ecb.europa.eu', retail=retail))
        except RadarError:
            continue
    return result


def publisher_for(article):
    # Distribution channels do not constitute independent confirmations.
    combined = str(article.get('author', '')) + ' ' + str(article.get('source', ''))
    for label in ('reuters', 'benzinga', 'bloomberg', 'associated press', 'coindesk', 'cointelegraph'):
        if label in combined.lower():
            return label.replace(' ', '_')
    return 'benzinga'  # this endpoint is a Benzinga distribution feed


class Collector:
    def __init__(self, client, *, channels=None):
        self.client, self.store = client, client.store
        self.channels = channels if channels is not None else [{'handle': h} for h in CHANNELS]
        self.health = []
        self.stopped = set()

    def attempt(self, source, function):
        group = source.split(':')[0]
        try:
            if group in self.stopped:
                raise RadarError('SOURCE_STOPPED_AFTER_THROTTLE')
            count = function()
            state = {'source': source, 'status': 'LIVE', 'items': count, 'checked_at': iso(self.store.clock())}
        except (RadarError, ValueError, KeyError, TypeError, UnicodeError, AttributeError, IndexError, OverflowError):
            # Avoid exception text from arbitrary parsers/transports reaching a report.
            import sys
            error = sys.exc_info()[1]
            code = str(error) if isinstance(error, RadarError) else 'INVALID_PROVIDER_SHAPE'
            if code == 'HTTP_429_STOP_SOURCE':
                self.stopped.add(group)
            state = {'source': source, 'status': 'DEAD' if code in ('HTTP_404', 'HTTP_410', 'NOT_A_FEED') else 'BLOCKED',
                     'reason': code, 'items': 0, 'checked_at': iso(self.store.clock())}
        self.health.append(state)
        self.store.append('health', source, state)

    def add(self, items):
        added = 0
        for story in items:
            if not self.store.latest('news', story['id']):
                self.store.append('news', story['id'], story)
                added += 1
        return added

    def feed(self, source, url, publisher, *, primary=False, retail=False):
        previous = self.store.latest('feed_cache', source) or {}
        headers = {k: previous[v] for k, v in [('If-None-Match', 'etag'), ('If-Modified-Since', 'last_modified')] if previous.get(v)}
        status, response, raw, received = self.client.get(url, 'youtube' if retail else 'rss', source, headers=headers)
        if status == 304:
            if not previous.get('verified'):
                raise RadarError('UNVERIFIED_304')
            return 0
        if status != 200:
            raise RadarError('HTTP_' + str(status))
        items = parse_feed(raw, source, publisher, received, primary=primary, retail=retail)
        normalized = {k.lower(): v for k, v in response.items()}
        self.store.append('feed_cache', source, {'etag': normalized.get('etag'), 'last_modified': normalized.get('last-modified'), 'verified': True})
        return self.add(items)

    def channel(self, config):
        handle = config.get('handle', '')
        channel_id = config.get('channel_id')
        if not channel_id:
            cached = self.store.latest('channel', handle)
            channel_id = cached['channel_id'] if cached else None
        if not channel_id:
            status, _, body, _ = self.client.get('https://www.youtube.com/' + quote(handle, safe='@'), 'youtube', 'discover:' + handle)
            if status != 200:
                raise RadarError('HTTP_' + str(status))
            # Metadata only; never downloads video, captions or stores the page.
            # Related videos can contain many channelId values. Prefer the
            # channel metadata's externalId rather than guessing from them.
            ids = re.findall(br'"externalId"\s*:\s*"(UC[A-Za-z0-9_-]{22})"', body)
            if not ids:
                ids = re.findall(br'"channelId"\s*:\s*"(UC[A-Za-z0-9_-]{22})"', body)
            if not ids or len(set(ids)) != 1:
                raise RadarError('CHANNEL_ID_UNVERIFIED')
            channel_id = ids[0].decode()
            self.store.append('channel', handle, {'channel_id': channel_id})
        if not re.fullmatch(r'UC[A-Za-z0-9_-]{22}', channel_id):
            raise RadarError('INVALID_CHANNEL_ID')
        publisher = next((label for label in ('cnbc', 'yahoo', 'bloomberg') if label in handle.lower()), channel_id)
        return self.feed('youtube:' + channel_id, 'https://www.youtube.com/feeds/videos.xml?' + urlencode({'channel_id': channel_id}), publisher, retail=True)

    def news(self):
        pending = self.store.latest('news_cursor', 'alpaca')
        if pending and pending.get('token'):
            start, end, token = pending['start'], pending['end'], pending['token']
        else:
            end = iso(self.store.clock())
            start = pending['end'] if pending else iso(self.store.clock() - timedelta(days=2))
            token = None
        total = 0
        for _ in range(12):
            params = {'start': start, 'end': end, 'sort': 'asc', 'limit': 50, 'include_content': 'false'}
            if token:
                params['page_token'] = token
            payload, received = self.client.json('https://data.alpaca.markets/v1beta1/news?' + urlencode(params), 'alpaca', 'alpaca:news')
            stories = [item('alpaca:news', publisher_for(a), a['url'], a['headline'], a.get('summary', ''),
                            a.get('created_at'), received, symbols=a.get('symbols', [])) for a in payload['news']]
            total += self.add(stories)
            token = payload.get('next_page_token')
            # Cursor is durable only after every page's items have been appended.
            self.store.append('news_cursor', 'alpaca', {'start': start, 'end': end, 'token': token, 'backlog': bool(token)})
            if not token:
                break
        return total

    def gdelt(self, theme, query):
        params = {'query': query, 'mode': 'artlist', 'format': 'json', 'timespan': '24h', 'maxrecords': 100, 'sort': 'datedesc'}
        payload, received = self.client.json('https://api.gdeltproject.org/api/v2/doc/doc?' + urlencode(params), 'gdelt', 'gdelt:' + theme)
        stories = []
        for article in payload.get('articles', []):
            url = article['url']
            # Do not fetch articles. GDELT seen time is explicitly discovery only.
            stories.append(item('gdelt:' + theme, article.get('domain') or urlsplit(url).hostname,
                                url, article['title'], '', None, received, discovery_at=article.get('seendate')))
        return self.add(stories)

    def screener(self, name):
        path = {'actives': '/v1beta1/screener/stocks/most-actives', 'stocks': '/v1beta1/screener/stocks/movers',
                'crypto': '/v1beta1/screener/crypto/movers'}[name]
        payload, received = self.client.json('https://data.alpaca.markets' + path + '?top=50', 'alpaca', 'alpaca:screener:' + name)
        rows = payload.get('most_actives', []) + payload.get('gainers', []) + payload.get('losers', [])
        if not rows:
            raise RadarError('EMPTY_SCREENER')
        self.store.append('screener', name, {'received_at': iso(received), 'rows': rows})
        return len(rows)

    def bars(self, symbols, *, crypto=False, minute=False, start=None, end=None):
        path = '/v1beta3/crypto/us/bars' if crypto else '/v2/stocks/bars'
        end = end or iso(self.store.clock())
        start = start or iso(self.store.clock() - timedelta(days=65))
        params = {'symbols': ','.join(symbols), 'timeframe': '1Min' if minute else '1Day',
                  'start': start, 'end': end, 'limit': 10000, 'sort': 'asc'}
        if not crypto:
            params.update(feed='iex', adjustment='all')
        total = 0
        for _ in range(12):
            payload, received = self.client.json('https://data.alpaca.markets' + path + '?' + urlencode(params), 'alpaca', 'alpaca:bars')
            for symbol, bars in payload['bars'].items():
                self.store.append('minute_bars' if minute else 'daily_bars', symbol,
                                  {'symbol': symbol, 'bars': bars, 'received_at': iso(received), 'provider': 'alpaca', 'feed': None if crypto else 'iex'})
                total += len(bars)
            token = payload.get('next_page_token')
            if not token:
                return total
            params['page_token'] = token
        raise RadarError('BARS_PAGINATION_INCOMPLETE')

    def collect(self, *, gdelt=True, sleep=None, markets=True):
        self.attempt('alpaca:news', self.news)
        if markets:
            for name in ('actives', 'stocks', 'crypto'):
                self.attempt('alpaca:screener:' + name, lambda name=name: self.screener(name))
        for source, url, publisher, primary in FEEDS:
            # Permanently bad candidates are dropped after verified dead response.
            previous = self.store.latest('health', 'rss:' + source)
            if previous and previous['status'] == 'DEAD':
                self.health.append(previous)
                continue
            self.attempt('rss:' + source, lambda source=source, url=url, publisher=publisher, primary=primary:
                         self.feed('rss:' + source, url, publisher, primary=primary))
        for config in self.channels:
            source = 'youtube:' + config.get('handle', config.get('channel_id', 'unknown'))
            previous = self.store.latest('health', source)
            if previous and previous['status'] == 'DEAD':
                self.health.append(previous)
                continue
            self.attempt(source, lambda config=config: self.channel(config))
        if gdelt:
            for index, (theme, query) in enumerate(THEMES.items()):
                if index and sleep and not '11:45' <= self.store.clock().strftime('%H:%M') < '12:30':
                    sleep(7)
                self.attempt('gdelt:' + theme, lambda theme=theme, query=query: self.gdelt(theme, query))
        if markets:
            symbols = universe()
            for offset in range(0, len(symbols), 100):
                batch = symbols[offset:offset + 100]
                self.attempt('alpaca:daily:' + str(offset), lambda batch=batch: self.bars(batch))
            # Isolate unsupported crypto candidates; one rejected symbol cannot
            # suppress BTC/ETH or the rest of the broad universe.
            for symbol in CRYPTO_NAMES:
                self.attempt('alpaca:crypto_daily:' + symbol, lambda symbol=symbol: self.bars([symbol + '/USD'], crypto=True))
        return self.health
