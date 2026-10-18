"""Read a private, request-indexed vendor archive. This module has no network client.

An index identifies actual cached responses, not purported original vintages. Every
request URL and compressed object is checked before interpreting vendor-shaped data.
"""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_bench import marketdata as md
from . import spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Archive:
    def __init__(self, root, index, *, before=None):
        self.root = Path(root).resolve()
        index = Path(index).resolve()
        self.cache = index.parent / 'cache'
        if self.root not in self.cache.resolve().parents:
            raise ReplayError('ARCHIVE_CACHE_OUTSIDE_DATA_ROOT')
        self.index = json.loads(index.read_text())
        if self.index.get('schema') != 'replay-vendor-request-index-v1':
            raise ReplayError('ARCHIVE_INDEX_INVALID')
        self.bars, self.news, self.filings, self.assets = {}, {}, {}, {}
        self.requests, self.objects, self.asset_statuses = [], {}, set()
        self.news_windows = []
        self.filing_symbols = set()
        revisions = {}
        for item in self.index['objects']:
            origin, path, params = item['origin'], item['path'], item.get('params', {})
            url = origin + path + ('?' + urlencode(sorted(params.items())) if params else '')
            key = hashlib.sha256(url.encode()).hexdigest()
            if key != item['cache_key']:
                raise ReplayError('ARCHIVE_REQUEST_HASH_MISMATCH')
            file = self.cache / key[:2] / (key + '.json.gz')
            if self.root not in file.resolve().parents or digest(file) != item['sha256']:
                raise ReplayError('ARCHIVE_OBJECT_HASH_MISMATCH')
            entry = json.loads(gzip.decompress(file.read_bytes()))
            received = instant(entry['received_at'])
            if before is not None and received > before:
                raise ReplayError('LIVE_ARCHIVE_RECEIVED_AFTER_T')
            body = entry['body']
            self.objects[key] = item['sha256']
            self.requests.append((origin, path, params, body))
            def put(kind, identity, value, target):
                prior = revisions.get((kind, identity))
                if prior is None or received > prior:
                    target[identity] = value
                    revisions[kind, identity] = received
                elif received == prior and target[identity] != value:
                    raise ReplayError('ARCHIVE_REVISION_CONFLICT')
            if origin == md.ALPACA and path in ('/v2/stocks/bars', '/v1beta3/crypto/us/bars'):
                if params.get('timeframe') != '1Day':
                    continue
                symbols = params['symbols'].split(',')
                md.guard(symbols, params['start'], params['end'])
                if path == '/v2/stocks/bars' and (params.get('adjustment') != 'raw' or params.get('feed') != 'sip'):
                    raise ReplayError('RAW_SIP_BARS_REQUIRED')
                for symbol, rows in body.get('bars', {}).items():
                    symbol = symbol.replace('/USD', '-USD')
                    dest = self.bars.setdefault(symbol, {})
                    for row in rows:
                        put('bars:' + symbol, row['t'], row, dest)
            elif origin == md.ALPACA and path == '/v1beta1/news':
                self.news_windows.append((params, body))
                for row in body.get('news', []):
                    put('news', row['id'], row, self.news)
            elif path == '/v2/assets' and origin in ('https://paper-api.alpaca.markets', 'https://api.alpaca.markets'):
                # Real Alpaca assets have no listing-date field. Listing history is
                # established from dated bars, never from an invented asset field.
                status = params.get('status', 'active')
                self.asset_statuses.update(('active', 'inactive') if status == 'all' else (status,))
                for row in body:
                    put('assets', row['symbol'], row, self.assets)
            elif origin == 'https://data.sec.gov' and path.startswith('/submissions/CIK'):
                cik = path.removeprefix('/submissions/CIK').removesuffix('.json')
                if body.get('cik') != cik or len(cik) != 10 or not cik.isdigit():
                    raise ReplayError('SEC_CIK_IDENTITY_MISMATCH')
                recent = body.get('filings', {}).get('recent', {})
                keys = ('form', 'acceptanceDateTime', 'accessionNumber')
                if len({len(recent.get(k, [])) for k in keys}) != 1:
                    raise ReplayError('SEC_PARALLEL_ARRAY_LENGTH_MISMATCH')
                for symbol in body.get('tickers', []):
                    self.filing_symbols.add(symbol)
                    dest = self.filings.setdefault(symbol, {})
                    for form, accepted, accession in zip(*(recent.get(k, []) for k in keys)):
                        put('filings:' + symbol, accession, {'form': form, 'accession': accession,
                            'accepted_at': md.iso(md.acceptance_upper_bound(accepted))}, dest)
        self.bars = {s: sorted(rows.values(), key=lambda r: r['t']) for s, rows in self.bars.items()}

    def require_news(self, start, end):
        """Verify a complete paginated all-symbol query covers this interval."""
        groups = {}
        for params, body in self.news_windows:
            if params.get('symbols') or instant(params['start']) > start or instant(params['end']) < end:
                continue
            group = tuple(sorted((k, str(v)) for k, v in params.items() if k != 'page_token'))
            groups.setdefault(group, {})[params.get('page_token')] = body.get('next_page_token')
        for pages in groups.values():
            token, seen = None, set()
            while token in pages and token not in seen:
                seen.add(token)
                token = pages[token]
                if token is None:
                    return
        raise ReplayError('NEWS_ARCHIVE_COVERAGE_MISSING')

    def require_assets(self):
        if not {'active', 'inactive'} <= self.asset_statuses:
            raise ReplayError('ACTIVE_AND_INACTIVE_ASSETS_REQUIRED')

    def require_bar_history(self, symbol, T):
        from datetime import timedelta
        groups = {}
        for origin, path, params, body in self.requests:
            if (origin != md.ALPACA or path != '/v2/stocks/bars' or params.get('timeframe') != '1Day' or
                    symbol not in params['symbols'].split(',') or
                    instant(params['start']) > T - timedelta(days=550) or instant(params['end']) < T):
                continue
            group = tuple(sorted((k, str(v)) for k, v in params.items() if k != 'page_token'))
            groups.setdefault(group, {})[params.get('page_token')] = body.get('next_page_token')
        for pages in groups.values():
            token, seen = None, set()
            while token in pages and token not in seen:
                seen.add(token)
                token = pages[token]
                if token is None:
                    return
        raise ReplayError('CANDIDATE_HISTORY_COVERAGE_MISSING', symbol)

    def proof(self):
        return {'index_hash': spec.sha256_canonical(self.index),
                'cache_directory': str(self.cache.relative_to(self.root)), 'objects': self.objects}
