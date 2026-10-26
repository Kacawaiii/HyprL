"""Read a private, request-indexed vendor archive. This module has no network client.

An index identifies actual cached responses, not purported original vintages. Every
request URL and compressed object is checked before interpreting vendor-shaped data.
"""
import gzip
import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import sqlite3
import tempfile
import weakref
from urllib.parse import urlencode

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_bench import marketdata as md
from . import spec, quality

# A query reaching 365 calendar days (>= 250 sessions) before T separates a missing history from
# a legitimately short one for the 200-session rule; the grant's daily bars start 2025-01-01.
HISTORY_DAYS = 365


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close_bars(connection, temporary):
    connection.close()
    temporary.cleanup()


class DiskBars(Mapping):
    """Disposable exact-row index; only 64 symbols' histories stay in memory.

    The immutable vendor objects remain the evidence. This scratch database is
    rebuilt on every read, never reused as an admitted store or input proof.
    """
    def __init__(self, root):
        temporary = tempfile.TemporaryDirectory(prefix='archive-read-', dir=root)
        self.connection = sqlite3.connect(str(Path(temporary.name) / 'bars.sqlite'))
        weakref.finalize(self, _close_bars, self.connection, temporary)
        self.connection.execute('PRAGMA journal_mode=OFF')
        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('CREATE TABLE bars (symbol TEXT, at TEXT, received TEXT, body TEXT, PRIMARY KEY(symbol,at)) WITHOUT ROWID')
        self.symbols, self.cache = set(), OrderedDict()
        # Receipt clocks only, never a second in-memory copy of raw histories.
        # Overlapping paginated requests repeat millions of already indexed rows.
        self.receipts = {}

    def put(self, symbol, row, received):
        receipts = self.receipts.setdefault(symbol, {})
        prior = receipts.get(row['t'])
        if prior is not None:
            if received < prior:
                return
            if received == prior:
                old = self.connection.execute('SELECT body FROM bars WHERE symbol=? AND at=?',
                                              (symbol, row['t'])).fetchone()[0]
                if json.loads(old) == row:
                    return
                raise ReplayError('ARCHIVE_REVISION_CONFLICT', {'kind': 'bars:' + symbol, 'identity': row['t']})
        stamp = received.isoformat(timespec='microseconds')
        body = json.dumps(row, sort_keys=True, separators=(',', ':'))
        self.connection.execute('INSERT OR REPLACE INTO bars VALUES (?,?,?,?)', (symbol, row['t'], stamp, body))
        receipts[row['t']] = received
        self.symbols.add(symbol)
        self.cache.pop(symbol, None)

    def __getitem__(self, symbol):
        if symbol not in self.symbols:
            raise KeyError(symbol)
        if symbol not in self.cache:
            self.cache[symbol] = [json.loads(row[0]) for row in self.connection.execute(
                'SELECT body FROM bars WHERE symbol=? ORDER BY at', (symbol,))]
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(symbol)
        return self.cache[symbol]

    def __iter__(self):
        return iter(sorted(self.symbols))

    def __len__(self):
        return len(self.symbols)


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
        self.bars, self.news, self.filings, self.assets = DiskBars(self.root), {}, {}, {}
        asset_records, asset_fingerprints = {}, {}
        self.ambiguous_assets = set()
        self.invalid_bars, self.invalid_news = {}, {}
        self.requests, self.objects, self.asset_statuses = [], {}, set()
        self.news_windows = []
        self.filing_symbols = set()
        revisions = {}
        news_fingerprints = {}
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
            if before is not None and received >= before:
                raise ReplayError('LIVE_ARCHIVE_RECEIVED_AFTER_T')
            body = entry['body']
            self.objects[key] = item['sha256']
            # Coverage needs pagination metadata only. Keeping every scoring response
            # and article body duplicates the private cache and exceeds host memory.
            pagination = {'next_page_token': body.get('next_page_token')} if isinstance(body, dict) else {}
            self.requests.append((origin, path, params, pagination))
            def put(kind, identity, value, target):
                prior = revisions.get((kind, identity))
                if prior is None or received > prior:
                    target[identity] = value
                    revisions[kind, identity] = received
                elif received == prior and target[identity] != value:
                    raise ReplayError('ARCHIVE_REVISION_CONFLICT', {'kind': kind, 'identity': identity,
                        'prior_sha256': spec.sha256_canonical(target[identity]), 'new_sha256': spec.sha256_canonical(value)})
            if origin == md.ALPACA and path in ('/v2/stocks/bars', '/v1beta3/crypto/us/bars'):
                if params.get('timeframe') != '1Day':
                    continue
                symbols = params['symbols'].split(',')
                md.guard(symbols, params['start'], params['end'])
                # Adjusted bars only detect corporate actions at scoring; they never become inputs.
                if path == '/v2/stocks/bars' and params.get('adjustment') == 'all' and params.get('feed') == 'sip':
                    continue
                if path == '/v2/stocks/bars' and (params.get('adjustment') != 'raw' or params.get('feed') != 'sip'):
                    raise ReplayError('RAW_SIP_BARS_REQUIRED')
                for symbol, rows in body.get('bars', {}).items():
                    symbol = symbol.replace('/USD', '-USD')
                    for row in rows:
                        try:
                            try:
                                instant(row['t'])
                            except (KeyError, TypeError, ValueError, AttributeError, ReplayError):
                                raise ReplayError('INVALID_BAR') from None
                            self.bars.put(symbol, row, received)
                        except ReplayError as error:
                            if error.code not in ('INVALID_BAR', 'ARCHIVE_REVISION_CONFLICT'):
                                raise
                            self.invalid_bars[symbol] = 'INVALID_BAR' if error.code == 'INVALID_BAR' else 'CONTRADICTORY_BARS'
            elif origin == md.ALPACA and path == '/v1beta1/news':
                self.news_windows.append((params, pagination))
                for row in body.get('news', []):
                    identity = row.get('id', 'invalid-' + spec.sha256_canonical(row))
                    prior = revisions.get(('news', identity))
                    if prior is None or received >= prior:
                        fingerprint = spec.sha256_canonical(row)
                        if received == prior and news_fingerprints[identity] != fingerprint:
                            self.invalid_news[str(identity)] = 'CONTRADICTORY_NEWS'
                            self.news.pop(identity, None)
                            continue
                        news_fingerprints[identity] = fingerprint
                    if str(identity) in self.invalid_news:
                        continue
                    fields = ('id', 'headline', 'source', 'symbols', 'created_at', 'updated_at')
                    put('news', identity, {k: row.get(k) for k in fields}, self.news)
            elif path == '/v2/assets' and origin in ('https://paper-api.alpaca.markets', 'https://api.alpaca.markets'):
                # Real Alpaca assets have no listing-date field. Listing history is
                # established from dated bars, never from an invented asset field.
                status = params.get('status', 'active')
                self.asset_statuses.update(('active', 'inactive') if status == 'all' else (status,))
                for row in body:
                    if not isinstance(row.get('symbol'), str) or not row['symbol']:
                        continue
                    symbol = row['symbol']
                    fingerprint = spec.sha256_canonical({k: v for k, v in row.items() if k != 'id'})
                    asset_fingerprints.setdefault(symbol, set()).add(fingerprint)
                    asset_records.setdefault(symbol, row)
            elif origin == 'https://data.sec.gov' and path.startswith('/submissions/CIK'):
                if spec.spec()['unblock_2_amendment']['sec_policy']['state'] == 'UNAVAILABLE':
                    continue
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
        # UUID-only duplicates are compatible. Every other contradiction excludes
        # the symbol globally, including contradictions across different receipts.
        for symbol, row in asset_records.items():
            if len(asset_fingerprints[symbol]) != 1:
                self.ambiguous_assets.add(symbol)
            else:
                self.assets[symbol] = row
        self.bars.connection.commit()

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
        earliest, last_close = T - timedelta(days=HISTORY_DAYS), md.previous_close(T)
        for origin, path, params, body in self.requests:
            if (origin != md.ALPACA or path != '/v2/stocks/bars' or params.get('timeframe') != '1Day' or
                    symbol not in params['symbols'].split(',') or
                    params.get('adjustment') != 'raw' or
                    instant(params['start']) > earliest or
                    instant(params['end']) < last_close):
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


class UnavailableArchive:
    """An absent data index excludes its day, with no invented source evidence."""
    def __init__(self, reason):
        self.reason = reason
        self.bars, self.news, self.filings, self.assets = {}, {}, {}, {}
        self.ambiguous_assets, self.filing_symbols = set(), set()
        self.invalid_bars, self.invalid_news, self.objects = {}, {}, {}

    def require_assets(self):
        raise ReplayError(self.reason)

    def require_news(self, start, end):
        raise ReplayError(self.reason)

    def proof(self):
        return {'state': 'EXCLUDED', 'reason': self.reason, 'objects': {}}
