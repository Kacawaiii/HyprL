"""Grant-bound GET collector that writes the request index the offline archive reads.

Only research data: daily bars, scoring fills/marks, news, asset lists and SEC submissions.
Every response lands in the private cache format of the replay transport, and every request
(cache hit or fresh) is recorded with its URL hash and compressed-object digest. No order,
position or account endpoint exists here; the archive reader stays network-free.
"""
import argparse
from datetime import timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from scripts.trading_lab.replay_bench.grant import (Budget, ReplayError, http_get, instant, iso, load_env, now)
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import spec

COLLECTION_GRANT = 'replay-data-collection-v1'
PAPER = 'https://paper-api.alpaca.markets'
SEC_DATA, SEC_WWW = 'https://data.sec.gov', 'https://www.sec.gov'
# (origin, path regex) -> granting file. Trades and the ticker map stay under replay-bench-v1.
ROUTES = {
    (md.ALPACA, r'/v2/stocks/bars'): COLLECTION_GRANT,
    (md.ALPACA, r'/v1beta3/crypto/us/bars'): COLLECTION_GRANT,
    (md.ALPACA, r'/v1beta1/news'): COLLECTION_GRANT,
    (PAPER, r'/v2/assets'): COLLECTION_GRANT,
    (SEC_DATA, r'/submissions/CIK\d{10}\.json'): COLLECTION_GRANT,
    (md.ALPACA, r'/v2/stocks/trades'): 'replay-bench-v1',
    (SEC_WWW, r'/files/company_tickers\.json'): 'replay-bench-v1',
}
SPACING = {md.ALPACA: 0.31, PAPER: 0.31, SEC_DATA: 0.21, SEC_WWW: 0.21}   # Alpaca 200/min, SEC 5/s
BARS_FROM = '2025-01-01T00:00:00Z'


class Grants:
    """Both operator files, rechecked unchanged and unexpired before every request."""

    def __init__(self, collection, bench, *, clock=now):
        self.paths = {COLLECTION_GRANT: Path(collection), 'replay-bench-v1': Path(bench)}
        self.payloads = {k: json.loads(p.read_text()) for k, p in self.paths.items()}
        self.clock = clock
        if self.payloads[COLLECTION_GRANT].get('id') != COLLECTION_GRANT:
            raise ReplayError('COLLECTION_GRANT_INVALID')
        if self.payloads['replay-bench-v1'].get('authorization') != 'replay-bench-v1':
            raise ReplayError('GRANT_INVALID')
        if (sha256_canonical(self.payloads[COLLECTION_GRANT]) != spec.spec()['data_collection']['collection_grant_sha256'] or
                sha256_canonical(self.payloads['replay-bench-v1']) != spec.spec()['authorization']['payload_sha256']):
            raise ReplayError('COLLECTION_GRANT_HASH_MISMATCH')
        self.check()

    def check(self):
        for name, path in self.paths.items():
            if json.loads(path.read_text()) != self.payloads[name]:
                raise ReplayError('GRANT_CHANGED', name)
        at = self.clock()
        if not at < instant(self.payloads[COLLECTION_GRANT]['expires'] + 'T23:59:59Z'):
            raise ReplayError('COLLECTION_GRANT_EXPIRED')
        bench = self.payloads['replay-bench-v1']
        if not instant(bench['granted_at']) <= at < instant(bench['not_after']):
            raise ReplayError('GRANT_EXPIRED_OR_NOT_STARTED')

    def route(self, origin, path, params):
        for (o, pattern), name in ROUTES.items():
            if o == origin and re.fullmatch(pattern, path):
                if path == '/v2/assets' and params.get('status') not in ('active', 'inactive'):
                    raise ReplayError('ASSET_STATUS_NOT_GRANTED')
                return name
        raise ReplayError('PATH_NOT_GRANTED', origin + path)

    def daily_limit(self, name, origin):
        if name == 'replay-bench-v1':
            return int(self.payloads[name]['scope'][origin]['max_requests_per_day'])
        return None


class Collector:
    """Same cache layout and URL hash as replay_bench.grant.Transport, plus a request index."""

    def __init__(self, grants, budget, cache_root, *, alpaca_env=None, sec_user_agent=None,
                 getter=http_get, sleep=time.sleep, clock=now, refresh=False, refresh_news=False, before=None):
        self.grants, self.budget, self.getter, self.sleep, self.clock = grants, budget, getter, sleep, clock
        self.cache = Path(cache_root)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._alpaca = load_env(alpaca_env) if alpaca_env else None
        self.sec_user_agent = sec_user_agent
        # Live polling reads a fresh receipt for the same URL only in a fresh per-day cache.
        self.refresh = refresh
        self.refresh_news, self.before = refresh_news, before
        self.index, self.requests, self._last = {}, 0, {}
        self.retry_missing, self.retried_keys = False, set()
        self.offline = False
        self.scoring_symbols = set()

    def _headers(self, origin):
        if origin in (md.ALPACA, PAPER):
            if not self._alpaca:
                raise ReplayError('ALPACA_KEYS_REQUIRED')
            return {'APCA-API-KEY-ID': self._alpaca['APCA_API_KEY_ID'],
                    'APCA-API-SECRET-KEY': self._alpaca['APCA_API_SECRET_KEY'], 'Accept': 'application/json'}
        if not self.sec_user_agent:
            raise ReplayError('SEC_USER_AGENT_REQUIRED')
        return {'User-Agent': self.sec_user_agent, 'Accept': 'application/json'}

    def get(self, origin, path, params=None):
        if origin in (SEC_WWW, SEC_DATA):
            raise ReplayError('SEC_COLLECTION_DISABLED')
        params = dict(params or {})
        self.grants.check()
        name = self.grants.route(origin, path, params)
        if origin == md.ALPACA and ('bars' in path or 'trades' in path):
            if not params.get('start') or not params.get('end') or not params.get('symbols'):
                raise ReplayError('BOUNDED_PRICE_QUERY_REQUIRED')
            md.guard(params['symbols'].split(','), params['start'], params['end'])
            symbols = set(params['symbols'].split(','))
            if params.get('timeframe') == '1Min' or 'trades' in path:
                if not symbols <= self.scoring_symbols:
                    raise ReplayError('SCORING_SYMBOL_NOT_FORECAST')
            elif params.get('timeframe') != '1Day':
                raise ReplayError('TIMEFRAME_NOT_GRANTED')
            if path == '/v2/stocks/bars' and (params.get('feed') != 'sip' or params.get('adjustment') not in ('raw', 'all')):
                raise ReplayError('RAW_OR_ADJUSTED_SIP_REQUIRED')
        if origin == md.ALPACA and path == '/v1beta1/news' and params.get('symbols'):
            md.guard(params['symbols'].split(','), params['start'], params['end'])
        from urllib.parse import urlencode
        url = origin + path + ('?' + urlencode(sorted(params.items())) if params else '')
        key = hashlib.sha256(url.encode()).hexdigest()
        file = self.cache / key[:2] / (key + '.json.gz')
        refresh = self.refresh or self.refresh_news and path == '/v1beta1/news'
        if self.retry_missing and key not in self.retried_keys and file.exists() and ('bars' in path or 'trades' in path):
            prior = json.loads(gzip.decompress(file.read_bytes()))['body']
            rows = prior.get('trades' if 'trades' in path else 'bars', {}) or {}
            if any(not rows.get(s) for s in params.get('symbols', '').split(',')):
                self.retried_keys.add(key)  # Durable job intent prevents reruns after a crash.
                backup = self.cache.parent / 'missing-label-originals' / key[:2] / (hashlib.sha256(file.read_bytes()).hexdigest() + '.json.gz')
                backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not backup.exists():
                    backup.write_bytes(file.read_bytes())
                self._fetch(name, origin, url, file)
        if not file.exists() or (refresh and key not in self.index):
            self._fetch(name, origin, url, file)
        entry = json.loads(gzip.decompress(file.read_bytes()))
        self.index[key] = {'origin': origin, 'path': path, 'params': params, 'cache_key': key,
                           'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
        return entry['body'], {'url_sha256': key, 'received_at': entry['received_at'],
                               'body_sha256': entry['body_sha256'], 'cached': True}

    def _fetch(self, name, origin, url, file):
        host = origin.split('//')[1]
        # Keep the older daily host ceiling as an additional bound even for collection-v1.
        limit = self.grants.daily_limit(name, origin) or (1500 if 'sec.gov' in host else 3000)
        delay = 5
        limiter = 'sec' if origin in (SEC_WWW, SEC_DATA) else 'alpaca'
        for attempt in range(6):
            wait = SPACING[origin] - (time.monotonic() - self._last.get(limiter, 0))
            if wait > 0:
                self.sleep(wait)
            self.grants.check()
            if self.before is not None and self.clock() >= self.before:
                raise ReplayError('LIVE_COLLECTION_DEADLINE_PASSED')
            self.budget.reserve(host, limit, self.clock())
            self._last[limiter] = time.monotonic()
            try:
                raw = self.getter(url, self._headers(origin))
                break
            except ReplayError as error:
                # Stop on every 4xx other than 429; back off exponentially on 429/5xx/network.
                retry = error.code in ('HTTP_429', 'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_504', 'HTTP_UNAVAILABLE')
                if not retry or attempt == 5:
                    raise
                self.sleep(delay)
                delay *= 2
        self.requests += 1
        received = iso(self.clock())
        if self.before is not None and instant(received) >= self.before:
            raise ReplayError('LIVE_COLLECTION_DEADLINE_PASSED')
        body = json.loads(raw)
        file.parent.mkdir(exist_ok=True, mode=0o700)
        temp = file.with_suffix('.tmp')
        temp.write_bytes(gzip.compress(json.dumps({'received_at': received, 'body_sha256': hashlib.sha256(raw).hexdigest(),
                                                   'body': body}).encode()))
        os.replace(temp, file)

    def save(self, index_file, *, merge=True):
        index_file = Path(index_file)
        objects = {}
        if merge and index_file.is_file():
            objects = {o['cache_key']: o for o in json.loads(index_file.read_text())['objects']}
        objects.update(self.index)
        value = {'schema': 'replay-vendor-request-index-v1', 'objects': [objects[k] for k in sorted(objects)]}
        value['sec_availability'] = spec.spec()['unblock_2_amendment']['sec_policy']
        atomic_json(index_file, value)
        return len(objects)


def news_rows(collector, start, end):
    """Preserve captured news, including bad clocks, until the shared input filter."""
    params = {'start': start, 'end': end, 'limit': 50, 'sort': 'asc', 'include_content': 'false'}
    return [row for body, _ in md.paginate(collector, '/v1beta1/news', params, 'news')
            for row in body.get('news', [])]


def news_symbols(rows, start, end):
    """Symbols that could become c/d candidates: 1-2 tagged equities, not fixed names or known ETFs."""
    out = set()
    from .inputs import news_at
    for row in news_at(rows, end):
        created = instant(row['created_at'])
        if not start <= created < end or not 1 <= len(row.get('symbols') or []) <= 2:
            continue
        out.update(s for s in row['symbols'] if sn.TICKER.fullmatch(s) and s not in sn.NOT_SMALL_MID)
    return out


def equities(collector, symbols):
    records, fingerprints = {}, {}
    for status in ('active', 'inactive'):
        body, _ = collector.get(PAPER, '/v2/assets', {'status': status, 'asset_class': 'us_equity'})
        for row in body:
            symbol = row['symbol']
            fingerprints.setdefault(symbol, set()).add(sha256_canonical({k:v for k,v in row.items() if k != 'id'}))
            records.setdefault(symbol, row)
    return {s for s in symbols if s in records and len(fingerprints[s]) == 1 and records[s].get('class') == 'us_equity'
            and not re.search(r'\bETF\b', records[s].get('name', ''), re.I)}


def sec_submissions(collector, symbols):
    body, _ = collector.get(SEC_WWW, '/files/company_tickers.json')
    ciks = {row['ticker'].upper(): int(row['cik_str']) for row in body.values()}
    done = set()
    for symbol in sorted(symbols):
        cik = ciks.get(symbol.upper().replace('.', '-'))
        if cik is not None and cik not in done:
            done.add(cik)
            collector.get(SEC_DATA, f'/submissions/CIK{cik:010d}.json')
    return len(done)


def sec_or_unavailable(collector, symbols, root):
    """Revision 10 operator policy: no collection and no retry, including cached SEC."""
    atomic_json(Path(root) / 'sec-availability.json', spec.spec()['unblock_2_amendment']['sec_policy'])
    return 0


def historical_inputs(collector, *, news_from='2026-06-26T00:00:00Z', news_to='2026-10-09T12:00:00Z',
                      bars_to='2026-10-17T00:00:00Z'):
    """Prediction inputs for every dev/test session: one paginated all-symbol news query,
    active+inactive assets and raw SIP daily bars from 2025-01-01 for every possible candidate."""
    rows = news_rows(collector, news_from, news_to)
    names = equities(collector, news_symbols(rows, instant(news_from), instant(news_to)))
    fixed = list(sn.STOCKS + sn.ETFS + sn.BENCHMARKS)
    md.stock_bars(collector, sorted(set(fixed) | names), BARS_FROM, bars_to)
    md.crypto_bars(collector, list(sn.CRYPTO), BARS_FROM, bars_to)
    return {'news_rows': len(rows), 'candidate_symbols': len(names)}


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_seal_forward.collect')
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'))
    parser.add_argument('--collection-grant', default=os.environ.get('REPLAY_COLLECTION_GRANT'))
    parser.add_argument('--authorization', default=os.environ.get('REPLAY_AUTHORIZATION'))
    parser.add_argument('--alpaca-env', default=os.environ.get('REPLAY_ALPACA_ENV'))
    parser.add_argument('--sec-agent-file', default=os.environ.get('REPLAY_SEC_AGENT_FILE'))
    parser.add_argument('--expected-commit')
    parser.add_argument('--missing-plan', type=Path)
    parser.add_argument('command', choices=('historical', 'historical-sec', 'historical-outcomes', 'historical-fixed-outcomes',
                                           'retry-missing-labels', 'live', 'live-outcomes'))
    args = parser.parse_args(argv)
    from . import spec
    root = Path(args.data_root).expanduser().resolve()
    if root == spec.ROOT or spec.ROOT in root.parents:
        raise SystemExit('DATA_ROOT_MUST_BE_OUTSIDE_GIT')
    grants = Grants(args.collection_grant, args.authorization)
    if args.expected_commit:
        from . import experiment
        experiment.committed()
        if experiment.head() != args.expected_commit:
            raise SystemExit('INSTALLED_SHA_MISMATCH')
    agent = None
    if args.sec_agent_file:
        # The contact stays in a private file and is never logged or committed.
        agent = json.loads(Path(args.sec_agent_file).read_text())['scope'][SEC_WWW]['user_agent']
    result = {'command': args.command}
    try:
        with owned(root / 'collector'):
            if args.command.startswith('historical') or args.command == 'retry-missing-labels':
                collector = Collector(grants, Budget(root / 'collector'), root / 'cache',
                                      alpaca_env=args.alpaca_env, sec_user_agent=agent)
                index = root / 'archive_index.json'
                try:
                    if args.command == 'historical':
                        result.update(historical_inputs(collector))
                    elif args.command == 'historical-sec':
                        result['ciks'] = sec_or_unavailable(collector, (), root)
                    elif args.command == 'retry-missing-labels':
                        from .collect_plan import retry_missing_labels
                        result.update(retry_missing_labels(root, collector, args.missing_plan))
                    else:
                        from .collect_plan import historical_outcomes
                        result.update(historical_outcomes(root, collector, include_news=args.command != 'historical-fixed-outcomes'))
                finally:
                    result['index_objects'] = collector.save(index)
            else:
                from .collect_plan import live, live_outcomes
                result.update((live if args.command == 'live' else live_outcomes)(root, grants, args.alpaca_env, agent))
            result['status'] = 'DONE'
    except ReplayError as error:
        result.update(status='BLOCKED', error=error.code)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'DONE' else 2


if __name__ == '__main__':
    sys.exit(main())
