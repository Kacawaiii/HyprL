"""Operator grant, per-UTC-day budgets and the GET-only cached transport. No bodies or credentials enter Git."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

UTC = timezone.utc
GRANT_SCHEMA = 'replay-bench-authorization-v1'
# Path templates of the grant, as regular expressions over the request path.
PATHS = {
    'https://data.alpaca.markets': [r'/v1beta1/news', r'/v2/stocks/bars', r'/v2/stocks/[A-Z][A-Z0-9.]{0,9}/bars',
                                    r'/v2/stocks/trades', r'/v1beta3/crypto/us/bars'],
    'https://www.sec.gov': [r'/files/company_tickers\.json'],
    'https://data.sec.gov': [r'/submissions/CIK\d{10}\.json'],
}
MODEL_KEYS = {'opus': 'opus_calls_per_day', 'sonnet': 'sonnet_calls_per_day'}


class ReplayError(RuntimeError):
    def __init__(self, code, evidence=None):
        self.code, self.evidence = code, evidence
        super().__init__(code)


def now():
    return datetime.now(UTC)


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ReplayError('NAIVE_TIME')
    return result.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


class Grant:
    """The operator authorization file, rechecked at every dispatch."""

    def __init__(self, path, *, clock=now):
        self.path = Path(path)
        self.payload = json.loads(self.path.read_text())
        self.clock = clock
        p = self.payload
        if p.get('schema') != GRANT_SCHEMA or p.get('operator_signed') is not True:
            raise ReplayError('GRANT_INVALID')
        if p.get('models', {}).get('gpt') not in (None, 'none') and 'none' not in str(p['models'].get('gpt')):
            raise ReplayError('GRANT_GPT_NOT_RETIRED')
        for origin in PATHS:
            if p['scope'].get(origin, {}).get('methods') != ['GET']:
                raise ReplayError('GRANT_SCOPE_INVALID', origin)
        self.check()

    def check(self):
        if json.loads(self.path.read_text()) != self.payload:
            raise ReplayError('GRANT_CHANGED')
        at = self.clock()
        if not instant(self.payload['granted_at']) <= at < instant(self.payload['not_after']):
            raise ReplayError('GRANT_EXPIRED_OR_NOT_STARTED')

    def origin_limit(self, origin):
        return int(self.payload['scope'][origin]['max_requests_per_day'])

    def model_limit(self, family):
        return int(self.payload['models']['claude_cli_subscription'][MODEL_KEYS[family]])


class Budget:
    """Durable per-UTC-day counters, reserved before any dispatch (a crash never refunds a call)."""

    def __init__(self, root):
        self.path = Path(root) / 'budget.json'
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def _locked(self):
        with open(self.path.with_suffix('.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() else {}
                yield data
                from .storage import atomic_json
                atomic_json(self.path, data)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, key, limit, at):
        day = iso(at)[:10]
        with self._locked() as data:
            used = data.setdefault(day, {}).get(key, 0)
            if used >= limit:
                raise ReplayError('BUDGET_EXHAUSTED', {'key': key, 'day': day, 'limit': limit})
            data[day][key] = used + 1
            return used + 1

    def reserve_model(self, family, limit, at, night_limit=150):
        """Atomically reserve both model and shared night allowance, including legacy calls."""
        with self._locked() as data:
            day = data.setdefault(iso(at)[:10], {})
            if day.get('stopped'):
                raise ReplayError('MODEL_USAGE_LIMIT')
            key = 'model:' + family
            if day.get(key, 0) >= limit or sum(v for k, v in day.items() if k.startswith('model:')) >= night_limit:
                raise ReplayError('BUDGET_EXHAUSTED')
            day[key] = day.get(key, 0) + 1

    def used(self, key, at):
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        return data.get(iso(at)[:10], {}).get(key, 0)

    def mark_stopped(self, reason, at):
        with self._locked() as data:
            data.setdefault(iso(at)[:10], {})['stopped'] = reason

    def stopped(self, at):
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        return data.get(iso(at)[:10], {}).get('stopped')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReplayError('REDIRECT_REFUSED')


def load_env(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def http_get(url, headers):
    try:
        with build_opener(_NoRedirect()).open(Request(url, headers=headers, method='GET'), timeout=40) as response:
            raw = response.read(30 * 1024 * 1024 + 1)
    except HTTPError as error:
        raise ReplayError('HTTP_' + str(error.code)) from None
    except (URLError, OSError):
        raise ReplayError('HTTP_UNAVAILABLE') from None
    if len(raw) > 30 * 1024 * 1024:
        raise ReplayError('RESPONSE_TOO_LARGE')
    return raw


class Transport:
    """Grant-bound GET with a private response cache. A cache hit costs no request and keeps its first received_at."""

    def __init__(self, grant, budget, cache_root, *, alpaca_env=None, sec_user_agent=None, getter=http_get,
                 sleep=time.sleep, offline=False):
        self.grant, self.budget, self.getter, self.sleep, self.offline = grant, budget, getter, sleep, offline
        self.cache = Path(cache_root)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._alpaca = load_env(alpaca_env) if alpaca_env else None
        self.sec_user_agent = sec_user_agent
        self._last = {}

    def _headers(self, origin):
        if origin == 'https://data.alpaca.markets':
            if not self._alpaca:
                raise ReplayError('ALPACA_KEYS_REQUIRED')
            return {'APCA-API-KEY-ID': self._alpaca['APCA_API_KEY_ID'],
                    'APCA-API-SECRET-KEY': self._alpaca['APCA_API_SECRET_KEY'], 'Accept': 'application/json'}
        if not self.sec_user_agent:
            raise ReplayError('SEC_USER_AGENT_REQUIRED')
        return {'User-Agent': self.sec_user_agent, 'Accept': 'application/json'}

    def get(self, origin, path, params=None):
        if origin not in PATHS or not any(re.fullmatch(p, path) for p in PATHS[origin]):
            raise ReplayError('PATH_REFUSED', path)
        if origin == 'https://data.alpaca.markets' and ('bars' in path or 'trades' in path):
            from .marketdata import guard
            query = params or {}
            symbols = query.get('symbols', '').split(',')
            if not any(symbols) and path.startswith('/v2/stocks/') and path.endswith('/bars'):
                symbols = [path.split('/')[3]]
            if not query.get('start') or not query.get('end') or not any(symbols):
                raise ReplayError('BOUNDED_PRICE_QUERY_REQUIRED')
            guard(symbols, query['start'], query['end'])
        # Check the actual grant, not just the transport's hardcoded maximum scope.
        granted = self.grant.payload['scope'].get(origin, {}).get('paths')
        if granted is not None:
            patterns = [re.escape(p).replace(r'\{symbol\}', r'[A-Z][A-Z0-9.]{0,9}')
                        .replace(r'\{cik\}', r'\d{10}') for p in granted]
            if not any(re.fullmatch(p, path) for p in patterns):
                raise ReplayError('PATH_NOT_GRANTED')
        url = origin + path + ('?' + urlencode(sorted((params or {}).items())) if params else '')
        key = hashlib.sha256(url.encode()).hexdigest()
        file = self.cache / key[:2] / (key + '.json.gz')
        allowed = getattr(self, 'allowed_cache', None)
        if allowed is not None:
            if key not in allowed or not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != allowed[key]:
                raise ReplayError('PREDICTION_CACHE_NOT_SEALED')
        if file.exists():
            entry = json.loads(gzip.decompress(file.read_bytes()))
            return entry['body'], {'url_sha256': key, 'received_at': entry['received_at'],
                                   'body_sha256': entry['body_sha256'], 'cached': True}
        if self.offline:
            raise ReplayError('OFFLINE_CACHE_MISS', url)
        self.grant.check()
        at = now()
        self.budget.reserve(urlsplit(origin).hostname, self.grant.origin_limit(origin), at)
        spacing = 0.35 if 'alpaca' in origin else 0.15
        wait = spacing - (time.monotonic() - self._last.get(origin, 0))
        if wait > 0:
            self.sleep(wait)
        self._last[origin] = time.monotonic()
        for attempt in range(3):
            self.grant.check()
            try:
                raw = self.getter(url, self._headers(origin))
                break
            except ReplayError as error:
                if error.code not in ('HTTP_429', 'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_UNAVAILABLE') or attempt == 2:
                    raise
                self.sleep(5 * (attempt + 1))
                self.budget.reserve(urlsplit(origin).hostname, self.grant.origin_limit(origin), now())
        received = iso(now())
        body = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        file.parent.mkdir(exist_ok=True, mode=0o700)
        tmp = file.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(json.dumps({'received_at': received, 'body_sha256': digest,
                                                  'body': body}).encode()))
        os.replace(tmp, file)
        return body, {'url_sha256': key, 'received_at': received, 'body_sha256': digest, 'cached': False}
