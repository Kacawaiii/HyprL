"""Private append-only evidence and fail-closed, durable capture budgets."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote
from urllib.request import Request, HTTPRedirectHandler, build_opener
from urllib.error import HTTPError, URLError
import fcntl
import json
import os
import sqlite3
import tempfile

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]


class RadarError(Exception):
    """Only fixed codes may reach console output; never response bodies or keys."""


def now():
    return datetime.now(UTC)


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timezone required')
    return result.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def database_time(value):
    # Fixed precision makes SQLite range comparisons chronological, including
    # the boundary between whole-second and fractional-second receipts.
    return value.astimezone(UTC).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(value.encode()).hexdigest()


def canonical_url(url, *, allow_http=False):
    if not isinstance(url, str) or any(ord(c) < 32 for c in url):
        raise RadarError('UNSAFE_URL')
    try:
        p = urlsplit(url)
        port = p.port
    except ValueError:
        raise RadarError('UNSAFE_URL') from None
    if p.scheme not in (('https', 'http') if allow_http else ('https',)) or not p.hostname or p.username or p.password or port not in (None, 443, 80 if allow_http else 443):
        raise RadarError('UNSAFE_URL')
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith('utm_') and k not in ('fbclid', 'gclid')]
    return urlunsplit((p.scheme, p.netloc.lower(), quote(p.path, safe='/%:@!$&*+=,;~-._'), urlencode(sorted(query)), ''))


def private_dir(path, *, secure_existing=True):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise RadarError('PRIVATE_OUTPUT_INSIDE_GIT')
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not existed or secure_existing:
        path.chmod(0o700)
    return path


def write_private(path, text):
    path = Path(path).expanduser()
    # The task authorizes radar-latest.md, not changing Claude book directory
    # permissions or other files. Never chmod an existing publication directory.
    folder = private_dir(path.parent, secure_existing=False)
    path = folder / path.name
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.radar-', dir=folder, delete=False) as stream:
            temp = Path(stream.name)
            os.chmod(temp, 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp and temp.exists():
            temp.unlink()


class Store:
    def __init__(self, root, *, clock=now):
        self.root, self.clock = private_dir(root), clock
        self.path = self.root / 'radar.sqlite'
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS evidence (
                seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, key TEXT NOT NULL,
                at TEXT NOT NULL, body TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS evidence_lookup ON evidence(kind,key,seq);
              CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON evidence
                BEGIN SELECT RAISE(ABORT,'append only'); END;
              CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON evidence
                BEGIN SELECT RAISE(ABORT,'append only'); END;
              CREATE TABLE IF NOT EXISTS dispatch (
                seq INTEGER PRIMARY KEY, day TEXT NOT NULL, group_name TEXT NOT NULL,
                source TEXT NOT NULL, at TEXT NOT NULL, grant_hash TEXT NOT NULL);
              CREATE TRIGGER IF NOT EXISTS dispatch_update BEFORE UPDATE ON dispatch
                BEGIN SELECT RAISE(ABORT,'append only'); END;
              CREATE TRIGGER IF NOT EXISTS dispatch_delete BEFORE DELETE ON dispatch
                BEGIN SELECT RAISE(ABORT,'append only'); END;
            ''')
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def owner(self):
        with (self.root / 'owner.lock').open('a') as stream:
            os.chmod(stream.name, 0o600)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RadarError('OWNER_BUSY') from None
            yield

    def append(self, kind, key, body, *, at=None):
        with self.connect() as db:
            db.execute('INSERT INTO evidence(kind,key,at,body) VALUES(?,?,?,?)',
                       (kind, key, database_time(at or self.clock()), json.dumps(body, ensure_ascii=False, allow_nan=False)))

    def latest(self, kind, key):
        with self.connect() as db:
            row = db.execute('SELECT body FROM evidence WHERE kind=? AND key=? ORDER BY seq DESC LIMIT 1', (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def rows(self, kind, *, since=None, limit=None):
        with self.connect() as db:
            query = 'SELECT body FROM evidence WHERE kind=? AND at>=? ORDER BY seq'
            params = [kind, database_time(since) if since else '']
            if limit is not None:
                query += ' DESC LIMIT ?'
                params.append(limit)
            rows = db.execute(query, params).fetchall()
            if limit is not None:
                rows.reverse()
        return [json.loads(row[0]) for row in rows]

    def reserve(self, grant, group, source, maximum, spacing=0):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            at = self.clock()
            grant.check(at)
            if group == 'gdelt' and '11:45' <= at.strftime('%H:%M') < '12:30':
                raise RadarError('GDELT_TRADER_WINDOW')
            maximum = min(maximum, grant.payload['budgets'].get(group, 0))
            count = db.execute('SELECT count(*) FROM dispatch WHERE day=? AND group_name=?',
                               (at.date().isoformat(), group)).fetchone()[0]
            if count >= maximum:
                raise RadarError('DAILY_BUDGET')
            # GDELT shares spacing across queries; feeds have per-feed spacing.
            row = db.execute('SELECT at FROM dispatch WHERE group_name=? AND (? OR source=?) ORDER BY seq DESC LIMIT 1',
                             (group, group == 'gdelt', source)).fetchone()
            spacing = max(spacing, grant.payload.get('spacing_seconds', {}).get(group, 0))
            if row and (at - instant(row[0])).total_seconds() < spacing:
                raise RadarError('POLL_SPACING')
            db.execute('INSERT INTO dispatch(day,group_name,source,at,grant_hash) VALUES(?,?,?,?,?)',
                       (at.date().isoformat(), group, source, iso(at), grant.identity))

    def counts(self):
        with self.connect() as db:
            return dict(db.execute('SELECT group_name,count(*) FROM dispatch WHERE day=? GROUP BY group_name',
                                   (self.clock().date().isoformat(),)))


class Authorization:
    """An operator-created file is required, including for public RSS and Claude."""
    def __init__(self, path):
        try:
            self.payload = json.loads(Path(path).expanduser().read_text())
            p = self.payload
            if (p['authorization'] != 'news-radar-v1' or not p['operator_signed']
                    or instant(p['not_after']) <= instant(p['granted_at'])
                    or not isinstance(p['scope'], dict) or not isinstance(p['budgets'], dict)
                    or any(type(v) is not int or v < 0 for v in p['budgets'].values())):
                raise ValueError()
            self.identity = digest(p)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise RadarError('AUTHORIZATION_MISSING_OR_INVALID') from None

    def check(self, at):
        if not instant(self.payload['granted_at']) <= at < instant(self.payload['not_after']):
            raise RadarError('AUTHORIZATION_EXPIRED_OR_NOT_STARTED')

    def permit(self, url):
        p = urlsplit(canonical_url(url))
        host = p.hostname
        # Live SEC and Fed adapters are intentionally absent from v1.
        if any(host == h or host.endswith('.' + h) for h in ('sec.gov', 'federalreserve.gov')):
            raise RadarError('OFFICIAL_SOURCE_DISABLED')
        rule = self.payload['scope'].get('https://' + p.netloc)
        if not isinstance(rule, dict) or rule.get('methods') != ['GET']:
            raise RadarError('SOURCE_OUTSIDE_GRANT')
        exact = rule.get('paths', [])
        prefixes = rule.get('path_prefixes', [])
        if not isinstance(exact, list) or not isinstance(prefixes, list):
            raise RadarError('PATH_OUTSIDE_GRANT')
        if p.path not in exact and not any(p.path.startswith(x) for x in prefixes if isinstance(x, str) and x.startswith('/')):
            raise RadarError('PATH_OUTSIDE_GRANT')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Return the redirect status/headers through HTTPError, without following.
        # Only the bounded channel resolver may authorize a new metadata URL.
        return None


def transport(url, headers):
    try:
        with build_opener(NoRedirect()).open(Request(url, headers=headers, method='GET'), timeout=20) as response:
            body = response.read(3_000_001)
            if len(body) > 3_000_000:
                raise RadarError('RESPONSE_TOO_LARGE')
            return response.status, dict(response.headers), body
    except HTTPError as error:
        return error.code, dict(error.headers), b''
    except (OSError, URLError):
        raise RadarError('HTTP_UNAVAILABLE') from None


class Client:
    def __init__(self, store, grant, *, send=transport, credentials=None):
        self.store, self.grant, self.send, self.credentials = store, grant, send, credentials

    def get(self, url, group, source, *, headers=None):
        self.grant.check(self.store.clock())
        self.grant.permit(url)
        cooldown = self.store.latest('cooldown', group)
        if cooldown and self.store.clock() < instant(cooldown['until']):
            raise RadarError('SOURCE_COOLDOWN')
        maximum = {'alpaca': 400, 'gdelt': 60, 'rss': 300, 'youtube': 400, 'market': 24}[group]
        spacing = {'rss': 900, 'youtube': 900, 'gdelt': 15}.get(group, 0)
        self.store.reserve(self.grant, group, source, maximum, spacing)
        request_headers = {'User-Agent': 'HyprL-NewsRadar/1.0', 'Accept': 'application/json, application/rss+xml, application/atom+xml, text/xml'}
        request_headers.update(headers or {})
        if urlsplit(url).hostname == 'data.alpaca.markets':
            request_headers.update(self.alpaca_headers())
        status, response_headers, body = self.send(url, request_headers)
        received = self.store.clock()  # after receipt; separate from publisher time
        self.store.append('http', source, {'url': url, 'status': status, 'received_at': iso(received), 'body_hash': sha256(body).hexdigest()})
        if status == 429:
            from email.utils import parsedate_to_datetime
            retry = next((v for k, v in response_headers.items() if k.lower() == 'retry-after'), None)
            failures = min(6, (cooldown or {}).get('failures', 0) + 1) if group == 'gdelt' else 1
            until = received + timedelta(hours=min(24, 2 ** (failures - 1)))
            try:
                until = max(until, received + timedelta(seconds=int(retry)))
            except (TypeError, ValueError):
                try:
                    until = max(until, parsedate_to_datetime(retry).astimezone(UTC))
                except (TypeError, ValueError, AttributeError):
                    pass
            self.store.append('cooldown', group, {'until': iso(until), 'status': status, 'failures': failures})
            raise RadarError('HTTP_429_STOP_SOURCE')
        if group == 'gdelt' and status == 200 and cooldown and cooldown.get('failures'):
            self.store.append('cooldown', group, {'until': iso(received), 'status': status, 'failures': 0})
        return status, response_headers, body, received

    def alpaca_headers(self):
        try:
            values = {}
            for line in Path(self.credentials).expanduser().read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    key, value = line.removeprefix('export ').split('=', 1)
                    values[key.strip()] = value.strip().strip('\"\'')
            key = values.get('APCA_API_KEY_ID') or values.get('ALPACA_API_KEY')
            secret = values.get('APCA_API_SECRET_KEY') or values.get('ALPACA_SECRET_KEY')
            if not key or not secret:
                raise ValueError()
            return {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
        except (OSError, TypeError, ValueError):
            raise RadarError('CREDENTIALS_MISSING_OR_INVALID') from None

    def json(self, url, group, source):
        status, _, raw, received = self.get(url, group, source)
        if status != 200:
            raise RadarError('HTTP_' + str(status))
        try:
            return json.loads(raw), received
        except (ValueError, UnicodeError):
            raise RadarError('INVALID_JSON') from None
