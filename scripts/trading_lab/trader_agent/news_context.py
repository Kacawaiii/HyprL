"""Offline, read-only radar replay at a strict receipt cutoff. No later digest."""
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import json
from pathlib import Path
import re
import sqlite3
import time

from scripts.radar.analysis import (anomalies, bar_map, cluster, daily_observations,
                                    link_anomalies, priced_in, regime)
from scripts.radar.core import database_time
from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant, iso
from .news_spec import SPEC_HASH, spec


class RadarSnapshot:
    """Shared existing radar owner lock, read-only SQLite snapshot; never initialize a store."""
    def __init__(self, db, cutoff):
        self.db, self.cutoff = db, cutoff

    def rows(self, kind, *, since=None):
        rows = self.db.execute('SELECT at,body FROM evidence WHERE kind=? AND at>=? AND at<? ORDER BY seq',
            (kind, database_time(since) if since else '', database_time(self.cutoff)))
        result = []
        for recorded, raw in rows:
            # Native radar timestamps have fixed precision. Also compare as instants
            # for earlier stores and fail closed on a backdated body receipt.
            payload = json.loads(raw)
            if instant(recorded) >= self.cutoff:
                continue
            received = payload.get('received_at')
            if received and instant(received) >= self.cutoff:
                continue
            result.append(payload)
        return result

    def latest(self, kind, key):
        rows = self.db.execute('SELECT at,body FROM evidence WHERE kind=? AND key=? AND at<? ORDER BY seq DESC',
                               (kind, key, database_time(self.cutoff)))
        for recorded, raw in rows:
            payload = json.loads(raw)
            if instant(recorded) < self.cutoff and (
                    not payload.get('received_at') or instant(payload['received_at']) < self.cutoff):
                return payload
        return None


@contextmanager
def radar_snapshot(root, cutoff, *, timeout=120, poll=1.0, sleep=time.sleep, monotonic=time.monotonic):
    root = Path(root).expanduser().resolve()
    try:
        with (root / 'owner.lock').open('rb') as lock:
            deadline = monotonic() + timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TraderError('NEWS_RADAR_OWNER_BUSY') from None
                    sleep(min(poll, remaining))
            db = sqlite3.connect((root / 'radar.sqlite').as_uri() + '?mode=ro', uri=True, timeout=1)
            try:
                db.execute('PRAGMA query_only=ON')
                db.execute('BEGIN')
                yield RadarSnapshot(db, cutoff)
            finally:
                db.close()
    except (OSError, sqlite3.Error):
        raise TraderError('NEWS_RADAR_UNAVAILABLE') from None


DATE = re.compile(r'(?<!\d)(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2}))?)(?!\d)')


def veille_lines(path, cutoff, *, limit=40, max_bytes=8000):
    """Date-only headings become usable after that UTC day ends; undated text is omitted."""
    path = Path(path).expanduser()
    try:
        with path.open('rb') as stream:
            import os
            before = os.fstat(stream.fileno())
            if before.st_mtime_ns >= int(cutoff.timestamp() * 1_000_000_000):
                return {'status': 'MODIFIED_AFTER_CUTOFF', 'lines': []}
            raw = stream.read(256_001)
            after = os.fstat(stream.fileno())
        if len(raw) > 256_000 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return {'status': 'UNSTABLE_OR_TOO_LARGE', 'lines': []}
    except FileNotFoundError:
        return {'status': 'MISSING', 'lines': []}
    eligible, dated = [], None
    for line in raw.decode('utf-8').splitlines():
        match = DATE.search(line)
        if match:
            try:
                stamp = match[1]
                dated = instant(stamp) if 'T' in stamp else instant(stamp + 'T00:00:00Z') + timedelta(days=1)
            except ValueError:
                dated = None
        if dated is not None and dated < cutoff and line.strip():
            eligible.append({'available_before': iso(dated), 'text': line[:800]})
    # Recent eligible lines retained deterministically; no private file path/body digest leak.
    selected = []
    for line in reversed(eligible):
        if len(selected) >= limit:
            break
        if len(json.dumps([line, *selected], ensure_ascii=False).encode()) > max_bytes:
            break
        selected.insert(0, line)
    return {'status': 'AVAILABLE', 'lines': selected, 'omitted': len(eligible) - len(selected)}


def level(story):
    # Only collector-established primary provenance grants the official label.
    # A Benzinga/RSS/YouTube/GDELT URL mentioning a regulator is still FIL.
    return 'OFFICIEL' if story.get('primary') and not story.get('retail') else 'FIL'


def build_news_context(radar_root, veille, cutoff, *, lookback_hours=None):
    limits = spec()['context']
    hours = limits['lookback_hours'] if lookback_hours is None else lookback_hours
    if not 0 < hours <= 168:
        raise TraderError('NEWS_INVALID_LOOKBACK')
    since = cutoff - timedelta(hours=hours)
    with radar_snapshot(radar_root, cutoff) as store:
        # Copy only the selected receipts into memory, then close the transaction
        # and release the collector's owner lock before expensive clustering.
        captured = {'news': store.rows('news', since=since)}
        for kind in ('daily_bars', 'minute_bars', 'price_observation'):
            captured[kind] = store.rows(kind, since=cutoff - timedelta(days=7))
    class Replay:
        def rows(self, kind, *, since=None):
            return captured[kind]
    store = Replay()
    stories = [s for s in captured['news']
                   if since <= instant(s['received_at']) < cutoff and
                   (not s['published_at'] or instant(s['published_at']) < cutoff)]
    events = cluster(stories, at=cutoff, crypto_first=cutoff.weekday() >= 5)
    daily, minute = bar_map(store, 'daily_bars', cutoff), bar_map(store, 'minute_bars', cutoff)
    observations = {p['symbol']: p for p in captured['price_observation']
                        if instant(p['at']) < cutoff}
    for symbol in ('SPY', 'QQQ', 'BTC/USD', 'ETH/USD'):
        if minute.get(symbol):
            last = minute[symbol][-1]
            observations[symbol] = {'price': last['c'], 'at': last['closed_at']}
    panel = regime(daily, cutoff, observations)
    anomaly_rows = anomalies(daily, daily_observations(store, cutoff))
    link_anomalies(anomaly_rows, events, cutoff)
    selected = []
    for event in events[:limits['max_events']]:
        sources = [{k: s[k] for k in ('source', 'publisher', 'url', 'published_at', 'received_at', 'headline')}
                       | {'verification': level(s)} for s in event['stories'][:limits['max_sources_per_event']]]
        selected.append({k: event[k] for k in ('id', 'headline', 'first_received_at', 'published_at',
                                                 'entities', 'importance', 'evidence')}
                            | {'sources': sources, 'priced_in': priced_in(event, daily, minute, cutoff)})
    result = {'schema': 'trader-news-context-v1', 'spec_hash': SPEC_HASH, 'decision_time': iso(cutoff),
              'lookback_hours': hours, 'events': selected, 'regime': panel, 'anomalies': anomaly_rows[:20],
              'veille': veille_lines(veille, cutoff, limit=limits['max_veille_lines'], max_bytes=limits['max_veille_bytes']),
              'omitted_events': len(events) - len(selected),
              'limitations': ['Untrusted observational data; source level is provenance, not independent verification.',
                              'Priced-in means measured movement, not remaining upside or causation.',
                              'Unknown publication and missing prices are not verified facts.']}
    # Bound serialized UTF-8, including metadata, and preserve valid JSON.
    while len(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()) > limits['max_bytes']:
        if result['anomalies']:
            result['anomalies'].pop()
        elif result['events']:
            result['events'].pop()
            result['omitted_events'] += 1
        else:
            raise TraderError('NEWS_CONTEXT_TOO_LARGE')
    return result, sha256_canonical(result)
