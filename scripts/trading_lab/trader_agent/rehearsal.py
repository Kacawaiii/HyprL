"""Daily infrastructure rehearsal. Production research and paper journals are readers only.

Only private monitoring state, source dispatches and a separate tiny inference
budget are writable. Reports contain codes, counts, suffixes and digests.
"""
from contextlib import contextmanager
from datetime import timedelta
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import sqlite3
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import Authorization, ROLES, TraderError, instant, iso, now, private_root, strict_json
from .data import PublicData, NoRedirect, calendar_session, equity_calendar, price_summary, protected
from .scoring import rows
from .ledger import Ledger


class MonitorState:
    def __init__(self, root):
        self.root = private_root(root)
        self.path = self.root / 'monitor.sqlite'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS attempts (
                    day TEXT, slot TEXT, body TEXT, PRIMARY KEY(day,slot));
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY, state TEXT);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY, body TEXT);
                CREATE TABLE IF NOT EXISTS acknowledgements (
                    identity TEXT PRIMARY KEY, code TEXT, reason TEXT, disposition TEXT, at TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def owner(self):
        with (self.root / 'owner.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TraderError('REHEARSAL_OWNER_BUSY') from None
            yield

    def get(self, day, slot):
        with self.connect() as db:
            row = db.execute('SELECT body FROM attempts WHERE day=? AND slot=?', (day, slot)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, day, slot, result):
        with self.connect() as db:
            db.execute('INSERT INTO attempts VALUES(?,?,?) ON CONFLICT(day,slot) DO UPDATE SET body=excluded.body',
                       (day, slot, json.dumps(result)))

    def alert(self, key, message, notify):
        # Reserve before dispatch: an uncertain Telegram response is never blindly replayed.
        with self.connect() as db:
            inserted = db.execute('INSERT OR IGNORE INTO notifications VALUES(?,?)', (key, 'RESERVED')).rowcount
            if not inserted:
                return db.execute('SELECT state FROM notifications WHERE id=?', (key,)).fetchone()[0]
        try:
            delivered = notify(key, message) if notify else 'UNAVAILABLE'
        except Exception:
            delivered = 'FAILED'
        delivered = delivered if delivered in {'DELIVERED', 'FAILED', 'UNAVAILABLE'} else 'FAILED'
        with self.connect() as db:
            db.execute('UPDATE notifications SET state=? WHERE id=?', (delivered, key))
        return delivered


def telegram(env_path):
    """No subprocess/shell environment loading; URL and exceptions never reach logs."""
    def send(key, message):
        try:
            values = {}
            for line in Path(env_path).read_text().splitlines():
                line = line.strip().removeprefix('export ')
                if not line or line.startswith('#'):
                    continue
                name, sep, raw = line.partition('=')
                tokens = shlex.split(raw, comments=True)
                if not sep or len(tokens) != 1 or name.strip() in values:
                    return 'FAILED'
                values[name.strip()] = tokens[0]
            token, recipient = values['TELEGRAM_BOT_TOKEN'], values['ADMIN_TELEGRAM_ID']
            if not token or any(c in token for c in '/?#\\\r\n'):
                return 'FAILED'
            request = Request('https://api.telegram.org/bot' + token + '/sendMessage',
                data=json.dumps({'chat_id': recipient, 'text': message[:3000],
                                 'disable_web_page_preview': True}).encode(),
                headers={'Content-Type': 'application/json'}, method='POST')
            with build_opener(NoRedirect()).open(request, timeout=15) as response:
                answer = strict_json(response.read(65537).decode())
            return 'DELIVERED' if answer.get('ok') is True else 'FAILED'
        except Exception:
            return 'FAILED'
    return send


def error_code(error):
    # Exception messages can contain credential-bearing URLs, private bodies and paths.
    if isinstance(error, TraderError) and all(c.isupper() or c.isdigit() or c == '_' for c in error.code):
        return error.code
    return 'UNEXPECTED_' + type(error).__name__.upper()


def gdelt_unavailable(code):
    # Availability mirrors PARTIAL_CONTEXT in the run. Admission, query and
    # budget failures remain blockers, even when they occur in this step.
    return code in {'SOURCE_UNAVAILABLE', 'SOURCE_HTTP_429', 'SOURCE_HTTP_408',
                    'SOURCE_COOLDOWN', 'REHEARSAL_GDELT_STALE_OR_EMPTY'} or (
        code.startswith('SOURCE_HTTP_5') and code.removeprefix('SOURCE_HTTP_').isdigit())


def write_report(state, reports, day, run_slot="am"):
    reports = private_root(reports)
    prefix = 'pm-' if run_slot == 'pm' else ''
    slots = ('primary', 'retry') if run_slot == 'pm' else ('early', 'primary', 'retry')
    attempts = [state.get(day, prefix + slot) for slot in slots]
    attempts = [a for a in attempts if a]
    lines = ['STATUS: ' + ('DONE' if attempts and attempts[-1]['state'] == 'GREEN' else 'BLOCKED'),
             '', 'Daily full-chain rehearsal ' + day, '',
             'GET-only AI paper accounts; no production decision, intent, label or order writes.', '']
    for attempt in attempts:
        lines += [attempt['slot'] + ': ' + attempt['state'] + ' at ' + attempt['at'], '',
                  '| Step | Result | Evidence |', '| --- | --- | --- |']
        for step in attempt.get('steps', []):
            lines.append('| ' + step['step'] + ' | ' + step['state'] + ' | ' +
                         (step.get('error', '') + ' ' + json.dumps(step.get('evidence', {}), sort_keys=True)) + ' |')
        if attempt.get('retry_at'):
            lines += ['', 'Second rehearsal due: ' + attempt['retry_at']]
        lines += ['', 'Alerts: ' + json.dumps(attempt.get('alerts', [])), '']
    lines += ['Unproven: a GET-only probe cannot establish that the broker will accept or fill a future order.',
              'Next: inspect the next entry, exit and label checkpoints; failures alert independently.', '']
    target = reports / ('rehearsal-' + day + ('-pm' if run_slot == 'pm' else '') + '.md')
    temp = target.with_suffix('.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(target)
    return target


def rehearse(state, steps, *, clock=now, retry=False, early=False, notify=None, reports, run_slot="am"):
    if early and (retry or run_slot != 'am'):
        raise TraderError('REHEARSAL_EARLY_AM_ONLY')
    prefix = 'pm-' if run_slot == 'pm' else ''
    at, slot = clock(), prefix + ('early' if early else 'retry' if retry else 'primary')
    day = at.date().isoformat()
    with state.owner():
        previous = state.get(day, prefix + 'primary')
        if retry and (not previous or previous['state'] == 'GREEN'):
            return {'state': 'NO_RETRY_NEEDED'}
        if retry and previous.get('retry_at') and at < instant(previous['retry_at']):
            return {'state': 'RETRY_NOT_DUE'}
        if state.get(day, slot):
            return {'state': 'ALREADY_ATTEMPTED'}
        result = {'schema': 'trader-rehearsal-v1', 'at': iso(at), 'slot': slot,
                  'run_slot': run_slot, 'state': 'RUNNING', 'steps': [], 'alerts': [], 'retry_at': None}
        # Reserve the whole attempt before calling anything. OnFailure sees RUNNING after a crash.
        state.save(day, slot, result)
        for name, check in steps:
            try:
                evidence = check()
                step = {'step': name, 'state': 'GREEN', 'evidence': evidence}
            except Exception as error:
                code = error_code(error)
                degraded = name == 'gdelt' and gdelt_unavailable(code)
                step = {'step': name, 'state': 'DEGRADED' if degraded else 'BLOCKED', 'error': code}
                if degraded:
                    step['evidence'] = {'run_effect': 'PARTIAL_CONTEXT', **(
                        error.evidence if isinstance(error.evidence, dict) else {})}
                if isinstance(error, TraderError) and error.code == 'REHEARSAL_PRICE_FAILURE':
                    step['evidence'] = {'failures': error.evidence}
                if name == 'alpaca_data' and isinstance(error, TraderError) and isinstance(error.evidence, dict):
                    step['evidence'] = error.evidence
                result['alerts'].append({'step': name, 'delivery': state.alert(
                    day + ':' + slot + ':' + name,
                    'Trader rehearsal ' + day + ' (' + slot + ') ' +
                    ('DEGRADED / PARTIAL_CONTEXT: ' if degraded else 'failed: ') + name + ' / ' + step['error'], notify)})
            result['steps'].append(step)
            state.save(day, slot, result)
        result['context_state'] = 'DEGRADED' if any(s['state'] == 'DEGRADED' for s in result['steps']) else 'COMPLETE'
        result['state'] = 'BLOCKED' if any(s['state'] == 'BLOCKED' for s in result['steps']) else 'GREEN'
        if result['context_state'] == 'DEGRADED' and any(a['delivery'] != 'DELIVERED' for a in result['alerts']):
            result['state'] = 'BLOCKED'
            result['steps'].append({'step': 'notification', 'state': 'BLOCKED', 'error': 'TELEGRAM_DELIVERY_FAILED'})
        if run_slot == 'pm' and result['state'] == 'GREEN':
            result['notification'] = state.alert(day + ':' + slot + ':green',
                'Trader pm pre-flight GREEN: data, model probes, executor GET and cockpit checks passed. Context: ' +
                result['context_state'] + (' / GDELT PARTIAL_CONTEXT.' if result['context_state'] == 'DEGRADED' else '.'), notify)
            if result['notification'] != 'DELIVERED':
                result['state'] = 'BLOCKED'
                result['steps'].append({'step': 'notification', 'state': 'BLOCKED', 'error': 'TELEGRAM_DELIVERY_FAILED'})
        if result['state'] == 'BLOCKED' and not retry and not early:
            scheduled = at.replace(hour=15 if run_slot == 'pm' else 11, minute=45 if run_slot == 'pm' else 15, second=0, microsecond=0)
            result['retry_at'] = iso(scheduled if at < scheduled else at + timedelta(minutes=30))
        result['finished_at'] = iso(clock())
        state.save(day, slot, result)
        write_report(state, reports, day, run_slot)
        return result


class SnapshotPaperLedger:
    """PaperLedger's projection/risk bindings, with append operations confined to memory."""
    def __new__(cls, *args, **kwargs):
        from .alpaca_paper import PaperLedger
        # Preserve all production methods; only storage mutation is replaced.
        class Snapshot(PaperLedger):
            def __init__(self, root, grant, clock):
                self.root, self.grant, self.clock = Path(root), grant, clock
                self.store = ResearchStore(self.root / 'paper-evidence', read_only=True)
                self.memory = []

            @contextmanager
            def owner(self):
                with (self.root / 'paper-owner.lock').open('rb') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise TraderError('PAPER_OWNER_BUSY') from None
                    self.store.verify()
                    self.memory = [r['payload'] for r in rows(self.store, 'replay-summary')]
                    bindings = self.events(event='binding')
                    self.validate_bindings(bindings)
                    if not bindings or any(bindings[-1].get(k) != v for k, v in self.binding_target().items()):
                        raise TraderError('PAPER_LEDGER_BINDING_MISMATCH')
                    self.source_stamp = self.stamp()
                # Never hold the writer's lock during HTTP. A changed snapshot
                # defers an audit rather than obstructing a scheduled execution.
                yield

            def stamp(self):
                files = (self.store.path, self.store.path.with_name(self.store.path.name + '-wal'))
                return tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in files)

            def unchanged(self):
                return self.source_stamp == self.stamp()

            def events(self, account=None, event=None):
                return [e for e in self.memory if (account is None or e['account'] == account)
                        and (event is None or e['event'] == event)]

            def append(self, event, account, **data):
                payload = {'event': event, 'account': account, 'at': iso(self.clock()), **data}
                self.memory.append(payload)
                return payload

            def halt(self, account, reason):
                self.append('halt', account, reason=reason)
                self.append('alert', account, code=reason)
        return Snapshot(*args, **kwargs)


def get_only_client(grant, name, *, clock=now, transport=None):
    from .alpaca_paper import PaperClient
    if name not in {'ia_actions', 'ia_crypto'}:
        raise TraderError('REHEARSAL_AI_ACCOUNTS_ONLY')
    class GetOnly(PaperClient):
        def request(self, method, path, *, params=None, body=None):
            if method != 'GET' or body is not None:
                raise TraderError('REHEARSAL_GET_ONLY')
            return super().request(method, path, params=params)
    return GetOnly(grant, name, clock=clock, **({'transport': transport} if transport else {}))


def readonly_executor(paper, research, *, clock=now, client_factory=get_only_client, quotes=None, pause_paths=()):
    from .alpaca_paper import PaperExecutor
    class Reader(PaperExecutor):
        def order_alert(self, name, key, code):
            self.ledger.append('alert', name, client_id=key, code=code)
    return Reader(paper, research, clock=clock, client_factory=client_factory, quotes=quotes, pause_paths=pause_paths)


def population(grant, at):
    grant.check(at)
    if calendar_session(at.date().isoformat()):
        return 'weekday'
    if grant.weekend_decision:
        from .weekend import check
        check(grant, at)
        return 'weekend_crypto'
    return 'closed'


def latest_equity_close(at):
    sessions = equity_calendar(at.year).sessions_between(at - timedelta(days=12), at)
    completed = [s.close_at for s in sessions if s.close_at <= at]
    if not completed:
        raise TraderError('REHEARSAL_CALENDAR_UNAVAILABLE')
    return completed[-1]


class LiveChecks:
    def __init__(self, args, state, *, clock=now):
        self.args, self.state, self.clock = args, state, clock
        self.ready = False

    def require(self):
        if not self.ready:
            raise TraderError('REHEARSAL_CONFIGURATION_UNAVAILABLE')
        self.grant.check_slot(getattr(self.args, 'slot', 'am'), self.clock(), preflight=True)

    def configuration(self):
        from .alpaca_paper import PaperAuthorization
        from .alpaca_data import DataAuthorization
        from .service import preregistration, skills
        from .paper_spec import execution_spec
        self.grant = Authorization.load(self.args.authorization)
        slot = getattr(self.args, 'slot', 'am')
        self.grant.check_slot(slot, self.clock(), preflight=True)
        self.mode = population(self.grant, self.clock())
        self.paper_grant = PaperAuthorization.load(self.args.paper_authorization, self.grant)
        self.paper_grant.check(self.clock())
        self.data_grant = DataAuthorization.load(self.args.data_authorization, self.paper_grant)
        self.data_grant.check(self.clock())
        self.ledger = Ledger(self.args.runtime, self.grant, clock=self.clock, budget_root=self.args.budget_root, slot=slot, preflight=True)
        if self.ledger.paused:
            raise TraderError('PAUSED')
        self.research = ResearchStore(Path(self.args.runtime) / 'evidence', read_only=True)
        self.research.verify()
        self.paper = SnapshotPaperLedger(self.args.paper_runtime, self.paper_grant, self.clock)
        with self.paper.owner():
            if slot == 'pm':
                self.paper.check_pm_registration()
        data_ledger = Ledger(self.state.root / 'sources', self.grant, clock=self.clock,
                             budget_root=self.args.budget_root, slot=slot, preflight=True)
        self.public = PublicData(data_ledger, clock=self.clock)
        skills()
        preregistration()
        execution_spec()
        self.ready = True
        return {'population': self.mode, 'grant_hash': self.grant.identity,
                'execution_spec_hash': self.paper.binding_target()['spec_hash']}

    def prices(self, crypto):
        self.require()
        assets = (self.grant.payload['universe']['crypto'] if crypto else
                  self.grant.payload['universe']['stocks'] + self.grant.payload['universe']['sector_etfs'] +
                  self.grant.payload['universe']['benchmarks_not_predicted'])
        universe_count = len(assets)
        if not crypto:
            # A provider probe per equity/ETF/benchmark path leaves room in the
            # unchanged 80-request grant for the actual run and its labels.
            assets = list(dict.fromkeys(self.grant.payload['universe']['stocks'][:1] +
                self.grant.payload['universe']['sector_etfs'][:1] + ['SPY']))
        # Equity reachability is still probed on holidays, with the last scheduled close as freshness clock.
        at = self.clock()
        start = at.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=70)
        latest = at.replace(hour=0, minute=0, second=0, microsecond=0) if crypto else latest_equity_close(at)
        cache = self.state.root / 'price-cache.json'
        saved = strict_json(cache.read_text()) if cache.exists() else {}
        failures, checked, reused = [], [], 0
        for asset in assets:
            try:
                if protected(asset, start, at):
                    raise TraderError('REHEARSAL_PROTECTED_PRICE')
                old = saved.get(asset, {})
                if (old.get('day') == at.date().isoformat() and old.get('summary', {}).get('last_price_at') == iso(latest)):
                    summary, source = old['summary'], old['source']
                    reused += 1
                else:
                    bars, source = (self.public.crypto(asset, start, at) if crypto else self.public.yahoo(asset, start, at))
                    summary = price_summary(bars, asset, min(at, instant(source['received_at'])), crypto)
                    if summary['last_price_at'] != iso(latest):
                        raise TraderError('PRICE_GAP_OR_STALE')
                    saved[asset] = {'day': at.date().isoformat(), 'summary': summary, 'source': source}
                    temp = cache.with_suffix('.tmp')
                    temp.write_text(json.dumps(saved))
                    temp.replace(cache)
                checked.append({'asset': asset, 'last_price_at': summary['last_price_at'], 'digest': source['digest']})
            except Exception as error:
                failures.append(asset + ':' + error_code(error))
        if failures:
            # Asset identities and codes are safe; never propagate broker/source bodies.
            raise TraderError('REHEARSAL_PRICE_FAILURE', evidence=failures)
        intraday = []
        if not crypto and getattr(self.args, 'slot', 'am') == 'pm':
            from .slots import label_window
            for asset in assets:
                anchor, _ = label_window(asset, latest.date().isoformat(), '1d', [])
                if protected(asset, anchor, latest):
                    raise TraderError('REHEARSAL_PROTECTED_PRICE')
                bars, source = self.public.yahoo(asset, anchor - timedelta(minutes=5), latest + timedelta(seconds=1), interval='5m')
                if (sum(b['bar_open_at'] == anchor for b in bars) != 1 or
                        sum(b['bar_open_at'] == latest - timedelta(minutes=5) for b in bars) != 1):
                    raise TraderError('REHEARSAL_PM_INTRADAY_ANCHOR_MISSING')
                intraday.append({'asset': asset, 'digest': source['digest']})
        return {'pm_intraday_paths': len(intraday), 'pm_intraday_hash': sha256_canonical(intraday),
                'assets_checked': len(checked), 'configured_assets': universe_count, 'cache_hits': reused, 'latest_due': iso(latest),
                'identities_hash': sha256_canonical(checked)}

    def gdelt(self):
        self.require()
        from .news_queries import news_query
        from .news_context import radar_snapshot
        assets = self.grant.universe + ['macro', 'politics', 'trade']
        for asset in assets:
            news_query(asset)
        cutoff = self.clock()
        with radar_snapshot(self.args.radar_root, cutoff) as snapshot:
            cooldown = snapshot.latest('cooldown', 'gdelt')
            receipts = [r for r in snapshot.rows('http', since=cutoff - timedelta(hours=2))
                        if urlsplit(r.get('url', '')).hostname == 'api.gdeltproject.org'
                        and cutoff - timedelta(hours=2) <= instant(r['received_at']) < cutoff]
            health = [r for r in snapshot.rows('health', since=cutoff - timedelta(hours=2))
                      if r.get('source', '').startswith('gdelt:')]
        if cooldown and cutoff < instant(cooldown['until']):
            raise TraderError('SOURCE_COOLDOWN', evidence={'retry_at': cooldown['until'],
                'failures': cooldown.get('failures', 0), 'live_requests': 0, 'backoff_source': 'radar'})
        if receipts:
            receipt = max(receipts, key=lambda r: instant(r['received_at']))
            evidence = {'reused_radar_receipt': True, 'received_at': receipt['received_at'],
                        'digest': receipt['body_hash'], 'http_status': receipt['status'],
                        'queries_validated': len(assets), 'live_requests': 0}
            if receipt['status'] != 200:
                raise TraderError('SOURCE_HTTP_' + str(receipt['status']), evidence=evidence)
            if not any(r['status'] == 'LIVE' and instant(receipt['received_at']) <= instant(r['checked_at']) < cutoff for r in health):
                raise TraderError('REHEARSAL_GDELT_STALE_OR_EMPTY', evidence=evidence)
            return {**evidence, 'clock': 'RADAR_RECEIPT_NOT_PUBLISHER_ATTESTED'}
        backoff = self.state.root / 'gdelt-backoff.json'
        prior = strict_json(backoff.read_text()) if backoff.exists() else {}
        if prior and cutoff < instant(prior['retry_at']):
            raise TraderError('SOURCE_COOLDOWN', evidence={**prior, 'live_requests': 0, 'backoff_source': 'rehearsal'})
        try:
            headlines, source = self.public.headlines(news_query('macro'), self.clock())
        except TraderError as error:
            if error.code == 'SOURCE_HTTP_429':
                from .data import rate_limit_retry_at
                failures = min(6, prior.get('failures', 0) + 1)
                until = max(instant(self.public.gdelt_retry_at),
                            instant(rate_limit_retry_at({}, self.clock(), failures)))
                saved = {'retry_at': iso(until), 'failures': failures, 'status': 429}
                temp = backoff.with_suffix('.tmp')
                temp.write_text(json.dumps(saved))
                temp.replace(backoff)
                raise TraderError(error.code, evidence={**saved, 'live_requests': 1}) from None
            raise
        backoff.unlink(missing_ok=True)
        if not headlines or max(instant(h['published_at']) for h in headlines) < self.clock() - timedelta(hours=6):
            raise TraderError('REHEARSAL_GDELT_STALE_OR_EMPTY')
        return {'articles': len(headlines), 'digest': source['digest'], 'queries_validated': len(assets),
                'clock': 'GDELT_SEEN_NOT_PUBLISHER_ATTESTED'}

    def news(self):
        self.require()
        from .news_context import build_news_context, radar_snapshot
        cutoff = self.clock()
        context, digest = build_news_context(self.args.radar_root, self.args.veille, cutoff)
        if context['veille']['status'] != 'AVAILABLE':
            raise TraderError('REHEARSAL_VEILLE_' + context['veille']['status'])
        # Selected events are ranked by importance, so their oldest first receipt
        # is not a collector freshness clock. Check all causal input receipts.
        with radar_snapshot(self.args.radar_root, cutoff) as snapshot:
            receipts = snapshot.rows('news', since=cutoff - timedelta(hours=2))
        receipts = [s for s in receipts if cutoff - timedelta(hours=2) <= instant(s['received_at']) < cutoff
                    and (not s.get('published_at') or instant(s['published_at']) < cutoff)]
        if not receipts:
            raise TraderError('REHEARSAL_RADAR_STALE_OR_EMPTY')
        return {'events': len(context['events']), 'veille_lines': len(context['veille']['lines']),
                'recent_receipts': len(receipts), 'cutoff': iso(cutoff), 'context_hash': digest}

    def archives(self):
        self.require()
        from .data import archived
        results = {}
        for kind, root in (('fomc', self.args.fomc_store), ('edgar', self.args.edgar_store)):
            if not root:
                continue
            view = archived(root, kind, self.clock())
            if view['state'] in {'INTEGRITY_ERROR', 'UNAVAILABLE', 'UNRESOLVED'}:
                raise TraderError('REHEARSAL_ARCHIVE_' + kind.upper())
            results[kind] = {'items': len(view['items']), 'binding_hash': sha256_canonical(view['binding'])}
        return results

    def model(self, role, retry=False):
        self.require()
        from .preflight import preflight
        from .runners import ModelRunner, command
        from .schemas import validate
        runner = ModelRunner(self.ledger, clock=self.clock)
        if role == 'analyst_gpt':
            path = self.ledger.root / 'gpt-preflight.json'
            prior = strict_json(path.read_text()) if path.exists() else {}
            if (prior.get('state') == 'GREEN' and prior.get('day') == self.clock().date().isoformat()
                    and prior.get('model', {}).get('model') == self.grant.gpt_model):
                # Revalidate the exact deployed schema binding even when reusing the web proof.
                from .schemas import HERE, GPT_SCHEMAS, openai_strict_problems
                schema = strict_json((HERE / 'schemas/analyst.gpt.json').read_text())
                if schema != GPT_SCHEMAS['analyst'] or openai_strict_problems(schema):
                    raise TraderError('GPT_SCHEMA_DRIFT')
                return {'reused_gpt_preflight': True, 'web_searches': prior['web_searches'], 'model': prior['model']}
            result = preflight(self.ledger, runner, retry=retry)
            if result['state'] != 'GREEN':
                raise TraderError(result.get('error', 'REHEARSAL_GPT_PREFLIGHT_FAILED'))
            return {'web_searches': result['web_searches'], 'model': result['model']}
        cache = self.state.root / (role + '-probe.json')
        binding = sha256_canonical(command(role, self.grant))
        prior = strict_json(cache.read_text()) if cache.exists() else {}
        if (prior.get('day') == self.clock().date().isoformat() and prior.get('command_hash') == binding):
            return {**prior['evidence'], 'cached': True}
        expected = {'verdicts': []} if role == 'reviewer' else {'regime': ['PREFLIGHT_OK'], 'views': []}
        prompt = ('Infrastructure health probe only. Synthetic empty analyst/reviewer input, no market data or trading decision. '
                  'Do not search, fetch, run tools or commands, or read/write files. Return exactly ' + json.dumps(expected))
        with self.ledger.owner():
            output = runner.once(role, prompt, probe=True)
            validate('reviewer' if role == 'reviewer' else 'analyst', output)
            if output != expected:
                raise TraderError('PREFLIGHT_OUTPUT_INVALID')
        evidence = {'model': runner.metadata[role], 'output_hash': sha256_canonical(output)}
        cache.write_text(json.dumps({'day': self.clock().date().isoformat(), 'command_hash': binding, 'evidence': evidence}))
        return evidence

    def alpaca_data(self):
        self.require()
        from .alpaca_data import DataClient
        from .alpaca_paper import decimal, symbol
        client = DataClient(self.data_grant, clock=self.clock)
        equities = self.grant.payload['universe']['stocks'] + self.grant.payload['universe']['sector_etfs']
        cryptos = self.grant.payload['universe']['crypto']
        checked, trade_ages, failures = [], [], []
        at = self.clock()
        session = calendar_session(at.date().isoformat())
        before_open = not session or at < session.open_at
        non_executable = 0
        for crypto, assets in ((False, equities), (True, cryptos)):
            prefix = '/v1beta3/crypto/us/latest/' if crypto else '/v2/stocks/'
            params = {'symbols': ','.join(symbol(a) for a in assets)}
            if not crypto:
                params['feed'] = 'iex'
            q = client.request('GET', prefix + ('quotes' if crypto else 'quotes/latest'), params=params)
            t = client.request('GET', prefix + ('trades' if crypto else 'trades/latest'), params=params)
            at = self.clock()
            session = calendar_session(at.date().isoformat())
            in_session = session and session.open_at <= at < session.close_at
            earliest = at - timedelta(seconds=60) if crypto or in_session else latest_equity_close(at) - timedelta(minutes=5)
            for asset in assets:
                field = 'symbol'
                try:
                    quote_row, trade = q['quotes'][symbol(asset)], t['trades'][symbol(asset)]
                    field = 'quote.t'
                    if not earliest <= instant(quote_row['t']) <= at:
                        raise TraderError('REHEARSAL_DATA_STALE')
                    # Match DataClient.latest: a current quote proves feed
                    # freshness; an infrequently traded asset's last trade is
                    # allowed to be older, but never from the future.
                    field = 'trade.t'
                    if instant(trade['t']) > at:
                        raise TraderError('DATA_TRADE_INVALID')
                    if not crypto and before_open and instant(trade['t']) < earliest:
                        raise TraderError('REHEARSAL_DATA_STALE')
                    field = 'trade.p'
                    if decimal(trade['p']) <= 0:
                        raise TraderError('REHEARSAL_DATA_PRICE_INVALID')
                    field = 'quote.bp/ap'
                    bid, ask = decimal(quote_row['bp']), decimal(quote_row['ap'])
                    executable = bid > 0 and ask >= bid
                    # IEX's last book update after the close can be one-sided or
                    # crossed. Before an upcoming DAY market entry it is only
                    # indicative; the completed-session trade proves prices.
                    # In-session/crypto quotes and production limit guards stay strict.
                    if bid < 0 or ask < 0 or (not executable and (crypto or not before_open)):
                        raise TraderError('REHEARSAL_DATA_PRICE_INVALID')
                    if not crypto and not executable:
                        non_executable += 1
                    checked.append({'asset': asset, 'quote_at': quote_row['t'], 'trade_at': trade['t']})
                    trade_ages.append((at - instant(trade['t'])).total_seconds())
                except (KeyError, TypeError):
                    failures.append({'asset': asset, 'field': field, 'code': 'REHEARSAL_DATA_MISSING_SYMBOL'})
                except TraderError as error:
                    failures.append({'asset': asset, 'field': field, 'code': error.code})
        if failures:
            raise TraderError(failures[0]['code'], evidence={'failures': failures})
        return {'assets_checked': len(checked), 'timestamps_hash': sha256_canonical(checked), 'feed': 'iex/crypto-us',
                'equity_quote_use': 'INDICATIVE_CLOSED_BOOK' if before_open else 'EXECUTABLE',
                'non_executable_equity_quotes': non_executable,
                'trades_older_than_60s': sum(age > 60 for age in trade_ages),
                'oldest_trade_age_seconds': round(max(trade_ages, default=0))}

    def executor(self):
        self.require()
        from .alpaca_paper import decimal
        executor = readonly_executor(self.paper, self.research, clock=self.clock,
                                     pause_paths=(Path(self.args.runtime) / 'PAUSED', Path(self.args.budget_root) / 'PAUSED'))
        result = executor.run('execute', dry_run=True)
        if not self.paper.unchanged():
            raise TraderError('REHEARSAL_PAPER_SNAPSHOT_CHANGED')
        if result['state'] != 'COMPLETE':
            reason = next((a.get('reason') for a in result['accounts'] if a.get('reason')), 'REHEARSAL_EXECUTOR_' + result['state'])
            raise TraderError(reason)
        if len(result['accounts']) != (1 if self.mode != 'weekday' else 2):
            raise TraderError('REHEARSAL_EXECUTOR_ACCOUNT_MISSING')
        checked = []
        for name in ('ia_actions', 'ia_crypto'):
            client = get_only_client(self.paper_grant, name, clock=self.clock)
            account = client.start()
            broker_clock = client.request('GET', '/v2/clock')
            if abs((self.clock() - instant(broker_clock['timestamp'])).total_seconds()) > 60:
                raise TraderError('REHEARSAL_BROKER_CLOCK_STALE')
            if account.get('status') != 'ACTIVE' or account.get('trading_blocked') or decimal(account['buying_power']) <= 0:
                raise TraderError('REHEARSAL_BUYING_POWER_OR_ACCOUNT_BLOCKED')
            if self.paper.halted(name):
                raise TraderError('REHEARSAL_PAPER_ACCOUNT_HALTED')
            if name == 'ia_actions':
                from .alpaca_paper import NY
                day = self.clock().date().isoformat()
                calendar = client.request('GET', '/v2/calendar', params={'start': day, 'end': day})
                session = calendar_session(day)
                if ((session is None and calendar) or (session and (len(calendar) != 1 or
                        calendar[0].get('date') != day or calendar[0].get('open') != session.open_at.astimezone(NY).strftime('%H:%M') or
                        calendar[0].get('close') != session.close_at.astimezone(NY).strftime('%H:%M')))):
                    raise TraderError('REHEARSAL_BROKER_CALENDAR_MISMATCH')
            checked.append({'account': name, 'suffix': client.config['account_suffix'], 'buying_power_positive': True,
                            'clock_digest': sha256_canonical(broker_clock)})
        return {'accounts': checked, 'planned_orders': sum(len(a['planned_orders']) for a in result['accounts']),
                'production_writes': 0, 'broker_methods': ['GET']}

    def order_types(self):
        self.require()
        from .alpaca_paper import order_terms, symbol
        checked = []
        session = calendar_session(self.clock().date().isoformat())
        planner = readonly_executor(self.paper, self.research, clock=self.clock)
        for name in ('ia_actions', 'ia_crypto'):
            client = get_only_client(self.paper_grant, name, clock=self.clock)
            account = client.start()
            assets = self.paper_grant.assets(name)
            for asset in assets:
                info = client.request('GET', '/v2/assets/' + quote(symbol(asset), safe=''))
                klass = 'crypto' if name == 'ia_crypto' else 'us_equity'
                if info.get('class') != klass or info.get('status') != 'active' or info.get('tradable') is not True:
                    raise TraderError('REHEARSAL_ASSET_NOT_TRADABLE')
                if name == 'ia_actions' and (not account.get('shorting_enabled') or not info.get('shortable')):
                    raise TraderError('REHEARSAL_ASSET_NOT_SHORTABLE')
                if name == 'ia_actions' and session:
                    for purpose, at in (('entry', session.open_at + timedelta(seconds=30)),
                                        ('exit', session.close_at - timedelta(minutes=5))):
                        terms = order_terms(name, purpose, at, session)
                        if terms != {'type': 'market', 'time_in_force': 'day'}:
                            raise TraderError('REHEARSAL_EQUITY_ORDER_TYPE')
                        planner.clock = lambda at=at: at
                        lot = {'symbol': symbol(asset), 'qty': '1', 'run_id': 'synthetic-rehearsal',
                               'lot_id': 'synthetic-rehearsal', 'session': session.session_date,
                               'variant': 'base'}
                        if any(plan['order'][k] != v for plan in planner.plans(name, [lot], purpose) for k, v in terms.items()):
                            raise TraderError('REHEARSAL_PLANNER_ORDER_TYPE')
                    terms = order_terms(name, 'entry', session.close_at, session, after_hours=True, bid='100', ask='100.01')
                    if terms['type'] != 'limit' or terms['time_in_force'] != 'day' or not terms['extended_hours']:
                        raise TraderError('REHEARSAL_EXTENDED_ORDER_TYPE')
                    if getattr(self.args, 'slot', 'am') == 'pm':
                        from . import slots
                        at = slots.start_at(session.session_date, 'pm')
                        terms = order_terms(name, 'entry', at, session, pm=True)
                        planner.clock = lambda: at
                        lot['slot'] = 'pm'
                        if any(plan['order'][k] != v for plan in planner.plans(name, [lot], 'entry') for k, v in terms.items()):
                            raise TraderError('REHEARSAL_PM_PLANNER_ORDER_TYPE')
                elif name == 'ia_crypto':
                    for purpose in ('entry', 'exit'):
                        planner.clock = self.clock
                        lot = {'symbol': symbol(asset), 'qty': '0.01', 'run_id': 'synthetic-rehearsal',
                               'lot_id': 'synthetic-rehearsal', 'session': self.clock().date().isoformat(), 'variant': 'base'}
                        terms = order_terms(name, purpose, self.clock(), session)
                        if terms != {'type': 'market', 'time_in_force': 'gtc'} or any(
                                plan['order'][k] != v for plan in planner.plans(name, [lot], purpose) for k, v in terms.items()):
                            raise TraderError('REHEARSAL_CRYPTO_ORDER_TYPE')
                checked.append({'asset': asset, 'class': klass, 'shortable': info.get('shortable', False)})
        return {'assets_checked': len(checked), 'capabilities_hash': sha256_canonical(checked),
                'equity': ['market/day', 'limit/day/extended'], 'crypto': ['market/gtc'],
                'acceptance_and_fills': 'UNPROVEN_GET_ONLY'}

    def scoring(self):
        self.require()
        from .scoring import realize, scorecard
        from .service import preregistration
        if not callable(realize):
            raise TraderError('REHEARSAL_LABEL_JOB_UNAVAILABLE')
        card = scorecard(self.research)
        return {'scorecard_hash': sha256_canonical(card), 'label_job_loaded': True, 'preregistration_hash': preregistration()}

    def cockpit(self):
        self.require()
        from scripts.trading_lab.app_api.trader import TraderViews
        api = TraderViews(self.args.runtime)
        responses = {path: api.dispatch('/api/v1/trader/' + path, {})
                     for path in ('today', 'context', 'runs', 'ledger', 'labels', 'scorecard', 'alerts', 'series', 'health')}
        encoded = json.dumps(responses, allow_nan=False)
        from scripts.radar.cockpit_export import build_home, load_trader, load_scorecard, write_snapshot
        from scripts.radar.cockpit_analysis import build_analysis
        reports = [p for p in (Path(self.args.reports) / 'radar').glob('*.json') if 'synthetic' not in p.name]
        if not reports:
            raise TraderError('REHEARSAL_COCKPIT_RADAR_REPORT_MISSING')
        radar = strict_json(max(reports, key=lambda p: p.stat().st_mtime_ns).read_text())
        trader = load_trader(self.research, load_scorecard(self.research))
        snapshots = {'radar-home.json': build_home(radar, trader, generated_at=iso(self.clock())),
                     'analysis.json': build_analysis(radar, trader, generated_at=iso(self.clock()))}
        for name, snapshot in snapshots.items():
            write_snapshot(self.state.root / 'cockpit-probe', name, snapshot)
        return {'endpoints': len(responses), 'export_bytes': len(encoded.encode()), 'export_hash': sha256_canonical(responses),
                'snapshots_checked': len(snapshots), 'snapshots_hash': sha256_canonical(snapshots)}

    def steps(self, retry=False):
        return [('configuration', self.configuration), ('yahoo', lambda: self.prices(False)),
                ('coinbase', lambda: self.prices(True)), ('gdelt', self.gdelt), ('radar_veille', self.news),
                ('archives', self.archives), *[(role, lambda role=role: self.model(role, retry)) for role in ROLES],
                ('alpaca_data', self.alpaca_data), ('executor', self.executor), ('order_types', self.order_types),
                ('scorer_labels', self.scoring), ('cockpit_export', self.cockpit)]


def arguments(parser):
    home = Path.home()
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--slot', choices=('am', 'pm'), default='am')
    parser.add_argument('--paper-authorization', required=True)
    parser.add_argument('--data-authorization', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--budget-root', default=str(home / '.local/share/hyprl/trader-agent-budget'))
    parser.add_argument('--paper-runtime', default=str(home / '.local/share/hyprl/trader-alpaca-paper'))
    parser.add_argument('--monitor-runtime', default=str(home / '.local/share/hyprl/trader-rehearsal'))
    parser.add_argument('--reports', default=str(home / 'reports'))
    parser.add_argument('--radar-root', default=str(home / 'private/radar-v1'))
    parser.add_argument('--veille', default=str(home / 'claude-book/veille.md'))
    parser.add_argument('--telegram-env', default=str(home / 'private/telegram.env'))
    parser.add_argument('--fomc-store')
    parser.add_argument('--edgar-store')


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    attempts = parser.add_mutually_exclusive_group()
    attempts.add_argument('--retry', action='store_true')
    attempts.add_argument('--early', action='store_true')
    parser.add_argument('--failure-hook', action='store_true')
    args = parser.parse_args(argv)
    if args.early and args.slot != 'am':
        parser.error('early rehearsal is for the morning slot')
    state = MonitorState(args.monitor_runtime)
    notify = telegram(args.telegram_env)
    if args.failure_hook:
        day = now().date().isoformat()
        prefix = 'pm-' if args.slot == 'pm' else ''
        slots = (prefix + 'primary', prefix + 'retry') if prefix else ('early', 'primary', 'retry')
        for slot in slots:
            result = state.get(day, slot)
            if result and result['state'] == 'RUNNING':
                result['state'] = 'BLOCKED'
                result['steps'].append({'step': 'service', 'state': 'BLOCKED', 'error': 'REHEARSAL_SERVICE_INTERRUPTED'})
                result['alerts'].append({'step': 'service', 'delivery': state.alert(
                    day + ':' + slot + ':service', 'Trader rehearsal failed: service interrupted', notify)})
                if slot == prefix + 'primary':
                    at = now()
                    due = at.replace(hour=15 if args.slot == 'pm' else 11, minute=45 if args.slot == 'pm' else 15, second=0, microsecond=0)
                    result['retry_at'] = iso(due if at < due else at + timedelta(minutes=30))
                state.save(day, slot, result)
                write_report(state, args.reports, day, args.slot)
        return 0
    result = rehearse(state, LiveChecks(args, state).steps(args.retry), retry=args.retry, early=args.early,
                      notify=notify, reports=args.reports, run_slot=args.slot)
    print(json.dumps(result, indent=2))
    return 1 if result['state'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
