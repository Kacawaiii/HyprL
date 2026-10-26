"""Durable dispatch reservations: crashes and failures never refund a request."""
from contextlib import contextmanager
import fcntl
import json
import os
import sqlite3

from .config import TraderError, iso, now, private_root


class Ledger:
    def __init__(self, root, grant, *, clock=now, budget_root=None, slot='am', preflight=False):
        self.root = private_root(root)
        self.budget_root = private_root(budget_root) if budget_root else self.root
        self.grant, self.clock = grant, clock
        self.budget_day = None
        self.arm = 'base'
        if slot not in {'am', 'pm'}:
            raise TraderError('INVALID_SLOT')
        self.slot = slot
        self.preflight = preflight
        self.path = self.budget_root / "dispatch.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dispatch (
                  seq INTEGER PRIMARY KEY, day TEXT NOT NULL, kind TEXT NOT NULL,
                  at TEXT NOT NULL, grant_hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS dispatch_no_update BEFORE UPDATE ON dispatch
                  BEGIN SELECT RAISE(ABORT,'dispatch immutable'); END;
                CREATE TRIGGER IF NOT EXISTS dispatch_no_delete BEFORE DELETE ON dispatch
                  BEGIN SELECT RAISE(ABORT,'dispatch immutable'); END;
                CREATE TABLE IF NOT EXISTS alert_state (
                  channel TEXT PRIMARY KEY, code TEXT NOT NULL, at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS work_reservations (
                  dispatch_seq INTEGER PRIMARY KEY, day TEXT NOT NULL, slot TEXT NOT NULL,
                  arm TEXT NOT NULL, role TEXT NOT NULL, tokens INTEGER NOT NULL,
                  spec_hash TEXT NOT NULL, approval_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS work_receipts (
                  dispatch_seq INTEGER PRIMARY KEY, new_tokens INTEGER NOT NULL,
                  cache_read_tokens INTEGER NOT NULL, code TEXT NOT NULL,
                  usage_complete INTEGER NOT NULL);
                CREATE TRIGGER IF NOT EXISTS work_reservations_no_update BEFORE UPDATE ON work_reservations
                  BEGIN SELECT RAISE(ABORT,'work reservation immutable'); END;
                CREATE TRIGGER IF NOT EXISTS work_reservations_no_delete BEFORE DELETE ON work_reservations
                  BEGIN SELECT RAISE(ABORT,'work reservation immutable'); END;
                CREATE TRIGGER IF NOT EXISTS work_receipts_no_update BEFORE UPDATE ON work_receipts
                  BEGIN SELECT RAISE(ABORT,'work receipt immutable'); END;
                CREATE TRIGGER IF NOT EXISTS work_receipts_no_delete BEFORE DELETE ON work_receipts
                  BEGIN SELECT RAISE(ABORT,'work receipt immutable'); END;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def owner(self):
        with (self.budget_root / "owner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TraderError("OWNER_BUSY") from None
            yield

    def reserve(self, kind):
        if self.paused:
            raise TraderError('PAUSED')
        at = self.clock()
        from . import retirement_spec
        retirement_spec.check_dispatch(kind, at)
        self.grant.check_slot(self.slot, at, preflight=self.preflight or kind.startswith(('rehearsal_', 'gpt_preflight')))
        if self.arm == 'news' and kind in self.grant.payload['data_sources']:
            raise TraderError('NEWS_ACQUISITION_FORBIDDEN')
        news = self.arm == 'news' and kind in {'run', *self.grant.payload['external_models']}
        if news:
            self.grant.check_news(at)
            kind = 'news_' + kind
        day = at.date().isoformat() if news else self.budget_day or at.date().isoformat()
        p = self.grant.payload
        if news:
            maximum = 1 if kind == 'news_run' else 2
        elif kind == "run":
            maximum = p["budgets"]["max_runs_per_day"]
        elif kind == "catchup_run":
            maximum = 1   # one operator-approved recovery per day; the service requires a FAILED daily run
        elif kind == 'gpt_preflight':
            maximum = 2   # preflight + one rehearsal retry; separate from decision budgets
        elif kind in {'rehearsal_analyst_claude', 'rehearsal_reviewer'}:
            maximum = 2   # daily rehearsal + one retry; failures never refund calls
        elif kind in p["external_models"]:
            m = p["external_models"][kind]
            maximum = m["calls_per_day"] + m["retries_per_day"]
        elif kind in ("yahoo_chart", "coinbase_exchange_public", "gdelt_doc_api"):
            maximum = p["data_sources"][kind]["max_requests_per_day"]
        else:
            raise TraderError("UNAUTHORIZED_DISPATCH")
        logical_kind = kind
        if self.slot == 'pm':
            kind = 'pm_' + kind
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            at = self.clock()
            retirement_spec.check_dispatch(logical_kind, at)
            day = at.date().isoformat() if news else self.budget_day or at.date().isoformat()
            self.grant.check_slot(self.slot, at, preflight=self.preflight or logical_kind.startswith(('rehearsal_', 'gpt_preflight')))
            if news:
                self.grant.check_news(at)
            from .work_budget import charge
            work = charge(self, logical_kind.removeprefix('news_'), at)
            if work:
                if work['arm'] not in {'base', 'news', 'monitor'}:
                    raise TraderError('WORK_BUDGET_APPROVAL_INVALID')
                used, arm_used, role_used = db.execute('''SELECT coalesce(sum(tokens),0),
                    coalesce(sum(CASE WHEN arm=? THEN tokens ELSE 0 END),0),
                    coalesce(sum(CASE WHEN arm=? AND role=? THEN tokens ELSE 0 END),0)
                    FROM work_reservations WHERE day=? AND slot=?''',
                    (work['arm'], work['arm'], work['role'], day, self.slot)).fetchone()
                if any(used + work['tokens'] > cap for used, cap in
                       ((used, work['slot_cap']), (arm_used, work['arm_cap']), (role_used, work['role_cap']))):
                    raise TraderError('WORK_BUDGET_EXHAUSTED')
                # Revision identities are immutable. A new policy cannot reinterpret
                # old envelopes, including a same-day deployment after rollback.
                mismatch = db.execute('SELECT 1 FROM work_reservations WHERE day=? AND slot=? '
                    'AND spec_hash!=? LIMIT 1', (day, self.slot, work['spec_hash'])).fetchone()
                if mismatch:
                    raise TraderError('WORK_BUDGET_SPEC_HASH_MISMATCH')
                prefix = 'pm_' if self.slot == 'pm' else ''
                model_kinds = [prefix + arm + role for arm in ('', 'news_', 'rehearsal_')
                               for role in ('analyst_claude', 'reviewer')]
                unaccounted = db.execute('''SELECT 1 FROM dispatch d LEFT JOIN work_reservations w
                    ON w.dispatch_seq=d.seq WHERE d.day=? AND d.kind IN (?,?,?,?,?,?)
                    AND w.dispatch_seq IS NULL LIMIT 1''', (day, *model_kinds)).fetchone()
                if unaccounted:
                    raise TraderError('WORK_BUDGET_UNACCOUNTED')
            count = db.execute("SELECT count(*) FROM dispatch WHERE day=? AND kind=?", (day, kind)).fetchone()[0]
            if count >= maximum:
                raise TraderError("BUDGET_EXHAUSTED")
            if logical_kind in p["external_models"]:
                prefix = 'pm_' if self.slot == 'pm' else ''
                total = db.execute("SELECT count(*) FROM dispatch WHERE day=? AND kind IN (?,?,?)",
                    (day, prefix + "analyst_claude", prefix + "analyst_gpt", prefix + "reviewer")).fetchone()[0]
                if total >= min(p["budgets"]["max_llm_calls_per_day"], 4 if retirement_spec.active(at) else 6):
                    raise TraderError("BUDGET_EXHAUSTED")
            if news and logical_kind != 'news_run':
                prefix = 'pm_' if self.slot == 'pm' else ''
                total = db.execute("SELECT count(*) FROM dispatch WHERE day=? AND kind IN (?,?,?)",
                    (day, prefix + 'news_analyst_claude', prefix + 'news_analyst_gpt', prefix + 'news_reviewer')).fetchone()[0]
                if total >= (4 if retirement_spec.active(at) else 6):
                    raise TraderError('BUDGET_EXHAUSTED')
            if logical_kind == "gdelt_doc_api":
                row = db.execute("SELECT at FROM dispatch WHERE kind IN (?,?) ORDER BY seq DESC LIMIT 1", ('gdelt_doc_api', 'pm_gdelt_doc_api')).fetchone()
                if row:
                    from .config import instant
                    if (at - instant(row[0])).total_seconds() < p["data_sources"][logical_kind]["min_spacing_seconds"]:
                        raise TraderError("SOURCE_SPACING")
            cursor = db.execute("INSERT INTO dispatch(day,kind,at,grant_hash) VALUES(?,?,?,?)",
                                (day, kind, iso(at), self.grant.pm_identity if self.slot == 'pm' else self.grant.identity))
            if work:
                db.execute('INSERT INTO work_reservations VALUES(?,?,?,?,?,?,?,?)',
                    (cursor.lastrowid, day, self.slot, work['arm'], work['role'], work['tokens'],
                     work['spec_hash'], work['approval_hash']))
            return cursor.lastrowid

    def record_work(self, seq, new_tokens, cached, code, *, usage_complete=True):
        if any(type(value) is not int or value < 0 for value in (new_tokens, cached)):
            raise TraderError('MODEL_USAGE_INVALID')
        with self.connect() as db:
            db.execute('INSERT INTO work_receipts VALUES(?,?,?,?,?)',
                       (seq, new_tokens, cached, code, int(usage_complete)))

    def work_counts(self, day=None):
        with self.connect() as db:
            return [dict(zip(('slot', 'arm', 'reserved', 'observed_new', 'observed_cache_read', 'incomplete_usage'), row)) for row in
                db.execute('''SELECT w.slot,w.arm,sum(w.tokens),coalesce(sum(r.new_tokens),0),
                    coalesce(sum(r.cache_read_tokens),0),sum(CASE WHEN r.dispatch_seq IS NULL OR r.usage_complete=0
                    THEN 1 ELSE 0 END) FROM work_reservations w LEFT JOIN work_receipts r
                    ON w.dispatch_seq=r.dispatch_seq WHERE w.day=? GROUP BY w.slot,w.arm''',
                    (day or self.budget_day or self.clock().date().isoformat(),))]

    @property
    def paused(self):
        return (self.root / 'PAUSED').exists() or (self.budget_root / 'PAUSED').exists()

    def counts(self, day=None):
        with self.connect() as db:
            counts = dict(db.execute("SELECT kind,count(*) FROM dispatch WHERE day=? GROUP BY kind",
                                   (day or (self.clock().date().isoformat() if self.arm == 'news' else
                                            self.budget_day or self.clock().date().isoformat()),)))
            return {k.removeprefix('pm_'): v for k, v in counts.items() if k.startswith('pm_')} if self.slot == 'pm' else {
                k: v for k, v in counts.items() if not k.startswith('pm_')}

    def alert(self, code, *, role=None):
        payload = {"schema": "trader-alert-v1", "at": iso(self.clock()), "code": code, "role": role}
        with (self.root / "alerts.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp = self.root / "alert.tmp"
        temp.write_text(json.dumps(payload))
        temp.replace(self.root / "alert.json")

    def alert_state(self, channel, code, *, initial=True, force=False, role=None):
        """Persist polling transitions across processes; actual outcomes can force an alert."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT code FROM alert_state WHERE channel=?', (channel,)).fetchone()
            if previous and previous[0] == code and not force:
                return False
            db.execute('INSERT INTO alert_state(channel,code,at) VALUES(?,?,?) '
                       'ON CONFLICT(channel) DO UPDATE SET code=excluded.code,at=excluded.at',
                       (channel, code, iso(self.clock())))
            if previous or initial or force:
                self.alert(code, role=role)
                return True
        return False
