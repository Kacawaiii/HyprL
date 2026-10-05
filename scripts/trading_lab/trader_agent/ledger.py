"""Durable dispatch reservations: crashes and failures never refund a request."""
from contextlib import contextmanager
import fcntl
import json
import os
import sqlite3

from .config import TraderError, iso, now, private_root


class Ledger:
    def __init__(self, root, grant, *, clock=now):
        self.root = private_root(root)
        self.grant, self.clock = grant, clock
        self.path = self.root / "dispatch.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dispatch (
                  seq INTEGER PRIMARY KEY, day TEXT NOT NULL, kind TEXT NOT NULL,
                  at TEXT NOT NULL, grant_hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS dispatch_no_update BEFORE UPDATE ON dispatch
                  BEGIN SELECT RAISE(ABORT,'dispatch immutable'); END;
                CREATE TRIGGER IF NOT EXISTS dispatch_no_delete BEFORE DELETE ON dispatch
                  BEGIN SELECT RAISE(ABORT,'dispatch immutable'); END;
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
        with (self.root / "owner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TraderError("OWNER_BUSY") from None
            yield

    def reserve(self, kind):
        at = self.clock()
        day = at.date().isoformat()
        p = self.grant.payload
        if kind == "run":
            maximum = p["budgets"]["max_runs_per_day"]
        elif kind in p["external_models"]:
            m = p["external_models"][kind]
            maximum = m["calls_per_day"] + m["retries_per_day"]
        elif kind in ("yahoo_chart", "coinbase_exchange_public", "gdelt_doc_api"):
            maximum = p["data_sources"][kind]["max_requests_per_day"]
        else:
            raise TraderError("UNAUTHORIZED_DISPATCH")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            at = self.clock()
            day = at.date().isoformat()
            self.grant.check(at)
            count = db.execute("SELECT count(*) FROM dispatch WHERE day=? AND kind=?", (day, kind)).fetchone()[0]
            if count >= maximum:
                raise TraderError("BUDGET_EXHAUSTED")
            if kind in p["external_models"]:
                total = db.execute("SELECT count(*) FROM dispatch WHERE day=? AND kind IN (?,?,?)",
                    (day, "analyst_claude", "analyst_gpt", "reviewer")).fetchone()[0]
                if total >= p["budgets"]["max_llm_calls_per_day"]:
                    raise TraderError("BUDGET_EXHAUSTED")
            if kind == "gdelt_doc_api":
                row = db.execute("SELECT at FROM dispatch WHERE kind=? ORDER BY seq DESC LIMIT 1", (kind,)).fetchone()
                if row:
                    from .config import instant
                    if (at - instant(row[0])).total_seconds() < p["data_sources"][kind]["min_spacing_seconds"]:
                        raise TraderError("SOURCE_SPACING")
            cursor = db.execute("INSERT INTO dispatch(day,kind,at,grant_hash) VALUES(?,?,?,?)",
                                (day, kind, iso(at), self.grant.identity))
            return cursor.lastrowid

    def counts(self, day=None):
        with self.connect() as db:
            return dict(db.execute("SELECT kind,count(*) FROM dispatch WHERE day=? GROUP BY kind",
                                   (day or self.clock().date().isoformat(),)))

    def alert(self, code, *, role=None):
        payload = {"schema": "trader-alert-v1", "at": iso(self.clock()), "code": code, "role": role}
        with (self.root / "alerts.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp = self.root / "alert.tmp"
        temp.write_text(json.dumps(payload))
        temp.replace(self.root / "alert.json")
