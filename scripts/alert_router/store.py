"""Private encrypted state, tenant boundaries and a transactional durable outbox."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .destinations import subscriptions, validate
from .model import KINDS, Alert, AlertError, canonical


def private_file(path, *, create=False):
    flags = (os.O_RDWR if create else os.O_RDONLY) | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise AlertError("UNSAFE_FILE_PERMISSIONS")
    return fd


def create_key(path):
    path = Path(path)
    with os.fdopen(private_file(path, create=True), "wb") as out:
        out.write(Fernet.generate_key())
        out.flush()
        os.fsync(out.fileno())


class Store:
    def __init__(self, root, key_path, *, allowed_customers, telegram_chats=None):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AlertError("UNSAFE_DIRECTORY_PERMISSIONS")
        with os.fdopen(private_file(key_path), "rb") as source:
            key = source.read(128)
        try:
            self.cipher = Fernet(key)
        except ValueError:
            raise AlertError("INVALID_ENCRYPTION_KEY") from None
        self.index_key = hashlib.sha256(b"hyprl-alert-index-v1" + key).digest()
        self.allowed = frozenset(allowed_customers)
        self.telegram_chats = telegram_chats or {}
        self.path = self.root / "alerts.sqlite3"
        try:
            fd = private_file(self.path, create=True)
        except FileExistsError:
            fd = private_file(self.path)
        os.close(fd)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, value BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS customers (id TEXT PRIMARY KEY, value BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS destinations (
                    id TEXT PRIMARY KEY, customer TEXT NOT NULL, value BLOB NOT NULL,
                    paused INTEGER NOT NULL DEFAULT 0, last_attempt REAL,
                    window INTEGER NOT NULL DEFAULT -1, used INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    customer TEXT NOT NULL, value BLOB NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS duels (
                    id TEXT PRIMARY KEY, customer TEXT NOT NULL, value BLOB NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, destination TEXT NOT NULL REFERENCES destinations(id) ON DELETE CASCADE,
                    value BLOB NOT NULL, state TEXT NOT NULL DEFAULT 'pending', due REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, lease TEXT, code TEXT, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state, due);
                CREATE INDEX IF NOT EXISTS events_customer ON events(customer, seq);
            """)
            row = db.execute("SELECT value FROM meta WHERE id=1").fetchone()
            if row:
                if self.open(row[0]) != {"version": 1}:
                    raise AlertError("STORE_VERSION_MISMATCH")
            else:
                db.execute("INSERT INTO meta VALUES(1,?)", (self.seal({"version": 1}),))

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def seal(self, value):
        return self.cipher.encrypt(canonical(value))

    def open(self, value):
        try:
            return json.loads(self.cipher.decrypt(value))
        except (InvalidToken, ValueError):
            raise AlertError("DECRYPTION_FAILED") from None

    def customer(self, customer):
        if customer not in self.allowed:
            raise AlertError("CUSTOMER_NOT_ENABLED")
        return hmac.new(self.index_key, customer.encode(), hashlib.sha256).hexdigest()

    def register(self, customer):
        """Create/rotate a read-only API token. Caller must keep it private."""
        owner = self.customer(customer)
        token = secrets.token_urlsafe(32)
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO customers VALUES(?,?)",
                (owner, self.seal({"token_sha256": hashlib.sha256(token.encode()).hexdigest()})),
            )
        return token

    def add(
        self,
        customer,
        channel,
        config,
        *,
        kinds=None,
        watchlist=None,
        min_interval=10,
        max_per_hour=60,
    ):
        owner = self.customer(customer)
        validate(channel, config)
        if channel == "telegram" and config["chat_id"] not in self.telegram_chats.get(customer, []):
            raise AlertError("TELEGRAM_CHAT_NOT_APPROVED")
        kinds, watchlist = subscriptions(kinds or sorted(KINDS), watchlist or [])
        if (
            type(min_interval) is not int
            or not 1 <= min_interval <= 3600
            or type(max_per_hour) is not int
            or not 1 <= max_per_hour <= 3600
        ):
            raise AlertError("INVALID_RATE_LIMIT")
        dest = uuid.uuid4().hex
        value = {
            "channel": channel,
            "config": config,
            "kinds": kinds,
            "watchlist": watchlist,
            "min_interval": min_interval,
            "max_per_hour": max_per_hour,
        }
        with self.db() as db:
            if (
                db.execute(
                    "SELECT count(*) FROM destinations WHERE customer=?", (owner,)
                ).fetchone()[0]
                >= 10
            ):
                raise AlertError("DESTINATION_LIMIT")
            db.execute(
                "INSERT INTO destinations(id,customer,value) VALUES(?,?,?)",
                (dest, owner, self.seal(value)),
            )
        return dest

    def list(self, customer):
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM destinations WHERE customer=? ORDER BY id",
                (self.customer(customer),),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "channel": self.open(r["value"])["channel"],
                    "paused": bool(r["paused"]),
                    "kinds": self.open(r["value"])["kinds"],
                    "watchlist": self.open(r["value"])["watchlist"],
                }
                for r in rows
            ]

    def owned(self, db, customer, destination):
        row = db.execute(
            "SELECT * FROM destinations WHERE id=? AND customer=?",
            (destination, self.customer(customer)),
        ).fetchone()
        if not row:
            raise AlertError("DESTINATION_NOT_FOUND")
        return row

    def pause(self, customer, destination, paused=True):
        if type(paused) is not bool:
            raise AlertError("INVALID_PAUSE")
        with self.db() as db:
            self.owned(db, customer, destination)
            db.execute("UPDATE destinations SET paused=? WHERE id=?", (int(paused), destination))

    def remove(self, customer, destination):
        with self.db() as db:
            self.owned(db, customer, destination)
            db.execute("DELETE FROM destinations WHERE id=?", (destination,))

    def enqueue(self, customer, alert, *, now, destination=None):
        owner = self.customer(customer)
        alert = Alert.parse(alert.event())
        e = alert.event()
        from .model import instant

        if instant(e["available_at"]).timestamp() > now:
            raise AlertError("EVENT_NOT_AVAILABLE")
        if e["kind"] == "duel_hashes" and now >= instant(e["facts"]["open_at"]).timestamp():
            raise AlertError("HASHES_NOT_BEFORE_OPEN")
        key = hashlib.sha256((owner + e["event_id"]).encode()).hexdigest()
        with self.db() as db:
            if destination:
                self.owned(db, customer, destination)
            if e["kind"] in ("duel_hashes", "duel_results"):
                duel_key = hashlib.sha256((owner + e["facts"]["session"]).encode()).hexdigest()
                commitment = db.execute(
                    "SELECT value FROM duels WHERE id=?", (duel_key,)
                ).fetchone()
                if e["kind"] == "duel_hashes":
                    if commitment and self.open(commitment[0])["facts"] != e["facts"]:
                        raise AlertError("DUEL_COMMITMENT_CONFLICT")
                    db.execute(
                        "INSERT OR IGNORE INTO duels VALUES(?,?,?,?)",
                        (duel_key, owner, self.seal(e), now),
                    )
                else:
                    if not commitment:
                        raise AlertError("DUEL_COMMITMENT_MISSING")
                    committed = self.open(commitment[0])
                    if any(
                        committed["facts"][k] != e["facts"][k] for k in ("a1_sha256", "e1_sha256")
                    ):
                        raise AlertError("DUEL_COMMITMENT_CONFLICT")
                    if instant(e["facts"]["close_at"]) <= instant(committed["facts"]["open_at"]):
                        raise AlertError("INVALID_DUEL_SESSION")
                    # A stored draft is not publication. Require a confirmed
                    # outbound receipt before the declared opening instant.
                    ids = [
                        hashlib.sha256((r[0] + committed["event_id"]).encode()).hexdigest()
                        for r in db.execute(
                            "SELECT id FROM destinations WHERE customer=?", (owner,)
                        )
                    ]
                    if not any(
                        db.execute(
                            "SELECT 1 FROM jobs WHERE id=? AND state='sent' AND due<?",
                            (job_id, instant(committed["facts"]["open_at"]).timestamp()),
                        ).fetchone()
                        for job_id in ids
                    ):
                        raise AlertError("DUEL_PREOPEN_DELIVERY_UNPROVEN")
            old = db.execute("SELECT value FROM events WHERE id=?", (key,)).fetchone()
            if old and self.open(old[0]) != e:
                raise AlertError("EVENT_ID_CONFLICT")
            db.execute(
                "INSERT OR IGNORE INTO events(id,customer,value,created) VALUES(?,?,?,?)",
                (key, owner, self.seal(e), now),
            )
            rows = db.execute("SELECT * FROM destinations WHERE customer=?", (owner,)).fetchall()
            queued = 0
            for row in rows:
                if destination and row["id"] != destination:
                    continue
                cfg = self.open(row["value"])
                if row["paused"] or cfg["channel"] in ("email", "api"):
                    continue
                if e["kind"] != "system_test" and (
                    e["kind"] not in cfg["kinds"]
                    or (e["symbols"] and not set(e["symbols"]) & set(cfg["watchlist"]))
                ):
                    continue
                if (
                    db.execute(
                        "SELECT count(*) FROM jobs WHERE destination=? AND state IN ('pending','sending')",
                        (row["id"],),
                    ).fetchone()[0]
                    >= 1000
                ):
                    raise AlertError("QUEUE_FULL")
                job = hashlib.sha256((row["id"] + e["event_id"]).encode()).hexdigest()
                queued += db.execute(
                    "INSERT OR IGNORE INTO jobs(id,destination,value,due,created) VALUES(?,?,?,?,?)",
                    (job, row["id"], self.seal(e), now, now),
                ).rowcount
            return queued

    def claim(self, *, now):
        """Reserve one send and its rate budget atomically, including retries."""
        with self.db() as db:
            rows = db.execute(
                """SELECT j.*, d.value AS config, d.paused, d.customer, d.last_attempt, d.window, d.used
                FROM jobs j JOIN destinations d ON d.id=j.destination
                WHERE j.state IN ('pending','sending') AND j.due<=? AND d.paused=0
                AND NOT EXISTS (SELECT 1 FROM jobs active WHERE active.destination=j.destination
                                AND active.state='sending' AND active.due>?)
                ORDER BY j.due,j.id LIMIT 1000""",
                (now, now),
            ).fetchall()
            for row in rows:
                if row["customer"] not in {self.customer(c) for c in self.allowed}:
                    continue
                cfg = self.open(row["config"])
                e = self.open(row["value"])
                if e["kind"] == "duel_hashes":
                    from .model import instant

                    if now >= instant(e["facts"]["open_at"]).timestamp():
                        db.execute(
                            "UPDATE jobs SET state='dead',code='HASH_WINDOW_CLOSED' WHERE id=?",
                            (row["id"],),
                        )
                        continue
                if row["attempts"] >= 5:
                    db.execute(
                        "UPDATE jobs SET state='dead',code='LEASE_EXHAUSTED' WHERE id=?",
                        (row["id"],),
                    )
                    continue
                window = int(now // 3600)
                used = row["used"] if row["window"] == window else 0
                due = max(
                    now,
                    (row["last_attempt"] + cfg["min_interval"])
                    if row["last_attempt"] is not None
                    else now,
                )
                if used >= cfg["max_per_hour"]:
                    due = max(due, (window + 1) * 3600)
                if due > now:
                    db.execute("UPDATE jobs SET due=? WHERE id=?", (due, row["id"]))
                    continue
                lease = secrets.token_hex(16)
                db.execute(
                    "UPDATE destinations SET last_attempt=?,window=?,used=? WHERE id=?",
                    (now, window, used + 1, row["destination"]),
                )
                db.execute(
                    "UPDATE jobs SET state='sending',attempts=attempts+1,due=?,lease=? WHERE id=?",
                    (now + 30, lease, row["id"]),
                )
                return {
                    "id": row["id"],
                    "destination": row["destination"],
                    "config": cfg,
                    "event": e,
                    "attempts": row["attempts"] + 1,
                    "lease": lease,
                }
        return None

    def finish(self, job, *, state, code, due):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET state=?,code=?,due=?,lease=NULL WHERE id=? AND lease=?",
                (state, code, due, job["id"], job["lease"]),
            )

    def can_send(self, job):
        with self.db() as db:
            row = db.execute(
                "SELECT paused,customer FROM destinations WHERE id=?", (job["destination"],)
            ).fetchone()
            return bool(
                row
                and not row["paused"]
                and row["customer"] in {self.customer(c) for c in self.allowed}
            )

    def jobs(self, customer, *, state=None):
        with self.db() as db:
            rows = db.execute(
                "SELECT j.id,j.state,j.code,j.attempts,j.due FROM jobs j JOIN destinations d ON d.id=j.destination WHERE d.customer=? ORDER BY j.created,j.id",
                (self.customer(customer),),
            ).fetchall()
            return [dict(r) for r in rows if state is None or r["state"] == state]

    def feed(self, token, *, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise AlertError("INVALID_PAGINATION")
        supplied = hashlib.sha256(token.encode()).hexdigest()
        with self.db() as db:
            owners = {self.customer(c) for c in self.allowed}
            owner = None
            for row in db.execute("SELECT * FROM customers"):
                if row["id"] in owners and hmac.compare_digest(
                    self.open(row["value"])["token_sha256"], supplied
                ):
                    owner = row["id"]
            if owner is None:
                raise AlertError("UNAUTHORIZED")
            return [
                {**Alert.parse(self.open(row["value"])).payload(), "cursor": row["seq"]}
                for row in db.execute(
                    "SELECT * FROM events WHERE customer=? AND seq>? ORDER BY seq LIMIT ?",
                    (owner, after, limit),
                )
            ]

    def prune(self, *, before):
        with self.db() as db:
            jobs = db.execute(
                "DELETE FROM jobs WHERE created<? AND state IN ('sent','dead')", (before,)
            ).rowcount
            events = db.execute("DELETE FROM events WHERE created<?", (before,)).rowcount
            db.execute("DELETE FROM duels WHERE created<?", (before,))
            return {"jobs": jobs, "events": events}
