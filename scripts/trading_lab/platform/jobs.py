"""Bounded persistent jobs, with one isolated spawned worker per lab root.

Only named local workloads are dispatched. SQLite stores structured log codes,
never exception text, HTTP bodies or secrets. CPU/memory/file/wall budgets are
enforced; no workload executes inside an API request thread.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sqlite3
import threading
import time
import uuid

from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical

TERMINAL = frozenset({"COMPLETE", "FAILED", "CANCELLED", "BLOCKED"})
KINDS = frozenset({"dataset", "experiment"})
MAX_PAYLOAD_BYTES = 65536
MAX_ARTIFACT_BYTES = 24 * 1024 * 1024


class ArtifactIntegrityError(ValueError):
    """A stored artifact no longer matches its content identity."""


@dataclass(frozen=True)
class ResourceLimits:
    wall_seconds: int = 120
    cpu_seconds: int = 60
    memory_mb: int = 1024
    output_mb: int = 32

    def __post_init__(self):
        for name, minimum, maximum in (("wall_seconds", 1, 180), ("cpu_seconds", 1, 90),
                                       ("memory_mb", 256, 1024), ("output_mb", 8, 32)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("resource budget outside lab limits")


class JobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.root / "jobs.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, limits_json TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    result_hash TEXT, error_code TEXT);
                CREATE TABLE IF NOT EXISTS logs (
                    sequence INTEGER PRIMARY KEY, job_id TEXT NOT NULL, code TEXT NOT NULL,
                    progress REAL NOT NULL, at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    hash TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def submit(self, kind, payload, limits=None, *, on_submit=None):
        if kind not in KINDS:
            raise ValueError("unknown job kind")
        encoded = canonical_bytes(payload)
        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise ValueError("job payload too large")
        limits = limits or ResourceLimits()
        identifier, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= 1000:
                raise ValueError("persistent lab job budget exhausted")
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('QUEUED','RUNNING')").fetchone()[0] >= 8:
                raise ValueError("lab queue budget exhausted")
            # Internal admission hook: tenant ownership and quotas commit with
            # the queued job, or roll back together. Never supplied by HTTP.
            if on_submit is not None:
                on_submit(db, identifier)
            db.execute("INSERT INTO jobs(id,kind,state,payload,limits_json,created_at,updated_at) VALUES(?,?,'QUEUED',?,?,?,?)",
                       (identifier, kind, encoded.decode(), json.dumps(asdict(limits)), now, now))
            db.execute("INSERT INTO logs(job_id,code,progress,at) VALUES(?,'QUEUED',0,?)", (identifier, now))
        return identifier

    def status(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError("unknown job")
            data = dict(row)
            data.pop("payload")
            data["limits"] = json.loads(data.pop("limits_json"))
            data["cancel_requested"] = bool(data["cancel_requested"])
            data["logs"] = [dict(r) for r in db.execute(
                "SELECT sequence,code,progress,at FROM logs WHERE job_id=? ORDER BY sequence", (identifier,))]
            return data

    def list(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 50")]
        return [self.status(i) for i in ids]

    def cancel(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError("unknown job")
            if row[0] not in TERMINAL:
                db.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (time.time(), identifier))
                if row[0] == "QUEUED":
                    db.execute("UPDATE jobs SET state='CANCELLED' WHERE id=?", (identifier,))
        return self.status(identifier)

    def checkpoint(self, identifier, progress, code):
        if not 0 <= progress <= 1 or code not in {"STARTED", "DATASET_BUILT", "TRAINING", "VALIDATING", "BACKTEST", "SHADOW", "RESULT_READY"}:
            raise ValueError("invalid structured progress")
        with self.connect() as db:
            row = db.execute("SELECT state,cancel_requested,progress FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None or row[0] != "RUNNING" or row[1]:
                raise InterruptedError("job no longer running")
            if progress < row[2]:
                raise ValueError("progress must be monotone")
            db.execute("UPDATE jobs SET progress=?,updated_at=? WHERE id=?", (progress, time.time(), identifier))
            db.execute("INSERT INTO logs(job_id,code,progress,at) VALUES(?,?,?,?)", (identifier, code, progress, time.time()))

    def put_artifact(self, kind, payload, *, identity=None):
        encoded = canonical_bytes(payload)
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact budget exceeded")
        key = identity or sha256_canonical(payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT kind,payload FROM artifacts WHERE hash=?", (key,)).fetchone()
            if existing and (existing[0] != kind or existing[1] != encoded.decode()):
                raise ValueError("immutable artifact identity collision")
            if not existing and db.execute("SELECT coalesce(sum(length(payload)),0) FROM artifacts").fetchone()[0] + len(encoded) > 128 * 1024 * 1024:
                raise ValueError("persistent artifact budget exhausted")
            db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?)", (key, kind, encoded.decode()))
        return key

    def artifact(self, key, *, kind):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM artifacts WHERE hash=? AND kind=?", (key, kind)).fetchone()
        if row is None:
            raise KeyError("unknown artifact")
        try:
            payload = json.loads(row[0])
            if kind == "dataset":
                from scripts.trading_lab.platform.datasets import verify_dataset
                actual = verify_dataset(payload).identity
            else:
                actual = sha256_canonical(payload)
            if actual != key:
                raise ValueError("artifact digest mismatch")
        except (ValueError, TypeError, KeyError):
            raise ArtifactIntegrityError("artifact digest mismatch") from None
        return payload

    def finish(self, identifier, state, *, result_hash=None, error_code=None):
        if state not in TERMINAL:
            raise ValueError("invalid terminal state")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state,cancel_requested FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None or row[0] in TERMINAL:
                return
            if row[1]:
                state, result_hash, error_code = "CANCELLED", None, None
            db.execute("UPDATE jobs SET state=?,result_hash=?,error_code=?,progress=CASE WHEN ?='COMPLETE' THEN 1 ELSE progress END,updated_at=? WHERE id=?",
                       (state, result_hash, error_code, state, time.time(), identifier))
            db.execute("INSERT INTO logs(job_id,code,progress,at) SELECT id,?,progress,? FROM jobs WHERE id=?", (state, time.time(), identifier))

    def result(self, identifier):
        status = self.status(identifier)
        if status["state"] != "COMPLETE":
            response = {"state": status["state"], "result": None, "error_code": status["error_code"]}
            if status["kind"] == "experiment":
                from dataclasses import replace
                from scripts.trading_lab.platform.contracts import ExperimentManifest
                with self.connect() as db:
                    raw = json.loads(db.execute("SELECT payload FROM jobs WHERE id=?", (identifier,)).fetchone()[0])
                prepared = ExperimentManifest.from_dict(raw["prepared"])
                view = replace(prepared, status="PREPARED" if status["state"] == "QUEUED" else status["state"])
                response.update(manifest=view.to_dict(), fingerprint=view.identity)
            return response
        return {"state": "COMPLETE", "result_hash": status["result_hash"],
                "result": self.artifact(status["result_hash"], kind="result")}


def _worker(root, identifier, parent_pid):
    # Set before importing any numeric stack in this spawned process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    store = JobStore(root)
    status = store.status(identifier)
    limits = ResourceLimits(**status["limits"])
    deadline = time.monotonic() + limits.wall_seconds

    def watch():
        while True:
            time.sleep(.1)
            try:
                if os.getppid() != parent_pid:
                    store.finish(identifier, "FAILED", error_code="PARENT_INTERRUPTED")
                    os._exit(1)
                if store.status(identifier)["cancel_requested"]:
                    store.finish(identifier, "CANCELLED")
                    os._exit(0)
                if time.monotonic() >= deadline:
                    store.finish(identifier, "FAILED", error_code="WALL_LIMIT")
                    os._exit(1)
            except Exception:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    try:
        # Covers orphan workers during API restarts as well as normal execution.
        with open(store.root / "execution.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (limits.output_mb * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            store.checkpoint(identifier, .01, "STARTED")
            with store.connect() as db:
                raw = db.execute("SELECT kind,payload FROM jobs WHERE id=?", (identifier,)).fetchone()
            from scripts.trading_lab.platform.experiments import execute_job
            result = execute_job(store, identifier, raw[0], json.loads(raw[1]))
            store.checkpoint(identifier, .99, "RESULT_READY")
            result_hash = store.put_artifact("result", result)
            store.finish(identifier, "COMPLETE", result_hash=result_hash)
    except InterruptedError:
        store.finish(identifier, "CANCELLED")
    except MemoryError:
        store.finish(identifier, "FAILED", error_code="MEMORY_LIMIT")
    except Exception as error:
        # Deliberately omit exception messages: they can contain paths and input data.
        store.finish(identifier, "FAILED", error_code="WORKLOAD_" + type(error).__name__)


class JobRunner:
    def __init__(self, root):
        self.store = JobStore(root)
        self._lock = open(self.store.root / "runner.lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise ValueError("a runner already owns this lab root") from None
        with self.store.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE state='RUNNING'")]
        for identifier in ids:
            self.store.finish(identifier, "FAILED", error_code="RUNNER_INTERRUPTED")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def _supervise(self):
        while not self._stop.is_set():
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT id FROM jobs WHERE state='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
                if row:
                    db.execute("UPDATE jobs SET state='RUNNING',updated_at=? WHERE id=? AND state='QUEUED'", (time.time(), row[0]))
            if row is None:
                self._stop.wait(.1)
                continue
            identifier = row[0]
            process = multiprocessing.get_context("spawn").Process(
                target=_worker, args=(str(self.store.root), identifier, os.getpid()), daemon=True)
            try:
                process.start()
                with self.store.connect() as db:
                    db.execute("UPDATE jobs SET worker_pid=? WHERE id=?", (process.pid, identifier))
                started = time.monotonic()
                while process.is_alive():
                    status = self.store.status(identifier)
                    if self._stop.is_set() or status["cancel_requested"] or time.monotonic() - started > status["limits"]["wall_seconds"] + 1:
                        process.terminate()
                        process.join(2)
                        if process.is_alive():
                            process.kill()
                        break
                    process.join(.1)
                process.join(2)
                status = self.store.status(identifier)
                if status["state"] not in TERMINAL:
                    code = "RUNNER_STOPPED" if self._stop.is_set() else (
                        "WALL_LIMIT" if time.monotonic() - started > status["limits"]["wall_seconds"] else "WORKER_EXIT")
                    self.store.finish(identifier, "FAILED", error_code=code)
            except Exception:
                if process.pid and process.is_alive():
                    process.terminate()
                    process.join(2)
                self.store.finish(identifier, "FAILED", error_code="WORKER_START_OR_SUPERVISION")
            finally:
                if process.pid and not process.is_alive():
                    process.close()

    def close(self):
        self._stop.set()
        self._thread.join(5)
        if self._thread.is_alive():
            raise RuntimeError("runner has not stopped")
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
