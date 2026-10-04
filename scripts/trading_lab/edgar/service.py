"""The EDGAR capture runner. It sends nothing without an operator authorization file that names this spec,
the CIKs, a request budget, an expiry and the declared User-Agent; it stops at the first of: the budget, the
expiry, a stop signal. `--check` validates the spec binding, the store and the authorization without
writing a byte or opening a socket.

    python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE --check
    python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE

Authorization file (JSON): {"authorizes": "sec_edgar_submissions_v1", "spec_hash": "<edgar spec hash>",
"ciks": ["320193"], "max_requests": 6, "not_after": "2026-10-05T18:00:00+00:00",
"user_agent": "Organization contact@example.org", "granted_by": "operator", "granted_at": "..."}
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.edgar.collector import EdgarCollector
from scripts.trading_lab.edgar.listing import cik10
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.edgar.transport import HttpsFetcher
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso, parse_iso
from scripts.trading_lab.sources.store import StoreBusy, admit_existing

MAX_AUTHORIZED_REQUESTS = 50
STORAGE_STALL_THRESHOLD_S = 10.0  # operational supervision; not an EDGAR parsing/spec rule


class CaptureRefused(RuntimeError):
    """No valid authorization: nothing is sent."""


class RealClock:
    def __init__(self, stop: threading.Event):
        self._stop = stop

    def wall(self) -> datetime:
        return datetime.now(timezone.utc)

    def mono(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self._stop.wait(max(seconds, 0.0))


def authorization_scope(auth: dict) -> dict:
    """Public bounds only: never persist the declared contact or operator details."""
    return {name: auth[name] for name in ("authorizes", "spec_hash", "ciks", "max_requests", "not_after")}


class StorageWatch:
    """Watch operation markers from a separate thread, including when the runner itself is stalled.

    The monitor never takes the store lock or writes. The runner commits ended incidents before
    reopening the request gate. A stalled incident write is itself monitored.
    """

    def __init__(self, collector, clock, *, threshold_s=STORAGE_STALL_THRESHOLD_S, log=print):
        if threshold_s <= 0:
            raise ValueError("storage stall threshold must be positive")
        self.collector, self.store, self.clock = collector, collector.store, clock
        self.threshold_s, self.log = threshold_s, log
        self.store.use_clock(clock.mono)
        self.incident = None
        self.pending = []
        self.alert_errors = []
        self._lock = threading.Lock()

    def watch_storage(self) -> None:
        alerts = []
        with self._lock:
            stalled = self.store.stalled()
            now = self.clock.mono()
            if stalled is not None and stalled[1] >= self.threshold_s:
                if self.incident is None:
                    self.incident = {"operation": stalled[0], "started_mono": now - stalled[1],
                                     "wall_detected": iso(self.clock.wall())}
                    alerts.append("STORAGE_INCIDENT_STARTED")
            elif self.incident is not None:
                self.pending.append(dict(self.incident, wall_ended=iso(self.clock.wall()),
                                         stalled_s=round(now - self.incident["started_mono"], 3)))
                self.incident = None
                alerts.append("STORAGE_INCIDENT_ENDED")
            self.collector.limiter.suspended = self.incident is not None or bool(self.pending)
        for kind in alerts:
            try:
                self.log(json.dumps({"alert": kind, "epoch": self.collector.epoch,
                                     "threshold_s": self.threshold_s}, sort_keys=True))
            except Exception as exc:
                self.alert_errors.append(type(exc).__name__)

    def record_incidents(self) -> bool:
        self.watch_storage()
        while True:
            with self._lock:
                if self.incident is not None:
                    return False
                if not self.pending:
                    self.collector.limiter.suspended = False
                    return True
                incident = self.pending[0]
            self.store.set_wait_bound(0.2)
            try:
                self.store.append("STORAGE_INCIDENT", [("STORAGE_INCIDENT", None, {
                    "epoch": self.collector.epoch, "threshold_s": self.threshold_s,
                    **{k: v for k, v in incident.items() if k != "started_mono"}})])
            except StoreBusy:
                return False
            finally:
                self.store.set_wait_bound(None)
            with self._lock:
                self.pending.pop(0)
            self.watch_storage()


def load_authorization(path: Path, now: datetime) -> dict:
    try:
        auth = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureRefused(f"no readable authorization at {path}: {exc}") from exc
    problems = []
    if auth.get("authorizes") != spec.PROVIDER_ID:
        problems.append(f"it authorizes {auth.get('authorizes')!r}, not {spec.PROVIDER_ID}")
    if auth.get("spec_hash") != spec.SPEC_HASH:
        problems.append("it names another EDGAR spec revision")
    ciks = auth.get("ciks")
    try:
        padded = [cik10(c) for c in ciks] if isinstance(ciks, list) else None
    except ValueError as exc:
        padded = None
        problems.append(str(exc))
    if not padded or not 1 <= len(padded) <= spec.WATCHLIST_MAX:
        problems.append(f"it must name 1 to {spec.WATCHLIST_MAX} CIKs")
    budget = auth.get("max_requests")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= MAX_AUTHORIZED_REQUESTS:
        problems.append(f"max_requests must be an integer from 1 to {MAX_AUTHORIZED_REQUESTS}")
    try:
        not_after = parse_iso(auth["not_after"])
        if not_after <= now:
            problems.append(f"it expired at {auth['not_after']}")
    except (KeyError, TypeError, ValueError):
        problems.append("not_after must be an ISO-8601 instant with its offset")
    if not isinstance(auth.get("user_agent"), str) or not spec.USER_AGENT.match(auth["user_agent"].strip()):
        problems.append("user_agent must name an organization and a contact e-mail")
    if not auth.get("granted_by"):
        problems.append("granted_by is required")
    if problems:
        raise CaptureRefused("authorization refused: " + "; ".join(problems))
    return dict(auth, ciks=padded)


def check(store_dir: Path, authorization: Path, *, now: datetime | None = None) -> dict:
    """Preflight without network or write: spec binding, store opening rule, authorization."""
    out = {"spec": [spec.SPEC_REVISION, spec.verify_spec_binding()]}
    admit_existing(Path(store_dir) / EdgarStore.DB_NAME, schema_version=spec.SCHEMA_VERSION, spec_hash=spec.SPEC_HASH)
    auth = load_authorization(authorization, now or datetime.now(timezone.utc))
    out.update(store=str(store_dir), ciks=auth["ciks"], max_requests=auth["max_requests"], not_after=auth["not_after"])
    return out


def run(store_dir: Path, authorization: Path, *, fetcher=None, clock=None, stop: threading.Event | None = None,
        log=print, stall_threshold_s=STORAGE_STALL_THRESHOLD_S, monitor_interval_s=0.1) -> dict:
    if stall_threshold_s <= 0 or monitor_interval_s <= 0:
        raise ValueError("storage watch intervals must be positive")
    stop = stop or threading.Event()
    clock = clock or RealClock(stop)
    auth = load_authorization(authorization, clock.wall())
    not_after = parse_iso(auth["not_after"])
    store = EdgarStore(Path(store_dir), wall_clock=clock.wall)
    fetcher = fetcher or HttpsFetcher(auth["user_agent"], wall=clock.wall, mono=clock.mono)
    collector = None
    monitor = None
    monitor_stop = threading.Event()
    try:
        collector = EdgarCollector(store, fetcher, clock, boot_id=f"boot-{int(time.time())}")
        watch = StorageWatch(collector, clock, threshold_s=stall_threshold_s, log=log)

        def monitor_storage():
            while not monitor_stop.wait(monitor_interval_s):
                watch.watch_storage()

        monitor = threading.Thread(target=monitor_storage, name="edgar-storage-watch", daemon=True)
        monitor.start()
        collector.submit_watchlist(auth["ciks"])
        auth_hash = sha256_canonical(auth)
        store.append("RUN_STARTED", [("RUN_STARTED", collector.epoch, {
            "epoch": collector.epoch, "authorization_sha256": auth_hash, "authorization": authorization_scope(auth)})])
        log(f"EDGAR capture: epoch {collector.epoch}, reconciled {collector.reconciled}, "
            f"budget {auth['max_requests']} requests until {auth['not_after']}")

        def sent() -> int:
            # Restarting with the same grant must not recreate its request budget.
            epochs = {r.body["epoch"] for r in store.rows("RUN_STARTED")
                      if r.body["authorization_sha256"] == auth_hash}
            return sum(1 for t in store.rows("TRANSPORT_INVOKED") if t.body["epoch"] in epochs)

        def request_allowed() -> bool:
            # This gate also runs after TRANSPORT_INVOKED: a slow commit can cross the expiry.
            while not watch.record_incidents():
                if stop.is_set() or clock.wall() >= not_after:
                    return False
                clock.sleep(0.1)
            return not stop.is_set() and clock.wall() < not_after

        reason = None
        while reason is None:
            for cik in collector.watchlist():
                if stop.is_set():
                    reason = "stopped"
                elif clock.wall() >= not_after:
                    reason = "authorization expired"
                elif sent() >= auth["max_requests"]:
                    reason = "request budget spent"
                if reason:
                    break
                result = collector.poll(cik, request_allowed=request_allowed)
                log(f"{iso(clock.wall())} {cik} {result['status']} {result.get('outcome', '')}")
                if result["status"] == "SOURCE_THROTTLED":
                    reason = "throttled (403/429): the trial stops, no further request"
                    break
                if result["status"] == "STOPPED":
                    reason = "stopped" if stop.is_set() else "authorization expired"
                    break
            if reason is None:
                clock.sleep(spec.POLL_INTERVAL_S)
        watch.record_incidents()
        summary = {"reason": reason, "requests": sent(), "records": len(store.rows("RESPONSE")),
                   "revisions": len(store.rows("FILING_REVISION")), "epoch": collector.epoch}
        store.append("RUN_ENDED", [("RUN_ENDED", collector.epoch, summary)])
        watch.record_incidents()
        log(f"EDGAR capture ended: {summary}")
        return summary
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join()
        if collector is not None:
            collector.close()
        store.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate only: no network, no write")
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(check(args.store, args.authorization), sort_keys=True))
            return 0
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
        signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
        run(args.store, args.authorization, stop=stop)
        return 0
    except CaptureRefused as exc:
        print(f"EDGAR capture refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
