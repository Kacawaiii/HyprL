"""The autonomous FOMC owner service (spec revision 23).

One owner thread ticks and never runs a task itself:
1. storage watch (storage_incident): in-process markers tell, without the store lock, whether a store
   operation has been in progress for 10 s. An incident raises an alert, suspends every FIX15 grant
   and promises no durable decision; when it ends the owner records it and reconciles from durable
   state (first outcome wins, no repair request, no budget recreated). A separate monitor thread runs
   the same watch, so an incident is detected even when the stalled operation is the owner's own;
2. `collector.reconcile()` closes what reached its bound: LOCAL_PERSISTENCE_FAILED at the 120 s
   admission bound (even while the saving task is blocked), INTERRUPTED 600 s after TRANSPORT_INVOKED
   or at once for an earlier epoch, DEAD runs 600 s after their start, replacement, poison, terminals;
3. grant dispatch (selection.grant_dispatch): at the instant FIX15 admits a start, a waiting redirect
   continuation with the smallest TRANSPORT_INVOKED goes first; otherwise one selection decision is
   made and its TRANSPORT_INVOKED committed before the next. Work is selected only when its grant
   can be attributed, so nothing is selected twice and no grant is consumed for a record that does not
   commit. Logical fetches of different items and the feed poll then run concurrently in worker
   threads, within the durable one-in-flight rules and with no other cap.

The owner waits for the store at most `owner_wait_s`: behind a stalled operation it gets StoreBusy
and skips the tick's durable work instead of blocking. A hung task keeps only its own thread; its late
result is fenced (first outcome wins, LATE_EVIDENCE, run fences), and a task only ever removes its own
entry from a registry. A restart is a new owner: attempts of earlier epochs without an outcome are
INTERRUPTED at its start and every durable budget stays counted.

    python -m scripts.trading_lab.fomc.service --store PATH --check   # preflight only, no network
    python -m scripts.trading_lab.fomc.service --store PATH           # real capture
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

from scripts.trading_lab.fomc import ledger, spec
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.store import StoreBusy, StoreRejected, _admit_existing

# Real capture is refused while any entry remains. Empty since spec revision 23: the 120 s bound is an
# admission bound and a stalled store is a storage incident (registry, "Runtime verdict").
CAPTURE_BLOCKERS: tuple[str, ...] = ()


class SystemClock:
    """Production clock: UTC wall reading, monotonic clock, real sleep."""

    @staticmethod
    def wall() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def mono() -> float:
        return time.monotonic()

    @staticmethod
    def sleep(seconds: float) -> None:
        time.sleep(max(seconds, 0.0))


class _Slot:
    """A worker waiting for the dispatcher to grant its validated redirect continuation."""

    def __init__(self, hop: int = 1, url: str = ""):
        self.hop, self.url = hop, url
        self.event = threading.Event()
        self.granted: tuple | None = None

    def release(self, granted: tuple | None) -> None:
        self.granted = granted
        self.event.set()


def _stderr_alert(alert: dict) -> None:
    print("FOMC-ALERT " + json.dumps(alert, sort_keys=True), file=sys.stderr, flush=True)


class FomcService:
    def __init__(self, collector, clock, *, tick_s: float = 1.0, settle_s: float = 0.0, on_alert=_stderr_alert,
                 owner_wait_s: float = 0.2, stall_threshold_s: float = spec.STORAGE_STALL_THRESHOLD_S):
        """`settle_s` > 0 is for simulated clocks only: after each tick the owner waits (real time, at
        most settle_s) for workers that are neither parked nor waiting for a grant, so simulated time
        advances between ticks deterministically. In production it is 0: the owner never waits on a
        worker. `on_alert(alert)` receives every storage alert (default: one JSON line on stderr)."""
        self.collector, self.clock, self.store = collector, clock, collector.store
        self.tick_s, self.settle_s, self.owner_wait_s = tick_s, settle_s, owner_wait_s
        self.stall_threshold_s, self.on_alert = stall_threshold_s, on_alert
        self.store.use_clock(clock.mono)  # operation markers and the watch share the service clock
        collector.inline = False
        collector.dispatch_run = self._dispatch_run
        collector.continuation = self._continuation
        self.tasks: dict[int, threading.Thread] = {}  # attempt -> the worker of its logical fetch
        self.workers: list[threading.Thread] = []
        self.errors: list[BaseException] = []
        self.alerts: list[dict] = []
        self.incident: dict | None = None  # the active storage incident
        self.ticks = self.busy_ticks = 0
        self.stopping = False
        self._slots: dict[int, _Slot] = {}  # attempt -> continuation request awaiting a grant
        self._unrecorded: list[dict] = []  # ended incidents not yet committed to the store
        self._parked: dict[int, threading.Event | None] = {}  # thread -> what it waits on (not working)
        self._lock = threading.Lock()  # tasks, slots
        self._incident_lock = threading.Lock()

    # ---- storage incident (storage_incident) ------------------------------------------------------
    def watch_storage(self) -> None:
        """Detect the start and the end of a storage incident from the store's operation markers.
        Touches neither the store lock nor the store; safe from the owner and from the monitor."""
        with self._incident_lock:
            stalled = self.store.stalled()
            now = self.clock.mono()
            if stalled is not None and stalled[1] >= self.stall_threshold_s:
                if self.incident is None:
                    self.incident = {"operation": stalled[0], "started_mono": now - stalled[1],
                                     "wall_detected": iso(self.clock.wall()), "fetches_in_flight": len(self.tasks)}
                    self.collector.limiter.suspended = True  # no grant, initial or continuation
                    self._alert("STORAGE_INCIDENT_STARTED", self.incident, stalled_s=round(stalled[1], 3))
            elif self.incident is not None:
                ended = dict(self.incident, stalled_s=round(now - self.incident["started_mono"], 3),
                             wall_ended=iso(self.clock.wall()))
                self.incident = None
                self.collector.limiter.suspended = False
                self._unrecorded.append(ended)
                self._alert("STORAGE_INCIDENT_ENDED", ended, stalled_s=ended["stalled_s"])

    def _alert(self, kind: str, incident: dict, *, stalled_s: float) -> None:
        alert = {"alert": kind, "provider_id": spec.PROVIDER_ID, "store": str(self.store.root),
                 "epoch": self.collector.epoch, "operation": incident["operation"], "stalled_s": stalled_s,
                 "threshold_s": self.stall_threshold_s, "fetches_in_flight": incident["fetches_in_flight"],
                 "grants_suspended": kind == "STORAGE_INCIDENT_STARTED", "wall": iso(self.clock.wall()),
                 "action": ("check the storage device; no request is granted and no decision is durable until the "
                            "store writes again" if kind == "STORAGE_INCIDENT_STARTED" else
                            "store writes again; reconciled from durable state, first outcome wins")}
        self.alerts.append(alert)
        try:
            self.on_alert(alert)
        except Exception as exc:  # an alert sink never stops the owner
            self.errors.append(exc)

    def _record_incidents(self) -> None:
        """Commit one STORAGE_INCIDENT record per ended incident, once the store writes again."""
        while self._unrecorded:
            incident = self._unrecorded[0]
            self.store.append("STORAGE_INCIDENT", [("STORAGE_INCIDENT", None, {
                "epoch": self.collector.epoch, "operation": incident["operation"], "stalled_s": incident["stalled_s"],
                "threshold_s": self.stall_threshold_s, "fetches_in_flight": incident["fetches_in_flight"],
                "wall_detected": incident["wall_detected"], "wall_ended": incident["wall_ended"]})])
            self._unrecorded.pop(0)

    # ---- workers ------------------------------------------------------------------------------
    def _spawn(self, name: str, target, *args, attempt: int | None = None) -> threading.Thread:
        def body():
            try:
                target(*args)
            except BaseException as exc:  # a task failure never reaches the owner thread
                self.errors.append(exc)
            finally:
                if attempt is not None:
                    with self._lock:  # a worker only ever removes its own task
                        if self.tasks.get(attempt) is thread:
                            del self.tasks[attempt]
        thread = threading.Thread(target=body, name=name, daemon=True)
        with self._lock:
            if attempt is not None:
                self.tasks[attempt] = thread
            self.workers.append(thread)
        thread.start()
        return thread

    def _dispatch_run(self, seq: int, run_id: str) -> None:
        self._spawn(f"fomc-run-{seq}", self.collector.finish_processing, seq, run_id)

    def _continuation(self, attempt: int, hop: int, url: str) -> tuple | None:
        """Called by a fetch worker whose redirect target is validated: wait for the dispatcher."""
        slot = _Slot(hop, url)
        with self._lock:
            if self.stopping:
                return None
            self._slots[attempt] = slot
        ident = threading.get_ident()
        self._parked[ident] = slot.event  # waiting for a grant, not working
        try:
            slot.event.wait()
        finally:
            self._parked.pop(ident, None)
        return slot.granted

    def park(self, release: threading.Event, timeout: float | None = None) -> None:
        """Test and soak hook: block the calling task until `release` is set (a hung task)."""
        ident = threading.get_ident()
        self._parked[ident] = release
        try:
            release.wait(timeout)
        finally:
            self._parked.pop(ident, None)

    def unpark(self, release: threading.Event, timeout: float = 5.0) -> None:
        """Test and soak hook: set `release`, wait (real time) until the tasks parked on it have resumed,
        then settle, so a simulated clock never runs ahead of a task that is working again."""
        release.set()
        deadline = time.monotonic() + timeout
        while release in list(self._parked.values()) and time.monotonic() < deadline:
            time.sleep(0.001)
        self._settle()

    def _settle(self) -> None:
        if self.settle_s > 0:
            deadline = time.monotonic() + self.settle_s
            for thread in list(self.workers):
                while thread.is_alive() and time.monotonic() < deadline:
                    parked = self._parked.get(thread.ident)
                    if parked is not None and not parked.is_set():
                        break
                    # A set event has released the worker even if it has not
                    # yet cleared its marker. Let it resume before advancing
                    # a simulated clock to another grant or deadline.
                    thread.join(0.002)
        with self._lock:
            self.workers = [t for t in self.workers if t.is_alive()]

    # ---- grant dispatch (selection.grant_dispatch) ------------------------------------------------
    def _next_continuation(self) -> tuple[int, _Slot] | None:
        """The waiting continuation of the attempt with the smallest TRANSPORT_INVOKED; requests of
        attempts that already have an outcome are released without a grant."""
        with self._lock:
            for attempt in sorted(self._slots):
                if ledger.outcome_of(self.store, attempt) is not None:
                    self._slots.pop(attempt).release(None)
                    continue
                return attempt, self._slots[attempt]
        return None

    def _dispatch(self) -> None:
        limiter = self.collector.limiter
        while not limiter.suspended and limiter.admissible(self.clock.mono()):
            waiting = self._next_continuation()
            if waiting is not None:  # continuations first, before any new attempt
                attempt, slot = waiting
                now = self.clock.mono()
                deadline = self.collector.transport.deadline()
                invoked = self.store.row_at("TRANSPORT_INVOKED", attempt)
                group = "FEED" if invoked.body["kind"] == "FEED_POLL" else invoked.body["sid"]
                try:  # journal first: a grant that is not durable in the journal is never consumed
                    self.collector.journal_grant(mono=now, attempt=attempt, group=group, hop=slot.hop, url=slot.url,
                                                 kind="CONTINUATION")
                except StoreBusy:
                    raise
                except Exception as exc:
                    self.errors.append(exc)
                    return
                limiter.register(now)
                with self._lock:
                    self._slots.pop(attempt, None)
                slot.release((now, deadline))
                continue
            chosen = self.collector.select()  # selected only now that a grant can be attributed
            if chosen is None:
                return
            started = self.collector.begin(chosen)  # TRANSPORT_INVOKED committed, then the start consumed
            if started is None:
                return
            self._spawn(f"fomc-fetch-{started['attempt']}", self.collector.complete, started, attempt=started["attempt"])

    # ---- the owner loop -----------------------------------------------------------------------
    def tick(self) -> None:
        self.ticks += 1
        self.store.set_wait_bound(self.owner_wait_s)  # this thread never waits behind a stalled store
        self.watch_storage()
        try:
            self._record_incidents()
            self.collector.reconcile()
            if self.incident is None and not self.stopping:
                self._dispatch()
        except StoreBusy:
            self.busy_ticks += 1  # no durable decision while the store cannot write
        self._settle()

    def run_for(self, seconds: float, between=None) -> None:
        """Tick for `seconds` of the service clock; `between(service)` runs after every tick."""
        end = self.clock.mono() + seconds
        while self.clock.mono() < end:
            self.tick()
            if between is not None:
                between(self)
            self.clock.sleep(self.tick_s)

    def run_until(self, stop: threading.Event) -> list[int]:
        """Production loop: tick until `stop`, with the storage monitor beside it, then stop cleanly."""
        def monitor():
            while not stop.is_set():
                self.watch_storage()
                stop.wait(1.0)
        threading.Thread(target=monitor, name="fomc-storage-monitor", daemon=True).start()
        while not stop.is_set():
            self.tick()
            stop.wait(self.tick_s)
        return self.stop()

    def stop(self, wait_s: float = 75.0) -> list[int]:
        """Clean stop: no new grant; fetches in flight finish within their own bounds (a physical
        attempt ends within 60 s) and are committed and processed; then ownership is released. Returns
        the attempts left without an outcome (none on a clean stop): the next owner interrupts them."""
        self.stopping = True
        with self._lock:
            slots, self._slots = list(self._slots.values()), {}
        for slot in slots:
            slot.release(None)  # no continuation grant: the attempt ends SOURCE_UNAVAILABLE and counts
        deadline = time.monotonic() + wait_s
        self.store.set_wait_bound(self.owner_wait_s)
        for _ in range(4):  # fetch workers, then the processing runs their records need
            for thread in list(self.workers):
                while thread.is_alive() and thread.ident not in self._parked and time.monotonic() < deadline:
                    thread.join(0.01)
            try:
                self.watch_storage()
                self._record_incidents()
                self.collector.reconcile()
            except StoreBusy:
                break
            with self._lock:
                self.workers = [t for t in self.workers if t.is_alive()]
                if not [t for t in self.workers if t.ident not in self._parked]:
                    break
        try:
            left = [a.seq for a in ledger.attempts_without_outcome(self.store)]
        except StoreBusy:
            left = sorted(self.tasks)
        self.collector.close()
        return left


def submit_manifest_once(collector, raw: bytes, provenance: str) -> dict:
    """Submit a backfill manifest through the owner unless a manifest with the same digest exists, so a
    restart never opens a second run of the same entries."""
    digest = spec.sha256_bytes(raw)
    existing = collector.store.rows("MANIFEST", key=digest)
    if existing:
        result = dict(existing[0].body, seq=existing[0].seq, submitted=False)
    else:
        result = dict(collector.submit_manifest(raw, provenance), submitted=True)
    print(f"FOMC manifest {digest[:12]} valid={result['valid']} submitted={result['submitted']} seq={result['seq']}",
          file=sys.stderr, flush=True)
    return result


def preflight(store_path: Path) -> list[str]:
    """Everything that can be checked without the network, without creating the store and without
    taking ownership. Returns the problems found (empty: ready)."""
    problems = list(CAPTURE_BLOCKERS)
    try:
        spec.verify_spec_binding()
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        _admit_existing(Path(store_path) / "fomc.sqlite3")
    except StoreRejected as exc:
        problems.append(str(exc))
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="store directory (created if absent)")
    parser.add_argument("--tick", type=float, default=1.0, help="owner tick in seconds (default 1)")
    parser.add_argument("--check", action="store_true", help="preflight only: no store created, no request")
    parser.add_argument("--manifest", type=Path, help="a HISTORICAL_BACKFILL manifest submitted by this owner at start, "
                                                      "once per manifest digest")
    args = parser.parse_args(argv)
    problems = preflight(args.store)
    if problems:
        print("FOMC capture refused:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    if args.check:
        print(f"FOMC capture preflight ok: spec revision {spec.SPEC_REVISION} {spec.SPEC_HASH}, "
              f"no capture blocker, store {args.store} admissible")
        return 0
    from scripts.trading_lab.fomc.collector import Collector
    from scripts.trading_lab.fomc.store import FomcStore
    from scripts.trading_lab.fomc.transport import HttpsConnector
    clock = SystemClock()
    store = FomcStore(args.store, wall_clock=clock.wall, mono=clock.mono)
    collector = Collector(store, HttpsConnector(), clock, boot_id=f"boot-{int(time.time())}", inline=False)
    if args.manifest is not None:
        submit_manifest_once(collector, args.manifest.read_bytes(), f"operator manifest {args.manifest.name}")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    left = FomcService(collector, clock, tick_s=args.tick).run_until(stop)
    print(f"FOMC service stopped; attempts left without outcome: {left}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
