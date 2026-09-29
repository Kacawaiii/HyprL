"""The autonomous FOMC owner service: a tick loop that never runs a task itself.

Each tick, on the owner thread:
1. `collector.reconcile()` closes what reached its bound: LOCAL_PERSISTENCE_FAILED 120 s after the
   network end (even while the saving task is blocked), INTERRUPTED 600 s after TRANSPORT_INVOKED or at
   once for an earlier epoch, DEAD processing runs 600 s after their start, then replacement runs and
   the poison guard (process_pending) and derived episode terminals;
2. when no logical fetch is in flight (none, or its attempt already has a durable outcome), the next
   selection decision (`collector.select`) is handed to a worker thread.
Processing runs are started by the owner and finished in worker threads. The 60 s physical deadline
is enforced inside the fetch by the transport. A task that hangs keeps only its own thread: the owner
closes it at its bound and its late result is fenced (first outcome wins, LATE_EVIDENCE, run fences).
A restart is a new owner (new epoch): attempts of earlier epochs without an outcome are INTERRUPTED
at the first tick and every durable budget stays counted; nothing is recreated.

Limits (docs/FOMC_V1_OFFLINE_SLICE.md): one logical fetch in flight at a time; a bound is closed at
the first tick at or after it; a SQLite COMMIT stalled in fsync holds the store lock and blocks the
owner too (CAPTURE_BLOCKERS).

    python -m scripts.trading_lab.fomc.service --store PATH   # real provider: refused while blocked
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time

from scripts.trading_lab.fomc import ledger, spec

# Real capture stays refused while any entry remains (see the registry, "Capture verdict").
CAPTURE_BLOCKERS = (
    "COMMIT_FSYNC_120S: the 120 s save bound is checked before SQLite's COMMIT; a COMMIT stalled in fsync "
    "can make a RESPONSE durable after +120 s and blocks the owner meanwhile "
    "(tests/crypto/test_fomc_service.py::test_capture_blocker_*)",
)


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


class FomcService:
    def __init__(self, collector, clock, *, tick_s: float = 1.0, settle_s: float = 0.0):
        """`settle_s` > 0 is for simulated clocks only: after each tick the owner waits (real time, at
        most settle_s) for its workers to finish or park, so that simulated time advances between
        ticks deterministically. In production it is 0 and the owner never waits on a worker."""
        self.collector, self.clock, self.tick_s, self.settle_s = collector, clock, tick_s, settle_s
        collector.inline = False
        collector.dispatch_run = self._dispatch_run
        self.workers: list[threading.Thread] = []
        self.errors: list[BaseException] = []
        self.ticks = 0
        self._fetch_thread: threading.Thread | None = None
        self._parked: set[int] = set()

    # ---- workers ------------------------------------------------------------------------------
    def _spawn(self, name: str, target, *args) -> threading.Thread:
        def body():
            try:
                target(*args)
            except BaseException as exc:  # a task failure never reaches the owner thread
                self.errors.append(exc)
        thread = threading.Thread(target=body, name=name, daemon=True)
        self.workers.append(thread)
        thread.start()
        return thread

    def _dispatch_run(self, seq: int, run_id: str) -> None:
        self._spawn(f"fomc-run-{seq}", self.collector.finish_processing, seq, run_id)

    def _fetch(self, chosen: dict) -> None:
        if chosen.get("poll"):
            self.collector.poll_feed()
        else:
            self.collector.run_episode(chosen["episode"])

    def fetch_in_flight(self) -> bool:
        """A logical fetch is in flight while its worker lives and it has not yet been closed durably:
        before its TRANSPORT_INVOKED (waiting for a FIX15 grant), or while its attempt has no outcome."""
        thread = self._fetch_thread
        if thread is None or not thread.is_alive():
            return False
        active = list(self.collector._active_attempts)
        return not active or any(ledger.outcome_of(self.collector.store, a) is None for a in active)

    def park(self, release: threading.Event, timeout: float | None = None) -> None:
        """Test and soak hook: block the calling task until `release` is set (a hung task)."""
        ident = threading.get_ident()
        self._parked.add(ident)
        try:
            release.wait(timeout)
        finally:
            self._parked.discard(ident)

    def _settle(self) -> None:
        if self.settle_s > 0:
            deadline = time.monotonic() + self.settle_s
            for thread in list(self.workers):
                while thread.is_alive() and thread.ident not in self._parked and time.monotonic() < deadline:
                    thread.join(0.002)
        self.workers = [t for t in self.workers if t.is_alive()]

    # ---- the owner loop -----------------------------------------------------------------------
    def tick(self) -> None:
        self.ticks += 1
        self.collector.reconcile()
        if not self.fetch_in_flight():
            chosen = self.collector.select()
            if chosen is not None:
                self._fetch_thread = self._spawn("fomc-fetch", self._fetch, chosen)
        self._settle()

    def run_for(self, seconds: float, between=None) -> None:
        """Tick for `seconds` of the service clock; `between(service)` runs after every tick."""
        end = self.clock.mono() + seconds
        while self.clock.mono() < end:
            self.tick()
            if between is not None:
                between(self)
            self.clock.sleep(self.tick_s)

    def run_until(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.tick()
            stop.wait(self.tick_s)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="store directory (created if absent)")
    parser.add_argument("--tick", type=float, default=1.0, help="owner tick in seconds (default 1)")
    args = parser.parse_args(argv)
    spec.verify_spec_binding()
    if CAPTURE_BLOCKERS:
        print("FOMC capture refused: open capture blockers:", file=sys.stderr)
        for blocker in CAPTURE_BLOCKERS:
            print(f"  - {blocker}", file=sys.stderr)
        return 3
    from scripts.trading_lab.fomc.collector import Collector
    from scripts.trading_lab.fomc.store import FomcStore
    from scripts.trading_lab.fomc.transport import HttpsConnector
    clock = SystemClock()
    store = FomcStore(args.store, wall_clock=clock.wall)  # an incompatible store is rejected here
    collector = Collector(store, HttpsConnector(), clock, boot_id=f"boot-{int(time.time())}")
    stop = threading.Event()
    try:
        FomcService(collector, clock, tick_s=args.tick).run_until(stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        collector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
