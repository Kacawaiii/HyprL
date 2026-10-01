"""Prolonged synthetic run of the autonomous service, offline: a simulated clock, a local provider on a
Unix socket, and a scripted day of FOMC traffic with faults - a correction, a historical backfill with
a failing and a clock-unverified entry, a save that hangs past 120 s, a processing run that hangs past
600 s, a COMMIT stalled for 200 s (a storage incident), one malformed feed, and two owner crashes
(during a save, during a processing run) each followed by a restart; it ends with a clean stop.
Snapshots are taken every hour during the run; afterwards the store is reopened and verified: every
snapshot re-reads identically at its (T, H), verified replay reproduces a subset, source health
re-derives, and the capture invariants hold.

    python -m scripts.trading_lab.fomc.soak [--hours 26] [--tick 10]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import tempfile
import threading
import time

from scripts.trading_lab.fomc import ledger, processing, snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.service import FomcService
from scripts.trading_lab.fomc.store import FomcStore, StoreBusy

A = syn.statement_path("20260617")
C = syn.statement_path("20260729")
BACKFILL = [syn.statement_path("20250129"), syn.statement_path("20250319"), syn.statement_path("20250507")]
STALE_DATE = "Mon, 01 Jun 2026 00:00:00 GMT"


def _item(path, guid):
    return {"title": spec.FEED_TITLE_EXACT, "link": syn.url(path), "guid": guid}


@dataclass
class _Run:
    root: Path
    clock: syn.SimClock
    provider: syn.LocalProvider
    store: FomcStore | None = None
    collector: Collector | None = None
    service: FomcService | None = None
    services: list = field(default_factory=list)
    boots: int = 0
    armed: dict = field(default_factory=dict)  # fault name -> state
    releases: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    log: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    stall: dict = field(default_factory=dict)  # the storage incident: monotonic start and end of the stall

    def start(self, tick_s: float) -> None:
        self.boots += 1
        self.store = FomcStore(self.root, wall_clock=self.clock.wall)
        self.collector = Collector(self.store, self.provider.connector(), self.clock, boot_id=f"boot-{self.boots}",
                                   inline=False)
        self.service = FomcService(self.collector, self.clock, tick_s=tick_s, settle_s=10.0, owner_wait_s=0.02,
                                   on_alert=self.alerts.append)
        self.services.append(self.service)
        self._install_faults()

    def crash(self, tick_s: float, why: str) -> None:
        """The owner process dies: its lock is released, its tasks never resume, a new owner starts."""
        self.log.append(f"{self.clock.true.isoformat()} crash: {why}")
        self.collector.close()
        self.start(tick_s)

    # ---- faults, installed on each owner --------------------------------------------------------
    def _install_faults(self) -> None:
        run, service, store = self, self.service, self.store
        put_raw = store.put_raw

        def put(data: bytes):
            fault = run.armed.get("save")
            if fault and fault["match"](data) and not fault.get("done"):
                fault["done"] = True
                release = threading.Event()
                fault["release"] = release
                run.releases.append(release)
                service.park(release)
            return put_raw(data)
        store.put_raw = put

        def processing_fault(resp):
            fault = run.armed.get("run")
            if fault and fault["match"](resp) and not fault.get("done"):
                fault["done"] = True
                release = threading.Event()
                fault["release"] = release
                run.releases.append(release)
                service.park(release)
        self.collector.processing_fault = processing_fault

        def store_fault(operation: str):
            fault = run.armed.get("commit")
            if fault and operation == fault["operation"] and not fault.get("done"):
                fault["done"] = True
                release = threading.Event()
                fault["release"] = release
                run.releases.append(release)
                run.stall["start"] = run.clock.mono()
                service.park(release)  # the COMMIT does not return: the store is stalled
        store.fault = store_fault


def _malformed_once(provider, items):
    served = {"n": 0}
    good = syn.feed_response(items)

    def route(_count):
        served["n"] += 1
        if served["n"] == 1:
            return syn.SyntheticResponse(body=b"<rss><channel><item></rss>", headers=list(syn.FEED_HEADERS))
        return good
    provider.routes[syn.FEED_PATH] = route


def run(root: Path, *, hours: float = 26.0, tick_s: float = 10.0, progress=None) -> dict:
    clock = syn.SimClock(syn.START)
    provider = syn.LocalProvider(clock)
    r = _Run(root, clock, provider)
    started_real = time.monotonic()
    try:
        provider.routes[syn.FEED_PATH] = syn.feed_response([])
        r.start(tick_s)
        t0 = clock.mono()

        def at(seconds):
            return t0 + seconds

        schedule = [
            (at(600), "feed lists statement A", lambda: (
                provider.routes.__setitem__(A, syn.page_response(body="The Committee decided to maintain the target range.")),
                provider.routes.__setitem__(syn.FEED_PATH, syn.feed_response([_item(A, "gA")])))),
            (at(2400), "statement A corrected upstream", lambda: provider.routes.__setitem__(
                A, syn.page_response(body="The Committee decided to lower the target range."))),
            (at(4000), "backfill manifest of three historical statements", lambda: (
                provider.routes.__setitem__(BACKFILL[0], syn.page_response(date_text="January 29, 2025")),
                provider.routes.__setitem__(BACKFILL[2], syn.page_response(date_text="May 7, 2025", date=STALE_DATE)),
                r.collector.submit_manifest(("{\"version\": 1, \"urls\": [" + ", ".join(f"\"{syn.url(p)}\"" for p in BACKFILL)
                                             + "]}").encode(), "soak operator"))),
            (at(5400), "the failing backfill entry comes back", lambda: provider.routes.__setitem__(
                BACKFILL[1], syn.page_response(date_text="March 19, 2025"))),
            (at(7200), "feed lists statement C; its first save will hang", lambda: (
                r.armed.__setitem__("save", {"match": lambda data: b"July 29, 2026" in data}),
                provider.routes.__setitem__(C, syn.page_response(date_text="July 29, 2026")),
                provider.routes.__setitem__(syn.FEED_PATH, syn.feed_response([_item(A, "gA"), _item(C, "gC")])))),
            (at(10000), "the next statement record's first processing run will hang", lambda: r.armed.__setitem__(
                "run", {"match": lambda resp: resp.body["surface"] == "primary"})),
            (at(15000), "the next record's COMMIT stalls for 200 s: a storage incident", lambda: r.armed.__setitem__(
                "commit", {"operation": "commit RESPONSE", "stall_s": 200})),
            (at(12600), "crash #1 armed: the next feed save hangs, then the owner dies", lambda: r.armed.__setitem__(
                "save", {"match": lambda data: data.startswith(b"<?xml"), "crash": True})),
            (at(18000), "one malformed feed", lambda: _malformed_once(provider, [_item(A, "gA"), _item(C, "gC")])),
            (at(21600), "crash #2 armed: the next feed processing run hangs, then the owner dies", lambda: r.armed.__setitem__(
                "run", {"match": lambda resp: resp.body["surface"] == "feed", "crash": True})),
        ]
        schedule.sort(key=lambda event: event[0])
        next_snapshot = at(3600)
        end = at(hours * 3600)
        while clock.mono() < end:
            r.service.tick()
            while schedule and clock.mono() >= schedule[0][0]:
                _t, what, action = schedule.pop(0)
                r.log.append(f"{clock.true.isoformat()} {what}")
                action()
            for name in ("save", "run"):
                fault = r.armed.get(name)
                if fault and fault.get("done") and fault.get("release") is not None and r.service._parked:
                    if fault.get("crash") and not fault.get("crashed"):
                        fault["crashed"] = True
                        r.crash(tick_s, f"owner dies with its {name} task hung")
                    elif not fault.get("crash") and "hung_since" not in fault:
                        fault["hung_since"] = clock.mono()
                if fault and not fault.get("crash") and "hung_since" in fault and not fault["release"].is_set() \
                        and clock.mono() - fault["hung_since"] >= (300 if name == "save" else 1200):
                    r.service.unpark(fault["release"])  # the hung task finally returns: its late result is fenced
                    r.log.append(f"{clock.true.isoformat()} hung {name} task released")
            commit = r.armed.get("commit")
            if commit and commit.get("release") is not None and not commit["release"].is_set() \
                    and clock.mono() - r.stall["start"] >= commit["stall_s"]:
                r.stall["end"] = clock.mono()
                r.service.unpark(commit["release"])  # the COMMIT returns
                r.log.append(f"{clock.true.isoformat()} stalled COMMIT returned")
            if clock.mono() >= next_snapshot:
                try:
                    snap = snapshot.events_as_of(r.store, clock.true)
                except StoreBusy:
                    snap = None  # the store is stalled: read at the next tick
                if snap is not None:
                    r.snapshots.append(snap)
                    next_snapshot += 3600
                    if progress:
                        progress(f"{clock.true.isoformat()} H={snap['H']} {snap['read_state']} "
                                 f"discovery={snap.get('discovery', {}).get('state')}")
            clock.sleep(tick_s)
        left = r.service.stop(wait_s=10)  # clean stop: nothing in flight, ownership released
        assert left == [], left
        r.log.append(f"{clock.true.isoformat()} clean stop")
        summary = verify(r)
    finally:
        for release in r.releases:
            if release is not None and not any(f.get("release") is release and f.get("crash") for f in r.armed.values()):
                release.set()
        provider.close()
    summary["real_seconds"] = round(time.monotonic() - started_real, 1)
    return summary


def verify(r: _Run) -> dict:
    """Reopen the store and check snapshots, replay, health and the capture invariants."""
    errors = [e for s in r.services for e in s.errors]
    assert errors == [], errors
    store = FomcStore(r.root, wall_clock=r.clock.wall)
    try:
        H = store.horizon()
        resolved = [s for s in r.snapshots if s["read_state"] == "FOMC_RESOLVED"]
        assert resolved and len(resolved) >= len(r.snapshots) - 1
        for snap in r.snapshots:  # re-read after reopening: the same state and identity at the same (T, H)
            again = snapshot.events_as_of(store, _t(snap), snap["H"])
            assert again == snap, f"snapshot at H={snap['H']} changed"
        replayed = resolved[::6] + [resolved[-1]]
        for snap in replayed:  # verified replay: raw digests, clock verdicts, re-derivation, health
            assert snapshot.replay(store, _t(snap), snap["H"]) == snap
        snapshot.verify_health(store, H)

        view = store.view()
        responses = view.rows("RESPONSE")
        assert all(len(view.rows("PROCESSING_OUTCOME", key=str(x.seq))) == 1 for x in responses)  # one terminal outcome
        invoked = view.rows("TRANSPORT_INVOKED")
        open_attempts = [t for t in invoked if not view.rows("ATTEMPT_OUTCOME", key=str(t.seq))]
        assert open_attempts == []  # the clean stop left nothing in flight
        for key in {t.key for t in invoked}:  # one feed poll and one fetch per item in flight, never two
            same = [t for t in invoked if t.key == key]
            assert all(view.rows("ATTEMPT_OUTCOME", key=str(a.seq))[0].seq < b.seq for a, b in zip(same, same[1:]))
        episodes = view.rows("EPISODE_OPEN")
        assert len({e.key for e in episodes}) == len(episodes)
        per_key = {}
        for t in invoked:
            if t.key != ledger.FEED_KEY:
                per_key[t.key] = per_key.get(t.key, 0) + 1
        assert max(per_key.values()) <= spec.ATTEMPTS_PER_EPISODE
        live_requests = {p: r.provider.requests.count(p) for p in (A, C)}
        assert all(n <= 120 for n in live_requests.values())
        grants = sorted(t.body["grant_mono"] for t in invoked)
        assert all(b - a >= spec.SPACING_S for a, b in zip(grants, grants[1:]))
        assert all(sum(1 for g in grants if 0 <= x - g < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for x in grants)
        zero = 0
        for cycle in view.rows("CYCLE_CONCLUSION"):
            if cycle.body["result"] != "EVENTS_OBSERVED_ZERO":
                continue
            zero += 1
            B = cycle.body["B"]
            feed = view.row_at("RESPONSE", B)
            assert state.live_eligible(feed)
            assert state.outstanding(view, upto=B - 1) == []  # no false zero: nothing outstanding before B
            assert all(state.processing_outcome(view, f.seq, upto=B - 1) is not None
                       for f in state.feed_responses(view, upto=B - 1))
        interrupted = [o for o in view.rows("ATTEMPT_OUTCOME") if o.body.get("reason") == "non-current epoch"]
        lpf = [o for o in view.rows("ATTEMPT_OUTCOME") if o.body["outcome"] == "LOCAL_PERSISTENCE_FAILED"]
        assert lpf and all(not view.rows("RESPONSE", key=str(o.body["attempt"])) for o in lpf)  # no late record
        dead = view.rows("RUN_DEAD")
        incidents = view.rows("STORAGE_INCIDENT")
        if r.stall:  # the storage incident: one durable record, both alerts, no grant while it lasted
            assert len(incidents) == 1 and incidents[0].body["stalled_s"] >= 200
            assert [a["alert"] for a in r.alerts] == ["STORAGE_INCIDENT_STARTED", "STORAGE_INCIDENT_ENDED"]
            threshold = spec.STORAGE_STALL_THRESHOLD_S + 2 * r.service.tick_s
            assert not [g for g in grants if r.stall["start"] + threshold < g < r.stall["end"]]
        epochs = view.rows("EPOCH")
        assert len(epochs) == r.boots
        sid_a, sid_c = (identity_of(p) for p in (A, C))
        assert state.anchor(view, sid_a) is not None and state.anchor(view, sid_c) is not None
        lb = state.now_lb(view)
        due = [o for o in state.obligations(view, sid_a) + state.obligations(view, sid_c)
               if o["due_at"] + timedelta(seconds=900) <= lb]
        assert due and all(o["satisfied"] for o in due)  # every recheck due long enough ago was served
        final = snapshot.events_as_of(store, _t(resolved[-1]), resolved[-1]["H"])
        return {
            "hours": round((r.clock.true - syn.START).total_seconds() / 3600, 2), "boots": r.boots,
            "transactions": H, "responses": len(responses), "attempts": len(invoked),
            "provider_requests": len(r.provider.requests), "live_item_requests": live_requests,
            "rechecks_due_and_served": len(due), "cycles": len(view.rows("CYCLE_CONCLUSION")), "zero_cycles": zero, "revisions": len(view.rows("REVISION")),
            "interrupted_by_restart": len(interrupted), "local_persistence_failed": len(lpf), "dead_runs": len(dead),
            "storage_incidents": [i.body["stalled_s"] for i in incidents], "alerts": [a["alert"] for a in r.alerts],
            "snapshots": len(r.snapshots), "replayed": len(replayed), "health_rows": len(view.rows("SOURCE_HEALTH")),
            "final_discovery": final["discovery"]["state"],
            "final_health": {k: v["result_state"] for k, v in final["health"].items()}, "log": r.log,
        }
    finally:
        store.close()


def identity_of(path: str) -> str:
    from scripts.trading_lab.fomc.identity import source_item_id
    return source_item_id(syn.url(path))


def _t(snap):
    from scripts.trading_lab.fomc.clock import parse_iso
    return parse_iso(snap["T"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=26.0)
    parser.add_argument("--tick", type=float, default=10.0)
    args = parser.parse_args(argv)
    spec.verify_spec_binding()
    root = Path(tempfile.mkdtemp(prefix="fomc-soak-")) / "store"
    summary = run(root, hours=args.hours, tick_s=args.tick, progress=print)
    log = summary.pop("log")
    print("\n".join(log))
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("soak verified: snapshots re-read, replay, health replay and invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
