"""The single-owner FOMC collector: epoch ownership, the logical fetch (TRANSPORT_INVOKED -> transport
-> response or outcome), reconciliation, bounded episodes, class-rotation selection, operator actions
and backfill manifests (retry_policy, reconciliation, selection, operator_resolution, surfaces.backfill_manifest).
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
import uuid
from datetime import timedelta

from scripts.trading_lab.fomc import clock as clockmod
from scripts.trading_lab.fomc import health, identity, ledger, processing, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.limiter import Limiter
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt, Rejected
from scripts.trading_lab.fomc.transport import FetchResult, Transport

CLASS_ORDER = ("FEED_DISCOVERY", "REOBSERVATION", "HISTORICAL_BACKFILL")
SAVE_RETRY_S = 5.0  # implementation choice: local save retry spacing inside the 120 s save deadline
FEED_WORK = {"kind": "FEED_POLL", "class": "FEED_DISCOVERY", "sid": None, "mode": "LIVE"}
CLASS_OF = {"FEED_POLL": "FEED_DISCOVERY", "LIVE_ACQUISITION": "FEED_DISCOVERY", "REOBSERVATION": "REOBSERVATION",
            "HISTORICAL_BACKFILL": "HISTORICAL_BACKFILL"}


class OwnershipUnavailable(RuntimeError):
    """Exclusive provider ownership cannot be established: no provider attempt may start."""


class Collector:
    def __init__(self, store: FomcStore, connector, clock, *, boot_id: str = "boot-1", inline: bool = True):
        self.store, self.clock = store, clock
        self._owner = open(store.root / "owner.lock", "a+")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._owner.close()
            raise OwnershipUnavailable("another collector owns this store") from exc
        self.epoch = uuid.uuid4().hex
        ledger.begin_epoch(store, self.epoch, boot_id)
        self.limiter = Limiter(clock.mono, clock.sleep)
        self.transport = Transport(connector, self.limiter, wall=clock.wall, mono=clock.mono)
        self.last_feed_grant: float | None = None
        self.processing_fault = None  # test hook
        self.persist_fault = None  # test hook: raise OSError to simulate a failing local save
        self._active_attempts: dict[int, float] = {}  # own attempts whose task is alive -> TRANSPORT_INVOKED mono
        self._active_runs: dict[int, str] = {}  # record seq -> run_id of a run whose task is alive
        self._save_deadlines: dict[int, float] = {}  # attempt -> network end + 120 s, kept from the network end
        # step-driven use processes inline; the autonomous service (service.py) sets inline = False,
        # dispatch_run and continuation, so that only its owner thread reconciles and dispatches grants
        # while network I/O, saves and processing runs happen in worker threads
        self.inline = inline
        self.dispatch_run = None
        self.continuation = None  # continuation(attempt) -> (grant, deadline) | None, set by the service
        # the task registries are shared with worker threads: every removal is compare-and-delete under
        # this lock, so a worker only ever removes its own task, never a newer one's
        self._registry = threading.Lock()
        if inline:
            self.reconcile()
        else:
            self.close_attempts()  # earlier-epoch attempts are interrupted at once; processing is dispatched

    def close(self) -> None:
        """Release ownership (also what a crash does to an flock)."""
        self._owner.close()

    # ------------------------------------------------------------------ one logical fetch ---------
    def _invoke(self, work: dict, grant: float) -> int:
        """Commit TRANSPORT_INVOKED and register its task in one locked step, so reconcile() never
        sees the durable attempt without its live task."""
        with self.store.locked("transaction TRANSPORT_INVOKED"):
            seq = ledger.transport_invoked(self.store, epoch=self.epoch, work=work, grant_mono=grant)
            self._active_attempts[seq] = self.clock.mono()
        if work["kind"] == "FEED_POLL":
            self.last_feed_grant = grant
        return seq

    def _open(self, attempt: int) -> bool:
        return ledger.outcome_of(self.store, attempt) is None

    def fetch(self, work: dict, url: str, surface: str, *, started: tuple | None = None) -> FetchResult | dict:
        """Grant, TRANSPORT_INVOKED, transport. The attempt stays active (its task alive) until commit().
        `started` = (attempt, grant, deadline) when the dispatcher already granted and invoked it."""
        try:
            if started is None:
                result = self.transport.fetch(url, surface, invoke=lambda grant: self._invoke(work, grant),
                                              may_continue=self._open)
            else:
                attempt, grant, deadline = started
                result = self.transport.fetch(url, surface, invoke=lambda _grant: attempt, may_continue=self._open,
                                              started=(grant, deadline), continuation=self.continuation)
        except Rejected as exc:
            return {"status": "REJECTED", "reason": str(exc)}
        self.network_ended(result)
        return result

    # ---- the dispatcher's two halves (service.py) ------------------------------------------------
    def begin(self, chosen: dict) -> dict | None:
        """Attribute the start FIX15 admits at this instant to `chosen` (selection.grant_dispatch):
        commit its TRANSPORT_INVOKED stamped with this instant, then consume the start. None when no
        start is admissible now or the record does not commit - then no grant is consumed and nothing
        is sent. Called by the single dispatcher thread only."""
        if chosen.get("poll"):
            work, url, surface = dict(FEED_WORK), spec.FEED_URL, "feed"
        else:
            work, url = self.work_of(chosen["episode"])
            surface = "primary"
        identity.admit_url(url)  # validation precedes the grant
        now = self.clock.mono()
        if self.limiter.suspended or not self.limiter.admissible(now):
            return None
        deadline = self.transport.deadline()  # the physical deadline starts at the grant
        try:
            attempt = self._invoke(work, now)
        except Rejected:
            return None
        self.limiter.register(now)
        return {"work": work, "url": url, "surface": surface, "attempt": attempt, "grant": now, "deadline": deadline}

    def complete(self, started: dict) -> dict:
        """A worker's part of a fetch begun by the dispatcher: network phase, then the local save."""
        result = self.fetch(started["work"], started["url"], started["surface"],
                            started=(started["attempt"], started["grant"], started["deadline"]))
        if isinstance(result, dict):
            return result
        return self.commit(result, started["work"], started["url"], started["surface"])

    def network_ended(self, result: FetchResult) -> None:
        """The network phase of `result` ends now: its 120 s local save deadline starts here and is
        kept by the owner, so reconcile() can close the attempt even if the saving task is blocked."""
        result.network_end_mono = self.clock.mono()
        if result.kind == "RESPONSE_200":
            self._save_deadlines[result.attempt_seq] = result.network_end_mono + spec.SAVE_DEADLINE_S

    def commit(self, result: FetchResult, work: dict, url: str, surface: str) -> dict:
        attempt = result.attempt_seq
        try:
            return self._commit(result, work, url, surface)
        finally:
            self._active_attempts.pop(attempt, None)  # the task ends here, with or without an outcome
            self._save_deadlines.pop(attempt, None)

    def _commit(self, result: FetchResult, work: dict, url: str, surface: str) -> dict:
        attempt = result.attempt_seq
        started = self._active_attempts.get(attempt)
        if started is not None and self.clock.mono() - started >= spec.ATTEMPT_ABSOLUTE_DEADLINE_S:
            # the 600 s bound closes the attempt first: whatever arrives now is LATE_EVIDENCE
            ledger.commit_attempt_outcome(self.store, attempt, "INTERRUPTED", {"reason": "no outcome 600 s after TRANSPORT_INVOKED"})
        if result.kind != "RESPONSE_200":
            outcome = "CANCELLED_AFTER_INVOKE" if result.kind == "ABANDONED" else result.kind
            ledger.commit_attempt_outcome(self.store, attempt, outcome, {"reason": result.reason})
            if self.inline:
                self.derive_terminals()
            return {"status": outcome, "reason": result.reason}
        save_deadline = self._save_deadlines.get(attempt, result.network_end_mono + spec.SAVE_DEADLINE_S)

        def admit(_store) -> None:  # inside the committing transaction: at 120 s, expired
            if self.clock.mono() >= save_deadline:
                raise ledger.SaveExpired()

        while True:
            if self.clock.mono() >= save_deadline:
                return self._persistence_failed(attempt, "local save not durable within 120 s of the network end")
            try:
                if self.persist_fault is not None:
                    self.persist_fault()
                digest = self.store.put_raw(result.body)
                seq, late = ledger.commit_response(self.store, attempt, self._fields(result, work, url, surface, digest), admit)
                break
            except ledger.SaveExpired:
                # the local operation finished at or after +120 s: never a RESPONSE nor LATE_EVIDENCE
                return self._persistence_failed(attempt, "local save not durable within 120 s of the network end")
            except RawCorrupt:
                # The immutable slot for these bytes holds other bytes: never overwrite it. The exact
                # bytes of this attempt are therefore not durable -> LOCAL_PERSISTENCE_FAILED, no RESPONSE.
                # Older records sharing the digest get integrity diagnostics; their outcomes stay.
                self._diagnose(spec.sha256_bytes(result.body))
                return self._persistence_failed(attempt, "raw slot holds other bytes; never overwritten")
            except (OSError, sqlite3.OperationalError):
                self.clock.sleep(min(SAVE_RETRY_S, max(save_deadline - self.clock.mono(), 0.0)))  # retry locally, no network
        if self.inline:
            self.process_pending()
            self.derive_terminals()
        return {"status": "RESPONSE", "record": seq, "late": late, "verified": self._verified(result)}

    @staticmethod
    def _verified(result: FetchResult) -> bool:
        return clockmod.is_clock_verified(result.wall_at_receipt, result.date_lines, result.age_lines)

    def _fields(self, result: FetchResult, work: dict, url: str, surface: str, digest: str) -> dict:
        verified = self._verified(result)
        return {
            "surface": surface, "mode": work["mode"], "sid": work.get("sid"), "work": work["kind"],
            "episode_key": work.get("episode_key"), "request_url": url, "final_url": result.final_url,
            "redirect_chain": [h.url for h in result.hops], "status": 200,
            "content_type_lines": result.content_type_lines, "content_encoding": result.content_encoding,
            "raw_sha": digest, "byte_length": len(result.body), "date_lines": result.date_lines,
            "age_lines": result.age_lines, "wall_at_receipt": iso(result.wall_at_receipt),
            "verdict": "CLOCK_VERIFIED" if verified else "CLOCK_UNVERIFIED",
            "observed_at": iso(result.wall_at_receipt) if verified else None,
        }

    def _persistence_failed(self, attempt: int, reason: str) -> dict:
        ledger.commit_attempt_outcome(self.store, attempt, "LOCAL_PERSISTENCE_FAILED", {"reason": reason})
        if self.inline:
            self.derive_terminals()
        return {"status": "LOCAL_PERSISTENCE_FAILED", "reason": reason}

    def run(self, work: dict, url: str, surface: str) -> dict:
        result = self.fetch(work, url, surface)
        if isinstance(result, dict):
            return result
        return self.commit(result, work, url, surface)

    def poll_feed(self) -> dict:
        return self.run(dict(FEED_WORK), spec.FEED_URL, "feed")

    # ------------------------------------------------------------------ reconciliation ------------
    def reconcile(self) -> None:
        """Close attempts at their bounds, then process pending records and derive episode terminals."""
        self.close_attempts()
        self.process_pending()
        self.derive_terminals()

    def close_attempts(self) -> None:
        """INTERRUPTED for attempts of other epochs, of ended tasks, or 600 s after TRANSPORT_INVOKED;
        LOCAL_PERSISTENCE_FAILED at or after the 120 s admission bound; a task that is still active is
        never declared dead before its deadline. Every closure is first-outcome-wins in the store."""
        now = self.clock.mono()
        for attempt in ledger.attempts_without_outcome(self.store):
            save_deadline = self._save_deadlines.get(attempt.seq) if attempt.body["epoch"] == self.epoch else None
            if save_deadline is not None and now >= save_deadline:
                # bytes in hand but not durable by +120 s, even if the saving task is still blocked
                ledger.commit_attempt_outcome(self.store, attempt.seq, "LOCAL_PERSISTENCE_FAILED",
                                              {"reason": "local save not durable within 120 s of the network end"})
                continue
            started = self._active_attempts.get(attempt.seq) if attempt.body["epoch"] == self.epoch else None
            if started is not None and now - started < spec.ATTEMPT_ABSOLUTE_DEADLINE_S:
                continue
            reason = ("non-current epoch" if attempt.body["epoch"] != self.epoch
                      else "task ended without outcome" if started is None else "no outcome 600 s after TRANSPORT_INVOKED")
            ledger.commit_attempt_outcome(self.store, attempt.seq, "INTERRUPTED", {"reason": reason})

    def _dead_runs(self, seq: int) -> int:
        dead_ids = {d.body["run_id"] for d in self.store.rows("RUN_DEAD", key=str(seq))}
        return sum(1 for run in self.store.rows("PROCESSING_RUN", key=str(seq))
                   if run.body["run_id"] in dead_ids or run.body["epoch"] != self.epoch)

    def _running(self, seq: int):
        dead_ids = {d.body["run_id"] for d in self.store.rows("RUN_DEAD", key=str(seq))}
        runs = [r for r in self.store.rows("PROCESSING_RUN", key=str(seq))
                if r.body["epoch"] == self.epoch and r.body["run_id"] not in dead_ids]
        return runs[-1] if runs else None

    def _mark_dead(self, seq: int, run_id: str, reason: str) -> None:
        """Mark exactly this run DEAD, once, and forget its task - never another run's."""
        def once(s: FomcStore) -> None:
            if any(d.body["run_id"] == run_id for d in s.rows("RUN_DEAD", key=str(seq))):
                raise Rejected("run already DEAD")
        try:
            self.store.append("RUN_DEAD", [("RUN_DEAD", str(seq), {"run_id": run_id, "reason": reason})], once)
        except Rejected:
            pass
        self._forget_run(seq, run_id)

    def _forget_run(self, seq: int, run_id: str) -> None:
        with self._registry:  # compare-and-delete: an old run's worker never erases a newer run's task
            if self._active_runs.get(seq) == run_id:
                del self._active_runs[seq]

    def start_processing(self, seq: int) -> str | None:
        run_id = processing.start_run(self.store, seq, epoch=self.epoch, start_mono=self.clock.mono())
        if run_id is not None:
            with self._registry:
                self._active_runs[seq] = run_id
        return run_id

    def finish_processing(self, seq: int, run_id: str) -> str | None:
        try:
            outcome = processing.finish_run(self.store, seq, epoch=self.epoch, run_id=run_id,
                                            mono=self.clock.mono, fault=self.processing_fault)
        except Exception:  # the task ended without an outcome: that run is DEAD
            self._mark_dead(seq, run_id, "task ended without outcome")
            return None
        if outcome is None and state.processing_outcome(self.store, seq) is None and self._running(seq) is not None:
            self._mark_dead(seq, run_id, "late result after the 600 s run deadline")
        self._forget_run(seq, run_id)
        return outcome

    def process_pending(self) -> None:
        view = self.store.view()
        head = state.open_work(view)  # a fresh view is at the head
        pending = list(head.pending) if head is not None else \
            [r.seq for r in view.rows("RESPONSE") if state.processing_outcome(view, r.seq) is None]
        for seq in pending:  # commit order; only this loop commits outcomes for these records
            if state.processing_outcome(self.store, seq) is not None:
                continue
            running = self._running(seq)
            if running is not None:
                active = self._active_runs.get(seq) == running.body["run_id"]
                if active and self.clock.mono() < running.body["deadline_mono"]:
                    continue  # RUNNING and before its deadline: never a second run, never counted
                self._mark_dead(seq, running.body["run_id"], "run deadline passed" if active else "task ended without outcome")
            if self._dead_runs(seq) >= spec.POISON_DEAD_RUNS:
                processing.poison(self.store, seq, epoch=self.epoch)
                continue
            run_id = self.start_processing(seq)
            if run_id is not None:
                if self.dispatch_run is not None:
                    self.dispatch_run(seq, run_id)  # the service's worker thread; the owner never waits on it
                else:
                    self.finish_processing(seq, run_id)

    def derive_terminals(self) -> None:
        view = self.store.view()  # SUCCEEDED and SUSPENDED are monotone: a view can only miss one for now
        for episode in view.rows("EPISODE_OPEN"):
            status = state.derived_status(view, episode)
            if status in ("SUCCEEDED", "SUSPENDED") and ledger.episode_status(view, episode.key) == "OPEN":
                ledger.close_episode(self.store, episode.key, status, "derived")

    # ------------------------------------------------------------------ episodes ------------------
    def _url_of(self, sid: str) -> str:
        return self.store.rows("CANDIDATE", key=sid)[0].body["url"]

    def open_episodes(self) -> None:
        store = self.store.view()  # decisions read the view; each open is atomic and unique by key
        for acq in store.rows("ACQUISITION"):
            sid = acq.key
            key = identity.acquisition_episode_key(sid, acq.seq)
            if state.anchor(store, sid) is None and not store.rows("EPISODE_OPEN", key=key):
                ledger.open_episode(self.store, key, {"kind": "LIVE_ACQUISITION", "sid": sid, "url": acq.body["url"], "mode": "LIVE"})
        for cand in store.rows("CANDIDATE"):
            sid = cand.key
            if any(e.body["kind"] == "REOBSERVATION" and state.derived_status(store, e) == "OPEN"
                   for e in state.item_episodes(store, sid)):
                continue
            for ob in state.obligations(store, sid):
                key = identity.reobservation_episode_key(sid, ob["anchor"], ob["offset"])
                if ob["pending_due"] and not store.rows("EPISODE_OPEN", key=key):
                    def check(s, sid=sid, ob=ob):
                        a = state.anchor(s, sid)
                        if a is None or a.body["observed_at"] != ob["anchor"] or ob["offset"] not in spec.OFFSETS_S:
                            raise Rejected("REOBSERVATION key does not match the item's anchor and frozen offsets")
                    ledger.open_episode(self.store, key, {"kind": "REOBSERVATION", "sid": sid, "url": cand.body["url"],
                                                     "anchor": ob["anchor"], "offset": ob["offset"], "mode": "LIVE"}, check)
                    break
        for manifest in store.rows("MANIFEST"):
            if not manifest.body["valid"]:
                continue
            for entry in manifest.body["entries"]:
                key = identity.backfill_episode_key(manifest.seq, entry["sid"])
                if not store.rows("EPISODE_OPEN", key=key):
                    ledger.open_episode(self.store, key, {"kind": "HISTORICAL_BACKFILL", "sid": entry["sid"], "url": entry["url"],
                                                     "manifest": manifest.seq, "mode": "HISTORICAL_BACKFILL"})
        for op in store.rows("OPERATOR"):
            if op.body["action"] != "MANUAL_RETRY":
                continue
            key = identity.manual_episode_key(op.seq)
            work = dict(op.body["work"])
            if store.rows("EPISODE_OPEN", key=key):
                continue
            if work["kind"] == "REOBSERVATION":
                due = [o for o in state.obligations(store, work["sid"]) if o["offset"] == work["offset"]]
                if not due or not due[0]["pending_due"]:
                    continue
                work["anchor"] = due[0]["anchor"]
            ledger.open_episode(self.store, key, {"kind": "MANUAL_RETRY", "work": work, "sid": work["sid"]})

    @staticmethod
    def _opening_instant(target: dict):
        """The server-time instant a first attempt's opening condition starts to hold, when it has one:
        a REOBSERVATION is PENDING_DUE from NOW_LB >= due_at + 92 s. Other work is eligible from its
        durable opening record, which orders before any server-time instant."""
        if target["kind"] != "REOBSERVATION":
            return None
        return clockmod.parse_iso(target["anchor"]) + timedelta(seconds=target["offset"] + spec.CLOCK_ERROR_BOUND_S)

    def _ready(self, episode, table, lb, store) -> tuple[bool, float]:
        """(eligible now, next eligible instant) for the episode's next attempt (retry_policy.schedule,
        first_attempt_eligibility); the instant orders eligible work within its class."""
        body = episode.body
        sid = body["sid"]
        opening = None
        outcomes = state.attempt_outcomes(store, episode.key)
        if any(o is None for _a, o in outcomes) or len(outcomes) >= spec.ATTEMPTS_PER_EPISODE:
            return False, 0.0
        if ledger.attempts_without_outcome(store, sid=sid):
            return False, 0.0  # one logical fetch per item in flight
        if any(state.processing_outcome(store, r.seq) is None for r in state.primary_responses(store, sid)):
            return False, 0.0  # processing gate
        if outcomes:
            last = outcomes[-1][1]
            gap = spec.BACKOFF_S[len(outcomes) - 1]
        else:
            opening = self._opening_instant(body["work"] if body["kind"] == "MANUAL_RETRY" else body)
            prior = [o for e in state.item_episodes(store, sid, live_only=False) if e.key != episode.key
                     for _a, o in state.attempt_outcomes(store, e.key) if o is not None and o.body["outcome"] != "RESPONSE"]
            if not prior:
                return True, opening.timestamp() if opening else float("-inf")
            last, gap = prior[-1], spec.FIRST_ATTEMPT_GAP_S
        start = state.avail_of(table, last.seq)
        if start is None or lb is None:
            return False, 0.0  # held until verified server time exists: never burned by waiting
        threshold = start + timedelta(seconds=gap)
        return lb >= threshold, max(threshold, opening).timestamp() if opening else threshold.timestamp()

    def feed_poll_due(self, view=None) -> bool:
        view = view or self.store.view()
        if ledger.attempts_without_outcome(view, feed=True):
            return False
        if processing.earlier_feed_unterminated(view, view.horizon() + 1):
            return False
        return self.last_feed_grant is None or self.clock.mono() >= self.last_feed_grant + spec.FEED_CADENCE_S

    def eligible_work(self) -> list[dict]:
        self.open_episodes()
        view = self.store.view()
        table = state.availability(view, view.horizon())
        lb = state.now_lb(view)
        work = []
        if self.feed_poll_due(view):
            work.append({"class": "FEED_DISCOVERY", "poll": True, "order": (0.0, "")})
        for episode in view.rows("EPISODE_OPEN"):
            if state.derived_status(view, episode) != "OPEN":
                continue
            ready, instant = self._ready(episode, table, lb, view)
            if not ready:
                continue
            target = episode.body["work"] if episode.body["kind"] == "MANUAL_RETRY" else episode.body
            work.append({"class": CLASS_OF[target["kind"]], "episode": episode, "order": _order(target, instant)})
        return work

    def step(self) -> dict | None:
        """One selection decision (ELIGIBLE_CLASS_ALTERNATION_V1) and its logical fetch. A redirect
        continuation never reaches a decision: it is followed inside its logical fetch, on its own
        grant, before any new attempt, and creates no TRANSPORT_INVOKED (rotation is unaffected)."""
        self.reconcile()
        chosen = self.select()
        if chosen is None:
            return None
        if chosen.get("poll"):
            return self.poll_feed()
        return self.run_episode(chosen["episode"])

    def select(self) -> dict | None:
        """The work the next selection decision takes, derived from durable records only (plus the
        feed cadence timer); nothing is fetched."""
        work = self.eligible_work()
        if not work:
            return None
        invoked = self.store.view().rows("TRANSPORT_INVOKED")
        last_class = invoked[-1].body["class"] if invoked else None
        start = (CLASS_ORDER.index(last_class) + 1) if last_class in CLASS_ORDER else 0
        for i in range(len(CLASS_ORDER)):
            klass = CLASS_ORDER[(start + i) % len(CLASS_ORDER)]
            choices = [w for w in work if w["class"] == klass]
            if not choices:
                continue
            if klass == "FEED_DISCOVERY" and len({bool(w.get("poll")) for w in choices}) == 2:
                feed_ti = [t for t in invoked if t.body["class"] == "FEED_DISCOVERY"]
                want_poll = not feed_ti or feed_ti[-1].body["kind"] != "FEED_POLL"
                choices = [w for w in choices if bool(w.get("poll")) == want_poll]
            return min(choices, key=lambda w: w["order"])
        return None

    def work_of(self, episode) -> tuple[dict, str]:
        """The logical fetch of an episode: its work and the URL of the retried work."""
        body = episode.body
        target = body["work"] if body["kind"] == "MANUAL_RETRY" else body
        work = {"kind": body["kind"], "class": CLASS_OF[target["kind"]], "episode_key": episode.key,
                "sid": body["sid"], "mode": target["mode"]}
        return work, target["url"]

    def run_episode(self, episode) -> dict:
        work, url = self.work_of(episode)
        return self.run(work, url, "primary")

    def run_until_idle(self, max_steps: int = 50) -> list[dict]:
        out = []
        for _ in range(max_steps):
            done = self.step()
            if done is None:
                break
            out.append(done)
        return out

    def _diagnose(self, digest: str) -> None:
        """For every record with these bytes and a terminal outcome: keep the outcome, and commit the
        integrity diagnostic and the surface's NO_PROVIDER_HEALTH_STATE (RAW_CORRUPTION) in one
        transaction, once per record. A record without an outcome is left to processing, which fails
        closed (CORRUPTION_FAIL_CLOSED) with its own health result. Nothing is fetched."""
        view = self.store.view()
        for resp in view.select("RESPONSE", "raw_sha", digest):
            outcome = state.processing_outcome(view, resp.seq)
            if outcome is None or view.rows("INTEGRITY_DIAGNOSTIC", key=str(resp.seq)):
                continue
            wall = self.store.wall_iso()
            rows = [("INTEGRITY_DIAGNOSTIC", str(resp.seq), {"record": resp.seq, "raw_sha": digest,
                                                             "outcome": outcome.body["outcome"]}),
                    health.row(health.surface_of_record(resp.body), wall, health.for_integrity_diagnostic(),
                               outcome=outcome.body["outcome"], attempt=resp.body["attempt"], record=resp.seq,
                               sid=resp.body.get("sid"), diagnostics={"reason": health.DIAGNOSTIC_REASON})]
            try:
                self.store.append("INTEGRITY_DIAGNOSTIC", rows, wall_at_commit=wall)
            except Rejected:
                pass  # already diagnosed (unique per record)

    def verify_integrity(self) -> list[int]:
        """RAW_INTEGRITY_EVERYWHERE_V1 outside replay: a mismatch found after a terminal outcome is a
        separate durable diagnostic; the outcome is never replaced and nothing is refetched."""
        corrupt = []
        for resp in self.store.rows("RESPONSE"):
            try:
                self.store.read_raw(resp.body["raw_sha"])
            except RawCorrupt:
                corrupt.append(resp.seq)
                self._diagnose(resp.body["raw_sha"])
        self.process_pending()
        return corrupt

    # ------------------------------------------------------------------ operator actions ----------
    def resolve(self, item_key: str, reason: str = "operator decision") -> dict:
        self.derive_terminals()
        body = {"action": "RESOLVE", "item_key": item_key, "reason": reason, "operator": "operator", "valid": False}

        def check(s: FomcStore) -> None:
            body["valid"], body["validity"] = state.resolve_validity(s, item_key)

        seq = self.store.append("OPERATOR", [("OPERATOR", item_key, body)], check)
        return {"seq": seq, "valid": body["valid"], "validity": body["validity"]}

    def manual_retry(self, work: dict, reason: str = "operator retry") -> int:
        """MANUAL_RETRY of a named work item (retry_policy.manual_retry). Backfill work is the validated
        manifest entry (manifest record, source item): its URL and manifest come from that entry, never
        from the feed, and the observation keeps mode HISTORICAL_BACKFILL."""
        kind, sid = work["kind"], work["sid"]
        if kind == "HISTORICAL_BACKFILL":
            manifest, url = self._manifest_entry(sid, work.get("manifest"))
            work = {"kind": kind, "sid": sid, "manifest": manifest, "url": url, "mode": "HISTORICAL_BACKFILL"}
        elif kind in ("LIVE_ACQUISITION", "REOBSERVATION"):
            work = dict(work, url=self._url_of(sid), mode="LIVE")
        else:
            raise ValueError(f"MANUAL_RETRY is valid only for LIVE_ACQUISITION, REOBSERVATION or HISTORICAL_BACKFILL work, not {kind}")
        return self.store.append("OPERATOR", [("OPERATOR", work["sid"], {"action": "MANUAL_RETRY", "work": work, "reason": reason})])

    def _manifest_entry(self, sid: str, manifest: int | None) -> tuple[int, str]:
        found = [(m.seq, e["url"]) for m in self.store.view().rows("MANIFEST") if m.body["valid"]
                 for e in m.body["entries"] if e["sid"] == sid and (manifest is None or m.seq == manifest)]
        if not found:
            raise ValueError("no validated manifest entry for this source item")
        if len(found) > 1:
            raise ValueError("the source item is in several manifests: name the manifest record")
        return found[0]

    def submit_manifest(self, raw: bytes, provenance: str) -> dict:
        body = {"sha": spec.sha256_bytes(raw), "length": len(raw), "provenance": provenance, "valid": False}
        try:
            entries, raw_count = _validate_manifest(raw)
            body.update(valid=True, version=1, raw_count=raw_count, dedup_count=len(entries), entries=entries)
        except ValueError as exc:
            body["reason"] = str(exc)
        seq = self.store.append("MANIFEST", [("MANIFEST", body["sha"], body)])
        return dict(body, seq=seq)


def _order(target: dict, instant: float) -> tuple:
    """Order of eligible work within its class (reobservation_scheduler.selection)."""
    if target["kind"] == "HISTORICAL_BACKFILL":
        return (target["manifest"], instant, target["sid"])  # (manifest commit_seq, next eligible instant, sid)
    if target["kind"] == "REOBSERVATION":
        return (instant, target["sid"], target["offset"])
    return (instant, target["sid"])  # LIVE_ACQUISITION


def _reject_duplicates(pairs):
    keys = [k for k, _v in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys")
    return dict(pairs)


def _validate_manifest(raw: bytes) -> tuple[list[dict], int]:
    """HISTORICAL_BACKFILL_MANIFEST_V1_BYTES, validated whole before any request."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("BOM not allowed")
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest is not UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != {"version", "urls"}:
        raise ValueError("manifest must be exactly {version, urls}")
    if type(doc["version"]) is not int or doc["version"] != 1:
        raise ValueError("version must be the JSON integer 1")
    urls = doc["urls"]
    if not isinstance(urls, list) or len(urls) > spec.MANIFEST_MAX_RAW_ENTRIES:
        raise ValueError("urls must be a list of at most 1000 raw entries")
    entries = {}
    for value in urls:
        if not isinstance(value, str) or not value:
            raise ValueError("every entry must be a non-empty string")
        try:
            admitted = identity.admit_url(value)
        except identity.UrlRejected as exc:
            raise ValueError(f"entry rejected: {exc}") from exc
        if not identity.is_statement_family_path(admitted.path):
            raise ValueError("entry outside the statement family path")
        entries[identity.source_item_id(admitted.canonical)] = admitted.canonical
    return [{"sid": sid, "url": entries[sid]} for sid in sorted(entries)], len(urls)
