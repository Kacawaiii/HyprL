"""The single-owner FOMC collector: epoch ownership, the logical fetch (TRANSPORT_INVOKED -> transport
-> response or outcome), reconciliation, bounded episodes, class-rotation selection, operator actions
and backfill manifests (retry_policy, reconciliation, selection, operator_resolution, surfaces.backfill_manifest).
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import uuid
from datetime import timedelta

from scripts.trading_lab.fomc import clock as clockmod
from scripts.trading_lab.fomc import identity, ledger, processing, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.limiter import Limiter
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt, Rejected
from scripts.trading_lab.fomc.transport import FetchResult, Transport

CLASS_ORDER = ("FEED_DISCOVERY", "REOBSERVATION", "HISTORICAL_BACKFILL")
SAVE_RETRY_S = 5.0  # implementation choice: local save retry spacing inside the 120 s save deadline
CLASS_OF = {"FEED_POLL": "FEED_DISCOVERY", "LIVE_ACQUISITION": "FEED_DISCOVERY", "REOBSERVATION": "REOBSERVATION",
            "HISTORICAL_BACKFILL": "HISTORICAL_BACKFILL"}


class OwnershipUnavailable(RuntimeError):
    """Exclusive provider ownership cannot be established: no provider attempt may start."""


class Collector:
    def __init__(self, store: FomcStore, connector, clock, *, boot_id: str = "boot-1"):
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
        self.reconcile()

    def close(self) -> None:
        """Release ownership (also what a crash does to an flock)."""
        self._owner.close()

    # ------------------------------------------------------------------ one logical fetch ---------
    def fetch(self, work: dict, url: str, surface: str) -> FetchResult | dict:
        """Grant, TRANSPORT_INVOKED, transport. The attempt stays active (its task alive) until commit()."""
        def invoke(grant: float) -> int:
            seq = ledger.transport_invoked(self.store, epoch=self.epoch, work=work, grant_mono=grant)
            self._active_attempts[seq] = self.clock.mono()
            if work["kind"] == "FEED_POLL":
                self.last_feed_grant = grant
            return seq
        try:
            result = self.transport.fetch(url, surface, invoke=invoke,
                                          may_continue=lambda a: ledger.outcome_of(self.store, a) is None)
        except Rejected as exc:
            return {"status": "REJECTED", "reason": str(exc)}
        result.network_end_mono = self.clock.mono()
        return result

    def commit(self, result: FetchResult, work: dict, url: str, surface: str) -> dict:
        attempt = result.attempt_seq
        try:
            return self._commit(result, work, url, surface)
        finally:
            self._active_attempts.pop(attempt, None)  # the task ends here, with or without an outcome

    def _commit(self, result: FetchResult, work: dict, url: str, surface: str) -> dict:
        attempt = result.attempt_seq
        started = self._active_attempts.get(attempt)
        if started is not None and self.clock.mono() - started >= spec.ATTEMPT_ABSOLUTE_DEADLINE_S:
            # the 600 s bound closes the attempt first: whatever arrives now is LATE_EVIDENCE
            ledger.commit_attempt_outcome(self.store, attempt, "INTERRUPTED", {"reason": "no outcome 600 s after TRANSPORT_INVOKED"})
        if result.kind != "RESPONSE_200":
            outcome = "CANCELLED_AFTER_INVOKE" if result.kind == "ABANDONED" else result.kind
            ledger.commit_attempt_outcome(self.store, attempt, outcome, {"reason": result.reason})
            self.derive_terminals()
            return {"status": outcome, "reason": result.reason}
        save_deadline = result.network_end_mono + spec.SAVE_DEADLINE_S
        while True:
            if self.clock.mono() >= save_deadline:
                return self._persistence_failed(attempt, "local save not durable within 120 s of the network end")
            try:
                if self.persist_fault is not None:
                    self.persist_fault()
                digest = self.store.put_raw(result.body)
                seq, late = ledger.commit_response(self.store, attempt, self._fields(result, work, url, surface, digest))
                break
            except RawCorrupt:
                # The immutable slot for these bytes holds other bytes: never overwrite it. The exact
                # bytes of this attempt are therefore not durable -> LOCAL_PERSISTENCE_FAILED, no RESPONSE.
                # Older records sharing the digest get integrity diagnostics; their outcomes stay.
                self._diagnose(spec.sha256_bytes(result.body))
                return self._persistence_failed(attempt, "raw slot holds other bytes; never overwritten")
            except (OSError, sqlite3.OperationalError):
                self.clock.sleep(min(SAVE_RETRY_S, max(save_deadline - self.clock.mono(), 0.0)))  # retry locally, no network
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
        self.derive_terminals()
        return {"status": "LOCAL_PERSISTENCE_FAILED", "reason": reason}

    def run(self, work: dict, url: str, surface: str) -> dict:
        result = self.fetch(work, url, surface)
        if isinstance(result, dict):
            return result
        return self.commit(result, work, url, surface)

    def poll_feed(self) -> dict:
        return self.run({"kind": "FEED_POLL", "class": "FEED_DISCOVERY", "sid": None, "mode": "LIVE"}, spec.FEED_URL, "feed")

    # ------------------------------------------------------------------ reconciliation ------------
    def reconcile(self) -> None:
        """INTERRUPTED for attempts of other epochs, of ended tasks, or 600 s after TRANSPORT_INVOKED; a
        task that is still active is never declared dead before its deadline."""
        now = self.clock.mono()
        for attempt in ledger.attempts_without_outcome(self.store):
            started = self._active_attempts.get(attempt.seq) if attempt.body["epoch"] == self.epoch else None
            if started is not None and now - started < spec.ATTEMPT_ABSOLUTE_DEADLINE_S:
                continue
            reason = ("non-current epoch" if attempt.body["epoch"] != self.epoch
                      else "task ended without outcome" if started is None else "no outcome 600 s after TRANSPORT_INVOKED")
            ledger.commit_attempt_outcome(self.store, attempt.seq, "INTERRUPTED", {"reason": reason})
        self.process_pending()
        self.derive_terminals()

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
        self.store.append("RUN_DEAD", [("RUN_DEAD", str(seq), {"run_id": run_id, "reason": reason})])
        if self._active_runs.get(seq) == run_id:
            self._active_runs.pop(seq)

    def start_processing(self, seq: int) -> str | None:
        run_id = processing.start_run(self.store, seq, epoch=self.epoch, start_mono=self.clock.mono())
        if run_id is not None:
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
        self._active_runs.pop(seq, None)
        return outcome

    def process_pending(self) -> None:
        for resp in self.store.rows("RESPONSE"):
            seq = resp.seq
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
                self.finish_processing(seq, run_id)

    def derive_terminals(self) -> None:
        for episode in self.store.rows("EPISODE_OPEN"):
            status = state.derived_status(self.store, episode)
            if status in ("SUCCEEDED", "SUSPENDED") and ledger.episode_status(self.store, episode.key) == "OPEN":
                ledger.close_episode(self.store, episode.key, status, "derived")

    # ------------------------------------------------------------------ episodes ------------------
    def _url_of(self, sid: str) -> str:
        return self.store.rows("CANDIDATE", key=sid)[0].body["url"]

    def open_episodes(self) -> None:
        store = self.store
        for acq in store.rows("ACQUISITION"):
            sid = acq.key
            key = identity.acquisition_episode_key(sid, acq.seq)
            if state.anchor(store, sid) is None and not store.rows("EPISODE_OPEN", key=key):
                ledger.open_episode(store, key, {"kind": "LIVE_ACQUISITION", "sid": sid, "url": acq.body["url"], "mode": "LIVE"})
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
                    ledger.open_episode(store, key, {"kind": "REOBSERVATION", "sid": sid, "url": cand.body["url"],
                                                     "anchor": ob["anchor"], "offset": ob["offset"], "mode": "LIVE"}, check)
                    break
        for manifest in store.rows("MANIFEST"):
            if not manifest.body["valid"]:
                continue
            for entry in manifest.body["entries"]:
                key = identity.backfill_episode_key(manifest.seq, entry["sid"])
                if not store.rows("EPISODE_OPEN", key=key):
                    ledger.open_episode(store, key, {"kind": "HISTORICAL_BACKFILL", "sid": entry["sid"], "url": entry["url"],
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
            ledger.open_episode(store, key, {"kind": "MANUAL_RETRY", "work": work, "sid": work["sid"]})

    def _ready(self, episode, table, lb) -> tuple[bool, float]:
        store = self.store
        body = episode.body
        sid = body["sid"]
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
            prior = [o for e in state.item_episodes(store, sid, live_only=False) if e.key != episode.key
                     for _a, o in state.attempt_outcomes(store, e.key) if o is not None and o.body["outcome"] != "RESPONSE"]
            if not prior:
                return True, 0.0
            last, gap = prior[-1], spec.FIRST_ATTEMPT_GAP_S
        start = state.avail_of(table, last.seq)
        if start is None or lb is None:
            return False, 0.0  # held until verified server time exists: never burned by waiting
        threshold = start + timedelta(seconds=gap)
        return lb >= threshold, threshold.timestamp()

    def feed_poll_due(self) -> bool:
        if ledger.attempts_without_outcome(self.store, feed=True):
            return False
        if processing.earlier_feed_unterminated(self.store, self.store.horizon() + 1):
            return False
        return self.last_feed_grant is None or self.clock.mono() >= self.last_feed_grant + spec.FEED_CADENCE_S

    def eligible_work(self) -> list[dict]:
        self.open_episodes()
        table = state.availability(self.store, self.store.horizon())
        lb = state.now_lb(self.store)
        work = []
        if self.feed_poll_due():
            work.append({"class": "FEED_DISCOVERY", "poll": True, "order": (0.0, "")})
        for episode in self.store.rows("EPISODE_OPEN"):
            if state.derived_status(self.store, episode) != "OPEN":
                continue
            ready, order = self._ready(episode, table, lb)
            if not ready:
                continue
            body = episode.body
            kind = body["work"]["kind"] if body["kind"] == "MANUAL_RETRY" else body["kind"]
            work.append({"class": CLASS_OF[kind], "episode": episode, "order": (order, body["sid"])})
        return work

    def step(self) -> dict | None:
        """One selection decision (ELIGIBLE_CLASS_ALTERNATION_V1) and its logical fetch."""
        self.reconcile()
        work = self.eligible_work()
        if not work:
            return None
        invoked = self.store.rows("TRANSPORT_INVOKED")
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
            chosen = sorted(choices, key=lambda w: w["order"])[0]
            if chosen.get("poll"):
                return self.poll_feed()
            return self.run_episode(chosen["episode"])
        return None

    def run_episode(self, episode) -> dict:
        body = episode.body
        target = body["work"] if body["kind"] == "MANUAL_RETRY" else body
        work = {"kind": body["kind"], "class": CLASS_OF[target["kind"]], "episode_key": episode.key,
                "sid": body["sid"], "mode": target["mode"]}
        return self.run(work, target["url"], "primary")

    def run_until_idle(self, max_steps: int = 50) -> list[dict]:
        out = []
        for _ in range(max_steps):
            done = self.step()
            if done is None:
                break
            out.append(done)
        return out

    def _diagnose(self, digest: str) -> None:
        for resp in self.store.rows("RESPONSE"):
            if (resp.body["raw_sha"] == digest and state.processing_outcome(self.store, resp.seq) is not None
                    and not self.store.rows("INTEGRITY_DIAGNOSTIC", key=str(resp.seq))):
                self.store.append("INTEGRITY_DIAGNOSTIC", [("INTEGRITY_DIAGNOSTIC", str(resp.seq),
                                                           {"record": resp.seq, "raw_sha": digest})])

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
        work = dict(work, url=self._url_of(work["sid"]),
                    mode="HISTORICAL_BACKFILL" if work["kind"] == "HISTORICAL_BACKFILL" else "LIVE")
        return self.store.append("OPERATOR", [("OPERATOR", work["sid"], {"action": "MANUAL_RETRY", "work": work, "reason": reason})])

    def submit_manifest(self, raw: bytes, provenance: str) -> dict:
        body = {"sha": spec.sha256_bytes(raw), "length": len(raw), "provenance": provenance, "valid": False}
        try:
            entries, raw_count = _validate_manifest(raw)
            body.update(valid=True, version=1, raw_count=raw_count, dedup_count=len(entries), entries=entries)
        except ValueError as exc:
            body["reason"] = str(exc)
        seq = self.store.append("MANIFEST", [("MANIFEST", body["sha"], body)])
        return dict(body, seq=seq)


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
