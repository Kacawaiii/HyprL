"""Tooling for the FOMC pilot capture. Nothing here sends a request to the provider: only the service
does, through its transport and FIX15 limiter. Store, raws, logs and fixtures stay outside Git
(redistribution.raw_storage = LOCAL_RESTRICTED).

    pilot snapshot  --store S --log snapshots.jsonl            record a snapshot read (resolved or not)
    pilot fixtures  --store S --out DIR NAME=URL...           export acquired backfill fixtures
    pilot supervise --store S --unit U --service-log L ...    alerts, liveness, hourly snapshots, closure
    pilot close     --store S --unit U --copy C ...           clean stop, consistent copy, offline checks

(`pilot` is `python -m scripts.trading_lab.fomc.pilot`.)
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import syslog
import time

from scripts.trading_lab.fomc import ledger, snapshot, spec, state
from scripts.trading_lab.fomc.clock import iso, parse_iso
from scripts.trading_lab.fomc.store import FomcStore

PROBES_S = (0, 120, 300, 900)  # read at now, then a little earlier, until a resolved read is found


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _open(store_dir: Path) -> FomcStore:
    return FomcStore(Path(store_dir), wall_clock=_now)


def _alert(message: str, priority=syslog.LOG_CRIT) -> None:
    syslog.openlog("fomc-pilot", syslog.LOG_PID, syslog.LOG_USER)
    syslog.syslog(priority, message)


# ------------------------------------------------------------------ progress and snapshots ---------
def progress(store) -> dict:
    """Durable progress counters (read-only)."""
    view = store.view()
    invoked = view.rows("TRANSPORT_INVOKED")
    responses = view.rows("RESPONSE")
    outcomes = Counter(o.body["outcome"] for o in view.rows("ATTEMPT_OUTCOME"))
    cycles = Counter(c.body["result"] for c in view.rows("CYCLE_CONCLUSION"))
    return {
        "horizon": view.horizon(), "epochs": len(view.rows("EPOCH")),
        "attempts": dict(Counter(t.body["kind"] for t in invoked)),
        "attempt_outcomes": dict(outcomes),
        "responses": len(responses), "verdicts": dict(Counter(r.body["verdict"] for r in responses)),
        "late_evidence": sum(1 for r in responses if r.body["late_evidence"]),
        "processing_outcomes": dict(Counter(o.body["outcome"] for o in view.rows("PROCESSING_OUTCOME"))),
        "cycles": dict(cycles), "candidates": dict(Counter(c.body["shape"] for c in view.rows("CANDIDATE"))),
        "revisions": len(view.rows("REVISION")), "manifests": len(view.rows("MANIFEST")),
        "storage_incidents": len(view.rows("STORAGE_INCIDENT")),
        "first_wall": view.txns()[0][2] if view.txns() else None, "last_wall": view.txns()[-1][2] if view.txns() else None,
    }


def take_snapshot(store_dir: Path, log: Path, *, now: datetime | None = None) -> dict:
    """Read events_as_of at T = now, then slightly earlier T, at the current horizon H, and append
    every read to `log` (JSON lines) until one is FOMC_RESOLVED. A recent unresolved read is normal
    until its causal witness exists; no rule is bent to make it resolve."""
    now = now or _now()
    store = _open(store_dir)
    try:
        H = store.horizon()
        records = []
        for back in PROBES_S:
            snap = snapshot.events_as_of(store, now - timedelta(seconds=back), H)
            record = {"recorded_at": iso(_now()), "T": snap["T"], "H": snap["H"], "read_state": snap["read_state"],
                      "resolved": snap["read_state"] == "FOMC_RESOLVED", "identity": snap["identity"],
                      "P": snap.get("P"), "discovery": snap.get("discovery"),
                      "health": {k: [v["result_state"], v.get("reason")] for k, v in snap.get("health", {}).items()},
                      "items": [{"sid": i["sid"], "step": i["step"], "state": i["state"],
                                 "revision": i.get("revision"), "live_available": i.get("live_available")}
                                for i in snap.get("items", [])],
                      "probe_back_s": back}
            records.append(record)
            if record["resolved"]:
                break
        record = dict(records[-1], progress=progress(store))
        records[-1] = record
        with open(log, "a", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps(r, sort_keys=True) + "\n")
        return record
    finally:
        store.close()


# ------------------------------------------------------------------ fixtures ----------------------
EXPECTED = {  # what each official fixture must show once processed (timestamps.release_time)
    "summer_edt": {"semantics": "EXACT", "zone": "EDT", "utc_hour": 18},
    "winter_est": {"semantics": "EXACT", "zone": "EST", "utc_hour": 19},
    "immediate_release": {"semantics": "IMMEDIATE", "text": "For immediate release"},
}


def export_fixtures(store_dir: Path, out: Path, wanted: dict[str, str]) -> dict:
    """Export the HISTORICAL_BACKFILL records of `wanted` {name: url}: exact bytes, the headers kept by
    the store, URLs, digests, provenance and the processed normalized fields, and check them against
    EXPECTED. Read-only on the store."""
    store = _open(store_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    try:
        view = store.view()
        for name, url in wanted.items():
            records = [r for r in view.rows("RESPONSE") if r.body["request_url"] == url
                       and r.body["mode"] == "HISTORICAL_BACKFILL" and r.body["surface"] == "primary"]
            verified = [r for r in records if r.body["verdict"] == "CLOCK_VERIFIED" and not r.body["late_evidence"]]
            if not records:
                index[name] = {"url": url, "status": "NOT_ACQUIRED"}
                continue
            record = (verified or records)[-1]
            body = store.read_raw(record.body["raw_sha"])
            outcome = state.processing_outcome(view, record.seq)
            link = view.rows("LINK", key=str(record.seq))
            revision = view.rows("REVISION", key=link[0].body["revision"])[0].body if link else None
            folder = out / name
            folder.mkdir(exist_ok=True)
            (folder / "body.html").write_bytes(body)
            meta = {
                "name": name, "url": url, "record": record.seq, "attempt": record.body["attempt"],
                "mode": record.body["mode"], "work": record.body["work"], "final_url": record.body["final_url"],
                "redirect_chain": record.body["redirect_chain"], "status": record.body["status"],
                "content_type_lines": record.body["content_type_lines"],
                "content_encoding": record.body["content_encoding"], "date_lines": record.body["date_lines"],
                "age_lines": record.body["age_lines"], "wall_at_receipt": record.body["wall_at_receipt"],
                "observed_at": record.body["observed_at"], "verdict": record.body["verdict"],
                "raw_sha256": record.body["raw_sha"], "byte_length": record.body["byte_length"],
                "body_sha256_recomputed": spec.sha256_bytes(body),
                "processing_outcome": outcome.body["outcome"] if outcome else None,
                "normalized": {k: revision.get(k) for k in (
                    "official_statement_date", "title", "declared_release_at", "declared_release_text",
                    "declared_release_semantics", "timestamp_semantics", "declared_release_trust_verdict",
                    "content_source_available_at", "observation_mode", "revision_id")} if revision else None,
                "provenance": {"store": str(store_dir), "spec_hash": spec.SPEC_HASH, "spec_revision": spec.SPEC_REVISION,
                               "acquired_by": "FomcService HISTORICAL_BACKFILL manifest (FIX15 limiter, single owner)",
                               "exported_at": iso(_now())},
            }
            meta["checks"] = _check_fixture(name, meta)
            (folder / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            index[name] = {"url": url, "status": "ACQUIRED", "raw_sha256": meta["raw_sha256"],
                           "byte_length": meta["byte_length"], "record": record.seq, "verdict": meta["verdict"],
                           "outcome": meta["processing_outcome"], "checks": meta["checks"]}
    finally:
        store.close()
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def _check_fixture(name: str, meta: dict) -> dict:
    expected = EXPECTED.get(name)
    normalized = meta["normalized"] or {}
    checks = {"digest_matches": meta["raw_sha256"] == meta["body_sha256_recomputed"],
              "historical_backfill": meta["mode"] == "HISTORICAL_BACKFILL",
              "in_scope": meta["processing_outcome"] in ("NORMALIZED_REVISION_COMMITTED", "NORMALIZED_SAME_CONTENT_NO_NEW_REVISION")}
    if expected:
        checks["semantics"] = normalized.get("declared_release_semantics") == expected["semantics"]
        if expected["semantics"] == "EXACT":
            declared = normalized.get("declared_release_at")
            checks["zone_label"] = (normalized.get("declared_release_text") or "").endswith(expected["zone"])
            checks["utc_hour"] = declared is not None and parse_iso(declared).hour == expected["utc_hour"]
        else:
            checks["text"] = normalized.get("declared_release_text") == expected["text"]
            checks["declared_null"] = normalized.get("declared_release_at") is None
    checks["ok"] = all(checks.values())
    return checks


# ------------------------------------------------------------------ closure -----------------------
def stop_service(unit: str | None, pid: int | None, timeout_s: float = 180.0) -> dict:
    """Clean stop: SIGTERM through systemd (or to the pid), then wait for the process to exit."""
    started = time.monotonic()
    if unit:
        active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True).stdout.strip()
        if active in ("active", "activating", "deactivating"):
            subprocess.run(["systemctl", "--user", "stop", unit], check=False, timeout=timeout_s)
        show = subprocess.run(["systemctl", "--user", "show", unit, "-p", "Result", "-p", "ExecMainStatus",
                               "-p", "ExecMainExitTimestamp"], capture_output=True, text=True).stdout
        return {"unit": unit, "was": active, "seconds": round(time.monotonic() - started, 1),
                **dict(line.split("=", 1) for line in show.strip().splitlines() if "=" in line)}
    if pid:
        import os
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return {"pid": pid, "was": "not running"}
        while time.monotonic() - started < timeout_s:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return {"pid": pid, "was": "running", "seconds": round(time.monotonic() - started, 1)}
            time.sleep(0.5)
        return {"pid": pid, "was": "running", "error": "did not exit"}
    return {"was": "no service given"}


def owner_free(store_dir: Path) -> bool:
    with open(Path(store_dir) / "owner.lock", "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def consistent_copy(store_dir: Path, copy_dir: Path) -> dict:
    """A transaction-consistent copy of the SQLite store, WAL frames included (SQLite backup API from a
    read-only connection, never a file copy of the database alone), then the immutable raw bodies."""
    copy_dir.mkdir(parents=True, exist_ok=False)
    source = sqlite3.connect(f"file:{Path(store_dir) / 'fomc.sqlite3'}?mode=ro", uri=True)
    target = sqlite3.connect(copy_dir / "fomc.sqlite3")
    try:
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {t: target.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("txn", "rec")}
        source_counts = {t: source.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("txn", "rec")}
    finally:
        target.close()
        source.close()
    shutil.copytree(Path(store_dir) / "raw", copy_dir / "raw")
    raws = sum(1 for p in (copy_dir / "raw").rglob("*") if p.is_file())
    return {"integrity_check": integrity, "rows": counts, "source_rows": source_counts, "raw_files": raws,
            "equal": counts == source_counts and integrity == "ok"}


def _in_flight_overlaps(view, invoked) -> list[dict]:
    """One fetch per source item across every class and episode, and one feed poll, in flight at a
    time: for consecutive attempts of the same item (or of the feed), the earlier one's outcome must be
    committed before the later TRANSPORT_INVOKED."""
    groups: dict[str, list] = {}
    for t in invoked:
        groups.setdefault("FEED" if t.key == ledger.FEED_KEY else f"item:{t.body.get('sid')}", []).append(t)
    overlaps = []
    for group, attempts in groups.items():
        for earlier, later in zip(attempts, attempts[1:]):
            outcome = view.rows("ATTEMPT_OUTCOME", key=str(earlier.seq))
            if not outcome or outcome[0].seq > later.seq:
                overlaps.append({"group": group, "earlier": earlier.seq, "later": later.seq,
                                 "earlier_outcome_seq": outcome[0].seq if outcome else None})
    return overlaps


def fix15_coverage(view, invoked) -> dict:
    """FIX15 over the starts the store actually records. Initial grants are durable (TRANSPORT_INVOKED
    grant_mono, per epoch). Continuation grants of redirect hops are not recorded by this store: they
    are covered only when every attempt is known to have had a single physical request; otherwise the
    check over all physical starts is NOT_PROVEN, never PASS."""
    per_epoch = {}
    for epoch in sorted({t.body["epoch"] for t in invoked}):
        grants = sorted(t.body["grant_mono"] for t in invoked if t.body["epoch"] == epoch)
        per_epoch[epoch] = {
            "initial_grants": len(grants),
            "spacing_ok": all(b - a >= spec.SPACING_S for a, b in zip(grants, grants[1:])),
            "window_ok": all(sum(1 for g in grants if 0 <= x - g < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for x in grants)}
    multi_hop = unknown = 0
    for t in invoked:
        record = view.rows("RESPONSE", key=str(t.seq))
        if not record:
            unknown += 1  # failed, interrupted or lost: how many physical requests it made is not recorded
        elif len(record[0].body["redirect_chain"]) > 1:
            multi_hop += 1  # its continuation grants are not recorded
    initial_ok = all(e["spacing_ok"] and e["window_ok"] for e in per_epoch.values())
    covered = multi_hop == 0 and unknown == 0
    return {"initial_grants": {"status": "PASS" if initial_ok else "FAIL", "per_epoch": per_epoch},
            "continuation_grants": {"status": "NONE_BY_RECORD" if covered else "NOT_PROVEN",
                                    "multi_hop_responses": multi_hop, "attempts_without_hop_record": unknown},
            "all_physical_starts": "FAIL" if not initial_ok else ("PASS" if covered else "NOT_PROVEN")}


def recheck_status(view) -> dict:
    """Every obligation of every LIVE-anchored item, from durable records only: SATISFIED, IN_PROGRESS
    (its episode is open), SUSPENDED (its episode suspended), PENDING_DUE (due, no episode yet) or
    NOT_YET_DUE (by the store's NOW_LB) - never inferred from elapsed time."""
    from scripts.trading_lab.fomc.identity import reobservation_episode_key
    items, counts = {}, {}
    for candidate in view.rows("CANDIDATE"):
        sid = candidate.key
        obligations = state.obligations(view, sid)
        if not obligations:
            continue
        rows = []
        for o in obligations:
            key = reobservation_episode_key(sid, o["anchor"], o["offset"])
            opened = view.rows("EPISODE_OPEN", key=key)
            if o["satisfied"]:
                status = "SATISFIED"
            elif opened:
                status = {"OPEN": "IN_PROGRESS", "SUSPENDED": "SUSPENDED"}.get(
                    state.derived_status(view, opened[0]), state.derived_status(view, opened[0]))
            else:
                status = "PENDING_DUE" if o["pending_due"] else "NOT_YET_DUE"
            rows.append({"offset": o["offset"], "due_at": iso(o["due_at"]), "status": status,
                         "attempts": len(ledger.attempts_of_episode(view, key)) if opened else 0})
            counts.setdefault(str(o["offset"]), Counter())[status] += 1
        items[sid] = {"anchor": obligations[0]["anchor"], "obligations": rows}
    lb = state.now_lb(view)
    return {"now_lb": iso(lb) if lb else None, "by_offset": {k: dict(v) for k, v in counts.items()}, "items": items}


def audit(store) -> dict:
    """Capture invariants on a closed store (read-only). Every check is computed from durable records;
    what the records cannot show is reported NOT_PROVEN, not passed."""
    view = store.view()
    invoked = view.rows("TRANSPORT_INVOKED")
    without_outcome = [t.seq for t in invoked if not view.rows("ATTEMPT_OUTCOME", key=str(t.seq))]
    per_key = Counter(t.key for t in invoked if t.key != ledger.FEED_KEY)
    overlaps = _in_flight_overlaps(view, invoked)
    fix15 = fix15_coverage(view, invoked)
    requests_per_item: Counter = Counter()  # upper bound: hops of each response, 4 for an attempt without one
    for t in invoked:
        if t.body.get("sid") is None or t.body.get("mode") != "LIVE":
            continue
        record = view.rows("RESPONSE", key=str(t.seq))
        requests_per_item[t.body["sid"]] += len(record[0].body["redirect_chain"]) if record else spec.MAX_PHYSICAL_PER_ATTEMPT
    responses = view.rows("RESPONSE")
    false_zero = []
    for cycle in view.rows("CYCLE_CONCLUSION"):
        if cycle.body["result"] != "EVENTS_OBSERVED_ZERO":
            continue
        B = cycle.body["B"]
        if not state.live_eligible(view.row_at("RESPONSE", B)) or state.outstanding(view, upto=B - 1):
            false_zero.append(B)
    checks = {
        "no_attempt_without_outcome": without_outcome == [],
        "at_most_six_attempts_per_key": max(per_key.values(), default=0) <= spec.ATTEMPTS_PER_EPISODE,
        "one_fetch_per_item_across_classes_and_one_feed_poll_in_flight": overlaps == [],
        "fix15_initial_grants_per_epoch": fix15["initial_grants"]["status"] == "PASS",
        "at_most_120_requests_per_live_item_upper_bound": max(requests_per_item.values(), default=0) <= 120,
        "one_processing_outcome_per_record": all(len(view.rows("PROCESSING_OUTCOME", key=str(r.seq))) == 1 for r in responses),
        "unique_episode_keys": len({e.key for e in view.rows("EPISODE_OPEN")}) == len(view.rows("EPISODE_OPEN")),
        "no_false_zero": false_zero == [],
    }
    not_proven = []
    if fix15["all_physical_starts"] == "NOT_PROVEN":
        not_proven.append("FIX15 over all physical starts: continuation grants of redirect hops are not recorded "
                          f"({fix15['continuation_grants']['multi_hop_responses']} multi-hop responses, "
                          f"{fix15['continuation_grants']['attempts_without_hop_record']} attempts without a hop record)")
    return {"checks": checks, "ok": all(checks.values()), "not_proven": not_proven, "fix15": fix15,
            "attempts_without_outcome": without_outcome, "in_flight_overlaps": overlaps,
            "max_attempts_per_key": max(per_key.values(), default=0),
            "max_requests_per_live_item_upper_bound": max(requests_per_item.values(), default=0),
            "false_zero_cycles": false_zero}


def verify_copy(copy_dir: Path, snapshots_log: Path | None) -> dict:
    """Offline verification of the copy: every recorded read - resolved or not - re-read and replayed
    at its (T, H) with the same identity, source health replay, capture invariants, recheck status.
    No network."""
    store = _open(copy_dir)
    try:
        H = store.horizon()
        recorded = [json.loads(line) for line in open(snapshots_log, encoding="utf-8")] if snapshots_log else []
        reread, replayed = [], []
        for r in recorded:
            T = parse_iso(r["T"])
            try:
                again = snapshot.events_as_of(store, T, r["H"])
                reread.append(True if again["identity"] == r["identity"] else f"identity differs at H={r['H']}")
            except snapshot.SnapshotFailed as exc:
                reread.append(f"failed: {exc}")
            try:
                replayed.append(True if snapshot.replay(store, T, r["H"])["identity"] == r["identity"]
                                else f"identity differs at H={r['H']}")
            except snapshot.SnapshotFailed as exc:
                replayed.append(f"failed: {exc}")
        health_ok, health_error = True, None
        try:
            snapshot.verify_health(store, H)
        except snapshot.ReplayFailed as exc:
            health_ok, health_error = False, str(exc)
        resolved = [r for r in recorded if r["resolved"]]
        result = {"horizon": H, "recorded_reads": len(recorded), "resolved_reads": len(resolved),
                  "unresolved_reads": len(recorded) - len(resolved),
                  "reread_identical": sum(1 for x in reread if x is True), "replay_identical": sum(1 for x in replayed if x is True),
                  "reread_failures": [x for x in reread if x is not True],
                  "replay_failures": [x for x in replayed if x is not True], "health_replay": health_ok,
                  "health_error": health_error, "audit": audit(store), "rechecks": recheck_status(store.view()),
                  "progress": progress(store)}
        result["ok"] = (bool(resolved) and all(x is True for x in reread) and all(x is True for x in replayed)
                        and health_ok and result["audit"]["ok"])
        return result
    finally:
        store.close()


def pilot_duration(stop: dict, progress_: dict, *, planned_start: datetime | None, planned_end: datetime | None,
                   closed_at: datetime) -> dict:
    """Whether the pilot ran for its planned duration - a separate verdict from integrity: the service
    must have been running when the closure stopped it, at or after the planned end, as one owner
    epoch, with durable activity up to the end."""
    was_running = stop.get("was") in ("active", "running")
    last = parse_iso(progress_["last_wall"]) if progress_.get("last_wall") else None
    first = parse_iso(progress_["first_wall"]) if progress_.get("first_wall") else None
    detail = {"service_running_at_closure": was_running, "epochs": progress_.get("epochs"),
              "planned_start": iso(planned_start) if planned_start else None,
              "planned_end": iso(planned_end) if planned_end else None, "closed_at": iso(closed_at),
              "first_durable_activity": progress_.get("first_wall"), "last_durable_activity": progress_.get("last_wall")}
    reasons = []
    if planned_end is None or planned_start is None:
        reasons.append("no planned window given")
    else:
        if closed_at < planned_end:
            reasons.append("closed before the planned end")
        if first is None or first > planned_start + timedelta(minutes=5):
            reasons.append("no durable activity from the planned start")
        if last is None or last < planned_end - timedelta(minutes=5):
            reasons.append("no durable activity up to the planned end")
    if not was_running:
        reasons.append("the service was not running when the closure came (premature stop or crash)")
    if progress_.get("epochs") != 1:
        reasons.append(f"{progress_.get('epochs')} owner epochs (restarts)")
    detail["verdict"] = "ACCOMPLISHED" if not reasons else "NOT_ACCOMPLISHED"
    detail["reasons"] = reasons
    return detail


def close(store_dir: Path, copy_dir: Path, report: Path, *, unit: str | None = None, pid: int | None = None,
          snapshots_log: Path | None = None, notify: bool = True, planned_start: datetime | None = None,
          planned_end: datetime | None = None, stopper=None) -> dict:
    """Clean stop, owner check, consistent copy and offline verification. Two separate verdicts:
    integrity (copy, re-read, replay, health replay, invariants) and pilot duration."""
    result = {"started": iso(_now()), "store": str(store_dir), "copy": str(copy_dir)}
    result["stop"] = stopper() if stopper is not None else stop_service(unit, pid)
    closed_at = _now()
    result["owner_free"] = owner_free(store_dir)
    if not result["owner_free"]:
        result["integrity"] = {"verdict": "NOT_VERIFIED", "reason": "the store still has an owner: no copy taken"}
        result["pilot_duration"] = {"verdict": "NOT_ACCOMPLISHED", "reasons": ["closure could not take the store"]}
    else:
        result["copy_check"] = consistent_copy(store_dir, copy_dir)
        verification = verify_copy(copy_dir, snapshots_log)
        result["verification"] = verification
        valid = result["copy_check"]["equal"] and verification["ok"]
        result["integrity"] = {"verdict": "VALID" if valid else "INVALID",
                               "not_proven": verification["audit"]["not_proven"]}
        result["pilot_duration"] = pilot_duration(result["stop"], verification["progress"],
                                                  planned_start=planned_start, planned_end=planned_end, closed_at=closed_at)
    result["integrity_ok"] = result["integrity"]["verdict"] == "VALID"
    result["duration_ok"] = result["pilot_duration"]["verdict"] == "ACCOMPLISHED"
    result["ok"] = result["integrity_ok"] and result["duration_ok"]
    result["finished"] = iso(_now())
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if notify:  # the monitored journal only hears about real closures
        _alert(f"FOMC pilot closure: integrity {result['integrity']['verdict']}, duration "
               f"{result['pilot_duration']['verdict']}, not proven {len(result['integrity'].get('not_proven', []))}: "
               f"report {report}", syslog.LOG_NOTICE if result["ok"] else syslog.LOG_CRIT)
    return result


# ------------------------------------------------------------------ supervision -------------------
def supervise(store_dir: Path, service_log: Path, alerts: Path, snapshots_log: Path, *, unit: str,
              close_at: datetime, copy_dir: Path, report: Path, snapshot_every_s: float = 3600.0,
              poll_s: float = 15.0, log_offset: int = 0, planned_start: datetime | None = None) -> dict:
    """Route FOMC-ALERT lines to syslog (journald, priority crit) and an alert file, raise an alert if
    the service stops before its closure, take a snapshot read every hour, then close at `close_at`.
    `log_offset` lets a replacement supervisor resume reading the service log where it is now."""
    offset, down_reported, next_snapshot = log_offset, False, time.monotonic()
    while _now() < close_at:
        if service_log.exists():
            with open(service_log, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset += len(chunk)
            for line in chunk.decode("utf-8", "replace").splitlines():
                if "FOMC-ALERT" in line or "Traceback" in line:
                    _alert(line)
                    with open(alerts, "a", encoding="utf-8") as out:
                        out.write(f"{iso(_now())} {line}\n")
        active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True).stdout.strip()
        if active != "active" and not down_reported:
            down_reported = True
            message = f"FOMC-SERVICE-DOWN unit={unit} state={active} before the scheduled closure"
            _alert(message)
            with open(alerts, "a", encoding="utf-8") as out:
                out.write(f"{iso(_now())} {message}\n")
        if time.monotonic() >= next_snapshot:
            try:
                take_snapshot(store_dir, snapshots_log)
            except Exception as exc:  # a failed read is reported, never fatal to supervision
                _alert(f"FOMC-SNAPSHOT-FAILED {type(exc).__name__}: {exc}")
            next_snapshot += snapshot_every_s
        time.sleep(poll_s)
    try:
        take_snapshot(store_dir, snapshots_log)  # a last read before the stop
    except Exception as exc:
        _alert(f"FOMC-SNAPSHOT-FAILED {type(exc).__name__}: {exc}")
    return close(store_dir, copy_dir, report, unit=unit, snapshots_log=snapshots_log,
                 planned_start=planned_start, planned_end=close_at)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p = sub.add_parser("fixtures")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("wanted", nargs="+", help="NAME=URL")
    p = sub.add_parser("supervise")
    for name in ("--store", "--service-log", "--alerts", "--snapshots", "--copy", "--report"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--close-at", required=True, help="UTC ISO instant of the closure")
    p.add_argument("--planned-start", help="UTC ISO instant the pilot started (duration verdict)")
    p.add_argument("--log-offset", default="0", help="byte offset in the service log to resume from, or 'end'")
    p = sub.add_parser("close")
    for name in ("--store", "--copy", "--report"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--snapshots", type=Path)
    p.add_argument("--unit")
    p.add_argument("--pid", type=int)
    p.add_argument("--planned-start")
    p.add_argument("--planned-end")
    args = parser.parse_args(argv)
    if args.cmd == "snapshot":
        record = take_snapshot(args.store, args.log)
        print(json.dumps({k: record[k] for k in ("T", "H", "read_state", "identity", "discovery", "health")}, sort_keys=True))
        return 0
    if args.cmd == "fixtures":
        index = export_fixtures(args.store, args.out, dict(w.split("=", 1) for w in args.wanted))
        print(json.dumps(index, indent=2, sort_keys=True))
        return 0 if all(v.get("checks", {}).get("ok") for v in index.values()) else 1
    if args.cmd == "supervise":
        offset = args.service_log.stat().st_size if args.log_offset == "end" and args.service_log.exists() else int(
            args.log_offset if args.log_offset != "end" else 0)
        result = supervise(args.store, args.service_log, args.alerts, args.snapshots, unit=args.unit,
                           close_at=parse_iso(args.close_at), copy_dir=args.copy, report=args.report, log_offset=offset,
                           planned_start=parse_iso(args.planned_start) if args.planned_start else None)
        return 0 if result["ok"] else 1
    result = close(args.store, args.copy, args.report, unit=args.unit, pid=args.pid, snapshots_log=args.snapshots,
                   planned_start=parse_iso(args.planned_start) if args.planned_start else None,
                   planned_end=parse_iso(args.planned_end) if args.planned_end else None)
    print(json.dumps({k: result[k] for k in ("integrity_ok", "duration_ok", "stop", "owner_free")}, sort_keys=True))
    return 0 if result["integrity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
