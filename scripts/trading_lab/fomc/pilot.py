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
        "max_gap_s": _max_gap(view.txns()),
    }


def _max_gap(txns) -> float | None:
    walls = [parse_iso(t[2]) for t in txns]
    return round(max((b - a).total_seconds() for a, b in zip(walls, walls[1:])), 1) if len(walls) > 1 else None


# A running owner commits at least one feed poll per FEED_CADENCE_S once the previous poll has ended (at most
# ATTEMPT_ABSOLUTE_DEADLINE_S): a longer silence means the capture itself was suspended (host sleep, VM pause).
CAPTURE_GAP_BOUND_S = spec.ATTEMPT_ABSOLUTE_DEADLINE_S + spec.FEED_CADENCE_S + 60


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


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
                    "content_source_available_at", "observation_mode", "revision_id", "content_hash",
                    "first_raw_sha256", "content_identity")} if revision else None,
                "content_identity": link[0].body.get("canonicalization") if link else None,
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


def fix15_journal(view, invoked) -> dict:
    """FIX15 proved over the grant journal (request_accounting.grant_journal): completeness first (orders
    contiguous per epoch, one initial grant per TRANSPORT_INVOKED in its transaction, per-attempt hops
    contiguous and equal to the recorded counts, redirect chains equal to the granted URLs), then the
    rules over every journaled grant (spacing, window, embargo, no reuse). A missing or inconsistent
    trace is NOT_PROVEN; a rule broken by the journal is FAIL; nothing is reconstructed."""
    grants = view.rows("GRANT")
    epochs = {e.key: e.body for e in view.rows("EPOCH")}
    missing, violations = [], []
    by_epoch: dict[str, list] = {}
    for g in grants:
        by_epoch.setdefault(g.body["epoch"], []).append(g)
    per_epoch = {}
    for epoch, rows in by_epoch.items():
        orders = [g.body["order"] for g in rows]
        if orders != list(range(1, len(rows) + 1)):
            missing.append(f"epoch {epoch[:8]}: grant orders are not contiguous from 1")
        monos = sorted(g.body["mono"] for g in rows)
        spacing = all(b - a >= spec.SPACING_S for a, b in zip(monos, monos[1:]))
        window = all(sum(1 for m in monos if 0 <= x - m < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for x in monos)
        begin = epochs.get(epoch, {}).get("limiter_epoch_begin_mono")
        embargo = begin is not None and monos[0] >= begin + spec.EMBARGO_S
        if begin is None:
            missing.append(f"epoch {epoch[:8]}: limiter epoch start not recorded")
        if not spacing:
            violations.append(f"epoch {epoch[:8]}: two grants less than {spec.SPACING_S} s apart")
        if not window:
            violations.append(f"epoch {epoch[:8]}: more than {spec.WINDOW_MAX_STARTS} grants in a {spec.WINDOW_S} s window")
        if begin is not None and not embargo:
            violations.append(f"epoch {epoch[:8]}: a grant inside the {spec.EMBARGO_S} s embargo")
        per_epoch[epoch] = {"grants": len(rows), "spacing_ok": spacing, "window_ok": window, "embargo_ok": embargo}
    hops: dict[int, list] = {}
    abandoned = 0
    for g in grants:
        attempt = g.seq if g.body["attempt"] == "SELF" else g.body["attempt"]
        if attempt is None:
            abandoned += 1
            continue
        hops.setdefault(attempt, []).append((g.body["hop"], g.body["url"], g))
    for t in invoked:
        initial = [h for h in hops.get(t.seq, []) if h[2].body["attempt"] == "SELF"]
        if len(initial) != 1 or initial[0][0] != 0 or initial[0][2].body["mono"] != t.body["grant_mono"]:
            missing.append(f"attempt {t.seq}: no single initial grant in its TRANSPORT_INVOKED transaction")
    for attempt, rows in hops.items():
        numbers = sorted(h for h, _u, _g in rows)
        if len(numbers) != len(set(numbers)):
            violations.append(f"attempt {attempt}: a hop granted twice")
        if numbers != list(range(len(numbers))):
            missing.append(f"attempt {attempt}: hops not contiguous from 0")
    counted = set()
    for record in view.rows("RESPONSE"):
        attempt, count = record.body["attempt"], record.body.get("grants")
        counted.add(attempt)
        rows = sorted(hops.get(attempt, []), key=lambda h: h[0])
        if count is None or len(rows) != count:
            missing.append(f"attempt {attempt}: {len(rows)} journaled grants, the response records {count}")
        elif [u for _h, u, _g in rows] != record.body["redirect_chain"]:
            missing.append(f"attempt {attempt}: journaled URLs differ from the redirect chain")
    for outcome in view.rows("ATTEMPT_OUTCOME"):
        if "grants" in outcome.body:
            attempt = outcome.body["attempt"]
            counted.add(attempt)
            if len(hops.get(attempt, [])) != outcome.body["grants"]:
                missing.append(f"attempt {attempt}: {len(hops.get(attempt, []))} journaled grants, its outcome records "
                               f"{outcome.body['grants']}")
    uncounted = sorted(t.seq for t in invoked if t.seq not in counted)
    status = "FAIL" if violations else ("NOT_PROVEN" if missing else "PROVEN")
    return {"status": status, "grants": len(grants), "abandoned_before_invoke": abandoned,
            "continuations": sum(1 for g in grants if g.body["kind"] == "CONTINUATION"), "per_epoch": per_epoch,
            "missing": missing, "violations": violations,
            "attempts_without_count": uncounted,  # INTERRUPTED: complete by construction (journal before grant)
            "all_physical_starts": status}


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
    fix15 = fix15_journal(view, invoked)
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
        "fix15_journal_not_failed": fix15["status"] != "FAIL",
        "at_most_120_requests_per_live_item_upper_bound": max(requests_per_item.values(), default=0) <= 120,
        "one_processing_outcome_per_record": all(len(view.rows("PROCESSING_OUTCOME", key=str(r.seq))) == 1 for r in responses),
        "unique_episode_keys": len({e.key for e in view.rows("EPISODE_OPEN")}) == len(view.rows("EPISODE_OPEN")),
        "no_false_zero": false_zero == [],
    }
    not_proven = []
    if fix15["status"] == "NOT_PROVEN":
        not_proven.append("FIX15 over all physical starts: the grant journal is incomplete: " + "; ".join(fix15["missing"][:5]))
    return {"checks": checks, "ok": all(checks.values()), "not_proven": not_proven, "fix15": fix15,
            "attempts_without_outcome": without_outcome, "in_flight_overlaps": overlaps,
            "max_attempts_per_key": max(per_key.values(), default=0),
            "max_requests_per_live_item_upper_bound": max(requests_per_item.values(), default=0),
            "false_zero_cycles": false_zero}


def pilot_criteria(view, audit_: dict, rechecks: dict, *, reread_ok: bool) -> dict:
    """The short real pilot's criteria, from durable records only: a content identity that stays the same over
    re-reads whose raw bytes really differ, every observation kept distinct (its own record, raw and link),
    identical snapshots and replays, a complete and compliant FIX15 journal, and the durable state of the +300 s
    and +1 h rechecks."""
    links = view.rows("LINK")
    by_item: dict[str, dict] = {}
    for link in links:
        record = view.row_at("RESPONSE", link.body["record"])
        item = by_item.setdefault(record.body["sid"], {"raws": set(), "identities": set(), "domains": Counter(), "links": 0})
        item["raws"].add(link.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"])
        item["identities"].add(link.body["content_sha256"])
        item["domains"][link.body["canonicalization"]["domain"]] += 1
        item["links"] += 1
    re_read = {sid: v for sid, v in by_item.items() if len(v["raws"]) >= 2}
    stable = [sid for sid, v in re_read.items() if len(v["identities"]) == 1]
    distinct = len({l.body["record"] for l in links}) == len(links) and all(
        l.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"] == view.row_at("RESPONSE", l.body["record"]).body["raw_sha"]
        for l in links)
    lb = parse_iso(rechecks["now_lb"]) if rechecks.get("now_lb") else None

    def served(offset):
        rows = [o for item in rechecks["items"].values() for o in item["obligations"] if o["offset"] == offset]
        due = [o for o in rows if lb is not None and parse_iso(o["due_at"]) + timedelta(seconds=2 * spec.CLOCK_ERROR_BOUND_S + 120) <= lb]
        return {"obligations": len(rows), "due_long_enough": len(due),
                "statuses": dict(Counter(o["status"] for o in rows)),
                "served_when_due": all(o["status"] == "SATISFIED" for o in due) and bool(due)}
    criteria = {
        "identity_stable_over_different_rereads": {"items_reread_with_different_raws": len(re_read),
                                                   "items_with_one_identity": len(stable), "ok": bool(re_read) and len(stable) == len(re_read),
                                                   "per_item": {sid[:12]: {"raws": len(v["raws"]), "identities": len(v["identities"]),
                                                                           "domains": dict(v["domains"])} for sid, v in by_item.items()}},
        "observations_distinct": {"links": len(links), "ok": distinct},
        "snapshots_and_replay_identical": {"ok": reread_ok},
        "fix15_journal_complete_and_compliant": {"status": audit_["fix15"]["status"], "ok": audit_["fix15"]["status"] == "PROVEN"},
        "rechecks_300s": served(300), "rechecks_1h": served(3600),
    }
    criteria["rechecks_300s"]["ok"] = criteria["rechecks_300s"]["served_when_due"]
    criteria["rechecks_1h"]["ok"] = criteria["rechecks_1h"]["served_when_due"]
    criteria["verdict"] = "MET" if all(v["ok"] for k, v in criteria.items() if isinstance(v, dict)) else "NOT_MET"
    return criteria


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
        result["criteria"] = pilot_criteria(store.view(), result["audit"], result["rechecks"],
                                            reread_ok=bool(recorded) and all(x is True for x in reread)
                                            and all(x is True for x in replayed))
        result["ok"] = (bool(resolved) and all(x is True for x in reread) and all(x is True for x in replayed)
                        and health_ok and result["audit"]["ok"])
        return result
    finally:
        store.close()


def pilot_duration(stop: dict, progress_: dict, *, planned_start: datetime | None, planned_end: datetime | None,
                   closed_at: datetime, launch_boot_id: str | None = None, closure_boot_id: str | None = None) -> dict:
    """Whether the pilot ran for its planned duration - a separate verdict from integrity: the service
    must have been running when the closure stopped it, at or after the planned end, as one owner
    epoch, with durable activity up to the end, on the boot it was launched on and without a silence
    longer than CAPTURE_GAP_BOUND_S."""
    was_running = stop.get("was") in ("active", "running")
    last = parse_iso(progress_["last_wall"]) if progress_.get("last_wall") else None
    first = parse_iso(progress_["first_wall"]) if progress_.get("first_wall") else None
    detail = {"service_running_at_closure": was_running, "epochs": progress_.get("epochs"),
              "planned_start": iso(planned_start) if planned_start else None,
              "planned_end": iso(planned_end) if planned_end else None, "closed_at": iso(closed_at),
              "first_durable_activity": progress_.get("first_wall"), "last_durable_activity": progress_.get("last_wall"),
              "max_gap_between_transactions_s": progress_.get("max_gap_s"), "capture_gap_bound_s": CAPTURE_GAP_BOUND_S,
              "launch_boot_id": launch_boot_id, "closure_boot_id": closure_boot_id}
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
    if launch_boot_id is not None and launch_boot_id != closure_boot_id:
        reasons.append(f"the host rebooted after the launch (boot {launch_boot_id[:8]} then {str(closure_boot_id)[:8]})")
    if progress_.get("max_gap_s") is not None and progress_["max_gap_s"] > CAPTURE_GAP_BOUND_S:
        reasons.append(f"no durable transaction for {progress_['max_gap_s']} s (> {CAPTURE_GAP_BOUND_S} s): the capture was suspended")
    detail["verdict"] = "ACCOMPLISHED" if not reasons else "NOT_ACCOMPLISHED"
    detail["reasons"] = reasons
    return detail


def close(store_dir: Path, copy_dir: Path, report: Path, *, unit: str | None = None, pid: int | None = None,
          snapshots_log: Path | None = None, notify: bool = True, planned_start: datetime | None = None,
          planned_end: datetime | None = None, stopper=None, launch_boot_id: str | None = None,
          closure_boot_id: str | None = None) -> dict:
    """Clean stop, owner check, consistent copy and offline verification. Separate verdicts: integrity
    (copy, re-read, replay, health replay, invariants), pilot duration and the pilot criteria; the pilot
    is VALIDATED only when all three hold and nothing is NOT_PROVEN."""
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
        result["pilot_duration"] = pilot_duration(
            result["stop"], verification["progress"], planned_start=planned_start, planned_end=planned_end,
            closed_at=closed_at, launch_boot_id=launch_boot_id,
            closure_boot_id=closure_boot_id if closure_boot_id is not None else boot_id())
        result["criteria"] = verification["criteria"]
    result["integrity_ok"] = result["integrity"]["verdict"] == "VALID"
    result["duration_ok"] = result["pilot_duration"]["verdict"] == "ACCOMPLISHED"
    result["criteria_ok"] = result.get("criteria", {}).get("verdict") == "MET"
    result["ok"] = result["integrity_ok"] and result["duration_ok"]
    missing = [name for name, ok in (("integrity VALID", result["integrity_ok"]),
                                     ("duration ACCOMPLISHED", result["duration_ok"]),
                                     ("criteria MET", result["criteria_ok"])) if not ok]
    missing += [f"NOT_PROVEN: {x}" for x in result["integrity"].get("not_proven", [])]
    result["validation"] = {"verdict": "VALIDATED" if not missing else "NOT_VALIDATED", "missing": missing}
    result["finished"] = iso(_now())
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if notify:  # the monitored journal only hears about real closures
        _alert(f"FOMC pilot closure: {result['validation']['verdict']}; integrity {result['integrity']['verdict']}, "
               f"duration {result['pilot_duration']['verdict']}, criteria {result.get('criteria', {}).get('verdict')}, "
               f"not proven {len(result['integrity'].get('not_proven', []))}: report {report}",
               syslog.LOG_NOTICE if result["validation"]["verdict"] == "VALIDATED" else syslog.LOG_CRIT)
    return result


# ------------------------------------------------------------------ supervision -------------------
def supervise(store_dir: Path, service_log: Path, alerts: Path, snapshots_log: Path, *, unit: str,
              close_at: datetime, copy_dir: Path, report: Path, snapshot_every_s: float = 3600.0,
              poll_s: float = 15.0, log_offset: int = 0, planned_start: datetime | None = None,
              closes: bool = True, launch_boot_id: str | None = None) -> dict:
    """Route FOMC-ALERT lines to syslog (journald, priority crit) and an alert file, raise an alert if
    the service stops before its closure, take a snapshot read every `snapshot_every_s`, then close at
    `close_at`. With `closes=False` a persistent timer owns the closure (it survives a stop of the host):
    the supervisor takes its last read 120 s before it and exits. `log_offset` lets a replacement
    supervisor resume reading the service log where it is now."""
    offset, down_reported, next_snapshot = log_offset, False, time.monotonic()
    until = close_at if closes else close_at - timedelta(seconds=120)
    while _now() < until:
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
    if not closes:
        return {"ok": True, "closed": False}
    return close(store_dir, copy_dir, report, unit=unit, snapshots_log=snapshots_log,
                 planned_start=planned_start, planned_end=close_at, launch_boot_id=launch_boot_id)


# ------------------------------------------------------------------ the successor pilot -----------
def _systemctl(*args: str) -> str:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True).stdout.strip()


def closure_units(unit: str, code: Path, run: Path, *, started: datetime, close_at: datetime, launch_boot: str) -> dict:
    """The closure as a persistent user timer and its one-shot service: it fires at `close_at`, or at the
    next boot when the host was down then (Persistent=true), from the frozen `code`."""
    service = "\n".join([
        "[Unit]", f"Description=FOMC {unit} closure (frozen code {code.name}; stop, consistent copy, offline verification)", "",
        "[Service]", "Type=oneshot", f"WorkingDirectory={code}",
        f"ExecStart=/usr/bin/python3 -m scripts.trading_lab.fomc.pilot close --store {run / 'store'} --copy {run / 'closure-copy'} "
        f"--report {run / 'closure-report.json'} --snapshots {run / 'snapshots.jsonl'} --unit {unit} "
        f"--planned-start {iso(started)} --planned-end {iso(close_at)} --launch-boot-id {launch_boot}",
        f"StandardOutput=append:{run / 'closure.log'}", f"StandardError=append:{run / 'closure.log'}", ""])
    timer = "\n".join([
        "[Unit]", f"Description=FOMC {unit} closure at its planned end (runs at the next boot if missed)", "",
        "[Timer]", f"OnCalendar={close_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", "Persistent=true",
        "AccuracySec=1s", f"Unit={unit}-closure.service", "", "[Install]", "WantedBy=timers.target", ""])
    return {f"{unit}-closure.service": service, f"{unit}-closure.timer": timer}


def successor(previous_run: Path, previous_unit: str, code: Path, base: Path, *, wait_until: datetime,
              minutes: float = 90.0, unit: str = "fomc-pilot-rev25", manifest: Path | None = None,
              authorization: dict | None = None, environment: dict | None = None, unit_dir: Path | None = None,
              snapshot_every_s: float = 900.0) -> dict:
    """Start the successor pilot only once the previous one allows it: its closure report shows integrity
    VALID and duration ACCOMPLISHED, its service is stopped and its store's ownership is free, and no other
    FOMC service runs - never a second emitter. An explicit operator `authorization` may waive the previous
    pilot's duration (`waives: ["previous_duration"]`) for one launch only; nothing else can be waived. Then:
    a new store, the service from the frozen `code`, a monitoring supervisor, and the closure after `minutes`
    as a persistent timer (consistent copy and offline verification, also after a stop of the host)."""
    decision = {"checked_at": None, "previous_run": str(previous_run), "launched": False, "reasons": [],
                "authorization": authorization, "waived": [], "environment": environment}
    report_path = previous_run / "closure-report.json"
    while not report_path.exists() and _now() < wait_until:
        time.sleep(30)
    decision["checked_at"] = iso(_now())
    reasons = decision["reasons"]
    if not report_path.exists():
        reasons.append(f"no closure report of the previous pilot by {iso(wait_until)}")
    else:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        decision["previous"] = {"integrity": previous.get("integrity", {}).get("verdict"),
                                "duration": previous.get("pilot_duration", {}).get("verdict"),
                                "not_proven": previous.get("integrity", {}).get("not_proven")}
        if decision["previous"]["integrity"] != "VALID":
            reasons.append(f"previous integrity {decision['previous']['integrity']}")
        if decision["previous"]["duration"] != "ACCOMPLISHED":
            if authorization and "previous_duration" in authorization.get("waives", []):
                decision["waived"].append(f"previous duration {decision['previous']['duration']} (operator authorization "
                                          f"of {authorization.get('granted_at')}, scope: {authorization.get('scope')})")
            else:
                reasons.append(f"previous duration {decision['previous']['duration']}")
    if authorization:
        unknown = set(authorization.get("waives", [])) - {"previous_duration"}
        if unknown:
            reasons.append(f"the authorization waives what cannot be waived: {sorted(unknown)}")
        if authorization.get("unit") != unit:
            reasons.append(f"the authorization is for {authorization.get('unit')}, not {unit}")
        earlier = base / "successor-decision.json"
        if earlier.exists() and json.loads(earlier.read_text(encoding="utf-8")).get("decision") == "LAUNCHED":
            reasons.append("an earlier launch already used a successor decision; this authorization covers one trial")
    state_ = _systemctl("is-active", previous_unit)
    if state_ in ("active", "activating", "deactivating", "reloading"):
        reasons.append(f"previous service {previous_unit} is {state_}")
    if (previous_run / "store").exists() and not owner_free(previous_run / "store"):
        reasons.append("the previous store still has an owner")
    others = [line.split()[0] for line in _systemctl("list-units", "fomc-pilot*", "--state=active", "--no-legend",
                                                      "--plain").splitlines() if line.strip()]
    emitters = [u for u in others if "supervisor" not in u and u.endswith(".service")]
    if emitters:
        reasons.append(f"another FOMC service is active: {emitters}")
    if subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True,
                      text=True).stdout.strip() != "yes":
        reasons.append("NTP not synchronized")
    try:
        decision["spec"] = [spec.SPEC_REVISION, spec.verify_spec_binding()]
    except RuntimeError as exc:
        reasons.append(str(exc))
    run = None
    if not reasons:
        run = base / f"run-rev{spec.SPEC_REVISION}-{_now().strftime('%Y%m%dT%H%M%SZ')}"
        run.mkdir(parents=True)
        args = ["/usr/bin/python3", "-m", "scripts.trading_lab.fomc.service", "--store", str(run / "store"), "--tick", "1"]
        if manifest is not None:
            shutil.copy(manifest, run / manifest.name)
            args += ["--manifest", str(run / manifest.name)]
        check = subprocess.run(["/usr/bin/python3", "-m", "scripts.trading_lab.fomc.service", "--store",
                                str(run / "store"), "--check"], capture_output=True, text=True, cwd=code)
        if check.returncode != 0:
            reasons.append(f"preflight refused: {check.stderr.strip()}")
    if reasons:
        decision["decision"] = "NOT_LAUNCHED"
        (base / "successor-decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
        _alert(f"FOMC successor NOT launched: {'; '.join(reasons)}")
        return decision
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", f"--working-directory={code}",
                    "--property=TimeoutStopSec=180", "--property=KillMode=mixed",
                    f"--property=StandardOutput=append:{run / 'service.log'}",
                    f"--property=StandardError=append:{run / 'service.log'}", "--", *args], check=True)
    started = _now()
    launch_boot = boot_id()
    close_at = started + timedelta(minutes=minutes)
    unit_dir = unit_dir or Path.home() / ".config" / "systemd" / "user"
    for name, text in closure_units(unit, code, run, started=started, close_at=close_at, launch_boot=launch_boot).items():
        (unit_dir / name).write_text(text, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{unit}-closure.timer"], check=True)
    time.sleep(5)
    subprocess.run(["systemd-run", "--user", f"--unit={unit}-supervisor", f"--working-directory={code}", "--",
                    "/usr/bin/python3", "-m", "scripts.trading_lab.fomc.pilot", "supervise", "--store", str(run / "store"),
                    "--unit", unit, "--service-log", str(run / "service.log"), "--alerts", str(run / "alerts.log"),
                    "--snapshots", str(run / "snapshots.jsonl"), "--copy", str(run / "closure-copy"),
                    "--report", str(run / "closure-report.json"), "--close-at", iso(close_at),
                    "--planned-start", iso(started), "--no-close", "--snapshot-every", str(snapshot_every_s),
                    "--launch-boot-id", launch_boot], check=True)
    decision.update(decision="LAUNCHED", launched=True, run=str(run), unit=unit, pid=_systemctl("show", unit, "-p", "MainPID", "--value"),
                    started=iso(started), close_at=iso(close_at), code=str(code), launch_boot_id=launch_boot,
                    closure_timer=f"{unit}-closure.timer (persistent, {unit_dir})")
    for path in (base / "successor-decision.json", run / "decision.json"):
        path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    _alert(f"FOMC successor launched: unit {unit}, run {run}, closure {iso(close_at)}", syslog.LOG_NOTICE)
    return decision


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
    p.add_argument("--no-close", action="store_true", help="monitor only: a persistent timer owns the closure")
    p.add_argument("--snapshot-every", type=float, default=3600.0)
    p.add_argument("--launch-boot-id")
    p = sub.add_parser("close")
    for name in ("--store", "--copy", "--report"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--snapshots", type=Path)
    p.add_argument("--unit")
    p.add_argument("--pid", type=int)
    p.add_argument("--planned-start")
    p.add_argument("--planned-end")
    p.add_argument("--launch-boot-id")
    p = sub.add_parser("successor")
    for name in ("--previous-run", "--code", "--base"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--previous-unit", required=True)
    p.add_argument("--wait-until", required=True)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--unit", default="fomc-pilot-rev25")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--authorization", type=Path, help="JSON operator authorization (one trial)")
    p.add_argument("--environment", type=Path, help="JSON environment check recorded in the decision")
    p.add_argument("--snapshot-every", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.cmd == "successor":
        decision = successor(args.previous_run, args.previous_unit, args.code, args.base,
                             wait_until=parse_iso(args.wait_until), minutes=args.minutes, unit=args.unit,
                             manifest=args.manifest, snapshot_every_s=args.snapshot_every,
                             authorization=json.loads(args.authorization.read_text()) if args.authorization else None,
                             environment=json.loads(args.environment.read_text()) if args.environment else None)
        print(json.dumps(decision, sort_keys=True))
        return 0 if decision["launched"] else 1
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
                           planned_start=parse_iso(args.planned_start) if args.planned_start else None,
                           closes=not args.no_close, snapshot_every_s=args.snapshot_every,
                           launch_boot_id=args.launch_boot_id)
        return 0 if result["ok"] else 1
    result = close(args.store, args.copy, args.report, unit=args.unit, pid=args.pid, snapshots_log=args.snapshots,
                   planned_start=parse_iso(args.planned_start) if args.planned_start else None,
                   planned_end=parse_iso(args.planned_end) if args.planned_end else None, launch_boot_id=args.launch_boot_id)
    print(json.dumps({k: result[k] for k in ("integrity_ok", "duration_ok", "criteria_ok", "validation", "stop", "owner_free")},
                     sort_keys=True))
    return 0 if result["integrity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
