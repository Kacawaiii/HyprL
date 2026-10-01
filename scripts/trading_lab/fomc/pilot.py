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


def audit(store) -> dict:
    """Capture invariants on a closed store (read-only)."""
    view = store.view()
    invoked = view.rows("TRANSPORT_INVOKED")
    without_outcome = [t.seq for t in invoked if not view.rows("ATTEMPT_OUTCOME", key=str(t.seq))]
    per_key = Counter(t.key for t in invoked if t.key != ledger.FEED_KEY)
    one_in_flight = all(view.rows("ATTEMPT_OUTCOME", key=str(a.seq))[0].seq < b.seq
                        for key in {t.key for t in invoked}
                        for a, b in zip([t for t in invoked if t.key == key], [t for t in invoked if t.key == key][1:])
                        if view.rows("ATTEMPT_OUTCOME", key=str(a.seq)))
    spacing_ok, window_ok = True, True
    for epoch in {t.body["epoch"] for t in invoked}:
        grants = sorted(t.body["grant_mono"] for t in invoked if t.body["epoch"] == epoch)
        spacing_ok &= all(b - a >= spec.SPACING_S for a, b in zip(grants, grants[1:]))
        window_ok &= all(sum(1 for g in grants if 0 <= x - g < spec.WINDOW_S) <= spec.WINDOW_MAX_STARTS for x in grants)
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
        "one_fetch_per_item_and_one_poll_in_flight": one_in_flight,
        "fix15_spacing_per_epoch": spacing_ok, "fix15_window_per_epoch": window_ok,
        "at_most_120_requests_per_live_item": max(requests_per_item.values(), default=0) <= 120,
        "one_processing_outcome_per_record": all(len(view.rows("PROCESSING_OUTCOME", key=str(r.seq))) == 1 for r in responses),
        "unique_episode_keys": len({e.key for e in view.rows("EPISODE_OPEN")}) == len(view.rows("EPISODE_OPEN")),
        "no_false_zero": false_zero == [],
    }
    return {"checks": checks, "ok": all(checks.values()), "attempts_without_outcome": without_outcome,
            "max_attempts_per_key": max(per_key.values(), default=0),
            "max_requests_per_live_item_upper_bound": max(requests_per_item.values(), default=0),
            "false_zero_cycles": false_zero}


def verify_copy(copy_dir: Path, snapshots_log: Path | None) -> dict:
    """Offline verification of the copy: recorded resolved snapshots re-read and replayed at their
    (T, H), source health replay, capture invariants. No network."""
    store = _open(copy_dir)
    try:
        H = store.horizon()
        recorded = [json.loads(line) for line in open(snapshots_log, encoding="utf-8")] if snapshots_log else []
        resolved = [r for r in recorded if r["resolved"]]
        reread, replayed = [], []
        for r in resolved:
            T = parse_iso(r["T"])
            again = snapshot.events_as_of(store, T, r["H"])
            reread.append(again["identity"] == r["identity"])
            try:
                replayed.append(snapshot.replay(store, T, r["H"])["identity"] == r["identity"])
            except snapshot.SnapshotFailed as exc:
                replayed.append(f"failed: {exc}")
        health_ok, health_error = True, None
        try:
            snapshot.verify_health(store, H)
        except snapshot.ReplayFailed as exc:
            health_ok, health_error = False, str(exc)
        result = {"horizon": H, "recorded_reads": len(recorded), "resolved_reads": len(resolved),
                  "reread_identical": sum(1 for x in reread if x is True), "replay_identical": sum(1 for x in replayed if x is True),
                  "replay_failures": [x for x in replayed if x is not True], "health_replay": health_ok,
                  "health_error": health_error, "audit": audit(store), "progress": progress(store)}
        result["ok"] = (bool(resolved) and all(x is True for x in reread) and all(x is True for x in replayed)
                        and health_ok and result["audit"]["ok"])
        return result
    finally:
        store.close()


def close(store_dir: Path, copy_dir: Path, report: Path, *, unit: str | None = None, pid: int | None = None,
          snapshots_log: Path | None = None, notify: bool = True) -> dict:
    result = {"started": iso(_now()), "store": str(store_dir), "copy": str(copy_dir)}
    result["stop"] = stop_service(unit, pid)
    result["owner_free"] = owner_free(store_dir)
    if not result["owner_free"]:
        result["ok"] = False
        result["error"] = "the store still has an owner: no copy taken"
    else:
        result["copy_check"] = consistent_copy(store_dir, copy_dir)
        result["verification"] = verify_copy(copy_dir, snapshots_log)
        result["ok"] = result["copy_check"]["equal"] and result["verification"]["ok"]
    result["finished"] = iso(_now())
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if notify:  # the monitored journal only hears about real closures
        _alert(f"FOMC pilot closure {'VERIFIED' if result['ok'] else 'FAILED'}: report {report}",
               syslog.LOG_NOTICE if result["ok"] else syslog.LOG_CRIT)
    return result


# ------------------------------------------------------------------ supervision -------------------
def supervise(store_dir: Path, service_log: Path, alerts: Path, snapshots_log: Path, *, unit: str,
              close_at: datetime, copy_dir: Path, report: Path, snapshot_every_s: float = 3600.0,
              poll_s: float = 15.0) -> dict:
    """Route FOMC-ALERT lines to syslog (journald, priority crit) and an alert file, raise an alert if
    the service stops before its closure, take a snapshot read every hour, then close at `close_at`."""
    offset, down_reported, next_snapshot = 0, False, time.monotonic()
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
    take_snapshot(store_dir, snapshots_log)  # a last read before the stop
    return close(store_dir, copy_dir, report, unit=unit, snapshots_log=snapshots_log)


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
    p = sub.add_parser("close")
    for name in ("--store", "--copy", "--report"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--snapshots", type=Path)
    p.add_argument("--unit")
    p.add_argument("--pid", type=int)
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
        result = supervise(args.store, args.service_log, args.alerts, args.snapshots, unit=args.unit,
                           close_at=parse_iso(args.close_at), copy_dir=args.copy, report=args.report)
        return 0 if result["ok"] else 1
    result = close(args.store, args.copy, args.report, unit=args.unit, pid=args.pid, snapshots_log=args.snapshots)
    print(json.dumps({k: result[k] for k in ("ok", "stop", "owner_free")}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
