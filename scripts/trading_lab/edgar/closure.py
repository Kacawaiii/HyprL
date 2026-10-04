"""Stop and close a bounded EDGAR run, then verify a consistent copy entirely offline.

    python -m scripts.trading_lab.edgar.closure close --store DIR --copy NEW_DIR --report FILE \
        [--unit EDGAR_USER_UNIT | --pid PID] [--authorization FILE] [--snapshots JSONL]
    python -m scripts.trading_lab.edgar.closure units --unit EDGAR_USER_UNIT --code DIR --run DIR \
        --authorization FILE --close-at ISO_INSTANT --out NEW_DIR

Unit generation writes templates only. It neither installs units nor starts a capture. Closure reports
separate store integrity from completion of the authorized budget/window. An interrupted run can have
valid evidence without being accomplished. Authorizations, copies and reports belong outside Git.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import time

from scripts.trading_lab.edgar import snapshot, spec
from scripts.trading_lab.edgar.collector import WATCHLIST_KEY
from scripts.trading_lab.edgar.listing import cik10
from scripts.trading_lab.edgar.qualify import qualify
from scripts.trading_lab.edgar.service import MAX_AUTHORIZED_REQUESTS, authorization_scope
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso, parse_iso


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _open(root: Path) -> EdgarStore:
    return EdgarStore(root, wall_clock=None, read_only=True)


@contextmanager
def _owner_guard(store_dir: Path):
    """Hold the existing owner lock through the copy, without creating or changing source files."""
    store_dir = Path(store_dir)
    try:
        handle = (store_dir / "owner.lock").open("rb")
    except FileNotFoundError:
        # Published closure copies have only the database and raws. A read-only directory without
        # an owner lock cannot admit a collector (it must create that lock). Never create one here.
        if not (store_dir / EdgarStore.DB_NAME).is_file() or os.access(store_dir, os.W_OK):
            raise
        yield "read-only snapshot directory; no owner lock or possible collector admission"
        return
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield "existing owner lock held exclusively through the copy"
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def owner_free(store_dir: Path) -> bool:
    try:
        with _owner_guard(store_dir):
            return True
    except (OSError, BlockingIOError):
        return False


def stop_service(unit: str | None = None, pid: int | None = None, *, timeout_s: float = 180.0) -> dict:
    """Stop only the explicitly named user capture unit/process and confirm its exit."""
    if unit and pid:
        raise ValueError("give a user unit or a pid, not both")
    if unit:
        _unit_name(unit)
        command = ["systemctl", "--user"]
        before = subprocess.run(command + ["is-active", unit], capture_output=True, text=True, timeout=timeout_s)
        was = before.stdout.strip()
        if was in ("active", "activating", "deactivating"):
            stopped = subprocess.run(command + ["stop", unit], capture_output=True, text=True, timeout=timeout_s)
            if stopped.returncode:
                return {"was": was, "stopped": False, "error": "user unit stop failed"}
        after = subprocess.run(command + ["is-active", unit], capture_output=True, text=True, timeout=timeout_s)
        left = after.stdout.strip()
        return {"was": was, "left": left, "stopped": left in ("inactive", "failed")}
    if pid is not None:
        if pid <= 1 or pid == os.getpid():
            raise ValueError("invalid capture pid")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return {"was": "not running", "stopped": True}
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return {"was": "running", "stopped": True}
            time.sleep(0.1)
        return {"was": "running", "stopped": False, "error": "capture did not exit"}
    return {"was": "offline closure; no process target supplied", "stopped": True}


def consistent_copy(store_dir: Path, copy_dir: Path) -> dict:
    """Backup a read-only SQLite connection (including WAL), then copy the immutable raw bodies.

    The caller must hold the owner lock: transaction consistency alone cannot freeze the raw set.
    The target must be new and outside the source. Never copy a database file without its WAL.
    """
    source_dir, copy_dir = Path(store_dir).resolve(), Path(copy_dir).resolve()
    if source_dir == copy_dir or source_dir in copy_dir.parents or copy_dir in source_dir.parents:
        raise ValueError("source and copy directories must be separate")
    source = _open(source_dir)
    try:
        copy_dir.mkdir(parents=True, exist_ok=False)
        target = sqlite3.connect(copy_dir / EdgarStore.DB_NAME)
        try:
            with source.locked("backup"):
                source._conn.backup(target)
                counts = {t: target.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("txn", "rec")}
                original = {t: source._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("txn", "rec")}
            integrity = [r[0] for r in target.execute("PRAGMA integrity_check")]
            target.execute("PRAGMA journal_mode=DELETE")  # stable, standalone read-only copy
        finally:
            target.close()
        shutil.copytree(source_dir / "raw", copy_dir / "raw")
    finally:
        source.close()
    return {"integrity_check": integrity, "rows": counts, "source_rows": original,
            "raw_files": sum(p.is_file() for p in (copy_dir / "raw").rglob("*")),
            "equal": counts == original and integrity == ["ok"]}


def _tree_digest(root: Path) -> str:
    """Digest names and contents, including any side files created by an accidental writable read."""
    files = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append([path.relative_to(root).as_posix(), digest.hexdigest()])
    return sha256_canonical(files)


def progress(store) -> dict:
    view = store.view()
    txns = view.txns()
    manifests = view.rows("MANIFEST", key=WATCHLIST_KEY)
    invoked = view.rows("TRANSPORT_INVOKED")
    walls = {seq: wall for seq, _kind, wall in txns}
    return {"horizon": view.horizon(), "epochs": len(view.rows("EPOCH")),
            "first_wall": txns[0][2] if txns else None, "last_wall": txns[-1][2] if txns else None,
            "requests": len(view.rows("TRANSPORT_INVOKED")), "responses": len(view.rows("RESPONSE")),
            "revisions": len(view.rows("FILING_REVISION")), "observations": len(view.rows("FILING_OBSERVATION")),
            "absences": len(view.rows("FILING_ABSENCE")), "health_rows": len(view.rows("SOURCE_HEALTH")),
            "storage_incidents": len(view.rows("STORAGE_INCIDENT")),
            "watchlist": manifests[0].body["ciks"] if manifests else [],
            "request_ciks": sorted({r.body["cik"] for r in invoked}),
            "request_walls": [walls[r.seq] for r in invoked],
            "open_attempts": sum(not view.rows("ATTEMPT_OUTCOME", key=str(r.seq))
                                 for r in view.rows("TRANSPORT_INVOKED")),
            "runs_started": [dict(r.body, wall_at_commit=wall) for r in view.rows("RUN_STARTED")
                             for seq, _kind, wall in txns if seq == r.seq],
            "runs_ended": [dict(r.body, wall_at_commit=wall) for r in view.rows("RUN_ENDED")
                           for seq, _kind, wall in txns if seq == r.seq]}


def verify_copy(copy_dir: Path, snapshots_log: Path | None = None, *, now: datetime | None = None) -> dict:
    """Verify every raw and replay the complete horizon, even if there are no resolved reads.

    Generate reads at each server attestation plus the unchanged causal bound, and at now. Optional
    recorded reads retain their exact (T, H, identity). Compare them after reopening, then replay them,
    including source health. Failure of any mandatory check is INVALID; nothing writes to the copy.
    """
    copy_dir = Path(copy_dir)
    before = _tree_digest(copy_dir)
    result = {"ok": False, "raws_verified": 0, "raw_failures": [], "read_failures": [],
              "reopen_identical": 0, "replay_identical": 0, "health_replay": False}
    store = None
    try:
        store = _open(copy_dir)
        result["integrity_check"] = [r[0] for r in store._conn.execute("PRAGMA integrity_check")]
        H = store.horizon()
        result["progress"] = progress(store)
        responses = store.rows("RESPONSE")
        result["responses"] = len(responses)
        for row in responses:
            try:
                body = store.read_raw(row.body["raw_sha"])
                if len(body) != row.body["byte_length"]:
                    raise ValueError("recorded byte length differs")
                result["raws_verified"] += 1
            except Exception as exc:
                result["raw_failures"].append({"record": row.seq, "error": type(exc).__name__})
        reads = [{"T": iso(parse_iso(r.body["observed_at"]) + spec.CLOCK_ERROR_BOUND), "H": H}
                 for r in responses if r.body["verdict"] == "CLOCK_VERIFIED" and not r.body["late_evidence"]]
        reads.append({"T": iso(now or _now()), "H": H})
        recorded = [json.loads(line) for line in Path(snapshots_log).read_text(encoding="utf-8").splitlines()
                    if line.strip()] if snapshots_log is not None else []
        reads.extend(recorded)
        result["recorded_reads"] = len(recorded)
        result["reads"] = []
        originals = []
        for read in reads:
            try:
                if not isinstance(read["H"], int) or isinstance(read["H"], bool) or not 0 <= read["H"] <= H:
                    raise ValueError("read horizon outside the copied store")
                snap = snapshot.filings_as_of(store, parse_iso(read["T"]), read["H"])
                if "identity" in read and read["identity"] != snap["identity"]:
                    raise ValueError("recorded identity differs")
                originals.append((read, snap))
                result["reads"].append({"T": snap["T"], "H": snap["H"], "identity": snap["identity"],
                                        "read_state": snap["read_state"], "filings": len(snap.get("filings", []))})
            except Exception as exc:
                result["read_failures"].append({"check": "causal read", "error": type(exc).__name__})
        try:
            result["qualification"] = qualify(store)
        except Exception as exc:
            # Rejected source listings are valid captured evidence. A qualification failure is
            # explicit, but the independent raw/replay checks determine storage integrity.
            result["qualification"] = {"error": type(exc).__name__, "verdict": "NOT_QUALIFIED"}
        store.close()
        store = _open(copy_dir)
        for read, original in originals:
            for name, reader in (("reopen", snapshot.filings_as_of), ("replay", snapshot.replay)):
                try:
                    if reader(store, parse_iso(read["T"]), read["H"]) != original:
                        raise ValueError("snapshot differs")
                    result[f"{name}_identical"] += 1
                except Exception as exc:
                    result["read_failures"].append({"check": name, "error": type(exc).__name__})
        try:
            snapshot.replay(store, now or _now(), H)  # covers all rows, including unresolved tail
            result["health_replay"] = True
        except Exception as exc:
            result["read_failures"].append({"check": "full replay including health", "error": type(exc).__name__})
        result["resolved_reads"] = sum(r["read_state"] == "EDGAR_RESOLVED" for r in result["reads"])
        result["unresolved_reads"] = len(result["reads"]) - result["resolved_reads"]
        result["ok"] = (result["integrity_check"] == ["ok"] and not result["raw_failures"]
                        and not result["read_failures"] and result["health_replay"])
    except Exception as exc:
        result["error"] = type(exc).__name__
    finally:
        if store is not None:
            store.close()
        result["copy_sha256_before"] = before
        result["copy_sha256_after"] = _tree_digest(copy_dir)
        result["copy_unchanged"] = result["copy_sha256_before"] == result["copy_sha256_after"]
        result["ok"] = result["ok"] and result["copy_unchanged"]
    return result


def _scope(authorization: Path | dict) -> dict:
    auth = json.loads(Path(authorization).read_text(encoding="utf-8")) if not isinstance(authorization, dict) else authorization
    scope = authorization_scope(auth)
    scope["ciks"] = [cik10(c) for c in scope["ciks"]]
    budget = scope["max_requests"]
    if (scope["authorizes"] != spec.PROVIDER_ID or scope["spec_hash"] != spec.SPEC_HASH
            or not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= MAX_AUTHORIZED_REQUESTS
            or not 1 <= len(scope["ciks"]) <= spec.WATCHLIST_MAX or len(set(scope["ciks"])) != len(scope["ciks"])):
        raise ValueError("invalid EDGAR authorization scope")
    parse_iso(scope["not_after"])  # expired authorizations must remain verifiable offline
    return scope


def run_completion(progress_: dict, *, authorization: Path | dict | None, closed_at: datetime,
                   stop: dict) -> dict:
    """Completion needs durable evidence of the authorized bound, never just a late closure time.

    Legacy runs have no RUN_ENDED: their committed request count can still prove a spent budget.
    Expiry needs a durable expiry termination, or closure of a still-running owner at its expiry.
    """
    reasons = []
    started, ended = progress_.get("runs_started", []), progress_.get("runs_ended", [])
    try:
        scope = _scope(authorization if authorization is not None else started[-1]["authorization"])
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return {"verdict": "NOT_ACCOMPLISHED", "reasons": ["no valid authorization bounds supplied or recorded"]}
    if progress_.get("epochs") != 1:
        reasons.append("the authorized run does not have exactly one owner epoch")
    if any(_scope(r["authorization"]) != scope for r in started):
        reasons.append("supplied bounds differ from the durable authorization")
    if progress_.get("watchlist") != scope["ciks"] or not set(progress_.get("request_ciks", [])) <= set(scope["ciks"]):
        reasons.append("the watchlist or request CIKs differ from the authorization")
    requests = progress_.get("requests", 0)
    if requests > scope["max_requests"]:
        reasons.append("request budget exceeded")
    expiry = parse_iso(scope["not_after"])
    if any(parse_iso(wall) >= expiry for wall in progress_.get("request_walls", [])):
        reasons.append("an invocation was committed at or after the authorization expiry")
    ending = ended[-1] if ended else None
    stopped = ending is not None and ending["reason"] not in ("request budget spent", "authorization expired")
    if stopped:
        reasons.append("the runner ended before an authorized bound (stop, throttle or interruption)")
    budget_reached = requests == scope["max_requests"] and progress_.get("open_attempts", 0) == 0
    expiry_reached = closed_at >= expiry and (
        (ending is not None and ending["reason"] == "authorization expired"
         and parse_iso(ending["wall_at_commit"]) >= expiry)
        or (ending is None and stop.get("was") in ("active", "running")
            and stop.get("checked_at") is not None
            and parse_iso(stop["checked_at"]) >= expiry))
    if not budget_reached and not expiry_reached:
        reasons.append("neither the request budget nor a witnessed authorization expiry was reached")
    return {"verdict": "NOT_ACCOMPLISHED" if reasons else "ACCOMPLISHED", "reasons": reasons,
            "authorization": scope, "requests": requests, "budget_reached": budget_reached,
            "expiry_reached": expiry_reached, "closed_at": iso(closed_at)}


def close(store_dir: Path, copy_dir: Path, report: Path, *, unit: str | None = None, pid: int | None = None,
          authorization: Path | dict | None = None, snapshots_log: Path | None = None,
          stopper=None, now: datetime | None = None) -> dict:
    """Stop, hold the owner lock, copy, verify; always write separate integrity and run verdicts."""
    result = {"spec_revision": spec.SPEC_REVISION, "spec_hash": spec.SPEC_HASH, "started": iso(now or _now()),
              "owner_free": False, "integrity": {"verdict": "INVALID"},
              "run": {"verdict": "NOT_ACCOMPLISHED", "reasons": ["closure did not verify the run"]}}
    try:
        spec.verify_spec_binding()
        # Output paths must never write into the original store, even on a refused closure.
        source = Path(store_dir).resolve()
        if source == Path(report).resolve() or source in Path(report).resolve().parents:
            raise ValueError("report must be outside the source store")
        checked_at = now or _now()
        result["stop"] = dict(stopper() if stopper is not None else stop_service(unit, pid), checked_at=iso(checked_at))
        if not result["stop"].get("stopped", False) or result["stop"].get("error"):
            raise RuntimeError("capture stop was not confirmed")
        closed_at = now or _now()
        with _owner_guard(store_dir) as owner_check:
            result["owner_free"] = True
            result["owner_check"] = owner_check
            result["copy_check"] = consistent_copy(store_dir, copy_dir)
        verification = verify_copy(copy_dir, snapshots_log, now=closed_at)
        result["verification"] = verification
        valid = result["copy_check"]["equal"] and verification["ok"]
        result["integrity"] = {"verdict": "VALID" if valid else "INVALID"}
        result["qualification"] = verification.get("qualification", {"verdict": "NOT_QUALIFIED"})
        result["run"] = run_completion(verification.get("progress", {}), authorization=authorization,
                                       closed_at=closed_at, stop=result["stop"])
    except Exception as exc:
        result["integrity"]["error"] = type(exc).__name__
    result["integrity_ok"] = result["integrity"]["verdict"] == "VALID"
    result["run_ok"] = result["run"]["verdict"] == "ACCOMPLISHED"
    result["ok"] = result["integrity_ok"] and result["run_ok"]
    result["finished"] = iso(now or _now())
    # Do not fall through to writing the forbidden source path after an earlier validation error.
    if Path(store_dir).resolve() in Path(report).resolve().parents or Path(report).resolve() == Path(store_dir).resolve():
        raise ValueError("report must be outside the source store")
    Path(report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _unit_name(unit: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", unit):
        raise ValueError("invalid user unit name")
    return unit


def _unit_arg(value) -> str:
    text = str(value)
    if any(c in text for c in ("\n", "\r", "\0")):
        raise ValueError("unit arguments must be single lines")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def closure_units(unit: str, code: Path, run: Path, *, close_at: datetime, authorization: Path,
                  python: Path = Path("/usr/bin/python3")) -> dict[str, str]:
    """Return persistent user timer/service templates; never install or enable them."""
    _unit_name(unit)
    if close_at.tzinfo is None:
        raise ValueError("closure needs an offset-aware instant")
    fire = close_at.astimezone(timezone.utc)
    if fire.microsecond:
        fire = fire.replace(microsecond=0) + timedelta(seconds=1)
    args = [python, "-m", "scripts.trading_lab.edgar.closure", "close", "--store", Path(run) / "store",
            "--copy", Path(run) / "closure-copy", "--report", Path(run) / "closure-report.json",
            "--authorization", authorization, "--unit", unit]
    service = "\n".join(["[Unit]", f"Description=EDGAR {unit} offline closure", "", "[Service]", "Type=oneshot",
                         "WorkingDirectory=" + _unit_arg(Path(code).resolve()).replace("$$", "$"),
                         "ExecStart=" + " ".join(_unit_arg(a) for a in args), ""])
    timer = "\n".join(["[Unit]", f"Description=EDGAR {unit} closure at its authorization expiry", "", "[Timer]",
                       f"OnCalendar={fire.strftime('%Y-%m-%d %H:%M:%S')} UTC", "Persistent=true", "AccuracySec=1s",
                       f"Unit={unit}-closure.service", "", "[Install]", "WantedBy=timers.target", ""])
    return {f"{unit}-closure.service": service, f"{unit}-closure.timer": timer}


def write_closure_units(out: Path, **kwargs) -> list[Path]:
    units = closure_units(**kwargs)
    Path(out).mkdir(parents=True, exist_ok=False)
    paths = []
    for name, content in units.items():
        path = Path(out) / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    closing = sub.add_parser("close")
    for name in ("store", "copy", "report"):
        closing.add_argument("--" + name, type=Path, required=True)
    target = closing.add_mutually_exclusive_group()
    target.add_argument("--unit")
    target.add_argument("--pid", type=int)
    closing.add_argument("--authorization", type=Path)
    closing.add_argument("--snapshots", type=Path)
    units = sub.add_parser("units")
    units.add_argument("--unit", required=True)
    for name in ("code", "run", "authorization", "out"):
        units.add_argument("--" + name, type=Path, required=True)
    units.add_argument("--close-at", type=parse_iso, required=True)
    args = parser.parse_args(argv)
    if args.command == "units":
        write_closure_units(args.out, unit=args.unit, code=args.code, run=args.run,
                            authorization=args.authorization, close_at=args.close_at)
        return 0
    result = close(args.store, args.copy, args.report, unit=args.unit, pid=args.pid,
                   authorization=args.authorization, snapshots_log=args.snapshots)
    print(json.dumps({"integrity": result["integrity"], "run": result["run"]}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
