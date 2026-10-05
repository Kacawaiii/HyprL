"""Read-only, redacted operations view for the cockpit. Unknown never means healthy."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

from scripts.trading_lab.ops.control import inspect, read_json, versions
from scripts.trading_lab.ops.supervisor import process_rss_bytes
from scripts.trading_lab.sources.store import read_only_connection

CODE = re.compile(r"[A-Z][A-Z0-9_]{0,79}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def age(value, now):
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if at.tzinfo is None:
            return None
        return max(0, (now - at).total_seconds()) if at <= now else None
    except (TypeError, ValueError, AttributeError):
        return None


def _jobs_schema(db):
    # JobStore v1 has no metadata table. Check the complete existing column binding.
    expected = ("id", "kind", "state", "payload", "limits_json", "progress", "cancel_requested",
                "worker_pid", "created_at", "updated_at", "result_hash", "error_code")
    if tuple(r[1] for r in db.execute("PRAGMA table_info(jobs)")) != expected:
        raise ValueError("job schema rejected")


def jobs(root):
    path = Path(root) / "lab" / "jobs.sqlite"
    if not path.is_file():
        return {"state": "NOT_OBSERVED", "workers": [], "budgets": None}
    with read_only_connection(path, validator=_jobs_schema) as db:
        counts = dict(db.execute("SELECT state,count(*) FROM jobs GROUP BY state"))
        if set(counts) - {"QUEUED", "RUNNING", "COMPLETE", "FAILED", "CANCELLED", "BLOCKED"}:
            raise ValueError("unknown job state")
        total = sum(counts.values())
        used = db.execute("SELECT coalesce(sum(length(payload)),0) FROM artifacts").fetchone()[0]
        from dataclasses import asdict
        from scripts.trading_lab.platform.jobs import ResourceLimits
        active = [{"pid": row[0], "progress": row[1], "limits": asdict(ResourceLimits(**json.loads(row[2])))}
                  for row in db.execute("SELECT worker_pid,progress,limits_json FROM jobs WHERE state='RUNNING'")]
        if any((row["pid"] is not None and (type(row["pid"]) is not int or row["pid"] <= 1))
               or type(row["progress"]) not in (float, int) or not 0 <= row["progress"] <= 1 for row in active):
            raise ValueError("invalid worker telemetry")
        errors = [r[0] if CODE.fullmatch(r[0] or "") else "WORKLOAD_ERROR"
                  for r in db.execute("SELECT error_code FROM jobs WHERE error_code IS NOT NULL ORDER BY updated_at DESC LIMIT 10")]
    return {"state": "OBSERVED", "states": counts, "workers": active, "errors": errors,
            "budgets": {"jobs_limit": 1000, "jobs_used": total, "jobs_remaining": max(0, 1000 - total),
                        "artifact_bytes_limit": 128 * 1024**2, "artifact_bytes_used": used,
                        "queue_limit": 8, "queue_used": counts.get("QUEUED", 0) + counts.get("RUNNING", 0), "worker_limit": 1}}


def edgar_runtime(root, process, now):
    path = Path(root) / "edgar" / "service-status.json"
    if not path.exists():
        return {"state": "NOT_OBSERVED", "budgets": None}
    data = read_json(path)
    elapsed = age(data.get("updated_at"), now)
    live = process["state"] == "RUNNING" and process["pid"] == data.get("pid")
    state = "OBSERVED" if live and elapsed is not None and elapsed <= 5 else "STALE"
    if data.get("state") == "ended" and process["state"] in ("STOPPED", "STALE"):
        state = "STOPPED"
    # Only explicitly bound public scalars cross the API boundary.
    payload = {"state": state, "age_seconds": elapsed, "freshness_limit_seconds": 5,
               "grants_suspended": bool(data.get("grants_suspended")),
               "storage_incident": data.get("storage_incident") is not None,
               "pending_incidents": data.get("pending_incidents", 0) if type(data.get("pending_incidents", 0)) is int else None,
               "budgets": None}
    from scripts.trading_lab.edgar.store import EdgarStore
    from scripts.trading_lab.edgar.service import authorization_state
    store = EdgarStore(Path(root) / "edgar", wall_clock=None, read_only=True)
    try:
        runs = store.rows("RUN_STARTED")
        if runs:
            run = runs[-1].body
            bound = run["authorization"]
            consumption = authorization_state(store, run["authorization_sha256"])
            payload["budgets"] = {"authorization_sha256": run["authorization_sha256"],
                "requests_limit": bound["max_requests"], "requests_used": consumption["requests"],
                "requests_remaining": max(0, bound["max_requests"] - consumption["requests"]),
                "not_after": bound["not_after"], "terminated": consumption["terminated"],
                "expired": datetime.fromisoformat(bound["not_after"].replace("Z", "+00:00")) <= now,
                "scope": "original durable store; never reuse a grant with another store"}
    finally:
        store.close()
    return payload


def health(*, ops_root=None, fomc=None, edgar=None, running_versions=None, now=None):
    now = now or datetime.now(timezone.utc)
    errors, sources, services = [], {}, {}
    for name, view in (("fomc", fomc), ("edgar", edgar)):
        if view is None:
            sources[name] = {"state": "NOT_CONFIGURED"}
            continue
        try:
            data = view.status()
            sources[name] = {"state": data["status"], "horizon": data.get("horizon"),
                "last_durable_activity": data.get("last_durable_activity"),
                "attested_as_of": data.get("suggested_as_of"),
                "age_seconds": age(data.get("suggested_as_of"), now), "read_only": True,
                "spec_hash": data["spec_hash"], "counts": data.get("counts"),
                "freshness_method": "archive attestation age; no claim of a live feed"}
            sources[name]["source_health"] = None
            if data["status"] == "AVAILABLE" and data.get("suggested_as_of"):
                read = view.snapshot(as_of=data["suggested_as_of"], horizon=data["horizon"], limit=1)
                sources[name]["read_state"] = read["snapshot"]["read_state"]
                states = {}
                for surface, row in (read.get("health") or {}).items():
                    if not isinstance(row, dict):
                        continue
                    state, reason = row.get("result_state"), row.get("reason")
                    states[surface] = {"result_state": state if isinstance(state, str) and CODE.fullmatch(state) else None,
                        "reason": reason if isinstance(reason, str) and CODE.fullmatch(reason) else None,
                        "check_at": row.get("check_at") if age(row.get("check_at"), now) is not None else None}
                    if state in ("SOURCE_UNAVAILABLE", "PARSER_FAILED", "NO_PROVIDER_HEALTH_STATE"):
                        errors.append(name.upper() + "_" + state)
                sources[name]["source_health"] = states
            if data["status"] == "REJECTED":
                errors.append(name.upper() + "_STORE_REJECTED")
        except Exception:
            sources[name] = {"state": "INTEGRITY_ERROR"}
            errors.append(name.upper() + "_READ_FAILED")
    operations, worker, capture = [], {"state": "NOT_CONFIGURED"}, {"state": "NOT_CONFIGURED"}
    if ops_root is not None:
        root = Path(ops_root)
        for name in ("app", "workers", "edgar"):
            state = inspect(root, name)
            services[name] = dict(state)
            services[name]["running_versions"] = None
            if state["state"] == "FOREIGN":
                errors.append(name.upper() + "_PROCESS_IDENTITY_REFUSED")
            try:
                ready_path = root / (name + ".ready.json")
                if state["state"] == "RUNNING" and ready_path.exists():
                    ready = read_json(ready_path)
                    if ready.get("pid") == state["pid"]:
                        recorded = ready.get("versions", {})
                        # Do not publish arbitrary edited contents of the private file.
                        sha = recorded.get("git_sha")
                        services[name]["running_versions"] = {"git_sha": sha if re.fullmatch(r"[a-f0-9]{40}", sha or "") else None,
                            "implementation_hash": recorded.get("implementation_hash") if HASH.fullmatch(recorded.get("implementation_hash", "")) else None,
                            "specs": {s: {"hash": v.get("hash") if HASH.fullmatch(v.get("hash", "")) else None,
                                "revision": v.get("revision") if type(v.get("revision")) is int else None}
                                for s, v in recorded.get("specs", {}).items() if s in ("fomc", "edgar")}}
                services[name]["rss_bytes"] = process_rss_bytes(state["pid"]) if state["state"] == "RUNNING" else None
            except Exception:
                errors.append(name.upper() + "_TELEMETRY_INVALID")
        try:
            path = root / "operations.json"
            for row in (read_json(path) if path.exists() else [])[-20:]:
                if row.get("action") in ("start", "stop", "resume", "backup", "restore") and row.get("service") in (*services, "all"):
                    code = row.get("code")
                    operations.append({"at": row.get("at") if age(row.get("at"), now) is not None else None,
                        "action": row["action"], "service": row["service"],
                        "state": row.get("state") if row.get("state") in ("COMPLETE", "BLOCKED") else "UNKNOWN",
                        "code": code if isinstance(code, str) and CODE.fullmatch(code) else None})
            worker = jobs(root)
            errors.extend(worker.get("errors", []))
        except Exception:
            worker = {"state": "INTEGRITY_ERROR"}
            errors.append("WORKER_TELEMETRY_READ_FAILED")
        try:
            capture = edgar_runtime(root, services["edgar"], now)
            if capture["state"] == "STALE":
                errors.append("EDGAR_STATUS_STALE")
            if capture.get("storage_incident") or capture.get("pending_incidents"):
                errors.append("EDGAR_STORAGE_INCIDENT")
        except Exception:
            capture = {"state": "INTEGRITY_ERROR"}
            errors.append("EDGAR_TELEMETRY_READ_FAILED")
    disk = None
    if ops_root is not None and Path(ops_root).is_dir():
        disk = shutil.disk_usage(ops_root).free
    return {"schema": "hyprl-ops-health-v1", "read_only": True, "observed_at": now.isoformat(),
        "status": "DEGRADED" if errors else "OBSERVED", "running_versions": running_versions or versions(),
        "sources": sources, "services": services, "last_operations": operations, "errors": sorted(set(errors)),
        "workers": worker, "edgar_service": capture,
        "resources": {"scope": "API process and configured runtime volume", "api_pid": os.getpid(),
            "api_rss_bytes": process_rss_bytes(os.getpid()), "api_cpu_seconds": time.process_time(),
            "api_threads": threading.active_count(), "runtime_disk_free_bytes": disk},
        "limitations": ["no live source request", "archive age is not a live freshness guarantee",
                        "missing telemetry remains unknown", "verify performs no workload or capture"]}
