"""Did we shut down cleanly last time, and is the log still trustworthy?

Two different questions that are easy to conflate. An unclean shutdown is
ordinary -- a laptop lid, a SIGKILL, a power cut -- and the append-only log
plus WAL is designed to survive it. A broken hash chain is not ordinary and
means the audit trail can no longer be trusted.

So the UI gets both facts separately. "Recovered after an unclean shutdown,
chain verified" is a calm, informational line, not a red banner: the system
did exactly what it was built to do. Painting successful recovery as an alarm
trains people to dismiss alarms.

The clean-shutdown marker is written on the way out. Its absence at boot is
what identifies a crash -- there is nothing else to ask, since a process that
was killed had no chance to record anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.trading_lab.ops.health import DEGRADED, ERROR, HEALTHY
from scripts.trading_lab.ops.runtime_paths import write_private

RECOVERY_SCHEMA_VERSION = "trading-lab.recovery.v1"

BOOTING = "BOOTING"
RUNNING = "RUNNING"
STOPPED = "STOPPED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_lifecycle(path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def mark_started(path, *, port=None, now=None) -> dict:
    """Record that a run began, carrying forward how the last one ended."""
    previous = read_lifecycle(path) or {}
    was_clean = bool(previous.get("clean_shutdown", True)) if previous else True
    record = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "phase": RUNNING,
        "started_at": now or _now(),
        "port": port,
        "clean_shutdown": False,            # until proven otherwise on the way out
        "previous_shutdown_clean": was_clean,
        "previous_stopped_at": previous.get("stopped_at"),
    }
    write_private(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def mark_stopped(path, *, now=None) -> dict:
    record = read_lifecycle(path) or {"schema_version": RECOVERY_SCHEMA_VERSION}
    record.update({
        "phase": STOPPED,
        "stopped_at": now or _now(),
        "clean_shutdown": True,
    })
    write_private(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def last_shutdown_clean(path, *, running: bool = False) -> bool | None:
    """Was the *previous* shutdown clean? None when there is no history.

    The distinction that matters: a file saying RUNNING means either a live
    process (in which case the last shutdown is the one recorded before this
    run started) or a process that was killed and never wrote an exit. Reading
    RUNNING as "unclean" unconditionally would make a perfectly healthy
    running app report a recovery banner for its entire uptime.
    """
    record = read_lifecycle(path)
    if record is None:
        return None
    if record.get("phase") == RUNNING:
        if running:
            return _previous(record)
        return False
    return bool(record.get("clean_shutdown", False))


def _previous(record):
    value = record.get("previous_shutdown_clean")
    return None if value is None else bool(value)


def verify_runtime(store, *, session_id=None) -> dict:
    """Check the event chain and the newest snapshot without repairing them.

    Repair is deliberately absent. A tool that silently "fixes" a hash chain
    produces a log that verifies and means nothing; the only honest response
    to a broken chain is to report it loudly and keep the evidence.
    """
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "event_chain_verified": None,
        "latest_snapshot_verified": None,
        "sessions": 0,
        "events": 0,
        "status": HEALTHY,
        "error_code": None,
    }
    if store is None:
        report["status"] = DEGRADED
        report["error_code"] = "PAPER_RUNTIME_ABSENT"
        report["detail"] = "no shadow session has ever been recorded"
        return report

    sessions = store.sessions()
    report["sessions"] = len(sessions)
    if not sessions:
        report["status"] = DEGRADED
        report["error_code"] = "PAPER_RUNTIME_EMPTY"
        return report

    target = session_id or sessions[-1]
    report["session_id"] = target
    try:
        chain = store.verify_chain(session_id=target)
    except Exception as error:                     # a store that cannot be read
        report["event_chain_verified"] = False
        report["status"] = ERROR
        report["error_code"] = "PAPER_EVENT_CHAIN_INVALID"
        report["detail"] = str(error)
        return report

    report["event_chain_verified"] = bool(chain.get("verified"))
    report["events"] = int(chain.get("events", 0))
    report["head_hash"] = chain.get("head_hash")
    if not report["event_chain_verified"]:
        report["status"] = ERROR
        report["error_code"] = "PAPER_EVENT_CHAIN_INVALID"
        return report

    verified, checked = _verify_snapshots(store, target)
    report["snapshots_checked"] = checked
    report["latest_snapshot_verified"] = verified
    if checked and not verified:
        report["status"] = ERROR
        report["error_code"] = "PAPER_SNAPSHOT_INVALID"
    return report


def _verify_snapshots(store, session_id):
    """A snapshot must still hash to its own state and name a real event."""
    from scripts.trading_lab.app_api.contracts import SUPPORTED_PRODUCTS

    checked = 0
    for product in SUPPORTED_PRODUCTS:
        try:
            snapshot = store.latest_snapshot(session_id=session_id, product=product)
        except Exception:
            # The store raises when a snapshot no longer hashes to its own
            # state. That is exactly the failure this check exists to catch.
            return False, checked + 1
        if snapshot is None:
            continue
        checked += 1
        anchor = store.events(session_id=session_id,
                              after_event_id=int(snapshot["last_event_id"]) - 1,
                              limit=1)
        if not anchor:
            return False, checked
        if anchor[0].event_id != int(snapshot["last_event_id"]):
            return False, checked
        if anchor[0].event_hash != snapshot.get("last_event_hash"):
            return False, checked
    return (True if checked else None), checked


def snapshot_pressure(store, *, session_id=None) -> dict:
    """How far the log has run past its newest snapshot.

    The Phase 5D live smoke found a trigger that could never fire: snapshots
    silently stopped and nothing noticed, because nothing was watching this
    number. Now something is. Falling behind is DEGRADED, never ERROR -- a
    missing snapshot costs replay time on restart, it does not corrupt
    anything.
    """
    from scripts.trading_lab.app_api.contracts import SUPPORTED_PRODUCTS
    from scripts.trading_lab.paper_event_store import SNAPSHOT_EVERY_EVENTS

    payload = {"snapshot_every_events": SNAPSHOT_EVERY_EVENTS, "products": {},
               "status": HEALTHY}
    if store is None:
        payload["status"] = DEGRADED
        return payload
    sessions = store.sessions()
    if not sessions:
        payload["status"] = DEGRADED
        return payload
    target = session_id or sessions[-1]
    payload["session_id"] = target
    for product in SUPPORTED_PRODUCTS:
        try:
            snapshot = store.latest_snapshot(session_id=target, product=product)
        except Exception:
            payload["status"] = ERROR
            payload["error_code"] = "PAPER_SNAPSHOT_INVALID"
            snapshot = None
        last_id = int(snapshot["last_event_id"]) if snapshot else 0
        # Counted in this product's own events. A global event-id difference
        # would make whichever product stopped receiving candles first look
        # thousands of events behind while nothing was actually wrong.
        since = store.count_after(session_id=target, product=product,
                                  after_event_id=last_id)
        # Two full intervals is the alarm point: one interval of lag is normal
        # between snapshots, two means the trigger is not firing.
        behind = since > 2 * SNAPSHOT_EVERY_EVENTS
        payload["products"][product] = {
            "events_since_last_snapshot": since,
            "snapshot_due": since >= SNAPSHOT_EVERY_EVENTS,
            "has_snapshot": snapshot is not None,
        }
        if behind:
            payload["status"] = DEGRADED
            payload["error_code"] = "PAPER_SNAPSHOT_OVERDUE"
    return payload
