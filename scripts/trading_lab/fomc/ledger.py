"""Atomic durable records whose predicates the spec requires inside the committing transaction:
limiter epochs (fencing), TRANSPORT_INVOKED, one outcome per attempt with LATE_EVIDENCE for late
responses, and episode open/terminal records (retry_policy, attempt_outcome_fence).
"""

from __future__ import annotations

from scripts.trading_lab.fomc import health, spec
from scripts.trading_lab.fomc.store import FomcStore, Rejected, Row

FEED_KEY = "FEED"


def begin_epoch(store: FomcStore, token: str, boot_id: str) -> int:
    """A new exclusive limiter epoch; its token fences every later TRANSPORT_INVOKED."""
    return store.append("EPOCH", [("EPOCH", token, {"token": token, "boot_id": boot_id})])


def current_epoch(store: FomcStore) -> str | None:
    rows = store.rows("EPOCH")
    return rows[-1].key if rows else None


def outcome_of(store: FomcStore, attempt_seq: int) -> Row | None:
    rows = store.rows("ATTEMPT_OUTCOME", key=str(attempt_seq))
    return rows[0] if rows else None


def attempts_without_outcome(store: FomcStore, *, sid: str | None = None, feed: bool = False) -> list[Row]:
    from scripts.trading_lab.fomc import state  # state derives from the ledger: imported late
    store = store.view()  # inside a write transaction the view is the committed state under the lock
    head = state.open_work(store)
    if head is not None:
        candidates = list(head.attempts)
    else:
        done = {row.key for row in store.rows("ATTEMPT_OUTCOME")}
        candidates = [row for row in store.rows("TRANSPORT_INVOKED") if str(row.seq) not in done]
    out = []
    for row in candidates:
        if feed and row.key == FEED_KEY:
            out.append(row)
        elif sid is not None and row.body.get("sid") == sid:
            out.append(row)
        elif not feed and sid is None:
            out.append(row)
    return out


def attempts_of_episode(store: FomcStore, key: str, *, upto: int | None = None) -> list[Row]:
    return store.rows("TRANSPORT_INVOKED", key=key, upto=upto)


def episode_status(store: FomcStore, key: str, *, upto: int | None = None) -> str:
    if not store.rows("EPISODE_OPEN", key=key, upto=upto):
        return "NONE"
    terminal = store.rows("EPISODE_TERMINAL", key=key, upto=upto)
    return terminal[0].body["status"] if terminal else "OPEN"


def transport_invoked(store: FomcStore, *, epoch: str, work: dict, grant_mono: float) -> int:
    """Commit the durable attempt record as the last step before transport. Once durable it counts."""

    def check(s: FomcStore) -> None:
        if current_epoch(s) != epoch:
            raise Rejected("fenced: not the current limiter epoch")
        if work["kind"] == "FEED_POLL":
            if attempts_without_outcome(s, feed=True):
                raise Rejected("a feed poll is still without outcome")
            return
        key = work["episode_key"]
        if episode_status(s, key) != "OPEN":
            raise Rejected("episode is not open")
        if len(attempts_of_episode(s, key)) >= spec.ATTEMPTS_PER_EPISODE:
            raise Rejected("attempt budget exhausted")
        if attempts_without_outcome(s, sid=work["sid"]):
            raise Rejected("an attempt of this item has no outcome yet")

    body = dict(work, epoch=epoch, grant_mono=grant_mono)
    key = FEED_KEY if work["kind"] == "FEED_POLL" else work["episode_key"]
    return store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", key, body)], check)


def commit_attempt_outcome(store: FomcStore, attempt_seq: int, outcome: str, detail: dict | None = None) -> int | None:
    """Commit a non-response outcome with its source-health result, in one transaction; returns None
    when the attempt already has an outcome (first wins, and no health result is written)."""
    body = {"attempt": attempt_seq, "outcome": outcome, **(detail or {})}
    rows = [("ATTEMPT_OUTCOME", str(attempt_seq), body)]
    invoked = store.row_at("TRANSPORT_INVOKED", attempt_seq)
    result = health.for_attempt_outcome(outcome)
    wall = store.wall_iso()  # check_at = the wall_at_commit of this transaction (durable provenance)
    if result is not None and invoked is not None:
        rows.append(health.row(health.surface_of(invoked.body["kind"]), wall, result, outcome=outcome,
                               attempt=attempt_seq, record=None, sid=invoked.body.get("sid"),
                               diagnostics={"reason": body.get("reason")}))
    try:
        return store.append("ATTEMPT_OUTCOME", rows, wall_at_commit=wall)
    except Rejected:
        return None


class SaveExpired(Exception):
    """The local save deadline has passed at commit time: the record is never admitted."""


def commit_response(store: FomcStore, attempt_seq: int, fields: dict, admit=None) -> tuple[int, bool]:
    """Commit a per-response record. It is its attempt's outcome only while the attempt has none;
    otherwise it is LATE_EVIDENCE, kept raw-first and treated as CLOCK_UNVERIFIED by every rule.
    `admit(store)` runs inside the committing transaction on both paths and may raise SaveExpired."""
    try:
        seq = store.append("RESPONSE", [
            ("RESPONSE", str(attempt_seq), dict(fields, attempt=attempt_seq, late_evidence=False)),
            ("ATTEMPT_OUTCOME", str(attempt_seq), {"attempt": attempt_seq, "outcome": "RESPONSE"}),
        ], admit)
        return seq, False
    except Rejected:
        seq = store.append("RESPONSE", [("RESPONSE", str(attempt_seq), dict(fields, attempt=attempt_seq, late_evidence=True))], admit)
        return seq, True


def open_episode(store: FomcStore, key: str, body: dict, check=None) -> int | None:
    try:
        return store.append("EPISODE_OPEN", [("EPISODE_OPEN", key, body)], check)
    except Rejected:
        return None


def close_episode(store: FomcStore, key: str, status: str, reason: str) -> int | None:
    assert status in ("SUCCEEDED", "SUSPENDED")
    try:
        return store.append("EPISODE_TERMINAL", [("EPISODE_TERMINAL", key, {"status": status, "reason": reason})])
    except Rejected:
        return None
