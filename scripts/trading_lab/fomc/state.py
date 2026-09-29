"""Pure derivations over durable records, up to a commit_seq horizon: causal predicates, server-attested
availability and NOW_LB, anchors and obligations, episode work state, item conclusion and RESOLVE
validity (causal_predicates, revision_policy.reobservation_*, retry_policy, zero_assertion).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta

from scripts.trading_lab.fomc import ledger, spec
from scripts.trading_lab.fomc.clock import parse_iso
from scripts.trading_lab.fomc.store import FomcStore, Row

BOUND = timedelta(seconds=spec.CLOCK_ERROR_BOUND_S)
NORMALIZED = frozenset({"NORMALIZED_REVISION_COMMITTED", "NORMALIZED_SAME_CONTENT_NO_NEW_REVISION"})
CONCLUDING = NORMALIZED | {"DEFINITELY_OUT_OF_SCOPE"}
FEED_TERMINAL = frozenset({"FEED_CLASSIFIED", "PARSER_FAILED", "INTERNAL_PROCESSING_ERROR", "CORRUPTION_FAIL_CLOSED"})


def outcome_class(outcome: str | None) -> str | None:
    if outcome in NORMALIZED:
        return "NORMALIZED"
    return outcome


# ------------------------------------------------------------------ causal predicates ------------
def verified(resp: Row) -> bool:
    """CLOCK_VERIFIED for every rule: LATE_EVIDENCE is treated as CLOCK_UNVERIFIED."""
    return resp.body["verdict"] == "CLOCK_VERIFIED" and not resp.body["late_evidence"]


def live_eligible(resp: Row) -> bool:
    """PROCESSABLE (every RESPONSE record is) + own clock verified + LIVE + not LATE_EVIDENCE."""
    return verified(resp) and resp.body["mode"] == "LIVE"


def observed_at(resp: Row) -> datetime | None:
    value = resp.body.get("observed_at")
    return parse_iso(value) if value else None


@dataclass(frozen=True)
class Avail:
    seq: int
    resolved: bool
    avail: datetime | None


def availability(store: FomcStore, horizon: int) -> list[Avail]:
    """CAUSAL_AVAILABILITY_V3 over txns <= horizon: avail(X) = max(V.observed_at + 92, observed_at
    concerned by X, avail of earlier txns), V = first verified non-late response whose
    TRANSPORT_INVOKED is after X; unresolved until V exists."""
    store = store.view(horizon)
    responses = store.rows("RESPONSE")
    refs = [r for r in responses if verified(r)]  # already in commit order
    concerned: dict[int, datetime] = {}
    for r in responses:
        if verified(r):
            concerned[r.seq] = max(concerned.get(r.seq, observed_at(r)), observed_at(r))
    for link in store.rows("LINK"):
        if link.body.get("observed_at"):
            t = parse_iso(link.body["observed_at"])
            concerned[link.seq] = max(concerned.get(link.seq, t), t)
    out, pointer, previous = [], 0, None
    for seq, _kind, _wall in store.txns():
        while pointer < len(refs) and refs[pointer].body["attempt"] <= seq:
            pointer += 1
        if pointer == len(refs) or (out and not out[-1].resolved):
            out.append(Avail(seq, False, None))
            continue
        value = observed_at(refs[pointer]) + BOUND
        for other in (concerned.get(seq), previous):
            if other is not None and other > value:
                value = other
        previous = value
        out.append(Avail(seq, True, value))
    return out


def now_lb(store: FomcStore, horizon: int | None = None) -> datetime | None:
    """SERVER_NOW_LB: max verified non-late observed_at - 92 s."""
    times = [observed_at(r) for r in store.view(horizon).rows("RESPONSE") if verified(r)]
    return max(times) - BOUND if times else None


def avail_of(table: list[Avail], seq: int) -> datetime | None:
    i = bisect_left(table, seq, key=lambda entry: entry.seq)  # the table is in commit_seq order
    if i < len(table) and table[i].seq == seq:
        return table[i].avail if table[i].resolved else None
    return None


# ------------------------------------------------------------------ records by item ---------------
def primary_responses(store: FomcStore, sid: str, *, upto: int | None = None, mode: str | None = None) -> list[Row]:
    return [r for r in store.select("RESPONSE", "sid", sid, upto=upto)
            if r.body.get("surface") == "primary" and (mode is None or r.body["mode"] == mode)]


def feed_responses(store: FomcStore, *, upto: int | None = None) -> list[Row]:
    return [r for r in store.select("RESPONSE", "surface", "feed", upto=upto) if r.body["mode"] == "LIVE"]


def processing_outcome(store: FomcStore, record_seq: int, *, upto: int | None = None) -> Row | None:
    rows = store.rows("PROCESSING_OUTCOME", key=str(record_seq), upto=upto)
    return rows[0] if rows else None


def anchor(store: FomcStore, sid: str, *, upto: int | None = None) -> Row | None:
    """The LIVE_ELIGIBLE primary record with the smallest commit_seq (fixed at commit)."""
    for r in primary_responses(store, sid, upto=upto):
        if live_eligible(r):
            return r
    return None


def obligations(store: FomcStore, sid: str, *, upto: int | None = None) -> list[dict]:
    a = anchor(store, sid, upto=upto)
    if a is None:
        return []
    base = observed_at(a)
    eligible = [observed_at(r) for r in primary_responses(store, sid, upto=upto) if live_eligible(r)]
    lb = now_lb(store, upto)
    out = []
    for offset in spec.OFFSETS_S:
        due = base + timedelta(seconds=offset)
        satisfied = any(t >= due for t in eligible)
        out.append({"offset": offset, "anchor": a.body["observed_at"], "due_at": due, "satisfied": satisfied,
                    "pending_due": (not satisfied) and lb is not None and lb >= due + BOUND})
    return out


# ------------------------------------------------------------------ episodes -----------------------
def work_satisfied(store: FomcStore, episode: Row, *, upto: int | None = None) -> bool:
    body = episode.body
    kind = body["kind"]
    if kind == "MANUAL_RETRY":
        return work_satisfied(store, Row(episode.seq, "EPISODE_OPEN", episode.key, body["work"]), upto=upto)
    if kind == "LIVE_ACQUISITION":
        return anchor(store, body["sid"], upto=upto) is not None
    if kind == "REOBSERVATION":
        due = parse_iso(body["anchor"]) + timedelta(seconds=body["offset"])
        return any(live_eligible(r) and observed_at(r) >= due for r in primary_responses(store, body["sid"], upto=upto))
    if kind == "HISTORICAL_BACKFILL":
        return any(verified(r) for a in ledger.attempts_of_episode(store, episode.key, upto=upto)
                   for r in store.rows("RESPONSE", key=str(a.seq), upto=upto))
    raise ValueError(kind)


def attempt_outcomes(store: FomcStore, key: str, *, upto: int | None = None) -> list[tuple[Row, Row | None]]:
    out = []
    for attempt in ledger.attempts_of_episode(store, key, upto=upto):
        rows = store.rows("ATTEMPT_OUTCOME", key=str(attempt.seq), upto=upto)
        out.append((attempt, rows[0] if rows else None))
    return out


def derived_status(store: FomcStore, episode: Row, *, upto: int | None = None) -> str:
    """OPEN, SUCCEEDED or SUSPENDED including terminal records that are derivable but not committed."""
    status = ledger.episode_status(store, episode.key, upto=upto)
    if status != "OPEN":
        return status
    if work_satisfied(store, episode, upto=upto):
        return "SUCCEEDED"
    outcomes = attempt_outcomes(store, episode.key, upto=upto)
    if len(outcomes) >= spec.ATTEMPTS_PER_EPISODE and all(o is not None for _a, o in outcomes):
        return "SUSPENDED"
    return "OPEN"


def item_episodes(store: FomcStore, sid: str, *, upto: int | None = None, live_only: bool = True) -> list[Row]:
    out = []
    for e in store.select("EPISODE_OPEN", "sid", sid, upto=upto):  # every episode body carries its item's sid
        work = e.body["work"] if e.body["kind"] == "MANUAL_RETRY" else e.body
        if live_only and work["kind"] == "HISTORICAL_BACKFILL":
            continue
        out.append(e)
    return out


# ------------------------------------------------------------------ items and zero -----------------
def item_record_seqs(store: FomcStore, sid: str, *, upto: int | None = None) -> list[int]:
    """OUTSTANDING_RELEVANT_ITEM_V1.item_records (HISTORICAL_BACKFILL records are never item records)."""
    seqs = [r.seq for r in primary_responses(store, sid, upto=upto, mode="LIVE")]
    seqs += [o.seq for s in list(seqs) for o in [processing_outcome(store, s, upto=upto)] if o is not None]
    seqs += [r.seq for r in store.rows("ACQUISITION", key=sid, upto=upto)]
    for e in item_episodes(store, sid, upto=upto):
        seqs.append(e.seq)
        seqs += [r.seq for r in store.rows("EPISODE_TERMINAL", key=e.key, upto=upto)]
        seqs += [a.seq for a in ledger.attempts_of_episode(store, e.key, upto=upto)]
    seqs += [m.seq for m in markers(store, sid, upto=upto)]
    return sorted(set(seqs))


def markers(store: FomcStore, sid: str, *, upto: int | None = None) -> list[Row]:
    return [d for d in store.select("DIAGNOSTIC_ONCE", "sid", sid, upto=upto) if d.body.get("marker")]


def resolutions(store: FomcStore, item_key: str, *, upto: int | None = None) -> list[Row]:
    return [r for r in store.select("OPERATOR", "item_key", item_key, upto=upto)
            if r.body["action"] == "RESOLVE" and r.body["valid"]]


def concluded(store: FomcStore, item: dict, *, upto: int) -> bool:
    """ITEM_SEMANTICALLY_CONCLUDED within the prefix <= upto."""
    key = item["key"]
    valid = resolutions(store, key, upto=upto)
    last_resolution = valid[-1].seq if valid else None
    if item["type"] == "CANDIDATE":
        records = item_record_seqs(store, item["sid"], upto=upto)
    elif item["type"] == "FAILED_FEED":
        records = [item["seq"]]
    else:
        records = [item["seq"]]
    if last_resolution is not None and not any(s > last_resolution for s in records):
        return True
    if item["type"] == "UNIDENTIFIABLE" or item.get("shape") == "UNSUPPORTED":
        return False
    if item["type"] == "FAILED_FEED":
        if item["reason"] != "PARSER_FAILED":
            return False
        for f in feed_responses(store, upto=upto):
            o = processing_outcome(store, f.seq, upto=upto)
            if f.seq > item["seq"] and live_eligible(f) and o is not None and o.body["outcome"] == "FEED_CLASSIFIED":
                return True
        return False
    return _naturally_concluded(store, item["sid"], upto, last_resolution)


def _naturally_concluded(store: FomcStore, sid: str, upto: int, last_resolution: int | None) -> bool:
    live = primary_responses(store, sid, upto=upto, mode="LIVE")
    outcomes = {r.seq: processing_outcome(store, r.seq, upto=upto) for r in live}
    if any(o is None for o in outcomes.values()):
        return False
    if any(derived_status(store, e, upto=upto) == "OPEN" for e in item_episodes(store, sid, upto=upto)):
        return False
    if any(last_resolution is None or m.seq > last_resolution for m in markers(store, sid, upto=upto)):
        return False
    eligible = [r for r in live if live_eligible(r)]
    if not eligible:
        return False
    latest = eligible[-1]
    kind = outcome_class(outcomes[latest.seq].body["outcome"])
    if outcomes[latest.seq].body["outcome"] not in CONCLUDING:
        return False
    return all(outcome_class(outcomes[r.seq].body["outcome"]) == kind for r in live if r.seq > latest.seq)


def items(store: FomcStore, *, upto: int) -> list[dict]:
    out = [{"type": "CANDIDATE", "key": f"SOURCE_ITEM:{c.key}", "sid": c.key, "shape": c.body["shape"], "seq": c.seq}
           for c in store.rows("CANDIDATE", upto=upto)]
    out += [{"type": "UNIDENTIFIABLE", "key": f"UNIDENTIFIABLE_ITEM:{u.key}", "seq": u.seq}
            for u in store.rows("UNIDENTIFIABLE", upto=upto)]
    for f in feed_responses(store, upto=upto):
        o = processing_outcome(store, f.seq, upto=upto)
        if o is not None and o.body["outcome"] in ("PARSER_FAILED", "INTERNAL_PROCESSING_ERROR", "CORRUPTION_FAIL_CLOSED"):
            out.append({"type": "FAILED_FEED", "key": f"FAILED_FEED:{f.seq}", "seq": f.seq, "reason": o.body["outcome"]})
    return out


def outstanding(store: FomcStore, *, upto: int) -> list[dict]:
    store = store.view(upto)
    return [item for item in items(store, upto=upto) if not concluded(store, item, upto=upto)]


def openable_acquisitions(store: FomcStore, sid: str, *, upto: int | None = None) -> list[Row]:
    """An acquisition is OPENABLE iff it has no EPISODE_OPEN and the item has no LIVE anchor."""
    if anchor(store, sid, upto=upto) is not None:
        return []
    from scripts.trading_lab.fomc.identity import acquisition_episode_key
    return [a for a in store.rows("ACQUISITION", key=sid, upto=upto)
            if not store.rows("EPISODE_OPEN", key=acquisition_episode_key(sid, a.seq), upto=upto)]


def resolve_validity(store: FomcStore, item_key: str) -> tuple[bool, str]:
    """OPERATOR_RESOLUTION_V2.validity evaluated on the durable state (called under the write lock)."""
    kind, _, value = item_key.partition(":")
    if kind != "SOURCE_ITEM":
        return True, "valid"
    store = store.view()  # refreshed under the write lock: exactly the committed state
    upto = store.horizon()
    live = primary_responses(store, value, upto=upto, mode="LIVE")
    if any(processing_outcome(store, r.seq) is None for r in live):
        return False, "a per-response item record is not processing-terminal"
    if any(derived_status(store, e) == "OPEN" for e in item_episodes(store, value)):
        return False, "a LIVE or manual-LIVE episode is open"
    if openable_acquisitions(store, value):
        return False, "a LIVE_ACQUISITION is still OPENABLE"
    return True, "valid"
