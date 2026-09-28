"""FOMC_CURRENT_CONTENT_SELECTION_V1: events_as_of(T) over the commit_seq prefix P(T) of a persisted
read horizon H, the provider-level read state, the discovery state and a canonical snapshot identity;
and an offline replay that re-verifies raw digests and clock verdicts and re-derives processing
before rebuilding the same snapshot (snapshot_binding, raw_policy.offline_replay).
"""

from __future__ import annotations

from datetime import datetime

from scripts.trading_lab.fomc import clock as clockmod
from scripts.trading_lab.fomc import parsing, processing, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt

POLICY_ID = "FOMC_CURRENT_CONTENT_SELECTION_V1"
BARRIER = "CURRENT_CONTENT_UNAVAILABLE_DUE_TO_NEWER_UNNORMALIZED_SOURCE"


class ReplayFailed(RuntimeError):
    """Verified replay fails closed (corrupt raw, verdict mismatch, re-derivation mismatch)."""


def prefix(store: FomcStore, T: datetime, H: int) -> tuple[str, int]:
    """Return (read_state, P): P is the longest prefix with resolved avail <= T; the FOMC part is
    admissible only if the next transaction is resolved (its avail > T)."""
    table = state.availability(store, H)
    end, beyond = 0, None
    for entry in table:
        if entry.resolved and entry.avail <= T:
            end = entry.seq
        else:
            beyond = entry
            break
    if beyond is None or not beyond.resolved:
        return "FOMC_CAUSAL_VISIBILITY_UNRESOLVED", end
    return "FOMC_RESOLVED", end


def _item_state(store: FomcStore, sid: str, P: int, table, outstanding_keys: set) -> dict:
    recs = state.primary_responses(store, sid, upto=P)
    O = recs[-1] if recs else None
    E = next((r for r in reversed(recs) if state.verified(r)), None)
    links = [link for link in store.rows("LINK", upto=P) if link.body["revision"].startswith(sid + ":")]

    def verified_revision(content_hash):
        key = f"{sid}:{content_hash}"
        if not store.rows("REVISION", key=key, upto=P):
            return None
        return key if any(l.body["revision"] == key and l.body["verified"] for l in links) else None

    def exposed(record):
        o = state.processing_outcome(store, record.seq, upto=P)
        return {"record": record.seq, "mode": record.body["mode"], "verdict": "CLOCK_VERIFIED" if state.verified(record) else "CLOCK_UNVERIFIED",
                "outcome": o.body["outcome"] if o else "PENDING"}

    out = {"sid": sid}
    if O is not None:
        H = O.body["raw_sha"]
        integrity = store.rows("INTEGRITY_DIAGNOSTIC", key=str(O.seq), upto=P)
        o_outcome = state.processing_outcome(store, O.seq, upto=P)
        current = None
        if integrity or (o_outcome and o_outcome.body["outcome"] == "CORRUPTION_FAIL_CLOSED"):
            out.update(step=3, state=BARRIER, newest=exposed(O))
        elif state.verified(O) and verified_revision(H):
            current = verified_revision(H)
            out.update(step=4, state="CURRENT_REVISION")
        elif not state.verified(O) and E is not None and E.body["raw_sha"] == H and verified_revision(H):
            current = verified_revision(H)
            out.update(step=5, state="CURRENT_REVISION")
        elif any(store.rows("REVISION", key=l.body["revision"], upto=P) for l in links):
            out.update(step=6, state=BARRIER, newest=exposed(O))
        elif state.verified(O) and o_outcome and o_outcome.body["outcome"] == "DEFINITELY_OUT_OF_SCOPE":
            out.update(step=7, state="NOT_IN_V1_SCOPE")
        else:
            out.update(step=8, state="SOURCE_ACTIVITY_NO_REVISION", newest=exposed(O))
        if current:
            revision = store.rows("REVISION", key=current, upto=P)[0].body
            cur_links = [{"record": l.body["record"], "mode": l.body["mode"], "observed_at": l.body["observed_at"],
                          "avail": iso(state.avail_of(table, l.seq)) if state.avail_of(table, l.seq) else None}
                         for l in links if l.body["revision"] == current and l.body["verified"]]
            live = [r for r in recs if state.live_eligible(r)]
            out.update(revision=current, content_hash=revision["content_hash"], normalized=revision, links=cur_links,
                       live_available=bool(live) and live[-1].body["raw_sha"] == revision["content_hash"])
    elif f"SOURCE_ITEM:{sid}" in outstanding_keys:
        out.update(step=8, state="SOURCE_ACTIVITY_NO_REVISION")
    else:
        out.update(step=9, state="NOT_OBSERVED")
    return out


def _discovery(store: FomcStore, P: int) -> dict:
    feeds = state.feed_responses(store, upto=P)
    if not feeds:
        return {"state": "NO_CURRENT_CYCLE"}
    F = feeds[-1]
    outcome = state.processing_outcome(store, F.seq, upto=P)
    if outcome is None or outcome.body["outcome"] not in state.FEED_TERMINAL:
        return {"state": "DISCOVERY_PENDING", "feed_record": F.seq}
    if not state.live_eligible(F):
        return {"state": "NO_CURRENT_CYCLE", "feed_record": F.seq}
    conclusion = store.rows("CYCLE_CONCLUSION", key=str(F.seq), upto=P)
    if not conclusion:
        return {"state": "DISCOVERY_PENDING", "feed_record": F.seq}
    body = conclusion[0].body
    return {"state": body["result"], "cycle_id": body["cycle_id"], "B": body["B"]}


def events_as_of(store: FomcStore, T: datetime, H: int | None = None, *, mode: str = "DURABLE_OBSERVED") -> dict:
    H = store.horizon() if H is None else H
    snap = {"policy": POLICY_ID, "spec_hash": spec.SPEC_HASH, "T": iso(T), "H": H, "mode": mode}
    if mode != "DURABLE_OBSERVED":
        snap["read_state"] = "FOMC_NOT_ADMISSIBLE_IN_MODE"
    else:
        read_state, P = prefix(store, T, H)
        snap["read_state"] = read_state
        if read_state == "FOMC_RESOLVED":
            table = state.availability(store, H)
            outstanding_keys = {i["key"] for i in state.outstanding(store, upto=P)}
            sids = sorted({c.key for c in store.rows("CANDIDATE", upto=P)} |
                          {r.body["sid"] for r in store.rows("RESPONSE", upto=P) if r.body.get("surface") == "primary"})
            snap.update(P=P, discovery=_discovery(store, P),
                        items=[_item_state(store, sid, P, table, outstanding_keys) for sid in sids])
    snap["identity"] = spec.sha256_canonical(snap)
    return snap


def replay(store: FomcStore, T: datetime, H: int) -> dict:
    """Offline verified replay at (T, H): no network, no scheduler; fails closed on any mismatch."""
    for resp in store.rows("RESPONSE", upto=H):
        try:
            body = store.read_raw(resp.body["raw_sha"])
        except RawCorrupt as exc:
            raise ReplayFailed(f"raw of record {resp.seq}: {exc}") from exc
        verified = clockmod.is_clock_verified(clockmod.parse_iso(resp.body["wall_at_receipt"]),
                                              resp.body["date_lines"], resp.body["age_lines"])
        if ("CLOCK_VERIFIED" if verified else "CLOCK_UNVERIFIED") != resp.body["verdict"]:
            raise ReplayFailed(f"clock verdict of record {resp.seq} does not re-derive")
        recorded = state.processing_outcome(store, resp.seq, upto=H)
        if recorded is None or recorded.body["outcome"] in ("INTERNAL_PROCESSING_ERROR", "CORRUPTION_FAIL_CLOSED"):
            continue
        if resp.body["surface"] == "feed":
            try:
                parsing.content_type_gate(resp.body["content_type_lines"], "feed")
                parsing.parse_feed(body)
                derived = "FEED_CLASSIFIED"
            except parsing.ParseFailed:
                derived = "PARSER_FAILED"
            if derived != recorded.body["outcome"]:
                raise ReplayFailed(f"feed record {resp.seq} re-derives {derived}")
        else:
            derived, detail, _rows = processing.classify_primary(store, resp, body)
            if state.outcome_class(derived) != state.outcome_class(recorded.body["outcome"]):
                raise ReplayFailed(f"primary record {resp.seq} re-derives {derived}")
    return events_as_of(store, T, H)
