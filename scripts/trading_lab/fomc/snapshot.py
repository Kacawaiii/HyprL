"""FOMC_CURRENT_CONTENT_SELECTION_V1: events_as_of(T) over the commit_seq prefix P(T) of a persisted
read horizon H, the provider-level read state, the discovery state and a canonical snapshot identity;
and an offline replay that re-verifies raw digests and clock verdicts and re-derives processing
before rebuilding the same snapshot (snapshot_binding, raw_policy.offline_replay).
"""

from __future__ import annotations

from datetime import datetime

from scripts.trading_lab.fomc import clock as clockmod
from scripts.trading_lab.fomc import canon, health, parsing, processing, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt

POLICY_ID = "FOMC_CURRENT_CONTENT_SELECTION_V1"
BARRIER = "CURRENT_CONTENT_UNAVAILABLE_DUE_TO_NEWER_UNNORMALIZED_SOURCE"


class SnapshotFailed(RuntimeError):
    """A verified snapshot depends on raw bytes that are absent or corrupt: it fails closed."""


class ReplayFailed(SnapshotFailed):
    """Verified replay fails closed (corrupt raw, verdict mismatch, re-derivation mismatch)."""


def prefix(store: FomcStore, T: datetime, H: int, table=None) -> tuple[str, int]:
    """Return (read_state, P): P is the longest prefix with resolved avail <= T; the FOMC part is
    admissible only if the next transaction is resolved (its avail > T)."""
    table = state.availability(store, H) if table is None else table
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


def content_identity(store: FomcStore, digest: str, cache: dict) -> str:
    """FOMC_CONTENT_IDENTITY_V1 of a record, recomputed from its raw, which is verified first: a corrupt
    raw fails the read closed even when its content identity would equal another record's."""
    if digest not in cache:
        try:
            cache[digest] = canon.canonicalize(store.read_raw(digest)).content_sha256
        except RawCorrupt as exc:
            raise SnapshotFailed(f"snapshot depends on raw {digest}: {exc}") from exc
    return cache[digest]


def _item_state(store: FomcStore, sid: str, P: int, table, outstanding_keys: set, cache: dict) -> tuple[dict, set]:
    """The item's selection state and the raw digests that state is derived from."""
    recs = state.primary_responses(store, sid, upto=P)
    O = recs[-1] if recs else None
    E = next((r for r in reversed(recs) if state.verified(r)), None)
    links = sorted((link for rev in store.select("REVISION", "source_item_id", sid, upto=P)
                    for link in store.select("LINK", "revision", rev.key, upto=P)), key=lambda link: link.seq)

    def verified_revision(content_hash):
        key = f"{sid}:{content_hash}"
        if not store.rows("REVISION", key=key, upto=P):
            return None
        return key if any(l.body["revision"] == key and l.body["verified"] for l in links) else None

    def exposed(record):
        o = state.processing_outcome(store, record.seq, upto=P)
        return {"record": record.seq, "mode": record.body["mode"], "verdict": "CLOCK_VERIFIED" if state.verified(record) else "CLOCK_UNVERIFIED",
                "outcome": o.body["outcome"] if o else "PENDING"}

    out, deps = {"sid": sid}, set()

    def content(record):  # its content identity; the read now depends on its raw
        deps.add(record.body["raw_sha"])
        return content_identity(store, record.body["raw_sha"], cache)

    if O is not None:
        integrity = store.rows("INTEGRITY_DIAGNOSTIC", key=str(O.seq), upto=P)
        o_outcome = state.processing_outcome(store, O.seq, upto=P)
        current = None
        if integrity or (o_outcome and o_outcome.body["outcome"] == "CORRUPTION_FAIL_CLOSED"):
            out.update(step=3, state=BARRIER, newest=exposed(O))  # states the corruption; derived from no raw
        elif state.verified(O) and verified_revision(H := content(O)):
            current = verified_revision(H)
            out.update(step=4, state="CURRENT_REVISION")
        elif not state.verified(O) and E is not None and content(E) == (H := content(O)) and verified_revision(H):
            current = verified_revision(H)
            out.update(step=5, state="CURRENT_REVISION")
        elif any(store.rows("REVISION", key=l.body["revision"], upto=P) for l in links):
            out.update(step=6, state=BARRIER, newest=exposed(O))
        elif state.verified(O) and o_outcome and o_outcome.body["outcome"] == "DEFINITELY_OUT_OF_SCOPE":
            out.update(step=7, state="NOT_IN_V1_SCOPE")
        else:
            out.update(step=8, state="SOURCE_ACTIVITY_NO_REVISION", newest=exposed(O))
        if out["step"] in (6, 7, 8) and o_outcome is not None:
            deps.add(O.body["raw_sha"])  # the exposed outcome (or the negative) was derived from O's bytes
        if current:
            created = store.rows("REVISION", key=current, upto=P)[0]
            revision = dict(created.body, ingested_at=iso(state.avail_of(table, created.seq)))  # resolved within P(T)
            cur_links = [dict(l.body, ingested_at=iso(state.avail_of(table, l.seq)))
                         for l in links if l.body["revision"] == current and l.body["verified"]]
            live = [r for r in recs if state.live_eligible(r)]
            out.update(revision=current, content_hash=revision["content_hash"], normalized=revision, links=cur_links,
                       live_available=bool(live) and content(live[-1]) == revision["content_hash"])
            deps.add(revision["first_raw_sha256"])  # a revision binds the raw that created it
            deps.update(l["raw_artifact_identities_and_hashes"][0]["raw_sha256"] for l in cur_links)  # and each exposed link
    elif f"SOURCE_ITEM:{sid}" in outstanding_keys:
        out.update(step=8, state="SOURCE_ACTIVITY_NO_REVISION")
    else:
        out.update(step=9, state="NOT_OBSERVED")
    return out, deps


def _discovery(store: FomcStore, P: int) -> tuple[dict, set]:
    """The discovery state and the raw digests it is derived from (the concluding feed record)."""
    feeds = state.feed_responses(store, upto=P)
    if not feeds:
        return {"state": "NO_CURRENT_CYCLE"}, set()
    F = feeds[-1]
    outcome = state.processing_outcome(store, F.seq, upto=P)
    if outcome is None or outcome.body["outcome"] not in state.FEED_TERMINAL:
        return {"state": "DISCOVERY_PENDING", "feed_record": F.seq}, set()
    if not state.live_eligible(F):
        return {"state": "NO_CURRENT_CYCLE", "feed_record": F.seq}, set()
    conclusion = store.rows("CYCLE_CONCLUSION", key=str(F.seq), upto=P)
    if not conclusion:
        return {"state": "DISCOVERY_PENDING", "feed_record": F.seq}, set()
    body = conclusion[0].body
    deps = set() if outcome.body["outcome"] == "CORRUPTION_FAIL_CLOSED" else {F.body["raw_sha"]}
    return {"state": body["result"], "cycle_id": body["cycle_id"], "B": body["B"]}, deps


def events_as_of(store: FomcStore, T: datetime, H: int | None = None, *, mode: str = "DURABLE_OBSERVED") -> dict:
    store = store.view(H)  # one consistent view of the horizon: no re-read of the store
    H = store.horizon() if H is None else H
    snap = {"policy": POLICY_ID, "spec_hash": spec.SPEC_HASH, "T": iso(T), "H": H, "mode": mode}
    if mode != "DURABLE_OBSERVED":
        snap["read_state"] = "FOMC_NOT_ADMISSIBLE_IN_MODE"
    else:
        table = state.availability(store, H)
        read_state, P = prefix(store, T, H, table)
        snap["read_state"] = read_state
        if read_state == "FOMC_RESOLVED":
            outstanding_keys = {i["key"] for i in state.outstanding(store, upto=P)}
            sids = sorted({c.key for c in store.rows("CANDIDATE", upto=P)} |
                          {r.body["sid"] for r in store.select("RESPONSE", "surface", "primary", upto=P)})
            discovery, deps = _discovery(store, P)
            item_states, cache = [], {}
            for sid in sids:
                item, item_deps = _item_state(store, sid, P, table, outstanding_keys, cache)
                item_states.append(item)
                deps |= item_deps
            _verify_dependencies(store, deps)
            snap.update(P=P, discovery=discovery, items=item_states, health=health.exposed(store, P))
    snap["identity"] = spec.sha256_canonical(snap)
    return snap


def _verify_dependencies(store: FomcStore, digests: set) -> None:
    """RAW_INTEGRITY_EVERYWHERE_V1 at read time: every raw a verified snapshot depends on is re-read
    and re-hashed now, so physical corruption fails the read closed even for an old (T, H) and
    without any earlier verify_integrity(). Read-only: nothing is written or fetched."""
    for digest in sorted(digests):
        try:
            store.read_raw(digest)
        except RawCorrupt as exc:
            raise SnapshotFailed(f"snapshot depends on raw {digest}: {exc}") from exc


def replay(store: FomcStore, T: datetime, H: int) -> dict:
    """Offline verified replay at (T, H): no network, no scheduler; fails closed on any mismatch."""
    view = store.view(H)
    for resp in view.rows("RESPONSE"):
        try:
            body = store.read_raw(resp.body["raw_sha"])
        except RawCorrupt as exc:
            raise ReplayFailed(f"raw of record {resp.seq}: {exc}") from exc
        verified = clockmod.is_clock_verified(clockmod.parse_iso(resp.body["wall_at_receipt"]),
                                              resp.body["date_lines"], resp.body["age_lines"])
        if ("CLOCK_VERIFIED" if verified else "CLOCK_UNVERIFIED") != resp.body["verdict"]:
            raise ReplayFailed(f"clock verdict of record {resp.seq} does not re-derive")
        recorded = state.processing_outcome(view, resp.seq)
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
            derived, detail, _rows = processing.classify_primary(view, resp, body)
            if state.outcome_class(derived) != state.outcome_class(recorded.body["outcome"]):
                raise ReplayFailed(f"primary record {resp.seq} re-derives {derived}")
            link = view.rows("LINK", key=str(resp.seq))
            if link:  # the content identity re-derives from the verified raw, and so does its revision key
                content = canon.canonicalize(body).content_sha256
                if (link[0].body["content_sha256"], link[0].body["revision"]) != (content, f"{resp.body['sid']}:{content}"):
                    raise ReplayFailed(f"content identity of record {resp.seq} does not re-derive")
    verify_health(store, H)
    return events_as_of(store, T, H)


def verify_health(store: FomcStore, H: int) -> None:
    """Replay of source health up to H, fail closed. Every health row must equal, field for field
    (key, provider_id, surface, check_at, result_state, reason, outcome, attempt, record, sid,
    diagnostics), the row re-derived from the durable data of its own transaction:
    - a processing outcome: from its record (surface, verdict, wall_at_receipt), the outcome row and
      the cycle conclusion of the same transaction;
    - a non-response attempt outcome: from its TRANSPORT_INVOKED and the outcome row, check_at being
      the transaction's wall_at_commit;
    - an integrity diagnostic: from the diagnosed record, the terminal outcome it keeps (committed
      earlier), check_at being the transaction's wall_at_commit; the diagnosed raw must still fail its
      digest (raw is never repaired).
    Every such transaction carries exactly one health row and no other transaction carries any.
    Limit: check_at repeats local wall readings (a transaction's wall_at_commit, a record's
    wall_at_receipt); replay proves the row repeats them, not that the readings were true, and a
    consistent rewrite of a reading together with its copy is not detectable."""
    view = store.view(H)
    walls = {seq: wall for seq, _kind, wall in view.txns()}
    actual: dict[int, object] = {}
    for row in view.rows("SOURCE_HEALTH"):
        if row.seq in actual:
            raise ReplayFailed(f"transaction {row.seq} carries two source-health results")
        actual[row.seq] = row
    expected: dict[int, tuple] = {}
    for outcome in view.rows("PROCESSING_OUTCOME"):
        resp = view.row_at("RESPONSE", outcome.body["record"])
        cycle = view.rows("CYCLE_CONCLUSION", key=str(resp.seq))
        cycle = cycle[0] if cycle and cycle[0].seq == outcome.seq else None
        result = health.for_record(state.verified(resp), outcome.body["outcome"], cycle.body["result"] if cycle else None)
        expected[outcome.seq] = health.row(health.surface_of_record(resp.body), resp.body["wall_at_receipt"], result,
                                           outcome=outcome.body["outcome"], attempt=resp.body["attempt"], record=resp.seq,
                                           sid=resp.body.get("sid"), diagnostics={"reason": outcome.body.get("reason")})
    for outcome in view.rows("ATTEMPT_OUTCOME"):
        result = None if outcome.body["outcome"] == "RESPONSE" else health.for_attempt_outcome(outcome.body["outcome"])
        if result is None:
            continue
        invoked = view.row_at("TRANSPORT_INVOKED", outcome.body["attempt"])
        expected[outcome.seq] = health.row(health.surface_of(invoked.body["kind"]), walls[outcome.seq], result,
                                           outcome=outcome.body["outcome"], attempt=outcome.body["attempt"], record=None,
                                           sid=invoked.body.get("sid"), diagnostics={"reason": outcome.body.get("reason")})
    for diag in view.rows("INTEGRITY_DIAGNOSTIC"):
        resp = view.row_at("RESPONSE", diag.body["record"])
        kept = state.processing_outcome(view, resp.seq, upto=diag.seq - 1) if resp is not None else None
        if kept is None or diag.body != {"record": resp.seq, "raw_sha": resp.body["raw_sha"], "outcome": kept.body["outcome"]}:
            raise ReplayFailed(f"integrity diagnostic {diag.seq} does not match a record with an earlier terminal outcome")
        try:
            store.read_raw(resp.body["raw_sha"])
        except RawCorrupt:
            pass
        else:
            raise ReplayFailed(f"integrity diagnostic {diag.seq}: the diagnosed raw verifies again (repaired or replaced)")
        expected[diag.seq] = health.row(health.surface_of_record(resp.body), walls[diag.seq], health.for_integrity_diagnostic(),
                                        outcome=kept.body["outcome"], attempt=resp.body["attempt"], record=resp.seq,
                                        sid=resp.body.get("sid"), diagnostics={"reason": health.DIAGNOSTIC_REASON})
    if set(actual) != set(expected):
        raise ReplayFailed("source-health results do not match the transactions that determine them")
    for seq, (_kind, surface, body) in expected.items():
        if actual[seq].key != surface or actual[seq].body != body:
            raise ReplayFailed(f"source-health result of transaction {seq} does not re-derive")
