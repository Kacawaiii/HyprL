"""Post-durable processing: every PROCESSABLE record reaches exactly one terminal outcome, whatever
its clock verdict (causal_predicates.PROCESSABLE, source_health.failure_classification_v1).
Feed records are classified strictly in commit_seq order (FEED_ITEM_CLASSIFICATION_V2) and a
LIVE_ELIGIBLE feed record's cycle conclusion commits in the same transaction (LIVE_DISCOVERY_CYCLE_V2).
Primary records are classified (PRIMARY_CLASSIFICATION_V2) and IN_SCOPE_V1 outcomes link to a
mode-neutral revision keyed by (source item, content hash) (REVISION_MODE_NEUTRALITY_V1).
"""

from __future__ import annotations

import uuid

from scripts.trading_lab.fomc import identity, ledger, parsing, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.store import FomcStore, RawCorrupt, Rejected, Row

BLOCKING_RAISES = frozenset({"NEW_FEED_CANDIDATE", "NEW_UNIDENTIFIABLE_ITEM", "FIRST_UNSUPPORTED_DISCOVERY_SHAPE",
                             "GUID_CONFLICT", "GUID_COLLISION", "FEED_TITLE_CHANGED"})


def _diag(kind: str, *parts, marker: bool = False, sid: str | None = None, value=None) -> tuple[str, str, dict]:
    return ("DIAGNOSTIC_ONCE", spec.sha256_canonical([kind, *parts]), {"type": kind, "sid": sid, "value": value, "marker": marker})


def earlier_feed_unterminated(store: FomcStore, seq: int) -> bool:
    store = store.view()
    for f in state.feed_responses(store, upto=seq - 1):
        o = state.processing_outcome(store, f.seq)
        if o is None or o.body["outcome"] not in state.FEED_TERMINAL:
            return True
    return False


# ------------------------------------------------------------------ feed ---------------------------
def classify_feed(store: FomcStore, resp: Row, feed_items: list[parsing.FeedItem]) -> tuple[list, list, list]:
    store = store.view()
    candidates = {c.key: c for c in store.rows("CANDIDATE")}
    seen_diag = {d.key for d in store.rows("DIAGNOSTIC_ONCE")}
    unidentified = {u.key for u in store.rows("UNIDENTIFIABLE")}
    titles: dict[str, set] = {}
    guids_by_url: dict[str, set] = {}
    urls_by_guid: dict[str, set] = {}
    for d in store.rows("DIAGNOSTIC_ONCE"):
        if d.body["type"] == "TITLE_SEEN":
            titles.setdefault(d.body["sid"], set()).add(d.body["value"])
        elif d.body["type"] == "GUID_SEEN":
            url, guid = d.body["value"]
            guids_by_url.setdefault(url, set()).add(guid)
            urls_by_guid.setdefault(guid, set()).add(url)
    rows, summary, raised = [], [], []

    def once(row):
        if row[1] not in seen_diag:
            seen_diag.add(row[1])
            rows.append(row)
            return True
        return False

    groups: dict[str, list] = {}
    order: list[str] = []
    for item in feed_items:
        link = item.single("link")
        admitted = None
        if link is not None:
            try:
                admitted = identity.admit_url(parsing.normalize_feed_text(link))
            except identity.UrlRejected:
                admitted = None
        if admitted is None:
            key = identity.unidentifiable_key(item.raw("link"), item.raw("title"), item.raw("guid"))
            if key not in unidentified:
                unidentified.add(key)
                rows.append(("UNIDENTIFIABLE", key, {"created_by": resp.seq}))
                raised.append("NEW_UNIDENTIFIABLE_ITEM")
            summary.append({"class": "UNIDENTIFIABLE_ITEM", "key": key})
            continue
        sid = identity.source_item_id(admitted.canonical)
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append((admitted, item))
    for sid in order:
        admitted = groups[sid][0][0]
        occ = [it for _a, it in groups[sid]]
        valid_titles = sorted({parsing.normalize_feed_text(it.single("title")) for it in occ if it.single("title") is not None})
        guid_values = sorted({parsing.normalize_feed_text(it.single("guid")) for it in occ if it.single("guid") is not None})
        if sid not in candidates:
            shape = "FAMILY" if identity.is_statement_family_path(admitted.path) else "UNSUPPORTED"
            birth = valid_titles[0] if valid_titles else None
            rows.append(("CANDIDATE", sid, {"url": admitted.canonical, "shape": shape, "birth_title": birth, "created_by": resp.seq}))
            if shape == "FAMILY":
                rows.append(("ACQUISITION", sid, {"url": admitted.canonical, "created_by": resp.seq}))
            raised.append("NEW_FEED_CANDIDATE" if shape == "FAMILY" else "FIRST_UNSUPPORTED_DISCOVERY_SHAPE")
            candidates[sid] = True
            if birth is None:
                once(_diag("INVALID_FEED_TITLE", sid, marker=True, sid=sid))
            else:
                once(_diag("TITLE_SEEN", sid, birth, sid=sid, value=birth))
            titles[sid] = {birth} if birth is not None else set()
            remaining = valid_titles[1:]
            if not guid_values:
                once(_diag("GUID_MISSING", sid, sid=sid))
            klass = "NEW_FEED_CANDIDATE" if shape == "FAMILY" else "UNSUPPORTED_DISCOVERY_SHAPE"
        else:
            remaining = valid_titles
            klass = "EXISTING"
        for title in remaining:
            if title not in titles.setdefault(sid, set()):
                titles[sid].add(title)
                once(_diag("TITLE_SEEN", sid, title, sid=sid, value=title))
                if once(_diag("FEED_TITLE_CHANGED", sid, title, marker=(title == spec.FEED_TITLE_EXACT), sid=sid, value=title)):
                    raised.append("FEED_TITLE_CHANGED")
        url = admitted.canonical
        for guid in guid_values:
            baseline = guids_by_url.setdefault(url, set())
            if baseline and guid not in baseline:
                if once(_diag("GUID_CONFLICT", url, guid, sid=sid, value=guid)):
                    raised.append("GUID_CONFLICT")
            others = urls_by_guid.setdefault(guid, set())
            if others and url not in others:
                if once(_diag("GUID_COLLISION", guid, url, sid=sid, value=url)):
                    raised.append("GUID_COLLISION")
            baseline.add(guid)
            others.add(url)
            once(_diag("GUID_SEEN", url, guid, sid=sid, value=[url, guid]))
        summary.append({"class": klass, "sid": sid})
    return rows, summary, raised


def cycle_conclusion(store: FomcStore, resp: Row, raised: list, failure: str | None) -> tuple[str, str, dict]:
    store = store.view()
    reasons = []
    if failure:
        reasons.append(f"feed processing failed: {failure}")
    if earlier_feed_unterminated(store, resp.seq):
        reasons.append("an earlier PROCESSABLE LIVE feed record is not FEED_RECORD_TERMINAL before B")
    for kind in sorted(set(raised) & BLOCKING_RAISES):
        reasons.append(kind)
    blocked = state.outstanding(store, upto=resp.seq - 1)
    if blocked:
        reasons.append(f"{len(blocked)} OUTSTANDING item(s) in the pre-cycle state")
    result = "NOT_ZERO" if reasons else "EVENTS_OBSERVED_ZERO"
    return ("CYCLE_CONCLUSION", str(resp.seq), {"cycle_id": resp.seq, "B": resp.seq, "result": result, "reasons": reasons})


# ------------------------------------------------------------------ primary ------------------------
def classify_primary(store: FomcStore, resp: Row, body: bytes) -> tuple[str, dict, list]:
    fields = resp.body
    sid = fields["sid"]
    try:
        if identity.source_item_id(identity.admit_url(fields["request_url"]).canonical) != sid:
            return "IDENTITY_CONFLICT", {}, []
    except identity.UrlRejected:
        return "IDENTITY_CONFLICT", {}, []
    try:
        parsing.content_type_gate(fields["content_type_lines"], "primary")
        parsed = parsing.parse_primary(body)
    except parsing.ParseFailed as exc:
        return "PARSER_FAILED", {"reason": str(exc)}, []
    if len(parsed.titles) != 1:
        return "PARSER_FAILED", {"reason": "title anchor absent or duplicated"}, []
    title = parsed.titles[0]
    if fields["mode"] == "HISTORICAL_BACKFILL":
        feed_exact = True  # the manifest entry replaces the feed-title conjunct
    else:
        cand = store.rows("CANDIDATE", key=sid)
        feed_exact = bool(cand) and cand[0].body["birth_title"] == spec.FEED_TITLE_EXACT
    primary_exact = title == spec.PRIMARY_TITLE_EXACT
    if feed_exact != primary_exact:
        return "CLASSIFICATION_CONFLICT", {"title": title}, []
    if not primary_exact:
        return "DEFINITELY_OUT_OF_SCOPE", {"title": title}, []
    if len(parsed.dates) != 1 or parsing.parse_statement_date(parsed.dates[0]) is None:
        return "PARSER_FAILED", {"reason": "date anchor absent, duplicated or unparseable"}, []
    statement_date = parsing.parse_statement_date(parsed.dates[0])
    if len(parsed.release_segments) == 1:
        semantics, declared = parsing.parse_release(parsed.release_segments[0], statement_date)
        release_text = parsed.release_segments[0]
    else:
        semantics, declared, release_text = "UNPARSED", None, None
    revision_key = f"{sid}:{fields['raw_sha']}"
    normalized = {
        "source_item_id": sid, "content_hash": fields["raw_sha"], "capture_spec_hash": spec.SPEC_HASH,
        "provider_id": spec.PROVIDER_ID, "event_family": spec.EVENT_FAMILY, "classification_state": "IN_SCOPE_V1",
        "canonical_source_url": store.rows("CANDIDATE", key=sid)[0].body["url"] if store.rows("CANDIDATE", key=sid) else fields["request_url"],
        "official_statement_date": statement_date.isoformat(), "title": title,
        "declared_release_at": iso(declared) if declared else None, "declared_release_text": release_text,
        "declared_release_semantics": semantics, "content_source_available_at": None, "source_updated_at": None,
        "created_by_mode": fields["mode"],
    }
    existing = store.rows("REVISION", key=revision_key)
    rows = [] if existing else [("REVISION", revision_key, normalized)]
    rows.append(("LINK", str(resp.seq), {"record": resp.seq, "revision": revision_key, "mode": fields["mode"],
                                          "observed_at": fields["observed_at"] if state.verified(resp) else None,
                                          "verified": state.verified(resp)}))
    outcome = "NORMALIZED_SAME_CONTENT_NO_NEW_REVISION" if existing else "NORMALIZED_REVISION_COMMITTED"
    return outcome, {"revision": revision_key, "release": semantics}, rows


# ------------------------------------------------------------------ one fenced run ----------------
def start_run(store: FomcStore, seq: int, *, epoch: str, start_mono: float) -> str | None:
    """Commit PROCESSING_RUN {run_id, epoch, start, deadline = start + 600 s} for a record that may be
    processed now; None when it already has an outcome, must wait (feed order) or we are not the owner."""
    resp = store.row_at("RESPONSE", seq)
    if state.processing_outcome(store, seq) is not None:
        return None
    if resp.body["surface"] == "feed" and earlier_feed_unterminated(store, seq):
        return None
    run_id = uuid.uuid4().hex

    def owner(s: FomcStore) -> None:  # only the current epoch owner may start a run
        if ledger.current_epoch(s) != epoch:
            raise Rejected("not the current owner")

    try:
        store.append("PROCESSING_RUN", [("PROCESSING_RUN", str(seq), {
            "run_id": run_id, "epoch": epoch, "start_mono": start_mono,
            "deadline_mono": start_mono + spec.RUN_DEADLINE_S})], owner)
    except Rejected:
        return None
    return run_id


def finish_run(store: FomcStore, seq: int, *, epoch: str, run_id: str, mono=None, fault=None) -> str | None:
    """Compute the terminal outcome and commit it only while this run is RUNNING in the current epoch
    and before its 600 s deadline; a late or fenced result is discarded (None)."""
    resp = store.row_at("RESPONSE", seq)
    run = next(r for r in store.rows("PROCESSING_RUN", key=str(seq)) if r.body["run_id"] == run_id)
    if fault is not None:
        fault(resp)  # test hook: raise (the task ends) or advance the clock (the run overruns)
    is_feed = resp.body["surface"] == "feed"
    extra: list = []
    try:
        body = store.read_raw(resp.body["raw_sha"])
    except RawCorrupt:
        outcome, detail = "CORRUPTION_FAIL_CLOSED", {}
    else:
        if is_feed:
            try:
                parsing.content_type_gate(resp.body["content_type_lines"], "feed")
                feed_items = parsing.parse_feed(body)
            except parsing.ParseFailed as exc:
                outcome, detail = "PARSER_FAILED", {"reason": str(exc), "channel_level": True}
            else:
                extra, summary, raised = classify_feed(store, resp, feed_items)
                outcome, detail = "FEED_CLASSIFIED", {"items": summary, "raised": raised}
        else:
            outcome, detail, extra = classify_primary(store, resp, body)
    rows = [("PROCESSING_OUTCOME", str(seq), {"record": seq, "outcome": outcome, **detail})] + extra
    if is_feed and state.live_eligible(resp):
        rows.append(cycle_conclusion(store, resp, detail.get("raised", []), None if outcome == "FEED_CLASSIFIED" else outcome))

    def fence(s: FomcStore) -> None:
        runs = s.rows("PROCESSING_RUN", key=str(seq))
        if not runs or runs[-1].body["run_id"] != run_id or ledger.current_epoch(s) != epoch:
            raise Rejected("fenced processing run")
        if any(d.body["run_id"] == run_id for d in s.rows("RUN_DEAD", key=str(seq))):
            raise Rejected("run already DEAD")
        if mono is not None and mono() >= run.body["deadline_mono"]:
            raise Rejected("run past its 600 s deadline: late result discarded")

    try:
        store.append("PROCESSING_OUTCOME", rows, fence)
    except Rejected:
        return None
    return outcome


def process_record(store: FomcStore, seq: int, *, epoch: str, fault=None, mono=None) -> str | None:
    """One fenced processing run (start then finish); returns the terminal outcome or None."""
    existing = state.processing_outcome(store, seq)
    if existing is not None:
        return existing.body["outcome"]
    run_id = start_run(store, seq, epoch=epoch, start_mono=mono() if mono else 0.0)
    if run_id is None:
        return None
    return finish_run(store, seq, epoch=epoch, run_id=run_id, mono=mono, fault=fault)


def poison(store: FomcStore, seq: int, *, epoch: str) -> str | None:
    """POISON_GUARD: after two DEAD runs the current owner commits INTERNAL_PROCESSING_ERROR."""
    resp = store.row_at("RESPONSE", seq)
    rows = [("PROCESSING_OUTCOME", str(seq), {"record": seq, "outcome": "INTERNAL_PROCESSING_ERROR"})]
    if resp.body["surface"] == "feed" and state.live_eligible(resp):
        rows.append(cycle_conclusion(store, resp, [], "INTERNAL_PROCESSING_ERROR"))

    def owner(s: FomcStore) -> None:
        if ledger.current_epoch(s) != epoch:
            raise Rejected("not the current owner")

    try:
        store.append("PROCESSING_OUTCOME", rows, owner)
        return "INTERNAL_PROCESSING_ERROR"
    except Rejected:
        return None
