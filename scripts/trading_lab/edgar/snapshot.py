"""Causal reads of the EDGAR store and their verified offline replay (spec: causal, replay).

filings_as_of(T, H) shows what later server-attested evidence had made available at T (sources.causal,
CAUSAL_AVAILABILITY_V3): watched CIKs, their source health, and every filing seen so far with its state
(PRESENT or ABSENT_FROM_LISTING), current revision, normalized metadata and provenance. acceptanceDateTime
is shown only as provenance text (E5); a filing's availability is the availability of the transaction that
first recorded it. An unresolved read shows nothing."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.edgar.collector import WATCHLIST_KEY, clock_verdict, derive, health_row
from scripts.trading_lab.edgar.listing import derived_urls
from scripts.trading_lab.sources import causal
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso, parse_iso
from scripts.trading_lab.sources.store import RawCorrupt

POLICY_ID = "EDGAR_FILINGS_AS_OF_V1"
DERIVED_KINDS = ("FILING_REVISION", "FILING_OBSERVATION", "FILING_ABSENCE")


class SnapshotFailed(RuntimeError):
    """A read depends on raw bytes that are absent or corrupt: it fails closed (E7)."""


class ReplayFailed(SnapshotFailed):
    """The offline replay does not re-derive what the store recorded (E7)."""


def availability(store, horizon: int) -> list[causal.Avail]:
    return causal.availability(store, horizon, bound=spec.CLOCK_ERROR_BOUND,
                               concerned_kinds=("FILING_OBSERVATION", "FILING_ABSENCE"))


def now_lb(store, horizon: int | None = None) -> datetime | None:
    """SERVER_NOW_LB: the latest verified observed_at minus the clock error bound; a read later than it is
    unresolved by construction."""
    seen = [causal.observed_at(r) for r in store.view(horizon).rows("RESPONSE") if causal.verified(r)]
    return max(seen) - spec.CLOCK_ERROR_BOUND if seen else None


def _avail_of(table: list[causal.Avail], seq: int) -> str | None:
    i = bisect_left(table, seq, key=lambda entry: entry.seq)
    if i < len(table) and table[i].seq == seq and table[i].resolved:
        return iso(table[i].avail)
    return None


def _filing(view, sid: str, table, deps: set) -> dict:
    observations = view.select("FILING_OBSERVATION", "source_item_id", sid)
    absences = view.select("FILING_ABSENCE", "source_item_id", sid)
    first, last_obs = observations[0], observations[-1]
    latest_absence = absences[-1] if absences else None
    absent = latest_absence is not None and latest_absence.seq > last_obs.seq
    revision = view.rows("FILING_REVISION", key=last_obs.body["revision"])[0]
    fields = revision.body["fields"]
    deps |= {first.body["raw_sha256"], last_obs.body["raw_sha256"]}
    if absent:
        deps.add(latest_absence.body["raw_sha256"])
    verified = [o for o in observations if o.body["observed_at"]]
    return {
        "source_item_id": sid, "accession_number": fields["accession_number"], "cik": fields["cik"],
        "state": "ABSENT_FROM_LISTING" if absent else "PRESENT",
        "revision": revision.key, "content_sha256": revision.body["content_sha256"],
        "revisions_seen": len({o.body["revision"] for o in observations}), "observations": len(observations),
        "fields": fields, "amends": revision.body["amends"], "amendment_link": revision.body["amendment_link"],
        "first_recorded_in": first.seq, "first_available_at": _avail_of(table, first.seq),
        "first_observed_at": verified[0].body["observed_at"] if verified else None,
        "provenance": {
            "acceptance_datetime_text": fields["acceptance_datetime_text"],  # provenance only (UV2, E5)
            "first_record": first.body["record"], "first_raw_sha256": first.body["raw_sha256"],
            "latest_record": last_obs.body["record"], "latest_raw_sha256": last_obs.body["raw_sha256"],
            "entity_name": last_obs.body["entity_name"],
            "absent_since_record": latest_absence.body["record"] if absent else None,
            **derived_urls(fields["cik"], fields["accession_number"], fields["primary_document"]),
        },
    }


def filings_as_of(store, T: datetime, H: int | None = None) -> dict:
    view = store.view(H)
    H = view.horizon() if H is None else H
    snap = {"policy": POLICY_ID, "spec_hash": spec.SPEC_HASH, "T": iso(T), "H": H, "mode": "DURABLE_OBSERVED"}
    table = availability(view, H)
    resolved, P = causal.prefix(table, T)
    snap["read_state"] = "EDGAR_RESOLVED" if resolved else "EDGAR_CAUSAL_VISIBILITY_UNRESOLVED"
    if resolved:
        at = view.view(P)
        manifest = at.rows("MANIFEST", key=WATCHLIST_KEY)
        watch = manifest[0].body["ciks"] if manifest else []
        health = {}
        for cik in watch:
            rows = at.select("SOURCE_HEALTH", "cik", cik)
            health[cik] = ({k: rows[-1].body[k] for k in ("result_state", "reason", "check_at", "attempt", "record")}
                           if rows else {"result_state": "SOURCE_NOT_CHECKED", "reason": "no check within P(T)",
                                         "check_at": None, "attempt": None, "record": None})
        deps: set = set()
        sids = sorted({o.body["source_item_id"] for o in at.rows("FILING_OBSERVATION")})
        filings = [_filing(at, sid, table, deps) for sid in sids]
        filings.sort(key=lambda f: (f["cik"], f["fields"]["filing_date"], f["accession_number"]))
        for digest in sorted(deps):  # read-time raw integrity: every raw the filings derive from
            try:
                store.read_raw(digest)
            except RawCorrupt as exc:
                raise SnapshotFailed(f"read depends on raw {digest}: {exc}") from exc
        snap.update(P=P, watchlist=watch, health=health, filings=filings)
    snap["identity"] = sha256_canonical(snap)
    return snap


def replay(store, T: datetime, H: int) -> dict:
    """Offline, from the store alone: raws re-verified, clock verdicts and processing re-derived in commit
    order, then the read at (T, H). Any mismatch fails closed."""
    view = store.view(H)
    for resp in view.rows("RESPONSE"):
        try:
            body = store.read_raw(resp.body["raw_sha"])
        except RawCorrupt as exc:
            raise ReplayFailed(f"raw of record {resp.seq}: {exc}") from exc
        if clock_verdict(parse_iso(resp.body["wall_at_receipt"]), resp.body["date_lines"], resp.body["age_lines"]) \
                != resp.body["verdict"]:
            raise ReplayFailed(f"clock verdict of record {resp.seq} does not re-derive")
        recorded = view.rows("PROCESSING_OUTCOME", key=str(resp.seq))
        if not recorded:
            continue  # not processed within H
        txn = recorded[0].seq
        outcome, detail, rows = derive(store.view(txn - 1), resp, body)
        if (outcome, detail) != (recorded[0].body["outcome"], recorded[0].body["detail"]):
            raise ReplayFailed(f"record {resp.seq} re-derives {outcome}")
        stored = [(r.kind, r.key, r.body) for kind in DERIVED_KINDS for r in view.rows(kind) if r.seq == txn]
        if sorted(stored, key=repr) != sorted(rows, key=repr):
            raise ReplayFailed(f"the filings derived from record {resp.seq} do not re-derive")
    _verify_health(view)
    return filings_as_of(store, T, H)


def _verify_health(view) -> None:
    """Every health row repeats its durable source: the processing outcome of its record, or the failed
    attempt outcome committed with it."""
    for row in view.rows("SOURCE_HEALTH"):
        if row.body["record"] is not None:
            outcome = view.rows("PROCESSING_OUTCOME", key=str(row.body["record"]))[0]
            resp = view.row_at("RESPONSE", row.body["record"])
            failure = None if outcome.body["outcome"] == "LISTING_CLASSIFIED" else outcome.body["outcome"]
            expected = health_row(resp.body["cik"], attempt=resp.body["attempt"], record=row.body["record"],
                                  result_state=failure, reason=outcome.body["detail"].get("reason"),
                                  check_at=resp.body["wall_at_receipt"])[2]
        else:
            failed = view.rows("ATTEMPT_OUTCOME", key=str(row.body["attempt"]))[0]
            invoked = view.row_at("TRANSPORT_INVOKED", row.body["attempt"])
            expected = health_row(invoked.body["cik"], attempt=row.body["attempt"], record=None,
                                  result_state=failed.body["outcome"], reason=failed.body["reason"],
                                  check_at=row.body["check_at"])[2]
            if failed.seq != row.seq:
                raise ReplayFailed(f"health row {row.seq} is not committed with its attempt outcome")
        if row.body != expected:
            raise ReplayFailed(f"health row {row.seq} does not repeat its source")
