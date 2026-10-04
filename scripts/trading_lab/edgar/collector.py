"""The EDGAR collector (step-driven, single owner): the watchlist manifest, one paced attempt per listing,
raw-first response records, and the processing that derives filings, revisions, observations, absences
and source health. `derive` is pure over a view of the store before the processing transaction, so the
offline replay re-runs exactly the same function."""

from __future__ import annotations

import fcntl
from bisect import bisect_right
from pathlib import Path
import uuid
from typing import Callable

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.edgar.listing import (
    ListingRejected, content_type_ok, filing_identity, parse_listing, source_item_id, submissions_url, cik10,
)
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.sources.httpclock import is_clock_verified, iso
from scripts.trading_lab.sources.limiter import RollingLimiter
from scripts.trading_lab.sources.store import Rejected

WATCHLIST_KEY = "watchlist"


class RequestCancelled(RuntimeError):
    """The runner's stop or authorization gate closed before transport could start."""


def clock_verdict(wall_at_receipt, date_lines: list[str], age_lines: list[str]) -> str:
    verified = is_clock_verified(wall_at_receipt, date_lines, age_lines, tolerance_s=spec.CLOCK_CHECK_TOLERANCE_S,
                                 age_max=spec.AGE_MAX, age_cap_s=spec.AGE_CAP_S)
    return "CLOCK_VERIFIED" if verified else "CLOCK_UNVERIFIED"


class _LatestEvents:
    """Versioned per-accession index: historical replay costs one lookup per
    filing, regardless of how many polls precede its processing transaction."""

    def __init__(self):
        self.ciks = {}

    def add(self, row):
        if row.kind not in {"FILING_OBSERVATION", "FILING_ABSENCE"}:
            return
        filings = self.ciks.setdefault(row.body["cik"], {})
        seqs, rows = filings.setdefault(row.body["accession_number"], ([], []))
        if seqs and seqs[-1] == row.seq:
            if row.kind == "FILING_ABSENCE":
                rows[-1] = row
        else:
            seqs.append(row.seq)
            rows.append(row)

    def at(self, cik, horizon):
        out = {}
        for accession, (seqs, rows) in self.ciks.get(cik, {}).items():
            i = bisect_right(seqs, horizon) - 1
            if i >= 0:
                out[accession] = rows[i]
        return out


def latest_events(view, cik: str) -> dict[str, object]:
    """The latest observation/absence per accession at this view's horizon."""
    latest = view.read_aggregate("edgar.latest_events", _LatestEvents, lambda index: index.at(cik, view.horizon()))
    view.store.reads["view_rows"] += len(latest)
    return latest


def derive(view, resp, body: bytes) -> tuple[str, dict, list[tuple[str, str, dict]]]:
    """The processing of one listing record over `view` (the store before its processing transaction)."""
    cik = resp.body["cik"]
    if not content_type_ok(resp.body["content_type_lines"]):
        return "PARSER_FAILED", {"reason": f"Content-Type {resp.body['content_type_lines']!r} is not {spec.JSON_MEDIA}"}, []
    try:
        listing = parse_listing(body, cik)
    except ListingRejected as exc:
        return "PARSER_FAILED", {"reason": str(exc)}, []
    rows, new, same = [], 0, 0
    for position, fields in enumerate(listing.in_scope):
        accession = fields["accession_number"]
        sid, content = source_item_id(accession), filing_identity(fields)
        key = f"{sid}:{content}"
        if view.rows("FILING_REVISION", key=key):
            same += 1
        else:
            new += 1
            rows.append(("FILING_REVISION", key, {
                "source_item_id": sid, "accession_number": accession, "cik": cik, "content_sha256": content,
                "identity": spec.METADATA_IDENTITY_ID, "fields": fields, "first_record": resp.seq,
                "amends": None, "amendment_link": "NOT_PROVIDED_BY_SOURCE" if fields["form"].endswith("/A") else None,
            }))
        rows.append(("FILING_OBSERVATION", f"{resp.seq}:{accession}", {
            "record": resp.seq, "source_item_id": sid, "accession_number": accession, "cik": cik, "revision": key,
            "observed_at": resp.body["observed_at"], "raw_sha256": resp.body["raw_sha"], "position": position,
            "entity_name": listing.entity_name,
        }))
    absences = 0
    for accession, event in sorted(latest_events(view, cik).items()):
        if event.kind != "FILING_OBSERVATION" or accession in listing.accessions:
            continue
        revision = view.rows("FILING_REVISION", key=event.body["revision"])[0]
        filed = revision.body["fields"]["filing_date"]
        if listing.oldest_filing_date is None or filed <= listing.oldest_filing_date:
            continue  # outside the documented window (VF2): nothing is inferred
        absences += 1
        rows.append(("FILING_ABSENCE", f"{resp.seq}:{accession}", {
            "record": resp.seq, "source_item_id": event.body["source_item_id"], "accession_number": accession,
            "cik": cik, "observed_at": resp.body["observed_at"], "raw_sha256": resp.body["raw_sha"],
            "filing_date": filed, "listing_oldest_filing_date": listing.oldest_filing_date,
        }))
    detail = {"rows": listing.rows, "in_scope": len(listing.in_scope), "new_revisions": new, "same_content": same,
              "absences": absences, "out_of_scope_forms": listing.out_of_scope_forms,
              "has_older_pages": listing.has_older_pages, "entity_name": listing.entity_name}
    return "LISTING_CLASSIFIED", detail, rows


def health_row(cik: str, *, attempt: int, record: int | None, result_state: str | None, reason: str | None,
               check_at: str | None) -> tuple[str, None, dict]:
    return ("SOURCE_HEALTH", None, {"surface": f"submissions:{cik}", "cik": cik, "attempt": attempt, "record": record,
                                    "result_state": result_state, "reason": reason or "NO_FAILURE", "check_at": check_at})


class EdgarCollector:
    def __init__(self, store: EdgarStore, fetcher, clock, *, boot_id: str = "boot-1"):
        self.store, self.fetcher, self.clock = store, fetcher, clock
        self._lock = open(Path(store.root) / "owner.lock", "a")  # single owner (flock), released on close
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise Rejected(f"another collector owns {store.root}") from exc
        self.epoch = uuid.uuid4().hex
        store.append("EPOCH", [("EPOCH", self.epoch, {"token": self.epoch, "boot_id": boot_id})])
        self.limiter = RollingLimiter(clock.mono, clock.sleep, spacing_s=spec.SPACING_S, window_s=spec.WINDOW_S,
                                      window_max=spec.WINDOW_MAX_STARTS, embargo_s=spec.EMBARGO_S)
        self.paused_until: float | None = None
        self.reconciled = self.reconcile()

    def reconcile(self) -> dict:
        """After a crash or a kill: an attempt of an earlier epoch without outcome is INTERRUPTED (its
        request may or may not have been sent; nothing is assumed), and a listing record without its
        processing outcome is processed now (its raw is durable: raw first)."""
        view = self.store.view()
        interrupted = 0
        for invoked in view.rows("TRANSPORT_INVOKED"):
            if invoked.body["epoch"] != self.epoch and not view.rows("ATTEMPT_OUTCOME", key=str(invoked.seq)):
                reason = "the owner stopped before the attempt ended"
                self.store.append("ATTEMPT_OUTCOME", [
                    ("ATTEMPT_OUTCOME", str(invoked.seq), {"attempt": invoked.seq, "outcome": "INTERRUPTED",
                                                            "status": None, "reason": reason}),
                    health_row(invoked.body["cik"], attempt=invoked.seq, record=None, result_state="INTERRUPTED",
                               reason=reason, check_at=self.store.wall_iso())])
                interrupted += 1
        processed = 0
        for resp in view.rows("RESPONSE"):
            if not view.rows("PROCESSING_OUTCOME", key=str(resp.seq)):
                self.process(resp.seq)
                processed += 1
        return {"interrupted": interrupted, "processed": processed}

    def close(self) -> None:
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()

    # ---- the watchlist (operator manifest, given once) -------------------------------------------
    def submit_watchlist(self, ciks: list[str]) -> int:
        padded = [cik10(c) for c in ciks]
        if not 1 <= len(padded) <= spec.WATCHLIST_MAX or len(set(padded)) != len(padded):
            raise ValueError(f"a watchlist holds 1 to {spec.WATCHLIST_MAX} distinct CIKs")
        existing = self.store.rows("MANIFEST", key=WATCHLIST_KEY)
        if existing:
            if existing[0].body["ciks"] != padded:
                raise Rejected("the watchlist is given once; this store already has another one")
            return existing[0].seq
        return self.store.append("MANIFEST", [("MANIFEST", WATCHLIST_KEY, {"ciks": padded})])

    def watchlist(self) -> list[str]:
        rows = self.store.rows("MANIFEST", key=WATCHLIST_KEY)
        return rows[0].body["ciks"] if rows else []

    # ---- one attempt ------------------------------------------------------------------------------
    def poll(self, cik: str, *, before_request: Callable[[], None] | None = None, request_allowed=None) -> dict:
        cik = cik10(cik)
        if cik not in self.watchlist():
            raise Rejected(f"CIK {cik} is not on the watchlist")
        if self.paused_until is not None and self.clock.mono() < self.paused_until:
            return {"status": "PAUSED", "until_mono": self.paused_until}

        def check_request():
            if before_request is not None:
                before_request()
            if request_allowed is not None and not request_allowed():
                raise RequestCancelled("the run stopped before the physical request")

        # Re-enter the storage/authorization gate during waits, even if grants are suspended.
        while True:
            check_request()
            grant = self.limiter.try_grant()
            if grant is not None:
                break
            self.clock.sleep(min(1.0, max(0.1, self.limiter.earliest(self.clock.mono()) - self.clock.mono())))
        check_request()
        url = submissions_url(cik)
        attempt = self.store.append("TRANSPORT_INVOKED", [("TRANSPORT_INVOKED", None, {
            "epoch": self.epoch, "cik": cik, "url": url, "grant_mono": grant})])
        try:
            check_request()  # the durable attempt may have waited past expiry, stop or a storage incident
        except RequestCancelled as exc:
            reason = str(exc)
            self.store.append("ATTEMPT_OUTCOME", [
                ("ATTEMPT_OUTCOME", str(attempt), {"attempt": attempt, "outcome": "INTERRUPTED",
                                                  "status": None, "reason": reason}),
                health_row(cik, attempt=attempt, record=None, result_state="INTERRUPTED",
                           reason=reason, check_at=self.store.wall_iso())])
            return {"status": "INTERRUPTED", "attempt": attempt, "reason": reason}
        result = self.fetcher.fetch(url, started=grant)
        fetch_seconds = round(self.clock.mono() - grant, 3)  # grant to the end of the fetch (deadline evidence)
        header_lines = [[name, value] for name, value in (result.headers or [])]
        if result.kind != "RESPONSE":
            check_at = iso(result.wall_at_receipt) if result.wall_at_receipt else self.store.wall_iso()
            self.store.append("ATTEMPT_OUTCOME", [
                ("ATTEMPT_OUTCOME", str(attempt), {"attempt": attempt, "outcome": result.kind, "status": result.status,
                                                    "reason": result.reason, "fetch_seconds": fetch_seconds,
                                                    "header_lines": header_lines}),
                health_row(cik, attempt=attempt, record=None, result_state=result.kind, reason=result.reason, check_at=check_at)])
            if result.kind == "SOURCE_THROTTLED":
                self.paused_until = self.clock.mono() + spec.THROTTLE_PAUSE_S
            return {"status": result.kind, "attempt": attempt, "reason": result.reason}
        wall = result.wall_at_receipt
        date_lines, age_lines = result.header_lines("Date"), result.header_lines("Age")
        verdict = clock_verdict(wall, date_lines, age_lines)
        digest = self.store.put_raw(result.body)  # raw first (E2)
        fields = {
            "attempt": attempt, "cik": cik, "url": url, "status": result.status,
            "content_type_lines": result.header_lines("Content-Type"),
            "content_encoding_lines": result.header_lines("Content-Encoding"),
            "date_lines": date_lines, "age_lines": age_lines, "wall_at_receipt": iso(wall), "verdict": verdict,
            "observed_at": iso(wall) if verdict == "CLOCK_VERIFIED" else None, "raw_sha": digest,
            "byte_length": len(result.body), "mode": "LIVE", "late_evidence": False,
            "header_lines": header_lines, "fetch_seconds": fetch_seconds,
        }
        record = self.store.append("RESPONSE", [
            ("RESPONSE", str(attempt), fields),
            ("ATTEMPT_OUTCOME", str(attempt), {"attempt": attempt, "outcome": "RESPONSE", "status": 200, "reason": None})])
        return {"status": "RESPONSE", "attempt": attempt, "record": record, **self.process(record)}

    def process(self, record: int) -> dict:
        resp = self.store.row_at("RESPONSE", record)
        body = self.store.read_raw(resp.body["raw_sha"])
        outcome, detail, rows = derive(self.store.view(), resp, body)
        failure = None if outcome == "LISTING_CLASSIFIED" else outcome
        self.store.append("PROCESSING", [
            ("PROCESSING_OUTCOME", str(record), {"record": record, "outcome": outcome, "detail": detail}),
            *rows,
            health_row(resp.body["cik"], attempt=resp.body["attempt"], record=record, result_state=failure,
                       reason=detail.get("reason"), check_at=resp.body["wall_at_receipt"])])
        return {"outcome": outcome, "detail": detail}

    def poll_all(self) -> list[dict]:
        return [self.poll(cik) for cik in self.watchlist()]
