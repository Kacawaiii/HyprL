"""Read-only views over an official event-source store (FOMC V1, spec revision 25).

The store is opened read-only for every request (sources.store: no byte written, every write refused),
so the API can serve a closed archive or a copy of a running capture without touching it. Nothing here
fetches, captures, repairs or decides: every value shipped is a durable record or a derivation the FOMC
code already performs (events_as_of, verified replay, source health). The store path is fixed at
construction; a client-supplied item id is matched against the snapshot, never used as a path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sqlite3

from scripts.trading_lab.app_api.contracts import (
    APP_API_VERSION,
    MAX_SOURCE_ITEMS,
    AppApiError,
    ConflictError,
    NotFoundError,
)
from scripts.trading_lab.fomc import snapshot, spec, state
from scripts.trading_lab.fomc.clock import iso
from scripts.trading_lab.fomc.store import SCHEMA_VERSION, FomcStore
from scripts.trading_lab.sources.store import StoreRejected

_SID = re.compile(r"^[0-9a-f]{64}$")


def _as_of(value) -> datetime:
    if value in (None, ""):
        raise AppApiError("as_of is required: an ISO-8601 instant with its UTC offset")
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AppApiError(f"as_of is not an ISO-8601 instant: {value!r}") from exc
    if moment.tzinfo is None:
        raise AppApiError("as_of needs an explicit UTC offset; a naive time names no instant")
    return moment


def _horizon(value, store: FomcStore) -> int:
    head = store.view().horizon()
    if value in (None, ""):
        return head
    try:
        horizon = int(str(value))
    except ValueError as exc:
        raise AppApiError(f"horizon must be an integer commit_seq: {value!r}") from exc
    if not 0 <= horizon <= head:
        raise AppApiError(f"horizon {horizon} is outside the store (0..{head}); a read never invents later commits")
    return horizon


def _header(snap: dict) -> dict:
    return {key: snap.get(key) for key in ("policy", "spec_hash", "mode", "T", "H", "P", "read_state", "identity")}


def _summary(item: dict) -> dict:
    normalized = item.get("normalized") or {}
    return {
        "sid": item["sid"], "state": item["state"], "step": item["step"], "revision": item.get("revision"),
        "content_hash": item.get("content_hash"), "live_available": item.get("live_available"),
        "title": normalized.get("title"), "official_statement_date": normalized.get("official_statement_date"),
        "declared_release_at": normalized.get("declared_release_at"),
        "declared_release_text": normalized.get("declared_release_text"),
        "declared_release_trust_verdict": normalized.get("declared_release_trust_verdict"),
        "observation_mode": normalized.get("observation_mode"),
        "canonical_source_url": normalized.get("canonical_source_url"),
        "content_domain": (normalized.get("content_identity") or {}).get("domain"),
        "observations": len(item.get("links") or []),
    }


class FomcViews:
    def __init__(self, store_root=None):
        self._root = Path(store_root).resolve() if store_root else None

    def _base(self) -> dict:
        return {"api_version": APP_API_VERSION, "source": "fomc", "provider_id": spec.PROVIDER_ID,
                "spec_revision": spec.SPEC_REVISION, "spec_hash": spec.SPEC_HASH, "schema_version": SCHEMA_VERSION,
                "read_only": True}

    def _open(self) -> FomcStore:
        if self._root is None:
            raise NotFoundError("no FOMC store is configured for this API (start it with --fomc-store DIR)")
        try:
            return FomcStore(self._root, wall_clock=None, read_only=True)
        except (StoreRejected, OSError, sqlite3.DatabaseError) as exc:
            raise ConflictError("the configured FOMC store cannot be opened with this schema/spec") from exc

    def status(self) -> dict:
        if self._root is None:
            return {**self._base(), "status": "NOT_CONFIGURED", "store": None}
        try:
            store = FomcStore(self._root, wall_clock=None, read_only=True)
        except (StoreRejected, OSError, sqlite3.DatabaseError):
            return {**self._base(), "status": "REJECTED", "store": "configured",
                    "reason": "the configured FOMC store cannot be opened with this schema/spec"}
        try:
            view = store.view()
            txns = view.txns()
            lower = state.now_lb(view)
            return {
                **self._base(), "status": "AVAILABLE", "store": "configured", "horizon": view.horizon(),
                "first_durable_activity": txns[0][2] if txns else None,
                "last_durable_activity": txns[-1][2] if txns else None,
                # SERVER_NOW_LB: the latest instant the store attests from a verified response; a read
                # later than it is UNRESOLVED by construction, so the cockpit opens here.
                "suggested_as_of": iso(lower) if lower else None,
                "counts": {"epochs": len(view.rows("EPOCH")), "responses": len(view.rows("RESPONSE")),
                           "revisions": len(view.rows("REVISION")), "observations": len(view.rows("LINK")),
                           "cycles": len(view.rows("CYCLE_CONCLUSION"))},
            }
        finally:
            store.close()

    def _read(self, store: FomcStore, as_of, horizon) -> dict:
        T, H = _as_of(as_of), _horizon(horizon, store)
        try:
            return snapshot.events_as_of(store, T, H)
        except snapshot.SnapshotFailed as exc:
            raise ConflictError(f"the read fails closed: {exc}") from exc

    def snapshot(self, *, as_of=None, horizon=None) -> dict:
        store = self._open()
        try:
            snap = self._read(store, as_of, horizon)
            items = snap.get("items") or []
            if len(items) > MAX_SOURCE_ITEMS:
                raise AppApiError(f"{len(items)} items exceed the bound {MAX_SOURCE_ITEMS}")
            return {**self._base(), "snapshot": _header(snap), "discovery": snap.get("discovery"),
                    "health": snap.get("health"), "items": [_summary(item) for item in items]}
        finally:
            store.close()

    def item(self, sid, *, as_of=None, horizon=None) -> dict:
        if not isinstance(sid, str) or not _SID.match(sid):
            raise AppApiError("an item id is the 64-hex source_item_id")
        store = self._open()
        try:
            snap = self._read(store, as_of, horizon)
            out = {**self._base(), "snapshot": _header(snap), "item": None, "revisions": [], "observations": []}
            if snap["read_state"] != "FOMC_RESOLVED":
                return out  # an unresolved read shows nothing as of T, not a guess
            item = next((i for i in snap["items"] if i["sid"] == sid), None)
            if item is None:
                raise NotFoundError("no such item in the snapshot as of this instant")
            view, P = store.view(snap["P"]), snap["P"]
            out["item"] = item
            out["revisions"] = [
                {"revision_id": r.key, "committed_seq": r.seq, "content_hash": r.body.get("content_hash"),
                 "first_raw_sha256": r.body.get("first_raw_sha256"), "observation_mode": r.body.get("observation_mode"),
                 "content_identity": r.body.get("content_identity")}
                for r in view.select("REVISION", "source_item_id", sid)]
            observations = []
            for resp in state.primary_responses(view, sid, upto=P):
                outcome = state.processing_outcome(view, resp.seq, upto=P)
                observations.append({
                    "record": resp.seq, "attempt": resp.body["attempt"], "mode": resp.body["mode"],
                    "verdict": resp.body["verdict"], "status": resp.body.get("status"),
                    "observed_at": resp.body.get("observed_at"), "wall_at_receipt": resp.body.get("wall_at_receipt"),
                    "request_url": resp.body.get("request_url"), "final_url": resp.body.get("final_url"),
                    "redirect_chain": resp.body.get("redirect_chain"), "raw_sha256": resp.body.get("raw_sha"),
                    "byte_length": resp.body.get("byte_length"),
                    "processing_outcome": outcome.body["outcome"] if outcome else None,
                    "revision": next((link.body.get("revision") for link in view.rows("LINK", key=str(resp.seq))), None)})
            out["observations"] = observations
            return out
        finally:
            store.close()

    def replay(self, *, as_of=None, horizon=None) -> dict:
        """Verified offline replay at (T, H) against the plain read: same identity or a named failure."""
        store = self._open()
        try:
            first = self._read(store, as_of, horizon)
            out = {**self._base(), "snapshot": _header(first), "replay_identity": None, "identical": False, "error": None}
            try:
                again = snapshot.replay(store, _as_of(as_of), first["H"])
            except snapshot.SnapshotFailed as exc:  # ReplayFailed included
                out["error"] = str(exc)
                return out
            out["replay_identity"] = again["identity"]
            out["identical"] = again == first
            return out
        finally:
            store.close()


# ------------------------------------------------------------------ SEC EDGAR (spec revision 1) ---------
from scripts.trading_lab.edgar import snapshot as edgar_snapshot  # noqa: E402
from scripts.trading_lab.edgar import spec as edgar_spec  # noqa: E402
from scripts.trading_lab.edgar.store import EdgarStore  # noqa: E402


def _edgar_summary(filing: dict) -> dict:
    fields = filing["fields"]
    return {
        "accession_number": filing["accession_number"], "cik": filing["cik"], "form": fields["form"],
        "filing_date": fields["filing_date"], "report_date": fields["report_date"], "items": fields["items"],
        "state": filing["state"], "revisions_seen": filing["revisions_seen"], "observations": filing["observations"],
        "first_available_at": filing["first_available_at"], "first_observed_at": filing["first_observed_at"],
        "amendment_link": filing["amendment_link"],
        # provenance only: never an availability (E5)
        "acceptance_datetime_text": filing["provenance"]["acceptance_datetime_text"],
        "entity_name": filing["provenance"]["entity_name"],
    }


class EdgarViews:
    """Read-only views over an EDGAR store (offline slice; the spec authorizes no capture)."""

    def __init__(self, store_root=None):
        self._root = Path(store_root).resolve() if store_root else None

    def _base(self) -> dict:
        return {"api_version": APP_API_VERSION, "source": "edgar", "provider_id": edgar_spec.PROVIDER_ID,
                "spec_revision": edgar_spec.SPEC_REVISION, "spec_hash": edgar_spec.SPEC_HASH,
                "schema_version": edgar_spec.SCHEMA_VERSION, "read_only": True}

    def _open(self) -> EdgarStore:
        if self._root is None:
            raise NotFoundError("no EDGAR store is configured for this API (start it with --edgar-store DIR)")
        try:
            return EdgarStore(self._root, wall_clock=None, read_only=True)
        except (StoreRejected, OSError, sqlite3.DatabaseError) as exc:
            raise ConflictError("the configured EDGAR store cannot be opened with this schema/spec") from exc

    def status(self) -> dict:
        if self._root is None:
            return {**self._base(), "status": "NOT_CONFIGURED", "store": None}
        try:
            store = EdgarStore(self._root, wall_clock=None, read_only=True)
        except (StoreRejected, OSError, sqlite3.DatabaseError):
            return {**self._base(), "status": "REJECTED", "store": "configured",
                    "reason": "the configured EDGAR store cannot be opened with this schema/spec"}
        try:
            view = store.view()
            txns = view.txns()
            lower = edgar_snapshot.now_lb(view)
            manifest = view.rows("MANIFEST")
            return {
                **self._base(), "status": "AVAILABLE", "store": "configured", "horizon": view.horizon(),
                "first_durable_activity": txns[0][2] if txns else None,
                "last_durable_activity": txns[-1][2] if txns else None,
                "suggested_as_of": iso(lower) if lower else None,
                "watchlist": manifest[0].body["ciks"] if manifest else [],
                "counts": {"epochs": len(view.rows("EPOCH")), "responses": len(view.rows("RESPONSE")),
                           "revisions": len(view.rows("FILING_REVISION")),
                           "observations": len(view.rows("FILING_OBSERVATION")),
                           "absences": len(view.rows("FILING_ABSENCE"))},
            }
        finally:
            store.close()

    def _read(self, store: EdgarStore, as_of, horizon) -> dict:
        T, H = _as_of(as_of), _horizon(horizon, store)
        try:
            return edgar_snapshot.filings_as_of(store, T, H)
        except edgar_snapshot.SnapshotFailed as exc:
            raise ConflictError(f"the read fails closed: {exc}") from exc

    @staticmethod
    def _header(snap: dict) -> dict:
        return {key: snap.get(key) for key in ("policy", "spec_hash", "mode", "T", "H", "P", "read_state", "identity")}

    def snapshot(self, *, as_of=None, horizon=None) -> dict:
        store = self._open()
        try:
            snap = self._read(store, as_of, horizon)
            filings = snap.get("filings") or []
            if len(filings) > MAX_SOURCE_ITEMS:
                raise AppApiError(f"{len(filings)} filings exceed the bound {MAX_SOURCE_ITEMS}")
            return {**self._base(), "snapshot": self._header(snap), "watchlist": snap.get("watchlist"),
                    "health": snap.get("health"), "filings": [_edgar_summary(f) for f in filings]}
        finally:
            store.close()

    def filing(self, accession, *, as_of=None, horizon=None) -> dict:
        if not isinstance(accession, str) or not edgar_spec.ACCESSION.match(accession):
            raise AppApiError("a filing is named by its accession number (0000000000-00-000000)")
        store = self._open()
        try:
            snap = self._read(store, as_of, horizon)
            out = {**self._base(), "snapshot": self._header(snap), "filing": None, "revisions": [],
                   "observations": [], "absences": []}
            if snap["read_state"] != "EDGAR_RESOLVED":
                return out
            filing = next((f for f in snap["filings"] if f["accession_number"] == accession), None)
            if filing is None:
                raise NotFoundError("no such filing in the snapshot as of this instant")
            view, sid = store.view(snap["P"]), filing["source_item_id"]
            out["filing"] = filing
            out["revisions"] = [{"revision": r.key, "committed_seq": r.seq, "content_sha256": r.body["content_sha256"],
                                 "first_record": r.body["first_record"], "fields": r.body["fields"]}
                                for r in view.select("FILING_REVISION", "source_item_id", sid)]
            out["observations"] = [{key: o.body[key] for key in ("record", "observed_at", "raw_sha256", "position",
                                                                "revision", "entity_name")}
                                   for o in view.select("FILING_OBSERVATION", "source_item_id", sid)]
            out["absences"] = [{key: a.body[key] for key in ("record", "observed_at", "raw_sha256", "filing_date",
                                                            "listing_oldest_filing_date")}
                               for a in view.select("FILING_ABSENCE", "source_item_id", sid)]
            return out
        finally:
            store.close()

    def replay(self, *, as_of=None, horizon=None) -> dict:
        store = self._open()
        try:
            first = self._read(store, as_of, horizon)
            out = {**self._base(), "snapshot": self._header(first), "replay_identity": None, "identical": False, "error": None}
            try:
                again = edgar_snapshot.replay(store, _as_of(as_of), first["H"])
            except edgar_snapshot.SnapshotFailed as exc:
                out["error"] = str(exc)
                return out
            out["replay_identity"] = again["identity"]
            out["identical"] = again == first
            return out
        finally:
            store.close()
