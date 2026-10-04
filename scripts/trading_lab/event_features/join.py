"""Pinned-horizon point-in-time join using CAUSAL_AVAILABILITY_V3 only.

The batch assembler reuses the source's selection and raw-verification functions,
once per distinct prefix. Snapshot headers and identities equal the public per-T
snapshot path (tested). Tables are computed once per reader; bisect selects P(T).
No cached payload survives a batch call, so a later raw corruption fails closed.
"""

from bisect import bisect_right
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from scripts.trading_lab.edgar import snapshot as es, spec as ep
from scripts.trading_lab.edgar.collector import WATCHLIST_KEY
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.fomc import snapshot as fs, spec as fp, state, health
from scripts.trading_lab.fomc.store import FomcStore
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso
from scripts.trading_lab.sources.store import RawCorrupt, StoreRejected

FAILURES = (fs.SnapshotFailed, es.SnapshotFailed, RawCorrupt, StoreRejected,
            OSError, sqlite3.DatabaseError, KeyError, ValueError, IndexError, TypeError)


def instant(value: datetime | str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("decision time must carry an explicit UTC offset")
    return t.astimezone(timezone.utc)


class SourceJoin:
    """A read-only source at one fixed H. Reopen to adopt newly committed evidence."""

    def __init__(self, source: str, root: Path | str | None = None, *, horizon: int | None = None):
        if source not in ("fomc", "edgar"):
            raise ValueError("source must be fomc or edgar")
        self.source, self.store, self.view, self.error = source, None, None, None
        self.table, self.resolved, self.values, self.by_seq = [], [], [], {}
        self.H, self.first_statement = None, {}
        if root is None:
            return
        try:
            cls = FomcStore if source == "fomc" else EdgarStore
            self.store = cls(Path(root), wall_clock=None, read_only=True)
            head = self.store.horizon()
            if horizon is not None and (type(horizon) is not int or not 0 <= horizon <= head):
                raise ValueError("horizon must be a commit_seq within the store")
            self.H = head if horizon is None else horizon
            self.view = self.store.view(self.H)
            self.table = (state.availability if source == "fomc" else es.availability)(self.view, self.H)
            self.resolved = [a for a in self.table if a.resolved]
            self.values = [a.avail for a in self.resolved]
            self.by_seq = {a.seq: a.avail for a in self.resolved}
            if source == "fomc":
                revisions = {r.key: r.body["source_item_id"] for r in self.view.rows("REVISION")}
                for link in self.view.rows("LINK"):
                    if link.body["verified"] and link.seq in self.by_seq:
                        sid = revisions[link.body["revision"]]
                        self.first_statement.setdefault(sid, self.by_seq[link.seq])
        except FAILURES as exc:
            # Public reason codes do not leak configured paths or source bodies.
            self.error = "STORE_READ_FAILED:" + type(exc).__name__

    def close(self):
        if self.store is not None:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def coverage(self) -> dict | None:
        if not self.values:
            return None
        return {"start": iso(self.values[0]), "end": iso(self.values[-1])}

    def prefix(self, T: datetime) -> tuple[bool, int]:
        i = bisect_right(self.values, T)
        # Equal availabilities are one indivisible boundary; at the last one,
        # the next transaction is unresolved, exactly like causal.prefix.
        return i < len(self.resolved), self.resolved[i - 1].seq if i else 0

    def _payload(self, P: int) -> dict:
        """Source-owned selection under P, with all its raw checks preserved."""
        view = self.view.view(P)
        deps = set()
        if self.source == "fomc":
            outstanding = {i["key"] for i in state.outstanding(self.view, upto=P)}
            sids = sorted({c.key for c in view.rows("CANDIDATE")} |
                          {r.body["sid"] for r in view.select("RESPONSE", "surface", "primary")})
            discovery, deps = fs._discovery(self.view, P)
            items, cache = [], {}
            for sid in sids:
                item, more = fs._item_state(self.view, sid, P, self.table, outstanding, cache)
                items.append(item)
                deps |= more
            fs._verify_dependencies(self.view, deps)
            return {"P": P, "discovery": discovery, "items": items, "health": health.exposed(self.view, P)}
        manifest = view.rows("MANIFEST", key=WATCHLIST_KEY)
        watch = manifest[0].body["ciks"] if manifest else []
        source_health = {}
        for cik in watch:
            rows = view.select("SOURCE_HEALTH", "cik", cik)
            source_health[cik] = ({k: rows[-1].body[k] for k in
                                   ("result_state", "reason", "check_at", "attempt", "record")}
                                  if rows else {"result_state": "SOURCE_NOT_CHECKED",
                                                "reason": "no check within P(T)", "check_at": None,
                                                "attempt": None, "record": None})
        sids = sorted({o.body["source_item_id"] for o in view.rows("FILING_OBSERVATION")})
        filings = [es._filing(view, sid, self.table, deps) for sid in sids]
        filings.sort(key=lambda f: (f["cik"], f["fields"]["filing_date"], f["accession_number"]))
        for digest in sorted(deps):
            self.view.read_raw(digest)
        return {"P": P, "watchlist": watch, "health": source_health, "filings": filings}

    def _snapshot(self, T, resolved, payload):
        module, spec = (fs, fp) if self.source == "fomc" else (es, ep)
        snap = {"policy": module.POLICY_ID, "spec_hash": spec.SPEC_HASH, "T": iso(T),
                "H": self.H, "mode": "DURABLE_OBSERVED",
                "read_state": self.source.upper() + ("_RESOLVED" if resolved else "_CAUSAL_VISIBILITY_UNRESOLVED")}
        if resolved:
            snap.update(payload)
        snap["identity"] = sha256_canonical(snap)
        return snap

    def _row(self, T, snap=None, error=None):
        status = ("INTEGRITY_ERROR" if error else "NOT_CONFIGURED" if self.store is None else
                  "NOT_OBSERVED" if self.values and T < self.values[0] else
                  "RESOLVED" if snap and snap["read_state"].endswith("_RESOLVED") else "UNRESOLVED")
        events = [] if status == "RESOLVED" else None
        if status == "RESOLVED":
            if self.source == "fomc":
                for item in snap["items"]:
                    if item["state"] == "CURRENT_REVISION":
                        events.append({"event_id": item["sid"], "revision": item["revision"],
                                       "available_at": iso(self.first_statement[item["sid"]]),
                                       "content_hash": item["content_hash"]})
            else:
                events = [{"event_id": f["source_item_id"], "revision": f["revision"],
                           "available_at": f["first_available_at"], "cik": f["cik"],
                           "form": f["fields"]["form"], "items": f["fields"]["items"],
                           "listing_state": f["state"]} for f in snap["filings"]]
        return {"source": self.source, "T": iso(T), "state": status, "reason": error,
                "H": self.H, "coverage": self.coverage, "snapshot": snap, "events": events}

    def read_many(self, times) -> list[dict]:
        times = [instant(t) for t in times]
        cache, rows = {}, []
        for T in times:
            if self.error or self.store is None:
                rows.append(self._row(T, error=self.error))
                continue
            resolved, P = self.prefix(T)
            try:
                if resolved and P not in cache:
                    cache[P] = self._payload(P)
                payload = cache.get(P, {})
                rows.append(self._row(T, self._snapshot(T, resolved, payload)))
            except FAILURES as exc:
                rows.append(self._row(T, error="SNAPSHOT_READ_FAILED:" + type(exc).__name__))
        return deepcopy(rows)

    def read_one(self, T) -> dict:
        """Independent reference path through the source's public snapshot function."""
        T = instant(T)
        if self.error or self.store is None:
            return self._row(T, error=self.error)
        try:
            function = fs.events_as_of if self.source == "fomc" else es.filings_as_of
            return self._row(T, function(self.view, T, self.H))
        except FAILURES as exc:
            return self._row(T, error="SNAPSHOT_READ_FAILED:" + type(exc).__name__)
