"""One bounded cache per configured source and pages bound to the complete snapshot.

The canonical identity covers the full read, so the first request must derive it in
linear time. Later pages slice that same read instead of deriving it again. Raw
integrity is checked on every request, including cache hits. No cache is persisted.
"""

from functools import wraps
import threading

from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError
from scripts.trading_lab.app_api.pagination import decode_cursor, encode_cursor, require_limit
from scripts.trading_lab.sources.store import RawCorrupt


def serialized_read(method):
    @wraps(method)
    def read(self, *args, **kwargs):
        if method.__name__ in {"snapshot", "item", "filing", "timeline"}:
            from scripts.trading_lab.app_api.sources import MAX_SOURCE_ITEMS
            require_limit(kwargs.get("limit"), default=min(200, MAX_SOURCE_ITEMS), maximum=MAX_SOURCE_ITEMS)
        with self._read_lock:
            return method(self, *args, **kwargs)
    return read


class PagedSource:
    def _init_pages(self):
        self._read_lock = threading.RLock()
        self._reader = self._stamp = self._cached = None
        self._detail = self._timeline = None

    def _open_cached(self, factory):
        # A replacement or same-head change invalidates the read. For an append
        # to the same admitted DB, retain the immutable prefix and let the mirror
        # decode only new rows. Refresh re-admits schema/spec before cache reuse.
        stamp = []
        for path in (self._root / factory.DB_NAME, self._root / (factory.DB_NAME + "-wal")):
            try:
                st = path.stat()
                stamp.append((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
            except FileNotFoundError:
                stamp.append(None)
        if self._reader is None or stamp != self._stamp:
            if (self._reader is not None and stamp[0] is not None and self._stamp[0] is not None
                    and stamp[0][:2] == self._stamp[0][:2]
                    and self._reader.horizon() > self._reader._mirror.upto):
                self._stamp = stamp
                return self._reader
            self.close()
            self._reader = factory(self._root, wall_clock=None, read_only=True)
            self._stamp = stamp
        return self._reader

    def close(self):
        with self._read_lock:
            if self._reader is not None:
                self._reader.close()
            self._reader = self._cached = self._detail = self._timeline = None

    def _cached_read(self, store, T, H, derive):
        key = (T.isoformat(), H)
        if self._cached is not None and self._cached[0] == key:
            for digest in self._cached[2]:
                try:
                    store.read_raw(digest)
                except RawCorrupt as exc:
                    raise ConflictError("the read fails its raw integrity check") from exc
            return self._cached[1]
        self._cached = self._detail = self._timeline = None
        digests = set()
        original = store.read_raw
        override = store.__dict__.get("read_raw")

        def tracked(digest):
            body = original(digest)
            digests.add(digest)
            return body

        store.read_raw = tracked
        try:
            snap = derive(store, T, H)
        finally:
            if override is None:
                del store.read_raw
            else:
                store.read_raw = override
        entities = {item["sid"]: item for item in snap.get("items", [])}
        entities.update({filing["accession_number"]: filing for filing in snap.get("filings", [])})
        self._cached = (key, snap, sorted(digests), entities)
        return snap

    def _pages(self, snap, collections, *, endpoint, product, limit, cursor, maximum):
        size = require_limit(limit, default=min(200, maximum), maximum=maximum)
        query = {"identity": snap["identity"]}
        start = 0
        if cursor:
            after = decode_cursor(cursor, endpoint=endpoint, product=product, query=query)
            if len(after) > 20 or not after.isascii() or not after.isdecimal():
                raise AppApiError("cursor position is malformed")
            start = int(after)
            if start >= max((len(rows) for rows in collections.values()), default=0):
                raise AppApiError("cursor position is outside this read")
        end = start + size
        pages = {name: rows[start:end] for name, rows in collections.items()}
        more = any(len(rows) > end for rows in collections.values())
        return pages, {"limit": size, "totals": {name: len(rows) for name, rows in collections.items()},
                       "next_cursor": encode_cursor(endpoint=endpoint, product=product,
                                                    last_timestamp=str(end), query=query) if more else None}

    @serialized_read
    def timeline(self, *, as_of=None, horizon=None, limit=None, cursor=None):
        from scripts.trading_lab.app_api.sources import MAX_SOURCE_ITEMS, _header, _verify_history_raws
        store = self._open()
        snap = self._read(store, as_of, horizon)
        if self._timeline is None:
            # Commit order, including each row within the same transaction.
            kinds = {"SOURCE_HEALTH", "PROCESSING_OUTCOME", "CYCLE_CONCLUSION", "REVISION", "LINK",
                     "FILING_REVISION", "FILING_OBSERVATION", "FILING_ABSENCE"}
            self._timeline = [{"committed_seq": row.seq, "kind": row.kind, "key": row.key, "body": row.body}
                              for row in store.view(snap["P"]).rows() if row.kind in kinds] if "P" in snap else []
        pages, pagination = self._pages(snap, {"rows": self._timeline}, endpoint="source.timeline",
                                        product=self._base()["source"], limit=limit, cursor=cursor,
                                        maximum=MAX_SOURCE_ITEMS)
        digests = []
        view = store.view(snap.get("P", 0))
        for row in pages["rows"]:
            body = row["body"]
            digests.extend(body[key] for key in ("raw_sha256", "first_raw_sha256") if body.get(key))
            digests.extend(raw["raw_sha256"] for raw in body.get("raw_artifact_identities_and_hashes", []))
            record = None
            if row["kind"] == "FILING_REVISION":
                record = body["first_record"]
            elif row["kind"] in {"SOURCE_HEALTH", "PROCESSING_OUTCOME"}:
                record = body.get("record")
            elif row["kind"] == "CYCLE_CONCLUSION":
                record = body["B"]
            if record is not None:
                digests.append(view.row_at("RESPONSE", record).body["raw_sha"])
        _verify_history_raws(store, digests)
        return {**self._base(), "snapshot": _header(snap), **pages, "pagination": pagination}
