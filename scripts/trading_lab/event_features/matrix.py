"""Coverage evidence from read-only stores and price metadata, never price rows."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.trading_lab.equity_corpus import CORPUS_SPEC_V2
from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso
from .join import instant
from .mapping import MAPPING_HASH, MAPPING, edgar_state

REPO = Path(__file__).resolve().parents[3]


def fingerprint(root):
    """Private file names never leave this function; only a canonical tree digest."""
    files = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(65536), b""):
                    digest.update(block)
            files.append([path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest()])
    return {"files": len(files), "sha256": sha256_canonical(files)}


def overlap(window, coverage):
    if coverage is None:
        return None
    start = max(instant(window["start"]), instant(coverage["start"]))
    end = min(instant(window["end"]), instant(coverage["end"]))
    return {"start": iso(start), "end": iso(end)} if start <= end else None


def price_windows(repo=REPO):
    """Only manifests/specs/fingerprints. Installation does not imply verification."""
    path = repo / "data/crypto/coinbase_history_v1/manifest.json"
    crypto = json.loads(path.read_text())
    equity = json.loads((repo / "docs/artifacts/us_equity_corpus_v2_fingerprint.json").read_text())
    if equity["requested_range"] != {"start": CORPUS_SPEC_V2.requested_start, "end": CORPUS_SPEC_V2.requested_end}:
        raise ValueError("equity fingerprint/spec windows differ")
    windows = {}
    for instrument in crypto["products"]:
        windows[instrument["product"]] = {
            **crypto["spec"]["requested_range"], "timeframe": crypto["spec"]["timeframe"],
            "corpus_spec_hash": crypto["corpus_spec_hash"], "rows": instrument["canonical_rows"],
            "content_hash": instrument["canonical_sha256"],
            "installed": (path.parent / instrument["canonical_path"]).is_file(),
            "evidence": "data/crypto/coinbase_history_v1/manifest.json"}
    for instrument in equity["instruments"]:
        windows[instrument["instrument_id"].split(":")[1]] = {
            "start": CORPUS_SPEC_V2.requested_start + "T00:00:00+00:00",
            "end": CORPUS_SPEC_V2.requested_end + "T23:59:59.999999+00:00", "timeframe": "1d",
            "corpus_spec_hash": equity["corpus_spec_hash"], "rows": instrument["canonical_rows"],
            "content_hash": instrument["instrument_content_hash"],
            "installed": (repo / equity["local_storage_root"]).is_dir(),
            "evidence": "docs/artifacts/us_equity_corpus_v2_fingerprint.json"}
    return windows


def source_evidence(reader):
    if reader.error or reader.store is None:
        return {"state": "INTEGRITY_ERROR" if reader.error else "NOT_CONFIGURED", "reason": reader.error,
                "coverage": reader.coverage, "H": reader.H}
    view, by_seq = reader.view, reader.by_seq
    events, revisions = {}, []
    if reader.source == "fomc":
        events = {sid: {"available_at": iso(t)} for sid, t in reader.first_statement.items()}
        revisions = [{"available_at": iso(by_seq[r.seq])} for r in view.rows("REVISION") if r.seq in by_seq]
    else:
        for row in view.rows("FILING_OBSERVATION"):
            if row.seq in by_seq:
                revision = view.rows("FILING_REVISION", key=row.body["revision"])[0]
                fields = revision.body["fields"]
                events.setdefault(row.body["source_item_id"], {"available_at": iso(by_seq[row.seq]),
                                                              "cik": fields["cik"], "form": fields["form"]})
        revisions = [{"available_at": iso(by_seq[r.seq])} for r in view.rows("FILING_REVISION") if r.seq in by_seq]
    # Also exercise every resolved selection prefix and its read-time raw checks.
    samples = sorted(set(reader.values))[:-1]
    reads = reader.read_many(samples)
    if any(r["state"] == "INTEGRITY_ERROR" for r in reads):
        return {"state": "INTEGRITY_ERROR", "reason": "SNAPSHOT_DEPENDENCY_FAILED", "H": reader.H,
                "coverage": reader.coverage}
    histogram = Counter(e["available_at"] for e in events.values())
    return {"state": "AVAILABLE", "H": reader.H, "coverage": reader.coverage,
            "store_digest": fingerprint(reader.store.root),
            "resolved_transactions": len(reader.resolved), "unresolved_transactions": len(reader.table) - len(reader.resolved),
            "attested_events": len(events), "attested_revisions": len(revisions),
            "events_by_availability": [{"available_at": t, "count": c} for t, c in sorted(histogram.items())],
            "events_by_issuer_form": dict(sorted(Counter(e.get("cik", "MACRO") + " " + e.get("form", "statement")
                                                        for e in events.values()).items())),
            "availability_table_hash": sha256_canonical([{ "seq": a.seq, "resolved": a.resolved,
                                                           "avail": iso(a.avail) if a.avail else None} for a in reader.table]),
            "snapshot_proof_hash": sha256_canonical([r["snapshot"]["identity"] for r in reads]),
            "events_hash": sha256_canonical(events), "spec_hash": (reads[0]["snapshot"]["spec_hash"] if reads else None),
            "events": list(events.values())}


def build_matrix(sources, repo=REPO):
    windows = price_windows(repo)
    holdout = {"start": iso(PROTECTED_WINDOW_V1.start_at), "end": iso(PROTECTED_WINDOW_V1.closes_at),
               "end_exclusive": True, "hash": PROTECTED_WINDOW_V1.holdout_hash}
    evidence = {s: source_evidence(r) for s, r in sources.items()}
    rows = []
    for mapping in MAPPING["products"]:
        product = mapping["product"]
        for source, ev in evidence.items():
            applicable = edgar_state(mapping) if source == "edgar" else None
            events = ev.get("events", [])
            if source == "edgar":
                events = [e for e in events if e["cik"] == mapping["cik"]]
            state = applicable or ev["state"]
            usable = state == "AVAILABLE"
            window = windows[product]
            rows.append({"source": source, "product": product, "state": state, "price_window": window,
                         "attested_coverage": ev["coverage"], "price_coverage_overlap": overlap(window, ev["coverage"]),
                         "holdout_coverage_overlap": overlap(holdout, ev["coverage"]),
                         "attested_events": len(events) if usable else None,
                         "events_by_availability": ([{"available_at": t, "count": c} for t, c in
                                                      sorted(Counter(e["available_at"] for e in events).items())]
                                                     if usable else None),
                         "events_in_price_window": sum(instant(window["start"]) <= instant(e["available_at"]) <= instant(window["end"])
                                                       for e in events) if usable else None,
                         "events_in_holdout": sum(instant(holdout["start"]) <= instant(e["available_at"]) < instant(holdout["end"])
                                                   for e in events) if usable else None})
    out = {"schema": "event-coverage-matrix-v1", "mapping_hash": MAPPING_HASH, "holdout": holdout,
           "window_counts_are_inventory_not_feature_values": True,
           "sources": {s: {k: v for k, v in e.items() if k != "events"} for s, e in evidence.items()}, "rows": rows}
    out["identity"] = sha256_canonical(out)
    return out
