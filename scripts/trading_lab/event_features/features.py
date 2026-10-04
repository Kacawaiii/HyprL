"""Feature v1: aggregate available events without reading any prices."""

from bisect import bisect_right
from datetime import timedelta

from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso
from .join import SourceJoin, instant
from .mapping import MAPPING_HASH, product_mapping, edgar_state

POLICY = "ATTESTED_EVENT_FEATURES_V1"
ITEMS = ("2.02", "5.02", "7.01", "8.01")


def values(events, T, source):
    """Open left, closed right windows, ordered exclusively by attested availability."""
    pairs = sorted(((instant(e["available_at"]), e) for e in events), key=lambda pair: (pair[0], pair[1]["event_id"]))
    times = [p[0] for p in pairs]
    right = bisect_right(times, T)
    left7 = bisect_right(times, T - timedelta(days=7))
    left30 = bisect_right(times, T - timedelta(days=30))
    last = times[right - 1] if right else None
    out = {"count_7d": right - left7, "count_30d": right - left30,
           "hours_since_last": (T - last).total_seconds() / 3600 if last else None}
    if source == "fomc":
        out["statement_within_24h"] = bool(last and T - last <= timedelta(hours=24))
    else:
        recent = [e for _, e in pairs[left7:right]]
        listed = {item.strip() for e in recent if e["items"] is not None for item in e["items"].split(",")}
        unknown = any(e["items"] is None for e in recent)
        out.update({"item_" + item.replace(".", "_"): True if item in listed else None if unknown else False
                    for item in ITEMS})
    return out


def missing_values(source):
    keys = ("count_7d", "count_30d", "hours_since_last")
    keys += (("statement_within_24h",) if source == "fomc" else
             tuple("item_" + item.replace(".", "_") for item in ITEMS))
    return dict.fromkeys(keys)


def source_features(row, T, mapping):
    source, snap = row["source"], row["snapshot"]
    override = edgar_state(mapping) if source == "edgar" else None
    status, reason = override or row["state"], row["reason"]
    # A mapped issuer outside this snapshot's watchlist was not observed; a
    # globally resolved source does not attest a zero for an unwatched issuer.
    if source == "edgar" and status == "RESOLVED" and mapping["cik"] not in snap["watchlist"]:
        status, reason = "NOT_OBSERVED", "ISSUER_NOT_IN_SNAPSHOT_WATCHLIST"
    events = row["events"] or []
    if source == "edgar":
        events = [e for e in events if e["cik"] == mapping["cik"] and e["form"] in ("8-K", "8-K/A")]
    return {"state": status, "source_state": row["state"], "reason": reason,
            "snapshot_identity": snap["identity"] if snap else None, "H": row["H"],
            "P": snap.get("P") if snap else None,
            "features": values(events, T, source) if status == "RESOLVED" else missing_values(source)}


class EventFeatures:
    def __init__(self, fomc_store=None, edgar_store=None):
        self.sources = {"fomc": SourceJoin("fomc", fomc_store), "edgar": SourceJoin("edgar", edgar_store)}

    def close(self):
        for source in self.sources.values():
            source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def rows(self, decisions) -> list[dict]:
        decisions = [(product_mapping(p), instant(t)) for p, t in decisions]
        times = sorted({t for _, t in decisions})
        reads = {source: dict(zip(times, reader.read_many(times))) for source, reader in self.sources.items()}
        result = []
        for mapping, T in decisions:
            row = {"policy": POLICY, "mapping_hash": MAPPING_HASH, "product": mapping["product"], "T": iso(T),
                   "inside_protected_holdout": PROTECTED_WINDOW_V1.start_at <= T < PROTECTED_WINDOW_V1.closes_at,
                   "sources": {s: source_features(reads[s][T], T, mapping) for s in self.sources}}
            row["identity"] = sha256_canonical(row)
            result.append(row)
        return result
