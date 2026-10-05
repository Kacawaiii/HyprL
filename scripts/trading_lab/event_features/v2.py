"""EVENT_FEATURES_V2: attested observations classified, counters named after what they measure.

V1 (`features.py`) counts every attested event, so the first listing of an issuer, which onboards
the issuer's whole existing inventory, looks like a burst of news. V2 classifies every attested
observation instead and counts only what is new:

* INITIAL_INVENTORY: an EDGAR accession present in the first valid listing of its issuer in the store; an
  FOMC statement acquired by a backfill manifest, or already a candidate at the first valid feed
  observation. Knowledge onboarding, never news.
* NEWLY_OBSERVED: an accession or statement first observed in a later read.
* REVISION: changed metadata/content of an item already observed.

Attested availability (CAUSAL_AVAILABILITY_V3) stays the only condition of use and orders every window;
filing dates and declared release times never contribute. Non-RESOLVED states keep null values.
Inventory sizes are reported as metadata, never as features. V1 is untouched.
"""

from bisect import bisect_right
from datetime import timedelta

from scripts.trading_lab.fomc import state
from scripts.trading_lab.research_protection import protection_flags
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso
from .join import SourceJoin, instant
from .mapping import MAPPING_HASH, edgar_state, product_mapping

POLICY = "ATTESTED_EVENT_FEATURES_V2"
CLASSIFICATION_POLICY = "ATTESTED_OBSERVATION_CLASSES_V1"
INITIAL_INVENTORY, NEWLY_OBSERVED, REVISION = "INITIAL_INVENTORY", "NEWLY_OBSERVED", "REVISION"
ITEMS = ("2.02", "5.02", "7.01", "8.01")
FORMS = ("8-K", "8-K/A")

FOMC_FEATURES = ("new_statements_attested_7d", "new_statements_attested_30d",
                 "statement_revisions_attested_7d", "statement_revisions_attested_30d",
                 "hours_since_last_new_statement", "new_statement_attested_within_24h")
EDGAR_FEATURES = ("new_accessions_attested_7d", "new_accessions_attested_30d",
                  "accession_revisions_attested_7d", "accession_revisions_attested_30d",
                  "hours_since_last_new_accession") + tuple(
                      "new_accession_item_" + item.replace(".", "_") + "_7d" for item in ITEMS)


def classify(reader: SourceJoin) -> list[dict]:
    """Every attested observation of the reader's store at its pinned H, in (availability, seq) order.

    An observation is attested only when its transaction's availability is resolved; observations that
    are not attested still advance the per-item state (they were observed), but are never emitted."""
    if reader.error or reader.store is None:
        return []
    out = (_fomc if reader.source == "fomc" else _edgar)(reader)
    return sorted(out, key=lambda o: (o["available_at"], o["seq"], o["event_id"]))


def _fomc(reader):
    view, by_seq = reader.view, reader.by_seq
    sid_of = {r.key: r.body["source_item_id"] for r in view.rows("REVISION")}
    candidate = {c.key: c.body["created_by"] for c in view.rows("CANDIDATE")}
    first_feed = next((r.seq for r in state.feed_responses(view, upto=reader.H)
                       if (o := state.processing_outcome(view, r.seq, upto=reader.H)) is not None
                       and o.body["outcome"] == "FEED_CLASSIFIED"), None)
    last, out = {}, []
    for link in sorted((l for l in view.rows("LINK") if l.body["verified"]), key=lambda l: l.seq):
        sid, revision = sid_of[link.body["revision"]], link.body["revision"]
        if sid not in last:
            inventory = (link.body["observation_mode"] == "HISTORICAL_BACKFILL"
                         or (first_feed is not None and candidate.get(sid) == first_feed))
            kind = INITIAL_INVENTORY if inventory else NEWLY_OBSERVED
        elif revision != last[sid]:
            kind = REVISION
        else:
            continue  # the same content seen again is not an event
        last[sid] = revision
        if link.seq in by_seq:
            out.append({"source": "fomc", "event_id": sid, "kind": kind, "seq": link.seq, "revision": revision,
                        "available_at": by_seq[link.seq]})
    return out


def _edgar(reader):
    view, by_seq = reader.view, reader.by_seq
    classified = {o.body["record"] for o in view.rows("PROCESSING_OUTCOME") if o.body["outcome"] == "LISTING_CLASSIFIED"}
    first_listing = {}
    for resp in view.rows("RESPONSE"):
        if resp.seq in classified:
            first_listing.setdefault(resp.body["cik"], resp.seq)
    last, out = {}, []
    for row in sorted(view.rows("FILING_OBSERVATION"), key=lambda r: r.seq):
        accession, revision = row.body["accession_number"], row.body["revision"]
        if accession not in last:
            kind = INITIAL_INVENTORY if row.body["record"] == first_listing.get(row.body["cik"]) else NEWLY_OBSERVED
        elif revision != last[accession]:
            kind = REVISION
        else:
            continue
        last[accession] = revision
        if row.seq in by_seq:
            fields = view.rows("FILING_REVISION", key=revision)[0].body["fields"]
            out.append({"source": "edgar", "event_id": row.body["source_item_id"], "kind": kind, "seq": row.seq,
                        "revision": revision, "available_at": by_seq[row.seq], "cik": fields["cik"],
                        "form": fields["form"], "items": fields["items"]})
    return out


def onboarding(reader: SourceJoin, cik: str | None) -> dict | None:
    """The attested moment each source's first valid read became available (EDGAR: per issuer)."""
    if reader.error or reader.store is None:
        return None
    view = reader.view
    if reader.source == "fomc":
        seq = next((r.seq for r in state.feed_responses(view, upto=reader.H)
                    if (o := state.processing_outcome(view, r.seq, upto=reader.H)) is not None
                    and o.body["outcome"] == "FEED_CLASSIFIED"), None)
        processing = view.rows("PROCESSING_OUTCOME", key=str(seq)) if seq else []
    else:
        records = {r.seq: r for r in view.rows("RESPONSE") if r.body["cik"] == cik}
        processing = sorted((o for o in view.rows("PROCESSING_OUTCOME")
                             if o.body["record"] in records and o.body["outcome"] == "LISTING_CLASSIFIED"),
                            key=lambda o: o.seq)[:1]
    if not processing or processing[0].seq not in reader.by_seq:
        return None
    return {"first_valid_read_available_at": reader.by_seq[processing[0].seq]}


def _windows(obs, T):
    times = [o["available_at"] for o in obs]
    right = bisect_right(times, T)
    return (right, bisect_right(times, T - timedelta(days=7)), bisect_right(times, T - timedelta(days=30)))


def values(observations: list[dict], T, source: str) -> tuple[dict, dict]:
    """(features, inventory metadata) at T from classified observations (already filtered to the source)."""
    by_kind = {k: [o for o in observations if o["kind"] == k] for k in (INITIAL_INVENTORY, NEWLY_OBSERVED, REVISION)}
    n_right, n_7, n_30 = _windows(by_kind[NEWLY_OBSERVED], T)
    r_right, r_7, r_30 = _windows(by_kind[REVISION], T)
    new, news = by_kind[NEWLY_OBSERVED], "statement" if source == "fomc" else "accession"
    last = new[n_right - 1]["available_at"] if n_right else None
    out = {f"new_{news}s_attested_7d": n_right - n_7, f"new_{news}s_attested_30d": n_right - n_30,
           f"{news}_revisions_attested_7d": r_right - r_7, f"{news}_revisions_attested_30d": r_right - r_30,
           f"hours_since_last_new_{news}": (T - last).total_seconds() / 3600 if last else None}
    if source == "fomc":
        out["new_statement_attested_within_24h"] = bool(last and T - last <= timedelta(hours=24))
    else:
        recent = new[n_7:n_right]
        listed = {i.strip() for o in recent if o["items"] is not None for i in o["items"].split(",")}
        unknown = any(o["items"] is None for o in recent)
        out.update({"new_accession_item_" + item.replace(".", "_") + "_7d":
                    True if item in listed else None if unknown else False for item in ITEMS})
    i_right = bisect_right([o["available_at"] for o in by_kind[INITIAL_INVENTORY]], T)
    inventory = by_kind[INITIAL_INVENTORY][:i_right]
    meta = {"inventory_observations_attested": len(inventory),
            "inventory_last_available_at": iso(inventory[-1]["available_at"]) if inventory else None}
    return out, meta


def missing_values(source):
    return dict.fromkeys(FOMC_FEATURES if source == "fomc" else EDGAR_FEATURES)


class EventFeaturesV2:
    """Same sources, states and readers as V1; features are computed from classified observations."""

    def __init__(self, fomc_store=None, edgar_store=None):
        self.sources = {"fomc": SourceJoin("fomc", fomc_store), "edgar": SourceJoin("edgar", edgar_store)}
        self._observations = {s: classify(r) for s, r in self.sources.items()}
        self._onboarding = {}

    def close(self):
        for source in self.sources.values():
            source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _source(self, source, read, T, mapping):
        override = edgar_state(mapping) if source == "edgar" else None
        status, reason, snap = override or read["state"], read["reason"], read["snapshot"]
        if source == "edgar" and status == "RESOLVED" and mapping["cik"] not in snap["watchlist"]:
            status, reason = "NOT_OBSERVED", "ISSUER_NOT_IN_SNAPSHOT_WATCHLIST"
        features, meta = missing_values(source), None
        if status == "RESOLVED":
            observations = self._observations[source]
            if source == "edgar":
                observations = [o for o in observations if o["cik"] == mapping["cik"] and o["form"] in FORMS]
            features, meta = values(observations, T, source)
            key = mapping["cik"] if source == "edgar" else None
            if (source, key) not in self._onboarding:
                self._onboarding[(source, key)] = onboarding(self.sources[source], key)
            first = self._onboarding[(source, key)]
            # as of T only: a first read attested after T is not yet known at T
            known = first and first["first_valid_read_available_at"] <= T
            meta["first_valid_read_available_at"] = iso(first["first_valid_read_available_at"]) if known else None
        return {"state": status, "source_state": read["state"], "reason": reason,
                "snapshot_identity": snap["identity"] if snap else None, "H": read["H"],
                "P": snap.get("P") if snap else None, "inventory": meta, "features": features}

    def rows(self, decisions) -> list[dict]:
        decisions = [(product_mapping(p), instant(t)) for p, t in decisions]
        times = sorted({t for _, t in decisions})
        reads = {s: dict(zip(times, reader.read_many(times))) for s, reader in self.sources.items()}
        result = []
        for mapping, T in decisions:
            row = {"policy": POLICY, "classification": CLASSIFICATION_POLICY, "mapping_hash": MAPPING_HASH,
                   "product": mapping["product"], "T": iso(T), "protection": protection_flags(mapping["product"], T),
                   "sources": {s: self._source(s, reads[s][T], T, mapping) for s in self.sources}}
            row["identity"] = sha256_canonical(row)
            result.append(row)
        return result
