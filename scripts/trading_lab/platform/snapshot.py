"""InformationSnapshot(T) v1 over source-owned causal reads, without a common global horizon."""
from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta

from scripts.trading_lab.edgar import spec as edgar_spec
from scripts.trading_lab.event_features import features as v1, v2
from scripts.trading_lab.event_features.join import FAILURES, SourceJoin, instant
from scripts.trading_lab.event_features.mapping import MAPPING_HASH, edgar_state, product_mapping
from scripts.trading_lab.fomc import spec as fomc_spec
from scripts.trading_lab.platform.contracts import InformationSnapshot
from scripts.trading_lab.platform.events import METHOD_HASH, understand
from scripts.trading_lab.platform.prices import POLICY as PRICE_POLICY
from scripts.trading_lab.platform.providers import source_contract
from scripts.trading_lab.research_protection import protection_flags
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.sources.httpclock import iso

POLICY = "INFORMATION_SNAPSHOT_V1"
DEPENDENCIES = "FEATURE_HISTORY_AND_CAUSAL_ATTESTATIONS_V1"


def verify_dependencies(reader, P):
    """Verify historical acquisitions/classification, revisions, and availability attestation raws.

    A current filing read verifies first/latest evidence. Features also use intermediate revisions,
    inventory onboarding and earlier new observations: verify the entire acquisition prefix. The
    first subsequent verified response for each transaction (and the next-prefix barrier) attests
    its availability, so those raw identities also belong to this snapshot's evidence.
    """
    view = reader.view
    responses = view.rows("RESPONSE")
    refs = [r for r in responses if r.body["verdict"] == "CLOCK_VERIFIED" and not r.body["late_evidence"]]
    attempts = [r.body["attempt"] for r in refs]
    raw = {r.body["raw_sha"] for r in responses if r.seq <= P}
    boundary = next((a.seq for a in reader.table if a.seq > P), P)
    for transaction, _, _ in view.txns(upto=boundary):
        index = bisect_right(attempts, transaction)
        if index < len(refs):
            raw.add(refs[index].body["raw_sha"])
    kinds = ("LINK", "REVISION") if reader.source == "fomc" else ("FILING_OBSERVATION", "FILING_REVISION", "FILING_ABSENCE")
    for kind in kinds:
        for row in view.rows(kind, upto=P):
            for key in ("raw_sha256", "first_raw_sha256"):
                if row.body.get(key):
                    raw.add(row.body[key])
            raw.update(r["raw_sha256"] for r in row.body.get("raw_artifact_identities_and_hashes", []))
    for value in sorted(raw):
        view.read_raw(value)
    return {"policy": DEPENDENCIES, "raw_sha256": sorted(raw), "count": len(raw),
            "identity": sha256_canonical(sorted(raw))}


def _source_summary(reader, read, proof):
    spec = fomc_spec if reader.source == "fomc" else edgar_spec
    contract = source_contract(reader.source)
    snap = read["snapshot"] or {}
    item_states = [{"event_id": r["sid"], "state": r["state"]} for r in snap.get("items", [])]
    if reader.source == "edgar":
        item_states = [{"event_id": f["source_item_id"], "state": f["state"]} for f in snap.get("filings", [])]
    health = snap.get("health")
    observations = ([instant(r.body["observed_at"]) for r in reader.view.rows("RESPONSE", upto=snap["P"])
                     if r.body.get("observed_at")] if "P" in snap else [])
    freshness = (instant(read["T"]) - max(observations)).total_seconds() if observations else None
    # Native identities bind the full source read without exposing raw bodies or exception text.
    return {"provider_id": spec.PROVIDER_ID, "provider_contract_hash": contract.identity,
            "spec_hash": spec.SPEC_HASH, "spec_revision": spec.SPEC_REVISION,
            "H": read["H"], "P": snap.get("P"), "state": read["state"], "reason": read["reason"],
            "snapshot_identity": snap.get("identity"), "policy": snap.get("policy"),
            "coverage": {"attested_bounds": read["coverage"], "complete_history": False,
                         "last_boundary_inclusive_readable": False},
            "health": health, "freshness_seconds": freshness, "item_states": item_states, "dependencies": proof,
            "discovery": snap.get("discovery"), "watchlist": snap.get("watchlist"),
            "quality": {"history": "PARTIAL", "causal_visibility": read["state"],
                        "integrity": "VERIFIED_DEPENDENCIES" if proof else "UNKNOWN"}}


def _observations(reader, P, T):
    observations = [o for o in v2.classify(reader) if o["seq"] <= P and o["available_at"] <= T]
    walls = {seq: wall for seq, _, wall in reader.view.txns(upto=P)}
    result = []
    for obs in observations:
        kind = "LINK" if reader.source == "fomc" else "FILING_OBSERVATION"
        rows = reader.view.rows_at(kind, obs["seq"])
        row = next(r for r in rows if r.body["revision"] == obs["revision"])
        record = row.body["record"]
        digest = (row.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"] if reader.source == "fomc"
                  else row.body["raw_sha256"])
        result.append({**obs, "available_at": iso(obs["available_at"]), "observed_at": row.body.get("observed_at"),
                       "ingested_at": walls[row.seq], "record": record, "raw_sha256": digest})
    return result


def _events(reader, read, observations, T, mappings):
    snap = read["snapshot"]
    view = reader.view.view(snap["P"])
    walls = {seq: wall for seq, _, wall in view.txns()}
    selected = snap["items"] if reader.source == "fomc" else snap["filings"]
    out = []
    for item in selected:
        if reader.source == "fomc" and item["state"] != "CURRENT_REVISION":
            continue
        sid = item["sid"] if reader.source == "fomc" else item["source_item_id"]
        relevant = [o for o in observations if o["event_id"] == sid]
        current = [o for o in relevant if o["revision"] == item["revision"]]
        if not current:
            continue  # no attested observation for this revision => no usable event
        latest = current[-1]
        event = {"source": reader.source, "provider_id": source_contract(reader.source).provider_id,
                 "event_id": sid, "revision": item["revision"], "available_at": latest["available_at"],
                 "first_available_at": relevant[0]["available_at"], "observation_class": latest["kind"],
                 "classification_policy": v2.CLASSIFICATION_POLICY,
                 "clocks": {"declared_publication": None, "observed_at": latest["observed_at"],
                            "ingested_at": latest["ingested_at"], "attested_available_at": latest["available_at"]},
                 "provenance": {"source_snapshot_identity": snap["identity"], "record": latest["record"],
                                "raw_sha256": latest["raw_sha256"], "observation_seq": latest["seq"]},
                 "links": [{"kind": "REVISION_OF_SAME_SOURCE_ITEM", "revision": o["revision"],
                            "evidence": {"event_id": sid, "raw_sha256": o["raw_sha256"], "seq": o["seq"]}}
                           for o in relevant if o["revision"] != item["revision"]],
                 "freshness_seconds": (T - instant(latest["available_at"])).total_seconds()}
        if reader.source == "fomc":
            normalized = item["normalized"]
            event.update(family=fomc_spec.EVENT_FAMILY, declared_publication=normalized["declared_release_at"])
            event["clocks"].update(declared_publication=normalized["declared_release_at"],
                                   declared_publication_trust=normalized["declared_release_trust_verdict"],
                                   source_updated_at=normalized["source_updated_at"])
            event["provenance"]["source_url"] = normalized["canonical_source_url"]
            created = view.rows("REVISION", key=item["revision"])[0]
            event["clocks"]["revision_created_ingested_at"] = walls[created.seq]
        else:
            fields = item["fields"]
            event.update(family=edgar_spec.EVENT_FAMILY, cik=item["cik"], form=fields["form"], items=fields["items"],
                         declared_publication=fields["acceptance_datetime_text"], listing_state=item["state"],
                         amendment_parent=item["amends"], amendment_link=item["amendment_link"])
            event["clocks"].update(declared_publication=fields["acceptance_datetime_text"],
                                   declared_publication_trust="PROVENANCE_TEXT_ONLY", filing_date=fields["filing_date"])
            event["provenance"]["source_url"] = edgar_spec.SUBMISSIONS_URL.format(cik10=item["cik"])
        understood = understand(event, mappings)
        if understood["products"]:
            out.append(understood)
    return out


class SnapshotBuilder:
    """One read-only pinned horizon per source; recreate to adopt later evidence."""
    def __init__(self, *, visibility_mode, fomc_store=None, edgar_store=None, horizons=None, prices=None, synthetic=False):
        if visibility_mode != "DURABLE_OBSERVED":
            raise ValueError("v1 supports explicit DURABLE_OBSERVED only")
        self.visibility_mode = visibility_mode
        horizons = horizons or {}
        if set(horizons) - {"fomc", "edgar"}:
            raise ValueError("horizons must name individual stores")
        self.readers = {s: SourceJoin(s, root, horizon=horizons.get(s)) for s, root in
                        (("fomc", fomc_store), ("edgar", edgar_store))}
        self.prices, self.synthetic = prices, synthetic

    def close(self):
        for reader in self.readers.values():
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def build(self, as_of, products):
        T = instant(as_of)
        mappings = sorted((product_mapping(p) for p in products), key=lambda m: m["product"])
        names = tuple(m["product"] for m in mappings)
        if not names or len(set(names)) != len(names):
            raise ValueError("products must be nonempty and unique")
        sources, histories, events, feature_rows = {}, {}, [], {p: {} for p in names}
        companies = {m["product"]: {"cik": m["cik"], "state": m["edgar"], "evidence": m["evidence"]} for m in mappings}
        for source, reader in self.readers.items():
            read = reader.read_one(T)
            observations, proof = [], None
            if read["state"] == "RESOLVED":
                try:
                    P = read["snapshot"]["P"]
                    proof = verify_dependencies(reader, P)
                    observations = _observations(reader, P, T)
                    events.extend(_events(reader, read, observations, T, mappings))
                except FAILURES as exc:
                    read = {**read, "state": "INTEGRITY_ERROR", "reason": "DEPENDENCY_READ_FAILED:" + type(exc).__name__,
                            "snapshot": None, "events": None}
                    observations, proof = [], None
            sources[source] = _source_summary(reader, read, proof)
            histories[source] = observations
            for mapping in mappings:
                state = read["state"]
                override = edgar_state(mapping) if source == "edgar" else None
                if override:
                    state = override
                elif source == "edgar" and state == "RESOLVED" and mapping["cik"] not in read["snapshot"]["watchlist"]:
                    state = "NOT_OBSERVED"
                if source == "edgar" and state == "RESOLVED":
                    first = v2.onboarding(reader, mapping["cik"])
                    if first is None or first["first_valid_read_available_at"] > T:
                        state = "NOT_OBSERVED"
                values1, values2, inventory = v1.missing_values(source), v2.missing_values(source), None
                if state == "RESOLVED":
                    obs = [o for o in observations if source == "fomc" or o["cik"] == mapping["cik"]]
                    values2, inventory = v2.values([{**o, "available_at": instant(o["available_at"])} for o in obs], T, source)
                    inventory_first = v2.onboarding(reader, mapping["cik"] if source == "edgar" else None)
                    inventory["first_valid_read_available_at"] = (iso(inventory_first["first_valid_read_available_at"])
                        if inventory_first and inventory_first["first_valid_read_available_at"] <= T else None)
                    selected = [e for e in read["events"] if source == "fomc" or e["cik"] == mapping["cik"]]
                    values1 = v1.values(selected, T, source)
                feature_rows[mapping["product"]][source] = {
                    "state": state, "source_state": read["state"], "reason": read["reason"],
                    "v1": values1, "v2": values2, "inventory": inventory, "history_complete": False}
        price_reads = {}
        for p in names:
            flags = protection_flags(p, T)
            feature_rows[p]["protection"] = flags
            if flags["inside"]:
                price_reads[p] = {"state": "PROTECTED", "reason": "DECISION_INSIDE_PROTECTED_INTERVAL", "price": None}
            elif self.prices is None:
                price_reads[p] = {"state": "NOT_CONFIGURED", "price": None}
            else:
                try:
                    price_reads[p] = self.prices.read(p, T)
                except FAILURES as exc:
                    price_reads[p] = {"state": "INTEGRITY_ERROR", "reason": "PRICE_READ_FAILED:" + type(exc).__name__, "price": None}
            if (price_reads[p].get("price") or {}).get("synthetic"):
                if not self.synthetic:
                    raise ValueError("synthetic price evidence requires a labelled synthetic snapshot")
        events.sort(key=lambda e: (e["source"], e["event_id"], e["revision"]))
        states = {s: v["state"] for s, v in sources.items()}
        return InformationSnapshot(
            as_of=iso(T), products=names, companies=companies, sources=sources, prices=price_reads,
            events=tuple(events), features={p: {**feature_rows[p], "observations": {
                s: [o for o in histories[s] if s == "fomc" or o["cik"] == companies[p]["cik"]] for s in histories}}
                for p in names},
            policies={"snapshot": POLICY, "visibility": self.visibility_mode, "minimum_causal_quality": "SERVER_ATTESTED_EVENT_PREFIX",
                      "timestamp_trust": "PROVIDER_SPEC_BOUND_V1", "market_causality": "PRICE_EVIDENCE_SELECTION_V1", "causal": "CAUSAL_AVAILABILITY_V3",
                      "dependencies": DEPENDENCIES, "event_features_v1": v1.POLICY, "event_features_v2": v2.POLICY,
                      "observation_classes": v2.CLASSIFICATION_POLICY, "mapping_hash": MAPPING_HASH,
                      "event_understanding_hash": METHOD_HASH, "prices": PRICE_POLICY},
            coverage={"state": "PARTIAL", "source_states": states, "complete_history": False,
                      "limits": ["source onboarding inventory is partial; no common source coverage is assumed",
                                 "counts include attested observations only; zero does not prove absence of news"]},
            quality={"state": "PARTIAL", "price_states": {p: r["state"] for p, r in price_reads.items()},
                     "source_states": states}, synthetic=self.synthetic)
