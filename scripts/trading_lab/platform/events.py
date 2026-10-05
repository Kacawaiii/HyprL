"""Evidence-bound rule interpretations; no sentiment model or causal market claim."""
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.event_features.mapping import MAPPING_HASH

METHOD = {
    "id": "EVENT_UNDERSTANDING_RULES_V1", "version": "1",
    "importance": {"FOMC_MONETARY_POLICY_STATEMENT": 3, "SEC_2.02_OR_5.02": 2, "OTHER_SEC_METADATA": 1},
    "relevance": {"verified_issuer": 1.0, "scoped_macro_hypothesis": 0.25},
    "uncertainty": "economic impact unknown (1.0); no price-response or calibration evidence",
    "limits": ["ordinal importance is a rule priority, not measured market impact",
               "metadata only; no statement/filing text interpretation",
               "inventory is onboarding, not new publication", "no causal event-to-return evidence"],
}
METHOD_HASH = sha256_canonical(METHOD)


def understand(event: dict, mappings: list[dict]) -> dict:
    source = event["source"]
    relations = []
    for mapping in mappings:
        if source == "fomc":
            relations.append({"product": mapping["product"], "evidence_level": "SCOPED_MACRO_HYPOTHESIS",
                              "evidence": {"file": "docs/OFFICIAL_EVENT_SOURCES.md", "scope": "FOMC macro relevance"},
                              "relevance": METHOD["relevance"]["scoped_macro_hypothesis"]})
        elif mapping["edgar"] == "VERIFIED" and mapping["cik"] == event.get("cik"):
            relations.append({"product": mapping["product"], "evidence_level": "VERIFIED_ENTITY_LINK",
                              "evidence": mapping["evidence"], "relevance": METHOD["relevance"]["verified_issuer"]})
    items = {i.strip() for i in (event.get("items") or "").split(",")}
    importance = 3 if source == "fomc" else 2 if items & {"2.02", "5.02"} else 1
    facts = {k: event[k] for k in ("event_id", "revision", "family", "declared_publication", "clocks", "provenance")}
    facts.update({k: event[k] for k in ("cik", "form", "items", "observation_class", "classification_policy") if k in event})
    inputs = {"facts": facts, "mapping_hash": MAPPING_HASH, "relations": relations}
    return {**event, "facts": facts,
            "entities": [{"kind": "institution", "id": "FOMC"}] if source == "fomc" else
                        [{"kind": "company", "id": event["cik"], "identity_scheme": "SEC_CIK"}],
            "products": [r["product"] for r in relations], "market_relations": relations,
            "dedup": {"method": "EXACT_PROVIDER_ITEM_REVISION_V1",
                      "key": [event["provider_id"], event["event_id"], event["revision"]],
                      "cross_source_cluster": None, "evidence": event["provenance"]},
            "interpretation": {"method": METHOD["id"], "version": METHOD["version"], "method_hash": METHOD_HASH,
                               "inputs": inputs, "inputs_hash": sha256_canonical(inputs), "importance": importance,
                               "importance_scale": "ordinal 1..3", "uncertainty": 1.0,
                               "uncertainty_meaning": METHOD["uncertainty"], "limits": METHOD["limits"]}}
