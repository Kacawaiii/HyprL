"""COMPARISON_PROTOCOL_V2: crypto primary with the V1 criteria unchanged, equities exploratory, V1 kept."""

import json
from pathlib import Path

from scripts.trading_lab import comparison_protocol as cp
from scripts.trading_lab.sources.canonical import sha256_canonical

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "docs/artifacts/comparison_protocol_v1.json"
V2 = REPO / "docs/artifacts/comparison_protocol_v2.json"
DOC = REPO / "docs/COMPARISON_PROTOCOL_V2.md"


def load(path):
    return json.loads(path.read_text())


def test_v2_artifact_is_generated_canonical_and_cited():
    data = load(V2)
    assert data == json.loads(json.dumps(cp.build_v2()))
    assert sha256_canonical({k: v for k, v in data.items() if k != "protocol_hash"}) == data["protocol_hash"]
    assert data["schema"] == "comparison-protocol-v2" and data["revision"] == 2
    assert data["results_exist"] is False and data["status"] == "PREREGISTERED_BEFORE_ANY_RESULT"
    assert data["protocol_hash"] in DOC.read_text() and data["protocol_hash"] != load(V1)["protocol_hash"]


def test_v1_is_kept_and_superseded_by_its_hash():
    v1, v2 = load(V1), load(V2)
    assert v1 == json.loads(json.dumps(cp.build()))
    assert v1["protocol_hash"].startswith("6d8d7829") and v1["revision"] == 1
    assert v2["supersedes"]["protocol_hash"] == v1["protocol_hash"] and v2["supersedes"]["revision"] == 1


def test_crypto_is_primary_with_every_v1_criterion_unchanged():
    p1, p2 = load(V1)["evaluation"]["primary"], load(V2)["evaluation"]["primary"]
    assert load(V2)["objective"]["primary"] == "CRYPTO"
    assert p2["families"] == {"CRYPTO": p1["families"]["CRYPTO"]}
    assert p2["metric"] == p1["metric"] and p2["label"]["crypto"] == p1["label"]["crypto"]
    assert p2["decision_rule"]["SUPPORTED"] == p1["decision_rule"]["SUPPORTED"]
    assert p2["decision_rule"]["NOT_SUPPORTED"] == p1["decision_rule"]["NOT_SUPPORTED"]
    assert p2["decision_rule"]["INCONCLUSIVE_INSUFFICIENT_SAMPLE"] == p1["decision_rule"]["INCONCLUSIVE_INSUFFICIENT_SAMPLE"]
    u1, u2 = p1["uncertainty"], p2["uncertainty"]
    for key in ("method", "resamples", "seed", "interval", "family_alpha_one_sided"):
        assert u2[key] == u1[key], key
    assert u2["block_decisions"]["crypto"] == u1["block_decisions"]["crypto"] == 24
    m1, m2 = p1["minimum_sample"], p2["minimum_sample"]
    assert m2["blocks_per_product"] == m1["blocks_per_product"]
    assert m2["paired_test_decisions_per_product"]["crypto"] == m1["paired_test_decisions_per_product"]["crypto"] == 240
    assert m2["crypto_fomc_statements_newly_observed_in_test"] == m1["crypto_fomc_statements_newly_observed_in_test"] == 2


def test_equities_are_exploratory_without_any_verdict_or_alpha():
    v2 = load(V2)
    primary = v2["evaluation"]["primary"]
    assert "EQUITY" not in primary["families"] and "equity" not in primary["label"]
    assert "equity" not in primary["minimum_sample"]["paired_test_decisions_per_product"]
    assert "equity_edgar_newly_observed_accessions_in_test" not in primary["minimum_sample"]
    assert primary["uncertainty"]["families_tested"] == 1
    exploratory = v2["evaluation"]["exploratory"]
    assert exploratory["family"] == "EQUITY" and exploratory["status"] == "EXPLORATORY_NO_CLAIM"
    assert exploratory["verdict"].startswith("none") and exploratory["effect_on_primary"].startswith("none")
    assert v2["products"]["roles"] == {"crypto": "PRIMARY_CONFIRMATORY", "equity": "EXPLORATORY_NO_CLAIM"}


def test_everything_outside_the_roles_is_unchanged():
    v1, v2 = load(V1), load(V2)
    for key in ("calendar", "protection", "warmup", "pairing", "admissible_decision", "requirements",
                "frozen_sources", "no_action", "variants", "changes_need"):
        assert v2[key] == v1[key], key
    for key in ("stopping", "exclusions_reported"):
        assert v2["evaluation"][key] == v1["evaluation"][key], key
    assert {k: v for k, v in v2["evaluation"]["secondary"].items() if k != "scope"} == v1["evaluation"]["secondary"]
    assert {k: v for k, v in v2["products"].items() if k != "roles"} == v1["products"]
