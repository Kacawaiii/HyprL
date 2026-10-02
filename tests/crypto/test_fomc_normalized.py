"""normalized_minimum_fields: the 27 fields, split between immutable properties of the (source item,
content hash) revision and what belongs to each observation-to-revision link; nulls where the spec
prescribes them and no reconstructed historical time. Backfill -> LIVE, content correction and A-B-A."""

from __future__ import annotations

import json

import pytest

from scripts.trading_lab.fomc import snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.clock import parse_iso as t

from tests.crypto.fomc_support import P1, SID1, Env, statement_item

BODY_A = "The Committee decided to maintain the target range."
BODY_B = "The Committee decided to lower the target range."


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def _item(env):
    env.drive(130, idle=30)  # a newer verified response resolves T = now
    snap = snapshot.events_as_of(env.store, env.clock.true)
    assert snap["read_state"] == "FOMC_RESOLVED"
    return next(i for i in snap["items"] if i["sid"] == SID1)


def _revisions(env):
    return {r.key: r.body for r in env.store.rows("REVISION")}


def _record(env, seq):
    return env.store.row_at("RESPONSE", seq).body


def test_the_27_fields_split_between_the_revision_and_each_observation(env):
    names = json.loads(spec.SPEC_PATH.read_text(encoding="utf-8"))["normalized_minimum_fields"]
    keys = {spec.NORMALIZED_KEYS.get(n, n) for n in names}
    assert len(names) == len(keys) == 27
    assert set(spec.REVISION_FIELDS) | set(spec.OBSERVATION_FIELDS) == keys
    assert set(spec.REVISION_FIELDS) & set(spec.OBSERVATION_FIELDS) == {"observation_mode", "ingested_at"}
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(240)
    item = _item(env)
    revision, links = item["normalized"], item["links"]
    assert set(spec.REVISION_FIELDS) <= set(revision) and all(set(spec.OBSERVATION_FIELDS) <= set(l) for l in links)
    assert revision["content_source_available_at"] is None and revision["source_updated_at"] is None  # always null in V1
    assert revision["declared_release_at"] == "2026-06-17T18:00:00+00:00"  # "2:00 p.m. EDT" on June 17, 2026
    assert (revision["timestamp_semantics"], revision["declared_release_trust_verdict"], revision["timestamp_trust_verdict"]) == \
        ("EXACT_INSTANT", "TRUSTED_EXACT", "UNTRUSTED")  # a trusted claim about the event, never content vintage
    assert (revision["provider_class"], revision["source_tier"], revision["taxonomy_type"], revision["taxonomy_version"]) == \
        ("CENTRAL_BANK_GOVERNMENT", "TIER_1_OFFICIAL", "CENTRAL_BANK", "trading-lab.event-taxonomy.v1")
    assert (revision["capture_spec_id"], revision["capture_scope_id"], revision["capture_spec_hash"]) == \
        ("federal_reserve_fomc_capture_v1", "standard_fomc_statement_release_pattern_v1", spec.SPEC_HASH)
    assert revision["revision_id"] == item["revision"] and revision["canonical_source_url"] == syn.url(P1)
    assert revision["official_statement_date"] == "2026-06-17" and revision["classification_state"] == "IN_SCOPE_V1"
    link = links[0]
    record = _record(env, link["record"])
    assert link["observation_mode"] == "LIVE" and link["observed_at"] == record["observed_at"]  # the receipt instant
    assert t(link["ingested_at"]) >= t(link["observed_at"]) and link["rss_guid_if_available"] == "g1"
    assert link["source_observation_id"] == spec.sha256_canonical(["SourceObservation", spec.PROVIDER_ID, SID1, revision["content_hash"]])
    assert link["raw_artifact_identities_and_hashes"] == [{"record": link["record"], "raw_sha256": revision["first_raw_sha256"],
                                                           "byte_length": record["byte_length"], "request_url": syn.url(P1),
                                                           "final_url": syn.url(P1), "redirect_chain": [syn.url(P1)]}]
    assert revision["ingested_at"] == link["ingested_at"]  # the revision was created by this observation's transaction


def test_immediate_release_and_unparsed_release_are_null_not_guessed(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(release="For immediate release")
    env.drive(240)
    revision = _item(env)["normalized"]
    assert revision["declared_release_at"] is None and revision["declared_release_text"] == "For immediate release"
    assert (revision["timestamp_semantics"], revision["declared_release_trust_verdict"]) == ("UNKNOWN", "UNKNOWN")
    assert revision["content_source_available_at"] is None


def test_backfill_then_live_keeps_one_immutable_revision_and_per_observation_facts(env):
    env.feed([])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.collector.submit_manifest(json.dumps({"version": 1, "urls": [syn.url(P1)]}).encode(), "operator")
    env.drive(240)
    before = _revisions(env)
    backfill = _item(env)
    env.feed([statement_item()])
    env.drive(300)
    live = _item(env)
    assert _revisions(env) == before and len(before) == 1  # the revision row is never rewritten
    assert live["normalized"]["observation_mode"] == "HISTORICAL_BACKFILL"  # creation provenance only
    links = {l["observation_mode"]: l for l in live["links"]}
    hb, lv = links["HISTORICAL_BACKFILL"], links["LIVE"]
    assert hb == backfill["links"][0]  # an observation's facts are fixed at its commit
    assert hb["observed_at"] == _record(env, hb["record"])["wall_at_receipt"]  # the actual backfill collection instant
    assert hb["observed_at"] != live["normalized"]["declared_release_at"]  # never reconstructed from the release time
    assert hb["rss_guid_if_available"] is None and lv["rss_guid_if_available"] == "g1"  # no GUID was listed before the backfill
    assert t(lv["observed_at"]) > t(hb["observed_at"]) and t(lv["ingested_at"]) > t(hb["ingested_at"])
    assert hb["source_observation_id"] == lv["source_observation_id"]  # content-derived: it may repeat
    assert hb["record"] != lv["record"]  # but each observation keeps its own per-response record


def test_content_correction_appends_a_revision_and_keeps_the_first_intact(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(240)
    first = _item(env)
    before = _revisions(env)
    env.provider.routes[P1] = syn.page_response(body=BODY_B, date_text="June 18, 2026")  # corrected body and date
    env.drive(900, idle=60)
    second = _item(env)
    after = _revisions(env)
    assert len(after) == 2 and after[first["revision"]] == before[first["revision"]]  # D1 retained, no overwrite
    assert second["revision"] != first["revision"] and second["normalized"]["official_statement_date"] == "2026-06-18"
    assert first["normalized"]["official_statement_date"] == "2026-06-17"
    assert second["normalized"]["source_item_id"] == first["normalized"]["source_item_id"] == SID1  # same logical item
    assert second["normalized"]["canonical_source_url"] == first["normalized"]["canonical_source_url"]


def test_aba_reuses_the_first_revision_with_a_link_per_observation(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(240)
    a = _item(env)
    env.provider.routes[P1] = syn.page_response(body=BODY_B)
    env.drive(900, idle=60)
    env.provider.routes[P1] = syn.page_response(body=BODY_A)
    env.drive(3600, idle=300)
    aba = _item(env)
    revisions = _revisions(env)
    assert len(revisions) == 2 and aba["revision"] == a["revision"] and aba["step"] == 4
    assert revisions[a["revision"]] == {k: v for k, v in a["normalized"].items() if k != "ingested_at"}  # unchanged row
    assert aba["normalized"] == a["normalized"]  # same immutable properties and creation provenance
    records = [l["record"] for l in aba["links"]]
    assert len(records) == 2 and records[0] == a["links"][0]["record"]  # the second A observation has its own link
    assert t(aba["links"][1]["observed_at"]) > t(aba["links"][0]["observed_at"])
    assert all(l["revision"] == a["revision"] for l in aba["links"])
    assert state.primary_responses(env.store, SID1)[-1].seq == records[1]
