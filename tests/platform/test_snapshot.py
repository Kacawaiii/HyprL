from datetime import timedelta
import hashlib
import json

import pytest

from scripts.trading_lab.capture_market_history import corpus_content_hash, manifest_content_sha256
from scripts.trading_lab.edgar import synthetic as esyn
from scripts.trading_lab.edgar.listing import source_item_id
from scripts.trading_lab.event_features import EventFeatures
from scripts.trading_lab.event_features import v2
from scripts.trading_lab.event_features.join import SourceJoin, instant
from scripts.trading_lab.platform.prices import CorpusPrices, MemoryPrices, select_prices
from scripts.trading_lab.platform.snapshot import SnapshotBuilder
from scripts.trading_lab.fomc import synthetic as fsyn
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.crypto.fomc_support import Env, P1, statement_item
from tests.crypto.test_event_features import A, ACC, AMEND, EdgarEnv


@pytest.fixture
def edgar(tmp_path):
    env = EdgarEnv(tmp_path)
    env.serve(A, esyn.filing(ACC, filed="2001-01-01", acceptance="2001-01-01T00:00:00.000Z"),
              esyn.filing(AMEND, form="8-K/A"))
    env.poll()
    env.T = env.settle()
    yield env
    env.close()


@pytest.fixture
def fomc(tmp_path):
    env = Env(tmp_path)
    env.feed([statement_item()])
    env.provider.routes[P1] = fsyn.page_response(body="Synthetic initial statement")
    env.drive(360)
    yield env
    env.collector.close()
    env.store.close()
    env.provider.close()


def price(**kw):
    args = dict(product="BTC-USD", provider_id="synthetic", bar_open_at="2026-06-17T17:00:00Z",
                bar_close_at="2026-06-17T18:00:00Z", available_at="2026-06-17T18:02:00Z",
                observed_at="2026-06-17T18:01:00Z", ingested_at="2026-06-17T18:01:30Z",
                open="100", high="102", low="99", close="101", volume="1", revision="r1",
                synthetic=True, availability_evidence="SYNTHETIC_ATTESTATION")
    args.update(kw)
    return args


def test_snapshot_states_and_independent_horizons(edgar, fomc):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=fomc.root, edgar_store=edgar.root, synthetic=True) as builder:
        s = builder.build(edgar.T, ["AAPL", "NVDA", "QQQ", "BTC-USD"])
        data = s.to_dict()
        assert data["sources"]["edgar"]["state"] == "RESOLVED"
        assert data["sources"]["fomc"]["state"] == "UNRESOLVED"
        assert data["sources"]["edgar"]["H"] != data["sources"]["fomc"]["H"]
        assert "H" not in data
        assert data["features"]["NVDA"]["edgar"]["state"] == "UNKNOWN_MAPPING"
        assert data["features"]["QQQ"]["edgar"]["state"] == "NOT_APPLICABLE"
        assert data["prices"]["BTC-USD"]["state"] == "PROTECTED"
        assert data["coverage"]["state"] == "PARTIAL" and not data["coverage"]["complete_history"]
        assert builder.build(edgar.T, ["QQQ", "BTC-USD", "NVDA", "AAPL"]).identity == s.identity
        assert all(e["products"] == ["AAPL"] for e in data["events"])
        assert s.identity == sha256_canonical(data)


def test_source_native_snapshot_identity_and_feature_v1_v2_reuse(edgar):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder, \
            SourceJoin("edgar", edgar.root) as reader, EventFeatures(edgar_store=edgar.root) as old, \
            v2.EventFeaturesV2(edgar_store=edgar.root) as new:
        snapshot = builder.build(edgar.T, ["AAPL"]).to_dict()
        src = snapshot["sources"]["edgar"]
        assert src["snapshot_identity"] == reader.read_one(edgar.T)["snapshot"]["identity"]
        features = snapshot["features"]["AAPL"]["edgar"]
        assert features["v1"] == old.rows([("AAPL", edgar.T)])[0]["sources"]["edgar"]["features"]
        assert features["v2"] == new.rows([("AAPL", edgar.T)])[0]["sources"]["edgar"]["features"]
        assert features["inventory"] == new.rows([("AAPL", edgar.T)])[0]["sources"]["edgar"]["inventory"]
        assert src["dependencies"]["count"] == len(src["dependencies"]["raw_sha256"]) > 0
        assert all(e["observation_class"] == "INITIAL_INVENTORY" for e in snapshot["events"])
        event = next(e for e in snapshot["events"] if e["event_id"] == source_item_id(ACC))
        assert event["declared_publication"].startswith("2001")
        assert instant(event["available_at"]).year == 2026
        assert event["clocks"]["observed_at"] != event["clocks"]["attested_available_at"]
        assert event["clocks"]["ingested_at"] != event["clocks"]["attested_available_at"]
        assert event["interpretation"]["inputs_hash"] == sha256_canonical(event["interpretation"]["inputs"])
        amend = next(e for e in snapshot["events"] if e["form"] == "8-K/A")
        assert amend["amendment_parent"] is None and amend["amendment_link"] == "NOT_PROVIDED_BY_SOURCE"


def test_horizon_pinning_and_revision_causality(edgar):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as fixed:
        before = fixed.build(edgar.T, ["AAPL"])
        H = before.sources["edgar"]["H"]
        edgar.serve(A, esyn.filing(ACC, items="9.01"), esyn.filing(AMEND, form="8-K/A"))
        edgar.poll()
        later = edgar.settle()
        assert fixed.build(edgar.T, ["AAPL"]).identity == before.identity
        with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, horizons={"edgar": H}, synthetic=True) as replay:
            assert replay.build(edgar.T, ["AAPL"]).identity == before.identity
        with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as fresh:
            past = fresh.build(edgar.T, ["AAPL"])
            current = fresh.build(later, ["AAPL"])
            past_event = next(e for e in past.events if e["event_id"] == source_item_id(ACC))
            changed = next(e for e in current.events if e["event_id"] == source_item_id(ACC))
            assert changed["revision"] != past_event["revision"] and changed["observation_class"] == "REVISION"
            assert changed["links"][0]["revision"] == past_event["revision"]
            assert current.features["AAPL"]["edgar"]["v2"]["accession_revisions_attested_7d"] == 1
            assert past.features["AAPL"]["edgar"]["v2"]["accession_revisions_attested_7d"] == 0


def test_intermediate_revision_raw_corruption_fails_feature_join(edgar):
    edgar.serve(A, esyn.filing(ACC, items="5.02"))
    edgar.poll()
    middle = edgar.settle()
    middle_digest = edgar.store.rows("FILING_REVISION")[-1].body["first_record"]
    digest = edgar.store.rows("RESPONSE", upto=middle_digest)[-1].body["raw_sha"]
    edgar.serve(A, esyn.filing(ACC, items="9.01"))
    edgar.poll()
    T = edgar.settle()
    # This raw is neither the initial observation nor the final current revision.
    with SourceJoin("edgar", edgar.root) as native:
        assert native.read_one(T)["state"] == "RESOLVED"
    # Mutate synthetic test evidence only, never a real closure store.
    raw_path = edgar.root / "raw" / digest[:2] / digest
    assert raw_path.exists()
    raw_path.write_bytes(b"synthetic damaged intermediate listing")
    with SourceJoin("edgar", edgar.root) as native:
        assert native.read_one(T)["state"] == "RESOLVED"  # native current read does not need the middle revision
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder:
        data = builder.build(T, ["AAPL"]).to_dict()
        assert data["sources"]["edgar"]["state"] == "INTEGRITY_ERROR"
        assert data["features"]["AAPL"]["edgar"]["state"] == "INTEGRITY_ERROR"
        assert all(v is None for v in data["features"]["AAPL"]["edgar"]["v2"].values())
        assert not data["events"]
        assert str(edgar.root) not in json.dumps(data)


def test_dependency_verification_on_every_read(edgar):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder:
        snap = builder.build(edgar.T, ["AAPL"])
        digest = snap.sources["edgar"]["dependencies"]["raw_sha256"][0]
        (edgar.root / "raw" / digest[:2] / digest).write_bytes(b"synthetic corruption after first read")
        assert builder.build(edgar.T, ["AAPL"]).sources["edgar"]["state"] == "INTEGRITY_ERROR"


def test_fomc_selected_revision_clocks_and_barrier(fomc):
    with SourceJoin("fomc", fomc.root) as reader:
        T = reader.values[-2] - timedelta(microseconds=1)
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=fomc.root, synthetic=True) as builder:
        snapshot = builder.build(T, ["AAPL"]).to_dict()
        assert snapshot["sources"]["fomc"]["state"] == "RESOLVED"
        assert len(snapshot["events"]) == 1
        event = snapshot["events"][0]
        assert event["family"] == "FOMC_MONETARY_POLICY_STATEMENT"
        assert event["market_relations"][0]["evidence_level"] == "SCOPED_MACRO_HYPOTHESIS"
        assert event["interpretation"]["uncertainty"] == 1.0
        assert "revision_created_ingested_at" in event["clocks"]
        assert builder.build("2020-01-01T00:00:00Z", ["AAPL"]).sources["fomc"]["state"] == "NOT_OBSERVED"
    fomc.provider.routes[P1] = fsyn.SyntheticResponse(body=b"<html>synthetic unnormalizable newer statement</html>",
                                                   headers=[("Content-Type", "text/html")])
    fomc.drive(600)
    with SourceJoin("fomc", fomc.root) as reader:
        T = reader.values[-2] - timedelta(microseconds=1)
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=fomc.root, synthetic=True) as builder:
        snapshot = builder.build(T, ["AAPL"]).to_dict()
        assert any(i["state"].startswith("CURRENT_CONTENT_UNAVAILABLE") for i in snapshot["sources"]["fomc"]["item_states"])
        assert snapshot["events"] == []


def test_absent_and_unwatched_sources_remain_visible(edgar, tmp_path):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder:
        data = builder.build(edgar.T, ["MSFT"]).to_dict()
        assert data["features"]["MSFT"]["edgar"]["state"] == "NOT_OBSERVED"  # watched, but never checked
        assert data["sources"]["edgar"]["health"]["0000789019"]["result_state"] == "SOURCE_NOT_CHECKED"
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=tmp_path / "missing", synthetic=True) as builder:
        data = builder.build(edgar.T, ["AAPL"]).to_dict()
        assert data["sources"]["fomc"]["state"] == "INTEGRITY_ERROR"
        assert data["sources"]["edgar"]["state"] == "NOT_CONFIGURED"
        assert str(tmp_path) not in json.dumps(data)


def test_synthetic_prices_never_use_future_revisions_or_replay_publication_times():
    prices = MemoryPrices([price(), price(revision="r2", close="102", available_at="2026-06-18T00:00:00Z")])
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", prices=prices, synthetic=True) as builder:
        assert builder.build("2026-06-17T18:01:59.999999Z", ["BTC-USD"]).prices["BTC-USD"]["state"] == "NOT_OBSERVED"
        exact = builder.build("2026-06-17T18:02:00Z", ["BTC-USD"])
        assert exact.prices["BTC-USD"]["price"]["revision"] == "r1"
        assert builder.build("2026-06-18T00:00:00Z", ["BTC-USD"]).prices["BTC-USD"]["price"]["revision"] == "r2"
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", prices=prices) as builder:
        with pytest.raises(ValueError, match="synthetic"):
            builder.build("2026-06-18T00:00:00Z", ["BTC-USD"])


@pytest.mark.parametrize("change", [{"low": "103"}, {"close": "NaN"}, {"volume": "-1"},
                                     {"available_at": "2026-06-17T17:59:00Z"}])
def test_invalid_prices_fail_closed(change):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", prices=MemoryPrices([price(**change)]), synthetic=True) as builder:
        data = builder.build("2026-06-18T00:00:00Z", ["BTC-USD"])
        assert data.prices["BTC-USD"]["state"] == "INTEGRITY_ERROR"
        assert data.prices["BTC-USD"]["price"] is None


def test_conflicting_price_revisions_fail_closed():
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", prices=MemoryPrices([price(), price(close="102", revision="r2")]), synthetic=True) as builder:
        assert builder.build("2026-06-18T00:00:00Z", ["BTC-USD"]).prices["BTC-USD"]["state"] == "INTEGRITY_ERROR"


def test_protected_decision_does_not_call_price_provider():
    class NoReads:
        def read(self, *_):
            pytest.fail("protected price data must never be opened")
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", prices=NoReads()) as builder:
        assert builder.build("2026-10-02T14:00:00Z", ["BTC-USD"]).prices["BTC-USD"]["state"] == "PROTECTED"
        assert builder.build("2026-12-15T14:00:00Z", ["AAPL"]).prices["AAPL"]["state"] == "PROTECTED"


def write_corpus(tmp_path, *, last="2026-06-17T17:00:00Z", acquired="2026-06-18T00:00:00Z"):
    root = tmp_path / "coinbase_history_v1"
    path = root / "BTC-USD" / "canonical.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in price(bar_open_at=last).items() if k in ("bar_open_at", "open", "high", "low", "close", "volume")}
    payload = (json.dumps(row) + "\n").encode()
    path.write_bytes(payload)
    manifest = {"spec": {"provider": "synthetic", "timeframe": "1h"}, "capture_completed_at": acquired,
                "corpus_content_hash": "a" * 64, "products": [{"product": "BTC-USD", "first_open": last,
                "last_open": last, "canonical_path": "BTC-USD/canonical.jsonl", "canonical_rows": 1,
                "canonical_sha256": hashlib.sha256(payload).hexdigest(), "missing_count": 0, "batches": []}]}
    manifest["corpus_content_hash"] = corpus_content_hash(manifest["products"])
    manifest["manifest_content_sha256"] = manifest_content_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_corpus_capture_time_is_real_gate_and_missing_ingestion_stays_unknown(tmp_path):
    write_corpus(tmp_path)
    reader = CorpusPrices(tmp_path)
    assert reader.read("BTC-USD", instant("2026-06-17T23:59:59Z"))["state"] == "NOT_OBSERVED"
    data = reader.read("BTC-USD", instant("2026-06-18T00:00:00Z"))
    assert data["state"] == "RESOLVED"
    assert data["price"]["available_at"] == "2026-06-18T00:00:00+00:00"
    assert data["price"]["ingested_at"] is None
    assert data["quality"]["availability_evidence"] == "CORPUS_LOCAL_CAPTURE_COMPLETION_V1"
    assert "not server attestation" in data["price"]["limits"][0]


def test_corpus_integrity_and_holdout_before_candle_read(tmp_path):
    path = write_corpus(tmp_path)
    path.write_bytes(b"synthetic damaged canonical file")
    reader = CorpusPrices(tmp_path)
    assert reader.read("BTC-USD", instant("2026-06-19T00:00:00Z"))["state"] == "INTEGRITY_ERROR"
    write_corpus(tmp_path, last="2026-10-01T00:00:00Z")
    path.unlink()
    assert reader.read("BTC-USD", instant("2026-12-02T00:00:00Z"))["state"] == "PROTECTED"


def test_core_mode_is_explicit():
    with pytest.raises(TypeError):
        SnapshotBuilder()
    with pytest.raises(ValueError, match="DURABLE_OBSERVED"):
        SnapshotBuilder(visibility_mode="RETROSPECTIVE_SOURCE")


def test_missing_issuer_observations_keep_null_features(edgar):
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder:
        snapshot = builder.build(edgar.T, ["MSFT"])
        src = snapshot.features["MSFT"]["edgar"]
        assert src["state"] == "NOT_OBSERVED" and src["source_state"] == "RESOLVED"
        assert all(v is None for v in src["v1"].values()) and all(v is None for v in src["v2"].values())


def test_snapshot_reads_are_offline_and_stores_are_read_only(edgar, monkeypatch):
    import socket
    def refuse(*args, **kwargs):
        pytest.fail("snapshot tried to contact a network source")
    monkeypatch.setattr(socket, "create_connection", refuse)
    before = edgar.store.horizon()
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=edgar.root, synthetic=True) as builder:
        assert builder.readers["edgar"].store.read_only
        assert builder.build(edgar.T, ["AAPL"]).sources["edgar"]["state"] == "RESOLVED"
    assert edgar.store.horizon() == before


def test_price_manifest_clock_tampering_fails_closed(tmp_path):
    write_corpus(tmp_path)
    path = tmp_path / "coinbase_history_v1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["capture_completed_at"] = "2026-06-17T18:00:00Z"
    path.write_text(json.dumps(manifest))
    assert CorpusPrices(tmp_path).read("BTC-USD", instant("2026-06-18T00:00:00Z"))["state"] == "INTEGRITY_ERROR"


def test_archived_demo_evidence_is_public_metadata_only():
    from pathlib import Path
    evidence = json.loads((Path(__file__).resolve().parents[2] / "docs/artifacts/information_snapshot_v1_evidence.json").read_text())
    assert evidence["read_only"] and evidence["network_requests"] == 0 and evidence["reproducible"]
    rows = evidence["snapshots"]
    assert [r["event_counts"] for r in rows] == [
        {"fomc": 0, "edgar": 0}, {"fomc": 6, "edgar": 0}, {"fomc": 0, "edgar": 104}]
    assert all(not r["synthetic"] and r["coverage_state"] == "PARTIAL" for r in rows)
    assert all(r["sources"]["fomc"]["H"] == 622 and r["sources"]["edgar"]["H"] == 26 for r in rows)
    assert all(len(r["fingerprint"]) == 64 for r in rows)
    assert not any(value in json.dumps(evidence) for value in ("/home/", "/srv/", "/tmp/", "body.html"))
