from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from scripts.trading_lab.event_features.join import instant
from scripts.trading_lab.paper_engine import series_from_rows
from scripts.trading_lab.platform.contracts import InformationSnapshot
from scripts.trading_lab.platform.datasets import build_versioned_dataset, synthetic_dataset, verify_dataset
from scripts.trading_lab.platform.prices import MemoryPrices
from scripts.trading_lab.platform.snapshot import SnapshotBuilder


def inputs(dataset):
    product = "BTC-USD"
    bars = dataset["bars"][product]
    proofs = [deepcopy(dataset["snapshots"][e]["prices"][product]["price"])
              for e in dataset["manifest"]["snapshot_hashes"]]
    proofs.sort(key=lambda p: p["bar_open_at"])
    return series_from_rows(bars, product=product), proofs


def test_manifest_fingerprints_and_partial_source_states(dataset):
    manifest = verify_dataset(dataset)
    assert manifest.synthetic and manifest.counts["included"] == 91
    assert manifest.counts["excluded"] == 29
    assert {r["reason"] for r in manifest.exclusions} == {"PRICE_FEATURE_WARMUP_OR_GAP", "LABEL_NOT_REALIZED_OR_GAP"}
    assert manifest.policies["protection_hash"]
    for row in dataset["rows"]:
        snap = InformationSnapshot.from_dict(dataset["snapshots"][row["snapshot_hash"]])
        assert row["decision_at"] == snap.as_of
        assert "label" not in snap.to_dict()
        assert row["source_states"] == {"fomc": "NOT_CONFIGURED", "edgar": "NOT_CONFIGURED"}
        assert instant(row["label_end"]) == instant(row["decision_at"]) + timedelta(hours=4)


@pytest.mark.parametrize("component", ["label", "bar", "snapshot", "features"])
def test_all_inputs_are_bound_to_identity(dataset, component):
    payload = deepcopy(dataset)
    if component == "label":
        payload["rows"][0]["label"] = "999"
    elif component == "bar":
        payload["bars"]["BTC-USD"][0]["open"] = "999"
    elif component == "features":
        payload["rows"][0]["features"][0][1] = "999"
    else:
        payload["snapshots"][next(iter(payload["snapshots"]))]["quality"]["state"] = "UNKNOWN"
    with pytest.raises(ValueError, match="digest|identity"):
        verify_dataset(payload)


def test_late_price_dependency_excludes_even_when_current_bar_is_available(dataset):
    series, proofs = inputs(dataset)
    proofs[0]["available_at"] = "2026-06-03T00:00:00+00:00"
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", synthetic=True, prices=MemoryPrices(proofs)) as builder:
        result = build_versioned_dataset(dataset_id="late", series_by_product={"BTC-USD": series},
            evidence_by_product={"BTC-USD": proofs}, builder=builder,
            start=dataset["manifest"]["decision_start"], end=dataset["manifest"]["decision_end"])
    assert any(r["reason"] == "PRICE_DEPENDENCY_NOT_AVAILABLE_AT_DECISION" for r in result["manifest"]["exclusions"])
    assert all(r["decision_at"] >= "2026-06-03T00:00:00+00:00" for r in result["rows"])


def test_selected_unknown_event_feature_excludes_without_imputation():
    data = synthetic_dataset(products=["BTC-USD"], bars=80,
                             event_columns=[("fomc", "new_statements_attested_7d")])
    assert data["rows"] == []
    assert "EVENT_NOT_CONFIGURED" in {r["reason"] for r in data["manifest"]["exclusions"]}


def test_real_shaped_synthetic_source_features_and_events_bind_to_dataset(tmp_path, monkeypatch):
    from scripts.trading_lab.fomc import synthetic as syn
    from tests.crypto.fomc_support import Env, P1, statement_item
    monkeypatch.setattr(syn, "START", instant("2026-06-17T17:00:00Z"))
    env = Env(tmp_path)
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response(body="Synthetic statement inventory")
        env.drive(4000)
        data = synthetic_dataset(products=["BTC-USD"], start="2026-06-13T00:00:00Z", bars=120)
        series, proofs = inputs(data)
        with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", synthetic=True,
                             fomc_store=env.root, prices=MemoryPrices(proofs)) as builder:
            result = build_versioned_dataset(dataset_id="synthetic-event-join", series_by_product={"BTC-USD": series},
                evidence_by_product={"BTC-USD": proofs}, builder=builder,
                start="2026-06-17T18:00:00Z", end="2026-06-17T18:01:00Z",
                event_columns=[("fomc", "new_statements_attested_7d")])
        assert len(result["rows"]) == 1
        row = result["rows"][0]
        assert row["features"][-1] == ["fomc.new_statements_attested_7d", "0"]
        assert row["event_ids"] and row["source_states"]["fomc"] == "RESOLVED"
        snap = result["snapshots"][row["snapshot_hash"]]
        assert snap["events"][0]["observation_class"] == "INITIAL_INVENTORY"
        assert snap["sources"]["fomc"]["dependencies"]["count"] > 0
        assert verify_dataset(result).synthetic
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()


def test_event_window_after_holdout_stays_excluded():
    data = synthetic_dataset(products=["BTC-USD"], start="2026-12-01T00:00:00Z", bars=80,
                             event_columns=[("fomc", "new_statements_attested_7d")])
    assert data["rows"] == []
    assert {r["reason"] for r in data["manifest"]["exclusions"]} == {"PROTECTED_EVENT_DEPENDENCY"}


def test_holdout_checked_before_any_price_access(dataset):
    series, _ = inputs(dataset)
    class ProtectedPoint:
        bar_open_at = "2026-09-01T00:00:00Z"
        @property
        def open(self):
            raise AssertionError("protected price was accessed")
    series = replace(series, points=(ProtectedPoint(),))
    with pytest.raises(ValueError, match="PROTECTED_INPUT"):
        build_versioned_dataset(dataset_id="protected", series_by_product={"BTC-USD": series},
            evidence_by_product={"BTC-USD": []}, builder=SimpleNamespace(synthetic=True),
            start="2026-08-01T00:00:00Z", end="2026-10-01T00:00:00Z")


def test_boundary_labels_excluded_and_synthetic_does_not_bypass_protection():
    data = synthetic_dataset(products=["BTC-USD"], start="2026-08-28T00:00:00Z", bars=96)
    assert any(e["reason"] == "PROTECTED_PRICE_OR_LABEL_DEPENDENCY" for e in data["manifest"]["exclusions"])
    assert all(instant(r["label_end"]) < instant("2026-09-01T00:00:00Z") for r in data["rows"])
    with pytest.raises(ValueError, match="PROTECTED_INPUT"):
        synthetic_dataset(start="2026-09-01T00:00:00Z", bars=80)


@pytest.mark.parametrize("config", [
    {"products": ["AAPL"]}, {"products": ["BTC-USD", "BTC-USD"]}, {"bars": True},
    {"target": "unregistered"}, {"start": "2026-06-01"}, {"horizon_seconds": 0},
    {"start": "2026-06-01T00:00:01Z"}, {"bars": 601}, {"seed": -1},
])
def test_invalid_dataset_configurations(config):
    with pytest.raises((ValueError, TypeError)):
        synthetic_dataset(**config)
