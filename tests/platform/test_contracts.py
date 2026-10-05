from dataclasses import FrozenInstanceError
import json

import pytest

from scripts.trading_lab.platform.contracts import (
    OUTPUTS, DatasetManifest, ExperimentManifest, InformationSnapshot, LabelRecord,
    ModelContract, PredictionRecord, ProviderContract, enrich_prediction,
)
from scripts.trading_lab.sources.canonical import sha256_canonical

HASH = "a" * 64
T = "2026-06-17T18:00:00+00:00"


def model(**kw):
    return ModelContract(model_id="synthetic-local", version="1", inputs={"features": ["close"]},
                         outputs={k: "float" if k == "return" else None for k in OUTPUTS},
                         horizons_seconds=(14400,), capabilities=("predict",), limits={"local": True},
                         implementation_version="synthetic-v1", **kw)


def prediction(**kw):
    args = dict(prediction_id="p1", model_id="synthetic-local", model_contract_hash=model().identity,
                artifact_hash=HASH, product="BTC-USD", decision_at=T, horizon_seconds=14400,
                snapshot_hash=HASH, features_hash=HASH, event_ids=(),
                outputs={k: 0.01 if k == "return" else None for k in OUTPUTS}, synthetic=True)
    args.update(kw)
    return PredictionRecord(**args)


def examples():
    yield model()
    yield prediction()
    yield ProviderContract(provider_id="synthetic", version="1", capabilities=("snapshot",),
                           **{k: {"rule": "synthetic"} for k in (
                               "identities", "formats", "clocks", "limits", "corrections",
                               "historical_availability", "health")}, evidence=({"fixture": "synthetic"},),
                           activation="WAITING_AUTHORIZATION", shape_verification="unverified")
    yield InformationSnapshot(as_of=T, products=("BTC-USD",), companies={},
                              sources={"fomc": {"H": 12, "state": "UNRESOLVED"},
                                       "edgar": {"H": 39, "state": "NOT_OBSERVED"}},
                              prices={"BTC-USD": {"state": "NOT_OBSERVED"}}, events=(),
                              features={"BTC-USD": {}}, policies={"visibility": "DURABLE_OBSERVED"}, coverage={}, quality={})
    yield DatasetManifest(dataset_id="d1", version="1", products=("BTC-USD",), decision_start=T,
                          decision_end="2026-07-01T00:00:00Z", target="return", horizon_seconds=14400,
                          snapshot_hashes=(HASH,), features_hash=HASH, exclusions=({"reason": "gap"},),
                          splits={"train": [T]}, policies={"v": 1}, counts={"accepted": 1}, synthetic=True)
    yield ExperimentManifest(experiment_id="e1", version="1", dataset_hash=HASH, model_contract_hash=HASH,
                             hypothesis={"falsify": "no advantage"}, parameters={}, splits={},
                             baselines=("zero",), decision_criteria={"metric": "MAE"}, costs={},
                             budgets={"seconds": 1}, status="PREPARED", artifacts={}, synthetic=True)
    yield LabelRecord(label_id="l1", prediction_id="p1", prediction_hash=prediction().identity,
                      product="BTC-USD", horizon_seconds=14400, realized_at="2026-06-17T22:00:00Z",
                      available_at="2026-06-18T00:00:00Z", recorded_at="2026-06-18T01:00:00Z",
                      target="return", value=0.02, provenance={"fixture": "synthetic"}, version="1")


@pytest.mark.parametrize("record", list(examples()))
def test_canonical_roundtrip_and_schema_bound_hash(record):
    payload = record.to_dict()
    restored = type(record).from_dict(json.loads(record.canonical_json()))
    assert restored == record and restored.identity == record.identity == sha256_canonical(payload)
    assert record.fingerprint == record.identity
    assert type(record).from_dict(dict(reversed(list(payload.items())))).identity == record.identity
    payload["schema"] += "-next"
    with pytest.raises(ValueError, match="schema"):
        type(record).from_dict(payload)


def test_deep_immutability_and_caller_copies():
    inputs = {"features": ["close"]}
    data = model().to_dict()
    data["inputs"] = inputs
    record = ModelContract.from_dict(data)
    before = record.identity
    inputs["features"].append("future_label")
    record.to_dict()["inputs"]["features"].append("future_label")
    assert record.identity == before and record.inputs["features"] == ("close",)
    with pytest.raises(TypeError):
        record.inputs["features"] = ()
    with pytest.raises(FrozenInstanceError):
        record.version = "2"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {3: "bad"}, object()])
def test_non_json_rejected(bad):
    with pytest.raises(ValueError):
        prediction(risk={"bad": bad})


@pytest.mark.parametrize("change", [
    {"capabilities": ["broker"]}, {"outputs": {"return": "float"}},
    {"outputs": dict.fromkeys(OUTPUTS)}, {"horizons_seconds": [0]}, {"horizons_seconds": [True]},
])
def test_invalid_model_declarations(change):
    data = model().to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        ModelContract.from_dict(data)


@pytest.mark.parametrize("change", [{"decision_at": "2026-06-17"}, {"snapshot_hash": "bad"},
                                     {"outputs": {"return": 1}}, {"horizon_seconds": -1}])
def test_invalid_prediction(change):
    with pytest.raises(ValueError):
        prediction(**change)


def test_late_label_enriches_without_rewriting_and_is_causal():
    pred = prediction()
    label = list(examples())[-1]
    before = pred.canonical_json(), pred.identity
    assert enrich_prediction(pred, (label,), as_of="2026-06-18T00:30:00Z")["label_state"] == "PENDING"
    view = enrich_prediction(pred, (label,), as_of="2026-06-18T01:00:00Z")
    assert view["labels"][0]["value"] == 0.02 and view["prediction_hash"] == before[1]
    assert (pred.canonical_json(), pred.identity) == before
    data = label.to_dict()
    data["prediction_hash"] = "b" * 64
    with pytest.raises(ValueError, match="bind"):
        enrich_prediction(pred, (LabelRecord.from_dict(data),), as_of="2026-06-19T00:00:00Z")
    data = label.to_dict()
    data["realized_at"] = T
    with pytest.raises(ValueError, match="horizon"):
        enrich_prediction(pred, (LabelRecord.from_dict(data),), as_of="2026-06-19T00:00:00Z")


def test_times_normalized_and_future_events_rejected():
    assert prediction(decision_at="2026-06-17T20:00:00+02:00").identity == prediction().identity
    data = list(examples())[3].to_dict()
    data["events"] = [{"available_at": "2026-06-17T18:00:00.000001Z"}]
    with pytest.raises(ValueError, match="future"):
        InformationSnapshot.from_dict(data)


@pytest.mark.parametrize("index,change", [
    (2, {"activation": "LIVE"}), (2, {"clocks": {}}),
    (3, {"sources": {"fomc": {"H": -1, "state": "RESOLVED"}}}),
    (3, {"sources": {"fomc": {"H": 1, "state": "PASS"}}}),
    (3, {"policies": {}}), (3, {"prices": {"BTC-USD": {"state": "RESOLVED", "price": {"available_at": "2027-01-01T00:00:00Z"}}}}),
    (4, {"decision_end": T}), (4, {"snapshot_hashes": ["bad"]}),
    (5, {"status": "PASS"}), (5, {"dataset_hash": "bad"}),
    (6, {"available_at": "2026-06-17T21:59:59Z"}), (6, {"recorded_at": "2026-06-17T23:59:59Z"}),
    (0, {"synthetic": "true"}), (0, {"model_id": ""}),
])
def test_invalid_records_rejected(index, change):
    record = list(examples())[index]
    data = record.to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        type(record).from_dict(data)
