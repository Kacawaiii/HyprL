from copy import deepcopy
from decimal import Decimal

import pytest

from scripts.trading_lab.platform.contracts import ExperimentManifest, PredictionRecord, OUTPUTS
from scripts.trading_lab.platform.experiments import prepare_experiment, temporal_splits
from scripts.trading_lab.platform.jobs import JobRunner
from scripts.trading_lab.platform.model_lab_demo import wait
from scripts.trading_lab.sources.canonical import sha256_canonical


@pytest.mark.parametrize("model_id", ["synthetic-ridge-v1", "local-momentum-v1"])
def test_full_isolated_experiment_reproduces_bit_for_bit_and_keeps_baselines(tmp_path, dataset, model_id):
    with JobRunner(tmp_path) as runner:
        runner.store.put_artifact("dataset", dataset, identity=dataset["fingerprint"])
        prepared = prepare_experiment(dataset, model_id=model_id)
        assert prepared.status == "PREPARED" and prepared.artifacts == {}
        jobs = [runner.store.submit("experiment", {"prepared": prepared.to_dict()}) for _ in range(2)]
        results = [wait(runner.store, identifier, 60) for identifier in jobs]
        assert results[0] == results[1]
        assert runner.store.status(jobs[0])["result_hash"] == runner.store.status(jobs[1])["result_hash"]
        result = results[0]
        complete = ExperimentManifest.from_dict(result["manifest"])
        assert complete.status == "COMPLETE" and complete.identity == result["fingerprint"]
        assert complete.artifacts["prepared_hash"] == prepared.identity
        assert result["synthetic"] and result["shadow"]["synthetic"]
        assert result["shadow"]["chain"]["verified"] is True
        assert result["shadow"]["events"] > 100 and result["shadow"]["broker_connected"] is False
        assert result["backtests"]["BTC-USD"]["live_execution"] is False
        from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
        assert result["backtests"]["BTC-USD"]["execution_spec"] == EXECUTION_SPEC_V1.canonical()
        selected, _ = temporal_splits(dataset["rows"], horizon_seconds=14400)
        if model_id == "synthetic-ridge-v1":
            # Every scaler mean equals TRAIN only, not the complete dataset.
            means = result["models"]["BTC-USD"]["feature_means"]
            for i, value in enumerate(means):
                from decimal import localcontext
                with localcontext() as context:
                    context.prec = 34
                    expected = sum(Decimal(r["features"][i][1]) for r in selected["train"]) / Decimal(len(selected["train"]))
                assert Decimal(value) == expected
        for role in ("validation", "test"):
            assert set(result["metrics"]["BTC-USD"][role]) == {"model", "ZERO", "TRAIN_MEAN"}
        if model_id == "local-momentum-v1":
            assert result["criteria_met"]["BTC-USD"] is False
        for pred in result["predictions"]:
            record = PredictionRecord.from_dict(pred["record"])
            assert record.identity == pred["fingerprint"]
            assert record.snapshot_hash in dataset["snapshots"]
            assert record.artifact_hash == complete.artifacts["model_hashes"][record.product]
            assert all(record.outputs[k] is None for k in OUTPUTS if k != "return")
            assert record.uncertainty is None


def test_temporal_purge_uses_label_availability_and_embargo(dataset):
    rows = deepcopy(dataset["rows"])
    selected, description = temporal_splits(rows, horizon_seconds=14400, embargo_seconds=7200)
    for role, boundary in (("train", description["validation_start"]), ("validation", description["test_start"])):
        assert all(r["label_end"] < boundary and r["label_available_at"] < boundary for r in selected[role])
    assert {e["reason"] for e in description["exclusions"]} == {"PURGE_LABEL_REACHES_NEXT_BLOCK", "TEMPORAL_EMBARGO"}
    late = rows[0]
    late["label_available_at"] = description["validation_start"]
    selected2, _ = temporal_splits(rows, horizon_seconds=14400)
    assert late not in selected2["train"]


def test_unimplemented_criterion_is_refused_before_execution(dataset):
    with pytest.raises(ValueError, match="decision criterion"):
        prepare_experiment(dataset, decision_criteria={"metric": "profit", "rule": "positive"})


@pytest.mark.parametrize("mutation", ["real", "horizon", "features", "frozen"])
def test_unsupported_experiment_refused_before_training(dataset, mutation):
    data = deepcopy(dataset)
    model = "synthetic-ridge-v1"
    if mutation == "frozen":
        model = "paper-ridge-v2"
    else:
        if mutation == "real":
            data["manifest"]["synthetic"] = False
        elif mutation == "horizon":
            data["manifest"]["horizon_seconds"] = 3600
        else:
            data["manifest"]["policies"]["columns"] = ["unsupported"]
        data["fingerprint"] = sha256_canonical(data["manifest"])
    with pytest.raises(ValueError):
        prepare_experiment(data, model_id=model)
