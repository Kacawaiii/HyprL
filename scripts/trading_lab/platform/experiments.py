"""Deterministic synthetic experiments reusing Ridge, scoring and execution engines."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import timedelta
from decimal import Decimal, localcontext
from importlib.metadata import version
from pathlib import Path
import platform
from types import SimpleNamespace

from scripts.trading_lab.event_features.join import instant
from scripts.trading_lab.platform.adapters import ModelRegistry
from scripts.trading_lab.platform.contracts import ExperimentManifest, PredictionRecord, OUTPUTS
from scripts.trading_lab.platform.datasets import synthetic_dataset, verify_dataset
from scripts.trading_lab.platform.jobs import ResourceLimits
from scripts.trading_lab.sources.canonical import sha256_canonical

DECISION_CRITERIA_V1 = {"metric": "test_mae", "rule": "strictly below ZERO and TRAIN_MEAN", "commercial_claim": False}


def temporal_splits(rows, *, horizon_seconds, embargo_seconds=3600):
    if type(embargo_seconds) is not int or not 0 <= embargo_seconds <= 86400:
        raise ValueError("embargo must be 0..86400 seconds")
    times = sorted({r["decision_at"] for r in rows})
    if len(times) < 24:
        raise ValueError("insufficient admissible temporal population")
    validation_start, test_start = times[len(times) // 2], times[3 * len(times) // 4]
    selected = {"train": [], "validation": [], "test": []}
    excluded = []
    for row in rows:
        at = row["decision_at"]
        label_end = max(instant(row["label_end"]), instant(row["label_available_at"]))
        role = "train" if at < validation_start else "validation" if at < test_start else "test"
        boundary = validation_start if role == "train" else test_start
        if role != "test" and label_end >= instant(boundary):
            excluded.append({"product": row["product"], "decision_at": at, "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"})
        elif role != "train" and instant(at) < instant(validation_start if role == "validation" else test_start) + timedelta(seconds=embargo_seconds):
            excluded.append({"product": row["product"], "decision_at": at, "reason": "TEMPORAL_EMBARGO"})
        else:
            selected[role].append(row)
    products = sorted({r["product"] for r in rows})
    if any(sum(r["product"] == p for r in selected[role]) < 2 for p in products for role in selected):
        raise ValueError("insufficient rows after purge/embargo")
    description = {"method": "temporal-purge-embargo-v1", "purge_seconds": horizon_seconds,
                   "embargo_seconds": embargo_seconds, "validation_start": validation_start, "test_start": test_start,
                   "train_label_available_before": validation_start, "validation_label_available_before": test_start,
                   "roles": {role: {"first": block[0]["decision_at"], "last": block[-1]["decision_at"],
                       "rows": len(block), "population_hash": sha256_canonical(
                           [[r["product"], r["decision_at"], r["features_hash"]] for r in block])}
                       for role, block in selected.items()}, "exclusions": excluded,
                   "refit_after_validation": False}
    return selected, description


def runtime_identity():
    root = Path(__file__).resolve().parents[1]
    # Public source digests, including reused engines, bind the exact implementation.
    modules = ("platform/contracts.py", "platform/datasets.py", "platform/adapters.py", "platform/local_momentum.py",
               "platform/experiments.py", "platform/jobs.py", "platform/snapshot.py", "platform/prices.py",
               "models.py", "market_dataset.py", "market_indicators.py", "walk_forward.py", "paper_model.py",
               "paper_engine.py", "paper_event_store.py", "economic_backtest.py", "signal_engine.py", "risk_engine.py",
               "research_protection.py", "event_features/v2.py")
    import hashlib
    return {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scikit-learn", "xgboost")},
            "decimal_precision": 34, "blas_threads": 1, "implementation": "model-lab-experiment-v1",
            "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in modules}}


def prepare_experiment(dataset, *, model_id="synthetic-ridge-v1", embargo_seconds=3600,
                       limits=None, hypothesis=None, decision_criteria=None):
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    if decision_criteria is not None and decision_criteria != DECISION_CRITERIA_V1:
        raise ValueError("unsupported decision criterion; requires another experiment revision")
    manifest = verify_dataset(dataset)
    adapter = ModelRegistry().create(model_id)
    if manifest.synthetic is not True or adapter.contract.synthetic is not True or "train" not in adapter.contract.capabilities:
        raise ValueError("new experiments require generated synthetic datasets and train-capable demo adapters")
    if manifest.horizon_seconds not in adapter.contract.horizons_seconds:
        raise ValueError("model does not support dataset horizon")
    if tuple(manifest.policies["columns"]) != tuple(adapter.contract.inputs["features"]):
        raise ValueError("model does not support dataset feature schema")
    if len(dataset["rows"]) > adapter.contract.limits["max_rows"]:
        raise ValueError("model row budget exceeded")
    _, splits = temporal_splits(dataset["rows"], horizon_seconds=manifest.horizon_seconds, embargo_seconds=embargo_seconds)
    parameters = {"model_id": model_id, "alpha": "1.0" if model_id == "synthetic-ridge-v1" else None,
                  "runtime": runtime_identity(), "transformations": "training only; no validation refit"}
    config = {"dataset_hash": manifest.identity, "model_contract_hash": adapter.contract.identity,
              "parameters": parameters, "splits": splits,
              "hypothesis": hypothesis or {"statement": "Synthetic infrastructure demonstration",
                  "mechanism": "price features may predict synthetic four-hour returns",
                  "falsification": "test MAE fails to beat both fixed baselines", "scientific_claim": False},
              "decision_criteria": DECISION_CRITERIA_V1,
              "budgets": asdict(limits or ResourceLimits())}
    return ExperimentManifest(experiment_id="synthetic-exp-" + sha256_canonical(config)[:24],
        version="model-lab-experiment-v1", dataset_hash=manifest.identity,
        model_contract_hash=adapter.contract.identity, hypothesis=config["hypothesis"],
        parameters=parameters, splits=splits, baselines=("ZERO", "TRAIN_MEAN"),
        decision_criteria=config["decision_criteria"],
        costs={"backtest": EXECUTION_SPEC_V1.canonical(), "backtest_hash": EXECUTION_SPEC_V1.execution_spec_hash,
               "paper": PAPER_EXECUTION_SPEC_V1.canonical(), "paper_hash": PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash,
               "signal_hash": SIGNAL_SPEC_V1.spec_hash, "risk_hash": RISK_SPEC_V1.risk_spec_hash,
               "calendar": manifest.policies["calendar"]},
        budgets=config["budgets"], status="PREPARED", artifacts={}, synthetic=True)


def model_rows(rows, *, labels=False):
    return tuple(SimpleNamespace(bar_open_at=r["bar_open_at"], usable=True,
        features=tuple((c, Decimal(v)) for c, v in r["features"]),
        **({"label": Decimal(r["label"])} if labels else {})) for r in rows)


def metrics(predictions, rows):
    from scripts.trading_lab.walk_forward import mean_absolute_error, root_mean_squared_error, rank_ic
    actuals = tuple(Decimal(r["label"]) for r in rows)
    return {"count": len(rows), "mae": str(mean_absolute_error(predictions, actuals)),
            "rmse": str(root_mean_squared_error(predictions, actuals)),
            "rank_ic": None if (ic := rank_ic(predictions, actuals)) is None else str(ic)}


def _backtest(dataset, rows, predictions, *, product, prepared, artifact_hash):
    from scripts.trading_lab.economic_backtest import (
        EXECUTION_SPEC_V1, EconomicBacktestSpec, build_position_targets, run_economic_backtest,
        ECONOMIC_BACKTEST_SCHEMA_VERSION, ECONOMIC_RESULT_SCHEMA_VERSION)
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
    stream = [{"bar_open_at": r["bar_open_at"], "prediction": str(p)} for r, p in zip(rows, predictions)]
    signals, targets = build_position_targets(stream, model_spec_hash=prepared.model_contract_hash,
        fitted_hash=artifact_hash, benchmark_spec_hash=prepared.identity)
    # Full test interval including the label tail; never training/validation bars.
    bars = [b for b in dataset["bars"][product] if instant(b["bar_open_at"]) >= instant(rows[0]["bar_open_at"])]
    spec = EconomicBacktestSpec(protocol_version=ECONOMIC_BACKTEST_SCHEMA_VERSION, product=product, timeframe="1h",
        source_benchmark_protocol=prepared.version, source_benchmark_spec_hash=prepared.identity,
        source_benchmark_results_hash=sha256_canonical(stream), signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
        risk_spec_hash=RISK_SPEC_V1.risk_spec_hash, execution_spec_hash=EXECUTION_SPEC_V1.execution_spec_hash,
        market_corpus_spec_hash=sha256_canonical(dataset["manifest"]["policies"]),
        market_corpus_content_hash=dataset["manifest"]["features_hash"], result_schema_version=ECONOMIC_RESULT_SCHEMA_VERSION)
    result = run_economic_backtest(product=product, targets=targets, bars=bars,
        bars_product=product, spec=spec, signal_series_hash=signals.series_hash)
    return result.canonical()


class _PaperBridge:
    def __init__(self, adapter, artifact_hash):
        self.adapter = adapter
        self.fitted = SimpleNamespace(fitted_model_hash=artifact_hash)
        self.model_spec_hash = adapter.contract.identity

    def predict(self, rows):
        return self.adapter.predict(rows)


def _shadow(store, identifier, dataset, selected, adapters, hashes, prepared):
    from scripts.trading_lab.paper_engine import PaperEngine, PaperSessionSpec, PAPER_EXECUTION_SPEC_V1
    from scripts.trading_lab.paper_event_store import PaperEventStore
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
    products = tuple(dataset["manifest"]["products"])
    models = {p: _PaperBridge(adapters[p], hashes[p]) for p in products}
    spec = PaperSessionSpec(products=products, timeframe="1h", paper_model_spec_hash=prepared.identity,
        model_fitted_hashes=tuple(sorted(hashes.items())), signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
        risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
        paper_execution_spec_hash=PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash,
        protected_holdout_hash=PROTECTED_WINDOW_V1.holdout_hash,
        initial_equity=str(PAPER_EXECUTION_SPEC_V1.initial_equity))
    # New private runtime file for each job, never a live or frozen replay database.
    paper = PaperEventStore(store.root / (identifier + "-synthetic-shadow.sqlite"))
    engine = PaperEngine(store=paper, models=models, session_id=prepared.experiment_id, session_spec=spec)
    first = min(instant(r["bar_open_at"]) for r in selected["test"])
    for p in products:
        engine.seed_history(p, [b for b in dataset["bars"][p] if instant(b["bar_open_at"]) < first])
    engine.start(now=first.isoformat())
    last = first
    for p in products:
        for b in dataset["bars"][p]:
            if instant(b["bar_open_at"]) < first:
                continue
            now = instant(b["bar_open_at"]) + timedelta(hours=1)
            last = max(last, now)
            engine.ingest_candle(p, b, now=now.isoformat(), row_product=p)
    engine.stop(now=last.isoformat())
    proof = paper.verify_chain(session_id=prepared.experiment_id)
    return {"synthetic": True, "session_spec": spec.canonical(), "session_hash": spec.session_spec_hash,
            "chain": proof, "events": paper.count(session_id=prepared.experiment_id),
            "products": {p: engine.state(p).payload() for p in products},
            "mode": "offline synthetic shadow replay", "broker_connected": False,
            "limitations": ["synthetic prices and costs; no scientific finding", "accounts independent per product",
                            "session paper_model_spec_hash binds the new experiment, never a frozen paper model"]}


def run_experiment(store, identifier, dataset, prepared):
    verify_dataset(dataset)
    model_id = prepared.parameters["model_id"]
    expected = prepare_experiment(dataset, model_id=model_id, embargo_seconds=prepared.splits["embargo_seconds"],
        limits=ResourceLimits(**dict(prepared.budgets)), hypothesis=dict(prepared.hypothesis),
        decision_criteria=dict(prepared.decision_criteria))
    if prepared.identity != expected.identity:
        raise ValueError("prepared experiment/runtime identity drift")
    selected, _ = temporal_splits(dataset["rows"], horizon_seconds=dataset["manifest"]["horizon_seconds"],
                                  embargo_seconds=prepared.splits["embargo_seconds"])
    store.checkpoint(identifier, .10, "TRAINING")
    registry = ModelRegistry()
    artifacts, hashes, adapters, scores, predictions, backtests = {}, {}, {}, {}, [], {}
    for product in dataset["manifest"]["products"]:
        adapter = registry.create(model_id)
        train = [r for r in selected["train"] if r["product"] == product]
        adapter.train(model_rows(train, labels=True), synthetic=True)
        artifact = adapter.serialize()
        hashes[product] = store.put_artifact("model", artifact)
        artifacts[product] = artifact
        # The serialized path performs inference, proving restoration is usable.
        adapter = registry.create(model_id).restore(artifact)
        adapters[product] = adapter
        from scripts.trading_lab.models import MeanTrainPredictor
        mean = MeanTrainPredictor().fit(model_rows(train, labels=True))
        scores[product] = {}
        for role in ("validation", "test"):
            block = [r for r in selected[role] if r["product"] == product]
            views = model_rows(block)
            output = tuple(adapter.predict(views))
            if len(output) != len(block) or any(not isinstance(v, Decimal) or not v.is_finite() for v in output):
                raise ValueError("adapter predictions violate declared output")
            scores[product][role] = {"model": metrics(output, block),
                "ZERO": metrics(tuple(Decimal(0) for _ in block), block), "TRAIN_MEAN": metrics(mean.predict(views), block)}
            for row, value in zip(block, output):
                record = PredictionRecord(prediction_id=sha256_canonical(
                    [prepared.experiment_id, product, row["decision_at"]]), model_id=model_id,
                    model_contract_hash=adapter.contract.identity, artifact_hash=hashes[product], product=product,
                    decision_at=row["decision_at"], horizon_seconds=dataset["manifest"]["horizon_seconds"],
                    snapshot_hash=row["snapshot_hash"], features_hash=row["features_hash"],
                    event_ids=tuple(row["event_ids"]), outputs={k: str(value) if k == "return" else None for k in OUTPUTS},
                    synthetic=True)
                predictions.append({"record": record.to_dict(), "fingerprint": record.identity, "split": role})
            if role == "test":
                backtests[product] = _backtest(dataset, block, output, product=product,
                                              prepared=prepared, artifact_hash=hashes[product])
    store.checkpoint(identifier, .55, "VALIDATING")
    prediction_hash = store.put_artifact("predictions", predictions)
    store.checkpoint(identifier, .65, "BACKTEST")
    backtest_hash = store.put_artifact("backtests", backtests)
    store.checkpoint(identifier, .75, "SHADOW")
    shadow = _shadow(store, identifier, dataset, selected, adapters, hashes, prepared)
    shadow_hash = store.put_artifact("shadow", shadow)
    artifact_bindings = {"prepared_hash": prepared.identity, "model_hashes": hashes,
                         "prediction_hash": prediction_hash, "backtest_hash": backtest_hash, "shadow_hash": shadow_hash}
    complete = replace(prepared, status="COMPLETE", artifacts=artifact_bindings)
    criteria = {p: all(Decimal(scores[p]["test"]["model"]["mae"]) < Decimal(scores[p]["test"][b]["mae"])
                       for b in prepared.baselines) for p in scores}
    return {"schema": "model-lab-result-v1", "synthetic": True, "manifest": complete.to_dict(),
            "fingerprint": complete.identity, "prepared": prepared.to_dict(), "models": artifacts,
            "metrics": scores, "criteria_met": criteria, "backtests": backtests, "shadow": shadow,
            "predictions": predictions, "limitations": ["synthetic infrastructure demonstration; no edge claim",
                "bit-for-bit reproducibility requires the recorded numeric runtime", "null/negative outcomes are retained"]}


def execute_job(store, identifier, kind, payload):
    with localcontext() as context:
        context.prec = 34
        if kind == "dataset":
            dataset = synthetic_dataset(**payload)
            store.checkpoint(identifier, .9, "DATASET_BUILT")
            key = store.put_artifact("dataset", dataset, identity=dataset["fingerprint"])
            return {"dataset_hash": key, "manifest": dataset["manifest"], "synthetic": True}
        if kind == "experiment":
            prepared = ExperimentManifest.from_dict(payload["prepared"])
            dataset = store.artifact(prepared.dataset_hash, kind="dataset")
            return run_experiment(store, identifier, dataset, prepared)
        raise ValueError("unknown job workload")
