"""Monitoring of one completed synthetic Model Lab experiment, as a bounded worker workload.

Reuses the research importer and monitor without refitting: the experiment's exact
predictions, snapshots and shadow chain are imported into a private append-only
ledger owned by this monitoring job, a reference is fixed on the validation split
and the test split is monitored against it. Labels not yet available at the
monitoring instant stay pending. Nothing here reads a real source or trains.
"""
from __future__ import annotations

from scripts.trading_lab.platform.contracts import digest
from scripts.trading_lab.sources.canonical import sha256_canonical


def monitor_experiment(store, identifier, payload):
    from scripts.trading_lab.research.importers import import_model_lab
    from scripts.trading_lab.research.monitoring import make_reference, monitor
    from scripts.trading_lab.research.store import ResearchStore, now

    if set(payload) != {"experiment_job_id"}:
        raise ValueError("monitoring accepts exactly one experiment job")
    subject = payload["experiment_job_id"]
    status = store.status(subject)
    if status["kind"] != "experiment" or status["state"] != "COMPLETE":
        raise ValueError("only a completed experiment can be monitored")
    digest(status["result_hash"])
    result = store.artifact(status["result_hash"], kind="result")
    digest(result["manifest"]["dataset_hash"])
    dataset = store.artifact(result["manifest"]["dataset_hash"], kind="dataset")
    ledger = ResearchStore(store.root / "monitoring" / identifier)
    imported = import_model_lab(ledger, dataset, result,
                                shadow_path=store.root / (subject + "-synthetic-shadow.sqlite"))
    store.checkpoint(identifier, .5, "VALIDATING")
    at = now()
    model_id = imported["model_id"]
    products = {}
    for product in dataset["manifest"]["products"]:
        reference = make_reference(ledger, reference_id=imported["experiment_hash"] + "-" + product + "-validation",
                                   as_of=at, product=product, model_id=model_id, split="validation")
        view = monitor(ledger, as_of=at, product=product, model_id=model_id, split="test", reference_hash=reference)
        products[product] = {"reference_hash": reference, "monitoring_hash": sha256_canonical(view), "view": view}
    return {"schema": "lab-experiment-monitoring-v1", "experiment_job_id": subject,
            "experiment_hash": imported["experiment_hash"], "model_id": model_id, "as_of": at,
            "predictions": imported["predictions"], "reference_split": "validation", "monitored_split": "test",
            "products": products, "ledger": ledger.verify(), "synthetic": True,
            "limitations": ["synthetic prices and labels; establishes no real edge",
                            "inference latency and errors were not measured for this offline run"]}
