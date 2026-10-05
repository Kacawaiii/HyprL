"""Offline archive evidence + reproducible synthetic snapshot-to-monitoring chain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.trading_lab.ops.control import local_path, save, OpsRefused
from scripts.trading_lab.sources.canonical import sha256_canonical

DEMO_AT = "2026-08-01T00:00:00+00:00"


def archive_evidence(*, fomc_store=None, edgar_store=None):
    from scripts.trading_lab.app_api.sources import FomcViews, EdgarViews
    from scripts.trading_lab.platform.snapshot import SnapshotBuilder
    from scripts.trading_lab.platform.demo import summary
    from scripts.trading_lab.edgar.closure import _tree_digest
    roots = {name: Path(root) for name, root in (("fomc", fomc_store), ("edgar", edgar_store)) if root}
    fingerprints = {name: _tree_digest(root) for name, root in roots.items()}
    fomc, edgar = FomcViews(fomc_store), EdgarViews(edgar_store)
    try:
        statuses = {"fomc": fomc.status(), "edgar": edgar.status()}
        reads = []
        with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=fomc_store,
                             edgar_store=edgar_store, prices=None) as builder:
            for name in roots:
                data = statuses[name]
                if data["status"] != "AVAILABLE" or not data.get("suggested_as_of"):
                    raise OpsRefused("ARCHIVE_EVIDENCE_UNAVAILABLE")
                snapshot = builder.build(data["suggested_as_of"], ["AAPL", "BTC-USD"])
                repeated = builder.build(data["suggested_as_of"], ["BTC-USD", "AAPL"])
                if snapshot.identity != repeated.identity or snapshot.sources[name]["state"] != "RESOLVED":
                    raise OpsRefused("ARCHIVE_SNAPSHOT_VERIFICATION_FAILED")
                # Replay uses precisely the source's own T/H; never a combined horizon.
                view = fomc if name == "fomc" else edgar
                replay = view.replay(as_of=data["suggested_as_of"], horizon=data["horizon"])
                if not replay["identical"]:
                    raise OpsRefused("ARCHIVE_REPLAY_FAILED")
                reads.append({"attesting_source": name, "snapshot": summary(snapshot), "replay_identical": True})
    finally:
        fomc.close()
        edgar.close()
    if any(_tree_digest(root) != fingerprints[name] for name, root in roots.items()):
        raise OpsRefused("ARCHIVE_BYTES_CHANGED")
    return {"kind": "REAL_ARCHIVED_EVIDENCE" if roots else "NOT_CONFIGURED", "read_only": True,
            "tree_digests": fingerprints, "unchanged": True, "reads": reads,
            "limits": ["no price data opened", "protected price decisions remain PROTECTED",
                       "archive event windows are not joined to synthetic training decisions"]}


def run(root, *, fomc_store=None, edgar_store=None, bars=120):
    root = local_path(root)
    if root.exists():
        raise OpsRefused("DEMO_REQUIRES_NEW_RUNTIME")
    archives = archive_evidence(fomc_store=fomc_store, edgar_store=edgar_store)
    from scripts.trading_lab.platform.jobs import JobRunner
    from scripts.trading_lab.platform.model_lab_demo import wait
    from scripts.trading_lab.research.store import ResearchStore
    from scripts.trading_lab.research.proposals import propose
    from scripts.trading_lab.research.importers import import_model_lab
    from scripts.trading_lab.research.monitoring import make_reference, monitor
    from scripts.trading_lab.research.demo import synthetic_scenarios
    root.mkdir(parents=True, mode=0o700)
    store = ResearchStore(root / "registry")
    results = []
    with JobRunner(root / "lab") as runner:
        identifier = runner.store.submit("dataset", {"products": ["BTC-USD", "ETH-USD"], "bars": bars})
        data = wait(runner.store, identifier)
        dataset = runner.store.artifact(data["dataset_hash"], kind="dataset")
        snapshots = {k: sha256_canonical(s) for k, s in dataset["snapshots"].items()}
        if any(key != identity for key, identity in snapshots.items()):
            raise OpsRefused("SYNTHETIC_SNAPSHOT_DIGEST_FAILED")
        for proposal in propose(dataset):
            plan, prepared = proposal["hypothesis"], proposal["prepared"]
            store.register(plan)
            store.start_trial(plan, prepared, trial_id=prepared.experiment_id, recorded_at=DEMO_AT)
            jobs, reproduced = [], []
            for _ in range(2):
                job = runner.store.submit("experiment", {"prepared": prepared.to_dict()})
                jobs.append(job)
                reproduced.append(wait(runner.store, job))
            hashes = [runner.store.status(job)["result_hash"] for job in jobs]
            if hashes[0] != hashes[1]:
                raise OpsRefused("MODEL_REPRODUCTION_FAILED")
            result = reproduced[0]
            imported = import_model_lab(store, dataset, result,
                shadow_path=runner.store.root / (jobs[0] + "-synthetic-shadow.sqlite"), recorded_at=DEMO_AT)
            monitoring = {}
            for product in dataset["manifest"]["products"]:
                reference = make_reference(store, reference_id=prepared.experiment_id + "-" + product,
                    as_of=DEMO_AT, product=product, model_id=prepared.parameters["model_id"], split="validation")
                view = monitor(store, as_of=DEMO_AT, product=product, model_id=prepared.parameters["model_id"],
                               split="test", reference_hash=reference)
                monitoring[product] = {"hash": sha256_canonical(view), "sample": view["sample"],
                                       "reference_hash": reference}
            met = all(result["criteria_met"].values())
            store.observe_trial(prepared.experiment_id, state="COMPLETE", outcome="POSITIVE" if met else "NEGATIVE",
                recorded_at=DEMO_AT, evidence={"result_hash": hashes[0], "synthetic": True, "scientific_claim": False})
            results.append({"model_id": prepared.parameters["model_id"], "result_hash": hashes[0],
                "reproduced_bit_for_bit": True, "predictions": imported["predictions"],
                "criteria_met": result["criteria_met"], "monitoring": monitoring})
    scenarios = synthetic_scenarios(store, at=DEMO_AT)
    evidence = {"schema": "hyprl-demo-chain-v1", "archives": archives,
        "synthetic": {"label": "SYNTHETIC DEMONSTRATION; NO SCIENTIFIC EDGE CLAIM", "dataset_hash": data["dataset_hash"],
            "snapshot_count": len(snapshots), "snapshot_inventory_hash": sha256_canonical(snapshots),
            "included_rows": len(dataset["rows"]), "models": results, "monitoring_scenarios": scenarios},
        "registry": store.verify(), "network_requests": 0, "new_real_training": False,
        "external_model_calls": 0, "broker_orders": 0, "holdout_price_reads": 0,
        "limitations": ["real archives prove event reads only; synthetic models use synthetic snapshots/prices",
                        "no causally overlapping real training population is claimed",
                        "synthetic outcomes prove the software path; absent inference telemetry stays unknown"]}
    save(root / "demo-evidence.json", dict(evidence, identity=sha256_canonical(evidence)))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="new directory beneath this worktree var")
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument("--bars", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        result = run(args.root, fomc_store=args.fomc_store, edgar_store=args.edgar_store, bars=args.bars)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"state": "BLOCKED", "code": str(error) if isinstance(error, OpsRefused) else "DEMO_CHAIN_FAILED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
