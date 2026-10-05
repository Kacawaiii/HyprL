"""Offline synthetic dataset -> isolated experiment -> backtest -> shadow demo."""
import argparse
import json
from pathlib import Path
import time

from scripts.trading_lab.platform.jobs import JobRunner, TERMINAL


def wait(store, identifier, timeout=190):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = store.status(identifier)
        if status["state"] in TERMINAL:
            if status["state"] != "COMPLETE":
                raise RuntimeError("demo job " + status["state"] + ": " + str(status["error_code"]))
            return store.result(identifier)["result"]
        time.sleep(.1)
    store.cancel(identifier)
    raise RuntimeError("demo job timed out")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="var/trading_lab/model-lab-demo")
    parser.add_argument("--model", choices=("synthetic-ridge-v1", "local-momentum-v1"), default="synthetic-ridge-v1")
    parser.add_argument("--bars", type=int, default=120)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    workspace = Path.cwd().resolve()
    if not root.is_relative_to(workspace / "var"):
        parser.error("demo root must be beneath this worktree's ignored var directory")
    with JobRunner(root) as runner:
        dataset_job = runner.store.submit("dataset", {"products": ["BTC-USD", "ETH-USD"], "bars": args.bars})
        data_result = wait(runner.store, dataset_job)
        from scripts.trading_lab.platform.experiments import prepare_experiment
        dataset = runner.store.artifact(data_result["dataset_hash"], kind="dataset")
        prepared = prepare_experiment(dataset, model_id=args.model)
        runner.store.put_artifact("experiment", prepared.to_dict())
        jobs, results = [], []
        for _ in range(2):
            job = runner.store.submit("experiment", {"prepared": prepared.to_dict()})
            jobs.append(job)
            results.append(wait(runner.store, job))
        result_hashes = [runner.store.status(j)["result_hash"] for j in jobs]
        if result_hashes[0] != result_hashes[1]:
            raise RuntimeError("bit-for-bit reproduction failed")
        print(json.dumps({"synthetic": True, "dataset_hash": data_result["dataset_hash"],
            "experiment_id": prepared.experiment_id, "model_id": args.model,
            "result_hash": result_hashes[0], "reproduced_bit_for_bit": True,
            "metrics": results[0]["metrics"], "shadow_events": results[0]["shadow"]["events"],
            "limitations": results[0]["limitations"]}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
