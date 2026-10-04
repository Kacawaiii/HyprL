"""Persist the out-of-sample walk-forward signal and position-target runs. Offline.

The committed economic backtest was driven by the walk-forward predictions stored in
``benchmark_results_v2``: every prediction was produced by a model fitted only on rows before its
fold's test window. Nothing is refitted and nothing is chosen here. This module replays those frozen
predictions through Signal V1 and Risk V1 -- the same per-fold provenance (the selected candidate's
model spec hash, the fold's final fit hash, the benchmark spec hash) that the economic backtest used
-- and writes the decisions and targets down so the cockpit can show them.

The acceptance gate is the committed ``signal_series_hash`` and ``position_target_series_hash`` of
``economic_backtest_v1/<product>.json``. A run that does not hash to them is refused, never adjusted.
The same check (``verify_run``) guards the API before it serves anything.

    python -m scripts.trading_lab.build_signal_runs [--root data/crypto] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from decimal import Decimal

from scripts.trading_lab.risk_engine import (
    RISK_SCALE_V1,
    RISK_SPEC_V1,
    generate_position_targets,
)
from scripts.trading_lab.signal_engine import (
    SIGNAL_SPEC_V1,
    SignalSeries,
    generate_signal,
)

RUNS_DIR = "signal_runs_v1"
BENCHMARK_DIR = "benchmark_results_v2"
ECONOMIC_DIR = "economic_backtest_v1"
PRODUCTS = ("BTC-USD", "ETH-USD")
SCHEMA_VERSION = "trading-lab.signal-run.v1"
MANIFEST_SCHEMA_VERSION = "trading-lab.signal-run-manifest.v1"
OUT_OF_SAMPLE = "walk_forward"


class SignalRunError(RuntimeError):
    """A run that cannot be built or does not hash to the committed backtest."""


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _load(path: pathlib.Path) -> tuple[dict, bytes]:
    body = path.read_bytes()
    return json.loads(body.decode("utf-8")), body


def _fold_table(benchmark: dict) -> dict[int, dict]:
    table = {}
    for fold in benchmark["selection"]["folds"]:
        chosen = next(candidate for candidate in fold["candidates"]
                      if candidate["candidate_id"] == fold["selected_candidate_id"])
        table[int(fold["fold_index"])] = {
            "fold_index": int(fold["fold_index"]),
            "selected_candidate_id": fold["selected_candidate_id"],
            "model_spec_hash": chosen["model_spec_hash"],
            "fitted_hash": fold["final_fit_hash"],
            "train_rows": fold["train_rows"],
            "validation_rows": fold["validation_rows"],
            "test_first": fold["test_first"],
            "test_last": fold["test_last"],
        }
    return table


def _replay(records, folds: dict[int, dict], benchmark_spec_hash: str):
    """Predictions -> signals -> targets, reading only a timestamp and a prediction."""
    decisions = []
    previous = None
    for record in records:
        stamp = record["bar_open_at"]
        if previous is not None and stamp <= previous:
            raise SignalRunError("predictions are not in strictly ascending order")
        previous = stamp
        fold = folds[int(record["fold_index"])]
        decisions.append(generate_signal(
            timestamp=stamp, prediction=Decimal(record["prediction"]),
            model_spec_hash=fold["model_spec_hash"], fitted_hash=fold["fitted_hash"],
            benchmark_spec_hash=benchmark_spec_hash, signal_spec=SIGNAL_SPEC_V1))
    series = SignalSeries(signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
                          decisions=tuple(decisions))
    return series, generate_position_targets(series)


def _decision_rows(series, records) -> list[dict]:
    return [{
        "timestamp": decision.timestamp,
        "prediction": decision.canonical()["prediction"],
        "direction": decision.direction,
        "strength": decision.canonical()["strength"],
        "fold_index": int(record["fold_index"]),
        "decision_hash": decision.decision_hash,
    } for decision, record in zip(series.decisions, records)]


def _target_rows(targets) -> list[dict]:
    return [{
        "timestamp": target.timestamp,
        "side": target.side,
        "target_exposure": str(target.target_exposure),
        "signal_strength": str(target.signal_strength),
        "raw_target_exposure": str(target.raw_target_exposure),
        "position_target_hash": target.position_target_hash,
    } for target in targets.targets]


def build_run(product: str, *, benchmark: dict, benchmark_sha256: str, economic: dict,
              economic_sha256: str) -> dict:
    """One product's run, or ``SignalRunError`` when it does not hash to the backtest."""
    if benchmark["product"] != product or economic["spec"]["product"] != product:
        raise SignalRunError(f"{product}: artefacts describe another product")
    if economic["spec"]["source_benchmark_results_hash"] != benchmark["benchmark_results_hash"]:
        raise SignalRunError(f"{product}: the backtest was not driven by this benchmark")
    folds = _fold_table(benchmark)
    records = benchmark["oos_records"]
    series, targets = _replay(records, folds, benchmark["benchmark_spec_hash"])
    for label, got, want in (
            ("signal_series_hash", series.series_hash, economic["signal_series_hash"]),
            ("position_target_series_hash", targets.series_hash,
             economic["position_target_series_hash"])):
        if got != want:
            raise SignalRunError(
                f"{product}: {label} {got} does not match the committed {want}")
    return {
        "schema_version": SCHEMA_VERSION,
        "product": product,
        "out_of_sample": OUT_OF_SAMPLE,
        "experiment_type": "exploratory",
        "confirmatory": False,
        "protocol": {
            "benchmark_protocol": benchmark["protocol_version"],
            "benchmark_spec_hash": benchmark["benchmark_spec_hash"],
            "benchmark_results_hash": benchmark["benchmark_results_hash"],
            "benchmark_result_file_sha256": benchmark_sha256,
            "economic_backtest_protocol": economic["spec"]["protocol_version"],
            "economic_backtest_spec_hash": economic["economic_backtest_spec_hash"],
            "economic_results_hash": economic["economic_results_hash"],
            "economic_result_file_sha256": economic_sha256,
            "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
            "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
            "risk_scale": str(RISK_SCALE_V1),
        },
        "corpus": {
            "corpus_content_hash": benchmark["corpus_content_hash"],
            "corpus_spec_hash": benchmark["corpus_spec_hash"],
            "dataset_hash": benchmark["dataset_hash"],
        },
        "signal_series_hash": series.series_hash,
        "position_target_series_hash": targets.series_hash,
        "counts": {"decisions": series.count, "targets": targets.count,
                   "folds": len(folds)},
        "window": {"first": series.first_timestamp, "last": series.last_timestamp},
        "folds": [folds[index] for index in sorted(folds)],
        "decisions": _decision_rows(series, records),
        "targets": _target_rows(targets),
    }


def verify_run(run: dict, *, economic: dict) -> None:
    """Refuse a run whose stored rows are not exactly what the frozen protocol yields.

    Rebuilds every decision and target from the stored predictions and fold provenance, requires the
    stored rows to equal the rebuilt ones, and requires both series hashes to equal the committed
    economic backtest's. A tampered artefact fails here, before anything is served.
    """
    try:
        if run["schema_version"] != SCHEMA_VERSION or run["out_of_sample"] != OUT_OF_SAMPLE:
            raise SignalRunError("unsupported signal run schema or label")
        if run["product"] != economic["spec"]["product"]:
            raise SignalRunError("the run and the backtest describe different products")
        protocol = run["protocol"]
        if (protocol["signal_spec_hash"] != SIGNAL_SPEC_V1.spec_hash
                or protocol["risk_spec_hash"] != RISK_SPEC_V1.risk_spec_hash
                or protocol["risk_scale"] != str(RISK_SCALE_V1)):
            raise SignalRunError("the run was built under other signal or risk specs")
        if protocol["economic_results_hash"] != economic["economic_results_hash"]:
            raise SignalRunError("the run was built from another economic backtest")
        folds = {fold["fold_index"]: fold for fold in run["folds"]}
        records = [{"bar_open_at": row["timestamp"], "prediction": row["prediction"],
                    "fold_index": row["fold_index"]} for row in run["decisions"]]
        series, targets = _replay(records, folds, protocol["benchmark_spec_hash"])
        if series.series_hash != economic["signal_series_hash"] \
                or run["signal_series_hash"] != economic["signal_series_hash"]:
            raise SignalRunError("signal series hash does not match the economic backtest")
        if targets.series_hash != economic["position_target_series_hash"] \
                or run["position_target_series_hash"] != economic["position_target_series_hash"]:
            raise SignalRunError(
                "position target series hash does not match the economic backtest")
        if run["decisions"] != _decision_rows(series, records):
            raise SignalRunError("stored decisions differ from the replayed ones")
        if run["targets"] != _target_rows(targets):
            raise SignalRunError("stored targets differ from the replayed ones")
        if run["counts"] != {"decisions": series.count, "targets": targets.count,
                             "folds": len(folds)}:
            raise SignalRunError("stored counts differ from the replayed ones")
    except SignalRunError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise SignalRunError(f"malformed signal run: {error!r}") from error


def build_all(root: pathlib.Path) -> dict[str, bytes]:
    """Every output file, as name -> bytes. Pure: nothing is written."""
    manifest_src, _ = _load(root / ECONOMIC_DIR / "manifest.json")
    files: dict[str, bytes] = {}
    entries = {}
    for product in PRODUCTS:
        benchmark, benchmark_body = _load(root / BENCHMARK_DIR / f"{product}.json")
        economic, economic_body = _load(root / ECONOMIC_DIR / f"{product}.json")
        run = build_run(product, benchmark=benchmark,
                        benchmark_sha256=_sha256(benchmark_body), economic=economic,
                        economic_sha256=_sha256(economic_body))
        body = _canonical_bytes(run)
        files[f"{product}.json"] = body
        entries[product] = {
            "result_file": f"{product}.json",
            "result_file_sha256": _sha256(body),
            "result_file_bytes": len(body),
            "decisions": run["counts"]["decisions"],
            "targets": run["counts"]["targets"],
            "folds": run["counts"]["folds"],
            "signal_series_hash": run["signal_series_hash"],
            "position_target_series_hash": run["position_target_series_hash"],
        }
    files["manifest.json"] = _canonical_bytes({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "out_of_sample": OUT_OF_SAMPLE,
        "experiment_type": "exploratory",
        "confirmatory": False,
        "refitted": False,
        "source_prediction_protocol": manifest_src["source_prediction_protocol"],
        "corpus": manifest_src["corpus"],
        "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
        "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
        "gate": "series hashes equal the committed economic_backtest_v1 hashes",
        "products": entries,
    })
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/crypto")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed files instead of writing them")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    try:
        files = build_all(root)
    except SignalRunError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    target = root / RUNS_DIR
    if args.check:
        stale = [name for name, body in files.items()
                 if not (target / name).is_file() or (target / name).read_bytes() != body]
        if stale:
            print(f"STALE: {stale}", file=sys.stderr)
            return 1
        print("signal runs reproduce byte for byte")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (target / name).write_bytes(body)
    manifest = json.loads(files["manifest.json"])
    for product, entry in manifest["products"].items():
        print(f"{product}: {entry['decisions']} decisions, {entry['targets']} targets, "
              f"{entry['folds']} folds; hash gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
