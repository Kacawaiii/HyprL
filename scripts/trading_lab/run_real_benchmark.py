"""Execute the pre-registered Benchmark V1 contract. Decide nothing.

This runner is deliberately dull. Every choice that could bias a result --
features, horizon, fold geometry, model configurations, the selection rule,
the robustness periods, the sensitivity scenarios -- was frozen and committed
before any score existed, and lives in `real_benchmark.py`. Nothing here
redefines any of it, and nothing here is configurable from the outside. There
is no `--alpha`, no `--features`, no `--train-size`: a knob a caller can turn
after seeing a number is a knob that will eventually be turned.

Three refusals happen before the first fit:

* the rebuilt `benchmark_spec_hash` must equal the one the caller pre-registered;
* the corpus must verify against its own manifest;
* the fold geometry must match what was measured when the contract was frozen.

A "nearly identical" contract is not the contract. If any of those disagree,
the run stops rather than producing a number under a protocol nobody agreed to.

The two products are separate experiments. Their observations are never pooled
into a single "crypto" score, and their metrics are never averaged together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
import pathlib
import shutil
import tempfile

from scripts.trading_lab.capture_market_history import (
    CORPUS_ID,
    CORPUS_PROVIDER,
    _coinbase_page,
    _iso,
    _parse_iso,
    load_canonical_rows,
    load_manifest,
    verify_corpus,
)
from scripts.trading_lab.coinbase_candles import MAX_CANDLES_PER_RESPONSE, TIMEFRAME_DURATIONS
from scripts.trading_lab.market_data_store import MarketDataStore
from scripts.trading_lab.market_dataset import build_dataset
from scripts.trading_lab.market_series import load_market_series
from scripts.trading_lab.market_snapshots import _materialize_snapshot
from scripts.trading_lab.model_robustness import (
    RobustnessScenario,
    analyse_robustness,
)
from scripts.trading_lab.model_selection import candidate, evaluate_selection
from scripts.trading_lab.models import (
    RidgeRegressionPredictor,
    XGBoostConfig,
    XGBoostRegressionPredictor,
)
from scripts.trading_lab.real_benchmark import (
    BENCHMARK_PROTOCOL_VERSION,
    DATASET_CONFIG_V1,
    FEATURE_COLUMNS_V1,
    RIDGE_ALPHA_V1,
    ROBUSTNESS_BOUNDARIES_V1,
    SENSITIVITY_SCENARIOS_V1,
    WALK_FORWARD_CONFIG_V1,
    RealBenchmarkError,
    build_benchmark_spec,
)
from scripts.trading_lab.walk_forward import build_folds, usable_rows

RUNNER_VERSION = "trading-lab.real-benchmark-runner.v1"
RESULT_SCHEMA_VERSION = "trading-lab.real-benchmark-result.v1"

# What the geometry measured when the contract was frozen, per product. These
# are an assertion, not a target: if the runner produces anything else it is
# not executing the protocol that was pre-registered.
EXPECTED_GEOMETRY = {
    "series_points": 8750,
    "missing_openings": 10,
    "usable_rows": 8663,
    "folds": 46,
    "min_effective_validation": 164,
    "oos_records": 7728,
}


class RealBenchmarkRunError(RuntimeError):
    """Raised when the frozen protocol cannot be executed exactly as written."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _metrics_payload(metrics) -> dict[str, object]:
    return {
        "rank_ic": _text(metrics.rank_ic),
        "mae": _text(metrics.mae),
        "rmse": _text(metrics.rmse),
        "observations": metrics.observations,
    }


# --- the frozen candidate set --------------------------------------------


def _candidates(alpha: Decimal = RIDGE_ALPHA_V1):
    """Fresh, unfitted candidates in the contract's configuration."""
    return (
        candidate("ridge", lambda: RidgeRegressionPredictor(
            feature_columns=FEATURE_COLUMNS_V1, alpha=alpha)),
        candidate("xgboost", lambda: XGBoostRegressionPredictor(
            feature_columns=FEATURE_COLUMNS_V1, config=XGBoostConfig())),
    )


def _scenarios():
    return tuple(
        RobustnessScenario(scenario_id, _candidates(alpha))
        for scenario_id, alpha in SENSITIVITY_SCENARIOS_V1
    )


# --- corpus -> series (offline) ------------------------------------------


def load_corpus_series(corpus_root, *, product: str, database_path):
    """Replay the frozen corpus into a temporary store. No network, ever."""
    manifest = load_manifest(corpus_root)
    entry = next((item for item in manifest["products"] if item["product"] == product), None)
    if entry is None:
        raise RealBenchmarkRunError(f"{product} is not part of this corpus")
    timeframe = manifest["spec"]["timeframe"]
    duration = TIMEFRAME_DURATIONS[timeframe]
    rows = load_canonical_rows(
        pathlib.Path(corpus_root) / CORPUS_ID / entry["canonical_path"])

    store = MarketDataStore(database_path)
    declared = manifest["capture_completed_at"]
    ingested = _iso(_parse_iso(declared, field="capture_completed_at")
                    + timedelta(seconds=1))
    for start in range(0, len(rows), MAX_CANDLES_PER_RESPONSE):
        store.ingest_coinbase_response(
            _coinbase_page(rows[start:start + MAX_CANDLES_PER_RESPONSE]),
            product_id=product, timeframe=timeframe,
            available_at=declared, ingested_at=ingested)

    connection = store._connect()
    try:
        last = _parse_iso(rows[-1]["bar_open_at"], field="bar_open_at")
        snapshot = _materialize_snapshot(
            connection, provider=CORPUS_PROVIDER, product_id=product,
            timeframe=timeframe, range_start=rows[0]["bar_open_at"],
            range_end=_iso(last + duration),
            as_of=_iso(_parse_iso(ingested, field="ingested_at") + timedelta(seconds=1)))
        return load_market_series(connection, snapshot_id=snapshot.snapshot_id)
    finally:
        connection.close()


# --- results ---------------------------------------------------------------


@dataclass(frozen=True)
class RealBenchmarkResult:
    """What the frozen protocol produced. Never what it should have produced."""

    schema_version: str
    runner_version: str
    protocol_version: str
    product: str
    benchmark_spec_hash: str
    corpus_spec_hash: str
    corpus_content_hash: str
    dataset_hash: str
    geometry: dict[str, object]
    selection: dict[str, object]
    robustness: dict[str, object]
    scenarios: tuple[dict[str, object], ...]
    oos_records: tuple[dict[str, str], ...]

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runner_version": self.runner_version,
            "protocol_version": self.protocol_version,
            "product": self.product,
            "benchmark_spec_hash": self.benchmark_spec_hash,
            "corpus_spec_hash": self.corpus_spec_hash,
            "corpus_content_hash": self.corpus_content_hash,
            "dataset_hash": self.dataset_hash,
            "geometry": self.geometry,
            "selection": self.selection,
            "robustness": self.robustness,
            "scenarios": [dict(entry) for entry in self.scenarios],
            "oos_records": [dict(record) for record in self.oos_records],
        }

    @property
    def benchmark_results_hash(self) -> str:
        return _sha256_canonical(self.canonical())


def _fold_payload(fold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        "train_rows": fold.train_rows,
        "validation_rows": fold.validation_rows,
        "candidates": [
            {
                "candidate_id": entry.candidate_id,
                "model_spec_hash": entry.model_spec_hash,
                "validation_rank_ic": _text(entry.rank_ic),
                "validation_mae": _text(entry.mae),
                "validation_rmse": _text(entry.rmse),
                "validation_observations": entry.observations,
                "selection_fit_hash": entry.selection_fit_hash,
            }
            for entry in fold.candidates
        ],
        "selected_candidate_id": fold.selected_candidate_id,
        "selection_reason": fold.selection_reason,
        "selection_fit_hash": fold.selection_fit_hash,
        "final_fit_hash": fold.final_fit_hash,
        "test_first": fold.test_records[0].bar_open_at if fold.test_records else None,
        "test_last": fold.test_records[-1].bar_open_at if fold.test_records else None,
        "test_metrics": _metrics_payload(fold.test_metrics),
    }


def _counts(values) -> dict[str, int]:
    tally: dict[str, int] = {}
    for value in values:
        tally[value] = tally.get(value, 0) + 1
    return dict(sorted(tally.items()))


def run_product_benchmark(corpus_dir, product: str, expected_benchmark_spec_hash: str,
                          *, database_path=None) -> RealBenchmarkResult:
    """Run the frozen contract for ONE product. Refuses before fitting if anything drifted."""
    corpus_root = pathlib.Path(corpus_dir)

    # 1. the contract must be exactly the pre-registered one
    spec = build_benchmark_spec(product, corpus_root=corpus_root)
    if spec.benchmark_spec_hash != expected_benchmark_spec_hash:
        raise RealBenchmarkError(
            f"{product}: benchmark spec hash {spec.benchmark_spec_hash} does not match the "
            f"pre-registered {expected_benchmark_spec_hash}; refusing to run a protocol "
            "nobody registered")

    # 2. the corpus must verify against its own manifest
    report = verify_corpus(corpus_root)
    if (report["corpus_spec_hash"] != spec.corpus_spec_hash
            or report["corpus_content_hash"] != spec.corpus_content_hash):
        raise RealBenchmarkRunError(f"{product}: corpus hashes drifted from the contract")

    owned = database_path is None
    workspace = pathlib.Path(tempfile.mkdtemp()) if owned else None
    try:
        target = (workspace / f"{product}.sqlite3") if owned else pathlib.Path(database_path)
        series = load_corpus_series(corpus_root, product=product, database_path=target)
        dataset = build_dataset(series, config=DATASET_CONFIG_V1)
        rows = usable_rows(dataset)
        folds = build_folds(dataset, config=WALK_FORWARD_CONFIG_V1)

        # 3. the geometry must be the one measured when the contract was frozen
        observed = {
            "series_points": len(series.points),
            "missing_openings": len(series.missing_openings),
            "usable_rows": len(rows),
            "folds": len(folds),
            "min_effective_validation": min(len(v) for _, v, _ in folds) if folds else 0,
            "oos_records": sum(len(t) for _, _, t in folds),
        }
        if observed != EXPECTED_GEOMETRY:
            raise RealBenchmarkRunError(
                f"{product}: geometry {observed} does not match the frozen "
                f"{EXPECTED_GEOMETRY}; the runner is not executing the registered contract")

        # --- from here on, the first real fits happen ---
        evaluation = evaluate_selection(dataset, config=WALK_FORWARD_CONFIG_V1,
                                        candidates=_candidates())
        report_robustness = analyse_robustness(
            evaluation, boundaries=ROBUSTNESS_BOUNDARIES_V1, scenarios=_scenarios(),
            dataset=dataset, config=WALK_FORWARD_CONFIG_V1)
    finally:
        if owned:
            shutil.rmtree(workspace, ignore_errors=True)

    stability = report_robustness.stability
    return RealBenchmarkResult(
        schema_version=RESULT_SCHEMA_VERSION,
        runner_version=RUNNER_VERSION,
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        product=product,
        benchmark_spec_hash=spec.benchmark_spec_hash,
        corpus_spec_hash=spec.corpus_spec_hash,
        corpus_content_hash=spec.corpus_content_hash,
        dataset_hash=dataset.dataset_hash,
        geometry=observed,
        selection={
            "results_hash": evaluation.results_hash,
            "spec_hash": evaluation.spec_hash,
            "global_test_metrics": _metrics_payload(evaluation.global_test_metrics),
            "selection_counts": _counts(
                fold.selected_candidate_id for fold in evaluation.folds),
            "selection_reasons": _counts(
                fold.selection_reason for fold in evaluation.folds),
            "folds": [_fold_payload(fold) for fold in evaluation.folds],
        },
        robustness={
            "results_hash": report_robustness.results_hash,
            "spec_hash": report_robustness.spec_hash,
            "stability": {
                "fold_count": stability.fold_count,
                "candidate_counts": [list(pair) for pair in stability.candidate_counts],
                "candidate_fractions": [[name, str(value)]
                                        for name, value in stability.candidate_fractions],
                "transition_count": stability.transition_count,
                "transition_rate": _text(stability.transition_rate),
                "longest_run": stability.longest_run,
                "selection_reasons": [list(pair) for pair in stability.selection_reasons],
            },
            "geometry": report_robustness.geometry.canonical(),
            "margins": [margin.canonical() for margin in report_robustness.margins],
            "periods": [period.canonical() for period in report_robustness.subperiods],
            "rank_stability": report_robustness.rank_stability.canonical(),
            "trading_cost_analysis_available":
                report_robustness.trading_cost_analysis_available,
        },
        scenarios=tuple(
            {
                "scenario_id": entry.scenario_id,
                "selection_spec_hash": entry.selection_spec_hash,
                "selection_results_hash": entry.selection_results_hash,
                "candidate_counts": [list(pair)
                                     for pair in entry.stability.candidate_counts],
                "transition_count": entry.stability.transition_count,
                "longest_run": entry.stability.longest_run,
                "global_test_metrics": _metrics_payload(entry.global_test_metrics),
            }
            for entry in report_robustness.scenarios
        ),
        oos_records=tuple(
            {
                "fold_index": str(record.fold_index),
                "bar_open_at": record.bar_open_at,
                "prediction": str(record.prediction),
                "actual_forward_return": str(record.actual_forward_return),
            }
            for record in evaluation.oos_records
        ),
    )


def write_result(result: RealBenchmarkResult, path) -> str:
    """Canonical JSON on disk. Decimals as strings, never as ambiguous floats."""
    payload = dict(result.canonical())
    payload["benchmark_results_hash"] = result.benchmark_results_hash
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (_canonical_json(payload) + "\n").encode("utf-8")
    target.write_bytes(body)
    return hashlib.sha256(body).hexdigest()
