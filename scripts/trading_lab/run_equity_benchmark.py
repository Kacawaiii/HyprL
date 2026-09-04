"""Execute the frozen equity benchmark once, offline, over the local corpus.

Row-level output -- features, targets, predictions -- is written to the
gitignored local research store and never to the repository. It is derived
from a source whose provider forbids redistribution, and derivation does not
launder that: a matrix of six returns per session plus a forward return is
enough to reconstruct a great deal of the prices it came from.

What may leave this machine is the summary: hashes, counts, fold boundaries
and metrics. Those describe the experiment without republishing the data.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from scripts.trading_lab.equity_benchmark import (
    aggregate, leakage_audit, run_instrument)
from scripts.trading_lab.equity_dataset import build_dataset
from scripts.trading_lab.equity_research import (
    EQUITY_RESEARCH_SPEC_V1, require_outside_holdout, sha256_canonical)
from scripts.trading_lab.local_research_corpus import LocalResearchCorpusRegistry

LOCAL_RESEARCH_ROOT = "var/trading_lab/research/equity_benchmark_v1"


def _gate(spec) -> None:
    """Refuse to start unless the reserved holdout is still untouched."""
    spec.holdout.require_not_observed()
    require_outside_holdout(spec.exploratory_start, spec.exploratory_end)


def run(*, corpus_root=None, fingerprint_path=None, output=None) -> dict:
    spec = EQUITY_RESEARCH_SPEC_V1
    _gate(spec)

    registry = LocalResearchCorpusRegistry(
        corpus_root=corpus_root, fingerprint_path=fingerprint_path)
    report = registry.report()
    if not report.available:
        raise RuntimeError(
            f"the local equity corpus is {report.status}; the benchmark reads "
            "verified local artefacts only and never fetches anything")

    datasets, results, audits = {}, [], {}
    for instrument_id in spec.instruments:
        dataset = build_dataset(instrument_id, registry=registry, spec=spec)
        result = run_instrument(dataset, spec=spec)
        datasets[instrument_id] = dataset
        audits[instrument_id] = leakage_audit(dataset, result, spec=spec)
        results.append(result)

    summary = {
        "schema_version": "trading-lab.equity-benchmark-result.v1",
        "research_spec_hash": spec.spec_hash,
        "source_corpus_content_hash": report.identity["corpus_content_hash"],
        "source_corpus_spec_hash": report.identity["corpus_spec_hash"],
        "calendar_spec_hash": report.identity["calendar_spec_hash"],
        "analytical_adjustment": spec.view.analytical_adjustment,
        "total_return": False,
        "instruments": {
            result["instrument_id"]: {
                key: result[key] for key in
                ("eligible_rows", "oos_rows", "folds", "ridge", "zero",
                 "train_mean")}
            for result in results},
        "dataset_identity": {
            instrument_id: {
                "analytical_view_hash": dataset["analytical_view_hash"],
                "feature_matrix_hash": dataset["feature_matrix_hash"],
                "target_vector_hash": dataset["target_vector_hash"],
                "source_sessions": dataset["source_sessions"],
                "eligible_rows": dataset["eligible_rows"],
            }
            for instrument_id, dataset in datasets.items()},
        "aggregate": aggregate(results),
        "leakage_audit": audits,
    }
    summary["result_hash"] = sha256_canonical(summary)

    local = pathlib.Path(output or LOCAL_RESEARCH_ROOT)
    (local / "rows").mkdir(parents=True, exist_ok=True)
    for instrument_id, dataset in datasets.items():
        slug = instrument_id.replace(":", "_")
        (local / "rows" / f"{slug}.dataset.json").write_text(
            json.dumps(dataset, indent=1), encoding="utf-8")
    for result in results:
        slug = result["instrument_id"].replace(":", "_")
        (local / "rows" / f"{slug}.oos.json").write_text(
            json.dumps([{**row, "prediction": str(row["prediction"]),
                         "actual": str(row["actual"]),
                         "train_mean": str(row["train_mean"])}
                        for row in result["_oos"]], indent=1), encoding="utf-8")
    (local / "summary.local.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None):  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(
        description="Run the frozen equity benchmark V1 (offline, local only)")
    parser.add_argument("--corpus-root", default=None)
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args(argv)
    summary = run(corpus_root=arguments.corpus_root,
                  fingerprint_path=arguments.fingerprint,
                  output=arguments.output)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("instruments", "leakage_audit")}, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
