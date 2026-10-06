"""Reproducible synthetic probability calibration and separate paper-exit demonstration."""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import random

from scripts.trading_lab.market_dataset import Dataset, DatasetConfig, DatasetRow, FeatureDefinition, LabelSpec
from scripts.trading_lab.market_series import SeriesPoint
from scripts.trading_lab.platform.contracts import ModelContract, OUTPUTS
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical
from scripts.trading_lab.walk_forward import WalkForwardConfig, build_folds
from .calibration import ScoreObservation, fit_isotonic, reliability, evaluate_test
from .risk import protection_plan, simulate
from .spec import SPEC_HASH, policy_spec


def synthetic_scores(product="BTC-USD", *, count=520):
    """Fixed synthetic source of binary scores, not a return-to-probability transformation."""
    rng = random.Random(721 if product == "BTC-USD" else 722)
    rows = []
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    for i in range(count):
        at = (start + timedelta(hours=i)).isoformat()
        score = (i % 17 + 1) / 18
        # Labels are simulated Bernoulli events; no observed price or real fitting.
        label = 1 if rng.random() < 0.15 + 0.7 * score else 0
        rows.append(DatasetRow(bar_open_at=at, features=(("synthetic_score", Decimal(str(score))),),
                               label=Decimal(1 if label else -1), usable=True))
    config = DatasetConfig(features=(FeatureDefinition("synthetic_score", "simple_return"),), label=LabelSpec(horizon=4))
    identity = sha256_canonical([[r.bar_open_at, str(r.features[0][1]), str(r.label)] for r in rows])
    return Dataset(schema_version="synthetic-calibration-population-v1", snapshot_id=identity,
        as_of=(start + timedelta(hours=count + 4)).isoformat(), provider="synthetic", product_id=product,
        timeframe="1h", entries_content_hash=identity, config=config, config_hash=config.config_hash,
        indicator_spec_hashes=(), rows=tuple(rows), dataset_hash=identity)


def model_contract():
    return ModelContract(model_id="synthetic-binary-score-v1", version="1", inputs={"synthetic_score": "float [0,1]"},
        outputs={**dict.fromkeys(OUTPUTS), "probabilities": {"event": "forward_return_gt_zero", "method": "synthetic-score-generator-v1", "calibrated": False}},
        horizons_seconds=(14400,), capabilities=("predict", "infer"), limits={"max_rows": 10000},
        implementation_version="synthetic-score-generator-v1", synthetic=True)


def observe(rows, *, product, split, artifact_hash):
    observations = []
    for row in rows:
        end = (datetime.fromisoformat(row.bar_open_at) + timedelta(hours=4)).isoformat()
        observations.append(ScoreObservation(product=product, model_id="synthetic-binary-score-v1",
            artifact_hash=artifact_hash, horizon_seconds=14400, decision_at=row.bar_open_at,
            label_end=end, label_available_at=end, score=float(row.features[0][1]),
            label=int(row.label > 0), split=split, synthetic=True))
    return tuple(observations)


def calibration_demo(product):
    dataset = synthetic_scores(product)
    contract = model_contract()
    artifact_hash = sha256_canonical({"model_contract_hash": contract.identity, "generator_seed": 721 if product == "BTC-USD" else 722})
    config = WalkForwardConfig(min_train_rows=300, validation_rows=40, test_rows=80, step_rows=80, purge_rows=4)
    fold_results, all_raw, all_calibrated, all_labels = [], [], [], []
    for i, (train, validation, test) in enumerate(build_folds(dataset, config=config)):
        training = observe(train, product=product, split="train", artifact_hash=artifact_hash)
        fitting_at = validation[0].bar_open_at
        artifact = fit_isotonic(training, fold_index=i, validation_start=fitting_at, fitted_at=fitting_at, synthetic=True)
        held_out = observe(test, product=product, split="test", artifact_hash=artifact_hash)
        calibrated = [artifact.predict(r.score, **{k: getattr(r, k) for k in
            ("product", "model_id", "artifact_hash", "horizon_seconds", "decision_at")}) for r in held_out]
        raw, labels = [r.score for r in held_out], [r.label for r in held_out]
        row = held_out[-1]
        sample = {"decision_at": row.decision_at,
                  "raw_probability": row.score, "calibrated_probability": calibrated[-1],
                  "event": artifact.event, "horizon_seconds": row.horizon_seconds,
                  "origin": row.model_id, "method": artifact.method, "calibration_hash": artifact.identity,
                  "artifact_hash": row.artifact_hash, "fold_index": i}
        # The issuance identity contains only information available at the decision.
        # Its future test label belongs to diagnostics, never to the prediction hash.
        sample = {**sample, "prediction_hash": sha256_canonical(sample)}
        fold_results.append({"artifact": artifact.to_dict(), "calibration_hash": artifact.identity,
            "test": {"first": held_out[0].decision_at, "last": row.decision_at, "count": len(test),
                     "population_hash": sha256_canonical([r.to_dict() for r in held_out])},
            **evaluate_test(artifact, held_out, as_of=dataset.as_of), "sample": sample})
        all_raw.extend(raw)
        all_calibrated.extend(calibrated)
        all_labels.extend(labels)
    return {"product": product, "synthetic": True, "model": contract.to_dict(), "model_contract_hash": contract.identity,
            "dataset_hash": dataset.dataset_hash, "walk_forward": config.canonical(), "folds": fold_results,
            "raw": reliability(all_raw, all_labels), "calibrated": reliability(all_calibrated, all_labels),
            "sample": fold_results[-1]["sample"], "interpretation": "synthetic test diagnostics; no real calibration or edge claim"}


def bar(at, op, hi, lo, cl):
    identity = sha256_canonical({"synthetic": True, "bar_open_at": at,
        "open": str(op), "high": str(hi), "low": str(lo), "close": str(cl), "volume": "1"})
    return SeriesPoint(bar_open_at=at, open=Decimal(str(op)), high=Decimal(str(hi)), low=Decimal(str(lo)),
                       close=Decimal(str(cl)), volume=Decimal(1), content_sha256=identity)


def build_report():
    calibrations = [calibration_demo(p) for p in ("BTC-USD", "ETH-USD")]
    simulations = []
    for calibrated in calibrations:
        entry_at = (datetime.fromisoformat(calibrated["sample"]["decision_at"]) + timedelta(hours=4)).isoformat()
        strategy = {"take_profit": {"value": "103", "source_hash": sha256_canonical({"strategy": "synthetic-target-v1"}),
                                    "method": "synthetic-target-v1", "provided_at": entry_at}}
        for name, side, candles, levels in (
            ("both-levels-touched", "LONG", [(100, 104, 98, 101)], strategy),
            ("adverse-gap", "LONG", [(97, 100, 96, 98)], None),
            ("favorable-short-gap", "SHORT", [(96, 98, 95, 97)], None),
        ):
            plan = protection_plan(product=calibrated["product"], mode="PAPER", side=side, entry_at=entry_at,
                entry_fill="100", horizon_seconds=14400, source_prediction_hash=calibrated["sample"]["prediction_hash"],
                synthetic=True, strategy=levels)
            bars = [bar(entry_at, *values) for values in candles]
            result = simulate(plan, bars, bar_seconds=3600,
                              as_of=(datetime.fromisoformat(entry_at) + timedelta(hours=4)).isoformat())
            simulations.append({"scenario": name, **result, "result_hash": sha256_canonical(result)})
    return {"schema": "calibration-risk-report-v1", "policy_hash": SPEC_HASH, "synthetic": True,
            "calibrations": calibrations, "simulations": simulations,
            "limitations": ["Synthetic demonstration only; real calibration fitting awaits authorization.",
                "No prediction intervals are provided.", "Unit positions with fixed policy distances; no portfolio or optimized risk claim.",
                "Exit prices are OHLC simulations; crossing time inside a bar is unknown.",
                "Frozen models, risk, backtest and replay references are not modified."]}


def save_report(root):
    root = Path(root).resolve()
    worktree = Path(__file__).resolve().parents[3]
    if not root.is_relative_to(worktree / "var"):
        raise ValueError("demo runtime must be under this worktree var")
    root.mkdir(parents=True, exist_ok=False)
    report = build_report()
    envelope = {"identity": sha256_canonical(report), "report": report}
    with (root / "report.json").open("xb") as stream:
        stream.write(canonical_bytes(envelope))
    return envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    envelope = save_report(args.root)
    print(json.dumps({"synthetic": True, "identity": envelope["identity"], "policy_hash": SPEC_HASH,
                      "folds": sum(len(c["folds"]) for c in envelope["report"]["calibrations"]),
                      "simulations": len(envelope["report"]["simulations"]), "state": "COMPLETE"}))


if __name__ == "__main__":
    main()
