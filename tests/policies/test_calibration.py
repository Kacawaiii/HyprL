"""Synthetic causal calibration; no real data or model training."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scripts.trading_lab.policies.calibration import (
    ScoreObservation, CalibrationArtifact, evaluate_test, fit_isotonic, reliability,
)
from scripts.trading_lab.policies.spec import SPEC_HASH, policy_spec
from scripts.trading_lab.sources.canonical import sha256_canonical

START = datetime(2026, 6, 1, tzinfo=timezone.utc)
FIT = (START + timedelta(hours=104)).isoformat()


@pytest.fixture
def rows():
    positives = (10, 0, 12, 15, 19)
    return tuple(ScoreObservation(product="BTC-USD", model_id="synthetic-score-v1",
        artifact_hash="a" * 64, horizon_seconds=14400,
        decision_at=(START + timedelta(hours=i)).isoformat(),
        label_end=(START + timedelta(hours=i + 4)).isoformat(),
        label_available_at=(START + timedelta(hours=i + 4)).isoformat(),
        score=(i // 20 + 1) / 10, label=int(i % 20 < positives[i // 20]),
        split="train", synthetic=True) for i in range(100))


def fitted(rows):
    return fit_isotonic(rows, fold_index=0, fitted_at=FIT, validation_start=FIT, synthetic=True)


def predict(artifact, score, **changes):
    args = dict(product="BTC-USD", model_id="synthetic-score-v1", artifact_hash="a" * 64,
                horizon_seconds=14400, decision_at=FIT)
    return artifact.predict(score, **{**args, **changes})


def test_pav_pools_violations_preserves_duplicates_and_step_endpoints(rows):
    artifact = fitted(rows)
    assert artifact.knots == ((.1, .25), (.2, .25), (.3, .6), (.4, .75), (.5, .95))
    assert [predict(artifact, s) for s in (0, .29, .3, .49, 1)] == [.25, .25, .6, .75, .95]
    assert artifact.train["count"] == 100
    assert artifact.train["class_counts"] == (44, 56)
    assert CalibrationArtifact.from_dict(artifact.to_dict()).identity == artifact.identity
    assert fitted(tuple(reversed(rows))).identity == artifact.identity
    assert sha256_canonical(policy_spec()) == SPEC_HASH


@pytest.mark.parametrize("change", [
    {"split": "validation"}, {"split": "test"}, {"synthetic": False},
    {"product": "ETH-USD"}, {"artifact_hash": "b" * 64},
    {"label_available_at": FIT},
])
def test_fit_refuses_leakage_and_mixed_origins(rows, change):
    with pytest.raises(ValueError):
        fitted((replace(rows[0], **change), *rows[1:]))


def test_real_fitting_duplicate_decisions_and_small_populations_refused(rows):
    with pytest.raises(ValueError, match="authorization"):
        fit_isotonic(rows, fold_index=0, fitted_at=FIT, validation_start=FIT, synthetic=False)
    for bad in (rows[:99], (rows[0], *rows[:-1]), tuple(replace(r, label=1) for r in rows),
                tuple(replace(r, score=.5) for r in rows)):
        with pytest.raises(ValueError):
            fitted(bad)


@pytest.mark.parametrize("change", [
    {"product": "ETH-USD"}, {"model_id": "other"}, {"artifact_hash": "b" * 64},
    {"horizon_seconds": 3600}, {"decision_at": "2026-06-01T00:00:00Z"},
])
def test_predict_refuses_incompatible_bindings(rows, change):
    with pytest.raises(ValueError):
        predict(fitted(rows), .5, **change)


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_non_probabilistic_scores_refused(rows, score):
    with pytest.raises(ValueError):
        predict(fitted(rows), score)


def test_realized_test_only_and_small_diagnostics_have_no_metric(rows):
    artifact = fitted(rows)
    test = replace(rows[0], decision_at=FIT, label_end=(START + timedelta(hours=108)).isoformat(),
                   label_available_at=(START + timedelta(hours=108)).isoformat(), split="test")
    report = evaluate_test(artifact, [test], as_of=test.label_end)
    assert report["calibrated"]["state"] == "REFUSED_SMALL_SAMPLE"
    assert report["calibrated"]["brier"] is None
    assert report["calibrated"]["bins"] == []
    for bad, clock in (([test], FIT), ([replace(test, split="train")], test.label_end),
                       ([replace(test, split="validation")], test.label_end), ([test, test], test.label_end)):
        with pytest.raises(ValueError):
            evaluate_test(artifact, bad, as_of=clock)


def test_brier_decomposition_preserves_binning_residual_and_empty_bins():
    p = [.21, .29] * 30
    y = [0, 1] * 30
    report = reliability(p, y)
    assert report["state"] == "AVAILABLE"
    assert report["binned_brier"] == pytest.approx(report["reliability"] - report["resolution"] + report["uncertainty"])
    assert report["brier"] == pytest.approx(report["binned_brier"] + report["binning_residual"])
    assert abs(report["binning_residual"]) > 0
    assert report["bins"][0]["mean_probability"] is None
    assert sum(b["count"] for b in report["bins"]) == 60
    assert reliability([1] * 60, [1] * 60)["state"] == "REFUSED_SMALL_SAMPLE"


def test_artifacts_are_immutable_and_old_hashes_stay_rejected(rows):
    artifact = fitted(rows)
    with pytest.raises(TypeError):
        artifact.train["count"] = 200
    with pytest.raises(ValueError):
        replace(artifact, policy_hash="b" * 64)
    with pytest.raises(ValueError):
        replace(artifact, train={**artifact.train, "class_counts": [30, 30]})


def test_protected_dates_are_refused_without_reading_data(rows):
    from scripts.trading_lab.research_protection import protection_table
    at = datetime.fromisoformat(protection_table()["BTC-USD"][0]["start"].replace("Z", "+00:00"))
    with pytest.raises(ValueError, match="protected"):
        replace(rows[0], decision_at=at.isoformat(), label_end=(at + timedelta(hours=4)).isoformat(),
                label_available_at=(at + timedelta(hours=4)).isoformat())
