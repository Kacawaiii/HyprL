"""Phase 3C: choosing between candidates using the validation block only.

Phase 2D cut every fold into three blocks and Phase 3A/3B put real models on
the train/test pair, leaving the middle block unused. This module finally
spends it -- and the whole design exists to answer one question honestly:

    could the test block have influenced which model was chosen?

The answer is made structural rather than promised. `validate_candidates` and
`select` do not take a test block at all; there is no parameter through which
test information could arrive. The fold driver calls them first, and only
touches the test rows afterwards, to predict on them. A reviewer does not have
to trust the ordering: the signatures make the earlier steps incapable of
seeing what comes later.

Two further points of honesty:

* **The model scored on validation is not the model that predicts test.** The
  winner is refitted from scratch on train + validation, so the test block is
  faced by a model that has seen strictly more data than the one that won the
  comparison. Both fitted hashes are recorded separately -- pretending they
  are the same model would be a comfortable lie.
* **The refit is only legal if validation labels were knowable.** A label at
  T reaches T+h, so train+validation is admissible only when every validation
  label window closes strictly before the test block opens. That is checked
  per fold in market time, not assumed from Phase 2D's purge.

Selection may pick a different candidate in every fold. That is the honest
output of a walk-forward meta-protocol; collapsing it into "candidate X won
7 folds so X is the model" would be a new, unvalidated selection rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.walk_forward import (
    METRIC_SCHEMA_VERSION,
    Metrics,
    PredictionRecord,
    build_folds,
    mean_absolute_error,
    rank_ic,
    root_mean_squared_error,
)

SELECTION_SCHEMA_VERSION = "trading-lab.model-selection.v1"
SELECTION_RULE_VERSION = "validation-rank-ic-mae-rmse-v1"
# A Spearman correlation over two non-constant points is mechanically +/-1, so a
# two-row validation block makes the primary criterion tie every time and lets
# MAE decide in silence. Three is the smallest count at which the ranking can
# carry information; the protocol refuses less rather than degrading quietly.
MIN_VALIDATION_OBSERVATIONS = 3
REFIT_POLICY_VERSION = "refit-train-plus-validation-v1"
MAX_CANDIDATES = 32


class ModelSelectionError(RuntimeError):
    """Raised when a selection cannot be made safely or reproducibly."""


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _decimal_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class CandidateSpec:
    """A candidate's identity plus a way to make fresh, unfitted instances.

    The factory is deliberately excluded from equality and from every hash:
    a bound method or lambda has no stable identity across processes, and
    hashing it would make results irreproducible for no benefit. Identity
    comes from the model's own deterministic spec hash.
    """

    candidate_id: str
    model_spec_hash: str
    factory: Callable[[], object] = field(compare=False, repr=False)

    def canonical(self) -> dict[str, str]:
        return {"candidate_id": self.candidate_id, "model_spec_hash": self.model_spec_hash}


def candidate(candidate_id: str, factory: Callable[[], object]) -> CandidateSpec:
    """Build a CandidateSpec, proving the factory is fresh and stable first."""
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ModelSelectionError("candidate_id must be a non-empty string")
    first, second = factory(), factory()
    if first is second:
        raise ModelSelectionError(
            f"factory for {candidate_id!r} returned the same instance twice; "
            "each fold needs an unfitted model"
        )
    hashes = {getattr(model, "model_spec_hash", None) for model in (first, second)}
    if len(hashes) != 1 or None in hashes:
        raise ModelSelectionError(
            f"factory for {candidate_id!r} must produce a stable model_spec_hash"
        )
    return CandidateSpec(candidate_id=candidate_id, model_spec_hash=hashes.pop(),
                         factory=factory)


def _canonical_candidates(candidates) -> tuple[CandidateSpec, ...]:
    ordered = tuple(candidates)
    if not ordered or len(ordered) > MAX_CANDIDATES:
        raise ModelSelectionError(f"between 1 and {MAX_CANDIDATES} candidates are required")
    identifiers = [spec.candidate_id for spec in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ModelSelectionError(f"candidate ids must be unique, got {identifiers}")
    # Sorting here is what makes the caller's argument order irrelevant, to the
    # comparison and to every hash derived from it.
    return tuple(sorted(ordered, key=lambda spec: spec.candidate_id))


@dataclass(frozen=True)
class CandidateValidation:
    candidate_id: str
    model_spec_hash: str
    rank_ic: Decimal | None
    mae: Decimal | None
    rmse: Decimal | None
    observations: int
    selection_fit_hash: str | None

    def canonical(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "model_spec_hash": self.model_spec_hash,
            "rank_ic": _decimal_or_none(self.rank_ic),
            "mae": _decimal_or_none(self.mae),
            "rmse": _decimal_or_none(self.rmse),
            "observations": self.observations,
            "selection_fit_hash": self.selection_fit_hash,
        }


@dataclass(frozen=True)
class SelectedFoldModel:
    fold_index: int
    candidates: tuple[CandidateValidation, ...]
    selected_candidate_id: str
    selection_reason: str
    selection_fit_hash: str | None
    final_fit_hash: str | None
    train_rows: int
    validation_rows: int
    test_records: tuple[PredictionRecord, ...]
    test_metrics: Metrics


@dataclass(frozen=True)
class SelectionSpec:
    dataset_hash: str
    walk_forward_config: dict[str, object]
    candidates: tuple[dict[str, str], ...]
    selection_rule_version: str
    refit_policy_version: str
    metric_version: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "dataset_hash": self.dataset_hash,
            "walk_forward_config": self.walk_forward_config,
            "candidates": [dict(entry) for entry in self.candidates],
            "selection_rule_version": self.selection_rule_version,
            "refit_policy_version": self.refit_policy_version,
            "metric_version": self.metric_version,
        }

    @property
    def spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class ModelSelectionEvaluation:
    spec: SelectionSpec
    spec_hash: str
    folds: tuple[SelectedFoldModel, ...]
    oos_records: tuple[PredictionRecord, ...]
    global_test_metrics: Metrics
    results_hash: str


# --- prediction plumbing --------------------------------------------------


def _require_predictions(values, expected: int) -> tuple[Decimal, ...]:
    predictions = tuple(values)
    if len(predictions) != expected:
        raise ModelSelectionError(
            f"candidate returned {len(predictions)} predictions for {expected} rows"
        )
    for value in predictions:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ModelSelectionError(f"prediction {value!r} is not a finite Decimal")
    return predictions


def _metrics_of(predictions, actuals) -> Metrics:
    """Assembled from the Phase 2D metric functions -- none of them reimplemented."""
    return Metrics(
        rank_ic=rank_ic(predictions, actuals),
        mae=mean_absolute_error(predictions, actuals),
        rmse=root_mean_squared_error(predictions, actuals),
        observations=len(predictions),
    )


def _fitted_hash(model) -> str | None:
    fitted = getattr(model, "fitted", None)
    return getattr(fitted, "fitted_model_hash", None)


# --- step A/B: score every candidate on validation ------------------------


def validate_candidates(train_rows, validation_rows, *, candidates
                        ) -> tuple[CandidateValidation, ...]:
    """Fit each candidate on TRAIN and score it on VALIDATION.

    This function has no test parameter. That is the point.
    """
    specs = _canonical_candidates(candidates)
    train = tuple(train_rows)
    validation = tuple(validation_rows)
    if not train or not validation:
        raise ModelSelectionError("selection needs a non-empty train and validation block")
    if len(validation) < MIN_VALIDATION_OBSERVATIONS:
        # Checked on the block actually produced for this fold, never on the
        # nominal parameters: the Phase 2D purge is expressed in market time and
        # gaps remove a different number of rows than the arithmetic suggests.
        raise ModelSelectionError(
            f"rank-ic-primary selection needs at least {MIN_VALIDATION_OBSERVATIONS} "
            f"effective validation observations, this fold scored {len(validation)}"
        )
    actuals = tuple(row.label for row in validation)
    if any(actual is None for actual in actuals):
        raise ModelSelectionError("validation rows must carry labels; nothing is imputed")

    results: list[CandidateValidation] = []
    for spec in specs:
        model = spec.factory()
        model.fit(train)
        predictions = _require_predictions(model.predict(validation), len(validation))
        metrics = _metrics_of(predictions, actuals)
        results.append(
            CandidateValidation(
                candidate_id=spec.candidate_id,
                model_spec_hash=spec.model_spec_hash,
                rank_ic=metrics.rank_ic,
                mae=metrics.mae,
                rmse=metrics.rmse,
                observations=metrics.observations,
                selection_fit_hash=_fitted_hash(model),
            )
        )
    return tuple(results)


# --- step C: the frozen selection rule ------------------------------------

_CRITERIA = ("rank_ic", "rank_ic", "mae", "rmse", "candidate_id")


def _selection_key(result: CandidateValidation):
    """Lexicographic, fully deterministic, no randomness anywhere.

    A defined rank_ic beats an undefined one; among defined ones, higher wins.
    `None` is never silently read as zero -- it sorts as "cannot be compared
    on this criterion", which is a different statement from "scored 0".
    """
    if result.mae is None or result.rmse is None:
        raise ModelSelectionError(
            f"candidate {result.candidate_id!r} has no comparable validation metrics"
        )
    undefined = result.rank_ic is None
    return (
        1 if undefined else 0,
        Decimal(0) if undefined else -result.rank_ic,
        result.mae,
        result.rmse,
        result.candidate_id,
    )


def select(results) -> tuple[str, str]:
    """Return the winning candidate_id and the criterion that decided it."""
    ranked = sorted(results, key=_selection_key)
    if not ranked:
        raise ModelSelectionError("no candidate to select from")
    if len(ranked) == 1:
        return ranked[0].candidate_id, "sole_candidate"
    best, runner_up = _selection_key(ranked[0]), _selection_key(ranked[1])
    for position, (left, right) in enumerate(zip(best, runner_up)):
        if left != right:
            return ranked[0].candidate_id, _CRITERIA[position]
    raise ModelSelectionError("two candidates share an identical selection key")


# --- step D: the refit, and the temporal contract that permits it ---------


def _require_validation_known_before_test(validation_rows, test_rows, *, horizon: int,
                                          duration: timedelta) -> None:
    """Refitting on validation is only legal if its labels were already knowable."""
    if not validation_rows or not test_rows:
        raise ModelSelectionError("a fold needs both a validation and a test block")
    latest_label_end = max(
        datetime.fromisoformat(row.bar_open_at) + duration * horizon for row in validation_rows
    )
    first_test = min(datetime.fromisoformat(row.bar_open_at) for row in test_rows)
    if latest_label_end >= first_test:
        raise ModelSelectionError(
            f"refit policy violated: a validation label window closes at {latest_label_end} "
            f"but the test block opens at {first_test}"
        )


# --- the fold driver ------------------------------------------------------


def select_over_folds(folds, *, candidates, horizon: int, duration: timedelta
                      ) -> tuple[SelectedFoldModel, ...]:
    specs = {spec.candidate_id: spec for spec in _canonical_candidates(candidates)}
    selected: list[SelectedFoldModel] = []
    for fold_index, (train, validation, test) in enumerate(folds):
        # A + B + C -- nothing below this point has been given the test block
        results = validate_candidates(train, validation, candidates=specs.values())
        winner_id, reason = select(results)

        # D -- the refit is legal only under the temporal contract
        _require_validation_known_before_test(validation, test, horizon=horizon,
                                              duration=duration)
        final = specs[winner_id].factory()
        final.fit(tuple(train) + tuple(validation))

        # E -- only now does the test block get touched
        test_rows = tuple(test)
        predictions = _require_predictions(final.predict(test_rows), len(test_rows))
        records = tuple(
            PredictionRecord(
                fold_index=fold_index,
                bar_open_at=row.bar_open_at,
                prediction=prediction,
                actual_forward_return=row.label,
            )
            for row, prediction in zip(test_rows, predictions)
        )
        chosen = next(result for result in results if result.candidate_id == winner_id)
        selected.append(
            SelectedFoldModel(
                fold_index=fold_index,
                candidates=results,
                selected_candidate_id=winner_id,
                selection_reason=reason,
                selection_fit_hash=chosen.selection_fit_hash,
                final_fit_hash=_fitted_hash(final),
                train_rows=len(train),
                validation_rows=len(validation),
                test_records=records,
                test_metrics=_metrics_of(
                    tuple(record.prediction for record in records),
                    tuple(record.actual_forward_return for record in records),
                ),
            )
        )
    return tuple(selected)


def evaluate_selection(dataset, *, config, candidates) -> ModelSelectionEvaluation:
    """Run validation-only selection across every walk-forward fold."""
    if dataset.timeframe not in TIMEFRAME_DURATIONS:
        raise ModelSelectionError(f"unsupported timeframe {dataset.timeframe!r}")
    specs = _canonical_candidates(candidates)
    folds = build_folds(dataset, config=config)
    selected = select_over_folds(
        folds,
        candidates=specs,
        horizon=dataset.config.label.horizon,
        duration=TIMEFRAME_DURATIONS[dataset.timeframe],
    )
    spec = SelectionSpec(
        dataset_hash=dataset.dataset_hash,
        walk_forward_config=config.canonical(),
        candidates=tuple(entry.canonical() for entry in specs),
        selection_rule_version=SELECTION_RULE_VERSION,
        refit_policy_version=REFIT_POLICY_VERSION,
        metric_version=METRIC_SCHEMA_VERSION,
    )
    oos_records = tuple(record for fold in selected for record in fold.test_records)
    results_hash = _sha256_canonical(
        {
            "spec": spec.canonical(),
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "selected_candidate_id": fold.selected_candidate_id,
                    "selection_reason": fold.selection_reason,
                    "selection_fit_hash": fold.selection_fit_hash,
                    "final_fit_hash": fold.final_fit_hash,
                    "candidates": [entry.canonical() for entry in fold.candidates],
                    "records": [
                        {
                            "bar_open_at": record.bar_open_at,
                            "prediction": str(record.prediction),
                            "actual_forward_return": str(record.actual_forward_return),
                        }
                        for record in fold.test_records
                    ],
                }
                for fold in selected
            ],
        }
    )
    return ModelSelectionEvaluation(
        spec=spec,
        spec_hash=spec.spec_hash,
        folds=selected,
        oos_records=oos_records,
        # Recomputed on the concatenated out-of-sample set, exactly as in 2D.
        global_test_metrics=_metrics_of(
            tuple(record.prediction for record in oos_records),
            tuple(record.actual_forward_return for record in oos_records),
        ),
        results_hash=results_hash,
    )
