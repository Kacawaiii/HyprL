"""Phase 3D: describing how stable the Phase 3C results are -- and nothing more.

This module is diagnostic. It reads a finished `ModelSelectionEvaluation` and
reports what happened: which candidate won each fold, how close the runner-up
was, whether the validation blocks were big enough for the primary criterion
to mean anything, and how the out-of-sample records score inside temporal
subperiods that were fixed in advance.

What it deliberately does NOT do is decide anything.

There is no `best_candidate`, no `majority_winner`, no `best_scenario`. A
module that counted fold wins and then named a champion would have invented a
second selection rule -- one chosen after seeing test results, which is the
exact failure Phase 3C was built to prevent. Counting is description;
crowning is selection. The distinction is the whole point of this file, and
the tests assert those attributes are absent rather than merely unused.

Two smaller rules follow from the same instinct:

* **Subperiod boundaries never depend on performance.** They are either given
  explicitly or derived from timestamps alone. Choosing where to cut after
  seeing the scores is how a bad period quietly disappears.
* **Nothing is reconstructed by averaging.** Every subperiod metric is
  recomputed from its own records, and the official global figure remains the
  one Phase 3C computed over all concatenated out-of-sample records. The
  average of subperiod correlations is a different statistic, and not one
  anybody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

from scripts.trading_lab.model_selection import (
    ModelSelectionEvaluation,
    evaluate_selection,
    select,
)
from scripts.trading_lab.walk_forward import (
    METRIC_SCHEMA_VERSION,
    Metrics,
    mean_absolute_error,
    rank_ic,
    root_mean_squared_error,
)

ROBUSTNESS_SCHEMA_VERSION = "trading-lab.model-robustness.v1"
ROBUSTNESS_PROTOCOL_VERSION = "descriptive-stability-v1"
ROBUSTNESS_PRECISION = 34
# Same arithmetic as the selection floor: a rank correlation over two points is
# mechanically +/-1. Subperiods below this are reported, and flagged, not hidden.
MIN_RELIABLE_RANK_OBSERVATIONS = 3
MAX_BOUNDARIES = 64
MAX_SCENARIOS = 8

TRADING_COST_ANALYSIS_AVAILABLE = False
TRADING_COST_UNAVAILABLE_REASON = (
    "no causal position or execution policy exists yet: there is no mapping from a "
    "predicted forward return to a position, no entry threshold, no turnover and no "
    "sizing rule, so any fee applied to the raw forward return would be arithmetic "
    "dressed up as a cost model"
)


class ModelRobustnessError(RuntimeError):
    """Raised when a diagnostic cannot be produced honestly."""


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _metrics_canonical(metrics: Metrics) -> dict[str, object]:
    return {
        "rank_ic": _text(metrics.rank_ic),
        "mae": _text(metrics.mae),
        "rmse": _text(metrics.rmse),
        "observations": metrics.observations,
    }


def _metrics_of(records) -> Metrics:
    predictions = tuple(record.prediction for record in records)
    actuals = tuple(record.actual_forward_return for record in records)
    return Metrics(
        rank_ic=rank_ic(predictions, actuals),
        mae=mean_absolute_error(predictions, actuals),
        rmse=root_mean_squared_error(predictions, actuals),
        observations=len(records),
    )


# --- selection stability --------------------------------------------------


@dataclass(frozen=True)
class SelectionStability:
    """Counts. Not a verdict.

    Deliberately exposes no `best_candidate`, `majority_winner` or
    `recommended_candidate`: turning these counts into a choice would be a new
    selection rule, decided after the test metrics were visible.
    """

    fold_count: int
    candidate_counts: tuple[tuple[str, int], ...]
    candidate_fractions: tuple[tuple[str, Decimal], ...]
    transition_count: int
    transition_rate: Decimal | None
    longest_run: int
    selection_reasons: tuple[tuple[str, int], ...]

    def canonical(self) -> dict[str, object]:
        return {
            "fold_count": self.fold_count,
            "candidate_counts": [list(pair) for pair in self.candidate_counts],
            "candidate_fractions": [[name, str(value)]
                                    for name, value in self.candidate_fractions],
            "transition_count": self.transition_count,
            "transition_rate": _text(self.transition_rate),
            "longest_run": self.longest_run,
            "selection_reasons": [list(pair) for pair in self.selection_reasons],
        }


def selection_stability(evaluation: ModelSelectionEvaluation) -> SelectionStability:
    winners = [fold.selected_candidate_id for fold in evaluation.folds]
    reasons = [fold.selection_reason for fold in evaluation.folds]
    if not winners:
        raise ModelRobustnessError("an evaluation without folds has no stability to describe")

    counts: dict[str, int] = {}
    for winner in winners:
        counts[winner] = counts.get(winner, 0) + 1
    reason_counts: dict[str, int] = {}
    for reason in reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    transitions = sum(1 for earlier, later in zip(winners, winners[1:]) if earlier != later)
    longest = current = 1
    for earlier, later in zip(winners, winners[1:]):
        current = current + 1 if earlier == later else 1
        longest = max(longest, current)

    with localcontext() as context:
        context.prec = ROBUSTNESS_PRECISION
        fractions = tuple(
            (name, Decimal(count) / Decimal(len(winners))) for name, count in sorted(counts.items())
        )
        rate = (Decimal(transitions) / Decimal(len(winners) - 1)) if len(winners) > 1 else None
    return SelectionStability(
        fold_count=len(winners),
        candidate_counts=tuple(sorted(counts.items())),
        candidate_fractions=fractions,
        transition_count=transitions,
        transition_rate=rate,
        longest_run=longest,
        selection_reasons=tuple(sorted(reason_counts.items())),
    )


# --- validation geometry --------------------------------------------------


@dataclass(frozen=True)
class FoldGeometry:
    fold_index: int
    nominal_validation_rows: int
    effective_validation_rows: int
    rank_ic_observation_count: int
    rank_ic_defined: tuple[tuple[str, bool], ...]
    selection_reason: str

    def canonical(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "nominal_validation_rows": self.nominal_validation_rows,
            "effective_validation_rows": self.effective_validation_rows,
            "rank_ic_observation_count": self.rank_ic_observation_count,
            "rank_ic_defined": [[name, flag] for name, flag in self.rank_ic_defined],
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class ValidationGeometry:
    folds: tuple[FoldGeometry, ...]
    minimum_effective_validation_rows: int
    folds_selected_by_rank_ic: int
    folds_selected_by_mae: int
    folds_selected_by_rmse: int
    folds_selected_by_tie: int

    def canonical(self) -> dict[str, object]:
        return {
            "folds": [fold.canonical() for fold in self.folds],
            "minimum_effective_validation_rows": self.minimum_effective_validation_rows,
            "folds_selected_by_rank_ic": self.folds_selected_by_rank_ic,
            "folds_selected_by_mae": self.folds_selected_by_mae,
            "folds_selected_by_rmse": self.folds_selected_by_rmse,
            "folds_selected_by_tie": self.folds_selected_by_tie,
        }


def validation_geometry(evaluation: ModelSelectionEvaluation) -> ValidationGeometry:
    """Makes it visible whether rank-ic-primary actually did any work."""
    nominal = evaluation.spec.walk_forward_config["validation_rows"]
    entries = tuple(
        FoldGeometry(
            fold_index=fold.fold_index,
            nominal_validation_rows=nominal,
            effective_validation_rows=fold.validation_rows,
            rank_ic_observation_count=(fold.candidates[0].observations
                                       if fold.candidates else 0),
            rank_ic_defined=tuple((entry.candidate_id, entry.rank_ic is not None)
                                  for entry in fold.candidates),
            selection_reason=fold.selection_reason,
        )
        for fold in evaluation.folds
    )
    if not entries:
        raise ModelRobustnessError("an evaluation without folds has no geometry to describe")
    reasons = [fold.selection_reason for fold in evaluation.folds]
    return ValidationGeometry(
        folds=entries,
        minimum_effective_validation_rows=min(entry.effective_validation_rows
                                              for entry in entries),
        folds_selected_by_rank_ic=reasons.count("rank_ic"),
        folds_selected_by_mae=reasons.count("mae"),
        folds_selected_by_rmse=reasons.count("rmse"),
        folds_selected_by_tie=reasons.count("candidate_id"),
    )


# --- selection margins ----------------------------------------------------


@dataclass(frozen=True)
class SelectionMargin:
    """How close the fold was. Never a single scalar mixing three criteria.

    The rule is lexicographic, so only the criterion that actually decided is
    meaningful; blending rank_ic, MAE and RMSE into one number would invent a
    quantity the rule never uses.
    """

    fold_index: int
    winning_criterion: str
    winner_candidate_id: str
    runner_up_candidate_id: str | None
    winner_value: Decimal | None
    runner_up_value: Decimal | None
    difference: Decimal | None

    def canonical(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "winning_criterion": self.winning_criterion,
            "winner_candidate_id": self.winner_candidate_id,
            "runner_up_candidate_id": self.runner_up_candidate_id,
            "winner_value": _text(self.winner_value),
            "runner_up_value": _text(self.runner_up_value),
            "difference": _text(self.difference),
        }


_LOWER_IS_BETTER = {"mae", "rmse"}


def selection_margins(evaluation: ModelSelectionEvaluation) -> tuple[SelectionMargin, ...]:
    margins: list[SelectionMargin] = []
    for fold in evaluation.folds:
        winner = next(entry for entry in fold.candidates
                      if entry.candidate_id == fold.selected_candidate_id)
        others = tuple(entry for entry in fold.candidates
                       if entry.candidate_id != fold.selected_candidate_id)
        if not others:
            margins.append(SelectionMargin(
                fold_index=fold.fold_index, winning_criterion=fold.selection_reason,
                winner_candidate_id=winner.candidate_id, runner_up_candidate_id=None,
                winner_value=None, runner_up_value=None, difference=None))
            continue
        # The runner-up is whoever the SAME public rule would pick next: no
        # private ordering is reimplemented here.
        runner_up_id, _ = select(others)
        runner_up = next(entry for entry in others if entry.candidate_id == runner_up_id)

        criterion = fold.selection_reason
        if criterion in {"candidate_id", "sole_candidate"}:
            # A tie broken by identity has no numeric distance, and pretending
            # it is zero would read as "they were equal on a criterion".
            winner_value = runner_up_value = difference = None
        else:
            winner_value = getattr(winner, criterion)
            runner_up_value = getattr(runner_up, criterion)
            if winner_value is None or runner_up_value is None:
                difference = None
            else:
                with localcontext() as context:
                    context.prec = ROBUSTNESS_PRECISION
                    difference = (runner_up_value - winner_value
                                  if criterion in _LOWER_IS_BETTER
                                  else winner_value - runner_up_value)
        margins.append(SelectionMargin(
            fold_index=fold.fold_index, winning_criterion=criterion,
            winner_candidate_id=winner.candidate_id, runner_up_candidate_id=runner_up_id,
            winner_value=winner_value, runner_up_value=runner_up_value,
            difference=difference))
    return tuple(margins)


# --- temporal subperiods --------------------------------------------------


@dataclass(frozen=True)
class SubperiodMetrics:
    start: str
    end: str
    metrics: Metrics
    rank_ic_degenerate: bool

    def canonical(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "metrics": _metrics_canonical(self.metrics),
            "rank_ic_degenerate": self.rank_ic_degenerate,
        }


@dataclass(frozen=True)
class RankStability:
    positive_periods: int
    negative_periods: int
    zero_periods: int
    undefined_periods: int
    degenerate_periods: int

    def canonical(self) -> dict[str, int]:
        return {
            "positive_periods": self.positive_periods,
            "negative_periods": self.negative_periods,
            "zero_periods": self.zero_periods,
            "undefined_periods": self.undefined_periods,
            "degenerate_periods": self.degenerate_periods,
        }


def _require_boundaries(boundaries) -> tuple[str, ...]:
    values = tuple(boundaries)
    if len(values) < 2 or len(values) > MAX_BOUNDARIES:
        raise ModelRobustnessError(f"between 2 and {MAX_BOUNDARIES} boundaries are required")
    parsed = []
    for value in values:
        if not isinstance(value, str):
            raise ModelRobustnessError(f"boundary {value!r} must be an ISO-8601 string")
        try:
            parsed.append(datetime.fromisoformat(value))
        except ValueError as error:
            raise ModelRobustnessError(f"boundary {value!r} is not ISO-8601") from error
    if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
        raise ModelRobustnessError("boundaries must be strictly increasing")
    return values


def subperiod_metrics(evaluation: ModelSelectionEvaluation, *, boundaries
                      ) -> tuple[SubperiodMetrics, ...]:
    """Score the out-of-sample records inside half-open intervals fixed in advance.

    The boundaries are an input. Nothing here inspects performance before
    deciding where to cut -- that is how a disappointing stretch quietly stops
    being reported.
    """
    edges = _require_boundaries(boundaries)
    parsed = [datetime.fromisoformat(edge) for edge in edges]
    periods: list[SubperiodMetrics] = []
    for start, end, start_text, end_text in zip(parsed, parsed[1:], edges, edges[1:]):
        inside = tuple(
            record for record in evaluation.oos_records
            if start <= datetime.fromisoformat(record.bar_open_at) < end
        )
        metrics = _metrics_of(inside)
        periods.append(SubperiodMetrics(
            start=start_text, end=end_text, metrics=metrics,
            rank_ic_degenerate=(metrics.rank_ic is not None
                                and metrics.observations < MIN_RELIABLE_RANK_OBSERVATIONS),
        ))
    return tuple(periods)


def equal_time_subperiods(evaluation: ModelSelectionEvaluation, *, parts: int
                          ) -> tuple[str, ...]:
    """Boundaries from timestamps alone: equal spans of TIME, never of score.

    Note these boundaries move when the record range changes, so they are a
    reporting convenience -- not a basis for proving that history stayed put.
    Use explicit boundaries for that.
    """
    if type(parts) is not int or parts < 1 or parts > MAX_BOUNDARIES - 1:
        raise ModelRobustnessError(f"parts must be an int in 1..{MAX_BOUNDARIES - 1}")
    if not evaluation.oos_records:
        raise ModelRobustnessError("no out-of-sample records to split")
    moments = [datetime.fromisoformat(record.bar_open_at)
               for record in evaluation.oos_records]
    first, last = min(moments), max(moments)
    span = (last - first) / parts
    edges = [first + span * index for index in range(parts)]
    # Intervals are half-open, so the closing edge must sit strictly after the
    # last record or that record would fall outside every period.
    edges.append(last + timedelta(microseconds=1))
    return tuple(edge.isoformat() for edge in edges)


def rank_stability(periods) -> RankStability:
    """Counts, never a score. `None` stays `None`; it is not a zero."""
    positive = negative = zero = undefined = degenerate = 0
    for period in periods:
        value = period.metrics.rank_ic
        if value is None:
            undefined += 1
        elif value > 0:
            positive += 1
        elif value < 0:
            negative += 1
        else:
            zero += 1
        if period.rank_ic_degenerate:
            degenerate += 1
    return RankStability(positive_periods=positive, negative_periods=negative,
                         zero_periods=zero, undefined_periods=undefined,
                         degenerate_periods=degenerate)


# --- declared scenarios ---------------------------------------------------


@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str
    candidates: tuple

    def canonical(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "candidates": [spec.canonical()
                           for spec in sorted(self.candidates,
                                              key=lambda spec: spec.candidate_id)],
        }


@dataclass(frozen=True)
class ScenarioResult:
    """A described scenario. There is deliberately no `best_scenario` anywhere."""

    scenario_id: str
    selection_spec_hash: str
    selection_results_hash: str
    stability: SelectionStability
    global_test_metrics: Metrics

    def canonical(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "selection_spec_hash": self.selection_spec_hash,
            "selection_results_hash": self.selection_results_hash,
            "stability": self.stability.canonical(),
            "global_test_metrics": _metrics_canonical(self.global_test_metrics),
        }


def _canonical_scenarios(scenarios) -> tuple[RobustnessScenario, ...]:
    ordered = tuple(scenarios)
    if len(ordered) > MAX_SCENARIOS:
        raise ModelRobustnessError(f"at most {MAX_SCENARIOS} scenarios may be declared")
    identifiers = [scenario.scenario_id for scenario in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ModelRobustnessError(f"scenario ids must be unique, got {identifiers}")
    return tuple(sorted(ordered, key=lambda scenario: scenario.scenario_id))


# --- the report -----------------------------------------------------------


@dataclass(frozen=True)
class RobustnessSpec:
    selection_spec_hash: str
    selection_results_hash: str
    boundaries: tuple[str, ...]
    scenarios: tuple[dict, ...]
    protocol_version: str
    metric_version: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": ROBUSTNESS_SCHEMA_VERSION,
            "selection_spec_hash": self.selection_spec_hash,
            "selection_results_hash": self.selection_results_hash,
            "boundaries": list(self.boundaries),
            "scenarios": [dict(entry) for entry in self.scenarios],
            "protocol_version": self.protocol_version,
            "metric_version": self.metric_version,
        }

    @property
    def spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class RobustnessReport:
    spec: RobustnessSpec
    spec_hash: str
    stability: SelectionStability
    geometry: ValidationGeometry
    margins: tuple[SelectionMargin, ...]
    subperiods: tuple[SubperiodMetrics, ...]
    rank_stability: RankStability
    scenarios: tuple[ScenarioResult, ...]
    global_test_metrics: Metrics
    trading_cost_analysis_available: bool
    trading_cost_unavailable_reason: str
    results_hash: str


def analyse_robustness(evaluation: ModelSelectionEvaluation, *, boundaries,
                       scenarios=(), dataset=None, config=None) -> RobustnessReport:
    """Describe a finished selection. Decide nothing."""
    edges = _require_boundaries(boundaries)
    declared = _canonical_scenarios(scenarios)
    if declared and (dataset is None or config is None):
        raise ModelRobustnessError("scenarios need the dataset and walk-forward config")

    periods = subperiod_metrics(evaluation, boundaries=edges)
    results: list[ScenarioResult] = []
    for scenario in declared:
        # Every scenario goes back through validation-only selection; there is
        # no shortcut that would let a scenario be scored straight on test.
        outcome = evaluate_selection(dataset, config=config, candidates=scenario.candidates)
        results.append(ScenarioResult(
            scenario_id=scenario.scenario_id,
            selection_spec_hash=outcome.spec_hash,
            selection_results_hash=outcome.results_hash,
            stability=selection_stability(outcome),
            global_test_metrics=outcome.global_test_metrics,
        ))

    spec = RobustnessSpec(
        selection_spec_hash=evaluation.spec_hash,
        selection_results_hash=evaluation.results_hash,
        boundaries=edges,
        scenarios=tuple(scenario.canonical() for scenario in declared),
        protocol_version=ROBUSTNESS_PROTOCOL_VERSION,
        metric_version=METRIC_SCHEMA_VERSION,
    )
    stability = selection_stability(evaluation)
    geometry = validation_geometry(evaluation)
    margins = selection_margins(evaluation)
    results_hash = _sha256_canonical(
        {
            "spec": spec.canonical(),
            "stability": stability.canonical(),
            "geometry": geometry.canonical(),
            "margins": [margin.canonical() for margin in margins],
            "subperiods": [period.canonical() for period in periods],
            "scenarios": [result.canonical() for result in results],
        }
    )
    return RobustnessReport(
        spec=spec,
        spec_hash=spec.spec_hash,
        stability=stability,
        geometry=geometry,
        margins=margins,
        subperiods=periods,
        rank_stability=rank_stability(periods),
        scenarios=tuple(results),
        # The official global figure stays the one Phase 3C computed over every
        # concatenated record; it is carried through, never rebuilt by averaging.
        global_test_metrics=evaluation.global_test_metrics,
        trading_cost_analysis_available=TRADING_COST_ANALYSIS_AVAILABLE,
        trading_cost_unavailable_reason=TRADING_COST_UNAVAILABLE_REASON,
        results_hash=results_hash,
    )
