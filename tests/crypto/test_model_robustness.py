"""Phase 3D: diagnostics that describe Phase 3C without ever deciding anything.

The dangerous failure here is not a wrong number. It is a module that quietly
becomes a second selection rule -- counting fold wins and naming a champion,
or moving a subperiod boundary until the picture improves. Most of what
follows tests for the absence of those powers.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json

import pytest

# Phase 3 is the ML layer: every test here needs the optional [ml] extra.
# Marked at module granularity -- see docs/TRADING_LAB_PHASE3.md.
pytestmark = pytest.mark.ml


GRID = datetime(2026, 11, 2, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
T_PUB = datetime(2026, 11, 20, tzinfo=timezone.utc)
HORIZON = 4


@pytest.fixture
def robustness():
    return importlib.import_module("scripts.trading_lab.model_robustness")


@pytest.fixture
def selection():
    return importlib.import_module("scripts.trading_lab.model_selection")


@pytest.fixture
def models():
    return importlib.import_module("scripts.trading_lab.models")


@pytest.fixture
def walk_forward():
    return importlib.import_module("scripts.trading_lab.walk_forward")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_dataset(tmp_path, *, name, count=120):
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    series_module = importlib.import_module("scripts.trading_lab.market_series")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + HOUR * index for index in range(count)]
    closes = [Decimal(220 + (index * 31) % 47 - (index % 7) * 3) for index in range(count)]
    rows = [
        [int(o.timestamp()), str(c - Decimal(2 + index % 4)), str(c + Decimal(1 + index % 6)),
         str(c), str(c), "1.0"]
        for index, (o, c) in enumerate(zip(opens, closes))
    ]
    store.ingest_coinbase_response(
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
        product_id="BTC-USD", timeframe="1h",
        available_at=_iso(T_PUB), ingested_at=_iso(T_PUB + timedelta(seconds=1)),
    )
    connection = store._connect()
    try:
        snapshot = snapshots._materialize_snapshot(
            connection, provider="coinbase_exchange_rest", product_id="BTC-USD",
            timeframe="1h", range_start=_iso(opens[0]),
            range_end=_iso(opens[-1] + HOUR), as_of=_iso(T_PUB + HOUR),
        )
        series = series_module.load_market_series(connection, snapshot_id=snapshot.snapshot_id)
    finally:
        connection.close()
    config = dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("sma5", "sma", (("period", 5),)),
            dataset_module.FeatureDefinition("rsi3", "rsi", (("period", 3),)),
            dataset_module.FeatureDefinition("atr4", "atr", (("period", 4),)),
        ),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    return dataset_module.build_dataset(series, config=config)


@pytest.fixture
def dataset(tmp_path):
    return _build_dataset(tmp_path, name="robust")


@pytest.fixture
def config(walk_forward):
    return walk_forward.WalkForwardConfig(min_train_rows=20, validation_rows=12,
                                          test_rows=6, step_rows=6, purge_rows=HORIZON)


@pytest.fixture
def candidates(selection, models, dataset):
    columns = models.feature_columns_of(dataset)
    return (
        selection.candidate(
            "ridge", lambda: models.RidgeRegressionPredictor(feature_columns=columns)),
        selection.candidate(
            "xgboost", lambda: models.XGBoostRegressionPredictor(feature_columns=columns)),
    )


@pytest.fixture
def evaluation(selection, dataset, config, candidates):
    return selection.evaluate_selection(dataset, config=config, candidates=candidates)


@pytest.fixture
def boundaries(evaluation, robustness):
    return robustness.equal_time_subperiods(evaluation, parts=3)


def _fixed_boundaries(evaluation):
    """Explicit, hand-written edges that do not move when the data grows."""
    return (_iso(GRID), _iso(GRID + HOUR * 45), _iso(GRID + HOUR * 75), _iso(GRID + HOUR * 300))


# --- selection stability: description, never a verdict --------------------


def test_the_stability_summary_counts_what_actually_happened(robustness, evaluation):
    stability = robustness.selection_stability(evaluation)
    winners = [fold.selected_candidate_id for fold in evaluation.folds]
    assert stability.fold_count == len(winners) >= 3
    assert dict(stability.candidate_counts) == {
        name: winners.count(name) for name in set(winners)
    }
    assert sum(count for _, count in stability.candidate_counts) == stability.fold_count
    assert stability.transition_count == sum(
        1 for a, b in zip(winners, winners[1:]) if a != b)
    assert dict(stability.selection_reasons) == {
        reason: [f.selection_reason for f in evaluation.folds].count(reason)
        for reason in {f.selection_reason for f in evaluation.folds}
    }


def test_the_fractions_and_the_transition_rate_are_exact(robustness, evaluation):
    stability = robustness.selection_stability(evaluation)
    with localcontext() as context:
        # references must be computed at the MODULE's precision; at the caller's
        # default a repeating fraction is simply a different number
        context.prec = robustness.ROBUSTNESS_PRECISION
        total = Decimal(stability.fold_count)
        for name, fraction in stability.candidate_fractions:
            assert fraction == Decimal(dict(stability.candidate_counts)[name]) / total
        assert sum(fraction for _, fraction in stability.candidate_fractions) == Decimal(1)
        assert stability.transition_rate == \
            Decimal(stability.transition_count) / Decimal(stability.fold_count - 1)


@pytest.mark.parametrize("winners,transitions,longest", [
    (["a", "a", "a"], 0, 3),
    (["a", "b", "a", "b"], 3, 1),
    (["a", "a", "b", "b", "b", "a"], 2, 3),
])
def test_transitions_and_runs_on_hand_checked_sequences(robustness, evaluation, winners,
                                                        transitions, longest):
    folds = tuple(
        replace(evaluation.folds[index % len(evaluation.folds)],
                fold_index=index, selected_candidate_id=name)
        for index, name in enumerate(winners)
    )
    stability = robustness.selection_stability(replace(evaluation, folds=folds))
    assert stability.transition_count == transitions
    assert stability.longest_run == longest


def test_the_module_offers_no_way_to_crown_a_global_winner(robustness, evaluation,
                                                           boundaries):
    """Counting is description. Crowning would be a second, unvalidated rule."""
    stability = robustness.selection_stability(evaluation)
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries)
    forbidden = ("best_candidate", "recommended_candidate", "majority_winner",
                 "winner", "best_scenario", "recommended_scenario", "champion")
    for name in forbidden:
        assert not hasattr(stability, name), name
        assert not hasattr(report, name), name
        assert not hasattr(robustness, name), name
    exported = [name for name in dir(robustness) if not name.startswith("_")]
    assert not any("best" in name.lower() or "recommend" in name.lower()
                   for name in exported), exported


def test_an_evaluation_without_folds_is_refused_rather_than_summarised(robustness,
                                                                       evaluation):
    empty = replace(evaluation, folds=())
    with pytest.raises(robustness.ModelRobustnessError, match="without folds"):
        robustness.selection_stability(empty)
    with pytest.raises(robustness.ModelRobustnessError, match="without folds"):
        robustness.validation_geometry(empty)


# --- validation geometry --------------------------------------------------


def test_the_geometry_report_shows_whether_rank_ic_did_any_work(robustness, evaluation,
                                                                config):
    geometry = robustness.validation_geometry(evaluation)
    assert len(geometry.folds) == len(evaluation.folds)
    for entry, fold in zip(geometry.folds, evaluation.folds):
        assert entry.nominal_validation_rows == config.validation_rows
        assert entry.effective_validation_rows == fold.validation_rows
        assert entry.effective_validation_rows < entry.nominal_validation_rows  # purged
        assert entry.rank_ic_observation_count == fold.candidates[0].observations
        assert dict(entry.rank_ic_defined) == {
            e.candidate_id: e.rank_ic is not None for e in fold.candidates}
        assert entry.selection_reason == fold.selection_reason
    counted = (geometry.folds_selected_by_rank_ic + geometry.folds_selected_by_mae
               + geometry.folds_selected_by_rmse + geometry.folds_selected_by_tie)
    assert counted == len(evaluation.folds)
    assert geometry.folds_selected_by_rank_ic > 0


def test_the_minimum_effective_validation_block_respects_the_phase_3c_floor(
    robustness, evaluation, selection
):
    """3C now refuses fewer than 3, so 3D can only ever observe compliance."""
    geometry = robustness.validation_geometry(evaluation)
    assert geometry.minimum_effective_validation_rows == min(
        fold.validation_rows for fold in evaluation.folds)
    assert geometry.minimum_effective_validation_rows >= \
        selection.MIN_VALIDATION_OBSERVATIONS


# --- selection margins ----------------------------------------------------


def test_a_margin_is_reported_per_fold_against_the_actual_runner_up(robustness, evaluation):
    margins = robustness.selection_margins(evaluation)
    assert len(margins) == len(evaluation.folds)
    for margin, fold in zip(margins, evaluation.folds):
        assert margin.winner_candidate_id == fold.selected_candidate_id
        assert margin.runner_up_candidate_id != margin.winner_candidate_id
        assert margin.winning_criterion == fold.selection_reason


def test_the_margin_sign_follows_the_direction_of_each_criterion(robustness, evaluation):
    """A better MAE is a SMALLER MAE. The difference must still be positive."""
    for margin in robustness.selection_margins(evaluation):
        if margin.difference is None:
            continue
        assert margin.difference > 0, margin
        with localcontext() as context:
            context.prec = robustness.ROBUSTNESS_PRECISION
            if margin.winning_criterion == "rank_ic":
                assert margin.difference == margin.winner_value - margin.runner_up_value
            else:
                assert margin.difference == margin.runner_up_value - margin.winner_value


def _fold_with(selection, evaluation, entries, reason, winner):
    fold = replace(evaluation.folds[0], candidates=entries,
                   selected_candidate_id=winner, selection_reason=reason)
    return replace(evaluation, folds=(fold,))


def _entry(selection, name, ric, mae, rmse):
    return selection.CandidateValidation(
        candidate_id=name, model_spec_hash="0" * 64,
        rank_ic=None if ric is None else Decimal(ric),
        mae=Decimal(mae), rmse=Decimal(rmse), observations=8, selection_fit_hash=None)


def test_margins_for_each_criterion_are_computed_from_the_right_pair(robustness, selection,
                                                                     evaluation):
    by_rank = _fold_with(selection, evaluation,
                         (_entry(selection, "a", "0.7", "0.5", "0.6"),
                          _entry(selection, "b", "0.4", "0.1", "0.2")), "rank_ic", "a")
    assert robustness.selection_margins(by_rank)[0].difference == Decimal("0.3")

    by_mae = _fold_with(selection, evaluation,
                        (_entry(selection, "a", "0.5", "0.30", "0.6"),
                         _entry(selection, "b", "0.5", "0.12", "0.2")), "mae", "b")
    assert robustness.selection_margins(by_mae)[0].difference == Decimal("0.18")

    by_rmse = _fold_with(selection, evaluation,
                         (_entry(selection, "a", "0.5", "0.10", "0.55"),
                          _entry(selection, "b", "0.5", "0.10", "0.40")), "rmse", "b")
    assert robustness.selection_margins(by_rmse)[0].difference == Decimal("0.15")


def test_a_tie_broken_by_identity_has_no_numeric_margin(robustness, selection, evaluation):
    """Zero would read as "equal on a criterion". They were equal on all of them."""
    tied = _fold_with(selection, evaluation,
                      (_entry(selection, "a", "0.5", "0.1", "0.2"),
                       _entry(selection, "b", "0.5", "0.1", "0.2")), "candidate_id", "a")
    margin = robustness.selection_margins(tied)[0]
    assert margin.difference is None
    assert margin.winner_value is None and margin.runner_up_value is None


def test_a_single_candidate_fold_reports_no_runner_up(robustness, selection, evaluation):
    alone = _fold_with(selection, evaluation,
                       (_entry(selection, "a", "0.5", "0.1", "0.2"),), "sole_candidate", "a")
    margin = robustness.selection_margins(alone)[0]
    assert margin.runner_up_candidate_id is None and margin.difference is None


# --- temporal subperiods --------------------------------------------------


def test_subperiod_metrics_are_recomputed_from_their_own_records(robustness, evaluation,
                                                                 walk_forward):
    edges = _fixed_boundaries(evaluation)
    periods = robustness.subperiod_metrics(evaluation, boundaries=edges)
    assert len(periods) == len(edges) - 1
    for period in periods:
        start = datetime.fromisoformat(period.start)
        end = datetime.fromisoformat(period.end)
        inside = tuple(r for r in evaluation.oos_records
                       if start <= datetime.fromisoformat(r.bar_open_at) < end)
        assert period.metrics.observations == len(inside)
        assert period.metrics.rank_ic == walk_forward.rank_ic(
            tuple(r.prediction for r in inside), tuple(r.actual_forward_return for r in inside))
        assert period.metrics.mae == walk_forward.mean_absolute_error(
            tuple(r.prediction for r in inside), tuple(r.actual_forward_return for r in inside))


def test_the_global_metric_is_never_rebuilt_from_the_subperiods(robustness, evaluation,
                                                                boundaries):
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries)
    assert report.global_test_metrics == evaluation.global_test_metrics
    values = [p.metrics.rank_ic for p in report.subperiods if p.metrics.rank_ic is not None]
    averaged = sum(values) / Decimal(len(values))
    assert report.global_test_metrics.rank_ic != averaged
    assert sum(p.metrics.observations for p in report.subperiods) == \
        report.global_test_metrics.observations


def test_subperiods_can_disagree_wildly_while_the_global_figure_stays_put(
    robustness, evaluation
):
    coarse = robustness.subperiod_metrics(evaluation, boundaries=_fixed_boundaries(evaluation))
    fine = robustness.subperiod_metrics(evaluation, boundaries=tuple(
        _iso(GRID + HOUR * step) for step in range(0, 320, 20)))
    assert len(fine) > len(coarse)
    assert {p.metrics.rank_ic for p in fine} != {p.metrics.rank_ic for p in coarse}
    # the official global number is a property of the records, not of the slicing
    for periods in (coarse, fine):
        assert sum(p.metrics.observations for p in periods) == \
            evaluation.global_test_metrics.observations


def test_an_undefined_subperiod_rank_ic_stays_undefined(robustness, evaluation):
    """An empty or single-observation window measures nothing. Not zero."""
    far_future = (_iso(GRID + HOUR * 5000), _iso(GRID + HOUR * 6000))
    empty = robustness.subperiod_metrics(evaluation, boundaries=far_future)[0]
    assert empty.metrics.observations == 0
    assert empty.metrics.rank_ic is None and empty.metrics.mae is None
    assert empty.rank_ic_degenerate is False


def test_a_two_observation_window_is_flagged_rather_than_trusted(robustness, evaluation):
    first_two = sorted(r.bar_open_at for r in evaluation.oos_records)[:2]
    edges = (first_two[0], _iso(datetime.fromisoformat(first_two[1]) + timedelta(seconds=1)))
    period = robustness.subperiod_metrics(evaluation, boundaries=edges)[0]
    assert period.metrics.observations == 2
    assert abs(period.metrics.rank_ic) == Decimal(1)   # mechanically, always
    assert period.rank_ic_degenerate is True


def test_the_rank_stability_summary_counts_periods_without_scoring_them(robustness,
                                                                        evaluation,
                                                                        boundaries):
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries)
    summary = report.rank_stability
    total = (summary.positive_periods + summary.negative_periods
             + summary.zero_periods + summary.undefined_periods)
    assert total == len(report.subperiods)
    assert summary.positive_periods == sum(
        1 for p in report.subperiods if p.metrics.rank_ic is not None and p.metrics.rank_ic > 0)
    assert summary.undefined_periods == sum(
        1 for p in report.subperiods if p.metrics.rank_ic is None)
    # no verdict of any kind is attached to those counts
    assert not hasattr(summary, "score")
    assert not hasattr(summary, "edge_confirmed")


@pytest.mark.parametrize("bad,fragment", [
    ((), "2 and"),
    (("2026-11-02T00:00:00+00:00",), "2 and"),
    (("2026-11-03T00:00:00+00:00", "2026-11-02T00:00:00+00:00"), "increasing"),
    (("2026-11-02T00:00:00+00:00", "2026-11-02T00:00:00+00:00"), "increasing"),
    (("not-a-date", "2026-11-03T00:00:00+00:00"), "ISO-8601"),
    ((1, 2), "ISO-8601"),
])
def test_malformed_boundaries_are_refused(robustness, evaluation, bad, fragment):
    with pytest.raises(robustness.ModelRobustnessError, match=fragment):
        robustness.subperiod_metrics(evaluation, boundaries=bad)


def test_equal_time_boundaries_come_from_timestamps_alone(robustness, evaluation):
    edges = robustness.equal_time_subperiods(evaluation, parts=4)
    assert len(edges) == 5
    parsed = [datetime.fromisoformat(edge) for edge in edges]
    assert parsed == sorted(parsed)
    spans = [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]
    assert max(spans) - min(spans) < 1.0            # equal spans of TIME
    covered = robustness.subperiod_metrics(evaluation, boundaries=edges)
    assert sum(p.metrics.observations for p in covered) == len(evaluation.oos_records)
    with pytest.raises(robustness.ModelRobustnessError, match="parts must be"):
        robustness.equal_time_subperiods(evaluation, parts=0)
    with pytest.raises(robustness.ModelRobustnessError, match="no out-of-sample"):
        robustness.equal_time_subperiods(replace(evaluation, oos_records=()), parts=2)


# --- anti-hindsight -------------------------------------------------------


def test_boundaries_never_move_because_a_period_scored_badly(robustness, evaluation):
    """The module analyses the periods it was given; it never picks flattering ones."""
    edges = _fixed_boundaries(evaluation)
    honest = robustness.subperiod_metrics(evaluation, boundaries=edges)
    assert len(honest) >= 3

    # invert the actuals of everything in the second window: that period now
    # scores as badly as it previously scored well
    start = datetime.fromisoformat(edges[1])
    end = datetime.fromisoformat(edges[2])
    ruined = replace(evaluation, oos_records=tuple(
        replace(record, actual_forward_return=-record.actual_forward_return)
        if start <= datetime.fromisoformat(record.bar_open_at) < end else record
        for record in evaluation.oos_records
    ))
    damaged = robustness.subperiod_metrics(ruined, boundaries=edges)

    assert [p.start for p in damaged] == [p.start for p in honest]
    assert [p.end for p in damaged] == [p.end for p in honest]
    assert len(damaged) == len(honest)
    assert [p.metrics.observations for p in damaged] == \
           [p.metrics.observations for p in honest]
    # the sabotage was real, and it is reported rather than removed
    walk = importlib.import_module("scripts.trading_lab.walk_forward")
    with localcontext() as context:
        # unary minus is itself a context operation: at prec=28 it would round
        # a value the module produced at prec=34
        context.prec = walk.EVALUATION_PRECISION
        assert damaged[1].metrics.rank_ic == -honest[1].metrics.rank_ic
    assert damaged[0].metrics.rank_ic == honest[0].metrics.rank_ic


# --- declared scenarios ---------------------------------------------------


@pytest.fixture
def scenarios(robustness, selection, models, dataset):
    columns = models.feature_columns_of(dataset)

    def ridge(alpha):
        return lambda: models.RidgeRegressionPredictor(feature_columns=columns,
                                                       alpha=Decimal(alpha))
    return (
        robustness.RobustnessScenario("alpha-0.5", (
            selection.candidate("ridge", ridge("0.5")),
            selection.candidate("xgboost", lambda: models.XGBoostRegressionPredictor(
                feature_columns=columns)))),
        robustness.RobustnessScenario("alpha-2.0", (
            selection.candidate("ridge", ridge("2.0")),
            selection.candidate("xgboost", lambda: models.XGBoostRegressionPredictor(
                feature_columns=columns)))),
    )


def test_each_scenario_goes_back_through_validation_only_selection(
    robustness, evaluation, boundaries, scenarios, dataset, config
):
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                           scenarios=scenarios, dataset=dataset,
                                           config=config)
    assert [result.scenario_id for result in report.scenarios] == ["alpha-0.5", "alpha-2.0"]
    for result in report.scenarios:
        assert len(result.selection_spec_hash) == 64
        assert result.selection_results_hash != result.selection_spec_hash
        assert result.stability.fold_count == len(evaluation.folds)
        assert result.global_test_metrics.observations == \
            evaluation.global_test_metrics.observations
    # different alphas are genuinely different protocols
    assert report.scenarios[0].selection_spec_hash != report.scenarios[1].selection_spec_hash


def test_no_api_exists_to_pick_a_winning_scenario(robustness, evaluation, boundaries,
                                                  scenarios, dataset, config):
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                           scenarios=scenarios, dataset=dataset,
                                           config=config)
    for name in ("best_scenario", "select_scenario", "recommended_scenario", "rank_scenarios"):
        assert not hasattr(report, name)
        assert not hasattr(robustness, name)
    assert all(not hasattr(result, "is_best") for result in report.scenarios)


def test_the_order_scenarios_are_declared_in_changes_nothing(robustness, evaluation,
                                                             boundaries, scenarios,
                                                             dataset, config):
    forward = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                            scenarios=scenarios, dataset=dataset,
                                            config=config)
    backward = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                             scenarios=tuple(reversed(scenarios)),
                                             dataset=dataset, config=config)
    assert forward.spec_hash == backward.spec_hash
    assert forward.results_hash == backward.results_hash
    assert forward.scenarios == backward.scenarios


def test_scenarios_need_the_data_they_would_be_re_run_on(robustness, evaluation,
                                                         boundaries, scenarios):
    with pytest.raises(robustness.ModelRobustnessError, match="need the dataset"):
        robustness.analyse_robustness(evaluation, boundaries=boundaries, scenarios=scenarios)


def test_duplicate_scenario_ids_are_refused(robustness, evaluation, boundaries, scenarios,
                                            dataset, config):
    with pytest.raises(robustness.ModelRobustnessError, match="unique"):
        robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                      scenarios=(scenarios[0], scenarios[0]),
                                      dataset=dataset, config=config)


# --- trading costs --------------------------------------------------------


def test_trading_costs_are_declared_unavailable_rather_than_invented(robustness, evaluation,
                                                                     boundaries):
    """Fees applied to a raw forward return would be arithmetic, not a cost model."""
    report = robustness.analyse_robustness(evaluation, boundaries=boundaries)
    assert report.trading_cost_analysis_available is False
    assert "no causal position or execution policy" in report.trading_cost_unavailable_reason
    for name in dir(robustness):
        assert "fee" not in name.lower() and "cost_model" not in name.lower()
    assert not hasattr(report, "net_metrics")


# --- identity, determinism, independence ---------------------------------


def test_the_report_is_reproducible_across_three_runs(robustness, evaluation, boundaries,
                                                      scenarios, dataset, config):
    runs = [robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                          scenarios=scenarios, dataset=dataset,
                                          config=config) for _ in range(3)]
    assert len({run.spec_hash for run in runs}) == 1
    assert len({run.results_hash for run in runs}) == 1
    assert runs[0].stability == runs[1].stability == runs[2].stability
    assert runs[0].margins == runs[2].margins
    assert runs[0].subperiods == runs[2].subperiods


def test_the_spec_hash_covers_every_boundary_and_every_scenario(robustness, evaluation,
                                                                boundaries, scenarios,
                                                                dataset, config):
    baseline = robustness.analyse_robustness(evaluation, boundaries=boundaries,
                                             scenarios=scenarios, dataset=dataset,
                                             config=config)
    assert baseline.spec_hash != baseline.results_hash
    assert robustness.analyse_robustness(
        evaluation, boundaries=boundaries[:-1] + (_iso(GRID + HOUR * 999),),
        scenarios=scenarios, dataset=dataset, config=config).spec_hash != baseline.spec_hash
    assert robustness.analyse_robustness(
        evaluation, boundaries=boundaries, scenarios=scenarios[:1], dataset=dataset,
        config=config).spec_hash != baseline.spec_hash
    assert robustness.analyse_robustness(
        evaluation, boundaries=boundaries).spec_hash != baseline.spec_hash
    assert baseline.spec.protocol_version == robustness.ROBUSTNESS_PROTOCOL_VERSION


def test_the_report_is_indifferent_to_the_callers_decimal_context(robustness, evaluation,
                                                                  boundaries):
    original = getcontext().prec
    seen = set()
    try:
        for precision in (7, 28, 34, 60):
            getcontext().prec = precision
            report = robustness.analyse_robustness(evaluation, boundaries=boundaries)
            seen.add((report.spec_hash, report.results_hash))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_future_observations_leave_fixed_boundaries_and_old_folds_alone(
    robustness, selection, models, tmp_path, config
):
    """Proved with EXPLICIT boundaries: equal-time edges legitimately move."""
    short = _build_dataset(tmp_path, name="short", count=100)
    long = _build_dataset(tmp_path, name="long", count=140)
    columns = models.feature_columns_of(short)

    def make():
        return (selection.candidate(
                    "ridge", lambda: models.RidgeRegressionPredictor(feature_columns=columns)),
                selection.candidate(
                    "xgboost",
                    lambda: models.XGBoostRegressionPredictor(feature_columns=columns)))

    short_eval = selection.evaluate_selection(short, config=config, candidates=make())
    long_eval = selection.evaluate_selection(long, config=config, candidates=make())
    assert len(long_eval.folds) > len(short_eval.folds)

    edges = (_iso(GRID), _iso(GRID + HOUR * 60), _iso(GRID + HOUR * 90))
    short_periods = robustness.subperiod_metrics(short_eval, boundaries=edges)
    long_periods = robustness.subperiod_metrics(long_eval, boundaries=edges)
    # windows entirely inside the shorter horizon are untouched
    assert short_periods[0] == long_periods[0]

    short_margins = robustness.selection_margins(short_eval)
    long_margins = robustness.selection_margins(long_eval)
    for earlier, later in zip(short_margins, long_margins):
        assert earlier == later
    for earlier, later in zip(short_eval.folds, long_eval.folds):
        assert earlier == later


def test_equal_time_boundaries_are_documented_as_moving_with_the_data(
    robustness, selection, models, tmp_path, config
):
    short = _build_dataset(tmp_path, name="mv-short", count=100)
    long = _build_dataset(tmp_path, name="mv-long", count=140)
    columns = models.feature_columns_of(short)

    def make():
        return (selection.candidate(
                    "ridge", lambda: models.RidgeRegressionPredictor(feature_columns=columns)),)

    short_edges = robustness.equal_time_subperiods(
        selection.evaluate_selection(short, config=config, candidates=make()), parts=3)
    long_edges = robustness.equal_time_subperiods(
        selection.evaluate_selection(long, config=config, candidates=make()), parts=3)
    # this is exactly why the independence proof above uses explicit boundaries
    assert short_edges != long_edges
    assert short_edges[0] == long_edges[0]


def test_an_undefined_period_is_counted_as_undefined_never_as_zero(robustness, evaluation):
    """The counts must actually exercise the undefined branch.

    A summary computed over windows that all happen to be well defined proves
    nothing about how `None` is handled: both sides of the assertion are zero.
    These boundaries deliberately mix a populated window, an inverted one and
    an empty one, so "undefined" and "exactly zero" cannot be confused.
    """
    populated = sorted(r.bar_open_at for r in evaluation.oos_records)
    edges = (populated[0],
             _iso(datetime.fromisoformat(populated[-1]) + timedelta(seconds=1)),
             _iso(GRID + HOUR * 5000),
             _iso(GRID + HOUR * 6000))
    periods = robustness.subperiod_metrics(evaluation, boundaries=edges)
    assert periods[0].metrics.rank_ic is not None
    assert periods[1].metrics.observations == 0 and periods[1].metrics.rank_ic is None
    assert periods[2].metrics.observations == 0 and periods[2].metrics.rank_ic is None

    summary = robustness.rank_stability(periods)
    assert summary.undefined_periods == 2
    assert summary.zero_periods == 0
    assert summary.positive_periods + summary.negative_periods == 1

    # and a genuinely negative window is counted as negative, not as undefined
    start, end = datetime.fromisoformat(edges[0]), datetime.fromisoformat(edges[1])
    inverted = replace(evaluation, oos_records=tuple(
        replace(record, actual_forward_return=-record.actual_forward_return)
        if start <= datetime.fromisoformat(record.bar_open_at) < end else record
        for record in evaluation.oos_records
    ))
    flipped = robustness.rank_stability(
        robustness.subperiod_metrics(inverted, boundaries=edges))
    assert flipped.undefined_periods == 2
    assert flipped.zero_periods == 0
    assert (flipped.positive_periods, flipped.negative_periods) == \
        (summary.negative_periods, summary.positive_periods)
