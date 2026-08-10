"""Phase 3C: validation-only model selection, and the proofs test never leaked in.

The selection rule itself is small. Almost everything below is about the one
failure mode that would quietly invalidate every number Phase 3 produces:
choosing a model because of how it happened to score on the block that is
supposed to be untouched.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
import hashlib
import importlib
import inspect
import json

import pytest


GRID = datetime(2026, 9, 7, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
T_PUB = datetime(2026, 9, 20, tzinfo=timezone.utc)
HORIZON = 4


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


def _build_dataset(tmp_path, *, name, count=95):
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    series_module = importlib.import_module("scripts.trading_lab.market_series")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + HOUR * index for index in range(count)]
    closes = [Decimal(180 + (index * 29) % 43 - (index % 8) * 3) for index in range(count)]
    rows = [
        [int(o.timestamp()), str(c - Decimal(2 + index % 3)), str(c + Decimal(1 + index % 5)),
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
    return _build_dataset(tmp_path, name="selection")


@pytest.fixture
def config(walk_forward):
    # validation_rows must exceed the purge, or the surviving block is too short
    # for a rank correlation and the primary criterion silently never applies:
    # at validation_rows=6 with purge_rows=4 only 2 rows survive, and a rank_ic
    # over two points is always exactly +/-1.
    return walk_forward.WalkForwardConfig(min_train_rows=20, validation_rows=12,
                                          test_rows=6, step_rows=6, purge_rows=HORIZON)


@pytest.fixture
def real_candidates(selection, models, dataset):
    columns = models.feature_columns_of(dataset)
    return (
        selection.candidate(
            "ridge", lambda: models.RidgeRegressionPredictor(feature_columns=columns)),
        selection.candidate(
            "xgboost", lambda: models.XGBoostRegressionPredictor(feature_columns=columns)),
    )


# --- fixture-only predictors ---------------------------------------------


class Scripted:
    """Fixture-only: emits a fixed cycle of predictions. Never a production API."""

    def __init__(self, label, values):
        self.label = label
        self.values = tuple(Decimal(str(value)) for value in values)
        self.fit_blocks = []

    @property
    def model_spec_hash(self):
        return hashlib.sha256(self.label.encode("utf-8")).hexdigest()

    def fit(self, rows):
        self.fit_blocks.append(tuple(row.bar_open_at for row in rows))

    def predict(self, rows):
        return tuple(self.values[index % len(self.values)] for index in range(len(rows)))


def _scripted(selection, label, values):
    return selection.candidate(label, lambda: Scripted(label, values))


def _validation(selection, candidate_id, rank_ic, mae, rmse, observations=6):
    return selection.CandidateValidation(
        candidate_id=candidate_id,
        model_spec_hash=hashlib.sha256(candidate_id.encode()).hexdigest(),
        rank_ic=None if rank_ic is None else Decimal(str(rank_ic)),
        mae=None if mae is None else Decimal(str(mae)),
        rmse=None if rmse is None else Decimal(str(rmse)),
        observations=observations,
        selection_fit_hash=None,
    )


class BlockedRow:
    """A test row that detonates if selection so much as looks at it."""

    def __init__(self, row, journal):
        self.bar_open_at = row.bar_open_at
        self._row = row
        self._journal = journal

    @property
    def features(self):
        self._journal.append(("test-row-touched", self.bar_open_at))
        raise AssertionError(f"selection read the features of {self.bar_open_at}")

    @property
    def label(self):
        self._journal.append(("test-row-touched", self.bar_open_at))
        raise AssertionError(f"selection read the label of {self.bar_open_at}")

    @property
    def usable(self):
        return True


# --- the selection rule ---------------------------------------------------


def test_the_highest_validation_rank_ic_wins_even_with_a_worse_error(selection):
    results = (_validation(selection, "a", "0.40", "0.30", "0.40"),
               _validation(selection, "b", "0.10", "0.05", "0.06"))
    assert selection.select(results) == ("a", "rank_ic")


def test_a_tie_on_rank_ic_falls_through_to_the_error(selection):
    results = (_validation(selection, "a", "0.20", "0.30", "0.40"),
               _validation(selection, "b", "0.20", "0.10", "0.50"))
    assert selection.select(results) == ("b", "mae")


def test_when_no_rank_ic_is_defined_the_error_decides(selection):
    results = (_validation(selection, "a", None, "0.10", "0.20"),
               _validation(selection, "b", None, "0.30", "0.05"))
    assert selection.select(results) == ("a", "mae")


def test_a_tie_on_rank_ic_and_mae_falls_through_to_rmse(selection):
    results = (_validation(selection, "a", "0.20", "0.10", "0.40"),
               _validation(selection, "b", "0.20", "0.10", "0.30"))
    assert selection.select(results) == ("b", "rmse")


def test_a_total_tie_is_broken_by_candidate_id_never_by_chance(selection):
    results = (_validation(selection, "zebra", "0.20", "0.10", "0.30"),
               _validation(selection, "alpha", "0.20", "0.10", "0.30"))
    assert selection.select(results) == ("alpha", "candidate_id")
    assert selection.select(tuple(reversed(results))) == ("alpha", "candidate_id")


def test_a_defined_rank_ic_beats_an_undefined_one_however_bad_it_is(selection):
    """The trap: if None were read as 0, a negative score would lose to nothing."""
    results = (_validation(selection, "measured", "-0.90", "0.50", "0.60"),
               _validation(selection, "unmeasurable", None, "0.01", "0.02"))
    assert selection.select(results) == ("measured", "rank_ic")


def test_a_single_candidate_is_selected_without_pretending_it_won(selection):
    assert selection.select((_validation(selection, "only", "0.1", "0.2", "0.3"),)) == \
        ("only", "sole_candidate")


def test_a_candidate_with_no_comparable_metric_fails_closed(selection):
    results = (_validation(selection, "a", "0.2", None, None),
               _validation(selection, "b", "0.1", "0.3", "0.4"))
    with pytest.raises(selection.ModelSelectionError, match="no comparable"):
        selection.select(results)
    with pytest.raises(selection.ModelSelectionError, match="no candidate"):
        selection.select(())


# --- structural test isolation -------------------------------------------


def test_the_selection_api_cannot_receive_a_test_block_at_all(selection):
    """The strongest statement available: there is no parameter for it."""
    for function in (selection.validate_candidates, selection.select):
        parameters = inspect.signature(function).parameters
        assert not any("test" in name for name in parameters), parameters


def test_selection_completes_without_touching_a_single_test_row(selection, models, dataset,
                                                                walk_forward, config,
                                                                real_candidates):
    """Order-based proof: every validation fit happens before the test block exists."""
    journal = []
    folds = walk_forward.build_folds(dataset, config=config)
    train, validation, test = folds[0]
    blocked = tuple(BlockedRow(row, journal) for row in test)

    results = selection.validate_candidates(train, validation, candidates=real_candidates)
    winner, reason = selection.select(results)
    assert journal == []                       # nothing reached the test rows
    assert winner in {"ridge", "xgboost"} and reason

    # and the trap is armed: the very next step would have fired it
    with pytest.raises(AssertionError, match="selection read"):
        selection.select_over_folds([(train, validation, blocked)],
                                    candidates=real_candidates, horizon=HORIZON,
                                    duration=HOUR)
    assert journal and journal[0][0] == "test-row-touched"


def test_sabotaging_the_test_labels_changes_no_choice_and_no_prediction(
    selection, dataset, walk_forward, config, real_candidates
):
    folds = walk_forward.build_folds(dataset, config=config)
    # Only fold 0's test block: an expanding window legitimately recycles a
    # scored test block into later training data, so sabotaging every test row
    # would move later folds for an entirely honest reason.
    fold_zero_test = {row.bar_open_at for row in folds[0][2]}
    sabotaged = replace(dataset, rows=tuple(
        replace(row, label=row.label * Decimal(-100))
        if row.bar_open_at in fold_zero_test and row.label is not None else row
        for row in dataset.rows
    ))
    honest = selection.evaluate_selection(dataset, config=config, candidates=real_candidates)
    tampered = selection.evaluate_selection(sabotaged, config=config,
                                            candidates=real_candidates)
    assert len(honest.folds) == len(tampered.folds) >= 3
    left, right = honest.folds[0], tampered.folds[0]
    assert left.selected_candidate_id == right.selected_candidate_id
    assert left.selection_reason == right.selection_reason
    assert left.candidates == right.candidates
    assert left.selection_fit_hash == right.selection_fit_hash
    assert left.final_fit_hash == right.final_fit_hash
    assert [r.prediction for r in left.test_records] == \
           [r.prediction for r in right.test_records]
    assert left.test_metrics != right.test_metrics      # only the scoring moved


def test_sabotaging_the_test_features_changes_no_choice(
    selection, dataset, walk_forward, config, real_candidates
):
    """Catches selection made indirectly on test performance."""
    folds = walk_forward.build_folds(dataset, config=config)
    fold_zero_test = {row.bar_open_at for row in folds[0][2]}
    sabotaged = replace(dataset, rows=tuple(
        replace(row, features=tuple((column, value * Decimal("777") + Decimal(13))
                                    for column, value in row.features))
        if row.bar_open_at in fold_zero_test else row
        for row in dataset.rows
    ))
    honest = selection.evaluate_selection(dataset, config=config, candidates=real_candidates)
    tampered = selection.evaluate_selection(sabotaged, config=config,
                                            candidates=real_candidates)
    left, right = honest.folds[0], tampered.folds[0]
    assert left.selected_candidate_id == right.selected_candidate_id
    assert left.selection_reason == right.selection_reason
    assert left.candidates == right.candidates
    assert left.selection_fit_hash == right.selection_fit_hash
    assert left.final_fit_hash == right.final_fit_hash
    # the sabotage was real: predictions on those features did move
    assert [r.prediction for r in left.test_records] != \
           [r.prediction for r in right.test_records]


# --- the refit contract ---------------------------------------------------


def test_the_winner_is_refitted_on_train_plus_validation_as_a_fresh_instance(
    selection, dataset, walk_forward, config
):
    seen = []

    class Spy(Scripted):
        def fit(self, rows):
            super().fit(rows)
            seen.append((id(self), tuple(row.bar_open_at for row in rows)))

    specs = (selection.candidate("spy", lambda: Spy("spy", ["0.1", "-0.2", "0.3"])),)
    folds = walk_forward.build_folds(dataset, config=config)[:1]
    selection.select_over_folds(folds, candidates=specs, horizon=HORIZON, duration=HOUR)
    train, validation, _ = folds[0]
    assert len(seen) == 2
    assert seen[0][1] == tuple(row.bar_open_at for row in train)
    assert seen[1][1] == tuple(row.bar_open_at for row in train) + \
        tuple(row.bar_open_at for row in validation)
    assert seen[0][0] != seen[1][0]      # a brand new object, never the scored one


def test_the_two_fitted_hashes_are_reported_separately_and_do_differ(
    selection, dataset, config, real_candidates
):
    """The model that faced test is not the model that won validation."""
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    for fold in evaluation.folds:
        assert fold.selection_fit_hash and fold.final_fit_hash
        assert fold.selection_fit_hash != fold.final_fit_hash
        assert fold.train_rows < fold.train_rows + fold.validation_rows


def test_the_refit_is_refused_when_validation_labels_are_not_yet_knowable(
    selection, dataset, walk_forward, config, real_candidates
):
    """Defence in depth: 2D purges, but 3C proves the contract for itself."""
    folds = walk_forward.build_folds(dataset, config=config)
    train, validation, test = folds[0]
    # move the test block to start immediately after validation: the last
    # validation label window would then reach into it
    illegal = tuple(replace(row, bar_open_at=_iso(
        datetime.fromisoformat(validation[-1].bar_open_at) + HOUR * (index + 1)))
        for index, row in enumerate(test))
    with pytest.raises(selection.ModelSelectionError, match="refit policy violated"):
        selection.select_over_folds([(train, validation, illegal)],
                                    candidates=real_candidates, horizon=HORIZON,
                                    duration=HOUR)
    # the honest fold satisfies the same check
    selection._require_validation_known_before_test(validation, test, horizon=HORIZON,
                                                    duration=HOUR)


def test_every_fold_gets_unfitted_candidates(selection, dataset, walk_forward, config):
    created = []

    def factory():
        model = Scripted("fresh", ["0.2", "-0.1"])
        created.append(model)
        return model

    specs = (selection.candidate("fresh", factory),)
    created.clear()
    folds = walk_forward.build_folds(dataset, config=config)
    selection.select_over_folds(folds, candidates=specs, horizon=HORIZON, duration=HOUR)
    assert len(created) == 2 * len(folds)          # one scored, one refitted, per fold
    assert len({id(model) for model in created}) == len(created)
    assert all(len(model.fit_blocks) == 1 for model in created)


def test_a_factory_that_recycles_one_instance_is_refused(selection):
    shared = Scripted("shared", ["0.1"])
    with pytest.raises(selection.ModelSelectionError, match="same instance"):
        selection.candidate("shared", lambda: shared)

    class Unstable:
        counter = 0
        def __init__(self):
            Unstable.counter += 1
            self.model_spec_hash = f"hash-{Unstable.counter}"
    with pytest.raises(selection.ModelSelectionError, match="stable model_spec_hash"):
        selection.candidate("unstable", Unstable)


# --- candidate set hygiene ------------------------------------------------


def test_the_candidate_set_is_validated(selection, models, dataset, real_candidates):
    columns = models.feature_columns_of(dataset)
    with pytest.raises(selection.ModelSelectionError, match="1 and"):
        selection.validate_candidates((), (), candidates=())
    duplicated = (real_candidates[0], real_candidates[0])
    with pytest.raises(selection.ModelSelectionError, match="unique"):
        selection._canonical_candidates(duplicated)
    with pytest.raises(selection.ModelSelectionError, match="non-empty string"):
        selection.candidate("", lambda: models.RidgeRegressionPredictor(
            feature_columns=columns))


def test_a_candidate_returning_malformed_predictions_is_refused(selection, dataset,
                                                                walk_forward, config):
    class Short(Scripted):
        def predict(self, rows): return tuple(Decimal(0) for _ in rows[:-1])

    class NotDecimal(Scripted):
        def predict(self, rows): return tuple(0.5 for _ in rows)

    train, validation, _ = walk_forward.build_folds(dataset, config=config)[0]
    for broken, fragment in ((Short, "predictions for"), (NotDecimal, "finite Decimal")):
        specs = (selection.candidate("broken", lambda broken=broken: broken("broken", ["0"])),)
        with pytest.raises(selection.ModelSelectionError, match=fragment):
            selection.validate_candidates(train, validation, candidates=specs)


def test_validation_rows_without_labels_are_refused_never_imputed(selection, dataset,
                                                                  walk_forward, config,
                                                                  real_candidates):
    train, validation, _ = walk_forward.build_folds(dataset, config=config)[0]
    stripped = tuple(replace(row, label=None) for row in validation)
    with pytest.raises(selection.ModelSelectionError, match="must carry labels"):
        selection.validate_candidates(train, stripped, candidates=real_candidates)
    with pytest.raises(selection.ModelSelectionError, match="non-empty train and validation"):
        selection.validate_candidates(train, (), candidates=real_candidates)


# --- order independence, identity, determinism ---------------------------


def test_the_order_the_candidates_arrive_in_changes_nothing(selection, dataset, config,
                                                            real_candidates):
    forward = selection.evaluate_selection(dataset, config=config, candidates=real_candidates)
    backward = selection.evaluate_selection(dataset, config=config,
                                            candidates=tuple(reversed(real_candidates)))
    assert forward.spec_hash == backward.spec_hash
    assert forward.results_hash == backward.results_hash
    assert [f.selected_candidate_id for f in forward.folds] == \
           [f.selected_candidate_id for f in backward.folds]
    assert forward.folds == backward.folds


def test_the_spec_hash_describes_the_protocol_and_the_results_hash_the_outcome(
    selection, models, dataset, walk_forward, config, real_candidates
):
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    assert len(evaluation.spec_hash) == 64
    assert evaluation.spec_hash != evaluation.results_hash
    assert evaluation.spec.selection_rule_version == selection.SELECTION_RULE_VERSION
    assert evaluation.spec.refit_policy_version == selection.REFIT_POLICY_VERSION
    assert evaluation.spec.metric_version == walk_forward.METRIC_SCHEMA_VERSION

    other_config = replace(config, min_train_rows=24)
    assert selection.evaluate_selection(
        dataset, config=other_config, candidates=real_candidates).spec_hash != \
        evaluation.spec_hash
    single = selection.evaluate_selection(dataset, config=config,
                                          candidates=real_candidates[:1])
    assert single.spec_hash != evaluation.spec_hash


def test_three_identical_runs_agree_on_everything(selection, dataset, config,
                                                  real_candidates):
    runs = [selection.evaluate_selection(dataset, config=config, candidates=real_candidates)
            for _ in range(3)]
    assert len({run.spec_hash for run in runs}) == 1
    assert len({run.results_hash for run in runs}) == 1
    assert runs[0].folds == runs[1].folds == runs[2].folds
    assert runs[0].oos_records == runs[2].oos_records


def test_the_selection_is_indifferent_to_the_callers_decimal_context(selection, dataset,
                                                                     config, real_candidates):
    original = getcontext().prec
    seen = set()
    try:
        for precision in (7, 28, 34, 60):
            getcontext().prec = precision
            evaluation = selection.evaluate_selection(dataset, config=config,
                                                      candidates=real_candidates)
            seen.add((evaluation.results_hash,
                      tuple(f.selected_candidate_id for f in evaluation.folds)))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_observations_added_later_cannot_change_an_earlier_fold(selection, dataset,
                                                                walk_forward, config,
                                                                real_candidates):
    folds = walk_forward.build_folds(dataset, config=config)
    boundary = folds[0][2][-1].bar_open_at
    rewritten = replace(dataset, rows=tuple(
        replace(row,
                label=row.label * Decimal(-6) if row.label is not None else None,
                features=tuple((column, None if value is None else value + Decimal("321"))
                               for column, value in row.features))
        if row.bar_open_at > boundary else row
        for row in dataset.rows
    ))
    before = selection.evaluate_selection(dataset, config=config, candidates=real_candidates)
    after = selection.evaluate_selection(rewritten, config=config, candidates=real_candidates)
    assert before.folds[0] == after.folds[0]
    assert before.folds[-1] != after.folds[-1]


# --- global metrics and real candidates ----------------------------------


def test_the_global_test_metrics_are_recomputed_on_the_concatenation(
    selection, dataset, config, real_candidates, walk_forward
):
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    concatenated = tuple(record for fold in evaluation.folds for record in fold.test_records)
    assert evaluation.oos_records == concatenated
    assert evaluation.global_test_metrics.observations == len(concatenated)
    assert evaluation.global_test_metrics.rank_ic == walk_forward.rank_ic(
        tuple(record.prediction for record in concatenated),
        tuple(record.actual_forward_return for record in concatenated),
    )
    fold_values = [fold.test_metrics.rank_ic for fold in evaluation.folds
                   if fold.test_metrics.rank_ic is not None]
    averaged = sum(fold_values) / Decimal(len(fold_values))
    assert evaluation.global_test_metrics.rank_ic != averaged
    timestamps = [record.bar_open_at for record in concatenated]
    assert len(set(timestamps)) == len(timestamps)


def test_both_real_candidates_are_scored_in_every_fold(selection, dataset, config,
                                                       real_candidates, walk_forward):
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    folds = walk_forward.build_folds(dataset, config=config)
    for fold, (_, validation, _) in zip(evaluation.folds, folds):
        assert [entry.candidate_id for entry in fold.candidates] == ["ridge", "xgboost"]
        # the validation block is PURGED, so it is shorter than validation_rows
        assert 0 < len(validation) <= config.validation_rows
        assert all(entry.observations == len(validation) for entry in fold.candidates)
        assert fold.selected_candidate_id in {"ridge", "xgboost"}
        winner = next(e for e in fold.candidates
                      if e.candidate_id == fold.selected_candidate_id)
        assert winner.model_spec_hash and len(winner.model_spec_hash) == 64


def test_the_selected_candidate_may_differ_from_fold_to_fold(selection, dataset, config,
                                                             real_candidates):
    """Per-fold choice is the honest output; no global vote is taken here."""
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    choices = [fold.selected_candidate_id for fold in evaluation.folds]
    assert len(choices) >= 3
    assert not hasattr(evaluation, "overall_winner")
    assert not hasattr(evaluation, "winning_candidate")
    # a scripted pair makes the alternation explicit rather than fixture-dependent
    early = _scripted(selection, "aaa", ["0.5", "-0.5", "0.5", "-0.5", "0.5", "-0.5"])
    late = _scripted(selection, "bbb", ["-0.5", "0.5", "-0.5", "0.5", "-0.5", "0.5"])
    scripted = selection.evaluate_selection(dataset, config=config,
                                            candidates=(early, late))
    assert {fold.selected_candidate_id for fold in scripted.folds} <= {"aaa", "bbb"}


def test_the_scripted_pair_exercises_the_rule_end_to_end(selection, dataset, config,
                                                         walk_forward):
    """A candidate whose validation ranking is perfect must win its fold."""
    train, validation, test = walk_forward.build_folds(dataset, config=config)[0]
    actuals = [row.label for row in validation]
    ordered = sorted(range(len(actuals)), key=lambda index: actuals[index])
    perfect = [Decimal(0)] * len(actuals)
    for rank, index in enumerate(ordered):
        perfect[index] = Decimal(rank)
    good = selection.candidate("good", lambda: Scripted("good", [str(v) for v in perfect]))
    bad = selection.candidate("bad", lambda: Scripted("bad", [str(-v) for v in perfect]))
    results = selection.validate_candidates(train, validation, candidates=(good, bad))
    winner, reason = selection.select(results)
    assert (winner, reason) == ("good", "rank_ic")
    scores = {entry.candidate_id: entry.rank_ic for entry in results}
    assert scores["good"] == Decimal(1) and scores["bad"] == Decimal(-1)


def test_a_validation_block_too_short_for_a_rank_correlation_is_refused(
    selection, dataset, walk_forward, real_candidates
):
    """The geometry trap, closed rather than documented.

    The purge eats `horizon` rows off the end of the validation block. Choose
    `validation_rows` barely above it and two observations survive -- and a
    Spearman correlation over two non-constant points is mechanically +/-1 for
    every candidate, so the primary criterion always ties and MAE silently
    becomes the real selector. Three is the smallest count at which the
    ranking can say something, so the protocol refuses less rather than
    degrading in silence.
    """
    degenerate = walk_forward.WalkForwardConfig(min_train_rows=20, validation_rows=6,
                                                test_rows=6, step_rows=6, purge_rows=HORIZON)
    folds = walk_forward.build_folds(dataset, config=degenerate)
    assert {len(validation) for _, validation, _ in folds} == {2}
    with pytest.raises(selection.ModelSelectionError, match="at least 3"):
        selection.evaluate_selection(dataset, config=degenerate, candidates=real_candidates)


def test_exactly_three_validation_observations_are_enough(selection, dataset, walk_forward,
                                                          real_candidates):
    """The bound is 3, not a comfortable round number pulled from the air."""
    config = walk_forward.WalkForwardConfig(min_train_rows=20, validation_rows=7,
                                            test_rows=6, step_rows=6, purge_rows=HORIZON)
    folds = walk_forward.build_folds(dataset, config=config)
    assert {len(validation) for _, validation, _ in folds} == {3}
    evaluation = selection.evaluate_selection(dataset, config=config,
                                              candidates=real_candidates)
    assert len(evaluation.folds) >= 3
    assert all(entry.observations == 3 for fold in evaluation.folds
               for entry in fold.candidates)


def test_the_effective_block_is_checked_not_the_nominal_parameter(
    selection, walk_forward, real_candidates, dataset
):
    """A nominal parameter says nothing: purge is market time and gaps bite.

    Here `validation_rows` is generous, but the block actually handed to the
    candidates holds two rows. Validating the config arithmetic would wave
    this through; validating the real block does not.
    """
    train, validation, test = walk_forward.build_folds(dataset, config=walk_forward.
                                                       WalkForwardConfig(
        min_train_rows=20, validation_rows=12, test_rows=6, step_rows=6,
        purge_rows=HORIZON))[0]
    assert len(validation) > 3          # nominally comfortable
    with pytest.raises(selection.ModelSelectionError, match="at least 3"):
        selection.validate_candidates(train, validation[:2], candidates=real_candidates)
    selection.validate_candidates(train, validation[:3], candidates=real_candidates)
