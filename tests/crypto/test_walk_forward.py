"""Phase 2D: expanding-window walk-forward evaluation and its leakage proofs.

The protocol under test has one job: make it impossible for a future model to
be scored on information it could not have had. Most of what follows is
therefore not "does the number look right" but "can the number have been
contaminated".
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json
import pathlib
import subprocess
import sys

import pytest


TESTS_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent.parent


GRID = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)
HORIZON = 4
HOUR = timedelta(hours=1)


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


@pytest.fixture
def snapshots_module():
    return importlib.import_module("scripts.trading_lab.market_snapshots")


@pytest.fixture
def series_module():
    return importlib.import_module("scripts.trading_lab.market_series")


@pytest.fixture
def dataset_module():
    return importlib.import_module("scripts.trading_lab.market_dataset")


@pytest.fixture
def walk_forward_module():
    return importlib.import_module("scripts.trading_lab.walk_forward")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _publish(store, opens, closes, *, at):
    rows = [[int(o.timestamp()), "1.0", "1000000.0", c, c, "1.0"] for o, c in zip(opens, closes)]
    return store.ingest_coinbase_response(
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
        product_id="BTC-USD", timeframe="1h",
        available_at=_iso(at), ingested_at=_iso(at + timedelta(seconds=1)),
    )


def _series_at(store, snapshots_module, series_module, as_of, opens):
    connection = store._connect()
    try:
        result = snapshots_module._materialize_snapshot(
            connection, provider="coinbase_exchange_rest", product_id="BTC-USD",
            timeframe="1h", range_start=_iso(opens[0]),
            range_end=_iso(opens[-1] + HOUR), as_of=_iso(as_of),
        )
        return series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def _ramp(count, *, start=100, step=7):
    return [str(start + index * step) for index in range(count)]


def _dataset(store_module, snapshots_module, series_module, dataset_module, tmp_path, *,
             name, closes, skip=()):
    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(len(closes))]
    kept = [(o, c) for index, (o, c) in enumerate(zip(opens, closes)) if index not in skip]
    _publish(store, [o for o, _ in kept], [c for _, c in kept], at=T1)
    series = _series_at(store, snapshots_module, series_module, T2, opens)
    config = dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("sma3", "sma", (("period", 3),)),
            dataset_module.FeatureDefinition("rsi2", "rsi", (("period", 2),)),
        ),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    return dataset_module.build_dataset(series, config=config)


def _config(walk_forward_module, **overrides):
    fields = {"min_train_rows": 10, "validation_rows": 5, "test_rows": 5,
              "step_rows": 5, "purge_rows": HORIZON}
    fields.update(overrides)
    return walk_forward_module.WalkForwardConfig(**fields)


class SpyPredictor:
    """Records everything it is shown, so leakage becomes an assertion."""

    name = "spy"
    version = "1"

    def __init__(self):
        self.fit_rows = []
        self.predict_rows = []
        self.fit_calls = 0

    def fit(self, rows):
        self.fit_calls += 1
        self.fit_rows.append(tuple(rows))

    def predict(self, rows):
        self.predict_rows.append(tuple(rows))
        return tuple(Decimal(index) for index in range(len(rows)))


class OraclePredictor:
    """Test fixture only: cheats by reading the label it is asked to predict."""

    name = "oracle"
    version = "1"

    def fit(self, rows): pass

    def predict(self, rows): return tuple(row.label for row in rows)


class ZeroPredictor:
    name = "zero"
    version = "1"

    def fit(self, rows): pass

    def predict(self, rows): return tuple(Decimal(0) for _ in rows)


@pytest.fixture
def dataset(tmp_path, store_module, snapshots_module, series_module, dataset_module):
    return _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="wf", closes=_ramp(50))


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize("field", ["min_train_rows", "validation_rows", "test_rows", "step_rows"])
@pytest.mark.parametrize("value", [0, -1, 1.0, "5", None, True, Decimal(5)])
def test_a_row_count_that_is_not_a_positive_int_is_refused(
    walk_forward_module, dataset, field, value
) -> None:
    """`True` is an int subclass; a bool must never be read as the count 1."""
    config = _config(walk_forward_module, **{field: value})
    with pytest.raises(walk_forward_module.WalkForwardError, match=field):
        walk_forward_module.build_folds(dataset, config=config)


def test_purge_rows_may_be_zero_but_not_negative(walk_forward_module, dataset) -> None:
    walk_forward_module.build_folds(dataset, config=_config(walk_forward_module, purge_rows=0))
    with pytest.raises(walk_forward_module.WalkForwardError, match="purge_rows"):
        walk_forward_module.build_folds(dataset, config=_config(walk_forward_module, purge_rows=-1))


def test_a_step_smaller_than_the_test_block_is_refused(walk_forward_module, dataset) -> None:
    """Overlapping test windows would double-count observations in the global score."""
    config = _config(walk_forward_module, test_rows=5, step_rows=4)
    with pytest.raises(walk_forward_module.WalkForwardError, match="overlap"):
        walk_forward_module.build_folds(dataset, config=config)
    walk_forward_module.build_folds(dataset, config=_config(walk_forward_module,
                                                            test_rows=5, step_rows=5))


def test_an_unknown_timeframe_is_refused_rather_than_guessed(
    walk_forward_module, dataset
) -> None:
    with pytest.raises(walk_forward_module.WalkForwardError, match="timeframe"):
        walk_forward_module.build_folds(replace(dataset, timeframe="3h"),
                                        config=_config(walk_forward_module))


def test_a_dataset_out_of_chronological_order_fails_closed(
    walk_forward_module, dataset
) -> None:
    """Reordering here would hide the very defect the ordering protects against."""
    scrambled = replace(dataset, rows=tuple(reversed(dataset.rows)))
    with pytest.raises(walk_forward_module.WalkForwardError, match="ascending"):
        walk_forward_module.usable_rows(scrambled)


# --- fold geometry --------------------------------------------------------


def test_the_training_window_expands_and_never_slides(walk_forward_module, dataset) -> None:
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    assert len(folds) >= 3
    for earlier, later in zip(folds, folds[1:]):
        train_earlier = [row.bar_open_at for row in earlier[0]]
        train_later = [row.bar_open_at for row in later[0]]
        # expanding, not rolling: the earlier train set is a strict prefix
        assert train_later[: len(train_earlier)] == train_earlier
        assert len(train_later) > len(train_earlier)
        assert train_later[0] == train_earlier[0]


def test_every_block_is_strictly_ordered_and_disjoint(walk_forward_module, dataset) -> None:
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    for train, validation, test in folds:
        assert train[-1].bar_open_at < validation[0].bar_open_at
        assert validation[-1].bar_open_at < test[0].bar_open_at
        openings = [row.bar_open_at for row in (*train, *validation, *test)]
        assert openings == sorted(openings)
        assert len(set(openings)) == len(openings)


def test_the_test_windows_of_two_folds_never_overlap(walk_forward_module, dataset) -> None:
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    seen = [row.bar_open_at for _, _, test in folds for row in test]
    assert len(set(seen)) == len(seen)


def test_unusable_rows_are_dropped_and_never_imputed(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, walk_forward_module
) -> None:
    """A missing feature is an absence. Filling it in would invent history."""
    data = _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="holes", closes=_ramp(50), skip={20, 21})
    assert any(not row.usable for row in data.rows)
    unusable = {row.bar_open_at for row in data.rows if not row.usable}
    folds = walk_forward_module.build_folds(data, config=_config(walk_forward_module))
    for block in folds:
        for rows in block:
            for row in rows:
                assert row.usable
                assert row.bar_open_at not in unusable
                assert row.label is not None
                assert all(value is not None for _, value in row.features)


# --- purging, proved in market time ---------------------------------------


def _label_end(row):
    return datetime.fromisoformat(row.bar_open_at) + HOUR * HORIZON


def test_no_training_label_window_reaches_the_next_block(
    walk_forward_module, dataset
) -> None:
    """The real invariant: max(train label end) < min(next block opening).

    Counting rows is not the proof -- a row's label reaches h bars of MARKET
    time forward, which is a different distance.
    """
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    assert folds
    for train, validation, test in folds:
        assert _label_end(train[-1]) < datetime.fromisoformat(validation[0].bar_open_at)
        assert _label_end(validation[-1]) < datetime.fromisoformat(test[0].bar_open_at)
        assert max(_label_end(row) for row in train) < min(
            datetime.fromisoformat(row.bar_open_at) for row in validation
        )


def test_a_gap_makes_row_counting_insufficient_and_market_time_decisive(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, walk_forward_module
) -> None:
    """With holes in the data, dropping `purge_rows` rows drops the wrong amount.

    Here the purge must still hold in market time, and it does -- because the
    boundary is computed from timestamps, not from an index arithmetic.
    """
    data = _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="gapped", closes=_ramp(60), skip={17, 18, 30, 31, 32})
    folds = walk_forward_module.build_folds(data, config=_config(walk_forward_module,
                                                                 purge_rows=0))
    assert folds
    for train, validation, test in folds:
        # purge_rows=0: nothing is dropped by count, yet the invariant holds
        assert _label_end(train[-1]) < datetime.fromisoformat(validation[0].bar_open_at)
        assert _label_end(validation[-1]) < datetime.fromisoformat(test[0].bar_open_at)


def test_the_purge_removes_exactly_the_rows_that_reach_over(
    walk_forward_module, dataset
) -> None:
    """Purging is bounded: it must not eat rows that were already safe."""
    rows = walk_forward_module.usable_rows(dataset)
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    train = folds[0][0]
    kept = len(train)
    # the first row NOT kept would have violated the boundary
    dropped = rows[kept]
    assert _label_end(dropped) >= datetime.fromisoformat(folds[0][1][0].bar_open_at)


# --- the leakage proof ----------------------------------------------------


def test_no_test_timestamp_is_ever_handed_to_its_own_fit(
    walk_forward_module, dataset
) -> None:
    """If this fails the whole evaluation is worthless, not merely inaccurate.

    The invariant is per fold, not global: an expanding window is SUPPOSED to
    absorb an already-scored test block into a later training set -- by then
    those labels are genuinely known. What must never happen is a fold seeing
    its own future.
    """
    spy = SpyPredictor()
    evaluation = walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                              predictor=spy)
    assert spy.fit_calls == len(evaluation.folds) >= 3
    for fit_rows, fold in zip(spy.fit_rows, evaluation.folds):
        trained = {row.bar_open_at for row in fit_rows}
        tested = {record.bar_open_at for record in fold.records}
        assert trained and tested
        assert trained.isdisjoint(tested)
        assert max(trained) < fold.test.first
        assert max(trained) < fold.validation.first


def test_a_test_block_can_only_be_learned_from_by_a_later_fold(
    walk_forward_module, dataset
) -> None:
    """The absorption is one-directional. Backwards absorption would be leakage."""
    spy = SpyPredictor()
    evaluation = walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                              predictor=spy)
    for fold in evaluation.folds:
        tested = {record.bar_open_at for record in fold.records}
        for fit_index, fit_rows in enumerate(spy.fit_rows):
            if {row.bar_open_at for row in fit_rows} & tested:
                assert fit_index > fold.fold_index


def test_fit_never_receives_the_validation_block_either(
    walk_forward_module, dataset
) -> None:
    spy = SpyPredictor()
    walk_forward_module.evaluate(dataset, config=_config(walk_forward_module), predictor=spy)
    folds = walk_forward_module.build_folds(dataset, config=_config(walk_forward_module))
    for fit_rows, (_, validation, _) in zip(spy.fit_rows, folds):
        assert {row.bar_open_at for row in fit_rows}.isdisjoint(
            {row.bar_open_at for row in validation}
        )


def test_observations_added_later_cannot_change_an_earlier_fold(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, walk_forward_module
) -> None:
    """A causal protocol is stable under extension of the future."""
    short = _dataset(store_module, snapshots_module, series_module, dataset_module,
                     tmp_path, name="short", closes=_ramp(45))
    long = _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="long", closes=_ramp(70))
    config = _config(walk_forward_module)
    short_evaluation = walk_forward_module.evaluate(short, config=config,
                                                    predictor=OraclePredictor())
    long_evaluation = walk_forward_module.evaluate(long, config=config,
                                                   predictor=OraclePredictor())
    assert len(long_evaluation.folds) > len(short_evaluation.folds)
    for earlier, later in zip(short_evaluation.folds, long_evaluation.folds):
        assert earlier.records == later.records
        assert earlier.train == later.train
        assert earlier.test == later.test
        assert earlier.metrics == later.metrics


# --- predictor contract ---------------------------------------------------


def test_a_predictor_returning_the_wrong_number_of_values_is_refused(
    walk_forward_module, dataset
) -> None:
    class Short(ZeroPredictor):
        def predict(self, rows): return tuple(Decimal(0) for _ in rows[:-1])

    with pytest.raises(walk_forward_module.WalkForwardError, match="predictions"):
        walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                     predictor=Short())


@pytest.mark.parametrize("value", [0.5, "0.5", None, Decimal("NaN"), Decimal("Infinity")])
def test_a_prediction_that_is_not_a_finite_decimal_is_refused(
    walk_forward_module, dataset, value
) -> None:
    class Bad(ZeroPredictor):
        def predict(self, rows): return tuple(value for _ in rows)

    with pytest.raises(walk_forward_module.WalkForwardError, match="prediction"):
        walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                     predictor=Bad())


# --- metrics --------------------------------------------------------------


def test_a_perfect_predictor_scores_a_rank_ic_of_one(walk_forward_module, dataset) -> None:
    evaluation = walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                              predictor=OraclePredictor())
    assert evaluation.oos_metrics.rank_ic == Decimal(1)
    assert evaluation.oos_metrics.mae == Decimal(0)
    assert evaluation.oos_metrics.rmse == Decimal(0)
    for fold in evaluation.folds:
        assert fold.metrics.rank_ic == Decimal(1)


def test_a_constant_predictor_has_no_rank_ic_rather_than_a_zero_one(
    walk_forward_module, dataset
) -> None:
    """Zero would claim "no relationship measured". None says "not measurable"."""
    evaluation = walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                              predictor=ZeroPredictor())
    assert evaluation.oos_metrics.rank_ic is None
    assert all(fold.metrics.rank_ic is None for fold in evaluation.folds)
    # the error metrics are still perfectly well defined
    assert evaluation.oos_metrics.mae is not None
    assert evaluation.oos_metrics.rmse > 0


def test_rank_ic_is_undefined_below_two_observations(walk_forward_module) -> None:
    assert walk_forward_module.rank_ic((), ()) is None
    assert walk_forward_module.rank_ic((Decimal(1),), (Decimal(2),)) is None
    assert walk_forward_module.mean_absolute_error((), ()) is None
    assert walk_forward_module.root_mean_squared_error((), ()) is None


def test_ties_are_ranked_by_their_average(walk_forward_module) -> None:
    """Competition ranking would silently bias every correlation with ties."""
    values = (Decimal(5), Decimal(1), Decimal(5), Decimal(3))
    assert walk_forward_module._average_ranks(values) == [
        Decimal("3.5"), Decimal(1), Decimal("3.5"), Decimal(2),
    ]
    # a monotone relation with ties still reaches +1
    assert walk_forward_module.rank_ic(
        (Decimal(1), Decimal(2), Decimal(2), Decimal(3)),
        (Decimal(10), Decimal(20), Decimal(20), Decimal(30)),
    ) == Decimal(1)


def test_rank_ic_is_minus_one_for_a_perfectly_inverted_predictor(walk_forward_module) -> None:
    predictions = tuple(Decimal(index) for index in range(6))
    actuals = tuple(Decimal(-index) for index in range(6))
    assert walk_forward_module.rank_ic(predictions, actuals) == Decimal(-1)


def test_rank_ic_ignores_scale_but_not_order(walk_forward_module) -> None:
    actuals = (Decimal(1), Decimal(4), Decimal(9), Decimal(16))
    linear = (Decimal(1), Decimal(2), Decimal(3), Decimal(4))
    assert walk_forward_module.rank_ic(linear, actuals) == Decimal(1)


def test_mismatched_metric_inputs_are_refused(walk_forward_module) -> None:
    for metric in (walk_forward_module.rank_ic, walk_forward_module.mean_absolute_error,
                   walk_forward_module.root_mean_squared_error):
        with pytest.raises(walk_forward_module.WalkForwardError, match="same length"):
            metric((Decimal(1), Decimal(2)), (Decimal(1),))


def test_the_metrics_do_not_depend_on_the_callers_decimal_context(
    walk_forward_module
) -> None:
    """The reference is computed at the MODULE's precision, not at prec=28."""
    predictions = (Decimal(0), Decimal(0), Decimal(0))
    actuals = (Decimal(1), Decimal(0), Decimal(0))
    with localcontext() as context:
        context.prec = walk_forward_module.EVALUATION_PRECISION
        expected_mae = Decimal(1) / Decimal(3)
        expected_rmse = (Decimal(1) / Decimal(3)).sqrt()

    original = getcontext().prec
    try:
        for precision in (6, 28, 50):
            getcontext().prec = precision
            assert walk_forward_module.mean_absolute_error(predictions, actuals) == expected_mae
            assert walk_forward_module.root_mean_squared_error(predictions, actuals) == expected_rmse
    finally:
        getcontext().prec = original


def test_the_global_score_is_recomputed_not_averaged(walk_forward_module, dataset) -> None:
    """Averaging fold correlations is a different, wrong statistic."""
    evaluation = walk_forward_module.evaluate(dataset, config=_config(walk_forward_module),
                                              predictor=SpyPredictor())
    concatenated = tuple(record for fold in evaluation.folds for record in fold.records)
    assert evaluation.oos_records == concatenated
    assert evaluation.oos_metrics.observations == len(concatenated)
    recomputed = walk_forward_module.rank_ic(
        tuple(record.prediction for record in concatenated),
        tuple(record.actual_forward_return for record in concatenated),
    )
    assert evaluation.oos_metrics.rank_ic == recomputed
    fold_values = [fold.metrics.rank_ic for fold in evaluation.folds]
    assert all(value is not None for value in fold_values)
    with localcontext() as context:
        context.prec = walk_forward_module.EVALUATION_PRECISION
        averaged = sum(fold_values) / Decimal(len(fold_values))
    assert evaluation.oos_metrics.rank_ic != averaged


# --- identity and reproducibility ----------------------------------------


def test_the_specification_hash_does_not_depend_on_the_results(
    walk_forward_module, dataset
) -> None:
    """A spec identifies the QUESTION; the results hash identifies the ANSWER."""
    config = _config(walk_forward_module)
    oracle = walk_forward_module.evaluate(dataset, config=config, predictor=OraclePredictor())
    zero = walk_forward_module.evaluate(dataset, config=config, predictor=ZeroPredictor())
    assert oracle.spec_hash != zero.spec_hash          # different predictor identity
    assert oracle.results_hash != zero.results_hash
    again = walk_forward_module.evaluate(dataset, config=config, predictor=OraclePredictor())
    assert again.spec_hash == oracle.spec_hash
    assert again.results_hash == oracle.results_hash


def test_the_spec_hash_changes_with_every_config_field(walk_forward_module, dataset) -> None:
    config = _config(walk_forward_module)
    baseline = walk_forward_module.evaluate(dataset, config=config,
                                            predictor=ZeroPredictor()).spec_hash
    for field, value in (("min_train_rows", 12), ("validation_rows", 6), ("test_rows", 4),
                         ("step_rows", 6), ("purge_rows", 3)):
        altered = _config(walk_forward_module, **{field: value})
        if field == "test_rows":
            altered = replace(altered, step_rows=altered.step_rows)
        assert walk_forward_module.evaluate(
            dataset, config=altered, predictor=ZeroPredictor()
        ).spec_hash != baseline, field


def test_the_evaluation_is_reproducible_across_a_process_restart(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, walk_forward_module,
) -> None:
    """Hashes must be a property of the data, not of one interpreter's memory."""
    data = _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="restart", closes=_ramp(50))
    evaluation = walk_forward_module.evaluate(data, config=_config(walk_forward_module),
                                              predictor=ZeroPredictor())
    script = f"""
import json, pathlib, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, {str(TESTS_ROOT)!r})
from decimal import Decimal
import test_walk_forward as harness
import importlib
store_module = importlib.import_module("scripts.trading_lab.market_data_store")
snapshots_module = importlib.import_module("scripts.trading_lab.market_snapshots")
series_module = importlib.import_module("scripts.trading_lab.market_series")
dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")
walk_forward_module = importlib.import_module("scripts.trading_lab.walk_forward")
data = harness._dataset(store_module, snapshots_module, series_module, dataset_module,
                        pathlib.Path({str(tmp_path)!r}), name="restart-child",
                        closes=harness._ramp(50))
result = walk_forward_module.evaluate(
    data, config=harness._config(walk_forward_module), predictor=harness.ZeroPredictor())
print(json.dumps({{"spec": result.spec_hash, "results": result.results_hash,
                   "folds": len(result.folds),
                   "rmse": str(result.oos_metrics.rmse)}}))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               cwd=str(REPO_ROOT), timeout=300)
    assert completed.returncode == 0, completed.stderr
    reloaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reloaded["spec"] == evaluation.spec_hash
    assert reloaded["results"] == evaluation.results_hash
    assert reloaded["folds"] == len(evaluation.folds)
    assert reloaded["rmse"] == str(evaluation.oos_metrics.rmse)


def test_the_evaluation_reports_the_blocks_it_actually_used(
    walk_forward_module, dataset
) -> None:
    config = _config(walk_forward_module)
    evaluation = walk_forward_module.evaluate(dataset, config=config,
                                              predictor=ZeroPredictor())
    folds = walk_forward_module.build_folds(dataset, config=config)
    assert len(evaluation.folds) == len(folds)
    for fold, (train, validation, test) in zip(evaluation.folds, folds):
        assert fold.train.count == len(train)
        assert fold.train.first == train[0].bar_open_at
        assert fold.train.last == train[-1].bar_open_at
        assert fold.validation.count == len(validation)
        assert fold.test.count == len(test) == config.test_rows
        assert [record.bar_open_at for record in fold.records] == [
            row.bar_open_at for row in test
        ]
        assert [record.actual_forward_return for record in fold.records] == [
            row.label for row in test
        ]


def test_a_dataset_too_short_for_one_fold_yields_no_fold_rather_than_a_partial_one(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, walk_forward_module
) -> None:
    tiny = _dataset(store_module, snapshots_module, series_module, dataset_module,
                    tmp_path, name="tiny", closes=_ramp(15))
    evaluation = walk_forward_module.evaluate(tiny, config=_config(walk_forward_module,
                                                                   min_train_rows=200),
                                              predictor=ZeroPredictor())
    assert evaluation.folds == ()
    assert evaluation.oos_records == ()
    assert evaluation.oos_metrics.observations == 0
    assert evaluation.oos_metrics.rank_ic is None
    assert evaluation.oos_metrics.mae is None
