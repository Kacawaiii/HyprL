"""Phase 3A: the ridge baseline and the proofs that it cannot cheat.

Most of these tests are not about regression quality. They are about the
three boundaries the model sits on: labels, feature order, and the float64
crossing. A model that scores well because it peeked is worse than no model.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json

import pytest


GRID = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
T_PUB = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
T_ASOF = datetime(2026, 4, 12, 1, 0, tzinfo=timezone.utc)
HORIZON = 4


@pytest.fixture
def models():
    return importlib.import_module("scripts.trading_lab.models")


@pytest.fixture
def walk_forward():
    return importlib.import_module("scripts.trading_lab.walk_forward")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_dataset(tmp_path, *, name, count=70, skip=()):
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    series_module = importlib.import_module("scripts.trading_lab.market_series")
    dataset_module = importlib.import_module("scripts.trading_lab.market_dataset")

    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + HOUR * index for index in range(count)]
    # deliberately not a clean ramp: a monotone series makes every model look
    # identical and hides feature-order bugs
    closes = [str(140 + (index * 23) % 61 - (index % 7) * 2) for index in range(count)]
    kept = [(index, o, c) for index, (o, c) in enumerate(zip(opens, closes)) if index not in skip]
    # real highs and lows: a constant band would make the true range constant,
    # and a constant feature is (correctly) refused by the model
    rows = [
        [int(o.timestamp()), str(Decimal(c) - Decimal(1 + index % 4)),
         str(Decimal(c) + Decimal(1 + index % 6)), c, c, "1.0"]
        for index, o, c in kept
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
            range_end=_iso(opens[-1] + HOUR), as_of=_iso(T_ASOF),
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
    return _build_dataset(tmp_path, name="models")


@pytest.fixture
def train_rows(dataset, walk_forward):
    return walk_forward.usable_rows(dataset)[:30]


def _config(walk_forward):
    return walk_forward.WalkForwardConfig(min_train_rows=14, validation_rows=5,
                                          test_rows=5, step_rows=5, purge_rows=HORIZON)


class LabelTrap:
    """A row that detonates if anything reads its label."""

    def __init__(self, row):
        self.bar_open_at = row.bar_open_at
        self.features = row.features
        self.usable = row.usable

    @property
    def label(self):
        raise AssertionError(f"predict() read the label of {self.bar_open_at}")


class RecordingRidge:
    """Ridge, plus a tape of what each fold learned."""

    name = "ridge_regression"
    version = "1"

    def __init__(self, models, columns, alpha=Decimal("1.0")):
        self.inner = models.RidgeRegressionPredictor(feature_columns=columns, alpha=alpha)
        self.fitted_hashes = []
        self.predictions = []

    def fit(self, rows):
        self.inner.fit(rows)
        self.fitted_hashes.append(self.inner.fitted.fitted_model_hash)

    def predict(self, rows):
        values = self.inner.predict(rows)
        self.predictions.append(values)
        return values


# --- happy path and the feature boundary ----------------------------------


def test_the_model_fits_and_predicts_finite_decimals(models, dataset, train_rows, walk_forward):
    columns = models.feature_columns_of(dataset)
    assert columns == ("sma5", "rsi3", "atr4")
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    model.fit(train_rows)
    later = walk_forward.usable_rows(dataset)[30:40]
    predictions = model.predict(later)
    assert len(predictions) == len(later)
    assert all(isinstance(value, Decimal) and value.is_finite() for value in predictions)
    assert model.fitted.train_rows == len(train_rows)
    assert len(model.fitted.coefficients) == len(columns)


def test_the_column_order_is_validated_not_trusted(models, dataset, train_rows):
    """A permuted matrix trains happily and predicts nonsense. Refuse it."""
    columns = models.feature_columns_of(dataset)
    permuted = (columns[1], columns[0], columns[2])
    model = models.RidgeRegressionPredictor(feature_columns=permuted)
    with pytest.raises(models.ModelError, match="exact order"):
        model.fit(train_rows)


@pytest.mark.parametrize("columns", [("sma5", "rsi3"), ("sma5", "rsi3", "atr4", "extra"),
                                     ("sma5", "rsi9", "atr4")])
def test_a_feature_schema_that_does_not_match_the_rows_is_refused(models, train_rows, columns):
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    with pytest.raises(models.ModelError, match="expects"):
        model.fit(train_rows)


@pytest.mark.parametrize("columns", [(), ("a",) * 300, ("a", "a"), ("a", ""), ("a", 3)])
def test_a_malformed_feature_schema_is_refused_at_construction(models, columns):
    with pytest.raises(models.ModelError):
        models.RidgeRegressionPredictor(feature_columns=columns)


@pytest.mark.parametrize("alpha", [0, -1, Decimal(0), Decimal("-0.5"), 1.0, "1", None,
                                   Decimal("NaN")])
def test_a_malformed_alpha_is_refused(models, dataset, alpha):
    with pytest.raises(models.ModelError, match="alpha"):
        models.RidgeRegressionPredictor(feature_columns=models.feature_columns_of(dataset),
                                        alpha=alpha)


# --- data quality: fail closed, never impute ------------------------------


def test_an_unusable_row_is_refused_rather_than_imputed(models, dataset, walk_forward):
    columns = models.feature_columns_of(dataset)
    unusable = [row for row in dataset.rows if not row.usable]
    assert unusable
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    with pytest.raises(models.ModelError, match="not usable"):
        model.fit(walk_forward.usable_rows(dataset)[:20] + (unusable[0],))


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_a_non_finite_feature_or_label_is_refused(models, dataset, train_rows, bad):
    columns = models.feature_columns_of(dataset)
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    poisoned_feature = replace(train_rows[3],
                               features=(("sma5", bad), *train_rows[3].features[1:]))
    with pytest.raises(models.ModelError, match="finite"):
        model.fit((*train_rows[:3], poisoned_feature, *train_rows[4:]))
    poisoned_label = replace(train_rows[3], label=bad)
    with pytest.raises(models.ModelError, match="finite"):
        model.fit((*train_rows[:3], poisoned_label, *train_rows[4:]))


def test_a_feature_that_never_moves_in_training_is_refused(models, train_rows):
    """Standardising a constant column is a division by zero, not a zero."""
    frozen = tuple(
        replace(row, features=(("sma5", Decimal("7")), *row.features[1:])) for row in train_rows
    )
    model = models.RidgeRegressionPredictor(feature_columns=("sma5", "rsi3", "atr4"))
    with pytest.raises(models.ModelError, match="constant"):
        model.fit(frozen)


def test_predicting_before_fitting_is_refused(models, dataset, train_rows):
    model = models.RidgeRegressionPredictor(feature_columns=models.feature_columns_of(dataset))
    with pytest.raises(models.ModelError, match="before fit"):
        model.predict(train_rows)
    with pytest.raises(models.ModelError, match="two training rows"):
        model.fit(train_rows[:1])


# --- the label boundary ---------------------------------------------------


def test_predict_never_reads_a_label_at_all(models, dataset, train_rows, walk_forward):
    """Structural, not defensive: the rows handed to predict have no readable label."""
    columns = models.feature_columns_of(dataset)
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    model.fit(train_rows)
    later = walk_forward.usable_rows(dataset)[30:40]
    trapped = [LabelTrap(row) for row in later]
    assert model.predict(trapped) == model.predict(later)


def test_changing_only_the_test_labels_changes_nothing_the_model_learns(
    models, dataset, walk_forward
):
    """The negative proof: labels of the block being scored are invisible."""
    config = _config(walk_forward)
    folds = walk_forward.build_folds(dataset, config=config)
    fold_zero_test = {row.bar_open_at for row in folds[0][2]}
    tampered = replace(dataset, rows=tuple(
        replace(row, label=row.label * Decimal(3) + Decimal("0.5"))
        if row.bar_open_at in fold_zero_test else row
        for row in dataset.rows
    ))
    columns = models.feature_columns_of(dataset)
    original = RecordingRidge(models, columns)
    altered = RecordingRidge(models, columns)
    first = walk_forward.evaluate(dataset, config=config, predictor=original)
    second = walk_forward.evaluate(tampered, config=config, predictor=altered)

    # what the model learned for fold 0, and what it predicted, are untouched
    assert original.fitted_hashes[0] == altered.fitted_hashes[0]
    assert original.predictions[0] == altered.predictions[0]
    assert [r.prediction for r in first.folds[0].records] == \
           [r.prediction for r in second.folds[0].records]
    # only the scoring moved, because only the actuals moved
    assert first.folds[0].metrics != second.folds[0].metrics
    assert [r.actual_forward_return for r in first.folds[0].records] != \
           [r.actual_forward_return for r in second.folds[0].records]


# --- the preprocessing boundary -------------------------------------------


def test_an_extreme_value_in_the_test_block_cannot_move_the_scaler(
    models, dataset, walk_forward
):
    """Standardisation fitted on train+test is the classic silent leak."""
    config = _config(walk_forward)
    folds = walk_forward.build_folds(dataset, config=config)
    fold_zero_test = {row.bar_open_at for row in folds[0][2]}
    spiked = replace(dataset, rows=tuple(
        replace(row, features=tuple((column, value * Decimal("1000000"))
                                    for column, value in row.features))
        if row.bar_open_at in fold_zero_test else row
        for row in dataset.rows
    ))
    columns = models.feature_columns_of(dataset)
    plain = models.RidgeRegressionPredictor(feature_columns=columns)
    plain.fit(folds[0][0])
    spiked_folds = walk_forward.build_folds(spiked, config=config)
    poisoned = models.RidgeRegressionPredictor(feature_columns=columns)
    poisoned.fit(spiked_folds[0][0])

    assert plain.fitted.feature_means == poisoned.fitted.feature_means
    assert plain.fitted.feature_stdevs == poisoned.fitted.feature_stdevs
    assert plain.fitted.coefficients == poisoned.fitted.coefficients
    assert plain.fitted.intercept == poisoned.fitted.intercept
    assert plain.fitted.fitted_model_hash == poisoned.fitted.fitted_model_hash


def test_rows_beyond_a_fold_cannot_reach_back_into_it(models, dataset, walk_forward):
    """Future observations may create new folds. They may not edit old ones."""
    config = _config(walk_forward)
    folds = walk_forward.build_folds(dataset, config=config)
    horizon_end = folds[0][2][-1].bar_open_at
    rewritten = replace(dataset, rows=tuple(
        replace(row,
                label=row.label * Decimal(-7) if row.label is not None else None,
                features=tuple((column, None if value is None else value + Decimal("999"))
                               for column, value in row.features))
        if row.bar_open_at > horizon_end else row
        for row in dataset.rows
    ))
    columns = models.feature_columns_of(dataset)
    before = walk_forward.evaluate(dataset, config=config,
                                   predictor=RecordingRidge(models, columns))
    after = walk_forward.evaluate(rewritten, config=config,
                                  predictor=RecordingRidge(models, columns))
    assert before.folds[0].records == after.folds[0].records
    assert before.folds[0].metrics == after.folds[0].metrics
    assert before.folds[0].train == after.folds[0].train
    # and the change was real: later folds did move
    assert before.folds[-1].records != after.folds[-1].records


def test_the_model_never_mutates_the_rows_it_is_given(models, dataset, walk_forward):
    snapshot = tuple(dataset.rows)
    columns = models.feature_columns_of(dataset)
    walk_forward.evaluate(dataset, config=_config(walk_forward),
                          predictor=models.RidgeRegressionPredictor(feature_columns=columns))
    assert dataset.rows == snapshot


# --- determinism and identity ---------------------------------------------


def test_three_identical_fits_produce_the_identical_model(models, dataset, train_rows,
                                                          walk_forward):
    columns = models.feature_columns_of(dataset)
    later = walk_forward.usable_rows(dataset)[30:40]
    results = []
    for _ in range(3):
        model = models.RidgeRegressionPredictor(feature_columns=columns)
        model.fit(train_rows)
        results.append((model.fitted.fitted_model_hash, model.predict(later)))
    assert len({item[0] for item in results}) == 1
    assert results[0][1] == results[1][1] == results[2][1]


def test_the_model_is_indifferent_to_the_callers_decimal_context(models, dataset, train_rows,
                                                                 walk_forward):
    columns = models.feature_columns_of(dataset)
    later = walk_forward.usable_rows(dataset)[30:40]
    original = getcontext().prec
    seen = set()
    try:
        for precision in (7, 28, 34, 60):
            getcontext().prec = precision
            model = models.RidgeRegressionPredictor(feature_columns=columns)
            model.fit(train_rows)
            seen.add((model.fitted.fitted_model_hash,
                      tuple(str(value) for value in model.predict(later))))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_the_spec_hash_describes_the_definition_and_the_fitted_hash_the_state(
    models, dataset, train_rows
):
    columns = models.feature_columns_of(dataset)
    baseline = models.RidgeRegressionPredictor(feature_columns=columns)
    same = models.RidgeRegressionPredictor(feature_columns=columns)
    assert baseline.model_spec_hash == same.model_spec_hash
    assert len(baseline.model_spec_hash) == 64

    other_alpha = models.RidgeRegressionPredictor(feature_columns=columns, alpha=Decimal("2.5"))
    other_schema = models.RidgeRegressionPredictor(feature_columns=columns[:2] + ("zzz",))
    assert other_alpha.model_spec_hash != baseline.model_spec_hash
    assert other_schema.model_spec_hash != baseline.model_spec_hash

    baseline.fit(train_rows)
    assert baseline.fitted.fitted_model_hash != baseline.model_spec_hash
    # a different training block leaves the definition untouched but the state changes
    shorter = models.RidgeRegressionPredictor(feature_columns=columns)
    shorter.fit(train_rows[:20])
    assert shorter.model_spec_hash == baseline.model_spec_hash
    assert shorter.fitted.fitted_model_hash != baseline.fitted.fitted_model_hash


def test_the_learned_parameters_cross_back_to_decimal_at_a_fixed_exponent(
    models, dataset, train_rows
):
    """The float64 river has one documented crossing, and it is quantised."""
    model = models.RidgeRegressionPredictor(feature_columns=models.feature_columns_of(dataset))
    model.fit(train_rows)
    for value in (*model.fitted.coefficients, model.fitted.intercept):
        assert isinstance(value, Decimal)
        assert value.as_tuple().exponent == models.COEFFICIENT_EXPONENT.as_tuple().exponent


# --- integration with the Phase 2 protocol --------------------------------


def test_the_model_runs_inside_the_walk_forward_protocol(models, dataset, walk_forward):
    config = _config(walk_forward)
    columns = models.feature_columns_of(dataset)
    evaluation = walk_forward.evaluate(
        dataset, config=config, predictor=models.RidgeRegressionPredictor(feature_columns=columns))
    assert len(evaluation.folds) >= 3
    assert evaluation.oos_metrics.observations == sum(f.metrics.observations
                                                      for f in evaluation.folds)
    assert evaluation.oos_metrics.mae is not None and evaluation.oos_metrics.rmse is not None
    assert evaluation.spec.predictor_name == "ridge_regression"
    again = walk_forward.evaluate(
        dataset, config=config, predictor=models.RidgeRegressionPredictor(feature_columns=columns))
    assert again.results_hash == evaluation.results_hash


def test_the_protocol_can_compare_the_model_against_trivial_controls(
    models, dataset, walk_forward
):
    """The point is comparability, not victory on synthetic data."""
    config = _config(walk_forward)
    columns = models.feature_columns_of(dataset)

    class Zero:
        name, version = "zero", "1"
        def fit(self, rows): pass
        def predict(self, rows): return tuple(Decimal(0) for _ in rows)

    ridge = walk_forward.evaluate(
        dataset, config=config, predictor=models.RidgeRegressionPredictor(feature_columns=columns))
    mean = walk_forward.evaluate(dataset, config=config, predictor=models.MeanTrainPredictor())
    zero = walk_forward.evaluate(dataset, config=config, predictor=Zero())

    assert ridge.oos_metrics.observations == mean.oos_metrics.observations \
        == zero.oos_metrics.observations
    # Both controls are constant WITHIN a fold, so their per-fold rank_ic is undefined.
    assert all(fold.metrics.rank_ic is None for fold in mean.folds)
    assert all(fold.metrics.rank_ic is None for fold in zero.folds)
    # The zero control is constant everywhere, so it stays undefined globally too.
    assert zero.oos_metrics.rank_ic is None
    # The mean control is NOT: its constant differs per fold, so a global rank
    # correlation becomes computable. That number measures fold-to-fold drift,
    # never within-fold skill -- a trap worth pinning down rather than admiring.
    assert mean.oos_metrics.rank_ic is not None
    assert ridge.oos_metrics.rank_ic is not None
    # the ridge actually uses its features: it is not a disguised constant
    assert len({record.prediction for record in ridge.oos_records}) > 1
    assert ridge.spec_hash != mean.spec_hash != zero.spec_hash
    for evaluation in (ridge, mean, zero):
        assert evaluation.oos_metrics.mae is not None
        assert evaluation.oos_metrics.rmse is not None


def test_the_mean_control_averages_the_training_block_only(models, dataset, walk_forward):
    rows = walk_forward.usable_rows(dataset)
    control = models.MeanTrainPredictor()
    control.fit(rows[:20])
    with localcontext() as context:
        context.prec = models.MODEL_PRECISION
        expected = sum(row.label for row in rows[:20]) / Decimal(20)
    assert control.mean == expected
    assert control.predict(rows[20:30]) == (expected,) * 10
    # the whole-dataset mean is a different number, and is not what it learned
    with localcontext() as context:
        context.prec = models.MODEL_PRECISION
        assert control.mean != sum(row.label for row in rows) / Decimal(len(rows))


def test_a_prediction_depends_on_its_own_row_and_nothing_else(models, dataset, train_rows,
                                                              walk_forward):
    """Row-wise independence is what makes `predict` free of block statistics.

    Any preprocessing refitted on the block being predicted -- the classic
    `fit_transform(X_test)` -- breaks this immediately, even though the
    fitted state and the determinism checks stay perfectly happy.
    """
    columns = models.feature_columns_of(dataset)
    model = models.RidgeRegressionPredictor(feature_columns=columns)
    model.fit(train_rows)
    block = walk_forward.usable_rows(dataset)[30:42]
    together = model.predict(block)
    one_at_a_time = tuple(model.predict([row])[0] for row in block)
    assert together == one_at_a_time
    # and the same row keeps its prediction inside a differently-shaped block
    shuffled = (block[7], block[0], block[3])
    assert model.predict(shuffled) == (together[7], together[0], together[3])
