"""Phase 3A: the ridge baseline and the proofs that it cannot cheat.

Most of these tests are not about regression quality. They are about the
three boundaries the model sits on: labels, feature order, and the float64
crossing. A model that scores well because it peeked is worse than no model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json

import pytest

# Phase 3 is the ML layer: every test here needs the optional [ml] extra.
# Marked at module granularity -- see docs/TRADING_LAB_PHASE3.md.
pytestmark = pytest.mark.ml


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


# ==========================================================================
# Phase 3B -- gradient-boosted candidate on the same rails
# ==========================================================================


@pytest.fixture
def xgb_columns(models, dataset):
    return models.feature_columns_of(dataset)


def _xgb(models, columns, **overrides):
    config = models.XGBoostConfig(**overrides) if overrides else None
    return models.XGBoostRegressionPredictor(feature_columns=columns, config=config)


def test_the_boosted_model_fits_and_predicts_finite_decimals(models, xgb_columns, train_rows,
                                                             walk_forward, dataset):
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    later = walk_forward.usable_rows(dataset)[30:40]
    predictions = model.predict(later)
    assert len(predictions) == len(later)
    assert all(isinstance(value, Decimal) and value.is_finite() for value in predictions)
    assert model.fitted.train_rows == len(train_rows)
    assert model.fitted.booster_bytes > 0
    assert len(model.fitted.booster_sha256) == 64
    assert model.predict(()) == ()


def test_what_is_hashed_is_what_the_backend_is_actually_given(models, xgb_columns, train_rows):
    """A parameter that runs but is not hashed makes the identity a fiction."""
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    declared = dict(model.config.as_hyperparameters())
    actual = model._booster.get_params()
    for name, value in declared.items():
        assert str(actual[name]) == value or float(actual[name]) == float(value), name
    # every declared hyperparameter also reaches the spec
    hashed = dict(model.spec.hyperparameters)
    assert set(declared).issubset(hashed)
    assert hashed["backend"] == "xgboost"
    assert hashed["backend_version"] == importlib.import_module("xgboost").__version__


def test_the_boosted_model_carries_no_scaler_at_all(models, xgb_columns, train_rows):
    """Trees split on order, not on scale. A scaler here would be pure surface area."""
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    assert not hasattr(model.fitted, "feature_means")
    assert not hasattr(model.fitted, "feature_stdevs")
    assert not hasattr(model, "feature_means")


@pytest.mark.parametrize("overrides,fragment", [
    ({"n_estimators": 0}, "n_estimators"),
    ({"max_depth": -2}, "max_depth"),
    ({"n_estimators": True}, "n_estimators"),
    ({"n_jobs": 4}, "n_jobs"),
    ({"subsample": Decimal("0.8")}, "stochastic"),
    ({"colsample_bytree": Decimal("0.5")}, "stochastic"),
    ({"learning_rate": Decimal(0)}, "learning_rate"),
    ({"learning_rate": 0.05}, "learning_rate"),
    ({"gamma": Decimal("-1")}, "gamma"),
    ({"tree_method": "gpu_hist"}, "tree_method"),
    ({"objective": ""}, "objective"),
    ({"random_state": -1}, "random_state"),
])
def test_a_configuration_that_breaks_reproducibility_is_refused(models, xgb_columns,
                                                                overrides, fragment):
    with pytest.raises(models.ModelError, match=fragment):
        _xgb(models, xgb_columns, **overrides)


def test_the_boosted_model_validates_the_feature_schema_like_the_ridge_one(models, dataset,
                                                                          train_rows):
    columns = models.feature_columns_of(dataset)
    permuted = (columns[2], columns[0], columns[1])
    with pytest.raises(models.ModelError, match="exact order"):
        _xgb(models, permuted).fit(train_rows)
    with pytest.raises(models.ModelError, match="expects"):
        _xgb(models, columns[:2]).fit(train_rows)


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity")])
def test_a_missing_value_is_refused_instead_of_handed_to_the_backend(models, xgb_columns,
                                                                     train_rows, bad):
    """XGBoost treats NaN as 'missing' natively. V1 does not want that silence."""
    poisoned = replace(train_rows[5], features=(("sma5", bad), *train_rows[5].features[1:]))
    with pytest.raises(models.ModelError, match="finite"):
        _xgb(models, xgb_columns).fit((*train_rows[:5], poisoned, *train_rows[6:]))
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    with pytest.raises(models.ModelError, match="finite"):
        model.predict((poisoned,))


def test_the_boosted_model_refuses_unusable_rows_and_premature_prediction(models, xgb_columns,
                                                                         dataset, train_rows,
                                                                         walk_forward):
    unusable = [row for row in dataset.rows if not row.usable]
    with pytest.raises(models.ModelError, match="not usable"):
        _xgb(models, xgb_columns).fit(walk_forward.usable_rows(dataset)[:20] + (unusable[0],))
    with pytest.raises(models.ModelError, match="before fit"):
        _xgb(models, xgb_columns).predict(train_rows)
    with pytest.raises(models.ModelError, match="at least two"):
        _xgb(models, xgb_columns).fit(train_rows[:1])
    with pytest.raises(models.ModelError, match="before fit"):
        _xgb(models, xgb_columns).feature_importances()


# --- boundaries -----------------------------------------------------------


def test_the_boosted_model_never_reads_a_label_when_predicting(models, xgb_columns, train_rows,
                                                               dataset, walk_forward):
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    later = walk_forward.usable_rows(dataset)[30:40]
    assert model.predict([LabelTrap(row) for row in later]) == model.predict(later)


def test_changing_test_labels_leaves_the_boosted_model_and_its_predictions_alone(
    models, dataset, walk_forward, xgb_columns
):
    config = _config(walk_forward)
    folds = walk_forward.build_folds(dataset, config=config)
    fold_zero_test = {row.bar_open_at for row in folds[0][2]}
    tampered = replace(dataset, rows=tuple(
        replace(row, label=row.label * Decimal(-4) - Decimal("0.25"))
        if row.bar_open_at in fold_zero_test else row
        for row in dataset.rows
    ))
    first = walk_forward.evaluate(dataset, config=config, predictor=_xgb(models, xgb_columns))
    second = walk_forward.evaluate(tampered, config=config, predictor=_xgb(models, xgb_columns))
    assert [r.prediction for r in first.folds[0].records] == \
           [r.prediction for r in second.folds[0].records]
    assert first.folds[0].metrics != second.folds[0].metrics


def test_a_boosted_prediction_depends_only_on_its_own_row(models, xgb_columns, train_rows,
                                                          dataset, walk_forward):
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    block = walk_forward.usable_rows(dataset)[30:42]
    together = model.predict(block)
    assert together == tuple(model.predict([row])[0] for row in block)


def test_rows_after_a_fold_cannot_reach_back_into_the_boosted_fold(models, dataset,
                                                                   walk_forward, xgb_columns):
    config = _config(walk_forward)
    folds = walk_forward.build_folds(dataset, config=config)
    boundary = folds[0][2][-1].bar_open_at
    rewritten = replace(dataset, rows=tuple(
        replace(row,
                label=row.label * Decimal(5) if row.label is not None else None,
                features=tuple((column, None if value is None else value - Decimal("500"))
                               for column, value in row.features))
        if row.bar_open_at > boundary else row
        for row in dataset.rows
    ))
    before = walk_forward.evaluate(dataset, config=config, predictor=_xgb(models, xgb_columns))
    after = walk_forward.evaluate(rewritten, config=config, predictor=_xgb(models, xgb_columns))
    assert before.folds[0].records == after.folds[0].records
    assert before.folds[0].metrics == after.folds[0].metrics
    assert before.folds[-1].records != after.folds[-1].records


def test_the_boosted_model_never_mutates_the_rows_it_is_given(models, dataset, walk_forward,
                                                              xgb_columns):
    snapshot = tuple(dataset.rows)
    walk_forward.evaluate(dataset, config=_config(walk_forward),
                          predictor=_xgb(models, xgb_columns))
    assert dataset.rows == snapshot


# --- determinism and identity ---------------------------------------------


def test_three_identical_boosted_fits_produce_the_identical_model(models, xgb_columns,
                                                                  train_rows, dataset,
                                                                  walk_forward):
    later = walk_forward.usable_rows(dataset)[30:40]
    results = []
    for _ in range(3):
        model = _xgb(models, xgb_columns)
        model.fit(train_rows)
        results.append((model.fitted.booster_sha256, model.fitted.fitted_model_hash,
                        model.predict(later)))
    assert len({item[0] for item in results}) == 1
    assert len({item[1] for item in results}) == 1
    assert results[0][2] == results[1][2] == results[2][2]


def test_the_boosted_model_is_indifferent_to_the_callers_decimal_context(models, xgb_columns,
                                                                        train_rows, dataset,
                                                                        walk_forward):
    later = walk_forward.usable_rows(dataset)[30:40]
    original = getcontext().prec
    seen = set()
    try:
        for precision in (5, 7, 28, 34, 80):
            getcontext().prec = precision
            model = _xgb(models, xgb_columns)
            model.fit(train_rows)
            seen.add((model.fitted.fitted_model_hash,
                      tuple(str(value) for value in model.predict(later))))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_every_boosted_hyperparameter_moves_the_spec_hash(models, xgb_columns):
    baseline = _xgb(models, xgb_columns).model_spec_hash
    assert len(baseline) == 64
    assert _xgb(models, xgb_columns).model_spec_hash == baseline
    for overrides in ({"n_estimators": 50}, {"max_depth": 5},
                      {"learning_rate": Decimal("0.1")}, {"min_child_weight": 3},
                      {"reg_alpha": Decimal("0.5")}, {"reg_lambda": Decimal("2.0")},
                      {"gamma": Decimal("0.1")}, {"tree_method": "exact"},
                      {"objective": "reg:absoluteerror"}, {"random_state": 7}):
        assert _xgb(models, xgb_columns, **overrides).model_spec_hash != baseline, overrides
    # and a different feature schema is a different model too
    assert models.XGBoostRegressionPredictor(
        feature_columns=xgb_columns[:2]).model_spec_hash != baseline
    # the two candidates are never the same model
    assert models.RidgeRegressionPredictor(
        feature_columns=xgb_columns).model_spec_hash != baseline


def test_the_fitted_hash_tracks_the_booster_not_just_the_configuration(models, xgb_columns,
                                                                       train_rows):
    """Hashing a handful of predictions would call two different trees equal."""
    first = _xgb(models, xgb_columns)
    first.fit(train_rows)
    relabelled = tuple(replace(row, label=row.label * Decimal(3) + Decimal("0.01"))
                       for row in train_rows)
    second = _xgb(models, xgb_columns)
    second.fit(relabelled)
    assert second.model_spec_hash == first.model_spec_hash      # same definition
    assert second.fitted.booster_sha256 != first.fitted.booster_sha256
    assert second.fitted.fitted_model_hash != first.fitted.fitted_model_hash
    # same data again reproduces the same learned state exactly
    third = _xgb(models, xgb_columns)
    third.fit(train_rows)
    assert third.fitted.fitted_model_hash == first.fitted.fitted_model_hash
    assert first.fitted.fitted_model_hash != first.model_spec_hash


def test_feature_importances_are_read_only_introspection(models, xgb_columns, train_rows):
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    before = model.fitted.fitted_model_hash
    importances = model.feature_importances()
    assert [column for column, _ in importances] == list(xgb_columns)
    assert all(isinstance(value, Decimal) and value >= 0 for _, value in importances)
    # looking at the model does not change the model, and nothing selects on it
    assert model.fitted.fitted_model_hash == before
    assert model.feature_importances() == importances


# --- the shared protocol --------------------------------------------------


def test_the_boosted_model_runs_inside_the_walk_forward_protocol(models, dataset, walk_forward,
                                                                 xgb_columns):
    config = _config(walk_forward)
    evaluation = walk_forward.evaluate(dataset, config=config,
                                       predictor=_xgb(models, xgb_columns))
    assert len(evaluation.folds) >= 3
    assert evaluation.spec.predictor_name == "xgboost_regression"
    assert evaluation.oos_metrics.observations == sum(f.metrics.observations
                                                      for f in evaluation.folds)
    again = walk_forward.evaluate(dataset, config=config, predictor=_xgb(models, xgb_columns))
    assert again.results_hash == evaluation.results_hash


def test_four_candidates_run_through_one_identical_protocol(models, dataset, walk_forward,
                                                            xgb_columns):
    """Phase 3B compares; it does not choose. Selection needs validation, later.

    These numbers come from a synthetic fixture: they are a protocol check,
    not evidence of a trading edge, and nothing here asserts a winner.
    """
    config = _config(walk_forward)

    class Zero:
        name, version = "zero", "1"
        def fit(self, rows): pass
        def predict(self, rows): return tuple(Decimal(0) for _ in rows)

    evaluations = {
        "zero": walk_forward.evaluate(dataset, config=config, predictor=Zero()),
        "mean": walk_forward.evaluate(dataset, config=config,
                                      predictor=models.MeanTrainPredictor()),
        "ridge": walk_forward.evaluate(
            dataset, config=config,
            predictor=models.RidgeRegressionPredictor(feature_columns=xgb_columns)),
        "xgboost": walk_forward.evaluate(dataset, config=config,
                                         predictor=_xgb(models, xgb_columns)),
    }
    counts = {name: evaluation.oos_metrics.observations
              for name, evaluation in evaluations.items()}
    assert len(set(counts.values())) == 1        # identical out-of-sample set
    timestamps = {name: tuple(r.bar_open_at for r in evaluation.oos_records)
                  for name, evaluation in evaluations.items()}
    assert len(set(timestamps.values())) == 1    # identical observations, not merely as many
    for name, evaluation in evaluations.items():
        assert evaluation.oos_metrics.mae is not None, name
        assert evaluation.oos_metrics.rmse is not None, name
    assert len({evaluation.spec_hash for evaluation in evaluations.values()}) == 4
    assert len({record.prediction for record in evaluations["xgboost"].oos_records}) > 1


def test_the_boosted_model_actually_learns_the_block_it_was_trained_on(models, xgb_columns,
                                                                       train_rows, walk_forward):
    """In-sample recall, the cheapest guard against a fit/predict mismatch.

    A model whose feature order differs between fitting and predicting still
    returns finite Decimals, still behaves deterministically, and still passes
    every boundary test -- it is simply wrong. Only asking it to reproduce
    what it was taught exposes that.

    The comparison is deliberately threshold-free. An absolute bound like
    "beat naive/10" is a fixture artefact: measured across two different
    synthetic series the correct model landed at naive/20.6 on one and
    naive/4.7 on the other, while the permuted variant landed at naive/4.8 and
    naive/0.86 -- no single constant separates them everywhere. So instead of
    guessing a constant, the test asks the model to rank the two orderings
    itself: whatever the fixture, feeding it the columns as it learned them
    must beat feeding it the same columns permuted.
    """
    model = _xgb(models, xgb_columns)
    model.fit(train_rows)
    labels = tuple(row.label for row in train_rows)
    permuted = tuple(
        replace(row, features=tuple(zip([column for column, _ in row.features],
                                        [value for _, value in row.features][::-1])))
        for row in train_rows
    )
    with localcontext() as context:
        context.prec = models.MODEL_PRECISION
        naive = tuple(sum(labels) / Decimal(len(labels)) for _ in labels)
    fitted_error = walk_forward.mean_absolute_error(model.predict(train_rows), labels)
    permuted_error = walk_forward.mean_absolute_error(model.predict(permuted), labels)
    naive_error = walk_forward.mean_absolute_error(naive, labels)
    assert fitted_error < naive_error          # it learned something at all
    assert fitted_error < permuted_error       # and it learned THIS column order


def test_no_configuration_field_can_escape_the_spec_hash(models, xgb_columns):
    """A hyperparameter added later must not run silently unhashed."""
    config = models.XGBoostConfig()
    assert set(config.HASHED_FIELDS) == {field.name for field in dataclasses.fields(config)}
    narrowed = type(config).HASHED_FIELDS
    try:
        type(config).HASHED_FIELDS = narrowed[:-1]
        with pytest.raises(models.ModelError, match="not covered by the spec hash"):
            config.as_hyperparameters()
    finally:
        type(config).HASHED_FIELDS = narrowed
    assert dict(config.as_hyperparameters())["random_state"] == "0"
