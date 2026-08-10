"""Phase 3A: the first real predictor plugged into the Phase 2 walk-forward protocol.

The model here is deliberately the least impressive one that can still be
wrong in interesting ways: a ridge regression on the canonical feature
columns of a Phase 2C dataset. Phase 3A exists to prove the predictive engine
is *sound*, not that it is profitable. A scientific control has to be simple
enough that any surviving edge cannot be blamed on the model's cleverness.

Three boundaries matter more than the estimator:

* **Numeric boundary.** Phase 1 and 2 are Decimal end to end. The ridge solve
  is not: `sklearn` works in float64. So the crossing is explicit and narrow
  -- standardisation is computed in Decimal, the design matrix is handed to
  the solver as float64, and the learned coefficients come straight back to
  Decimal, quantised to a fixed exponent. Inference is then pure Decimal:
  once fitted, no float ever touches a prediction. This module does not
  pretend to be Decimal end to end; it is Decimal on both banks of one
  float64 river.
* **Feature boundary.** Columns are taken in the canonical order the dataset
  declares. The order is validated against every row rather than trusted,
  because a silently permuted matrix trains fine and predicts nonsense.
* **Label boundary.** `fit` reads labels. `predict` never touches `.label` at
  all -- not defensively, structurally: it only ever reads `.features`.

Nothing here opens a store, a snapshot, or a clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json

from sklearn.linear_model import Ridge

MODEL_SCHEMA_VERSION = "trading-lab.model.v1"
FEATURE_SCHEMA_VERSION = "trading-lab.model-features.v1"
MODEL_PRECISION = 34
# Learned parameters are rounded to this exponent on the way back from
# float64, so a fitted model has one exact textual form to hash.
COEFFICIENT_EXPONENT = Decimal("1E-18")
DEFAULT_RIDGE_ALPHA = Decimal("1.0")
RIDGE_SOLVER = "cholesky"  # closed form, no RNG, no iteration count
MAX_FEATURES = 256


class ModelError(RuntimeError):
    """Raised when a model refuses the data or the configuration it is given."""


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FeatureSchema:
    """The columns a model consumes, in the order it consumes them."""

    columns: tuple[str, ...]
    version: str = FEATURE_SCHEMA_VERSION

    def canonical(self) -> dict[str, object]:
        return {"version": self.version, "columns": list(self.columns)}


@dataclass(frozen=True)
class ModelSpec:
    """The DEFINITION of a model. Never its learned state."""

    name: str
    version: str
    hyperparameters: tuple[tuple[str, str], ...]
    feature_schema: FeatureSchema

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "hyperparameters": [list(pair) for pair in self.hyperparameters],
            "feature_schema": self.feature_schema.canonical(),
        }

    @property
    def model_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class FittedRidge:
    """The learned state. Hashed separately from the spec that produced it."""

    spec: ModelSpec
    model_spec_hash: str
    feature_means: tuple[Decimal, ...]
    feature_stdevs: tuple[Decimal, ...]
    coefficients: tuple[Decimal, ...]
    intercept: Decimal
    train_rows: int

    @property
    def fitted_model_hash(self) -> str:
        return _sha256_canonical(
            {
                "spec": self.spec.canonical(),
                "feature_means": [str(value) for value in self.feature_means],
                "feature_stdevs": [str(value) for value in self.feature_stdevs],
                "coefficients": [str(value) for value in self.coefficients],
                "intercept": str(self.intercept),
                "train_rows": self.train_rows,
            }
        )


def feature_columns_of(dataset) -> tuple[str, ...]:
    """The canonical column order, taken from the dataset's own configuration."""
    return tuple(feature.column for feature in dataset.config.features)


def _require_decimal(value: object, *, where: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ModelError(f"{where} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise ModelError(f"{where} must be finite, got {value}")
    return value


def feature_vector(row, columns: tuple[str, ...]) -> tuple[Decimal, ...]:
    """Read one row's features in canonical order, refusing anything doubtful.

    Only `.features` is read. A row object whose `.label` explodes on access
    passes through here untouched -- which is what makes the label isolation
    a structural property rather than a promise.
    """
    if not row.usable:
        raise ModelError(f"row {row.bar_open_at} is not usable and must not be imputed")
    present = tuple(column for column, _ in row.features)
    if present != columns:
        raise ModelError(
            f"row {row.bar_open_at} has feature columns {present} "
            f"but the model expects {columns} in that exact order"
        )
    return tuple(
        _require_decimal(value, where=f"feature {column} of {row.bar_open_at}")
        for column, value in row.features
    )


def _standardisation(matrix: tuple[tuple[Decimal, ...], ...]):
    """Mean and population standard deviation, per column, over TRAIN only."""
    count = Decimal(len(matrix))
    means: list[Decimal] = []
    stdevs: list[Decimal] = []
    with localcontext() as context:
        context.prec = MODEL_PRECISION
        for index in range(len(matrix[0])):
            column = [row[index] for row in matrix]
            mean = sum(column) / count
            variance = sum((value - mean) ** 2 for value in column) / count
            if variance == 0:
                raise ModelError(
                    f"feature at position {index} is constant over the training block; "
                    "it carries no information and cannot be standardised"
                )
            means.append(mean)
            stdevs.append(variance.sqrt())
    return tuple(means), tuple(stdevs)


def _standardise(
    vector: tuple[Decimal, ...],
    means: tuple[Decimal, ...],
    stdevs: tuple[Decimal, ...],
) -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec = MODEL_PRECISION
        return tuple(
            (value - mean) / stdev for value, mean, stdev in zip(vector, means, stdevs)
        )


def _to_decimal(value: float) -> Decimal:
    """Cross back from float64 to Decimal at a fixed, caller-independent exponent.

    `quantize` obeys the *ambient* context precision, so this must run pinned
    to the module's own precision -- otherwise a caller working at prec=7
    turns the crossing into an InvalidOperation.
    """
    with localcontext() as context:
        context.prec = MODEL_PRECISION
        return Decimal(repr(float(value))).quantize(
            COEFFICIENT_EXPONENT, rounding=ROUND_HALF_EVEN
        )


class RidgeRegressionPredictor:
    """L2-penalised linear regression on the dataset's canonical features."""

    name = "ridge_regression"
    version = "1"

    def __init__(self, *, feature_columns, alpha: Decimal = DEFAULT_RIDGE_ALPHA) -> None:
        columns = tuple(feature_columns)
        if not columns or len(columns) > MAX_FEATURES:
            raise ModelError(f"feature_columns must hold 1..{MAX_FEATURES} names")
        if len(set(columns)) != len(columns):
            raise ModelError(f"feature_columns contains duplicates: {columns}")
        if not all(isinstance(column, str) and column for column in columns):
            raise ModelError("every feature column must be a non-empty string")
        if not isinstance(alpha, Decimal) or not alpha.is_finite() or alpha <= 0:
            raise ModelError(f"alpha must be a finite positive Decimal, got {alpha!r}")
        self.feature_columns = columns
        self.alpha = alpha
        self.spec = ModelSpec(
            name=self.name,
            version=self.version,
            hyperparameters=(
                ("alpha", str(alpha)),
                ("fit_intercept", "true"),
                ("solver", RIDGE_SOLVER),
                ("standardised", "true"),
                ("coefficient_exponent", str(COEFFICIENT_EXPONENT)),
            ),
            feature_schema=FeatureSchema(columns=columns),
        )
        self.fitted: FittedRidge | None = None

    @property
    def model_spec_hash(self) -> str:
        return self.spec.model_spec_hash

    def fit(self, train_rows) -> "RidgeRegressionPredictor":
        rows = tuple(train_rows)
        if len(rows) < 2:
            raise ModelError("ridge regression needs at least two training rows")
        matrix = tuple(feature_vector(row, self.feature_columns) for row in rows)
        targets = tuple(
            _require_decimal(row.label, where=f"label of {row.bar_open_at}") for row in rows
        )
        # Standardisation is part of the learned state: TRAIN only, always.
        means, stdevs = _standardisation(matrix)
        design = [[float(value) for value in _standardise(row, means, stdevs)] for row in matrix]
        solver = Ridge(alpha=float(self.alpha), fit_intercept=True, solver=RIDGE_SOLVER)
        solver.fit(design, [float(target) for target in targets])
        self.fitted = FittedRidge(
            spec=self.spec,
            model_spec_hash=self.spec.model_spec_hash,
            feature_means=means,
            feature_stdevs=stdevs,
            coefficients=tuple(_to_decimal(value) for value in solver.coef_),
            intercept=_to_decimal(solver.intercept_),
            train_rows=len(rows),
        )
        return self

    def predict(self, rows) -> tuple[Decimal, ...]:
        fitted = self.fitted
        if fitted is None:
            raise ModelError("predict() called before fit()")
        predictions: list[Decimal] = []
        with localcontext() as context:
            context.prec = MODEL_PRECISION
            for row in rows:
                vector = _standardise(
                    feature_vector(row, self.feature_columns),
                    fitted.feature_means,
                    fitted.feature_stdevs,
                )
                predictions.append(
                    sum(
                        (coefficient * value
                         for coefficient, value in zip(fitted.coefficients, vector)),
                        fitted.intercept,
                    )
                )
        return tuple(predictions)


class MeanTrainPredictor:
    """Control: predicts the mean TRAIN label, whatever the features say.

    Not a strawman -- for a forward return this is a genuinely hard baseline
    to beat, and any model that cannot separate itself from it has shown
    nothing.
    """

    name = "mean_train"
    version = "1"

    def __init__(self) -> None:
        self.spec = ModelSpec(
            name=self.name,
            version=self.version,
            hyperparameters=(),
            feature_schema=FeatureSchema(columns=()),
        )
        self.mean: Decimal | None = None

    @property
    def model_spec_hash(self) -> str:
        return self.spec.model_spec_hash

    def fit(self, train_rows) -> "MeanTrainPredictor":
        rows = tuple(train_rows)
        if not rows:
            raise ModelError("mean predictor needs at least one training row")
        with localcontext() as context:
            context.prec = MODEL_PRECISION
            self.mean = sum(
                _require_decimal(row.label, where=f"label of {row.bar_open_at}") for row in rows
            ) / Decimal(len(rows))
        return self

    def predict(self, rows) -> tuple[Decimal, ...]:
        if self.mean is None:
            raise ModelError("predict() called before fit()")
        return tuple(self.mean for _ in rows)
