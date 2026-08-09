"""Phase 2D: causal expanding-window walk-forward evaluation.

This module builds the PROTOCOL a future model must enter, not a model. It
takes a Phase 2C `Dataset` and a predictor, and produces per-fold and global
out-of-sample scores that a later run can reproduce exactly.

The design is shaped by one adversary: a label at T reaches T+h, so a training
row can know something about the block that follows it. Three rules answer
that.

* **Purge is measured in market time, not in rows.** Usable rows are not
  contiguous -- indicator warm-up and market gaps punch holes in them -- so
  "drop 4 rows" does not mean "drop 4 bars". The boundary is enforced by
  timestamp: rows are dropped from the end of a block until no remaining
  label window can reach the first row of the next block. `purge_rows` is a
  floor on that, never the whole mechanism.
* **`fit` only ever sees train.** Validation is a separate block for later
  calibration work; the test block is never passed to `fit` at all.
* **Global out-of-sample is recomputed, not averaged.** Per-fold metrics are
  reported, but the headline numbers come from re-scoring the concatenated
  test observations. Averaging fold correlations would weight a 10-row fold
  like a 200-row one, and `step_rows >= test_rows` keeps those observations
  from being counted twice.

Metric note: `rank_ic` here is a Spearman rank correlation over the
observations of one evaluation block for a single instrument. It is NOT a
cross-sectional information coefficient, and nothing in this module pretends
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.market_dataset import Dataset, DatasetRow

WALK_FORWARD_SCHEMA_VERSION = "trading-lab.walk-forward.v1"
METRIC_SCHEMA_VERSION = "trading-lab.walk-forward-metrics.v1"
EVALUATION_PRECISION = 34
MAX_ROWS_PARAMETER = 1_000_000


class WalkForwardError(RuntimeError):
    """Raised when an evaluation cannot be produced safely."""


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_rows: int
    validation_rows: int
    test_rows: int
    step_rows: int
    purge_rows: int

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": WALK_FORWARD_SCHEMA_VERSION,
            "min_train_rows": self.min_train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "step_rows": self.step_rows,
            "purge_rows": self.purge_rows,
        }


@dataclass(frozen=True)
class BlockSpan:
    first: str | None
    last: str | None
    count: int


@dataclass(frozen=True)
class PredictionRecord:
    fold_index: int
    bar_open_at: str
    prediction: Decimal
    actual_forward_return: Decimal


@dataclass(frozen=True)
class Metrics:
    rank_ic: Decimal | None
    mae: Decimal | None
    rmse: Decimal | None
    observations: int


@dataclass(frozen=True)
class FoldEvaluation:
    fold_index: int
    train: BlockSpan
    validation: BlockSpan
    test: BlockSpan
    records: tuple[PredictionRecord, ...]
    metrics: Metrics


@dataclass(frozen=True)
class EvaluationSpec:
    """The DEFINITION of an evaluation -- never its outputs."""

    dataset_hash: str
    config: WalkForwardConfig
    metric_version: str
    predictor_name: str
    predictor_version: str

    def canonical(self) -> dict[str, object]:
        return {
            "dataset_hash": self.dataset_hash,
            "config": self.config.canonical(),
            "metric_version": self.metric_version,
            "predictor": {"name": self.predictor_name, "version": self.predictor_version},
        }

    @property
    def spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class WalkForwardEvaluation:
    spec: EvaluationSpec
    spec_hash: str
    folds: tuple[FoldEvaluation, ...]
    oos_records: tuple[PredictionRecord, ...]
    oos_metrics: Metrics
    results_hash: str


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _require_count(name: str, value: object, *, minimum: int = 1) -> int:
    # bool is an int subclass; True must not read as 1 here.
    if type(value) is not int or not minimum <= value <= MAX_ROWS_PARAMETER:
        raise WalkForwardError(
            f"{name} must be an int in {minimum}..{MAX_ROWS_PARAMETER}, got {value!r}"
        )
    return value


def _require_config(config: WalkForwardConfig) -> None:
    _require_count("min_train_rows", config.min_train_rows)
    _require_count("validation_rows", config.validation_rows)
    _require_count("test_rows", config.test_rows)
    _require_count("step_rows", config.step_rows)
    _require_count("purge_rows", config.purge_rows, minimum=0)
    if config.step_rows < config.test_rows:
        # Overlapping test windows would count the same observation twice in
        # the global out-of-sample score.
        raise WalkForwardError(
            f"step_rows ({config.step_rows}) must be >= test_rows ({config.test_rows}) "
            "so out-of-sample windows never overlap"
        )


# --- metrics --------------------------------------------------------------


def _average_ranks(values: tuple[Decimal, ...]) -> list[Decimal]:
    """Ranks with ties resolved to their average, as Spearman requires."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks: list[Decimal] = [Decimal(0)] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        # positions are 0-based; ranks are 1-based averages over the tie group
        shared = (Decimal(position + 1) + Decimal(end + 1)) / Decimal(2)
        for slot in range(position, end + 1):
            ranks[order[slot]] = shared
        position = end + 1
    return ranks


def _pearson(left: list[Decimal], right: list[Decimal]) -> Decimal | None:
    count = Decimal(len(left))
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    if variance_left == 0 or variance_right == 0:
        # Undefined, not zero: a constant vector has no direction to correlate.
        return None
    return covariance / (variance_left.sqrt() * variance_right.sqrt())


def rank_ic(predictions: tuple[Decimal, ...], actuals: tuple[Decimal, ...]) -> Decimal | None:
    """Spearman rank correlation over one block of observations.

    None when it is mathematically undefined: fewer than two observations, or
    a constant vector on either side. Never silently reported as 0.
    """
    if len(predictions) != len(actuals):
        raise WalkForwardError("predictions and actuals must have the same length")
    if len(predictions) < 2:
        return None
    with localcontext() as context:
        context.prec = EVALUATION_PRECISION
        return _pearson(_average_ranks(predictions), _average_ranks(actuals))


def mean_absolute_error(
    predictions: tuple[Decimal, ...], actuals: tuple[Decimal, ...]
) -> Decimal | None:
    if len(predictions) != len(actuals):
        raise WalkForwardError("predictions and actuals must have the same length")
    if not predictions:
        return None
    with localcontext() as context:
        context.prec = EVALUATION_PRECISION
        return sum(abs(p - a) for p, a in zip(predictions, actuals)) / Decimal(len(predictions))


def root_mean_squared_error(
    predictions: tuple[Decimal, ...], actuals: tuple[Decimal, ...]
) -> Decimal | None:
    if len(predictions) != len(actuals):
        raise WalkForwardError("predictions and actuals must have the same length")
    if not predictions:
        return None
    with localcontext() as context:
        context.prec = EVALUATION_PRECISION
        mean_square = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / Decimal(
            len(predictions)
        )
        return mean_square.sqrt()


def _metrics(records: tuple[PredictionRecord, ...]) -> Metrics:
    predictions = tuple(record.prediction for record in records)
    actuals = tuple(record.actual_forward_return for record in records)
    return Metrics(
        rank_ic=rank_ic(predictions, actuals),
        mae=mean_absolute_error(predictions, actuals),
        rmse=root_mean_squared_error(predictions, actuals),
        observations=len(records),
    )


# --- folds ----------------------------------------------------------------


def usable_rows(dataset: Dataset) -> tuple[DatasetRow, ...]:
    """The rows an evaluation may touch, in the order the dataset holds them.

    Fail closed rather than repair: a dataset whose openings are not already
    ascending is a broken upstream contract, not something to sort here.
    """
    rows = tuple(row for row in dataset.rows if row.usable)
    openings = [row.bar_open_at for row in rows]
    if openings != sorted(openings):
        raise WalkForwardError("dataset rows are not in ascending opening order")
    return rows


def _purge(
    earlier: tuple[DatasetRow, ...],
    later: tuple[DatasetRow, ...],
    *,
    horizon: int,
    duration: timedelta,
    minimum: int,
) -> tuple[DatasetRow, ...]:
    """Drop the tail of `earlier` whose label window reaches into `later`.

    Row counting alone is not enough: usable rows can be far apart in market
    time. The condition enforced is the one that actually matters --
    timestamp + horizon bars must land strictly before the next block starts.
    """
    if not earlier or not later:
        return earlier
    boundary = datetime.fromisoformat(later[0].bar_open_at)
    kept = earlier[: max(len(earlier) - minimum, 0)]
    while kept:
        label_end = datetime.fromisoformat(kept[-1].bar_open_at) + duration * horizon
        if label_end < boundary:
            break
        kept = kept[:-1]
    return kept


def build_folds(dataset: Dataset, *, config: WalkForwardConfig):
    """Expanding-window folds: train always starts at the first usable row."""
    _require_config(config)
    if dataset.timeframe not in TIMEFRAME_DURATIONS:
        raise WalkForwardError(f"unsupported timeframe {dataset.timeframe!r}")
    duration = TIMEFRAME_DURATIONS[dataset.timeframe]
    horizon = dataset.config.label.horizon
    rows = usable_rows(dataset)

    folds: list[tuple[tuple[DatasetRow, ...], tuple[DatasetRow, ...], tuple[DatasetRow, ...]]] = []
    train_end = config.min_train_rows
    while True:
        validation_end = train_end + config.validation_rows
        test_end = validation_end + config.test_rows
        if test_end > len(rows):
            break
        train = _purge(rows[:train_end], rows[train_end:validation_end],
                       horizon=horizon, duration=duration, minimum=config.purge_rows)
        validation = _purge(rows[train_end:validation_end], rows[validation_end:test_end],
                            horizon=horizon, duration=duration, minimum=config.purge_rows)
        if train and validation:
            folds.append((train, validation, rows[validation_end:test_end]))
        train_end += config.step_rows
    return tuple(folds)


def _span(rows: tuple[DatasetRow, ...]) -> BlockSpan:
    if not rows:
        return BlockSpan(first=None, last=None, count=0)
    return BlockSpan(first=rows[0].bar_open_at, last=rows[-1].bar_open_at, count=len(rows))


def _require_predictions(
    predictions: object, expected: int
) -> tuple[Decimal, ...]:
    values = tuple(predictions)
    if len(values) != expected:
        raise WalkForwardError(
            f"predictor returned {len(values)} predictions for {expected} rows"
        )
    for value in values:
        if not isinstance(value, Decimal):
            raise WalkForwardError(f"prediction {value!r} is not a Decimal")
        if not value.is_finite():
            raise WalkForwardError(f"prediction {value!r} is not finite")
    return values


def evaluate(dataset: Dataset, *, config: WalkForwardConfig, predictor) -> WalkForwardEvaluation:
    """Run the walk-forward protocol and score it.

    `predictor` is fitted on the train block of each fold and only ever
    predicts on the test block. The test block is never handed to `fit`.
    """
    folds_rows = build_folds(dataset, config=config)
    spec = EvaluationSpec(
        dataset_hash=dataset.dataset_hash,
        config=config,
        metric_version=METRIC_SCHEMA_VERSION,
        predictor_name=getattr(predictor, "name", type(predictor).__name__),
        predictor_version=getattr(predictor, "version", "unversioned"),
    )
    evaluations: list[FoldEvaluation] = []
    oos: list[PredictionRecord] = []
    for fold_index, (train, validation, test) in enumerate(folds_rows):
        predictor.fit(train)
        predictions = _require_predictions(predictor.predict(test), len(test))
        records = tuple(
            PredictionRecord(
                fold_index=fold_index,
                bar_open_at=row.bar_open_at,
                prediction=prediction,
                actual_forward_return=row.label,
            )
            for row, prediction in zip(test, predictions)
        )
        evaluations.append(
            FoldEvaluation(
                fold_index=fold_index,
                train=_span(train),
                validation=_span(validation),
                test=_span(test),
                records=records,
                metrics=_metrics(records),
            )
        )
        oos.extend(records)

    oos_records = tuple(oos)
    results_hash = _sha256_canonical(
        {
            "spec": spec.canonical(),
            "records": [
                {
                    "fold_index": record.fold_index,
                    "bar_open_at": record.bar_open_at,
                    "prediction": str(record.prediction),
                    "actual_forward_return": str(record.actual_forward_return),
                }
                for record in oos_records
            ],
        }
    )
    return WalkForwardEvaluation(
        spec=spec,
        spec_hash=spec.spec_hash,
        folds=tuple(evaluations),
        oos_records=oos_records,
        # Recomputed on the concatenated observations, never averaged per fold.
        oos_metrics=_metrics(oos_records),
        results_hash=results_hash,
    )
