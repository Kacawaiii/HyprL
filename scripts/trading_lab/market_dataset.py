"""Phase 2C: causal datasets and forward-return labels.

Turns a Phase 2A `MarketSeries` plus Phase 2B indicators into rows a model can
be trained on, under one rule that shapes every design choice here:

    FEATURES at T use only data available at or before T.
    The LABEL at T deliberately looks forward to T+h -- and never touches a
    single feature.

The two live in separate fields of `DatasetRow` precisely so no transformation
can mix them by accident. Everything else follows:

* **Gaps stop a label.** A forward return is only defined when T and T+h sit
  in the same run of strictly adjacent openings. Across a hole the horizon is
  not h bars of market time, so the label is `None` -- never stretched, never
  imputed.
* **The tail has no future.** The last h rows of each segment simply have no
  label. They are still emitted, marked unusable, rather than silently dropped
  or given a shortened horizon.
* **Splits are chronological and purged.** A row labelled with T+h knows
  something about the next h bars, so the last h rows of each block are purged
  before the boundary. Without that, a training label would overlap the
  validation period.
* **Identity is explicit.** `config_hash` names the definition (features,
  parameters, versions, horizon); `dataset_hash` names definition *and* data.
  Same snapshot plus same config always yields the same pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json

from scripts.trading_lab.market_indicators import (
    INDICATOR_PRECISION,
    IndicatorSpec,
    average_true_range,
    contiguous_segments,
    exponential_moving_average,
    relative_strength_index,
    simple_moving_average,
    simple_return,
    true_range,
)
from scripts.trading_lab.market_series import MarketSeries

DATASET_SCHEMA_VERSION = "trading-lab.market-dataset.v1"
LABEL_SCHEMA_VERSION = "trading-lab.market-label.v1"
DEFAULT_LABEL_HORIZON = 4
MAX_LABEL_HORIZON = 1_000

# Indicators are referenced by name, never by callable: a function object has
# no canonical form, and the whole point of a config hash is to be storable.
INDICATOR_REGISTRY = {
    "sma": simple_moving_average,
    "ema": exponential_moving_average,
    "rsi": relative_strength_index,
    "atr": average_true_range,
    "true_range": true_range,
    "simple_return": simple_return,
}
_PARAMETERLESS = frozenset({"true_range", "simple_return"})


class MarketDatasetError(RuntimeError):
    """Raised when a dataset cannot be built safely."""


@dataclass(frozen=True)
class FeatureDefinition:
    """One named column and the indicator that fills it."""

    column: str
    indicator: str
    parameters: tuple[tuple[str, object], ...] = ()

    def canonical(self) -> dict[str, object]:
        return {
            "column": self.column,
            "indicator": self.indicator,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class LabelSpec:
    """Forward return over `horizon` bars: close[T+h] / close[T] - 1."""

    name: str = "forward_return"
    version: str = LABEL_SCHEMA_VERSION
    horizon: int = DEFAULT_LABEL_HORIZON

    def canonical(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "horizon": self.horizon}


@dataclass(frozen=True)
class DatasetConfig:
    features: tuple[FeatureDefinition, ...]
    label: LabelSpec = LabelSpec()

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "features": [feature.canonical() for feature in self.features],
            "label": self.label.canonical(),
        }

    @property
    def config_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class DatasetRow:
    """One observation. `features` and `label` are separate on purpose."""

    bar_open_at: str
    features: tuple[tuple[str, Decimal | None], ...]
    label: Decimal | None
    usable: bool


@dataclass(frozen=True)
class Dataset:
    schema_version: str
    snapshot_id: str
    as_of: str
    provider: str
    product_id: str
    timeframe: str
    entries_content_hash: str
    config: DatasetConfig
    config_hash: str
    indicator_spec_hashes: tuple[tuple[str, str], ...]
    rows: tuple[DatasetRow, ...]
    dataset_hash: str


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological blocks with the overlapping tail of each one purged."""

    train: tuple[DatasetRow, ...]
    validation: tuple[DatasetRow, ...]
    test: tuple[DatasetRow, ...]
    purged: int


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _require_horizon(horizon: object) -> int:
    # bool is an int subclass; True must not read as horizon 1 here.
    if type(horizon) is not int or not 1 <= horizon <= MAX_LABEL_HORIZON:
        raise MarketDatasetError(
            f"horizon must be an int in 1..{MAX_LABEL_HORIZON}, got {horizon!r}"
        )
    return horizon


def _require_config(config: DatasetConfig) -> None:
    if not config.features:
        raise MarketDatasetError("a dataset needs at least one feature")
    columns = [feature.column for feature in config.features]
    if len(set(columns)) != len(columns):
        raise MarketDatasetError(f"duplicate feature columns: {columns!r}")
    for feature in config.features:
        if feature.indicator not in INDICATOR_REGISTRY:
            raise MarketDatasetError(f"unknown indicator {feature.indicator!r}")
        if feature.indicator in _PARAMETERLESS and feature.parameters:
            raise MarketDatasetError(
                f"indicator {feature.indicator!r} takes no parameters, "
                f"got {feature.parameters!r}"
            )
    _require_horizon(config.label.horizon)
    if config.label.name != "forward_return":
        raise MarketDatasetError(f"unsupported label {config.label.name!r}")


def forward_return_labels(series: MarketSeries, *, horizon: int) -> tuple[Decimal | None, ...]:
    """close[T+h] / close[T] - 1, and None wherever that is not defined.

    Undefined means one of two honest situations: T+h falls outside the
    contiguous segment that contains T (a gap sits between them), or it falls
    past the end of the data. Neither is filled in.
    """
    horizon = _require_horizon(horizon)
    labels: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        for start, end in contiguous_segments(series):
            for index in range(start, end):
                future = index + horizon
                if future >= end:  # the horizon would cross a gap or the tail
                    continue
                current_close = series.points[index].close
                if current_close == 0:
                    continue
                labels[index] = series.points[future].close / current_close - Decimal(1)
    return tuple(labels)


def build_dataset(series: MarketSeries, *, config: DatasetConfig) -> Dataset:
    """Assemble causal features and forward labels into aligned rows."""
    _require_config(config)
    columns: list[tuple[str, tuple[Decimal | None, ...]]] = []
    spec_hashes: list[tuple[str, str]] = []
    for feature in config.features:
        function = INDICATOR_REGISTRY[feature.indicator]
        result = function(series, **dict(feature.parameters))
        columns.append((feature.column, result.values))
        spec_hashes.append((feature.column, result.spec_hash))

    labels = forward_return_labels(series, horizon=config.label.horizon)
    rows: list[DatasetRow] = []
    for index, point in enumerate(series.points):
        values = tuple((column, series_values[index]) for column, series_values in columns)
        label = labels[index]
        rows.append(
            DatasetRow(
                bar_open_at=point.bar_open_at,
                features=values,
                label=label,
                usable=label is not None and all(value is not None for _, value in values),
            )
        )

    config_hash = config.config_hash
    dataset_hash = _sha256_canonical(
        {
            "config": config.canonical(),
            "snapshot_id": series.snapshot_id,
            "as_of": series.as_of,
            "entries_content_hash": series.entries_content_hash,
            "indicator_spec_hashes": [list(pair) for pair in spec_hashes],
            "rows": [
                {
                    "bar_open_at": row.bar_open_at,
                    "features": [
                        [column, None if value is None else str(value)]
                        for column, value in row.features
                    ],
                    "label": None if row.label is None else str(row.label),
                }
                for row in rows
            ],
        }
    )
    return Dataset(
        schema_version=DATASET_SCHEMA_VERSION,
        snapshot_id=series.snapshot_id,
        as_of=series.as_of,
        provider=series.provider,
        product_id=series.product_id,
        timeframe=series.timeframe,
        entries_content_hash=series.entries_content_hash,
        config=config,
        config_hash=config_hash,
        indicator_spec_hashes=tuple(spec_hashes),
        rows=tuple(rows),
        dataset_hash=dataset_hash,
    )


def temporal_split(
    dataset: Dataset, *, train: Decimal | str = "0.6", validation: Decimal | str = "0.2"
) -> TemporalSplit:
    """Split chronologically, purging the label overlap at each boundary.

    Rows are never shuffled and never sorted: they are already in grid order,
    and a random split would put a row's own future in the training set. The
    last `horizon` rows of the train and validation blocks are dropped, because
    their labels are computed from bars that belong to the next block.
    """
    train_fraction = Decimal(train)
    validation_fraction = Decimal(validation)
    if not (0 < train_fraction < 1) or not (0 < validation_fraction < 1):
        raise MarketDatasetError("train and validation fractions must sit in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise MarketDatasetError("train + validation must leave room for a test block")

    total = len(dataset.rows)
    horizon = dataset.config.label.horizon
    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    return TemporalSplit(
        train=dataset.rows[: max(train_end - horizon, 0)],
        validation=dataset.rows[train_end : max(validation_end - horizon, train_end)],
        test=dataset.rows[validation_end:],
        purged=horizon,
    )
