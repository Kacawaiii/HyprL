"""Phase 2C: causal datasets, forward labels, and the proofs against leakage.

Product contract fixed for this milestone: regression on `forward_return`,
horizon 4 bars, 1h timeframe.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json

import pytest


GRID = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)
T4 = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
HORIZON = 4


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
            range_end=_iso(opens[-1] + timedelta(hours=1)), as_of=_iso(as_of),
        )
        return series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def _build(store_module, snapshots_module, series_module, tmp_path, *, name,
           closes, skip=(), at=T1, as_of=T2):
    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(len(closes))]
    kept = [(o, c) for index, (o, c) in enumerate(zip(opens, closes)) if index not in skip]
    _publish(store, [o for o, _ in kept], [c for _, c in kept], at=at)
    return store, opens, _series_at(store, snapshots_module, series_module, as_of, opens)


def _config(dataset_module, *, horizon=HORIZON):
    return dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("sma3", "sma", (("period", 3),)),
            dataset_module.FeatureDefinition("rsi2", "rsi", (("period", 2),)),
            dataset_module.FeatureDefinition("tr", "true_range"),
        ),
        label=dataset_module.LabelSpec(horizon=horizon),
    )


def _ramp(count): return [str(100 + index * 10) for index in range(count)]


# --- labels ---------------------------------------------------------------


def test_the_forward_return_label_matches_a_hand_computed_value(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="label", closes=_ramp(10))
    labels = dataset_module.forward_return_labels(series, horizon=HORIZON)
    indicators = importlib.import_module("scripts.trading_lab.market_indicators")
    with localcontext() as context:
        # The reference must be computed at the module's own precision -- at the
        # caller's default a repeating decimal would simply be a different number.
        context.prec = indicators.INDICATOR_PRECISION
        assert labels[0] == Decimal("140") / Decimal("100") - 1   # exact: 0.4
        assert labels[5] == Decimal("190") / Decimal("150") - 1   # repeating
    # the last h rows have no future left: None, never a shortened horizon
    assert labels[-HORIZON:] == (None,) * HORIZON
    assert len(labels) == len(series.points)


def test_a_label_never_spans_a_gap(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """T+h must be h bars of MARKET time away, not h rows away."""
    _, opens, series = _build(store_module, snapshots_module, series_module, tmp_path,
                              name="gap", closes=_ramp(12), skip={6})
    assert series.missing_openings == (opens[6].isoformat(),)
    labels = dataset_module.forward_return_labels(series, horizon=HORIZON)
    segments = importlib.import_module(
        "scripts.trading_lab.market_indicators").contiguous_segments(series)
    assert segments == ((0, 6), (6, 11))
    # Inside the first segment only indices 0..1 have a full horizon.
    assert labels[0] is not None and labels[1] is not None
    assert labels[2] is None and labels[3] is None and labels[4] is None and labels[5] is None
    # Second segment: index 6 has 6+4=10 < 11, so it is defined; 7.. are not.
    assert labels[6] is not None
    assert labels[7] is None


# --- dataset --------------------------------------------------------------


def test_a_dataset_carries_provenance_and_separates_features_from_the_label(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ds", closes=_ramp(12))
    dataset = dataset_module.build_dataset(series, config=_config(dataset_module))

    assert dataset.schema_version == "trading-lab.market-dataset.v1"
    assert dataset.snapshot_id == series.snapshot_id
    assert dataset.as_of == series.as_of
    assert dataset.entries_content_hash == series.entries_content_hash
    assert len(dataset.rows) == len(series.points)
    assert [row.bar_open_at for row in dataset.rows] == [p.bar_open_at for p in series.points]
    assert [column for column, _ in dataset.rows[0].features] == ["sma3", "rsi2", "tr"]
    # Every indicator that filled a column is traceable to its definition.
    assert dict(dataset.indicator_spec_hashes).keys() == {"sma3", "rsi2", "tr"}
    assert all(len(value) == 64 for value in dict(dataset.indicator_spec_hashes).values())
    # A row is usable only when every feature AND the label exist.
    for row in dataset.rows:
        assert row.usable == (row.label is not None
                              and all(value is not None for _, value in row.features))
    assert any(row.usable for row in dataset.rows)
    assert all(not row.usable for row in dataset.rows[-HORIZON:])


def test_features_are_causal_even_though_the_label_looks_forward(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """Truncating the series must leave every earlier FEATURE untouched. The
    label may disappear -- its future was cut off -- but never change."""
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="causal", closes=_ramp(12))
    config = _config(dataset_module)
    full = dataset_module.build_dataset(series, config=config).rows
    for cut in range(1, len(series.points) + 1):
        truncated = dataset_module.build_dataset(
            replace(series, points=series.points[:cut]), config=config
        ).rows
        assert [row.features for row in truncated] == [row.features for row in full[:cut]], cut
        for index, row in enumerate(truncated):
            assert row.label in (None, full[index].label), (cut, index)


def test_a_dataset_does_not_change_when_a_later_revision_lands(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    store, opens, series = _build(store_module, snapshots_module, series_module, tmp_path,
                                  name="leak", closes=_ramp(12))
    config = _config(dataset_module)
    before = dataset_module.build_dataset(series, config=config)

    _publish(store, opens, ["999"] * 12, at=T3)  # knowledge that did not exist at T2

    recomputed = dataset_module.build_dataset(
        _series_at(store, snapshots_module, series_module, T2, opens), config=config)
    assert recomputed.dataset_hash == before.dataset_hash
    assert recomputed.rows == before.rows

    later = dataset_module.build_dataset(
        _series_at(store, snapshots_module, series_module, T4, opens), config=config)
    assert later.dataset_hash != before.dataset_hash
    assert later.config_hash == before.config_hash  # the definition did not change


# --- identity -------------------------------------------------------------


def test_config_hash_names_the_definition_and_dataset_hash_names_the_data(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="idA", closes=_ramp(12))
    other = _build(store_module, snapshots_module, series_module, tmp_path,
                   name="idB", closes=[str(500 - index * 7) for index in range(12)])[2]
    config = _config(dataset_module)
    first = dataset_module.build_dataset(series, config=config)

    assert dataset_module.build_dataset(series, config=config).dataset_hash == first.dataset_hash
    same_config_other_data = dataset_module.build_dataset(other, config=config)
    assert same_config_other_data.config_hash == first.config_hash
    assert same_config_other_data.dataset_hash != first.dataset_hash

    for changed in (
        _config(dataset_module, horizon=3),
        replace(config, features=config.features[:2]),
        replace(config, features=(
            dataset_module.FeatureDefinition("sma3", "sma", (("period", 5),)),
        ) + config.features[1:]),
    ):
        assert changed.config_hash != first.config_hash
        assert dataset_module.build_dataset(series, config=changed).dataset_hash \
            != first.dataset_hash
    assert len(first.dataset_hash) == 64 and len(first.config_hash) == 64


# --- temporal split -------------------------------------------------------


def test_the_split_is_chronological_and_purges_the_label_overlap(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="split", closes=_ramp(60))
    dataset = dataset_module.build_dataset(series, config=_config(dataset_module))
    split = dataset_module.temporal_split(dataset)

    assert split.purged == HORIZON
    for block in (split.train, split.validation, split.test):
        assert block, "every block must hold rows"
        opens = [row.bar_open_at for row in block]
        assert opens == sorted(opens), "no shuffling, ever"
    assert split.train[-1].bar_open_at < split.validation[0].bar_open_at
    assert split.validation[-1].bar_open_at < split.test[0].bar_open_at
    # No row appears twice.
    everything = [r.bar_open_at for r in split.train + split.validation + split.test]
    assert len(everything) == len(set(everything))
    # Purging really happened: rows were dropped at each boundary.
    assert len(everything) == len(dataset.rows) - 2 * HORIZON


def test_purging_prevents_a_training_label_from_reading_the_validation_block(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """The decisive check: the last training label's forward window must end
    strictly before the first validation row."""
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="purge", closes=_ramp(60))
    dataset = dataset_module.build_dataset(series, config=_config(dataset_module))
    split = dataset_module.temporal_split(dataset)
    openings = [row.bar_open_at for row in dataset.rows]

    last_train = openings.index(split.train[-1].bar_open_at)
    first_validation = openings.index(split.validation[0].bar_open_at)
    assert last_train + HORIZON < first_validation, (last_train, first_validation)

    last_validation = openings.index(split.validation[-1].bar_open_at)
    first_test = openings.index(split.test[0].bar_open_at)
    assert last_validation + HORIZON < first_test


# --- determinism and validation ------------------------------------------


def test_the_dataset_is_deterministic_across_rebuilds_and_databases(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="det", closes=_ramp(20))
    config = _config(dataset_module)
    hashes = {dataset_module.build_dataset(series, config=config).dataset_hash for _ in range(3)}
    assert len(hashes) == 1
    twin = _build(store_module, snapshots_module, series_module, tmp_path,
                  name="twin", closes=_ramp(20))[2]
    assert dataset_module.build_dataset(twin, config=config).dataset_hash == hashes.pop()


def test_labels_do_not_depend_on_the_callers_decimal_context(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ctx", closes=["100", "103", "97", "111", "108", "125", "119", "140"])
    config = _config(dataset_module)
    reference = dataset_module.build_dataset(series, config=config).dataset_hash
    original = getcontext().prec
    try:
        for precision in (6, 60):
            getcontext().prec = precision
            assert dataset_module.build_dataset(series, config=config).dataset_hash == reference
    finally:
        getcontext().prec = original


@pytest.mark.parametrize("horizon", [0, -1, 1.0, True, "4", None, 1001])
def test_an_invalid_horizon_is_refused(
    tmp_path, store_module, snapshots_module, series_module, dataset_module, horizon
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="badh", closes=_ramp(8))
    with pytest.raises(dataset_module.MarketDatasetError):
        dataset_module.forward_return_labels(series, horizon=horizon)
    with pytest.raises(dataset_module.MarketDatasetError):
        dataset_module.build_dataset(
            series, config=replace(_config(dataset_module),
                                   label=dataset_module.LabelSpec(horizon=horizon)))


def test_an_invalid_config_is_refused(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="badc", closes=_ramp(8))
    base = _config(dataset_module)
    bad = [
        replace(base, features=()),
        replace(base, features=(dataset_module.FeatureDefinition("x", "not_an_indicator"),)),
        replace(base, features=(dataset_module.FeatureDefinition("tr", "true_range",
                                                                 (("period", 3),)),)),
        replace(base, features=base.features + (base.features[0],)),  # duplicate column
        replace(base, label=dataset_module.LabelSpec(name="magic")),
    ]
    for config in bad:
        with pytest.raises(dataset_module.MarketDatasetError):
            dataset_module.build_dataset(series, config=config)


def test_building_a_dataset_never_mutates_the_series(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="pure", closes=_ramp(12))
    before = replace(series)
    dataset_module.build_dataset(series, config=_config(dataset_module))
    assert series == before


# --- one-bar causal return through the dataset layer (Phase 4B) ------------


def test_the_return_indicator_is_resolvable_as_a_dataset_feature(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """The whole point of the extension: `return_1` can be NAMED in a config."""
    assert "simple_return" in dataset_module.INDICATOR_REGISTRY
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ds-ret", closes=_ramp(12))
    config = dataset_module.DatasetConfig(
        features=(dataset_module.FeatureDefinition("return_1", "simple_return"),),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    dataset = dataset_module.build_dataset(series, config=config)
    assert [column for column, _ in dataset.rows[0].features] == ["return_1"]
    assert dataset.rows[0].features[0][1] is None          # no predecessor
    assert dataset.rows[0].usable is False
    indicators = importlib.import_module("scripts.trading_lab.market_indicators")
    expected = indicators.simple_return(series).values
    assert [row.features[0][1] for row in dataset.rows] == list(expected)
    assert dict(dataset.indicator_spec_hashes)["return_1"] == \
        indicators.simple_return(series).spec_hash


def test_the_return_feature_takes_no_parameters(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ds-ret-param", closes=_ramp(8))
    config = dataset_module.DatasetConfig(
        features=(dataset_module.FeatureDefinition("return_1", "simple_return",
                                                   (("period", 1),)),),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    with pytest.raises(dataset_module.MarketDatasetError):
        dataset_module.build_dataset(series, config=config)


def test_a_correction_published_later_cannot_change_an_earlier_return(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """Anti-future-leakage for the new primitive, through the real snapshot path.

    A read at as_of=T must not see a value the venue only published after T,
    even when that value overwrites a bar T already covered.
    """
    store = store_module.MarketDataStore(tmp_path / "ret-asof.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(6)]
    _publish(store, opens, ["100", "110", "121", "133", "146", "160"], at=T1)
    before = _series_at(store, snapshots_module, series_module, T2, opens)

    indicators = importlib.import_module("scripts.trading_lab.market_indicators")
    early = indicators.simple_return(before).values

    # the venue restates one bar, and only publishes the restatement afterwards
    _publish(store, [opens[3]], ["999"], at=T3)
    unchanged = _series_at(store, snapshots_module, series_module, T2, opens)
    later = _series_at(store, snapshots_module, series_module, T4, opens)

    assert indicators.simple_return(unchanged).values == early
    assert indicators.simple_return(later).values != early     # the restatement is real
    assert later.points[3].close == Decimal("999")
    assert unchanged.points[3].close == Decimal("133")


# --- relative primitives through the dataset layer (Phase 4D) -------------


V2_AT = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
V2_ASOF = datetime(2026, 8, 14, 13, 0, tzinfo=timezone.utc)


def test_the_relative_primitives_are_resolvable_as_dataset_features(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """The V2 feature set must be nameable in a config, exactly like V1's."""
    for name in ("return_over_period", "ema_spread", "atr_percent"):
        assert name in dataset_module.INDICATOR_REGISTRY, name
    closes = [str(100 + (index * 11) % 37) for index in range(80)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ds-v2", closes=closes, at=V2_AT, as_of=V2_ASOF)
    config = dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("return_1", "simple_return"),
            dataset_module.FeatureDefinition("return_4", "return_over_period",
                                             (("period", 4),)),
            dataset_module.FeatureDefinition("return_12", "return_over_period",
                                             (("period", 12),)),
            dataset_module.FeatureDefinition("ema_spread_12_26", "ema_spread",
                                             (("fast_period", 12), ("slow_period", 26))),
            dataset_module.FeatureDefinition("rsi_14", "rsi", (("period", 14),)),
            dataset_module.FeatureDefinition("atr_pct_14", "atr_percent",
                                             (("period", 14),)),
        ),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    dataset = dataset_module.build_dataset(series, config=config)
    assert [column for column, _ in dataset.rows[0].features] == [
        "return_1", "return_4", "return_12", "ema_spread_12_26", "rsi_14", "atr_pct_14"]
    indicators = importlib.import_module("scripts.trading_lab.market_indicators")
    expected = {
        "return_4": indicators.return_over_period(series, period=4),
        "ema_spread_12_26": indicators.ema_spread(series, fast_period=12, slow_period=26),
        "atr_pct_14": indicators.atr_percent(series, period=14),
    }
    for column, result in expected.items():
        position = [c for c, _ in dataset.rows[0].features].index(column)
        assert [row.features[position][1] for row in dataset.rows] == list(result.values)
        assert dict(dataset.indicator_spec_hashes)[column] == result.spec_hash
    assert any(row.usable for row in dataset.rows)


def test_the_two_return_primitives_keep_separate_identities_in_a_dataset(
    tmp_path, store_module, snapshots_module, series_module, dataset_module
) -> None:
    """V1's `simple_return` identity must survive the arrival of V2's family."""
    closes = [str(100 + (index * 7) % 19) for index in range(40)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ds-two-returns", closes=closes, at=V2_AT, as_of=V2_ASOF)
    config = dataset_module.DatasetConfig(
        features=(
            dataset_module.FeatureDefinition("return_1", "simple_return"),
            dataset_module.FeatureDefinition("also_1", "return_over_period",
                                             (("period", 1),)),
        ),
        label=dataset_module.LabelSpec(horizon=HORIZON),
    )
    dataset = dataset_module.build_dataset(series, config=config)
    hashes = dict(dataset.indicator_spec_hashes)
    assert hashes["return_1"] != hashes["also_1"]          # different definitions
    assert [row.features[0][1] for row in dataset.rows] == \
           [row.features[1][1] for row in dataset.rows]    # ... same numbers
