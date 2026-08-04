from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest


MODULE_NAME = "scripts.trading_lab.market_bar"
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "trading_lab"
    / "market_bar.py"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "trading_lab"
    / "market-bar-v1.schema.json"
)


def _load_market_bar_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, f"missing module {MODULE_PATH}"
    return importlib.import_module(MODULE_NAME)


def _build(module, **overrides):
    values = {
        "asset": "BTC/USD",
        "venue": "coinbase_exchange",
        "provider": "coinbase_exchange",
        "timeframe": "1h",
        "bar_open_at": datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc),
        "bar_close_at": datetime(2026, 8, 2, 11, 0, tzinfo=timezone.utc),
        "available_at": datetime(2026, 8, 2, 11, 0, 5, tzinfo=timezone.utc),
        "ingested_at": datetime(2026, 8, 2, 11, 0, 7, tzinfo=timezone.utc),
        "open_price": "64123.12345678901",
        "high_price": "64200.00000000004",
        "low_price": "64000.00000000001",
        "close_price": "64150.50000000005",
        "volume": "12.345678901234567891",
        "raw_payload_sha256": "a" * 64,
    }
    values.update(overrides)
    return module.build_market_bar(**values)


def test_market_bar_builds_exact_hashed_schema_valid_record() -> None:
    module = _load_market_bar_module()

    record = _build(module)

    assert record["schema_version"] == "trading-lab.market-bar.v1"
    assert record["asset"] == "BTC/USD"
    assert record["timeframe"] == "1h"
    assert record["bar_status"] == "complete"
    assert record["bar_open_at"] == "2026-08-02T10:00:00+00:00"
    assert record["bar_close_at"] == "2026-08-02T11:00:00+00:00"
    assert record["available_at"] == "2026-08-02T11:00:05+00:00"
    assert record["ingested_at"] == "2026-08-02T11:00:07+00:00"
    assert record["open"] == "64123.1234567890"
    assert record["high"] == "64200.0000000000"
    assert record["low"] == "64000.0000000000"
    assert record["close"] == "64150.5000000000"
    assert record["volume"] == "12.345678901234567891"
    assert record["bar_id"].startswith("hyprl-market-bar-")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    ).validate(record)

    expected_hash = record.pop("content_sha256")
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert expected_hash == hashlib.sha256(canonical).hexdigest()


def test_market_bar_accepts_aligned_eth_daily_bar() -> None:
    module = _load_market_bar_module()

    record = _build(
        module,
        asset="ETH/USD",
        timeframe="1d",
        bar_open_at="2026-08-01T00:00:00Z",
        bar_close_at="2026-08-02T00:00:00Z",
        available_at="2026-08-02T00:01:00Z",
        ingested_at="2026-08-02T00:01:01Z",
    )

    assert record["asset"] == "ETH/USD"
    assert record["timeframe"] == "1d"


@pytest.mark.parametrize("asset", ["SOL/USD", "BTC/USDT", "ETH/BTC"])
def test_market_bar_rejects_assets_outside_btc_eth(asset: str) -> None:
    module = _load_market_bar_module()

    with pytest.raises(ValueError, match="asset"):
        _build(module, asset=asset)


@pytest.mark.parametrize("timeframe", ["5m", "4h", "1w"])
def test_market_bar_rejects_non_v1_timeframes(timeframe: str) -> None:
    module = _load_market_bar_module()

    with pytest.raises(ValueError, match="timeframe"):
        _build(module, timeframe=timeframe)


@pytest.mark.parametrize(
    "overrides",
    [
        {"available_at": "2026-08-02T10:59:59Z"},
        {"ingested_at": "2026-08-02T11:00:04Z"},
        {"bar_close_at": "2026-08-02T10:59:59Z"},
        {
            "bar_open_at": "2026-08-02T10:30:00Z",
            "bar_close_at": "2026-08-02T11:30:00Z",
            "available_at": "2026-08-02T11:30:01Z",
            "ingested_at": "2026-08-02T11:30:02Z",
        },
    ],
)
def test_market_bar_rejects_non_causal_or_misaligned_timestamps(overrides) -> None:
    module = _load_market_bar_module()

    with pytest.raises(ValueError, match="timestamp|interval|aligned"):
        _build(module, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"high_price": "64100"},
        {"low_price": "64140"},
        {"open_price": "0"},
        {"close_price": "NaN"},
        {"volume": "-0.000000000000000001"},
    ],
)
def test_market_bar_rejects_impossible_or_non_finite_ohlcv(overrides) -> None:
    module = _load_market_bar_module()

    with pytest.raises(ValueError, match="OHLC|price|volume|finite"):
        _build(module, **overrides)


def test_market_bar_identity_is_stable_but_content_hash_tracks_ingestion() -> None:
    module = _load_market_bar_module()
    first = _build(module)
    replay = _build(
        module,
        ingested_at="2026-08-02T11:00:08Z",
    )

    assert first["bar_id"] == replay["bar_id"]
    assert first["bar_version_id"] == replay["bar_version_id"]
    assert first["content_sha256"] != replay["content_sha256"]


def test_market_bar_provider_revision_creates_new_append_only_version() -> None:
    module = _load_market_bar_module()
    first = _build(module)
    revised = _build(module, close_price="64151")

    assert first["bar_id"] == revised["bar_id"]
    assert first["bar_version_id"] != revised["bar_version_id"]
    assert first["content_sha256"] != revised["content_sha256"]


def test_market_bar_page_repackaging_is_not_a_provider_revision() -> None:
    module = _load_market_bar_module()
    first = _build(module)
    repackaged = _build(module, raw_payload_sha256="b" * 64)

    assert first["bar_id"] == repackaged["bar_id"]
    assert first["bar_version_id"] == repackaged["bar_version_id"]
    assert first["content_sha256"] != repackaged["content_sha256"]


def test_market_bar_schema_rejects_zero_price_and_unknown_fields() -> None:
    module = _load_market_bar_module()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    record = _build(module)

    record["open"] = "0.0000000000"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)

    record = _build(module)
    record["partial"] = True
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_market_bar_rejects_quantization_carry_beyond_magnitude_bound() -> None:
    module = _load_market_bar_module()
    carrying = ("9" * 100) + ".99999999996"

    with pytest.raises(ValueError, match="magnitude"):
        _build(
            module,
            open_price=carrying,
            high_price=carrying,
            close_price=carrying,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"venue": None},
        {"provider": []},
        {"raw_payload_sha256": None},
    ],
)
def test_market_bar_rejects_invalid_source_metadata_types(overrides) -> None:
    module = _load_market_bar_module()

    with pytest.raises(ValueError, match="venue|provider|hash"):
        _build(module, **overrides)


def test_market_bar_accepts_equivalent_scientific_and_expanded_boundary_values() -> None:
    module = _load_market_bar_module()
    expanded = "1" + ("0" * 99)

    scientific_record = _build(
        module,
        open_price="1e99",
        high_price="1e99",
        close_price="1e99",
        volume="1e99",
    )
    expanded_record = _build(
        module,
        open_price=expanded,
        high_price=expanded,
        close_price=expanded,
        volume=expanded,
    )

    for field in ("open", "high", "close", "volume"):
        assert scientific_record[field] == expanded_record[field]


def test_market_bar_rejects_oversized_decimal_text_before_parsing(
    monkeypatch,
) -> None:
    module = _load_market_bar_module()
    real_decimal = module.Decimal

    def guarded_decimal(value):
        if isinstance(value, str) and len(value) > 256:
            pytest.fail("oversized decimal reached Decimal parsing")
        return real_decimal(value)

    monkeypatch.setattr(module, "Decimal", guarded_decimal)
    with pytest.raises(ValueError, match="input is too long"):
        _build(module, open_price="1" * 10_000)
