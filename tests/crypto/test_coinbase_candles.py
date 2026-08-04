from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts" / "trading_lab" / "coinbase_candles.py"
SCHEMA_PATH = ROOT / "schemas" / "trading_lab" / "market-bar-v1.schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "crypto" / "coinbase_btc_usd_1h.json"
ETH_FIXTURE_PATH = (
    ROOT / "tests" / "fixtures" / "crypto" / "coinbase_eth_usd_1d.json"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("coinbase_candles_under_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapt(module, payload: bytes, **overrides):
    values = {
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "available_at": "2026-08-02T11:00:05Z",
        "ingested_at": "2026-08-02T11:00:06Z",
    }
    values.update(overrides)
    return module.adapt_coinbase_candles(payload, **values)


def test_coinbase_fixture_adapts_to_sorted_causal_market_bars() -> None:
    module = _load_adapter()
    payload = FIXTURE_PATH.read_bytes()
    records = _adapt(module, payload)
    validator = jsonschema.Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )

    assert len(records) == 2
    assert [record["bar_open_at"] for record in records] == [
        "2026-08-02T09:00:00+00:00",
        "2026-08-02T10:00:00+00:00",
    ]
    assert records[0]["asset"] == "BTC/USD"
    assert records[0]["venue"] == "coinbase_exchange"
    assert records[0]["provider"] == "coinbase_exchange_rest"
    assert records[0]["bar_close_at"] == "2026-08-02T10:00:00+00:00"
    assert records[0]["open"] == "63900.0000000000"
    assert records[0]["high"] == "64200.0000000000"
    assert records[0]["low"] == "63800.0000000000"
    assert records[0]["close"] == "64100.0000000000"
    assert records[0]["volume"] == "10.250000000000000000"
    assert records[0]["raw_payload_sha256"] == hashlib.sha256(payload).hexdigest()
    for record in records:
        validator.validate(record)


def test_coinbase_page_repackaging_preserves_bar_versions() -> None:
    module = _load_adapter()
    payload = FIXTURE_PATH.read_bytes()
    compact = json.dumps(json.loads(payload), separators=(",", ":")).encode("utf-8")

    first = _adapt(module, payload)
    repackaged = _adapt(module, compact)

    assert [row["bar_id"] for row in first] == [row["bar_id"] for row in repackaged]
    assert [row["bar_version_id"] for row in first] == [
        row["bar_version_id"] for row in repackaged
    ]
    assert [row["raw_payload_sha256"] for row in first] != [
        row["raw_payload_sha256"] for row in repackaged
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"product_id": "SOL-USD"}, "product"),
        ({"timeframe": "5m"}, "timeframe"),
    ],
)
def test_coinbase_adapter_rejects_out_of_scope_identity(overrides, message) -> None:
    module = _load_adapter()

    with pytest.raises(module.CoinbaseCandleError, match=message):
        _adapt(module, FIXTURE_PATH.read_bytes(), **overrides)


def test_coinbase_adapter_rejects_partial_bar_atomically() -> None:
    module = _load_adapter()

    with pytest.raises(module.CoinbaseCandleError, match="invalid"):
        _adapt(
            module,
            FIXTURE_PATH.read_bytes(),
            available_at="2026-08-02T10:30:00Z",
            ingested_at="2026-08-02T10:30:01Z",
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b'{"error":"not a candle page"}',
        b"[[1785664800,1,2,1,2]]",
        b"[[1785664800.5,1,2,1,2,1]]",
        b"[[1785664800,1,2,1,2,NaN]]",
        b"[[1785664800,1,2,1,2,true]]",
        (
            b"[[1785664800,1,2,1,2,1],"
            b"[1785664800,1,2,1,2,1]]"
        ),
    ],
)
def test_coinbase_adapter_rejects_malformed_pages(payload: bytes) -> None:
    module = _load_adapter()

    with pytest.raises(module.CoinbaseCandleError, match="invalid"):
        _adapt(module, payload)


def test_coinbase_adapter_bounds_page_size_and_decimal_tokens() -> None:
    module = _load_adapter()
    too_many = json.dumps([[index, 1, 2, 1, 2, 1] for index in range(301)]).encode()
    huge_decimal = b"[[1785664800,1,2,1,2," + (b"1" * 257) + b"]]"

    with pytest.raises(module.CoinbaseCandleError, match="invalid"):
        _adapt(module, too_many)
    with pytest.raises(module.CoinbaseCandleError, match="invalid"):
        _adapt(module, huge_decimal)
    with pytest.raises(module.CoinbaseCandleError, match="too large"):
        _adapt(module, b" " * (module.MAX_RESPONSE_BYTES + 1))


def test_coinbase_adapter_accepts_eth_daily_closed_bar() -> None:
    module = _load_adapter()

    records = _adapt(
        module,
        ETH_FIXTURE_PATH.read_bytes(),
        product_id="ETH-USD",
        timeframe="1d",
        available_at="2026-08-02T00:00:01Z",
        ingested_at="2026-08-02T00:00:02Z",
    )

    assert len(records) == 1
    assert records[0]["asset"] == "ETH/USD"
    assert records[0]["bar_open_at"] == "2026-08-01T00:00:00+00:00"
    assert records[0]["bar_close_at"] == "2026-08-02T00:00:00+00:00"


def test_coinbase_adapter_preserves_gaps_without_synthetic_bars() -> None:
    module = _load_adapter()
    payload = (
        b'[[1785668400,"64200","64400","64250","64300","4"],'
        b'[1785661200,"63800","64200","63900","64100","10.25"]]'
    )

    records = _adapt(
        module,
        payload,
        available_at="2026-08-02T12:00:01Z",
        ingested_at="2026-08-02T12:00:02Z",
    )

    assert [record["bar_open_at"] for record in records] == [
        "2026-08-02T09:00:00+00:00",
        "2026-08-02T11:00:00+00:00",
    ]


def test_coinbase_adapter_has_no_network_or_broker_imports() -> None:
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    forbidden = ("alpaca", "broker", "requests", "urllib", "socket", "src.hyprl.crypto")
    assert not any(
        any(part in imported.lower() for part in forbidden)
        for imported in imports
    )
