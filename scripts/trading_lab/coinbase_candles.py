"""Offline adapter from raw Coinbase Exchange candle pages to MarketBar V1."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json

from scripts.trading_lab.market_bar import build_market_bar


MAX_RESPONSE_BYTES = 1_000_000
MAX_CANDLES_PER_RESPONSE = 300
MAX_DECIMAL_TOKEN_CHARS = 256
PRODUCT_ASSETS = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}
TIMEFRAME_DURATIONS = {
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


class CoinbaseCandleError(ValueError):
    """Raised when an offline Coinbase candle page violates the adapter contract."""


def _parse_decimal_token(token: str) -> Decimal:
    if len(token) > MAX_DECIMAL_TOKEN_CHARS:
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise CoinbaseCandleError("Coinbase candle payload is invalid") from exc
    if not value.is_finite():
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    return value


def _reject_constant(_token: str) -> None:
    raise CoinbaseCandleError("Coinbase candle payload is invalid")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CoinbaseCandleError("Coinbase candle payload is invalid")
        result[key] = value
    return result


def _decode_page(raw_payload: bytes) -> list[object]:
    if not isinstance(raw_payload, bytes):
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    if len(raw_payload) > MAX_RESPONSE_BYTES:
        raise CoinbaseCandleError("Coinbase candle payload is too large")
    try:
        text = raw_payload.decode("utf-8")
        payload = json.loads(
            text,
            parse_float=_parse_decimal_token,
            parse_int=_parse_decimal_token,
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    except CoinbaseCandleError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CoinbaseCandleError("Coinbase candle payload is invalid") from exc
    if not isinstance(payload, list) or len(payload) > MAX_CANDLES_PER_RESPONSE:
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    return payload


def _bar_open(value: object) -> datetime:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    integral = value.to_integral_value()
    if value != integral:
        raise CoinbaseCandleError("Coinbase candle payload is invalid")
    try:
        return datetime.fromtimestamp(int(integral), tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise CoinbaseCandleError("Coinbase candle payload is invalid") from exc


def adapt_coinbase_candles(
    raw_payload: bytes,
    *,
    product_id: str,
    timeframe: str,
    available_at: datetime | str,
    ingested_at: datetime | str,
) -> list[dict[str, object]]:
    """Validate one captured response atomically and return sorted MarketBar records."""

    if not isinstance(product_id, str) or product_id not in PRODUCT_ASSETS:
        raise CoinbaseCandleError("Coinbase product is unsupported")
    if not isinstance(timeframe, str) or timeframe not in TIMEFRAME_DURATIONS:
        raise CoinbaseCandleError("Coinbase timeframe is unsupported")

    page = _decode_page(raw_payload)
    duration = TIMEFRAME_DURATIONS[timeframe]
    raw_payload_sha256 = hashlib.sha256(raw_payload).hexdigest()
    parsed_rows: list[tuple[datetime, list[object]]] = []
    seen_timestamps: set[datetime] = set()

    for row in page:
        if not isinstance(row, list) or len(row) != 6:
            raise CoinbaseCandleError("Coinbase candle payload is invalid")
        opened = _bar_open(row[0])
        if opened in seen_timestamps:
            raise CoinbaseCandleError("Coinbase candle payload is invalid")
        seen_timestamps.add(opened)
        parsed_rows.append((opened, row))

    records: list[dict[str, object]] = []
    try:
        for opened, row in sorted(parsed_rows, key=lambda item: item[0]):
            records.append(
                build_market_bar(
                    asset=PRODUCT_ASSETS[product_id],
                    venue="coinbase_exchange",
                    provider="coinbase_exchange_rest",
                    timeframe=timeframe,
                    bar_open_at=opened,
                    bar_close_at=opened + duration,
                    available_at=available_at,
                    ingested_at=ingested_at,
                    low_price=row[1],
                    high_price=row[2],
                    open_price=row[3],
                    close_price=row[4],
                    volume=row[5],
                    raw_payload_sha256=raw_payload_sha256,
                )
            )
    except (TypeError, ValueError) as exc:
        raise CoinbaseCandleError("Coinbase candle payload is invalid") from exc
    return records
