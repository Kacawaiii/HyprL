"""Deterministic causal MarketBar V1 records for BTC/USD and ETH/USD."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import re


SCHEMA_VERSION = "trading-lab.market-bar.v1"
ASSETS = frozenset({"BTC/USD", "ETH/USD"})
TIMEFRAME_DURATIONS = {
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}
PRICE_DECIMAL_PLACES = 10
VOLUME_DECIMAL_PLACES = 18
MAX_INTEGER_DIGITS = 100
MAX_DECIMAL_INPUT_CHARS = 256
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _timestamp(value: datetime | str, *, field: str) -> datetime:
    try:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} timestamp is invalid") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise ValueError(f"{field} timestamp must be timezone-aware")
    try:
        offset = parsed.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} timestamp is invalid") from exc
    if offset is None:
        raise ValueError(f"{field} timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _integer_digits(value: Decimal) -> int:
    if value.is_zero():
        return 1
    return max(value.copy_abs().adjusted() + 1, 1)


def _decimal(
    value: object,
    *,
    field: str,
    places: int,
    allow_zero: bool,
) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field} must be an exact decimal value")
    try:
        raw_value = str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if len(raw_value) > MAX_DECIMAL_INPUT_CHARS:
        raise ValueError(f"{field} input is too long")
    try:
        parsed = Decimal(raw_value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    integer_digits = _integer_digits(parsed)
    if integer_digits > MAX_INTEGER_DIGITS:
        raise ValueError(f"{field} exceeds the magnitude limit")

    precision = max(
        len(parsed.as_tuple().digits),
        integer_digits + places,
        32,
    ) + 4
    quantum = Decimal(1).scaleb(-places)
    try:
        with localcontext() as context:
            context.prec = precision
            quantized = parsed.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError(f"{field} cannot be quantized") from exc
    if _integer_digits(quantized) > MAX_INTEGER_DIGITS:
        raise ValueError(f"{field} exceeds the magnitude limit")
    if quantized < 0 or (not allow_zero and quantized == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return quantized


def _json_decimal(value: Decimal, *, places: int) -> str:
    if value.is_zero():
        value = abs(value)
    return format(value, f".{places}f")


def _hash_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_market_bar(
    *,
    asset: str,
    venue: str,
    provider: str,
    timeframe: str,
    bar_open_at: datetime | str,
    bar_close_at: datetime | str,
    available_at: datetime | str,
    ingested_at: datetime | str,
    open_price: object,
    high_price: object,
    low_price: object,
    close_price: object,
    volume: object,
    raw_payload_sha256: str,
) -> dict[str, object]:
    """Build one immutable, complete and causally timestamped market bar."""

    if not isinstance(asset, str) or asset not in ASSETS:
        raise ValueError("asset must be BTC/USD or ETH/USD")
    if not isinstance(timeframe, str) or timeframe not in TIMEFRAME_DURATIONS:
        raise ValueError("timeframe must be 1h or 1d")
    if not isinstance(venue, str) or not _NAME_PATTERN.fullmatch(venue):
        raise ValueError("venue is invalid")
    if not isinstance(provider, str) or not _NAME_PATTERN.fullmatch(provider):
        raise ValueError("provider is invalid")
    if not isinstance(raw_payload_sha256, str) or not _SHA256_PATTERN.fullmatch(
        raw_payload_sha256
    ):
        raise ValueError("raw payload hash is invalid")

    opened = _timestamp(bar_open_at, field="bar_open_at")
    closed = _timestamp(bar_close_at, field="bar_close_at")
    available = _timestamp(available_at, field="available_at")
    ingested = _timestamp(ingested_at, field="ingested_at")
    expected_duration = TIMEFRAME_DURATIONS[timeframe]
    if closed - opened != expected_duration:
        raise ValueError("bar interval does not match timeframe")
    if opened.minute != 0 or opened.second != 0 or opened.microsecond != 0:
        raise ValueError("bar_open_at is not aligned")
    if timeframe == "1d" and opened.hour != 0:
        raise ValueError("daily bar_open_at is not aligned")
    if available < closed or ingested < available:
        raise ValueError("market bar timestamps are not causal")

    opened_price = _decimal(
        open_price,
        field="open price",
        places=PRICE_DECIMAL_PLACES,
        allow_zero=False,
    )
    highest_price = _decimal(
        high_price,
        field="high price",
        places=PRICE_DECIMAL_PLACES,
        allow_zero=False,
    )
    lowest_price = _decimal(
        low_price,
        field="low price",
        places=PRICE_DECIMAL_PLACES,
        allow_zero=False,
    )
    closed_price = _decimal(
        close_price,
        field="close price",
        places=PRICE_DECIMAL_PLACES,
        allow_zero=False,
    )
    parsed_volume = _decimal(
        volume,
        field="volume",
        places=VOLUME_DECIMAL_PLACES,
        allow_zero=True,
    )
    if highest_price < max(opened_price, closed_price, lowest_price):
        raise ValueError("OHLC high is inconsistent")
    if lowest_price > min(opened_price, closed_price, highest_price):
        raise ValueError("OHLC low is inconsistent")

    open_text = opened.isoformat()
    close_text = closed.isoformat()
    logical_identity: dict[str, object] = {
        "asset": asset,
        "venue": venue,
        "provider": provider,
        "timeframe": timeframe,
        "bar_open_at": open_text,
        "bar_close_at": close_text,
    }
    logical_hash = _hash_payload(logical_identity)
    bar_id = f"hyprl-market-bar-{logical_hash}"
    open_value = _json_decimal(opened_price, places=PRICE_DECIMAL_PLACES)
    high_value = _json_decimal(highest_price, places=PRICE_DECIMAL_PLACES)
    low_value = _json_decimal(lowest_price, places=PRICE_DECIMAL_PLACES)
    close_value = _json_decimal(closed_price, places=PRICE_DECIMAL_PLACES)
    volume_value = _json_decimal(parsed_volume, places=VOLUME_DECIMAL_PLACES)
    version_hash = _hash_payload(
        {
            "bar_id": bar_id,
            "open": open_value,
            "high": high_value,
            "low": low_value,
            "close": close_value,
            "volume": volume_value,
        }
    )

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "bar_id": bar_id,
        "bar_version_id": f"hyprl-market-bar-version-{version_hash}",
        "asset": asset,
        "venue": venue,
        "provider": provider,
        "timeframe": timeframe,
        "bar_status": "complete",
        "bar_open_at": open_text,
        "bar_close_at": close_text,
        "available_at": available.isoformat(),
        "ingested_at": ingested.isoformat(),
        "open": open_value,
        "high": high_value,
        "low": low_value,
        "close": close_value,
        "volume": volume_value,
        "raw_payload_sha256": raw_payload_sha256,
    }
    payload["content_sha256"] = _hash_payload(payload)
    return payload
