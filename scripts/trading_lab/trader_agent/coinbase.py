"""Native Coinbase candles for the amended trader universe.

The frozen BTC/ETH MarketBar v1 contract is intentionally not extended. Trader
features need only completed closes; labels use the exact minute candle open.
"""
from datetime import datetime, timedelta, timezone
import math

from .config import TraderError


def candles(payload):
    if not isinstance(payload, list) or len(payload) > 300:
        raise TraderError('CRYPTO_SHAPE_INVALID')
    seen, output = set(), []
    for row in payload:
        if (not isinstance(row, list) or len(row) != 6 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)):
            raise TraderError('CRYPTO_SHAPE_INVALID')
        stamp, low, high, op, close, volume = row
        if (stamp != int(stamp) or stamp in seen or not 0 < low <= min(op, close) <= max(op, close) <= high
                or volume < 0):
            raise TraderError('CRYPTO_SHAPE_INVALID')
        try:
            opening = datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            raise TraderError('CRYPTO_SHAPE_INVALID') from None
        seen.add(stamp)
        output.append({'bar_open_at': opening, 'open': float(op), 'close': float(close)})
    return sorted(output, key=lambda r: r['bar_open_at'])


def daily_closes(payload, before):
    rows = candles(payload)
    if any(r['bar_open_at'].timestamp() % 86400 for r in rows):
        raise TraderError('CRYPTO_SHAPE_INVALID')
    return [{'bar_open_at': r['bar_open_at'], 'close': r['close']} for r in rows
            if r['bar_open_at'] + timedelta(days=1) <= before]
