"""Preregistered constants and shared validation; never relax the cutoff."""

from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-08-31T23:59:00Z")
LAST_DAY = CUTOFF.normalize()
FIRST_DAY = pd.Timestamp("2014-12-01T00:00:00Z")  # Coinbase Exchange has no earlier candles
OOS_START = pd.Timestamp("2023-01-01T00:00:00Z")
DAYS_YEAR = 365.25
CORE = ("BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD")
JOKERS = ("AVAX-USD", "XRP-USD")
# Identity-based exclusions, frozen before analysis; no price/volume outcome screening.
STABLE_BASES = frozenset({
    "USDC", "USDT", "DAI", "SAI", "PAX", "USDP", "TUSD", "BUSD", "GUSD",
    "UST", "USTC", "PYUSD", "RLUSD", "USDS", "SUSD", "LUSD", "FRAX", "FDUSD",
    "USDE", "SUSDE", "DUSD", "USDJ", "USDN", "USDD", "USDK", "USDX", "USD1",
    "EURC", "EUROC", "EURT", "EURS", "EURCV", "CUSD", "CEUR", "FEI", "MIM",
    "CRVUSD", "DOLA", "ALUSD", "USDBC", "USDSC", "USDEB", "GHO", "USD0",
})
REGIMES = (
    ("2018-2020 bear (operator label; overlaps bull)", "2018-01-01", "2020-12-31"),
    ("2020-21 bull (operator label)", "2020-01-01", "2021-12-31"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
    ("2023-25 bull (overlaps final bear)", "2023-01-01", "2025-12-31"),
    ("2025-10 to 2026-08 bear", "2025-10-01", "2026-08-31"),
)


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Timezone-aware UTC timestamp required")
    return value.tz_convert("UTC")


def assert_cutoff(index):
    """Reject forbidden observations rather than silently truncating an input."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        raise ValueError("Timezone-aware UTC dates required")
    if len(idx) and (idx > CUTOFF).any():
        raise ValueError("Forbidden observation after 2026-08-31T23:59Z")
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if len(idx) and not (idx == idx.normalize()).all():
        raise ValueError("Daily bars must be labelled at UTC midnight")


def validate_bars(bars):
    assert_cutoff(bars.index)
    expected = ["low", "high", "open", "close", "volume"]
    if any(col not in bars for col in expected):
        raise ValueError("Incomplete OHLCV candle")
    values = bars[expected].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite candle")
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Nonpositive price or negative volume")
    if (bars.high < bars[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle high")
    if (bars.low > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle low")


def outside_repo(path):
    path = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if path == repo or repo in path.parents:
        raise ValueError("Data, cache, grants and outputs must stay outside Git")
    return path
