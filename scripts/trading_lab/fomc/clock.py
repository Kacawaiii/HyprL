"""HTTP clock evidence (timestamps.http_clock_header_parser, collector_clock_check) and the
canonical timestamp binding (timestamps.canonical_serialization -> market_data_store._canonical_timestamp),
bound to the FOMC spec's tolerance and Age bounds (the parsers live in sources.httpclock).
"""

from __future__ import annotations

from datetime import datetime

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.sources import httpclock
from scripts.trading_lab.sources.httpclock import parse_http_date, parse_iso  # noqa: F401 - FOMC names


def iso(value: datetime) -> str:
    return httpclock.iso(value, field="fomc timestamp")


def server_time(date_lines: list[str], age_lines: list[str]) -> datetime | None:
    """Date + Age of the final 200 response, or None when the reference is absent or invalid."""
    return httpclock.server_time(date_lines, age_lines, age_max=spec.AGE_MAX, age_cap_s=spec.AGE_CAP_S)


def is_clock_verified(wall_at_receipt: datetime, date_lines: list[str], age_lines: list[str]) -> bool:
    return httpclock.is_clock_verified(wall_at_receipt, date_lines, age_lines, tolerance_s=spec.CLOCK_CHECK_TOLERANCE_S,
                                       age_max=spec.AGE_MAX, age_cap_s=spec.AGE_CAP_S)
