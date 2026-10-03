"""HTTP clock evidence: strict IMF-fixdate `Date`, `Age`, the server reference time and the per-response
clock check, plus the canonical timestamp form (market_data_store._canonical_timestamp). Each source
passes its own tolerance and Age bounds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

from scripts.trading_lab.market_data_store import _canonical_timestamp

_IMF = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), ([0-9]{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"([0-9]{4}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) GMT$"
)
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def iso(value: datetime, *, field: str = "timestamp") -> str:
    return _canonical_timestamp(value, field=field)


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _ows(value: str) -> str:
    return value.strip(" \t")


def parse_http_date(value: str) -> datetime | None:
    value = _ows(value)
    if not value.isascii():
        return None
    match = _IMF.match(value)
    if not match:
        return None
    day_name, dd, mon, yyyy, hh, mm, ss = match.groups()
    try:
        instant = datetime(int(yyyy), _MONTHS[mon], int(dd), int(hh), int(mm), int(ss), tzinfo=timezone.utc)
    except ValueError:  # invalid date, hour > 23, minute/second > 59 (leap second 60 is invalid)
        return None
    if _DAYS[instant.weekday()] != day_name:
        return None
    return instant


def server_time(date_lines: list[str], age_lines: list[str], *, age_max: int, age_cap_s: int) -> datetime | None:
    """Date + Age of the final response, or None when the reference is absent or invalid."""
    if len(date_lines) != 1 or len(age_lines) > 1:
        return None
    date = parse_http_date(date_lines[0])
    if date is None:
        return None
    age = 0
    if age_lines:
        raw = _ows(age_lines[0])
        if not raw or not raw.isascii() or not raw.isdigit():
            return None
        age = int(raw)
        if age > age_max or age > age_cap_s:
            return None
    return date + timedelta(seconds=age)


def is_clock_verified(wall_at_receipt: datetime, date_lines: list[str], age_lines: list[str], *, tolerance_s: float,
                      age_max: int, age_cap_s: int) -> bool:
    reference = server_time(date_lines, age_lines, age_max=age_max, age_cap_s=age_cap_s)
    if reference is None:
        return False
    return abs((wall_at_receipt - reference).total_seconds()) <= tolerance_s
