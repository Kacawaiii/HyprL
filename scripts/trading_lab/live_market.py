"""Live ingestion of CLOSED public candles, with the holdout guard at the source.

Only closed candles enter. A candle still forming has a moving close, and a
feature computed from it would change under the model's feet — so the engine
waits for `bar_close_at`, plus a small fixed settle delay that is infrastructure
tolerance, not a trading parameter.

The protected-window guard runs twice and both times *before* anything is kept:
once on the request range, so reserved data is never even asked for, and once
per parsed candle, so a source that returns more than it was asked for cannot
smuggle one in. Nothing protected reaches the store, the features, the model or
the UI.

Public market data only. No key, no header, no account, no private endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import urllib.error

from scripts.trading_lab.capture_market_history import (
    COINBASE_GRANULARITY,
    CORPUS_PROVIDER,
    MAX_ATTEMPTS,
    _http_get,
    _iso,
    _parse_iso,
    canonical_rows_from_payload,
)
from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.protected_holdout import (
    PROTECTED_WINDOW_V1,
    ProtectedHoldoutError,
    ProtectedResearchWindow,
    require_tradeable_now,
    require_unprotected_bar,
    require_unprotected_request,
)

LIVE_MARKET_SCHEMA_VERSION = "trading-lab.live-market.v1"
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
LIVE_USER_AGENT = "hyprl-trading-lab-shadow/1.0"

# Infrastructure tolerance so a venue has finished writing the bar. Not a
# trading parameter, and never used in a price or a decision.
CANDLE_SETTLE_DELAY_SECONDS = 5
MAX_LIVE_CANDLES_PER_POLL = 300


class LiveMarketStatus:
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    EMBARGOED = "EMBARGOED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class LiveMarketError(RuntimeError):
    """Raised when live data cannot be trusted."""


def latest_closed_bar_open(now, *, timeframe: str = "1h",
                           settle_seconds: int = CANDLE_SETTLE_DELAY_SECONDS):
    """The opening of the most recent candle that has certainly closed."""
    duration: timedelta = TIMEFRAME_DURATIONS[timeframe]
    moment = _parse_iso(now, field="now") if isinstance(now, str) else now
    moment = moment.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed = (moment - epoch) % duration
    current_open = moment - elapsed
    # The bar that just closed is only trustworthy after the settle delay.
    if elapsed < timedelta(seconds=settle_seconds):
        return current_open - duration * 2
    return current_open - duration


@dataclass(frozen=True)
class LiveMarketState:
    product: str
    timeframe: str
    status: str
    last_closed_bar: str | None
    last_ingested_bar: str | None
    last_success_at: str | None
    next_expected_bar: str | None
    gap_count: int
    error_code: str | None
    embargo: dict

    def as_payload(self) -> dict:
        return {
            "product": self.product, "timeframe": self.timeframe,
            "status": self.status, "last_closed_bar": self.last_closed_bar,
            "last_ingested_bar": self.last_ingested_bar,
            "last_success_at": self.last_success_at,
            "next_expected_bar": self.next_expected_bar,
            "gap_count": self.gap_count, "error_code": self.error_code,
            "embargo": self.embargo,
        }


def _refuse_protected_payload(product: str, raw: bytes, *,
                              window: ProtectedResearchWindow) -> None:
    """Reject a response containing reserved openings, before it is adapted.

    Checked on the raw timestamps so a protected candle never reaches the
    parser, the store, or a Decimal. A malformed body is left to the adapter,
    which is the component that owns payload validation.
    """
    if not window.protects_product(product):
        return
    try:
        page = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(page, list):
        return
    for row in page:
        if not isinstance(row, list) or not row:
            continue
        try:
            opened = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            continue
        if window.covers(product, opened):
            raise ProtectedHoldoutError(
                f"the venue returned a {product} candle at {_iso(opened)}, inside the "
                f"reserved research holdout {window.start}..{window.end}; the whole "
                "response is discarded")


def _request_url(product: str, *, start: str, end: str, timeframe: str) -> str:
    base = COINBASE_CANDLES_URL.format(product=product)
    return (f"{base}?granularity={COINBASE_GRANULARITY[timeframe]}"
            f"&start={start}&end={end}")


def fetch_closed_candles(product: str, *, start, end, timeframe: str = "1h",
                         fetch=_http_get,
                         window: ProtectedResearchWindow = PROTECTED_WINDOW_V1
                         ) -> tuple[dict, ...]:
    """Fetch a bounded range of closed candles, refusing protected data twice.

    The first refusal happens before the request leaves the process. The second
    happens on every parsed row, because a source is free to return more than it
    was asked for and this one must not be trusted to respect a range.
    """
    first = _parse_iso(start, field="start") if isinstance(start, str) else start
    last = _parse_iso(end, field="end") if isinstance(end, str) else end
    duration = TIMEFRAME_DURATIONS[timeframe]
    if last < first:
        raise LiveMarketError("requested end precedes its start")
    if int((last - first) / duration) + 1 > MAX_LIVE_CANDLES_PER_POLL:
        raise LiveMarketError(
            f"a single poll may request at most {MAX_LIVE_CANDLES_PER_POLL} candles")

    # (1) never even ask for reserved data
    require_unprotected_request(product, start=_iso(first), end=_iso(last),
                                window=window)

    url = _request_url(product, start=_iso(first), end=_iso(last), timeframe=timeframe)
    last_error = None
    raw = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = fetch(url)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as error:
            last_error = error
    if raw is None:
        raise LiveMarketError(f"live fetch failed for {product}: {last_error!r}")

    # (2) scan the raw timestamps BEFORE parsing. A venue is free to return more
    #     than it was asked for, and a protected bar must not even be adapted.
    _refuse_protected_payload(product, raw, window=window)

    marker = _iso(last + duration)
    rows = canonical_rows_from_payload(raw, product=product, timeframe=timeframe,
                                       marker=marker)
    # (3) and again on the parsed openings, in their canonical form
    for row in rows:
        require_unprotected_bar(product, row["bar_open_at"], window=window)
    return rows


def poll_closed_candles(product: str, *, now, since=None, timeframe: str = "1h",
                        fetch=_http_get, max_candles: int = 24,
                        window: ProtectedResearchWindow = PROTECTED_WINDOW_V1
                        ) -> tuple[dict, ...]:
    """Everything closed and admissible since `since`, up to a bounded count."""
    require_tradeable_now(product, now=now, window=window)
    duration = TIMEFRAME_DURATIONS[timeframe]
    newest = latest_closed_bar_open(now, timeframe=timeframe)
    if since is None:
        oldest = newest - duration * (max_candles - 1)
    else:
        previous = _parse_iso(since, field="since") if isinstance(since, str) else since
        oldest = previous + duration
    if oldest > newest:
        return ()
    span = int((newest - oldest) / duration) + 1
    if span > max_candles:
        oldest = newest - duration * (max_candles - 1)
    return fetch_closed_candles(product, start=oldest, end=newest,
                                timeframe=timeframe, fetch=fetch, window=window)
