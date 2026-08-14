"""A credential-free daily source for US equities, and its limits.

Corpus V1 is blocked on a Massive key. This is the second source, chosen so a
US equity corpus can exist at all without one. It is *not* a replacement for
V1 and does not touch it: V1's spec, hash, timeframe and adjustment policy are
frozen and stay exactly as they are. This is a different corpus with a
different identity, and the differences are not cosmetic.

**What this source can and cannot do**, established by asking it rather than
by assuming:

* Daily bars over the full two-year range: 501 rows for AAPL, and every one of
  those timestamps falls exactly on a session open from the pinned
  ``US_EQUITY_REGULAR`` calendar -- 501 of 501, no off-grid rows, no missing
  sessions, across both daylight-saving transitions. An independent source
  agreeing with the calendar bar-for-bar is the strongest check the Phase 6D
  work has had.
* **30-minute bars over that range: refused, HTTP 422.** The endpoint serves
  intraday only for roughly the last month. So this corpus is daily, and that
  is a fact about the source rather than a preference. A 30-minute corpus over
  two years still requires a paid provider.

**Honest limitations, recorded because a corpus outlives the person who made
it.** This endpoint is undocumented and unofficial. There is no published
contract, no stability guarantee, and no vendor commitment that the shape
below will hold tomorrow -- which is precisely why every response is stored
raw and every field this code reads is checked before use. Nothing here is
suitable for redistribution, and a corpus built from it is a research input,
not a licensed dataset.

**Prices are RAW.** The chart payload carries unadjusted OHLC in `quote` and a
separate adjusted close in `adjclose`. This provider reads the raw quote and
records corporate actions alongside, rather than claiming a provider-adjusted
series. That is the honest description of what arrives, and it composes with
the existing ``apply_splits`` rather than duplicating it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from scripts.trading_lab.equity_market import (
    ADJUSTMENT_RAW, EquityMarketError, require_adjustment_policy)
from scripts.trading_lab.instruments import InstrumentId, normalize_symbol
from scripts.trading_lab.market_providers import (
    FRESHNESS_END_OF_DAY, MarketDataProvider, MarketProviderError,
    ProviderCapabilities, UnsupportedCapabilityError)
from scripts.trading_lab.trading_calendar import US_EQUITY_REGULAR

YAHOO_PROVIDER_SCHEMA_VERSION = "trading-lab.yahoo-chart-provider.v1"

YAHOO_CHART_DAILY_V1 = "yahoo-chart-daily-v1"

YAHOO_BASE_URL = "https://query1.finance.yahoo.com"

# One endpoint. The ticker travels in the path, so the allowlist is a pattern
# with the ticker constrained to what a US listing can actually contain -- a
# crafted "ticker" cannot walk the path or smuggle a query string.
_TICKER = r"[A-Z][A-Z0-9.\-]{0,15}"
ALLOWED_PATH_PATTERNS = (
    re.compile(rf"^/v8/finance/chart/{_TICKER}\Z"),
)
ALLOWED_PATHS = ("/v8/finance/chart/{ticker}",)

FORBIDDEN_PATH_MARKERS = (
    "account", "balance", "order", "position", "wallet", "withdraw",
    "transfer", "trade", "execution", "portfolio", "quote/marketsummary",
)

# Only what this source can actually serve over a multi-year range. 30m is
# deliberately absent: the endpoint answers 422 for it beyond about a month,
# and offering it here would promise a corpus that cannot be captured.
TIMEFRAME_INTERVALS = {"1d": "1d"}

# The fields read out of a chart response. Named so a schema drift is reported
# against a list rather than discovered as a KeyError mid-capture.
QUOTE_FIELDS = ("open", "high", "low", "close", "volume")


class YahooProviderError(MarketProviderError):
    """Raised when a chart response is malformed, unsafe, or impossible."""


class YahooTransport:
    """The seam between this provider and the network."""

    name = "abstract"

    def request(self, path: str, params: dict, headers: dict) -> dict:
        raise NotImplementedError


class NoNetworkYahooTransport(YahooTransport):
    """The default. Refuses, by construction.

    Not a stub returning empty data: that would let a caller believe an
    instrument simply had no bars. It raises, so an accidental network
    dependency surfaces as a failure rather than a quietly empty series.
    """

    name = "no-network"

    def request(self, path: str, params: dict, headers: dict) -> dict:
        raise YahooProviderError(
            f"{YAHOO_CHART_DAILY_V1} has no transport configured, so {path} "
            "cannot be requested. This build makes no outbound calls unless a "
            "transport is explicitly injected.")


class RecordedYahooTransport(YahooTransport):
    """Serves fixtures keyed by path. Records every call."""

    name = "recorded"

    def __init__(self, responses: dict):
        self.responses = dict(responses)
        self.calls: list[dict] = []

    def request(self, path: str, params: dict, headers: dict) -> dict:
        self.calls.append({"path": path, "params": dict(params),
                           "headers": dict(headers)})
        if path not in self.responses:
            raise YahooProviderError(f"no recorded response for {path}")
        payload = self.responses[path]
        return payload(params) if callable(payload) else payload


YAHOO_CHART_CAPABILITIES = ProviderCapabilities(
    historical_bars=True,
    # Daily history only. Claiming a latest closed bar would let a live caller
    # poll an end-of-day source for a current price.
    latest_closed_bar=False,
    realtime_ticks=False,
    order_book=False,
    corporate_actions=True,
    market_calendar=US_EQUITY_REGULAR,
    data_freshness=FRESHNESS_END_OF_DAY,
    # The whole point of this source: no key, and therefore structurally no
    # account behind it either.
    authenticated=False,
    private_account_data=False,
)


def chart_path(symbol: str) -> str:
    return f"/v8/finance/chart/{symbol}"


def require_interval(timeframe: str) -> str:
    """A HyprL timeframe as an interval this source will actually serve."""
    if timeframe not in TIMEFRAME_INTERVALS:
        raise YahooProviderError(
            f"this source serves {sorted(TIMEFRAME_INTERVALS)} over a "
            f"multi-year range, not {timeframe!r}. Intraday beyond about a "
            "month is refused by the endpoint itself (HTTP 422), so offering "
            "it here would promise a corpus that cannot be captured.")
    return TIMEFRAME_INTERVALS[timeframe]


def _number_text(value, *, field: str, index: int) -> str:
    """A JSON number to exact text, or a refusal.

    The payload carries JSON numbers, so Python has already parsed them to
    float. Nothing can restore digits lost before this line; what it can do is
    record exactly which float arrived, so a rebuild from the stored raw bytes
    lands on the identical string rather than re-rounding differently.
    """
    if isinstance(value, bool) or value is None:
        raise YahooProviderError(
            f"row {index}: {field} is {value!r}, not a number")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise YahooProviderError(
        f"row {index}: {field} is {type(value).__name__}, not a number")


@dataclass(frozen=True)
class YahooReferenceTicker:
    """What the source says the instrument is. Read, never assumed."""

    symbol: str
    exchange_name: str
    exchange_timezone: str
    currency: str
    instrument_type: str

    def payload(self) -> dict:
        return {"symbol": self.symbol, "exchange_name": self.exchange_name,
                "exchange_timezone": self.exchange_timezone,
                "currency": self.currency,
                "instrument_type": self.instrument_type}


def parse_chart_meta(payload: dict) -> YahooReferenceTicker:
    """The identity block, checked before any price is read."""
    result = _require_result(payload)
    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise YahooProviderError("the chart response carries no meta block")
    symbol = str(meta.get("symbol", "")).strip().upper()
    if not symbol:
        raise YahooProviderError("the chart meta names no symbol")
    currency = str(meta.get("currency", "")).strip().upper()
    if currency != "USD":
        raise YahooProviderError(
            f"{symbol} is quoted in {currency!r}; this corpus is USD only and "
            "will not convert silently")
    timezone_name = str(meta.get("exchangeTimezoneName", "")).strip()
    if timezone_name != "America/New_York":
        raise YahooProviderError(
            f"{symbol} reports exchange timezone {timezone_name!r}; the "
            "US_EQUITY_REGULAR calendar describes America/New_York and a "
            "different exchange is a different instrument")
    return YahooReferenceTicker(
        symbol=symbol,
        exchange_name=str(meta.get("exchangeName", "")).strip(),
        exchange_timezone=timezone_name,
        currency=currency,
        instrument_type=str(meta.get("instrumentType", "")).strip().upper())


def _require_result(payload: dict) -> dict:
    if not isinstance(payload, dict) or "chart" not in payload:
        raise YahooProviderError(
            "the response has no 'chart' block; the source schema does not "
            "match this capture's contract")
    chart = payload["chart"]
    error = chart.get("error")
    if error:
        raise YahooProviderError(f"the source reports an error: {error}")
    results = chart.get("result")
    if not results:
        raise YahooProviderError("the chart response carries no result")
    result = results[0]
    if not isinstance(result, dict):
        raise YahooProviderError("the chart result is not an object")
    return result


def adapt_chart_rows(payload: dict, *, instrument_id: str) -> list[dict]:
    """Turn a chart response into rows this platform can validate.

    The payload is column-oriented: one timestamp array and one array per OHLC
    field, all meant to be read positionally. Lengths that disagree are a
    corrupt response, not a shape to be zipped over -- zip would silently
    truncate to the shortest and produce a shorter, plausible series.

    Rows with a null price are dropped rather than guessed at, and the drop is
    visible: the gap audit compares what arrived against the calendar, so a
    dropped row appears as a missing expected bar instead of disappearing.
    """
    result = _require_result(payload)
    meta = parse_chart_meta(payload)
    # Both sides canonicalised before they meet. The source reports an
    # exchange *name* ("NMS"), not a MIC, so a full instrument identity cannot
    # be built from the response -- the venue comes from the registry and is
    # asserted there. What is checkable here is that the response is about the
    # symbol that was requested, and this comparison IS that rule.
    wanted = normalize_symbol(instrument_id.split(":")[-1])
    reported = normalize_symbol(meta.symbol)
    if reported != wanted:
        raise YahooProviderError(
            f"asked for {wanted} and the response is about {reported!r}; "
            "refusing to store one instrument under another's name")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators")
    if not isinstance(indicators, dict) or not indicators.get("quote"):
        raise YahooProviderError(
            f"{instrument_id}: the chart response carries no quote series; "
            f"the source schema does not match this capture's contract "
            f"(expected fields {list(QUOTE_FIELDS)})")
    blocks = indicators["quote"]
    if not isinstance(blocks, list) or len(blocks) != 1:
        # The documented shape carries exactly one quote block. Two is an
        # anomaly nobody has characterised, and reading the first would be a
        # silent choice about which series is the real one -- the same class
        # of mistake as keeping the first of two duplicate bars.
        raise YahooProviderError(
            f"{instrument_id}: expected exactly one quote block, found "
            f"{len(blocks) if isinstance(blocks, list) else 'a non-list'}; "
            "refusing to choose between them")
    quote = blocks[0]
    missing = [name for name in QUOTE_FIELDS if name not in quote]
    if missing:
        raise YahooProviderError(
            f"{instrument_id}: the quote series is missing {missing}")

    for name in QUOTE_FIELDS:
        series = quote[name]
        if not isinstance(series, list) or len(series) != len(timestamps):
            raise YahooProviderError(
                f"{instrument_id}: the {name!r} series has "
                f"{len(series) if isinstance(series, list) else 'no'} entries "
                f"for {len(timestamps)} timestamps; a column-oriented response "
                "whose columns disagree is corrupt, and zipping it would "
                "silently produce a shorter series")

    rows = []
    for index, stamp in enumerate(timestamps):
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise YahooProviderError(
                f"{instrument_id}: timestamp {stamp!r} at row {index} is not "
                "epoch seconds")
        values = {name: quote[name][index] for name in QUOTE_FIELDS}
        if any(value is None for value in values.values()):
            # A hole in the source. Left out rather than filled, so the gap
            # audit reports it against the calendar.
            continue
        rows.append({
            "bar_open_at": datetime.fromtimestamp(int(stamp), tz=timezone.utc),
            **{name: _number_text(values[name], field=name, index=index)
               for name in QUOTE_FIELDS},
        })
    return rows


def adapt_split_events(payload: dict, *, instrument_id: str) -> list[dict]:
    """Split events, for provenance. Never used to recompute a price."""
    result = _require_result(payload)
    events = result.get("events") or {}
    splits = events.get("splits") or {}
    if not isinstance(splits, dict):
        raise YahooProviderError(f"{instrument_id}: 'splits' is not an object")
    rows = []
    for entry in splits.values():
        if not isinstance(entry, dict):
            raise YahooProviderError(f"{instrument_id}: a split is not an object")
        for key in ("date", "numerator", "denominator"):
            if key not in entry:
                raise YahooProviderError(
                    f"{instrument_id}: a split event is missing {key!r}")
        effective = datetime.fromtimestamp(int(entry["date"]), tz=timezone.utc)
        rows.append({
            "effective_date": effective.date().isoformat(),
            # numerator/denominator are new-per-old, so a 4-for-1 is 4/1.
            # Reversed, it would invert every adjustment.
            "ratio_numerator": int(entry["numerator"]),
            "ratio_denominator": int(entry["denominator"]),
        })
    return sorted(rows, key=lambda row: row["effective_date"])


class YahooChartDailyProvider(MarketDataProvider):
    """Daily US equity history, no credential, over an injected transport."""

    provider_id = YAHOO_CHART_DAILY_V1
    display_name = "Yahoo chart (US equities, daily, unofficial)"
    capabilities = YAHOO_CHART_CAPABILITIES

    def __init__(self, instruments=(), *, transport: YahooTransport | None = None,
                 base_url: str = YAHOO_BASE_URL,
                 adjustment_policy: str = ADJUSTMENT_RAW):
        self.supported_instruments = tuple(
            InstrumentId.coerce(item).canonical for item in instruments)
        self.transport = transport or NoNetworkYahooTransport()
        self.base_url = base_url.rstrip("/")
        self.adjustment_policy = require_adjustment_policy(adjustment_policy)
        if self.adjustment_policy != ADJUSTMENT_RAW:
            raise EquityMarketError(
                "this source serves unadjusted OHLC with a separate adjusted "
                f"close, so it can only honestly claim {ADJUSTMENT_RAW}; "
                f"{self.adjustment_policy} would relabel the data")

    def _require_allowed(self, path: str) -> str:
        lowered = path.lower()
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in lowered:
                raise YahooProviderError(
                    f"refusing to request {path!r}: this provider is market "
                    f"data only, and {marker!r} is not market data.")
        if not any(pattern.match(path) for pattern in ALLOWED_PATH_PATTERNS):
            raise YahooProviderError(
                f"{path!r} does not match this provider's allowlist "
                f"{list(ALLOWED_PATHS)}")
        return path

    def _headers(self) -> dict:
        """No credential exists for this source, so none is ever sent."""
        return {"Accept": "application/json"}

    def chart_params(self, *, timeframe: str, start, end) -> dict:
        return {
            "interval": require_interval(timeframe),
            "period1": _epoch(start),
            "period2": _epoch(end),
            "events": "div,split",
        }

    def get_historical_bars(self, instrument, timeframe, *, start, end):
        raise UnsupportedCapabilityError(
            f"{self.provider_id} builds bars through the corpus capture "
            "runner, which is calendar-aware; a daily bar closes at the "
            "session close and this provider cannot know that on its own")

    def get_latest_closed_bars(self, instrument, timeframe, *, now=None,
                               since=None, max_bars=None):
        raise UnsupportedCapabilityError(
            f"{self.provider_id} is an {FRESHNESS_END_OF_DAY} historical "
            "source and has no latest closed bar")

    def payload(self) -> dict:
        base = super().payload()
        return {
            **base,
            "schema_version": YAHOO_PROVIDER_SCHEMA_VERSION,
            "base_url": self.base_url,
            "transport": self.transport.name,
            "network_enabled": not isinstance(self.transport,
                                              NoNetworkYahooTransport),
            "adjustment_policy": self.adjustment_policy,
            "allowed_paths": list(ALLOWED_PATHS),
            "brokerage_endpoints": [],
            "credential_required": False,
            "official_contract": False,
            "redistribution_permitted": False,
            "note": ("Undocumented endpoint with no stability guarantee. "
                     "Research input, not a licensed dataset."),
        }


def _epoch(value) -> int:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(moment.timestamp())
    text = str(value).strip()[:10]
    return int(datetime.fromisoformat(text).replace(
        tzinfo=timezone.utc).timestamp())


__all__ = [
    "ALLOWED_PATHS", "ALLOWED_PATH_PATTERNS", "FORBIDDEN_PATH_MARKERS",
    "NoNetworkYahooTransport", "QUOTE_FIELDS", "RecordedYahooTransport",
    "TIMEFRAME_INTERVALS", "YAHOO_BASE_URL", "YAHOO_CHART_CAPABILITIES",
    "YAHOO_CHART_DAILY_V1", "YAHOO_PROVIDER_SCHEMA_VERSION",
    "YahooChartDailyProvider", "YahooProviderError", "YahooReferenceTicker",
    "YahooTransport", "adapt_chart_rows", "adapt_split_events", "chart_path",
    "parse_chart_meta", "require_interval",
]
