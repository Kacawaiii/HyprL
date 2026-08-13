"""Massive as a source of US equity history, and nothing else.

Three boundaries define this provider, and each exists because of a specific
way the code could otherwise go wrong.

**The transport is injected.** Nothing here opens a socket by itself. A
provider built without a transport has none, and asking it for bars raises
rather than reaching the internet. That is what makes Phase 6D's tests
genuinely offline: not a mock that intercepts a request, but a provider that
structurally cannot make one. It also means the parsing, the validation, the
grid checks and the corporate-action logic are all exercised against fixtures,
deterministically, with no key and no network.

**The credential never becomes a string here.** It lives in a ``Secret``, is
read at request time, goes into a header, and is dropped. It is not stored on
the instance, not in ``payload()``, not in an exception message, not in a
repr. Every one of those is a real place API keys have leaked from.

**This is market data.** No account, no balance, no order, no position, no
brokerage endpoint -- not unimplemented, absent. The URL allowlist below is the
mechanical version of that promise: a path that is not a market-data path
cannot be requested even by a caller who wants to.

Freshness is END_OF_DAY and is declared as such. This provider is a source of
history for research; it is never a live feed, and nothing downstream may treat
its most recent row as a current price.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from scripts.trading_lab.credentials import (
    CredentialError, CredentialProvider, MASSIVE_API_KEY_ENV,
    MissingCredentialProvider, massive_credentials)
from scripts.trading_lab.equity_market import (
    ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED, CashDividend, EquityMarketError,
    StockSplit, build_equity_bar, require_adjustment_policy,
    require_single_policy)
from scripts.trading_lab.instruments import InstrumentId, Timeframe
from scripts.trading_lab.market_providers import (
    FRESHNESS_END_OF_DAY, MarketDataProvider, MarketProviderError,
    ProviderCapabilities, UnsupportedCapabilityError)
from scripts.trading_lab.trading_calendar import US_EQUITY_REGULAR

MASSIVE_PROVIDER_SCHEMA_VERSION = "trading-lab.massive-provider.v1"

MASSIVE_STOCKS_HISTORICAL_V1 = "massive-stocks-historical-v1"

MASSIVE_BASE_URL = "https://api.massive.com"

# The documented endpoint shapes, as patterns rather than literals: the
# ticker travels in the path, so a fixed set cannot express them. Still an
# allowlist -- a path that does not match one of these three cannot be
# requested, and a denylist would have to anticipate every account endpoint
# the vendor might ever add and only has to miss one.
#
# A ticker is constrained to the characters a US listing can actually use, so
# a crafted "ticker" cannot walk the path or smuggle a query string.
_TICKER = r"[A-Z][A-Z0-9.\-]{0,15}"
_DATE = r"\d{4}-\d{2}-\d{2}"

ALLOWED_PATH_PATTERNS = (
    # Aggregate bars: /v2/aggs/ticker/{ticker}/range/30/minute/{from}/{to}
    re.compile(rf"^/v2/aggs/ticker/{_TICKER}/range/\d{{1,3}}/"
               rf"(minute|hour|day)/{_DATE}/{_DATE}\Z"),
    # Ticker reference details: /v3/reference/tickers/{ticker}
    re.compile(rf"^/v3/reference/tickers/{_TICKER}\Z"),
    # Corporate actions: /stocks/v1/splits
    re.compile(r"^/stocks/v1/splits\Z"),
)

# Kept for the report and for tests: what the allowlist admits, in words.
ALLOWED_PATHS = (
    "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}",
    "/v3/reference/tickers/{ticker}",
    "/stocks/v1/splits",
)


def bars_path(symbol: str, *, multiplier: int, timespan: str,
              start: str, end: str) -> str:
    """The aggregates endpoint, with the ticker in the path where it belongs."""
    return (f"/v2/aggs/ticker/{symbol}/range/{int(multiplier)}/{timespan}"
            f"/{start}/{end}")


def reference_path(symbol: str) -> str:
    return f"/v3/reference/tickers/{symbol}"


SPLITS_ENDPOINT = "/stocks/v1/splits"

# How a HyprL timeframe becomes a vendor (multiplier, timespan) pair. An
# explicit table rather than string surgery: "30m" must mean 30 minutes to
# both sides, and a timeframe with no documented mapping is refused rather
# than approximated.
TIMEFRAME_AGGREGATES = {
    "30m": (30, "minute"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
}

# What ``adjusted=true`` buys, stated once. The vendor's flag is a boolean and
# means split-adjusted; it does NOT apply dividends. That is exactly HyprL's
# SPLIT_ADJUSTED, and exactly not TOTAL_RETURN -- so RAW must send false and
# anything else must be refused rather than mapped to a guess.
ADJUSTED_FLAG = {
    ADJUSTMENT_SPLIT_ADJUSTED: "true",
    ADJUSTMENT_RAW: "false",
}


# Paths that are refused loudly rather than merely absent from the allowlist,
# so the error says *why* instead of "unknown path".
FORBIDDEN_PATH_MARKERS = (
    "account", "balance", "order", "position", "wallet", "withdraw",
    "transfer", "trade", "execution", "portfolio",
)


class MassiveProviderError(MarketProviderError):
    """Raised when a Massive request is malformed, unsafe, or impossible."""


class OfflineTransportError(MassiveProviderError):
    """Raised when a provider with no transport is asked to fetch."""


# --- transport -------------------------------------------------------------


class MassiveTransport:
    """The seam between this provider and the network.

    A transport receives a path, query parameters and headers, and returns a
    decoded payload. It is the *only* thing in this module that could ever
    touch a socket, and the default implementation cannot.
    """

    name = "abstract"

    def request(self, path: str, params: dict, headers: dict) -> dict:
        raise NotImplementedError


class NoNetworkTransport(MassiveTransport):
    """The default. Refuses, by construction.

    Not a stub returning empty data -- that would let a caller believe an
    instrument simply had no bars in the requested window. It raises, so an
    accidental network dependency surfaces immediately as a failure rather
    than as a quietly empty series.
    """

    name = "no-network"

    def request(self, path: str, params: dict, headers: dict) -> dict:
        raise OfflineTransportError(
            f"{MASSIVE_STOCKS_HISTORICAL_V1} has no transport configured, so "
            f"{path} cannot be requested. This build makes no outbound market "
            "data calls unless a transport is explicitly injected.")


class RecordedTransport(MassiveTransport):
    """Serves fixtures from a dict keyed by path.

    The tests' transport. Records every request it receives -- including the
    headers -- so a test can assert both what was asked for and, critically,
    what was sent: the credential leak checks read this log.
    """

    name = "recorded"

    def __init__(self, responses: dict):
        self.responses = dict(responses)
        self.calls: list[dict] = []

    def request(self, path: str, params: dict, headers: dict) -> dict:
        self.calls.append({"path": path, "params": dict(params),
                           "headers": dict(headers)})
        if path not in self.responses:
            raise MassiveProviderError(f"no recorded response for {path}")
        payload = self.responses[path]
        return payload(params) if callable(payload) else payload


# --- capabilities ----------------------------------------------------------

MASSIVE_STOCKS_CAPABILITIES = ProviderCapabilities(
    historical_bars=True,
    # Deliberately false. An end-of-day source has no "latest closed bar" in
    # the sense the live engine means it, and claiming otherwise would let the
    # shadow runtime poll this provider for a current price.
    latest_closed_bar=False,
    realtime_ticks=False,
    order_book=False,
    corporate_actions=True,
    market_calendar=US_EQUITY_REGULAR,
    data_freshness=FRESHNESS_END_OF_DAY,
    # A market-data key. Authenticated, yes -- but authentication here buys
    # access to prices and to nothing that belongs to an account.
    authenticated=True,
    private_account_data=False,
)


@dataclass(frozen=True)
class MassiveReferenceTicker:
    """Reference metadata: what the vendor says a ticker actually is.

    Exists so that venue and asset class are *read* rather than assumed. The
    seed universe is registered from this and only from this; guessing that
    AAPL is NASDAQ because it usually is would be right today and unverified
    forever.
    """

    symbol: str
    venue: str
    asset_class: str
    display_name: str
    currency: str = "USD"
    active: bool = True

    def payload(self) -> dict:
        return {"symbol": self.symbol, "venue": self.venue,
                "asset_class": self.asset_class, "currency": self.currency,
                "display_name": self.display_name, "active": self.active}


# Vendor exchange codes to canonical venues. Anything not listed is refused,
# because a venue this platform cannot name is a venue it cannot canonicalise,
# and a wrong venue means a wrong instrument id.
VENUE_BY_EXCHANGE_CODE = {
    "XNAS": "xnas",
    "NASDAQ": "xnas",
    "XNYS": "xnys",
    "NYSE": "xnys",
    "ARCX": "arcx",
    "NYSEARCA": "arcx",
    "BATS": "bats",
}

ASSET_CLASS_BY_VENDOR_TYPE = {
    "CS": "EQUITY",
    "COMMON_STOCK": "EQUITY",
    "ADRC": "EQUITY",
    "ETF": "ETF",
    "ETP": "ETF",
}


class MassiveStocksHistoricalProvider(MarketDataProvider):
    """US equity and ETF history from Massive, over an injected transport."""

    provider_id = MASSIVE_STOCKS_HISTORICAL_V1
    display_name = "Massive (US stocks, historical)"
    capabilities = MASSIVE_STOCKS_CAPABILITIES

    def __init__(self, instruments=(), *, transport: MassiveTransport | None = None,
                 credentials: CredentialProvider | None = None,
                 base_url: str = MASSIVE_BASE_URL,
                 adjustment_policy: str = ADJUSTMENT_RAW):
        self.supported_instruments = tuple(
            InstrumentId.coerce(item).canonical for item in instruments)
        # No transport means no network. The default is the refusing one, not
        # a real HTTP client that happens to be unused.
        self.transport = transport or NoNetworkTransport()
        # No credential provider means none is configured -- distinct from one
        # that is configured and empty.
        self.credentials = credentials or MissingCredentialProvider(
            MASSIVE_API_KEY_ENV)
        self.base_url = base_url.rstrip("/")
        self.adjustment_policy = require_adjustment_policy(adjustment_policy)

    # --- request plumbing -------------------------------------------------

    def _require_allowed(self, path: str) -> str:
        lowered = path.lower()
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in lowered:
                raise MassiveProviderError(
                    f"refusing to request {path!r}: this provider is market "
                    f"data only, and {marker!r} is not market data. No account "
                    "or order endpoint exists on this interface.")
        if not any(pattern.match(path) for pattern in ALLOWED_PATH_PATTERNS):
            raise MassiveProviderError(
                f"{path!r} does not match any endpoint on this provider's "
                f"allowlist {list(ALLOWED_PATHS)}")
        return path

    def _headers(self) -> dict:
        """Build auth headers at call time. The value never outlives the call.

        Returned rather than stored, and the caller passes it straight to the
        transport. Nothing in this class holds it afterwards.
        """
        secret = self.credentials.get()
        return {"Authorization": f"Bearer {secret.reveal()}",
                "Accept": "application/json"}

    def _fetch(self, path: str, params: dict) -> dict:
        safe_path = self._require_allowed(path)
        if isinstance(self.transport, NoNetworkTransport):
            # Refused before the credential is even read. Nothing is going to
            # be sent, so there is no reason for the secret to be in memory,
            # and no reason to report a missing key when the real obstacle is
            # that this build makes no requests at all.
            return self.transport.request(safe_path, dict(params), {})
        try:
            return self.transport.request(safe_path, dict(params), self._headers())
        except CredentialError:
            raise
        except MassiveProviderError:
            raise
        except Exception as error:
            # The vendor's exception may quote the request it failed on, which
            # can include a header. Re-raised with our own message and without
            # chaining the original text into the payload.
            raise MassiveProviderError(
                f"Massive request to {safe_path} failed: "
                f"{type(error).__name__}") from None

    # --- historical bars --------------------------------------------------

    def get_historical_bars(self, instrument, timeframe, *, start, end,
                            adjustment_policy: str | None = None):
        """Bars over a closed range, on the session grid, one policy throughout.

        The grid check is the point. A vendor row landing at 09:47 on a session
        that opens at 09:30 is not a 30-minute bar however well-formed it looks,
        and a row on a holiday is not a bar at all.
        """
        identity = self.require_supported(instrument)
        frame = Timeframe.parse(timeframe)
        policy = require_adjustment_policy(
            adjustment_policy or self.adjustment_policy)
        if frame.duration >= timedelta(days=1):
            # A daily equity bar ends at the session close, not 24 hours after
            # it opened, and this provider deliberately cannot load a calendar
            # -- that would drag pandas into every crypto process. The corpus
            # capture runner is the calendar-aware path; this convenience
            # method refuses rather than inventing a close.
            raise UnsupportedCapabilityError(
                f"{self.provider_id} cannot close a {frame.label} bar without "
                "a trading calendar; use the corpus capture runner, which is "
                "session-aware")
        multiplier, timespan = require_aggregate_window(frame.label)
        payload = self._fetch(
            bars_path(identity.symbol, multiplier=multiplier,
                      timespan=timespan, start=_day(start), end=_day(end)),
            {"adjusted": ADJUSTED_FLAG[policy], "sort": "asc",
             "limit": MAX_ROWS_PER_PAGE})
        rows = adapt_aggregate_rows(payload, instrument_id=identity.canonical,
                                    requested_policy=policy)
        bars = tuple(
            build_equity_bar(
                instrument=identity, timeframe=frame.label,
                provider_id=self.provider_id, adjustment_policy=policy,
                bar_open_at=row["bar_open_at"],
                # Intraday only, and every supported intraday grid divides a
                # regular and an early-close session exactly, so a bar can
                # never overrun its close here. The capture runner still does
                # the session-membership check that proves it.
                bar_close_at=row["bar_open_at"] + frame.duration,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"])
            for row in rows)
        if bars:
            require_single_policy(bars)
        return bars

    def get_latest_closed_bars(self, instrument, timeframe, *, now=None,
                               since=None, max_bars=None):
        """Refused. END_OF_DAY data cannot answer "what just closed"."""
        raise UnsupportedCapabilityError(
            f"{self.provider_id} is an {FRESHNESS_END_OF_DAY} historical "
            "source. It has no latest closed bar, and returning its most "
            "recent row here would hand a live caller a stale price.")

    # --- corporate actions ------------------------------------------------

    def get_splits(self, instrument, *, start, end) -> tuple[StockSplit, ...]:
        identity = self.require_supported(instrument)
        payload = self._fetch(SPLITS_ENDPOINT, {
            "ticker": identity.symbol,
            "execution_date.gte": _day(start),
            "execution_date.lte": _day(end)})
        return tuple(
            StockSplit(instrument_id=identity,
                       effective_date=split["effective_date"],
                       ratio_numerator=split["ratio_numerator"],
                       ratio_denominator=split["ratio_denominator"],
                       source=self.provider_id)
            for split in adapt_split_rows(payload,
                                          instrument_id=identity.canonical))

    def get_dividends(self, instrument, *, start, end):
        """Refused: no dividends endpoint is documented for this provider.

        Corpus V1 is SPLIT_ADJUSTED and records dividends nowhere, so nothing
        depends on this. A method that built a path the allowlist would reject
        would be worse than one that says plainly there is no such endpoint.
        """
        raise UnsupportedCapabilityError(
            f"{self.provider_id} has no documented dividends endpoint; "
            "SPLIT_ADJUSTED prices do not apply dividends, and Corpus V1 does "
            "not claim total return")

    # --- reference metadata -----------------------------------------------

    def get_reference_ticker(self, symbol: str) -> MassiveReferenceTicker:
        """What the vendor says this ticker is. Venue is read, never guessed."""
        payload = self._fetch(reference_path(str(symbol)), {})
        results = payload.get("results")
        if not results:
            raise MassiveProviderError(f"no reference metadata for {symbol!r}")
        # The single-ticker endpoint returns one object; the search endpoint
        # returns a list. Both spellings are documented, so both are read --
        # this is two known shapes, not a permissive fallback for unknown data.
        row = results[0] if isinstance(results, list) else results
        if not isinstance(row, dict):
            raise MassiveProviderError(
                f"reference metadata for {symbol!r} is not an object")
        return parse_reference_ticker(row)

    def payload(self) -> dict:
        """Everything about this provider that is safe to publish.

        Says whether a credential is configured, because that is operationally
        useful, and nothing else about it -- not the value, not the variable it
        lives in, and no field named after it. A redacted ``api_key`` key would
        be safe today and would be the obvious place for someone to put the
        real one later; the field simply does not exist.
        """
        base = super().payload()
        return {
            **base,
            "schema_version": MASSIVE_PROVIDER_SCHEMA_VERSION,
            "base_url": self.base_url,
            "transport": self.transport.name,
            "network_enabled": not isinstance(self.transport, NoNetworkTransport),
            "adjustment_policy": self.adjustment_policy,
            "allowed_paths": list(ALLOWED_PATHS),
            "brokerage_endpoints": [],
            "credential_required": True,
            "credential_configured": self.credentials.available(),
        }


MAX_ROWS_PER_PAGE = 50000

# The aggregate row, as the vendor sends it. Terse keys, and a timestamp in
# milliseconds since the epoch rather than an ISO string.
AGGREGATE_FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close",
                    "v": "volume", "t": "bar_open_at"}


def require_aggregate_window(timeframe: str) -> tuple[int, str]:
    """A HyprL timeframe as a vendor (multiplier, timespan) pair, or a refusal."""
    if timeframe not in TIMEFRAME_AGGREGATES:
        raise MassiveProviderError(
            f"no documented aggregate window for timeframe {timeframe!r}; "
            f"mapped: {sorted(TIMEFRAME_AGGREGATES)}")
    return TIMEFRAME_AGGREGATES[timeframe]


def _decimal_text(value, *, field: str, row_index: int):
    """A vendor number to exact text, refusing anything that already lost digits.

    The vendor sends JSON numbers, which Python has already parsed to float by
    the time this sees them. A float carries the loss; ``repr`` at least
    records exactly which float arrived, so the canonical value is reproducible
    from the raw bytes rather than being re-rounded differently on each rebuild.
    """
    if isinstance(value, bool) or value is None:
        raise MassiveProviderError(
            f"aggregate row {row_index}: {field} is {value!r}, not a number")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value.strip()
    raise MassiveProviderError(
        f"aggregate row {row_index}: {field} is {type(value).__name__}, "
        "not a number")


def adapt_aggregate_rows(payload: dict, *, instrument_id: str,
                         requested_policy: str) -> list[dict]:
    """Turn an aggregates response into rows this platform can validate.

    Two checks come before any price is read.

    The vendor echoes ``adjusted`` as a boolean. It must agree with what was
    asked for: a response carrying raw prices while the request said adjusted
    is not a response to relabel, because the numbers are different numbers.

    The vendor also echoes ``ticker``. It must be the one requested -- a
    response about another instrument would otherwise be filed under this
    instrument's name.
    """
    status = payload.get("status")
    if status is not None and str(status).upper() not in ("OK", "DELAYED"):
        raise MassiveProviderError(
            f"{instrument_id}: the provider reports status {status!r}")

    declared = payload.get("adjusted")
    if declared is not None:
        expected = ADJUSTED_FLAG[requested_policy] == "true"
        if bool(declared) is not expected:
            raise EquityMarketError(
                f"{instrument_id}: requested {requested_policy} bars but the "
                f"response declares adjusted={declared!r}; refusing to relabel "
                "data")

    echoed = payload.get("ticker")
    wanted = instrument_id.split(":")[-1]
    if echoed is not None and str(echoed).strip().upper() != wanted:
        raise MassiveProviderError(
            f"asked for {wanted} aggregates and the response is about "
            f"{echoed!r}; refusing to store one instrument under another's name")

    results = payload.get("results")
    if results is None:
        # An empty window legitimately returns no results key at all, but only
        # when the vendor also says so. Anything else is a shape this contract
        # does not recognise, and guessing would produce plausible nonsense.
        if payload.get("resultsCount") in (0, None) and status is not None:
            return []
        raise MassiveProviderError(
            f"{instrument_id}: the aggregates response has no 'results' field; "
            f"the provider schema does not match this capture's contract "
            f"(expected row keys {sorted(AGGREGATE_FIELDS)})")
    if not isinstance(results, list):
        raise MassiveProviderError(f"{instrument_id}: 'results' is not a list")

    rows = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise MassiveProviderError(
                f"{instrument_id}: aggregate row {index} is not an object")
        missing = [key for key in AGGREGATE_FIELDS if key not in row]
        if missing:
            raise MassiveProviderError(
                f"{instrument_id}: aggregate row {index} is missing {missing}; "
                "the provider schema does not match this capture's contract")
        opening = _epoch_ms_to_utc(row["t"], row_index=index)
        rows.append({
            "bar_open_at": opening,
            "open": _decimal_text(row["o"], field="o", row_index=index),
            "high": _decimal_text(row["h"], field="h", row_index=index),
            "low": _decimal_text(row["l"], field="l", row_index=index),
            "close": _decimal_text(row["c"], field="c", row_index=index),
            "volume": _decimal_text(row["v"], field="v", row_index=index),
        })
    return rows


def _epoch_ms_to_utc(value, *, row_index: int) -> datetime:
    """Milliseconds since the epoch to an aware UTC instant.

    The vendor's aggregate timestamp is the START of the window. Treating it
    as the end would shift every bar by one interval and still look plausible.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveProviderError(
            f"aggregate row {row_index}: timestamp {value!r} is not epoch "
            "milliseconds")
    if isinstance(value, float) and not value.is_integer():
        raise MassiveProviderError(
            f"aggregate row {row_index}: timestamp {value!r} is not a whole "
            "number of milliseconds")
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def adapt_split_rows(payload: dict, *, instrument_id: str) -> list[dict]:
    """Split records, with the vendor's from/to pair read as a ratio.

    ``split_from``/``split_to`` are shares before and after: a 4-for-1 is
    from 1 to 4. Reading them the wrong way round inverts every adjustment,
    which would look like a 16x error rather than a 4x one and is exactly the
    kind of mistake that survives a casual review.
    """
    results = payload.get("results")
    if results is None:
        raise MassiveProviderError(
            f"{instrument_id}: the splits response has no 'results' field")
    if not isinstance(results, list):
        raise MassiveProviderError(f"{instrument_id}: 'results' is not a list")

    wanted = instrument_id.split(":")[-1]
    rows = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise MassiveProviderError(
                f"{instrument_id}: split row {index} is not an object")
        echoed = str(row.get("ticker", wanted)).strip().upper()
        if echoed != wanted:
            raise MassiveProviderError(
                f"{instrument_id}: split row {index} is about {echoed!r}")
        for key in ("execution_date", "split_from", "split_to"):
            if key not in row:
                raise MassiveProviderError(
                    f"{instrument_id}: split row {index} is missing {key!r}")
        rows.append({
            "effective_date": str(row["execution_date"])[:10],
            "ratio_numerator": int(row["split_to"]),
            "ratio_denominator": int(row["split_from"]),
        })
    return rows


def _day(value) -> str:
    """A request bound as the vendor wants it: a plain calendar date."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat()
    return str(value).strip()[:10]


def parse_reference_ticker(row: dict) -> MassiveReferenceTicker:
    """Turn a vendor reference row into a venue this platform can name.

    Refuses unknown exchange codes and unknown security types instead of
    defaulting. A ticker registered under the wrong venue is a different
    instrument with a plausible name, and every artefact referencing it would
    be wrong in a way no test about prices would catch.
    """
    symbol = str(row.get("ticker", "")).strip().upper()
    if not symbol:
        raise MassiveProviderError("a reference row must name a ticker")
    code = str(row.get("primary_exchange", "")).strip().upper()
    if code not in VENUE_BY_EXCHANGE_CODE:
        raise MassiveProviderError(
            f"unknown exchange code {code!r} for {symbol}; this platform will "
            "not guess a venue. Add an explicit mapping or leave it unregistered.")
    vendor_type = str(row.get("type", "")).strip().upper()
    if vendor_type not in ASSET_CLASS_BY_VENDOR_TYPE:
        raise MassiveProviderError(
            f"unknown security type {vendor_type!r} for {symbol}; an ETF and a "
            "common share are not interchangeable and neither is a default.")
    currency = str(row.get("currency_name", "USD")).strip().upper() or "USD"
    if currency != "USD":
        raise MassiveProviderError(
            f"{symbol} is quoted in {currency}; this platform's US equity "
            "support is USD only and will not convert silently.")
    return MassiveReferenceTicker(
        symbol=symbol,
        venue=VENUE_BY_EXCHANGE_CODE[code],
        asset_class=ASSET_CLASS_BY_VENDOR_TYPE[vendor_type],
        display_name=str(row.get("name", "")).strip() or symbol,
        currency=currency,
        active=bool(row.get("active", True)))


def _iso(value) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise MassiveProviderError("a request bound must carry a timezone")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def build_massive_provider(instruments=(), *, transport=None, credentials=None,
                           adjustment_policy: str = ADJUSTMENT_RAW):
    """Build the provider. Offline unless a transport is handed in explicitly."""
    return MassiveStocksHistoricalProvider(
        instruments, transport=transport, credentials=credentials,
        adjustment_policy=adjustment_policy)


def build_live_massive_provider(instruments=()):
    """The shape a networked build would take. Not wired to a transport here.

    Kept as a named function so that the one place a real transport would be
    injected is obvious and reviewable, rather than appearing inline in some
    future capture script. Phase 6D ships no HTTP transport at all, so this
    still cannot reach the network.
    """
    return MassiveStocksHistoricalProvider(
        instruments, transport=None, credentials=massive_credentials(),
        adjustment_policy=ADJUSTMENT_SPLIT_ADJUSTED)


__all__ = [
    "ADJUSTED_FLAG", "AGGREGATE_FIELDS", "ALLOWED_PATHS",
    "ALLOWED_PATH_PATTERNS", "ASSET_CLASS_BY_VENDOR_TYPE",
    "FORBIDDEN_PATH_MARKERS", "MAX_ROWS_PER_PAGE", "SPLITS_ENDPOINT",
    "TIMEFRAME_AGGREGATES", "adapt_aggregate_rows", "adapt_split_rows",
    "bars_path", "reference_path", "require_aggregate_window",
    "MASSIVE_BASE_URL", "MASSIVE_PROVIDER_SCHEMA_VERSION",
    "MASSIVE_STOCKS_CAPABILITIES", "MASSIVE_STOCKS_HISTORICAL_V1",
    "MassiveProviderError", "MassiveReferenceTicker",
    "MassiveStocksHistoricalProvider", "MassiveTransport", "NoNetworkTransport",
    "OfflineTransportError", "RecordedTransport", "VENUE_BY_EXCHANGE_CODE",
    "build_live_massive_provider", "build_massive_provider",
    "parse_reference_ticker",
]
