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

from dataclasses import dataclass
from datetime import datetime, timezone
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

# The only paths this provider may request. An allowlist rather than a
# denylist: a denylist has to anticipate every account endpoint a vendor might
# ever add, and it only has to miss one.
ALLOWED_PATHS = (
    "/v1/stocks/bars",
    "/v1/stocks/splits",
    "/v1/stocks/dividends",
    "/v1/reference/tickers",
)

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
        if path not in ALLOWED_PATHS:
            raise MassiveProviderError(
                f"{path!r} is not on this provider's allowlist "
                f"{list(ALLOWED_PATHS)}")
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
        payload = self._fetch("/v1/stocks/bars", {
            "symbol": identity.symbol,
            "venue": identity.venue,
            "timeframe": frame.label,
            "start": _iso(start),
            "end": _iso(end),
            "adjustment": policy,
        })
        rows = payload.get("bars", [])
        if not isinstance(rows, list):
            raise MassiveProviderError("bars payload is not a list")
        declared = payload.get("adjustment")
        if declared is not None and declared != policy:
            # The vendor answering with a different policy than was asked for
            # is exactly the silent mismatch this whole module guards against.
            raise EquityMarketError(
                f"requested {policy} bars but the response declares "
                f"{declared!r}; refusing to relabel data")
        bars = tuple(
            build_equity_bar(
                instrument=identity, timeframe=frame.label,
                provider_id=self.provider_id, adjustment_policy=policy,
                bar_open_at=row["bar_open_at"], bar_close_at=row["bar_close_at"],
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
                session_date=str(row.get("session_date", ""))[:10])
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
        payload = self._fetch("/v1/stocks/splits", {
            "symbol": identity.symbol, "venue": identity.venue,
            "start": _iso(start), "end": _iso(end)})
        return tuple(
            StockSplit(instrument_id=identity,
                       effective_date=str(row["effective_date"])[:10],
                       ratio_numerator=int(row["ratio_numerator"]),
                       ratio_denominator=int(row["ratio_denominator"]),
                       source=self.provider_id)
            for row in payload.get("splits", []))

    def get_dividends(self, instrument, *, start, end) -> tuple[CashDividend, ...]:
        identity = self.require_supported(instrument)
        payload = self._fetch("/v1/stocks/dividends", {
            "symbol": identity.symbol, "venue": identity.venue,
            "start": _iso(start), "end": _iso(end)})
        return tuple(
            CashDividend(instrument_id=identity,
                         ex_date=str(row["ex_date"])[:10],
                         amount=Decimal(str(row["amount"])),
                         currency=str(row.get("currency", "USD")),
                         source=self.provider_id)
            for row in payload.get("dividends", []))

    # --- reference metadata -----------------------------------------------

    def get_reference_ticker(self, symbol: str) -> MassiveReferenceTicker:
        """What the vendor says this ticker is. Venue is read, never guessed."""
        payload = self._fetch("/v1/reference/tickers", {"symbol": str(symbol)})
        rows = payload.get("results", [])
        if not rows:
            raise MassiveProviderError(f"no reference metadata for {symbol!r}")
        return parse_reference_ticker(rows[0])

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


def parse_reference_ticker(row: dict) -> MassiveReferenceTicker:
    """Turn a vendor reference row into a venue this platform can name.

    Refuses unknown exchange codes and unknown security types instead of
    defaulting. A ticker registered under the wrong venue is a different
    instrument with a plausible name, and every artefact referencing it would
    be wrong in a way no test about prices would catch.
    """
    symbol = str(row.get("symbol", "")).strip().upper()
    if not symbol:
        raise MassiveProviderError("a reference row must name a symbol")
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
    "ALLOWED_PATHS", "ASSET_CLASS_BY_VENDOR_TYPE", "FORBIDDEN_PATH_MARKERS",
    "MASSIVE_BASE_URL", "MASSIVE_PROVIDER_SCHEMA_VERSION",
    "MASSIVE_STOCKS_CAPABILITIES", "MASSIVE_STOCKS_HISTORICAL_V1",
    "MassiveProviderError", "MassiveReferenceTicker",
    "MassiveStocksHistoricalProvider", "MassiveTransport", "NoNetworkTransport",
    "OfflineTransportError", "RecordedTransport", "VENUE_BY_EXCHANGE_CODE",
    "build_live_massive_provider", "build_massive_provider",
    "parse_reference_ticker",
]
