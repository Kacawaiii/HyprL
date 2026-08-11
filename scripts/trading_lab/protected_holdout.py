"""The confirmatory holdout guard.

Phase 4D registered a window nobody may look at before it is used, once, to
answer whether the V2 feature representation carries any predictive signal:

    BTC-USD and ETH-USD, 2026-09-01T00:00:00Z through 2026-11-30T23:00:00Z.

Shadow trading runs continuously and would, left alone, march straight into it
on the first of September. That is the failure this module exists to make
impossible. Two properties matter more than convenience:

**The guard sits at the source, not at the display.** A protected candle must
never be requested, parsed, persisted, featurised, predicted on, or streamed.
Hiding it after download would leave it in a database, in a cache, and in the
session's hash chain -- and a holdout that has been stored is a holdout that
has been observed.

**No human has to remember.** The boundary is a property of the clock and the
data, checked on every request and every candle. A session running on the 31st
of August at 23:59 crosses into EMBARGOED by itself, with an explicit event,
and stays there until December.

The window is not restated here: it is read from the committed Phase 4D
contract, so the two cannot drift apart. A test asserts they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.research_holdout import CONFIRMATORY_HOLDOUT_V2

PROTECTED_HOLDOUT_SCHEMA_VERSION = "trading-lab.protected-holdout.v1"


class ProtectedHoldoutError(RuntimeError):
    """Raised when anything tries to touch reserved research data."""


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value))
        except (TypeError, ValueError) as error:
            raise ProtectedHoldoutError(f"{field} is not ISO-8601: {value!r}") from error
    if moment.tzinfo is None:
        # A naive timestamp would be compared against an aware boundary and
        # silently mis-classify a protected bar. Refuse it.
        raise ProtectedHoldoutError(f"{field} must carry a timezone: {value!r}")
    return moment.astimezone(timezone.utc)


def _symbol_of(value: object) -> object:
    """Reduce any identity to its bare symbol, for comparison.

    Handles an InstrumentSpec explicitly rather than letting it fall through
    to the fail-closed branch. Failing closed would give the right answer for
    a protected instrument and the *wrong* one for every other spec ever
    registered -- a future SOL-USD would be reported as reserved.
    """
    identity = getattr(value, "instrument_id", None)     # InstrumentSpec
    if identity is not None:
        value = identity
    symbol = getattr(value, "symbol", None)              # InstrumentId
    if symbol is not None:
        return symbol
    if isinstance(value, str) and ":" in value:
        return value.partition(":")[2]
    return value


@dataclass(frozen=True)
class ProtectedResearchWindow:
    """A range of market data reserved for one future confirmatory test."""

    holdout_id: str
    products: tuple[str, ...]
    start: str
    end: str
    purpose: str
    single_use: bool = True
    observed: bool = False
    timeframe: str = "1h"

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": PROTECTED_HOLDOUT_SCHEMA_VERSION,
            "holdout_id": self.holdout_id,
            "products": list(self.products),
            "start": self.start,
            "end": self.end,
            "timeframe": self.timeframe,
            "purpose": self.purpose,
            "single_use": self.single_use,
            "observed": self.observed,
        }

    @property
    def holdout_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
            .encode("utf-8")).hexdigest()

    @property
    def start_at(self) -> datetime:
        return _parse(self.start, field="start")

    @property
    def end_at(self) -> datetime:
        """The last PROTECTED bar opening, inclusive."""
        return _parse(self.end, field="end")

    @property
    def closes_at(self) -> datetime:
        """The instant after which no protected bar can still be produced.

        `end` names the last protected bar's OPENING, so the window is not over
        when that instant passes -- the bar is still forming. Lifting the
        embargo at 23:00 on the last day would hand the session the final
        protected candle an hour later. It is over one bar duration later.
        """
        return self.end_at + TIMEFRAME_DURATIONS[self.timeframe]

    def protects_product(self, product: object) -> bool:
        """Is this instrument reserved -- however the caller spelled it?

        This used to be ``product in self.products``: a membership test on raw
        text. It meant every spelling but the exact one reported *unprotected*
        and walked straight past the embargo -- ``btc-usd``, ``BTCUSD``,
        ``BTC/USD``, ``" BTC-USD"``, ``coinbase:BTC-USD``, twelve of thirteen
        variants tried. Nothing in the codebase happened to send those
        spellings, so the hole stayed open and invisible.

        Comparison now happens on the canonical symbol, and it is deliberately
        venue-blind: a protected symbol quoted against some other venue is
        still refused. Being over-broad costs nothing today -- there is one
        venue -- while being narrow reopens the bypass.

        A value too malformed to canonicalise is treated as protected rather
        than waved through. If a caller cannot say what market it means, this
        guard will not guess in the permissive direction.
        """
        from scripts.trading_lab.instruments import (
            InstrumentError, normalize_symbol)

        if isinstance(product, str) and product in self.products:
            return True                              # fast path, same answer
        try:
            candidate = normalize_symbol(_symbol_of(product))
        except InstrumentError:
            # Unreadable input, and this is a safety guard: fail closed.
            return True
        for protected in self.products:
            try:
                if normalize_symbol(_symbol_of(protected)) == candidate:
                    return True
            except InstrumentError:                  # pragma: no cover
                continue
        return False

    def covers(self, product: str, bar_open_at: object) -> bool:
        """Is this specific candle inside the reserved window?"""
        if not self.protects_product(product):
            return False
        moment = _parse(bar_open_at, field="bar_open_at")
        return self.start_at <= moment <= self.end_at

    def active_at(self, moment: object) -> bool:
        """Is the window currently in force, as of a wall-clock instant?"""
        return _parse(moment, field="now") >= self.start_at

    def elapsed_at(self, moment: object) -> bool:
        return _parse(moment, field="now") >= self.closes_at


PROTECTED_WINDOW_V1 = ProtectedResearchWindow(
    holdout_id=CONFIRMATORY_HOLDOUT_V2["holdout_id"],
    products=tuple(CONFIRMATORY_HOLDOUT_V2["products"]),
    start=CONFIRMATORY_HOLDOUT_V2["range_start"],
    end=CONFIRMATORY_HOLDOUT_V2["range_end"],
    purpose="V2 confirmatory holdout",
    single_use=bool(CONFIRMATORY_HOLDOUT_V2["single_use"]),
    observed=False,
    timeframe=CONFIRMATORY_HOLDOUT_V2["timeframe"],
)


def require_unprotected_bar(product: str, bar_open_at: object, *,
                            window: ProtectedResearchWindow = PROTECTED_WINDOW_V1) -> None:
    """Refuse a single candle. Called before parsing, storing or using one."""
    if window.covers(product, bar_open_at):
        raise ProtectedHoldoutError(
            f"{product} candle at {bar_open_at} lies inside the reserved research "
            f"holdout {window.start}..{window.end}; shadow trading must not observe it")


def require_unprotected_request(product: str, *, start: object, end: object,
                                window: ProtectedResearchWindow = PROTECTED_WINDOW_V1
                                ) -> None:
    """Refuse a whole request whose range overlaps the window at all.

    Checked before the fetch, so protected data is never even asked for.
    """
    if not window.protects_product(product):
        return
    first = _parse(start, field="start")
    last = _parse(end, field="end")
    if first > last:
        raise ProtectedHoldoutError("request end precedes its start")
    if first <= window.end_at and last >= window.start_at:
        raise ProtectedHoldoutError(
            f"requested {product} range {start}..{end} overlaps the reserved research "
            f"holdout {window.start}..{window.end}; the request is refused")


def embargo_state(product: str, *, now: object,
                  window: ProtectedResearchWindow = PROTECTED_WINDOW_V1) -> dict:
    """What a session may do with this product right now, and why."""
    protected = window.protects_product(product)
    active = window.active_at(now)
    elapsed = window.elapsed_at(now)
    embargoed = protected and active and not elapsed
    if not protected:
        reason = "product is not part of any reserved research holdout"
    elif embargoed:
        reason = (f"paper trading for {product} is disabled to preserve the "
                  f"confirmatory research holdout {window.start}..{window.end}")
    elif elapsed:
        reason = ("the reserved window has elapsed; it may only be used once, by the "
                  "registered confirmatory evaluation")
    else:
        reason = (f"paper trading for {product} is allowed until the embargo boundary "
                  f"at {window.start}")
    return {
        "product": product,
        "protected_product": protected,
        "embargoed": embargoed,
        "window_active": active,
        "window_elapsed": elapsed,
        "start": window.start,
        "end": window.end,
        "closes_at": _iso_z(window.closes_at),
        "holdout_id": window.holdout_id,
        "holdout_hash": window.holdout_hash,
        "observed": window.observed,
        "reason": reason,
    }


def require_tradeable_now(product: str, *, now: object,
                          window: ProtectedResearchWindow = PROTECTED_WINDOW_V1) -> None:
    """Refuse to run a product at all while its window is in force."""
    state = embargo_state(product, now=now, window=window)
    if state["embargoed"]:
        raise ProtectedHoldoutError(state["reason"])
