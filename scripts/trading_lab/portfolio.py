"""One portfolio holding several instruments on shared capital.

Until now BTC and ETH were two synthetic accounts of 100 000 each that never
met. That is not a portfolio; it is two backtests printed side by side, and
adding their percentages is meaningless because neither ever competed with
the other for a dollar. This module makes them share one cash ledger, one
equity figure and one exposure budget.

Nothing here predicts anything. It consumes the PositionTargets that Signal
V1 and Risk V1 already produce and decides how they coexist under shared
capital. No threshold, no cap and no cost in this file was derived from any
observed P&L.

**The order-independence problem is the whole design.** If BTC is filled
first, cash and equity move, and sizing ETH from the new equity makes the
result depend on the order two instruments happen to appear in a list. That
is not a rounding difference: it is a different portfolio. So one
pre-trade equity is computed once per timestamp, every target in the batch is
sized from that same number, and only then are fills applied. Instrument
order affects serialization and nothing else, which is asserted by permuting
inputs and comparing the result hash.

**Nothing is forward-filled.** If an open position cannot be priced at a
timestamp, the portfolio has no honest equity to report, so no snapshot is
produced -- rather than marking yesterday's price and calling it today's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

from scripts.trading_lab.economic_backtest import (
    ECONOMIC_PRECISION,
    EXECUTION_SPEC_V1,
    ExecutionSpec,
    apply_quantity_delta,
)
from scripts.trading_lab.identity import IdentityError, resolve_instrument

PORTFOLIO_SCHEMA_VERSION = "trading-lab.portfolio.v1"
PORTFOLIO_RESULT_SCHEMA_VERSION = "trading-lab.portfolio-result.v1"

_ZERO = Decimal(0)
_ONE = Decimal(1)

# Stated once, here, so that no reader has to infer it from a number.
PORTFOLIO_LIMIT_V1_IS_NOT_OPTIMIZED = True

# How exactly per-instrument contributions must add up to the change in
# portfolio equity, as a fraction of equity. The decomposition is exact in
# arithmetic; each accumulation rounds at 34 significant digits, so the
# residual lands around 1E-34 relative. A real accounting error -- a fee
# charged to the wrong book, a fill counted twice -- is of order cents, about
# 1E-5 relative, so this bound is four orders tighter than needed to be
# generous and twenty-five orders tighter than any mistake worth catching.
ATTRIBUTION_RELATIVE_TOLERANCE = Decimal("1E-30")

# The limit check recomputes exposure as quantity x price / equity, while the
# quantity was derived as exposure x equity / price. That round trip is exact
# in arithmetic and lands one unit in the last place away at 34 significant
# digits, so an exposure allocated at exactly 0.25 can read back as
# 0.2500000000000000000000000000000001. Refusing that would be reporting a
# limit breach caused by division, not by a position. Anything a trader would
# call a breach is at least 1E-6 of NAV, twenty-four orders larger.
LIMIT_ROUNDING_TOLERANCE = Decimal("1E-30")


class PortfolioError(RuntimeError):
    """Raised when a portfolio operation cannot proceed safely."""


class PortfolioIdentityMismatch(PortfolioError):
    """Raised when one instrument's data would be used as another's."""


class PortfolioLimitBreached(PortfolioError):
    """Raised when a limit cannot be satisfied by the specified rule."""


class PortfolioValuationUnavailable(PortfolioError):
    """Raised when an open position cannot be priced at a timestamp."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _instrument_id(value, *, context: str) -> str:
    """Any accepted spelling to the canonical id, or refuse.

    Every identity entering this module goes through here. The Phase 6A-R
    audit found two boundaries where a raw string comparison let one
    instrument's data be used as another's; both produced plausible, wrong
    numbers. Nothing in this file compares a product spelling directly.
    """
    try:
        return resolve_instrument(value, context=context).canonical_id
    except IdentityError as error:
        raise PortfolioIdentityMismatch(str(error)) from error


# --- specification --------------------------------------------------------


@dataclass(frozen=True)
class PortfolioSpec:
    """How shared capital is allocated. Frozen before any result was seen.

    The 50 % gross cap is not an optimum and was not fitted. RiskSpec V1 caps
    a single instrument at 25 % of NAV, and there are two instruments, so 50 %
    is the gross exposure the existing rules already permit. The cap
    formalises the structure that was there; it does not tighten or loosen it.
    """

    protocol_version: str = PORTFOLIO_SCHEMA_VERSION
    base_currency: str = "USD"
    initial_equity: Decimal = Decimal("100000")
    max_instrument_abs_exposure: Decimal = Decimal("0.25")
    max_gross_exposure: Decimal = Decimal("0.50")
    max_net_abs_exposure: Decimal = Decimal("0.50")
    allocation_rule: str = "proportional-gross-cap-v1"
    simultaneous_rebalance_rule: str = "single-pretrade-equity-batch-v1"
    cash_model: str = "shared-cash-v1"
    short_model: str = "synthetic-linear-short-v1"
    optimized: bool = False

    def canonical(self) -> dict:
        return {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "base_currency": self.base_currency,
            "initial_equity": str(self.initial_equity),
            "max_instrument_abs_exposure": str(self.max_instrument_abs_exposure),
            "max_gross_exposure": str(self.max_gross_exposure),
            "max_net_abs_exposure": str(self.max_net_abs_exposure),
            "allocation_rule": self.allocation_rule,
            "simultaneous_rebalance_rule": self.simultaneous_rebalance_rule,
            "cash_model": self.cash_model,
            "short_model": self.short_model,
            "optimized": self.optimized,
        }

    @property
    def portfolio_spec_hash(self) -> str:
        return _sha256(self.canonical())


PORTFOLIO_SPEC_V1 = PortfolioSpec()


# --- inputs ---------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentPositionTarget:
    """One instrument's desired exposure, bound to that instrument."""

    instrument_id: str
    target_exposure: Decimal
    source_position_target_hash: str
    side: str = ""

    def canonical(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "target_exposure": str(self.target_exposure),
            "side": self.side,
            "source_position_target_hash": self.source_position_target_hash,
        }

    @property
    def instrument_target_hash(self) -> str:
        return _sha256(self.canonical())


@dataclass(frozen=True)
class PortfolioTargetSet:
    """Every target that applies at one instant. At most one per instrument."""

    timestamp: str
    targets: tuple[InstrumentPositionTarget, ...]
    portfolio_spec_hash: str

    def __post_init__(self):
        seen = set()
        normalised = []
        for target in self.targets:
            canonical = _instrument_id(target.instrument_id,
                                       context="portfolio target")
            if canonical in seen:
                raise PortfolioIdentityMismatch(
                    f"{canonical} appears twice in the batch at {self.timestamp}; "
                    "one instrument cannot hold two simultaneous targets")
            seen.add(canonical)
            normalised.append(
                target if target.instrument_id == canonical
                else InstrumentPositionTarget(
                    instrument_id=canonical,
                    target_exposure=target.target_exposure,
                    source_position_target_hash=target.source_position_target_hash,
                    side=target.side))
        # Sorted for serialization only. The economics must not depend on it,
        # which is what the permutation tests exist to prove.
        object.__setattr__(self, "targets",
                           tuple(sorted(normalised,
                                        key=lambda item: item.instrument_id)))

    def canonical(self) -> dict:
        return {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "portfolio_spec_hash": self.portfolio_spec_hash,
            "targets": [target.canonical() for target in self.targets],
        }

    @property
    def target_set_hash(self) -> str:
        return _sha256(self.canonical())

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(target.instrument_id for target in self.targets)


@dataclass(frozen=True)
class PortfolioMarketFrame:
    """Every observable price at one instant, bound to its instrument."""

    timestamp: str
    prices: tuple[tuple[str, Decimal], ...]

    def __post_init__(self):
        seen = {}
        for instrument, price in self.prices:
            canonical = _instrument_id(instrument, context="market frame")
            if canonical in seen:
                raise PortfolioIdentityMismatch(
                    f"{canonical} is priced twice at {self.timestamp}")
            if not isinstance(price, Decimal):
                raise PortfolioError(
                    f"{canonical} price must be a Decimal, not "
                    f"{type(price).__name__}")
            if price <= 0:
                raise PortfolioError(f"{canonical} price at {self.timestamp} "
                                     "must be positive")
            seen[canonical] = price
        object.__setattr__(self, "prices",
                           tuple(sorted(seen.items(), key=lambda item: item[0])))

    def price_of(self, instrument: str) -> Decimal:
        canonical = _instrument_id(instrument, context="market frame lookup")
        for name, price in self.prices:
            if name == canonical:
                return price
        raise PortfolioValuationUnavailable(
            f"no observable price for {canonical} at {self.timestamp}")

    def has(self, instrument: str) -> bool:
        canonical = _instrument_id(instrument, context="market frame lookup")
        return any(name == canonical for name, _ in self.prices)

    def canonical(self) -> dict:
        return {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "prices": [[name, str(price)] for name, price in self.prices],
        }

    @property
    def frame_hash(self) -> str:
        return _sha256(self.canonical())


# --- state ----------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentPositionState:
    instrument_id: str
    quantity: Decimal
    mark_price: Decimal
    target_exposure: Decimal = _ZERO
    cumulative_fees: Decimal = _ZERO
    cumulative_slippage_cost: Decimal = _ZERO
    cumulative_gross_pnl: Decimal = _ZERO
    turnover_sum: Decimal = _ZERO

    @property
    def market_value(self) -> Decimal:
        # Inside the economic context, not the caller's. A property that
        # multiplies at whatever precision happens to be active makes the
        # portfolio's equity depend on who asked for it.
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return self.quantity * self.mark_price

    def canonical(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "quantity": str(self.quantity),
            "mark_price": str(self.mark_price),
            "market_value": str(self.market_value),
            "target_exposure": str(self.target_exposure),
            "cumulative_fees": str(self.cumulative_fees),
            "cumulative_slippage_cost": str(self.cumulative_slippage_cost),
            "cumulative_gross_pnl": str(self.cumulative_gross_pnl),
        }

    @property
    def position_hash(self) -> str:
        return _sha256(self.canonical())


@dataclass(frozen=True)
class PortfolioState:
    """One cash ledger, several positions, one equity."""

    timestamp: str
    cash: Decimal
    positions: tuple[InstrumentPositionState, ...]

    def __post_init__(self):
        object.__setattr__(self, "positions",
                           tuple(sorted(self.positions,
                                        key=lambda item: item.instrument_id)))

    @property
    def equity(self) -> Decimal:
        """cash + the marked value of every position. The only definition."""
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return self.cash + sum((position.market_value
                                    for position in self.positions), _ZERO)

    @property
    def gross_exposure(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            equity = self.equity
            if equity == 0:
                return _ZERO
            return sum((abs(position.market_value)
                        for position in self.positions), _ZERO) / equity

    @property
    def net_exposure(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            equity = self.equity
            if equity == 0:
                return _ZERO
            return sum((position.market_value
                        for position in self.positions), _ZERO) / equity

    @property
    def cumulative_fees(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return sum((position.cumulative_fees
                        for position in self.positions), _ZERO)

    @property
    def cumulative_slippage_cost(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return sum((position.cumulative_slippage_cost
                        for position in self.positions), _ZERO)

    def position(self, instrument: str) -> InstrumentPositionState | None:
        canonical = _instrument_id(instrument, context="position lookup")
        for position in self.positions:
            if position.instrument_id == canonical:
                return position
        return None

    def canonical(self) -> dict:
        return {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "cash": str(self.cash),
            "equity": str(self.equity),
            "gross_exposure": str(self.gross_exposure),
            "net_exposure": str(self.net_exposure),
            "cumulative_fees": str(self.cumulative_fees),
            "cumulative_slippage_cost": str(self.cumulative_slippage_cost),
            "positions": [position.canonical() for position in self.positions],
        }

    @property
    def portfolio_state_hash(self) -> str:
        return _sha256(self.canonical())


@dataclass(frozen=True)
class PortfolioFill:
    timestamp: str
    instrument_id: str
    side: str
    reference_price: Decimal
    fill_price: Decimal
    quantity_delta: Decimal
    notional: Decimal
    fee: Decimal
    slippage_cost: Decimal
    position_after: Decimal
    source_position_target_hash: str

    def canonical(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "reference_price": str(self.reference_price),
            "fill_price": str(self.fill_price),
            "quantity_delta": str(self.quantity_delta),
            "notional": str(self.notional),
            "fee": str(self.fee),
            "slippage_cost": str(self.slippage_cost),
            "position_after": str(self.position_after),
            "source_position_target_hash": self.source_position_target_hash,
        }

    @property
    def fill_hash(self) -> str:
        return _sha256(self.canonical())


@dataclass(frozen=True)
class RebalanceOutcome:
    state: PortfolioState
    fills: tuple[PortfolioFill, ...]
    pre_trade_equity: Decimal
    portfolio_scale: Decimal
    requested_gross: Decimal
    scaled: bool = False


def initial_portfolio_state(spec: PortfolioSpec = PORTFOLIO_SPEC_V1, *,
                            timestamp: str = "") -> PortfolioState:
    return PortfolioState(timestamp=timestamp, cash=spec.initial_equity,
                          positions=())


# --- the batch ------------------------------------------------------------


def mark_to_market(state: PortfolioState, frame: PortfolioMarketFrame
                   ) -> PortfolioState:
    """Reprice every open position, accruing each one's gross P&L.

    Refuses rather than forward-filling. A portfolio whose equity depends on a
    price nobody observed is not a measurement; it is an estimate presented as
    one, and it would silently smooth exactly the drawdowns a reader is
    looking for.
    """
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        marked = []
        for position in state.positions:
            if position.quantity == 0:
                # A flat position is worth nothing at any price, so a missing
                # quote cannot make the equity uncertain. Take the new mark
                # when it exists so the next P&L accrual starts from a real
                # price, and keep the old one otherwise.
                price = (frame.price_of(position.instrument_id)
                         if frame.has(position.instrument_id)
                         else position.mark_price)
                marked.append(InstrumentPositionState(
                    instrument_id=position.instrument_id, quantity=_ZERO,
                    mark_price=price,
                    target_exposure=position.target_exposure,
                    cumulative_fees=position.cumulative_fees,
                    cumulative_slippage_cost=position.cumulative_slippage_cost,
                    cumulative_gross_pnl=position.cumulative_gross_pnl,
                    turnover_sum=position.turnover_sum))
                continue
            if not frame.has(position.instrument_id):
                raise PortfolioValuationUnavailable(
                    f"{position.instrument_id} holds {position.quantity} units at "
                    f"{frame.timestamp} and has no observable price; the "
                    "portfolio has no honest equity at this timestamp")
            price = frame.price_of(position.instrument_id)
            marked.append(InstrumentPositionState(
                instrument_id=position.instrument_id,
                quantity=position.quantity, mark_price=price,
                target_exposure=position.target_exposure,
                cumulative_fees=position.cumulative_fees,
                cumulative_slippage_cost=position.cumulative_slippage_cost,
                cumulative_gross_pnl=position.cumulative_gross_pnl
                + position.quantity * (price - position.mark_price),
                turnover_sum=position.turnover_sum))
        return PortfolioState(timestamp=frame.timestamp, cash=state.cash,
                              positions=tuple(marked))


def allocate(target_set: PortfolioTargetSet, spec: PortfolioSpec = PORTFOLIO_SPEC_V1):
    """Apply the per-instrument cap, then scale the batch to the gross cap.

    Proportional, never prioritised. Giving BTC its full request and cutting
    ETH -- or cutting whichever arrived second -- would make the allocation
    depend on registry order or list order, which is a portfolio decision
    nobody specified and nobody could reproduce.
    """
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        requested = []
        for target in target_set.targets:
            exposure = target.target_exposure
            if abs(exposure) > (spec.max_instrument_abs_exposure
                            + LIMIT_ROUNDING_TOLERANCE):
                raise PortfolioLimitBreached(
                    f"{target.instrument_id} requests {exposure} at "
                    f"{target_set.timestamp}, beyond the per-instrument cap of "
                    f"{spec.max_instrument_abs_exposure}")
            requested.append((target.instrument_id, exposure, target))
        requested_gross = sum((abs(item[1]) for item in requested), _ZERO)
        if requested_gross <= spec.max_gross_exposure or requested_gross == 0:
            scale = _ONE
        else:
            scale = spec.max_gross_exposure / requested_gross
        allocated = tuple((instrument, exposure * scale, target)
                          for instrument, exposure, target in requested)
        net = sum((item[1] for item in allocated), _ZERO)
        if abs(net) > spec.max_net_abs_exposure:
            # No second, unspecified scaling. A net cap that bites under this
            # rule set means the rule set is wrong, and inventing an algorithm
            # here would hide that.
            raise PortfolioLimitBreached(
                f"net exposure {net} at {target_set.timestamp} exceeds "
                f"{spec.max_net_abs_exposure} after gross scaling; the "
                "specified allocation rule cannot satisfy both caps")
        return allocated, scale, requested_gross


def rebalance(state: PortfolioState, *, target_set: PortfolioTargetSet,
              frame: PortfolioMarketFrame,
              spec: PortfolioSpec = PORTFOLIO_SPEC_V1,
              execution_spec: ExecutionSpec = EXECUTION_SPEC_V1
              ) -> RebalanceOutcome:
    """Move every instrument to its allocated exposure, atomically.

    The steps are ordered so that no economically observable state exists
    between one instrument and the next: mark everything, compute one shared
    pre-trade equity, size every target from that number, then apply the
    fills.
    """
    if target_set.timestamp != frame.timestamp:
        raise PortfolioError(
            f"target batch at {target_set.timestamp} cannot be executed against "
            f"a market frame at {frame.timestamp}")

    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION

        # A. mark, B. one shared pre-trade equity
        marked = mark_to_market(state, frame)
        pre_trade_equity = marked.equity
        if pre_trade_equity <= 0:
            raise PortfolioError(
                f"portfolio equity reached {pre_trade_equity} at "
                f"{frame.timestamp}; the synthetic account is insolvent")

        # C, D. validate and scale
        allocated, scale, requested_gross = allocate(target_set, spec)

        # E. every target quantity from the SAME pre-trade equity
        positions = {position.instrument_id: position
                     for position in marked.positions}
        planned = []
        for instrument, exposure, target in allocated:
            if not frame.has(instrument):
                raise PortfolioValuationUnavailable(
                    f"{instrument} has a target at {frame.timestamp} but no "
                    "observable price")
            reference = frame.price_of(instrument)
            target_quantity = (exposure * pre_trade_equity) / reference
            held = positions.get(instrument)
            current = held.quantity if held else _ZERO
            planned.append((instrument, exposure, reference,
                            target_quantity - current, target))

        # F, G, H. price every delta, then apply the aggregate cash impact
        cash = marked.cash
        fills = []
        updated = dict(positions)
        for instrument, exposure, reference, delta, target in planned:
            held = positions.get(instrument)
            base = held or InstrumentPositionState(
                instrument_id=instrument, quantity=_ZERO, mark_price=reference)
            if delta == 0:
                updated[instrument] = InstrumentPositionState(
                    instrument_id=instrument, quantity=base.quantity,
                    mark_price=reference, target_exposure=exposure,
                    cumulative_fees=base.cumulative_fees,
                    cumulative_slippage_cost=base.cumulative_slippage_cost,
                    cumulative_gross_pnl=base.cumulative_gross_pnl,
                    turnover_sum=base.turnover_sum)
                continue
            effect = apply_quantity_delta(quantity_delta=delta,
                                          reference_price=reference,
                                          execution_spec=execution_spec)
            cash += effect.cash_delta
            quantity = base.quantity + delta
            updated[instrument] = InstrumentPositionState(
                instrument_id=instrument, quantity=quantity,
                mark_price=reference, target_exposure=exposure,
                cumulative_fees=base.cumulative_fees + effect.fee,
                cumulative_slippage_cost=base.cumulative_slippage_cost
                + effect.slippage_cost,
                cumulative_gross_pnl=base.cumulative_gross_pnl,
                turnover_sum=base.turnover_sum
                + abs(delta * reference) / pre_trade_equity)
            fills.append(PortfolioFill(
                timestamp=frame.timestamp, instrument_id=instrument,
                side=effect.side, reference_price=reference,
                fill_price=effect.fill_price, quantity_delta=delta,
                notional=effect.notional, fee=effect.fee,
                slippage_cost=effect.slippage_cost, position_after=quantity,
                source_position_target_hash=target.source_position_target_hash))

        # I. final positions
        final = PortfolioState(timestamp=frame.timestamp, cash=cash,
                               positions=tuple(updated.values()))

        # J. invariants
        _require_limits(final, spec, pre_trade_equity=pre_trade_equity)
        return RebalanceOutcome(
            state=final,
            fills=tuple(sorted(fills, key=lambda item: item.instrument_id)),
            pre_trade_equity=pre_trade_equity, portfolio_scale=scale,
            requested_gross=requested_gross, scaled=scale != _ONE)


def _require_limits(state: PortfolioState, spec: PortfolioSpec, *,
                    pre_trade_equity: Decimal) -> None:
    """The caps hold on the exposure the batch actually put on.

    Measured against the pre-trade equity the targets were sized from, not
    against equity after the fills. Those differ by exactly the execution
    cost, so checking a 25 % target against post-trade equity would report a
    breach on every single trade -- the cap and the sizing rule would be
    talking about two different denominators.

    Realised exposure also drifts afterwards as prices move, and that is not a
    limit breach either: it is the market, and no rebalance happened. The cap
    governs what the portfolio *asks for* and what it *executes*; a
    continuously enforced realised band would be a different, unspecified
    strategy that trades on price moves alone.

    A position held without a target still consumes the gross budget, so it is
    counted here. Rather than silently rescaling something nobody asked to
    trade, this refuses -- the allocation rule covers requested exposure, and
    stretching it to cover untargeted holdings would be an invented algorithm.
    """
    if pre_trade_equity <= 0:                        # pragma: no cover - refused earlier
        return
    gross = sum((abs(position.market_value) for position in state.positions),
                _ZERO) / pre_trade_equity
    net = sum((position.market_value for position in state.positions),
              _ZERO) / pre_trade_equity
    if gross > spec.max_gross_exposure + LIMIT_ROUNDING_TOLERANCE:
        raise PortfolioLimitBreached(
            f"gross exposure {gross} at {state.timestamp} exceeds "
            f"{spec.max_gross_exposure} after rebalancing")
    if abs(net) > spec.max_net_abs_exposure + LIMIT_ROUNDING_TOLERANCE:
        raise PortfolioLimitBreached(
            f"net exposure {net} at {state.timestamp} exceeds "
            f"{spec.max_net_abs_exposure} after rebalancing")
    for position in state.positions:
        exposure = position.market_value / pre_trade_equity
        if abs(exposure) > (spec.max_instrument_abs_exposure
                            + LIMIT_ROUNDING_TOLERANCE):
            raise PortfolioLimitBreached(
                f"{position.instrument_id} holds {exposure} of NAV at "
                f"{state.timestamp}, beyond {spec.max_instrument_abs_exposure}")


# --- running a portfolio over time ----------------------------------------


@dataclass(frozen=True)
class PortfolioSnapshot:
    """One observed portfolio valuation. Never interpolated."""

    timestamp: str
    cash: Decimal
    equity: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    cumulative_fees: Decimal
    cumulative_slippage_cost: Decimal
    positions: tuple[tuple[str, Decimal, Decimal, Decimal], ...]

    def canonical(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cash": str(self.cash),
            "equity": str(self.equity),
            "gross_exposure": str(self.gross_exposure),
            "net_exposure": str(self.net_exposure),
            "cumulative_fees": str(self.cumulative_fees),
            "cumulative_slippage_cost": str(self.cumulative_slippage_cost),
            "positions": [[name, str(quantity), str(price), str(exposure)]
                          for name, quantity, price, exposure in self.positions],
        }


def _snapshot(state: PortfolioState) -> PortfolioSnapshot:
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        return _build_snapshot(state)


def _build_snapshot(state: PortfolioState) -> PortfolioSnapshot:
    equity = state.equity
    return PortfolioSnapshot(
        timestamp=state.timestamp, cash=state.cash, equity=equity,
        gross_exposure=state.gross_exposure, net_exposure=state.net_exposure,
        cumulative_fees=state.cumulative_fees,
        cumulative_slippage_cost=state.cumulative_slippage_cost,
        positions=tuple(
            (position.instrument_id, position.quantity, position.mark_price,
             (position.market_value / equity) if equity else _ZERO)
            for position in state.positions))


@dataclass(frozen=True)
class InstrumentAttribution:
    """What one instrument contributed, and what it cost to hold it.

    Gross P&L is quantity held times the price move between marks -- an
    instrument's own arithmetic, not a share of the shared cash. Execution
    costs attach to that instrument's own fills. The net contributions of all
    instruments must add up to the change in portfolio equity, which is
    asserted rather than assumed.
    """

    instrument_id: str
    gross_pnl: Decimal
    fees: Decimal
    slippage_cost: Decimal
    turnover: Decimal
    average_abs_exposure: Decimal
    fill_count: int

    @property
    def execution_cost(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return self.fees + self.slippage_cost

    @property
    def net_pnl(self) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return self.gross_pnl - self.execution_cost

    def canonical(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "gross_pnl": str(self.gross_pnl),
            "fees": str(self.fees),
            "slippage_cost": str(self.slippage_cost),
            "execution_cost": str(self.execution_cost),
            "net_pnl": str(self.net_pnl),
            "turnover": str(self.turnover),
            "average_abs_exposure": str(self.average_abs_exposure),
            "fill_count": self.fill_count,
        }


@dataclass(frozen=True)
class PortfolioMetrics:
    initial_equity: Decimal
    final_equity: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    gross_return: Decimal
    net_return: Decimal
    total_fees: Decimal
    total_slippage_cost: Decimal
    total_execution_cost: Decimal
    portfolio_turnover: Decimal
    max_drawdown: Decimal
    annualized_sharpe: Decimal
    average_gross_exposure: Decimal
    average_abs_net_exposure: Decimal
    max_observed_gross_exposure: Decimal
    fill_count: int
    rebalance_count: int

    def canonical(self) -> dict:
        return {name: (value if isinstance(value, int) else str(value))
                for name, value in self.__dict__.items()}


def _metrics(snapshots, fills, *, spec: PortfolioSpec, rebalances: int,
             periods_per_year: int) -> PortfolioMetrics:
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        initial = spec.initial_equity
        final = snapshots[-1].equity if snapshots else initial
        fees = sum((fill.fee for fill in fills), _ZERO)
        slippage = sum((fill.slippage_cost for fill in fills), _ZERO)
        net_pnl = final - initial
        gross_pnl = net_pnl + fees + slippage
        peak = snapshots[0].equity if snapshots else initial
        drawdown = _ZERO
        for snapshot in snapshots:
            peak = max(peak, snapshot.equity)
            if peak > 0:
                drawdown = min(drawdown, (snapshot.equity - peak) / peak)
        gross_sum = sum((snapshot.gross_exposure for snapshot in snapshots), _ZERO)
        net_sum = sum((abs(snapshot.net_exposure) for snapshot in snapshots), _ZERO)
        count = Decimal(len(snapshots)) if snapshots else _ONE
        turnover = sum((abs(fill.quantity_delta * fill.reference_price)
                        for fill in fills), _ZERO) / initial
        return PortfolioMetrics(
            initial_equity=initial, final_equity=final,
            gross_pnl=gross_pnl, net_pnl=net_pnl,
            gross_return=gross_pnl / initial, net_return=net_pnl / initial,
            total_fees=fees, total_slippage_cost=slippage,
            total_execution_cost=fees + slippage,
            portfolio_turnover=turnover, max_drawdown=drawdown,
            annualized_sharpe=_sharpe(snapshots, periods_per_year),
            average_gross_exposure=gross_sum / count,
            average_abs_net_exposure=net_sum / count,
            max_observed_gross_exposure=max(
                (snapshot.gross_exposure for snapshot in snapshots), default=_ZERO),
            fill_count=len(fills), rebalance_count=rebalances)


def _sharpe(snapshots, periods_per_year: int) -> Decimal:
    """Annualised Sharpe of the equity curve, at zero risk-free rate.

    The annualisation factor comes from the trading calendar rather than a
    constant, so an instrument that does not trade 24/7 cannot be scaled by
    a crypto year.
    """
    if len(snapshots) < 3:
        return _ZERO
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        returns = []
        for previous, current in zip(snapshots, snapshots[1:]):
            if previous.equity <= 0:
                return _ZERO
            returns.append((current.equity - previous.equity) / previous.equity)
        count = Decimal(len(returns))
        mean = sum(returns, _ZERO) / count
        variance = sum(((value - mean) ** 2 for value in returns), _ZERO) / count
        if variance <= 0:
            return _ZERO
        return (mean / variance.sqrt()) * Decimal(periods_per_year).sqrt()


def _attribution(final_state: PortfolioState, fills, snapshots
                 ) -> tuple[InstrumentAttribution, ...]:
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        instruments = sorted({position.instrument_id
                              for position in final_state.positions}
                             | {fill.instrument_id for fill in fills})
        count = Decimal(len(snapshots)) if snapshots else _ONE
        records = []
        for instrument in instruments:
            position = final_state.position(instrument)
            own = [fill for fill in fills if fill.instrument_id == instrument]
            exposure_sum = _ZERO
            for snapshot in snapshots:
                for name, _, _, exposure in snapshot.positions:
                    if name == instrument:
                        exposure_sum += abs(exposure)
            records.append(InstrumentAttribution(
                instrument_id=instrument,
                gross_pnl=position.cumulative_gross_pnl if position else _ZERO,
                fees=sum((fill.fee for fill in own), _ZERO),
                slippage_cost=sum((fill.slippage_cost for fill in own), _ZERO),
                turnover=position.turnover_sum if position else _ZERO,
                average_abs_exposure=exposure_sum / count,
                fill_count=len(own)))
        return tuple(records)


@dataclass(frozen=True)
class PortfolioBacktestResult:
    spec: PortfolioSpec
    experiment_type: str
    confirmatory: bool
    live_execution: bool
    cost_model: str
    instruments: tuple[str, ...]
    metrics: PortfolioMetrics
    gross_metrics: PortfolioMetrics
    attribution: tuple[InstrumentAttribution, ...]
    fills: tuple[PortfolioFill, ...]
    equity_curve: tuple[PortfolioSnapshot, ...]
    target_set_hashes: tuple[str, ...]
    unavailable_valuations: tuple[str, ...] = ()
    scaled_batches: tuple[str, ...] = ()

    def canonical(self) -> dict:
        return {
            "schema_version": PORTFOLIO_RESULT_SCHEMA_VERSION,
            "portfolio_spec": self.spec.canonical(),
            "portfolio_spec_hash": self.spec.portfolio_spec_hash,
            "experiment_type": self.experiment_type,
            "confirmatory": self.confirmatory,
            "live_execution": self.live_execution,
            "cost_model": self.cost_model,
            "instruments": list(self.instruments),
            "metrics": self.metrics.canonical(),
            "gross_metrics": self.gross_metrics.canonical(),
            "attribution": [record.canonical() for record in self.attribution],
            "fills": [fill.canonical() for fill in self.fills],
            "equity_curve": [snapshot.canonical() for snapshot in self.equity_curve],
            "target_set_hashes": list(self.target_set_hashes),
            "unavailable_valuations": list(self.unavailable_valuations),
            "scaled_batches": list(self.scaled_batches),
        }

    @property
    def result_hash(self) -> str:
        return _sha256(self.canonical())

    def reconciliation_residual(self) -> Decimal:
        """Sum of net contributions minus the change in portfolio equity."""
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            total = sum((record.net_pnl for record in self.attribution), _ZERO)
            return total - (self.metrics.final_equity
                            - self.metrics.initial_equity)

    def attribution_reconciles(self) -> bool:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            equity = abs(self.metrics.final_equity)
            if equity == 0:
                return self.reconciliation_residual() == _ZERO
            return (abs(self.reconciliation_residual()) / equity
                    <= ATTRIBUTION_RELATIVE_TOLERANCE)


def run_portfolio_backtest(batches, *, spec: PortfolioSpec = PORTFOLIO_SPEC_V1,
                           execution_spec: ExecutionSpec = EXECUTION_SPEC_V1,
                           periods_per_year: int = 8760,
                           experiment_type: str = "exploratory"
                           ) -> PortfolioBacktestResult:
    """Walk a sequence of (frame, target_set) in time order.

    ``target_set`` may be None: a timestamp can be a pure valuation with no
    rebalance. A timestamp where an open position cannot be priced produces no
    snapshot at all and is reported, rather than being marked at a stale
    price.
    """
    state = initial_portfolio_state(spec)
    fills: list[PortfolioFill] = []
    snapshots: list[PortfolioSnapshot] = []
    unavailable: list[str] = []
    scaled: list[str] = []
    target_hashes: list[str] = []
    rebalances = 0

    for frame, target_set in batches:
        try:
            if target_set is None:
                state = mark_to_market(state, frame)
            else:
                if target_set.timestamp != frame.timestamp:
                    raise PortfolioError(
                        f"batch timestamp {target_set.timestamp} does not match "
                        f"frame {frame.timestamp}")
                outcome = rebalance(state, target_set=target_set, frame=frame,
                                    spec=spec, execution_spec=execution_spec)
                state = outcome.state
                fills.extend(outcome.fills)
                target_hashes.append(target_set.target_set_hash)
                rebalances += 1
                if outcome.scaled:
                    scaled.append(frame.timestamp)
        except PortfolioValuationUnavailable:
            unavailable.append(frame.timestamp)
            continue
        snapshots.append(_snapshot(state))

    net = _metrics(tuple(snapshots), tuple(fills), spec=spec,
                   rebalances=rebalances, periods_per_year=periods_per_year)
    gross = _gross_view(net)
    return PortfolioBacktestResult(
        spec=spec, experiment_type=experiment_type, confirmatory=False,
        live_execution=False, cost_model="synthetic",
        instruments=tuple(sorted({position.instrument_id
                                  for position in state.positions})),
        metrics=net, gross_metrics=gross,
        attribution=_attribution(state, tuple(fills), tuple(snapshots)),
        fills=tuple(fills), equity_curve=tuple(snapshots),
        target_set_hashes=tuple(target_hashes),
        unavailable_valuations=tuple(unavailable), scaled_batches=tuple(scaled))


def _gross_view(net: PortfolioMetrics) -> PortfolioMetrics:
    """The same target stream with execution costs removed from the books.

    Not a second strategy and not a second run: identical decisions, identical
    fills, costs taken out of the accounting so the reader can see what the
    execution model consumed.
    """
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        return PortfolioMetrics(
            initial_equity=net.initial_equity,
            final_equity=net.final_equity + net.total_execution_cost,
            gross_pnl=net.gross_pnl, net_pnl=net.gross_pnl,
            gross_return=net.gross_return, net_return=net.gross_return,
            total_fees=_ZERO, total_slippage_cost=_ZERO,
            total_execution_cost=_ZERO,
            portfolio_turnover=net.portfolio_turnover,
            max_drawdown=net.max_drawdown,
            annualized_sharpe=net.annualized_sharpe,
            average_gross_exposure=net.average_gross_exposure,
            average_abs_net_exposure=net.average_abs_net_exposure,
            max_observed_gross_exposure=net.max_observed_gross_exposure,
            fill_count=net.fill_count, rebalance_count=net.rebalance_count)
