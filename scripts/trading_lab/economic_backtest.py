"""Phase 5C: turn frozen position targets into a simulated portfolio.

This is the first layer in HyprL that produces money-shaped numbers, so it is
also the first layer that can flatter itself. Three rules keep it honest.

**A decision cannot be executed at a price it helped to observe.** The features
behind a prediction at bar T use the *whole* of candle T, including its close.
So a target derived from candle T is not executable at `open[T]` -- that price
is already in the past when the candle finishes. The frozen fill policy is the
open of the *next contiguous* bar, and a test asserts the same-bar fill is
impossible rather than merely unused.

**Costs are a contract, not a discovery.** 10 bps of fee and 5 bps of slippage
were fixed before the first simulated fill. They are a synthetic infrastructure
model: they are not any exchange's schedule, they were not tuned, and they are
not claimed to reproduce a real account. Slippage always moves the price
against the trade, deterministically, in both directions.

**A gap expires a target; it does not liquidate a position.** If the bar after
a decision is missing, that target is stale and is dropped -- executing it
hours later would be trading on an intention the model never had. But closing a
position *before* a gap would use knowledge that the gap was coming. Existing
positions are carried across and revalued when the market reappears.

What this engine cannot do is manufacture predictive information. V1 and V2
both measured rank correlations near zero; an execution simulator applied to a
signal with no edge produces, at best, an honest picture of costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS

EXECUTION_SCHEMA_VERSION = "trading-lab.execution.v1"
ECONOMIC_BACKTEST_SCHEMA_VERSION = "trading-lab.economic-backtest.v1"
ECONOMIC_RESULT_SCHEMA_VERSION = "trading-lab.economic-backtest-result.v1"
ECONOMIC_PRECISION = 34

INSTRUMENT_MODEL_V1 = "synthetic-linear-usd-notional-v1"
FILL_POLICY_V1 = "next-contiguous-bar-open-after-decision-v1"
MARK_POLICY_V1 = "next-observable-open-v1"
FINAL_LIQUIDATION_POLICY_V1 = "next-observable-open-after-last-fill-v1"
TARGET_QUANTITY_BASIS_V1 = "reference-market-open-v1"

FEE_RATE_V1 = Decimal("0.0010")          # 10 bps per fill, on absolute notional
SLIPPAGE_RATE_V1 = Decimal("0.0005")     # 5 bps, always against the trade
INITIAL_EQUITY_V1 = Decimal("100000")
CURRENCY_V1 = "USD"

# Stated so no reader can mistake the cost model for an account statement.
EXECUTION_COST_V1_IS_NOT_OPTIMIZED = True
EXECUTION_COST_V1_IS_NOT_EXCHANGE_ACCOUNT_SPECIFIC = True

PERIODS_PER_YEAR = 8760                  # crypto trades 24/7; hourly bars
MAX_BACKTEST_BARS = 1_000_000
_ZERO = Decimal(0)
_ONE = Decimal(1)


class EconomicBacktestError(RuntimeError):
    """Raised when a simulation cannot be produced honestly."""


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _require_decimal(value: object, *, field: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise EconomicBacktestError(f"{field} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise EconomicBacktestError(f"{field} must be finite, got {value}")
    return value


def _parse(value: str, *, field: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise EconomicBacktestError(f"{field} is not ISO-8601: {value!r}") from error
    if moment.tzinfo is None:
        raise EconomicBacktestError(f"{field} must carry a timezone: {value!r}")
    return moment


# --- the frozen execution contract ---------------------------------------


@dataclass(frozen=True)
class ExecutionSpec:
    """How an intention becomes a fill. Frozen before the first simulated trade."""

    protocol_version: str = EXECUTION_SCHEMA_VERSION
    instrument_model: str = INSTRUMENT_MODEL_V1
    fill_policy: str = FILL_POLICY_V1
    mark_policy: str = MARK_POLICY_V1
    final_liquidation_policy: str = FINAL_LIQUIDATION_POLICY_V1
    target_quantity_basis: str = TARGET_QUANTITY_BASIS_V1
    fee_rate: Decimal = FEE_RATE_V1
    slippage_rate: Decimal = SLIPPAGE_RATE_V1
    initial_equity: Decimal = INITIAL_EQUITY_V1
    currency: str = CURRENCY_V1
    allow_long: bool = True
    allow_short: bool = True
    funding_rate: Decimal = _ZERO
    borrow_rate: Decimal = _ZERO

    def validate(self) -> None:
        for field in ("fee_rate", "slippage_rate", "funding_rate", "borrow_rate"):
            value = _require_decimal(getattr(self, field), field=field)
            if value < 0:
                raise EconomicBacktestError(f"{field} must not be negative, got {value}")
        equity = _require_decimal(self.initial_equity, field="initial_equity")
        if equity <= 0:
            raise EconomicBacktestError("initial_equity must be strictly positive")
        if self.funding_rate != 0 or self.borrow_rate != 0:
            # V1 has frozen no accrual schedule, so a non-zero rate could only
            # mean an undefined rule.
            raise EconomicBacktestError(
                "execution spec v1 models no funding or borrow accrual; a later spec "
                "must freeze the accrual basis and schedule first")
        for field, expected in (("fill_policy", FILL_POLICY_V1),
                                ("mark_policy", MARK_POLICY_V1),
                                ("instrument_model", INSTRUMENT_MODEL_V1),
                                ("final_liquidation_policy", FINAL_LIQUIDATION_POLICY_V1),
                                ("target_quantity_basis", TARGET_QUANTITY_BASIS_V1)):
            if getattr(self, field) != expected:
                raise EconomicBacktestError(
                    f"unsupported {field} {getattr(self, field)!r} in execution spec v1")
        if not self.allow_long and not self.allow_short:
            raise EconomicBacktestError("an execution spec must allow at least one side")

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "instrument_model": self.instrument_model,
            "fill_policy": self.fill_policy,
            "mark_policy": self.mark_policy,
            "final_liquidation_policy": self.final_liquidation_policy,
            "target_quantity_basis": self.target_quantity_basis,
            "fee_rate": str(self.fee_rate),
            "slippage_rate": str(self.slippage_rate),
            "initial_equity": str(self.initial_equity),
            "currency": self.currency,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "funding_rate": str(self.funding_rate),
            "borrow_rate": str(self.borrow_rate),
            "cost_model": "synthetic",
            "optimized": not EXECUTION_COST_V1_IS_NOT_OPTIMIZED,
            "exchange_account_specific":
                not EXECUTION_COST_V1_IS_NOT_EXCHANGE_ACCOUNT_SPECIFIC,
        }

    @property
    def execution_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


EXECUTION_SPEC_V1 = ExecutionSpec()
# The same target stream, priced as if execution were free. Not a second
# strategy -- a second accounting view, used only to measure what costs took.
GROSS_EXECUTION_SPEC_V1 = ExecutionSpec(fee_rate=_ZERO, slippage_rate=_ZERO)


# --- immutable records ----------------------------------------------------


@dataclass(frozen=True)
class SimulatedFill:
    timestamp: str
    side: str
    reference_price: Decimal
    fill_price: Decimal
    quantity_delta: Decimal
    notional: Decimal
    fee: Decimal
    slippage_cost: Decimal
    position_after: Decimal
    cash_after: Decimal
    equity_after: Decimal
    source_position_target_hash: str

    def canonical(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "side": self.side,
            "reference_price": str(self.reference_price),
            "fill_price": str(self.fill_price),
            "quantity_delta": str(self.quantity_delta),
            "notional": str(self.notional),
            "fee": str(self.fee),
            "slippage_cost": str(self.slippage_cost),
            "position_after": str(self.position_after),
            "cash_after": str(self.cash_after),
            "equity_after": str(self.equity_after),
            "source_position_target_hash": self.source_position_target_hash,
        }

    @property
    def fill_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: str
    cash: Decimal
    position_quantity: Decimal
    mark_price: Decimal
    position_value: Decimal
    equity: Decimal
    target_exposure: Decimal
    realized_exposure: Decimal
    cumulative_fees: Decimal
    cumulative_slippage_cost: Decimal

    def canonical(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "cash": str(self.cash),
            "position_quantity": str(self.position_quantity),
            "mark_price": str(self.mark_price),
            "position_value": str(self.position_value),
            "equity": str(self.equity),
            "target_exposure": str(self.target_exposure),
            "realized_exposure": str(self.realized_exposure),
            "cumulative_fees": str(self.cumulative_fees),
            "cumulative_slippage_cost": str(self.cumulative_slippage_cost),
        }


@dataclass(frozen=True)
class ExpiredTarget:
    timestamp: str
    expected_fill_at: str
    reason: str
    source_position_target_hash: str

    def canonical(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "expected_fill_at": self.expected_fill_at,
            "reason": self.reason,
            "source_position_target_hash": self.source_position_target_hash,
        }


@dataclass(frozen=True)
class EconomicMetrics:
    initial_equity: Decimal
    final_equity: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    gross_return: Decimal
    net_return: Decimal
    total_fees: Decimal
    total_slippage_cost: Decimal
    total_execution_cost: Decimal
    turnover_ratio: Decimal
    max_drawdown: Decimal
    annualized_sharpe: Decimal | None
    fill_count: int
    rebalance_count: int
    expired_target_count: int
    average_abs_exposure: Decimal
    exposure_time_fraction: Decimal

    def canonical(self) -> dict[str, object]:
        return {
            "initial_equity": str(self.initial_equity),
            "final_equity": str(self.final_equity),
            "gross_pnl": str(self.gross_pnl),
            "net_pnl": str(self.net_pnl),
            "gross_return": str(self.gross_return),
            "net_return": str(self.net_return),
            "total_fees": str(self.total_fees),
            "total_slippage_cost": str(self.total_slippage_cost),
            "total_execution_cost": str(self.total_execution_cost),
            "turnover_ratio": str(self.turnover_ratio),
            "max_drawdown": str(self.max_drawdown),
            "annualized_sharpe": _text(self.annualized_sharpe),
            "periods_per_year": PERIODS_PER_YEAR,
            "fill_count": self.fill_count,
            "rebalance_count": self.rebalance_count,
            "expired_target_count": self.expired_target_count,
            "average_abs_exposure": str(self.average_abs_exposure),
            "exposure_time_fraction": str(self.exposure_time_fraction),
        }


# --- the simulator --------------------------------------------------------


@dataclass(frozen=True)
class PortfolioState:
    """Everything one fill needs to know, and nothing about time or schedule.

    Extracted from the backtest loop in Phase 5D so that shadow trading can
    reuse the *same* accounting instead of growing a second implementation of
    it. Two economic engines that agree today will disagree eventually, and
    the disagreement will be discovered in a number nobody can reproduce.
    """

    cash: Decimal
    quantity: Decimal
    cumulative_fees: Decimal = _ZERO
    cumulative_slippage_cost: Decimal = _ZERO
    turnover_sum: Decimal = _ZERO

    def equity_at(self, mark_price: Decimal) -> Decimal:
        with localcontext() as context:
            context.prec = ECONOMIC_PRECISION
            return self.cash + self.quantity * mark_price


def initial_portfolio_state(execution_spec: ExecutionSpec) -> PortfolioState:
    return PortfolioState(cash=execution_spec.initial_equity, quantity=_ZERO)


@dataclass(frozen=True)
class FillEffect:
    """What moving a quantity costs, independent of whose book it lands in.

    Deliberately says nothing about equity, exposure or position size. Those
    depend on the account doing the trading, and a single-product account and
    a shared multi-instrument portfolio compute them differently -- while the
    cost of moving N units at a price is the same in both. Splitting it here
    is what lets the portfolio engine reuse this arithmetic instead of
    growing a second copy that agrees today and drifts later.
    """

    side: str
    fill_price: Decimal
    notional: Decimal
    fee: Decimal
    slippage_cost: Decimal
    cash_delta: Decimal


def apply_quantity_delta(*, quantity_delta: Decimal, reference_price: Decimal,
                         execution_spec: ExecutionSpec) -> FillEffect:
    """Price a quantity change at an observable open, with costs against it.

    Slippage always moves the fill price the wrong way for the trader, and the
    fee is charged on the slipped notional -- the same convention the
    single-product engine has used since Phase 5C, extracted verbatim so the
    committed results still reproduce byte for byte.
    """
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        if quantity_delta > 0:
            fill_price = reference_price * (_ONE + execution_spec.slippage_rate)
            side = "buy"
        else:
            fill_price = reference_price * (_ONE - execution_spec.slippage_rate)
            side = "sell"
        notional = abs(quantity_delta) * fill_price
        fee = notional * execution_spec.fee_rate
        slippage_cost = abs(quantity_delta) * abs(fill_price - reference_price)
        cash_delta = -(quantity_delta * fill_price) - fee
        return FillEffect(side=side, fill_price=fill_price, notional=notional,
                          fee=fee, slippage_cost=slippage_cost,
                          cash_delta=cash_delta)


def apply_position_target(state: PortfolioState, *, target_exposure: Decimal,
                          reference_price: Decimal, timestamp: str,
                          execution_spec: ExecutionSpec,
                          source_position_target_hash: str):
    """Move to `target_exposure` of pre-trade equity at an observable open.

    Returns the new state and the fill, or the unchanged state and None when
    the position already matches the target. This is the single place where
    cash, fees and slippage are computed; both the backtest and the shadow
    engine call it.
    """
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        pre_trade_equity = state.cash + state.quantity * reference_price
        if pre_trade_equity <= 0:
            raise EconomicBacktestError(
                f"equity reached {pre_trade_equity} at {timestamp}; the synthetic "
                "account is insolvent and the simulation cannot continue")
        # Target quantity is priced at the OBSERVABLE open, so slippage stays a
        # separable execution cost rather than silently reshaping the intended
        # exposure.
        target_quantity = (target_exposure * pre_trade_equity) / reference_price
        delta = target_quantity - state.quantity
        if delta == 0:
            return state, None
        effect = apply_quantity_delta(quantity_delta=delta,
                                      reference_price=reference_price,
                                      execution_spec=execution_spec)
        fill_price = effect.fill_price
        side = effect.side
        notional = effect.notional
        fee = effect.fee
        slippage_cost = effect.slippage_cost
        cash = state.cash + effect.cash_delta
        quantity = state.quantity + delta
        moved = PortfolioState(
            cash=cash, quantity=quantity,
            cumulative_fees=state.cumulative_fees + fee,
            cumulative_slippage_cost=state.cumulative_slippage_cost + slippage_cost,
            turnover_sum=state.turnover_sum
            + abs(delta * reference_price) / pre_trade_equity)
        fill = SimulatedFill(
            timestamp=timestamp, side=side, reference_price=reference_price,
            fill_price=fill_price, quantity_delta=delta, notional=notional,
            fee=fee, slippage_cost=slippage_cost, position_after=quantity,
            cash_after=cash, equity_after=cash + quantity * reference_price,
            source_position_target_hash=source_position_target_hash)
        return moved, fill


def _bar_index(bars, *, timeframe: str):
    """Index observable bars by opening, refusing anything unusable."""
    if timeframe not in TIMEFRAME_DURATIONS:
        raise EconomicBacktestError(f"unsupported timeframe {timeframe!r}")
    entries = tuple(bars)
    if not entries:
        raise EconomicBacktestError("no market bars were provided")
    if len(entries) > MAX_BACKTEST_BARS:
        raise EconomicBacktestError(f"at most {MAX_BACKTEST_BARS} bars per simulation")
    opens: dict[str, Decimal] = {}
    order: list[str] = []
    previous = None
    for bar in entries:
        stamp = bar["bar_open_at"]
        moment = _parse(stamp, field="bar_open_at")
        if previous is not None and moment <= previous:
            raise EconomicBacktestError("market bars are not in ascending order")
        previous = moment
        price = _require_decimal(Decimal(bar["open"]), field=f"open at {stamp}")
        if price <= 0:
            raise EconomicBacktestError(f"open price at {stamp} must be positive")
        opens[stamp] = price
        order.append(stamp)
    return opens, tuple(order)


def simulate_targets(targets, bars, *, timeframe: str = "1h",
                     execution_spec: ExecutionSpec = EXECUTION_SPEC_V1):
    """Run one target stream through one execution contract.

    The whole causal argument lives in the first loop: a target stamped at bar
    T is scheduled onto bar T + one interval, and only if that exact bar is
    observable. Nothing else can ever place a fill.
    """
    execution_spec.validate()
    duration: timedelta = TIMEFRAME_DURATIONS[timeframe]
    opens, order = _bar_index(bars, timeframe=timeframe)
    entries = tuple(getattr(targets, "targets", targets))

    stamps = [target.timestamp for target in entries]
    if stamps != sorted(stamps):
        raise EconomicBacktestError("position targets are not in ascending order")
    if len(set(stamps)) != len(stamps):
        raise EconomicBacktestError("position targets contain duplicate timestamps")

    scheduled: dict[str, object] = {}
    expired: list[ExpiredTarget] = []
    for target in entries:
        decided_at = _parse(target.timestamp, field="target timestamp")
        available_at = decided_at + duration
        key = available_at.isoformat()
        if key not in opens:
            # The next bar never arrived. Executing later would trade on an
            # intention the model did not have at that later price.
            expired.append(ExpiredTarget(
                timestamp=target.timestamp, expected_fill_at=key,
                reason="no contiguous bar open after the decision",
                source_position_target_hash=target.position_target_hash))
            continue
        scheduled[key] = target

    if not scheduled:
        raise EconomicBacktestError("no position target became executable")

    first_fill_at = min(scheduled)
    last_fill_at = max(scheduled)
    tail = [stamp for stamp in order if stamp > last_fill_at]
    if not tail:
        raise EconomicBacktestError(
            "final liquidation price unavailable: the corpus ends at the last fill")
    liquidation_at = tail[0]

    cash = execution_spec.initial_equity
    quantity = _ZERO
    cumulative_fees = _ZERO
    cumulative_slippage = _ZERO
    turnover_sum = _ZERO
    fills: list[SimulatedFill] = []
    snapshots: list[PortfolioSnapshot] = []
    rebalance_count = 0
    current_target = _ZERO

    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        window = [stamp for stamp in order
                  if first_fill_at <= stamp <= liquidation_at]
        for stamp in window:
            reference = opens[stamp]

            def _trade(exposure: Decimal, source_hash: str):
                """Move to `exposure` of pre-trade equity at this bar's open."""
                nonlocal cash, quantity, cumulative_fees, cumulative_slippage
                nonlocal turnover_sum
                state, fill = apply_position_target(
                    PortfolioState(cash=cash, quantity=quantity,
                                   cumulative_fees=cumulative_fees,
                                   cumulative_slippage_cost=cumulative_slippage,
                                   turnover_sum=turnover_sum),
                    target_exposure=exposure, reference_price=reference,
                    timestamp=stamp, execution_spec=execution_spec,
                    source_position_target_hash=source_hash)
                if fill is None:
                    return
                cash = state.cash
                quantity = state.quantity
                cumulative_fees = state.cumulative_fees
                cumulative_slippage = state.cumulative_slippage_cost
                turnover_sum = state.turnover_sum
                fills.append(fill)

            if stamp == liquidation_at:
                current_target = _ZERO
                _trade(_ZERO, "final-liquidation")
            elif stamp in scheduled:
                target = scheduled[stamp]
                current_target = _require_decimal(
                    target.target_exposure, field="target_exposure")
                rebalance_count += 1
                _trade(current_target, target.position_target_hash)

            equity = cash + quantity * reference
            snapshots.append(PortfolioSnapshot(
                timestamp=stamp, cash=cash, position_quantity=quantity,
                mark_price=reference, position_value=quantity * reference,
                equity=equity, target_exposure=current_target,
                realized_exposure=(quantity * reference) / equity if equity else _ZERO,
                cumulative_fees=cumulative_fees,
                cumulative_slippage_cost=cumulative_slippage))

    return {
        "fills": tuple(fills),
        "snapshots": tuple(snapshots),
        "expired": tuple(expired),
        "rebalance_count": rebalance_count,
        "turnover_sum": turnover_sum,
        "first_fill_at": first_fill_at,
        "last_fill_at": last_fill_at,
        "liquidation_at": liquidation_at,
    }


# --- metrics --------------------------------------------------------------


def _max_drawdown(snapshots) -> Decimal:
    """Negative by convention: -0.15 means a 15 % fall from the running peak."""
    peak = None
    worst = _ZERO
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        for snapshot in snapshots:
            equity = snapshot.equity
            peak = equity if peak is None or equity > peak else peak
            if peak > 0:
                drawdown = equity / peak - _ONE
                if drawdown < worst:
                    worst = drawdown
    return worst


def annualized_sharpe(snapshots) -> Decimal | None:
    """Descriptive only. No significance is claimed, and none is testable here."""
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        returns = []
        for previous, current in zip(snapshots, snapshots[1:]):
            if previous.equity <= 0:
                return None
            returns.append(current.equity / previous.equity - _ONE)
        if len(returns) < 2:
            return None
        count = Decimal(len(returns))
        mean = sum(returns) / count
        variance = sum((value - mean) ** 2 for value in returns) / (count - _ONE)
        if variance <= 0:
            return None
        return (mean / variance.sqrt()) * Decimal(PERIODS_PER_YEAR).sqrt()


def _metrics(run, *, execution_spec: ExecutionSpec, gross_final_equity: Decimal
             ) -> EconomicMetrics:
    snapshots = run["snapshots"]
    initial = execution_spec.initial_equity
    final = snapshots[-1].equity
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        fees = snapshots[-1].cumulative_fees
        slippage = snapshots[-1].cumulative_slippage_cost
        exposures = [abs(snapshot.realized_exposure) for snapshot in snapshots]
        invested = sum(1 for snapshot in snapshots if snapshot.position_quantity != 0)
        return EconomicMetrics(
            initial_equity=initial,
            final_equity=final,
            gross_pnl=gross_final_equity - initial,
            net_pnl=final - initial,
            gross_return=gross_final_equity / initial - _ONE,
            net_return=final / initial - _ONE,
            total_fees=fees,
            total_slippage_cost=slippage,
            total_execution_cost=fees + slippage,
            turnover_ratio=run["turnover_sum"],
            max_drawdown=_max_drawdown(snapshots),
            annualized_sharpe=annualized_sharpe(snapshots),
            fill_count=len(run["fills"]),
            rebalance_count=run["rebalance_count"],
            expired_target_count=len(run["expired"]),
            average_abs_exposure=sum(exposures) / Decimal(len(exposures)),
            exposure_time_fraction=Decimal(invested) / Decimal(len(snapshots)),
        )


# --- identity -------------------------------------------------------------


@dataclass(frozen=True)
class EconomicBacktestSpec:
    """The DEFINITION of one economic simulation. Contains no result."""

    protocol_version: str
    product: str
    timeframe: str
    source_benchmark_protocol: str
    source_benchmark_spec_hash: str
    source_benchmark_results_hash: str
    signal_spec_hash: str
    risk_spec_hash: str
    execution_spec_hash: str
    market_corpus_spec_hash: str
    market_corpus_content_hash: str
    result_schema_version: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": ECONOMIC_BACKTEST_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "product": self.product,
            "timeframe": self.timeframe,
            "source_benchmark_protocol": self.source_benchmark_protocol,
            "source_benchmark_spec_hash": self.source_benchmark_spec_hash,
            "source_benchmark_results_hash": self.source_benchmark_results_hash,
            "signal_spec_hash": self.signal_spec_hash,
            "risk_spec_hash": self.risk_spec_hash,
            "execution_spec_hash": self.execution_spec_hash,
            "market_corpus_spec_hash": self.market_corpus_spec_hash,
            "market_corpus_content_hash": self.market_corpus_content_hash,
            "result_schema_version": self.result_schema_version,
        }

    @property
    def economic_backtest_spec_hash(self) -> str:
        return _sha256_canonical(self.canonical())


@dataclass(frozen=True)
class EconomicBacktestResult:
    """What the frozen contract produced. Explicitly exploratory."""

    spec: EconomicBacktestSpec
    experiment_type: str
    confirmatory: bool
    live_execution: bool
    cost_model: str
    execution_spec: dict[str, object]
    metrics: EconomicMetrics
    gross_metrics: EconomicMetrics
    fills: tuple[SimulatedFill, ...]
    equity_curve: tuple[PortfolioSnapshot, ...]
    expired_targets: tuple[ExpiredTarget, ...]
    position_target_series_hash: str
    signal_series_hash: str
    window: dict[str, str]

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": ECONOMIC_RESULT_SCHEMA_VERSION,
            "spec": self.spec.canonical(),
            "economic_backtest_spec_hash": self.spec.economic_backtest_spec_hash,
            "experiment_type": self.experiment_type,
            "confirmatory": self.confirmatory,
            "live_execution": self.live_execution,
            "cost_model": self.cost_model,
            "execution_spec": self.execution_spec,
            "metrics": self.metrics.canonical(),
            "gross_metrics": self.gross_metrics.canonical(),
            "fills": [fill.canonical() for fill in self.fills],
            "equity_curve": [point.canonical() for point in self.equity_curve],
            "expired_targets": [entry.canonical() for entry in self.expired_targets],
            "position_target_series_hash": self.position_target_series_hash,
            "signal_series_hash": self.signal_series_hash,
            "window": self.window,
        }

    @property
    def economic_results_hash(self) -> str:
        return _sha256_canonical(self.canonical())


# --- label isolation ------------------------------------------------------


class PredictionView:
    """A stored PredictionRecord, with its realised label made unreachable.

    Signal generation must never see `actual_forward_return`: the whole point
    of a backtest is that profit comes from prices the strategy could have
    traded, not from the label used to score the model. Making the attribute
    raise turns that from a convention into a property of the object.
    """

    __slots__ = ("bar_open_at", "prediction", "fold_index")

    def __init__(self, record: dict):
        self.bar_open_at = record["bar_open_at"]
        self.prediction = Decimal(record["prediction"])
        self.fold_index = record.get("fold_index")

    @property
    def actual_forward_return(self):
        raise AssertionError(
            f"signal generation read the realised label at {self.bar_open_at}")


def build_position_targets(records, *, model_spec_hash: str, fitted_hash: str,
                           benchmark_spec_hash: str, signal_spec=None, risk_spec=None):
    """Predictions -> signals -> targets, without ever touching a label."""
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1, generate_position_targets
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1, generate_signals

    signals = generate_signals(
        [PredictionView(record) for record in records],
        model_spec_hash=model_spec_hash, fitted_hash=fitted_hash,
        benchmark_spec_hash=benchmark_spec_hash,
        signal_spec=signal_spec or SIGNAL_SPEC_V1)
    targets = generate_position_targets(signals, risk_spec=risk_spec or RISK_SPEC_V1)
    return signals, targets


def run_economic_backtest(*, product: str, targets, bars, spec: EconomicBacktestSpec,
                          signal_series_hash: str, bars_product,
                          execution_spec: ExecutionSpec = EXECUTION_SPEC_V1,
                          timeframe: str = "1h") -> EconomicBacktestResult:
    """Simulate one target stream twice: with costs, and as if they were free.

    ``bars_product`` names the market the price series came from and is
    required, not optional. A PositionTargetSeries carries spec hashes and
    timestamps but no instrument, and a bar is six numbers and a timestamp, so
    nothing in either argument said which market it belonged to. One target
    stream labelled BTC-USD ran against ETH-priced bars and produced a
    complete, plausible, entirely wrong result -- different final equity, same
    confident label, no complaint anywhere.

    A default of None would leave that hole open for every caller that forgot
    the argument, which is the same shape of mistake as the one being fixed,
    so there is no default.
    """
    from scripts.trading_lab.identity import require_same_instrument

    require_same_instrument(product, bars_product,
                            context="economic backtest",
                            left_label="result product",
                            right_label="market bars")
    net = simulate_targets(targets, bars, timeframe=timeframe,
                           execution_spec=execution_spec)
    gross = simulate_targets(targets, bars, timeframe=timeframe,
                             execution_spec=ExecutionSpec(
                                 fee_rate=_ZERO, slippage_rate=_ZERO,
                                 initial_equity=execution_spec.initial_equity,
                                 currency=execution_spec.currency,
                                 allow_long=execution_spec.allow_long,
                                 allow_short=execution_spec.allow_short))
    gross_final = gross["snapshots"][-1].equity
    return EconomicBacktestResult(
        spec=spec,
        experiment_type="exploratory",
        confirmatory=False,
        live_execution=False,
        cost_model="synthetic",
        execution_spec=execution_spec.canonical(),
        metrics=_metrics(net, execution_spec=execution_spec,
                         gross_final_equity=gross_final),
        gross_metrics=_metrics(gross, execution_spec=execution_spec,
                               gross_final_equity=gross_final),
        fills=net["fills"],
        equity_curve=net["snapshots"],
        expired_targets=net["expired"],
        position_target_series_hash=targets.series_hash,
        signal_series_hash=signal_series_hash,
        window={"first_fill_at": net["first_fill_at"],
                "last_fill_at": net["last_fill_at"],
                "liquidation_at": net["liquidation_at"]},
    )
