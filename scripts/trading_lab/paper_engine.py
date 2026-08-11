"""Shadow trading: the live pipeline, with no money and no broker anywhere.

    closed candle -> causal features -> frozen shadow model -> Signal V1
                  -> Risk V1 -> simulated fill -> paper portfolio -> event log

Every contract below this line already existed and is reused verbatim: the
signal rule, the risk sizing, and — importantly — the fill accounting, which
comes from `economic_backtest.apply_position_target` rather than a second
implementation. Two economic engines that agree today will disagree eventually,
and the disagreement surfaces as a number nobody can reproduce.

**Execution timing is the one genuinely new thing.** A target derived from
candle T is priced, exactly as in the backtest, at the open of candle T+1h —
a price that exists at the instant the target does, so nothing is backdated.
But a system that only sees closed candles does not *observe* that open until
T+2h. So the fill is recorded an hour late while remaining priced correctly,
and `PaperExecutionSpec` states that latency instead of hiding it. A live
session also never liquidates: it is open-ended, unlike a backtest.

The protected-window guard is checked again here, at the top of the pipeline,
even though ingestion already refused. Defence in depth is cheap and the cost
of being wrong once is a spent holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.economic_backtest import (
    EXECUTION_SPEC_V1,
    PortfolioState,
    apply_position_target,
    initial_portfolio_state,
)
from scripts.trading_lab.market_dataset import INDICATOR_REGISTRY
from scripts.trading_lab.market_series import MARKET_SERIES_SCHEMA_VERSION, MarketSeries, SeriesPoint
from scripts.trading_lab.paper_event_store import (
    PaperEventStore,
    PaperEventStoreError,
    SNAPSHOT_EVERY_EVENTS,
)
from scripts.trading_lab.protected_holdout import (
    PROTECTED_WINDOW_V1,
    ProtectedHoldoutError,
    embargo_state,
    require_tradeable_now,
    require_unprotected_bar,
)
from scripts.trading_lab.real_benchmark_v2 import FEATURE_COLUMNS_V2, FEATURE_SET_V2
from scripts.trading_lab.risk_engine import RISK_SPEC_V1, generate_position_target
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1, generate_signal

PAPER_SESSION_SCHEMA_VERSION = "trading-lab.paper-session.v1"
PAPER_EXECUTION_SCHEMA_VERSION = "trading-lab.paper-execution.v1"
PAPER_INITIAL_EQUITY = Decimal("100000")
PAPER_WARMUP_BARS = 40          # the 26-period EMA dominates; 40 is comfortable


class PaperEngineError(RuntimeError):
    """Raised when the shadow pipeline cannot proceed safely."""


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class PaperExecutionSpec:
    """What a live shadow session can actually execute, and when it learns it."""

    protocol_version: str = PAPER_EXECUTION_SCHEMA_VERSION
    fill_price_policy: str = "next-contiguous-bar-open-after-decision-v1"
    fill_observation_policy: str = "recorded-when-the-fill-bar-closes-v1"
    mark_policy: str = "latest-observed-open-v1"
    terminal_liquidation: bool = False
    fee_rate: Decimal = EXECUTION_SPEC_V1.fee_rate
    slippage_rate: Decimal = EXECUTION_SPEC_V1.slippage_rate
    initial_equity: Decimal = PAPER_INITIAL_EQUITY
    currency: str = EXECUTION_SPEC_V1.currency
    cost_model: str = "synthetic"

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": PAPER_EXECUTION_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "fill_price_policy": self.fill_price_policy,
            "fill_observation_policy": self.fill_observation_policy,
            "mark_policy": self.mark_policy,
            "terminal_liquidation": self.terminal_liquidation,
            "fee_rate": str(self.fee_rate),
            "slippage_rate": str(self.slippage_rate),
            "initial_equity": str(self.initial_equity),
            "currency": self.currency,
            "cost_model": self.cost_model,
            "backtest_execution_spec_hash": EXECUTION_SPEC_V1.execution_spec_hash,
            "differs_from_backtest": [
                "a live session never liquidates a terminal position",
                "a fill is recorded one bar after the price it is filled at becomes "
                "observable, so paper and backtest timelines are not directly comparable",
            ],
        }

    @property
    def paper_execution_spec_hash(self) -> str:
        return _sha256(self.canonical())

    def as_backtest_spec(self):
        """The cost contract, in the shape the shared accounting expects."""
        from dataclasses import replace
        return replace(EXECUTION_SPEC_V1, fee_rate=self.fee_rate,
                       slippage_rate=self.slippage_rate,
                       initial_equity=self.initial_equity, currency=self.currency)


PAPER_EXECUTION_SPEC_V1 = PaperExecutionSpec()


@dataclass(frozen=True)
class PaperSessionSpec:
    products: tuple[str, ...]
    timeframe: str
    paper_model_spec_hash: str
    model_fitted_hashes: tuple[tuple[str, str], ...]
    signal_spec_hash: str
    risk_spec_hash: str
    paper_execution_spec_hash: str
    protected_holdout_hash: str
    initial_equity: str

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": PAPER_SESSION_SCHEMA_VERSION,
            "products": list(self.products),
            "timeframe": self.timeframe,
            "paper_model_spec_hash": self.paper_model_spec_hash,
            "model_fitted_hashes": [list(pair) for pair in self.model_fitted_hashes],
            "signal_spec_hash": self.signal_spec_hash,
            "risk_spec_hash": self.risk_spec_hash,
            "paper_execution_spec_hash": self.paper_execution_spec_hash,
            "protected_holdout_hash": self.protected_holdout_hash,
            "initial_equity": self.initial_equity,
            "real_money": False,
            "broker_connected": False,
            "shadow_mode": True,
        }

    @property
    def session_spec_hash(self) -> str:
        return _sha256(self.canonical())


def build_session_spec(models: dict, *, products, timeframe: str = "1h",
                       execution_spec: PaperExecutionSpec = PAPER_EXECUTION_SPEC_V1,
                       window=PROTECTED_WINDOW_V1) -> PaperSessionSpec:
    from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
    fitted = tuple(sorted(
        (product, models[product].fitted.fitted_model_hash) for product in products))
    return PaperSessionSpec(
        products=tuple(products), timeframe=timeframe,
        paper_model_spec_hash=PAPER_MODEL_SPEC_V1.paper_model_spec_hash,
        model_fitted_hashes=fitted,
        signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
        risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
        paper_execution_spec_hash=execution_spec.paper_execution_spec_hash,
        protected_holdout_hash=window.holdout_hash,
        initial_equity=str(execution_spec.initial_equity))


def series_from_rows(rows, *, product: str, timeframe: str = "1h") -> MarketSeries:
    """An in-memory series for feature computation. No snapshot is invented."""
    points = tuple(
        SeriesPoint(
            bar_open_at=row["bar_open_at"], open=Decimal(row["open"]),
            high=Decimal(row["high"]), low=Decimal(row["low"]),
            close=Decimal(row["close"]), volume=Decimal(row["volume"]),
            content_sha256=_sha256({k: row[k] for k in (
                "bar_open_at", "open", "high", "low", "close", "volume")}))
        for row in rows)
    duration = TIMEFRAME_DURATIONS[timeframe]
    missing = ()
    if points:
        first = datetime.fromisoformat(points[0].bar_open_at)
        last = datetime.fromisoformat(points[-1].bar_open_at)
        present = {point.bar_open_at for point in points}
        span = int((last - first) / duration) + 1
        missing = tuple(_iso(first + duration * index) for index in range(span)
                        if _iso(first + duration * index) not in present)
    return MarketSeries(
        schema_version=MARKET_SERIES_SCHEMA_VERSION,
        snapshot_id="paper-live-series", snapshot_request_id="paper-live-series",
        entries_content_hash=_sha256([point.content_sha256 for point in points]),
        provider="coinbase_exchange_rest", product_id=product, timeframe=timeframe,
        range_start=points[0].bar_open_at if points else "",
        range_end=points[-1].bar_open_at if points else "",
        as_of=points[-1].bar_open_at if points else "",
        points=points, missing_openings=missing)


def causal_feature_vector(series: MarketSeries) -> tuple[tuple[str, Decimal | None], ...]:
    """The V2 feature row for the LAST point, computed from the prefix only."""
    values = []
    for feature in FEATURE_SET_V2:
        function = INDICATOR_REGISTRY[feature.indicator]
        result = function(series, **dict(feature.parameters))
        values.append((feature.column, result.values[-1]))
    return tuple(values)


@dataclass(frozen=True)
class PaperProductState:
    product: str
    portfolio: PortfolioState
    last_bar: str | None
    last_mark: Decimal | None
    fill_count: int
    pending_target: dict | None

    def payload(self) -> dict:
        return {
            "product": self.product,
            "cash": str(self.portfolio.cash),
            "position_quantity": str(self.portfolio.quantity),
            "cumulative_fees": str(self.portfolio.cumulative_fees),
            "cumulative_slippage_cost": str(self.portfolio.cumulative_slippage_cost),
            "turnover_sum": str(self.portfolio.turnover_sum),
            "last_bar": self.last_bar,
            "last_mark": None if self.last_mark is None else str(self.last_mark),
            "fill_count": self.fill_count,
            "pending_target": self.pending_target,
        }

    def equity(self) -> Decimal | None:
        if self.last_mark is None:
            return self.portfolio.cash
        return self.portfolio.equity_at(self.last_mark)


class PaperEngine:
    """Orchestrates one shadow session over one or more products."""

    def __init__(self, *, store: PaperEventStore, models: dict, session_id: str,
                 session_spec: PaperSessionSpec,
                 execution_spec: PaperExecutionSpec = PAPER_EXECUTION_SPEC_V1,
                 window=PROTECTED_WINDOW_V1, timeframe: str = "1h"):
        self.store = store
        self.models = models
        self.session_id = session_id
        self.session_spec = session_spec
        self.execution_spec = execution_spec
        self.window = window
        self.timeframe = timeframe
        self._bars: dict[str, list[dict]] = {}
        self._state: dict[str, PaperProductState] = {}
        self._status: dict[str, str] = {}

    # --- lifecycle -------------------------------------------------------

    def seed_history(self, product: str, rows) -> None:
        """Warm-up bars from the already-spent corpus. Never live, never protected."""
        for row in rows:
            require_unprotected_bar(product, row["bar_open_at"], window=self.window)
        self._bars[product] = [dict(row) for row in rows]

    def start(self, *, now) -> None:
        from scripts.trading_lab.live_market import LiveMarketStatus
        self.store.append(
            session_id=self.session_id, event_type="SESSION_STARTED",
            event_at=str(now), natural_key="session",
            payload={"session_spec": self.session_spec.canonical(),
                     "session_spec_hash": self.session_spec.session_spec_hash,
                     "paper_execution_spec": self.execution_spec.canonical()})
        for product in self.session_spec.products:
            self._state.setdefault(product, PaperProductState(
                product=product,
                portfolio=initial_portfolio_state(self.execution_spec.as_backtest_spec()),
                last_bar=None, last_mark=None, fill_count=0, pending_target=None))
            self._bars.setdefault(product, [])
            state = embargo_state(product, now=now, window=self.window)
            if state["embargoed"]:
                self._status[product] = LiveMarketStatus.EMBARGOED
                self._record_embargo(product, now=now, reason=state["reason"])
            else:
                self._status[product] = LiveMarketStatus.RUNNING

    def stop(self, *, now) -> None:
        self.store.append(session_id=self.session_id, event_type="SESSION_STOPPED",
                          event_at=str(now), natural_key="session",
                          payload={"products": list(self.session_spec.products)})
        for product in self.session_spec.products:
            from scripts.trading_lab.live_market import LiveMarketStatus
            self._status[product] = LiveMarketStatus.STOPPED

    def status(self, product: str) -> str:
        from scripts.trading_lab.live_market import LiveMarketStatus
        return self._status.get(product, LiveMarketStatus.STOPPED)

    def state(self, product: str) -> PaperProductState:
        return self._state[product]

    def _record_embargo(self, product: str, *, now, reason: str) -> None:
        """An embargo event carries the boundary, never a protected price."""
        try:
            self.store.append(
                session_id=self.session_id,
                event_type="PROTECTED_HOLDOUT_BOUNDARY_REACHED",
                event_at=str(now), product=product, natural_key=self.window.start,
                payload={"reason": reason, "start": self.window.start,
                         "end": self.window.end,
                         "holdout_id": self.window.holdout_id,
                         "holdout_hash": self.window.holdout_hash})
        except PaperEventStoreError:
            pass    # already recorded for this session; the boundary is idempotent

    # --- the pipeline ----------------------------------------------------

    def ingest_candle(self, product: str, row: dict, *, now) -> dict:
        """Run one closed candle through the whole chain. Idempotent per candle."""
        from scripts.trading_lab.live_market import LiveMarketStatus
        opening = row["bar_open_at"]

        # Defence in depth: ingestion already refused, and so does this.
        try:
            require_tradeable_now(product, now=now, window=self.window)
            require_unprotected_bar(product, opening, window=self.window)
        except ProtectedHoldoutError as error:
            self._status[product] = LiveMarketStatus.EMBARGOED
            self._record_embargo(product, now=now, reason=str(error))
            raise

        if self.store.has_event(session_id=self.session_id,
                                event_type="CANDLE_INGESTED",
                                product=product, natural_key=opening):
            return {"product": product, "bar_open_at": opening, "skipped": True,
                    "reason": "already processed in this session"}

        bars = self._bars.setdefault(product, [])
        if bars and opening <= bars[-1]["bar_open_at"]:
            raise PaperEngineError(
                f"{product}: candle {opening} does not follow {bars[-1]['bar_open_at']}")
        duration = TIMEFRAME_DURATIONS[self.timeframe]
        if bars:
            expected = _iso(datetime.fromisoformat(bars[-1]["bar_open_at"]) + duration)
            if opening != expected:
                self.store.append(
                    session_id=self.session_id, event_type="GAP_DETECTED",
                    event_at=str(now), product=product, natural_key=opening,
                    payload={"expected": expected, "observed": opening})

        bars.append(dict(row))
        self.store.append(session_id=self.session_id, event_type="CANDLE_INGESTED",
                          event_at=str(now), product=product, natural_key=opening,
                          payload={"bar": dict(row)})

        outcome = {"product": product, "bar_open_at": opening, "skipped": False,
                   "filled": False, "predicted": False}
        # (1) execute the target decided at the previous candle, at THIS open
        outcome.update(self._execute_pending(product, row=row, now=now))
        # (2) then derive this candle's own decision
        outcome.update(self._decide(product, opening=opening, now=now))
        self._maybe_snapshot(product)
        return outcome

    def _execute_pending(self, product: str, *, row: dict, now) -> dict:
        state = self._state[product]
        pending = state.pending_target
        opening = row["bar_open_at"]
        reference = Decimal(row["open"])
        if pending is None:
            self._state[product] = PaperProductState(
                product=product, portfolio=state.portfolio, last_bar=opening,
                last_mark=reference, fill_count=state.fill_count, pending_target=None)
            self._write_portfolio(product, opening=opening, now=now, mark=reference)
            return {"filled": False}
        duration = TIMEFRAME_DURATIONS[self.timeframe]
        expected = _iso(datetime.fromisoformat(pending["timestamp"]) + duration)
        if expected != opening:
            # The bar the target was meant for never arrived; it is stale.
            self.store.append(
                session_id=self.session_id, event_type="GAP_DETECTED",
                event_at=str(now), product=product,
                natural_key=f"expired:{pending['timestamp']}",
                payload={"reason": "target expired: no contiguous bar after the decision",
                         "target_timestamp": pending["timestamp"],
                         "expected_fill_at": expected})
            self._state[product] = PaperProductState(
                product=product, portfolio=state.portfolio, last_bar=opening,
                last_mark=reference, fill_count=state.fill_count, pending_target=None)
            self._write_portfolio(product, opening=opening, now=now, mark=reference)
            return {"filled": False, "expired": True}

        portfolio, fill = apply_position_target(
            state.portfolio, target_exposure=Decimal(pending["target_exposure"]),
            reference_price=reference, timestamp=opening,
            execution_spec=self.execution_spec.as_backtest_spec(),
            source_position_target_hash=pending["position_target_hash"])
        fills = state.fill_count
        if fill is not None:
            fills += 1
            self.store.append(
                session_id=self.session_id, event_type="SIMULATED_FILL",
                event_at=str(now), product=product, natural_key=opening,
                payload={"fill": fill.canonical(), "fill_hash": fill.fill_hash,
                         "decided_at": pending["timestamp"],
                         "observation_policy":
                             self.execution_spec.fill_observation_policy})
        self._state[product] = PaperProductState(
            product=product, portfolio=portfolio, last_bar=opening,
            last_mark=reference, fill_count=fills, pending_target=None)
        self._write_portfolio(product, opening=opening, now=now, mark=reference)
        return {"filled": fill is not None}

    def _decide(self, product: str, *, opening: str, now) -> dict:
        bars = self._bars[product]
        if len(bars) < PAPER_WARMUP_BARS:
            return {"predicted": False, "reason": "WARMING_UP"}
        series = series_from_rows(bars, product=product, timeframe=self.timeframe)
        features = causal_feature_vector(series)
        if any(value is None for _, value in features):
            return {"predicted": False, "reason": "WARMING_UP"}
        feature_hash = _sha256([[name, str(value)] for name, value in features])
        self.store.append(
            session_id=self.session_id, event_type="FEATURES_READY",
            event_at=str(now), product=product, natural_key=opening,
            payload={"feature_columns": list(FEATURE_COLUMNS_V2),
                     "feature_vector_hash": feature_hash})

        model = self.models[product]

        class _Row:
            bar_open_at = opening
            usable = True
            def __init__(self, values): self.features = values
        prediction = model.predict([_Row(features)])[0]
        duration = TIMEFRAME_DURATIONS[self.timeframe]
        available_at = _iso(datetime.fromisoformat(opening) + duration)
        prediction_payload = {
            "product": product, "bar_open_at": opening,
            "decision_available_at": available_at, "prediction": str(prediction),
            "paper_model_spec_hash": self.session_spec.paper_model_spec_hash,
            "fitted_hash": model.fitted.fitted_model_hash,
            "feature_vector_hash": feature_hash,
        }
        prediction_payload["prediction_hash"] = _sha256(prediction_payload)
        self.store.append(
            session_id=self.session_id, event_type="PREDICTION_CREATED",
            event_at=str(now), product=product, natural_key=opening,
            payload=prediction_payload)

        signal = generate_signal(
            timestamp=opening, prediction=prediction,
            model_spec_hash=model.model_spec_hash,
            fitted_hash=model.fitted.fitted_model_hash,
            benchmark_spec_hash=self.session_spec.paper_model_spec_hash,
            signal_spec=SIGNAL_SPEC_V1)
        self.store.append(
            session_id=self.session_id, event_type="SIGNAL_CREATED",
            event_at=str(now), product=product, natural_key=opening,
            payload={"signal": signal.canonical(),
                     "signal_decision_hash": signal.decision_hash})

        target = generate_position_target(signal=signal, risk_spec=RISK_SPEC_V1)
        self.store.append(
            session_id=self.session_id, event_type="POSITION_TARGET_CREATED",
            event_at=str(now), product=product, natural_key=opening,
            payload={"target": target.canonical(),
                     "position_target_hash": target.position_target_hash})

        state = self._state[product]
        self._state[product] = PaperProductState(
            product=product, portfolio=state.portfolio, last_bar=state.last_bar,
            last_mark=state.last_mark, fill_count=state.fill_count,
            pending_target={"timestamp": opening,
                            "target_exposure": str(target.target_exposure),
                            "position_target_hash": target.position_target_hash})
        return {"predicted": True, "prediction": str(prediction),
                "side": signal.direction, "target_exposure": str(target.target_exposure)}

    def _write_portfolio(self, product: str, *, opening: str, now, mark: Decimal) -> None:
        state = self._state[product]
        equity = state.portfolio.equity_at(mark)
        self.store.append(
            session_id=self.session_id, event_type="PORTFOLIO_SNAPSHOT",
            event_at=str(now), product=product, natural_key=opening,
            payload={"timestamp": opening, "cash": str(state.portfolio.cash),
                     "position_quantity": str(state.portfolio.quantity),
                     "mark_price": str(mark),
                     "position_value": str(state.portfolio.quantity * mark),
                     "equity": str(equity),
                     "cumulative_fees": str(state.portfolio.cumulative_fees),
                     "cumulative_slippage_cost":
                         str(state.portfolio.cumulative_slippage_cost),
                     "fill_count": state.fill_count})

    def _maybe_snapshot(self, product: str) -> None:
        # Measure the distance from this product's last snapshot rather than
        # testing the running total for divisibility: a candle appends several
        # events at once, so an exact multiple is never observed.
        events = self.store.latest_events(session_id=self.session_id, limit=1)
        if not events:
            return
        head = events[-1]
        previous = self.store.latest_snapshot(
            session_id=self.session_id, product=product)
        since = head.event_id - (previous["last_event_id"] if previous else 0)
        if since < SNAPSHOT_EVERY_EVENTS:
            return
        self.store.write_snapshot(
            session_id=self.session_id, product=product, last_event_id=head.event_id,
            last_event_hash=head.event_hash, state=self._state[product].payload())

    # --- restart ---------------------------------------------------------

    def restore(self, *, history: dict | None = None) -> None:
        """Rebuild state from the verified log. A restart replays nothing twice."""
        self.store.verify_chain(session_id=self.session_id)
        for product in self.session_spec.products:
            self._state[product] = PaperProductState(
                product=product,
                portfolio=initial_portfolio_state(self.execution_spec.as_backtest_spec()),
                last_bar=None, last_mark=None, fill_count=0, pending_target=None)
            self._bars[product] = [dict(row) for row in (history or {}).get(product, [])]

        after = 0
        while True:
            events = self.store.events(session_id=self.session_id,
                                       after_event_id=after, limit=1000)
            if not events:
                break
            for event in events:
                after = event.event_id
                self._replay(event)
        return None

    def _replay(self, event) -> None:
        product = event.product
        if product is None or product not in self._state:
            return
        state = self._state[product]
        if event.event_type == "CANDLE_INGESTED":
            self._bars[product].append(dict(event.payload["bar"]))
        elif event.event_type == "SIMULATED_FILL":
            fill = event.payload["fill"]
            portfolio = PortfolioState(
                cash=Decimal(fill["cash_after"]),
                quantity=Decimal(fill["position_after"]),
                cumulative_fees=state.portfolio.cumulative_fees + Decimal(fill["fee"]),
                cumulative_slippage_cost=state.portfolio.cumulative_slippage_cost
                + Decimal(fill["slippage_cost"]),
                turnover_sum=state.portfolio.turnover_sum)
            self._state[product] = PaperProductState(
                product=product, portfolio=portfolio, last_bar=fill["timestamp"],
                last_mark=Decimal(fill["reference_price"]),
                fill_count=state.fill_count + 1, pending_target=None)
        elif event.event_type == "PORTFOLIO_SNAPSHOT":
            payload = event.payload
            portfolio = PortfolioState(
                cash=Decimal(payload["cash"]),
                quantity=Decimal(payload["position_quantity"]),
                cumulative_fees=Decimal(payload["cumulative_fees"]),
                cumulative_slippage_cost=Decimal(payload["cumulative_slippage_cost"]),
                turnover_sum=state.portfolio.turnover_sum)
            self._state[product] = PaperProductState(
                product=product, portfolio=portfolio,
                last_bar=payload["timestamp"], last_mark=Decimal(payload["mark_price"]),
                fill_count=payload["fill_count"], pending_target=state.pending_target)
        elif event.event_type == "POSITION_TARGET_CREATED":
            target = event.payload["target"]
            current = self._state[product]
            self._state[product] = PaperProductState(
                product=product, portfolio=current.portfolio, last_bar=current.last_bar,
                last_mark=current.last_mark, fill_count=current.fill_count,
                pending_target={"timestamp": target["timestamp"],
                                "target_exposure": target["target_exposure"],
                                "position_target_hash":
                                    event.payload["position_target_hash"]})
