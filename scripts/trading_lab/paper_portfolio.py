"""The shadow portfolio, live: several instruments, one pot of capital.

No real money, no broker, no exchange key. Nothing here places an order.

The problem this module exists to solve is arrival order. Live candles do not
arrive in a defined sequence -- BTC may land before ETH, or after, or a
minute later -- and the naive shape is to rebalance whichever one arrived.
That would move cash and equity, so the second instrument would be sized from
a different number, and the portfolio would depend on network timing. Phase 6B
established that this is not a rounding difference but a different portfolio.

So arrival never triggers execution. A ``PendingPortfolioBatch`` collects each
instrument's target for one decision timestamp, and only when the batch is
complete does a single atomic rebalance run through the Phase 6B engine, from
one shared pre-trade equity. Arrival order affects nothing but the wall-clock
stamps on monitoring events.

FLAT is a target, not a missing one. An instrument whose signal is FLAT
produces an exposure of zero, which may well close an open position; treating
it as "not ready" would stall the batch forever on a market that is mostly
flat -- which this one is.

The causal boundary from Phase 5D is unchanged: a target decided on bar T is
priced at the open of bar T + one interval, and that price is only observable
once that bar closes. The fill is therefore recorded one bar later than it is
priced, and the specification says so rather than hiding it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.identity import resolve_instrument
from scripts.trading_lab.paper_engine import (
    PAPER_EXECUTION_SPEC_V1,
    PAPER_WARMUP_BARS,
    PaperExecutionSpec,
    causal_feature_vector,
    series_from_rows,
)
from scripts.trading_lab.paper_portfolio_store import (
    SNAPSHOT_EVERY_EVENTS,
    PaperPortfolioStore,
    PaperPortfolioStoreError,
)
from scripts.trading_lab.portfolio import (
    PORTFOLIO_SPEC_V1,
    InstrumentPositionTarget,
    PortfolioError,
    PortfolioIdentityMismatch,
    PortfolioMarketFrame,
    PortfolioSpec,
    PortfolioState,
    PortfolioTargetSet,
    PortfolioValuationUnavailable,
    initial_portfolio_state,
    mark_to_market,
    rebalance,
)
from scripts.trading_lab.protected_holdout import (
    PROTECTED_WINDOW_V1,
    ProtectedHoldoutError,
    embargo_state,
    require_tradeable_now,
    require_unprotected_bar,
)
from scripts.trading_lab.real_benchmark_v2 import FEATURE_COLUMNS_V2
from scripts.trading_lab.risk_engine import RISK_SPEC_V1, generate_position_target
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1, generate_signal

PAPER_PORTFOLIO_SESSION_SCHEMA_VERSION = "trading-lab.paper-portfolio-session.v1"

_ZERO = Decimal(0)


class PaperPortfolioError(RuntimeError):
    """Raised when the shared shadow portfolio cannot proceed safely."""


class PortfolioStatus:
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    WAITING = "WAITING_FOR_PORTFOLIO_BATCH"
    EMBARGOED = "EMBARGOED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


# --- session specification -------------------------------------------------


@dataclass(frozen=True)
class PaperPortfolioSessionSpec:
    """Every frozen contract this session runs under, named once."""

    protocol_version: str
    instruments: tuple[str, ...]
    timeframe: str
    paper_model_spec_hash: str
    model_fitted_hashes: tuple[tuple[str, str], ...]
    signal_spec_hash: str
    risk_spec_hash: str
    paper_execution_spec_hash: str
    portfolio_spec_hash: str
    protected_holdout_hash: str
    initial_equity: str

    def canonical(self) -> dict:
        return {
            "schema_version": PAPER_PORTFOLIO_SESSION_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "instruments": list(self.instruments),
            "timeframe": self.timeframe,
            "paper_model_spec_hash": self.paper_model_spec_hash,
            "model_fitted_hashes": [list(pair) for pair in self.model_fitted_hashes],
            "signal_spec_hash": self.signal_spec_hash,
            "risk_spec_hash": self.risk_spec_hash,
            "paper_execution_spec_hash": self.paper_execution_spec_hash,
            "portfolio_spec_hash": self.portfolio_spec_hash,
            "protected_holdout_hash": self.protected_holdout_hash,
            "initial_equity": self.initial_equity,
            "shared_capital": True,
            "real_money": False,
            "broker_connected": False,
            "shadow_mode": True,
        }

    @property
    def session_spec_hash(self) -> str:
        return _sha256(self.canonical())


def build_portfolio_session_spec(models: dict, *, instruments=None,
                                 timeframe: str = "1h",
                                 execution_spec: PaperExecutionSpec = PAPER_EXECUTION_SPEC_V1,
                                 portfolio_spec: PortfolioSpec = PORTFOLIO_SPEC_V1,
                                 window=PROTECTED_WINDOW_V1
                                 ) -> PaperPortfolioSessionSpec:
    """Canonical identities, canonically ordered, whatever the caller passed."""
    from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1

    names = tuple(sorted(
        resolve_instrument(name, context="session instrument").canonical_id
        for name in (instruments if instruments is not None else models)))
    fitted = []
    for canonical in names:
        model = _model_for(models, canonical)
        fitted.append((canonical, model.fitted.fitted_model_hash))
    return PaperPortfolioSessionSpec(
        protocol_version=PAPER_PORTFOLIO_SESSION_SCHEMA_VERSION,
        instruments=names, timeframe=timeframe,
        paper_model_spec_hash=PAPER_MODEL_SPEC_V1.paper_model_spec_hash,
        model_fitted_hashes=tuple(sorted(fitted)),
        signal_spec_hash=SIGNAL_SPEC_V1.spec_hash,
        risk_spec_hash=RISK_SPEC_V1.risk_spec_hash,
        paper_execution_spec_hash=execution_spec.paper_execution_spec_hash,
        portfolio_spec_hash=portfolio_spec.portfolio_spec_hash,
        protected_holdout_hash=window.holdout_hash,
        initial_equity=str(portfolio_spec.initial_equity))


def _model_for(models: dict, canonical: str):
    """Find a model however its key is spelled, or refuse."""
    for key, model in models.items():
        try:
            if resolve_instrument(key, context="model key").canonical_id == canonical:
                return model
        except Exception:                            # unresolvable key
            continue
    raise PaperPortfolioError(f"no shadow model supplied for {canonical}")


# --- the pending batch -----------------------------------------------------


@dataclass
class PendingPortfolioBatch:
    """One decision timestamp, collecting targets until every instrument is in.

    Not executable until complete. This is the whole defence against arrival
    order deciding the portfolio.
    """

    decision_at: str
    required: tuple[str, ...]
    targets: dict = field(default_factory=dict)

    @property
    def ready_instruments(self) -> tuple[str, ...]:
        return tuple(sorted(self.targets))

    @property
    def missing_instruments(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.required) - set(self.targets)))

    @property
    def complete(self) -> bool:
        return not self.missing_instruments

    def payload(self) -> dict:
        return {
            "decision_at": self.decision_at,
            "required": list(self.required),
            "ready": list(self.ready_instruments),
            "missing": list(self.missing_instruments),
            "complete": self.complete,
        }


def portfolio_batch_id(*, session_spec_hash: str, decision_at: str,
                       targets: dict, frame_hash: str) -> str:
    """A deterministic identity for one rebalance.

    Derived from what the batch *is*, never from a UUID or a clock: a restart
    that re-derives the same batch must produce the same id, or the
    exactly-once constraint has nothing to match on.
    """
    return _sha256({
        "session_spec_hash": session_spec_hash,
        "decision_at": decision_at,
        "instruments": sorted(targets),
        "position_target_hashes": [
            [name, targets[name]["position_target_hash"]] for name in sorted(targets)],
        "market_frame_hash": frame_hash,
    })


# --- the runtime -----------------------------------------------------------


class PaperPortfolioEngine:
    """Live shadow trading over one shared portfolio.

    Owns no financial arithmetic. Allocation, the gross cap, quantity deltas,
    fees, slippage and cash all come from the Phase 6B engine; this class
    decides *when* a rebalance may happen and records what happened.
    """

    def __init__(self, *, store: PaperPortfolioStore, models: dict,
                 session_id: str, session_spec: PaperPortfolioSessionSpec,
                 portfolio_spec: PortfolioSpec = PORTFOLIO_SPEC_V1,
                 execution_spec: PaperExecutionSpec = PAPER_EXECUTION_SPEC_V1,
                 window=PROTECTED_WINDOW_V1):
        self.store = store
        self.session_id = session_id
        self.session_spec = session_spec
        self.portfolio_spec = portfolio_spec
        self.execution_spec = execution_spec
        self.window = window
        self.timeframe = session_spec.timeframe
        self.models = {name: _model_for(models, name)
                       for name in session_spec.instruments}
        self.state: PortfolioState = initial_portfolio_state(portfolio_spec)
        self._bars: dict[str, list] = {name: [] for name in session_spec.instruments}
        self._opens: dict[str, dict] = {name: {} for name in session_spec.instruments}
        self._pending: dict[str, PendingPortfolioBatch] = {}
        self._completed: set[str] = set()
        # batch_id -> decision timestamp, for rebalances that began but whose
        # snapshot never landed.
        self._started: dict[str, str] = {}
        self._status = PortfolioStatus.STOPPED
        self._fill_count = 0
        self._rebalance_count = 0

    # --- identity --------------------------------------------------------

    def _require_instrument(self, value) -> str:
        canonical = resolve_instrument(
            value, context="paper portfolio instrument").canonical_id
        if canonical not in self.session_spec.instruments:
            raise PortfolioIdentityMismatch(
                f"{canonical} is not one of this session's instruments "
                f"{list(self.session_spec.instruments)}")
        return canonical

    # --- lifecycle -------------------------------------------------------

    def seed_history(self, instrument, rows) -> None:
        canonical = self._require_instrument(instrument)
        for row in rows:
            require_unprotected_bar(canonical, row["bar_open_at"], window=self.window)
        self._bars[canonical] = [dict(row) for row in rows]
        for row in rows:
            self._opens[canonical][str(row["bar_open_at"])] = Decimal(str(row["open"]))

    def start(self, *, now) -> None:
        self.store.register_session(
            session_id=self.session_id, session_spec=self.session_spec.canonical(),
            session_spec_hash=self.session_spec.session_spec_hash,
            started_at=str(now))
        self.store.append(
            session_id=self.session_id, event_type="PORTFOLIO_SESSION_STARTED",
            event_at=str(now), natural_key="session",
            payload={"session_spec": self.session_spec.canonical(),
                     "session_spec_hash": self.session_spec.session_spec_hash})
        self._status = PortfolioStatus.RUNNING

    def stop(self, *, now) -> None:
        try:
            self.store.append(
                session_id=self.session_id, event_type="PORTFOLIO_SESSION_STOPPED",
                event_at=str(now), natural_key="session",
                payload={"fills": self._fill_count,
                         "rebalances": self._rebalance_count})
        except PaperPortfolioStoreError:
            pass                                     # already stopped is not a failure
        self._status = PortfolioStatus.STOPPED

    @property
    def status(self) -> str:
        return self._status

    # --- ingestion -------------------------------------------------------

    def ingest_candle(self, instrument, row: dict, *, now,
                      row_instrument=None) -> dict:
        """Take one closed candle. May complete a batch, or may not."""
        canonical = self._require_instrument(instrument)
        if row_instrument is not None:
            # The poller knows which market it fetched; a candle is six numbers
            # and passing one under another's name is caught by no schema.
            if self._require_instrument(row_instrument) != canonical:
                raise PortfolioIdentityMismatch(
                    f"candle came from {row_instrument} but was offered as "
                    f"{canonical}")
        opening = str(row["bar_open_at"])

        try:
            require_tradeable_now(canonical, now=now, window=self.window)
            require_unprotected_bar(canonical, opening, window=self.window)
        except ProtectedHoldoutError as error:
            self._record_embargo(canonical, now=now, reason=str(error))
            raise

        bars = self._bars[canonical]
        if bars and opening <= str(bars[-1]["bar_open_at"]):
            return {"instrument_id": canonical, "bar_open_at": opening,
                    "skipped": True, "reason": "already processed"}
        bars.append(dict(row))
        self._opens[canonical][opening] = Decimal(str(row["open"]))

        outcome = {"instrument_id": canonical, "bar_open_at": opening,
                   "skipped": False}
        outcome.update(self._decide(canonical, opening=opening, now=now))
        # This bar's OPEN is the execution price for the batch decided one
        # interval earlier, so its arrival is what can complete that batch.
        outcome.update(self._try_execute(decision_at=self._previous(opening),
                                         now=now))
        self._maybe_snapshot()
        return outcome

    def _previous(self, opening: str) -> str:
        duration = TIMEFRAME_DURATIONS[self.timeframe]
        return _iso(datetime.fromisoformat(opening) - duration)

    def _record_embargo(self, instrument, *, now, reason: str) -> None:
        self._status = PortfolioStatus.EMBARGOED
        try:
            self.store.append(
                session_id=self.session_id,
                event_type="PROTECTED_HOLDOUT_BOUNDARY_REACHED",
                event_at=str(now), instrument_id=instrument, natural_key="boundary",
                payload={"reason": reason,
                         "holdout_hash": self.window.holdout_hash,
                         "observed": False})
        except PaperPortfolioStoreError:
            pass                                     # recorded once is enough

    # --- decision --------------------------------------------------------

    def _decide(self, instrument: str, *, opening: str, now) -> dict:
        bars = self._bars[instrument]
        if len(bars) < PAPER_WARMUP_BARS:
            return {"ready": False, "reason": "WARMING_UP"}
        series = series_from_rows(bars, product=instrument.split(":")[-1],
                                  timeframe=self.timeframe)
        features = causal_feature_vector(series)
        if any(value is None for _, value in features):
            return {"ready": False, "reason": "WARMING_UP"}

        model = self.models[instrument]

        class _Row:
            bar_open_at = opening
            usable = True
            def __init__(self, values): self.features = values

        prediction = model.predict([_Row(features)])[0]
        signal = generate_signal(
            timestamp=opening, prediction=prediction,
            model_spec_hash=model.model_spec_hash,
            fitted_hash=model.fitted.fitted_model_hash,
            benchmark_spec_hash=self.session_spec.paper_model_spec_hash,
            signal_spec=SIGNAL_SPEC_V1)
        # FLAT is an exposure of zero, not an absent decision. Treating it as
        # missing would stall every batch on a market that is mostly flat.
        target = generate_position_target(signal=signal, risk_spec=RISK_SPEC_V1)

        batch = self._pending.get(opening)
        if batch is None:
            batch = PendingPortfolioBatch(
                decision_at=opening, required=self.session_spec.instruments)
            self._pending[opening] = batch
            self._append_once(
                event_type="PORTFOLIO_BATCH_OPENED", event_at=str(now),
                natural_key=opening, payload={"decision_at": opening,
                                              "required": list(batch.required)})

        entry = {
            "instrument_id": instrument,
            "prediction": str(prediction),
            "signal_direction": signal.direction,
            "signal_decision_hash": signal.decision_hash,
            "target_exposure": str(target.target_exposure),
            "position_target_hash": target.position_target_hash,
            "feature_vector_hash": _sha256(
                [[name, str(value)] for name, value in features]),
            "feature_columns": list(FEATURE_COLUMNS_V2),
            "paper_model_spec_hash": self.session_spec.paper_model_spec_hash,
            "fitted_hash": model.fitted.fitted_model_hash,
        }
        batch.targets[instrument] = entry
        self._append_once(
            event_type="PORTFOLIO_INSTRUMENT_READY", event_at=str(now),
            instrument_id=instrument, natural_key=opening, payload=entry)

        if batch.complete:
            self._append_once(
                event_type="PORTFOLIO_BATCH_READY", event_at=str(now),
                natural_key=opening, payload=batch.payload())
            # A batch whose execution bar falls inside the reserved window can
            # never complete, and waiting for it would leave it pending
            # forever. Decided from the timestamp alone: the protected candle
            # is never requested, and would be refused if it were.
            if self._execution_is_protected(opening):
                self._expire(batch, now=now, reason="HOLDOUT_BOUNDARY",
                             detail=f"the execution bar at {self._next(opening)} "
                                    "lies inside the reserved research holdout")
                self._status = PortfolioStatus.EMBARGOED
                return {"ready": True, "prediction": str(prediction),
                        "side": signal.direction,
                        "target_exposure": str(target.target_exposure),
                        "batch_complete": True, "batch_missing": [],
                        "expired": "HOLDOUT_BOUNDARY"}
        return {"ready": True, "prediction": str(prediction),
                "side": signal.direction,
                "target_exposure": str(target.target_exposure),
                "batch_complete": batch.complete,
                "batch_missing": list(batch.missing_instruments)}

    def _execution_is_protected(self, decision_at: str) -> bool:
        execute_at = self._next(decision_at)
        return any(self.window.covers(name, execute_at)
                   for name in self.session_spec.instruments)

    def _append_once(self, **kwargs):
        """Append, tolerating an event this session already committed."""
        try:
            return self.store.append(session_id=self.session_id, **kwargs)
        except PaperPortfolioStoreError:
            return None

    # --- execution -------------------------------------------------------

    def _try_execute(self, *, decision_at: str, now) -> dict:
        """Rebalance if, and only if, the whole batch can be executed."""
        batch = self._pending.get(decision_at)
        if batch is None:
            return {}
        if decision_at in self._completed:
            return {}
        if not batch.complete:
            self._status = PortfolioStatus.WAITING
            return {"rebalanced": False, "reason": "WAITING_FOR_PORTFOLIO_BATCH",
                    "missing": list(batch.missing_instruments)}

        execute_at = self._next(decision_at)
        # Every instrument in the batch needs an execution price, and every
        # open position needs a mark, or the portfolio has no honest equity.
        needed = set(batch.targets) | {
            position.instrument_id for position in self.state.positions
            if position.quantity != 0}
        missing_price = sorted(
            name for name in needed if execute_at not in self._opens[name])
        if missing_price:
            self._status = PortfolioStatus.WAITING
            return {"rebalanced": False, "reason": "WAITING_FOR_PRICES",
                    "missing_prices": missing_price}

        try:
            for name in needed:
                require_unprotected_bar(name, execute_at, window=self.window)
        except ProtectedHoldoutError as error:
            # Completing this batch would require protected data. Expire it;
            # never ask for the candle.
            self._expire(batch, now=now, reason="HOLDOUT_BOUNDARY",
                         detail=str(error))
            return {"rebalanced": False, "reason": "HOLDOUT_BOUNDARY"}

        frame = PortfolioMarketFrame(
            timestamp=execute_at,
            prices=tuple((name, self._opens[name][execute_at])
                         for name in sorted(needed)))
        target_set = PortfolioTargetSet(
            timestamp=execute_at,
            portfolio_spec_hash=self.portfolio_spec.portfolio_spec_hash,
            targets=tuple(
                InstrumentPositionTarget(
                    instrument_id=name,
                    target_exposure=Decimal(entry["target_exposure"]),
                    source_position_target_hash=entry["position_target_hash"],
                    side=entry["signal_direction"])
                for name, entry in sorted(batch.targets.items())))
        batch_id = portfolio_batch_id(
            session_spec_hash=self.session_spec.session_spec_hash,
            decision_at=decision_at, targets=batch.targets,
            frame_hash=frame.frame_hash)

        if self.store.has_event(session_id=self.session_id,
                                event_type="PORTFOLIO_SNAPSHOT",
                                instrument_id=None, natural_key=batch_id):
            # An early-out, not the guarantee. Exactly-once is enforced by the
            # store's UNIQUE (session, event_type, instrument, natural_key)
            # constraint, which refuses a duplicate whether or not any code
            # path remembers to ask. This check only avoids redoing the work.
            self._completed.add(decision_at)
            self._pending.pop(decision_at, None)
            return {"rebalanced": False, "reason": "ALREADY_EXECUTED",
                    "batch_id": batch_id}
        # A target set without its snapshot is a batch that was interrupted
        # between deciding and finishing. It is resumed, not abandoned: the
        # decision is already on record, and dropping it would silently lose
        # a trade the log says was made. Re-appending is refused by the
        # database, so resuming cannot duplicate anything.
        resuming = self.store.has_event(
            session_id=self.session_id, event_type="PORTFOLIO_TARGET_SET_CREATED",
            instrument_id=None, natural_key=batch_id)

        self._append_once(
            event_type="PORTFOLIO_TARGET_SET_CREATED", event_at=str(now),
            natural_key=batch_id,
            payload={"batch_id": batch_id, "decision_at": decision_at,
                     "execute_at": execute_at,
                     "target_set": target_set.canonical(),
                     "target_set_hash": target_set.target_set_hash,
                     "market_frame_hash": frame.frame_hash})

        try:
            outcome = rebalance(self.state, target_set=target_set, frame=frame,
                                spec=self.portfolio_spec,
                                execution_spec=self.execution_spec.as_backtest_spec())
        except PortfolioValuationUnavailable as error:
            self._append_once(
                event_type="PORTFOLIO_VALUATION_UNAVAILABLE", event_at=str(now),
                natural_key=batch_id, payload={"reason": str(error)})
            self._status = PortfolioStatus.DEGRADED
            return {"rebalanced": False, "reason": "VALUATION_UNAVAILABLE"}
        except PortfolioError as error:
            self._append_once(event_type="ERROR", event_at=str(now),
                              natural_key=batch_id,
                              payload={"error_code": "PORTFOLIO_REBALANCE_REFUSED",
                                       "detail": str(error)})
            self._status = PortfolioStatus.ERROR
            raise

        if outcome.scaled:
            self._append_once(
                event_type="PORTFOLIO_TARGET_SCALED", event_at=str(now),
                natural_key=batch_id,
                payload={"batch_id": batch_id,
                         "portfolio_scale": str(outcome.portfolio_scale),
                         "requested_gross": str(outcome.requested_gross)})

        appended = 0
        for fill in outcome.fills:
            if self._append_once(
                    event_type="PORTFOLIO_FILL", event_at=str(now),
                    instrument_id=fill.instrument_id, natural_key=batch_id,
                    payload={**fill.canonical(), "batch_id": batch_id,
                             "fill_hash": fill.fill_hash}) is not None:
                appended += 1
        # Only what this call wrote. A resumed batch whose fills were already
        # on disk must not count them twice.
        self._fill_count += appended
        self._rebalance_count += 1
        self.state = outcome.state
        self._completed.add(decision_at)
        self._pending.pop(decision_at, None)

        self._append_once(
            event_type="PORTFOLIO_SNAPSHOT", event_at=str(now),
            natural_key=batch_id,
            payload={"batch_id": batch_id, **self.state.canonical(),
                     "pre_trade_equity": str(outcome.pre_trade_equity),
                     "fill_count": self._fill_count,
                     "rebalance_count": self._rebalance_count})
        self._status = PortfolioStatus.RUNNING
        return {"rebalanced": True, "batch_id": batch_id,
                "fills": len(outcome.fills),
                "equity": str(self.state.equity),
                "gross_exposure": str(self.state.gross_exposure)}

    def _next(self, opening: str) -> str:
        duration = TIMEFRAME_DURATIONS[self.timeframe]
        return _iso(datetime.fromisoformat(opening) + duration)

    def _expire(self, batch: PendingPortfolioBatch, *, now, reason: str,
                detail: str = "") -> None:
        self._append_once(
            event_type="PORTFOLIO_BATCH_INCOMPLETE", event_at=str(now),
            natural_key=batch.decision_at,
            payload={**batch.payload(), "reason": reason, "detail": detail})
        self._completed.add(batch.decision_at)
        self._pending.pop(batch.decision_at, None)

    def expire_incomplete_before(self, *, cutoff: str, now, reason: str) -> int:
        """Close batches that can no longer be completed. Never silently."""
        expired = 0
        for decision_at in sorted(self._pending):
            if decision_at < cutoff:
                self._expire(self._pending[decision_at], now=now, reason=reason)
                expired += 1
        return expired

    # --- snapshots -------------------------------------------------------

    def _maybe_snapshot(self) -> None:
        """Measured from the last snapshot, not by testing a running total.

        The Phase 5D trigger used ``total % N == 0``. A batch appends several
        events at once, so a constant stride steps over every multiple and the
        trigger never fires -- silently, for an entire session.
        """
        events = self.store.latest_events(session_id=self.session_id, limit=1)
        if not events:
            return
        head = events[-1]
        previous = self.store.latest_snapshot(session_id=self.session_id)
        since = head.event_id - (previous["last_event_id"] if previous else 0)
        if since < SNAPSHOT_EVERY_EVENTS:
            return
        self.store.write_snapshot(
            session_id=self.session_id, last_event_id=head.event_id,
            last_event_hash=head.event_hash, state=self._state_payload())

    def _state_payload(self) -> dict:
        return {
            "state": self.state.canonical(),
            "fill_count": self._fill_count,
            "rebalance_count": self._rebalance_count,
            "completed_batches": sorted(self._completed),
            "pending": {key: batch.payload()
                        for key, batch in sorted(self._pending.items())},
        }

    # --- restart ---------------------------------------------------------

    def restore(self, *, history: dict | None = None) -> dict:
        """Rebuild state and pending batches from the verified log.

        Replays rather than trusting a snapshot alone: a snapshot is an
        optimisation, and a portfolio that can only restart from one cannot
        restart at all when the trigger has not fired yet.
        """
        self.store.verify_chain(session_id=self.session_id)
        self.state = initial_portfolio_state(self.portfolio_spec)
        self._pending, self._completed, self._started = {}, set(), {}
        self._fill_count = self._rebalance_count = 0
        for instrument, rows in (history or {}).items():
            canonical = self._require_instrument(instrument)
            self._bars[canonical] = [dict(row) for row in rows]
            for row in rows:
                self._opens[canonical][str(row["bar_open_at"])] = Decimal(
                    str(row["open"]))

        after, replayed = None, 0
        while True:
            page = self.store.events(session_id=self.session_id,
                                     after_event_id=after, limit=5000)
            if not page:
                break
            for event in page:
                self._replay(event)
                replayed += 1
            after = page[-1].event_id
        self._status = PortfolioStatus.RUNNING
        return {"replayed_events": replayed,
                "pending_batches": sorted(self._pending),
                "completed_batches": len(self._completed),
                "fill_count": self._fill_count,
                "equity": str(self.state.equity)}

    def _replay(self, event) -> None:
        kind = event.event_type
        if kind == "PORTFOLIO_BATCH_OPENED":
            decision_at = event.payload["decision_at"]
            self._pending.setdefault(decision_at, PendingPortfolioBatch(
                decision_at=decision_at,
                required=tuple(event.payload.get("required",
                                                 self.session_spec.instruments))))
        elif kind == "PORTFOLIO_INSTRUMENT_READY":
            decision_at = event.natural_key
            batch = self._pending.setdefault(decision_at, PendingPortfolioBatch(
                decision_at=decision_at, required=self.session_spec.instruments))
            batch.targets[event.instrument_id] = dict(event.payload)
        elif kind == "PORTFOLIO_BATCH_INCOMPLETE":
            self._completed.add(event.payload["decision_at"])
            self._pending.pop(event.payload["decision_at"], None)
        elif kind == "PORTFOLIO_TARGET_SET_CREATED":
            # Deciding is not finishing. The batch stays pending until its
            # snapshot lands, so an interrupted rebalance is resumed rather
            # than quietly dropped.
            self._started[event.payload["batch_id"]] = event.payload["decision_at"]
        elif kind == "PORTFOLIO_FILL":
            self._fill_count += 1
        elif kind == "PORTFOLIO_SNAPSHOT":
            batch_id = event.payload.get("batch_id")
            decision_at = self._started.pop(batch_id, None)
            if decision_at is not None:
                self._completed.add(decision_at)
                self._pending.pop(decision_at, None)
            self._rebalance_count = event.payload.get(
                "rebalance_count", self._rebalance_count)
            self._fill_count = event.payload.get("fill_count", self._fill_count)
            self.state = _state_from_payload(event.payload)

    # --- reporting -------------------------------------------------------

    def payload(self, *, now=None) -> dict:
        moment = now or _iso(datetime.now(timezone.utc))
        pending = {key: batch.payload() for key, batch in sorted(self._pending.items())}
        return {
            "mode": "SHARED_PORTFOLIO",
            "status": self._status,
            "session_id": self.session_id,
            "session_spec_hash": self.session_spec.session_spec_hash,
            "portfolio_spec_hash": self.portfolio_spec.portfolio_spec_hash,
            "instruments": list(self.session_spec.instruments),
            "shared_capital": True,
            "real_money": False,
            "broker_connected": False,
            "cash": str(self.state.cash),
            "equity": str(self.state.equity),
            "gross_exposure": str(self.state.gross_exposure),
            "net_exposure": str(self.state.net_exposure),
            "cumulative_fees": str(self.state.cumulative_fees),
            "cumulative_slippage_cost": str(self.state.cumulative_slippage_cost),
            "fill_count": self._fill_count,
            "rebalance_count": self._rebalance_count,
            "positions": [position.canonical() for position in self.state.positions],
            "pending_batches": pending,
            "embargo": {name: embargo_state(name, now=moment, window=self.window)
                        for name in self.session_spec.instruments},
        }


def _state_from_payload(payload: dict) -> PortfolioState:
    from scripts.trading_lab.portfolio import InstrumentPositionState

    return PortfolioState(
        timestamp=payload.get("timestamp", ""),
        cash=Decimal(payload["cash"]),
        positions=tuple(
            InstrumentPositionState(
                instrument_id=item["instrument_id"],
                quantity=Decimal(item["quantity"]),
                mark_price=Decimal(item["mark_price"]),
                target_exposure=Decimal(item["target_exposure"]),
                cumulative_fees=Decimal(item["cumulative_fees"]),
                cumulative_slippage_cost=Decimal(item["cumulative_slippage_cost"]),
                cumulative_gross_pnl=Decimal(item["cumulative_gross_pnl"]))
            for item in payload.get("positions", [])))
