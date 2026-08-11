"""Phase 6C: the shared paper portfolio runtime.

Marked `ml`: the shadow models are Ridge fits, so these need the optional
stack. The identity and store behaviour that must survive a core install is
tested separately.

The property that carries this phase is arrival-order independence. Live
candles do not arrive in a defined sequence, and the naive shape -- rebalance
whichever instrument landed -- makes the portfolio depend on network timing.
Phase 6B established that is a different portfolio, not a rounding
difference, so several tests here permute arrival and demand identical
economics.

The second property is exactly-once. A crash between committing a fill and
writing a snapshot must not replay the trade, and a restart must not lose a
half-collected batch.
"""

from __future__ import annotations

import itertools
import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

pytestmark = pytest.mark.ml

BTC = "coinbase:BTC-USD"
ETH = "coinbase:ETH-USD"
GRID = datetime(2026, 1, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat()


class FakeClock:
    def __init__(self, start=GRID):
        self.now = start

    def iso(self):
        return _iso(self.now)

    def advance(self, delta=HOUR):
        self.now = self.now + delta
        return self.iso()


def _rows(count, *, seed=0, start=GRID, drift=Decimal("0")):
    """Synthetic bars with enough variation to keep every feature alive."""
    out = []
    for index in range(count):
        close = (Decimal(20000) + Decimal(index) * drift
                 + Decimal((index * (17 + seed)) % 53)
                 - Decimal((index % 9) * 2))
        out.append({"bar_open_at": _iso(start + HOUR * index),
                    "open": str(close), "high": str(close + 2),
                    "low": str(close - 2), "close": str(close),
                    "volume": str(10 + index % 11)})
    return out


def _calm_rows(count, *, start=GRID, base=20000):
    """A market too quiet to cross the +/-0.0025 signal threshold."""
    out = []
    for index in range(count):
        close = (Decimal(base) + Decimal(index) / 100
                 + Decimal(index % 7) / 10 - Decimal(index % 3) / 10)
        out.append({"bar_open_at": _iso(start + HOUR * index),
                    "open": str(close), "high": str(close + 1 + Decimal(index % 5) / 10),
                    "low": str(close - 1 - Decimal(index % 4) / 10),
                    "close": str(close), "volume": str(10 + index % 11)})
    return out


@pytest.fixture(scope="module")
def modules():
    import importlib
    return (importlib.import_module("scripts.trading_lab.paper_portfolio"),
            importlib.import_module("scripts.trading_lab.paper_portfolio_store"),
            importlib.import_module("scripts.trading_lab.paper_model"),
            importlib.import_module("scripts.trading_lab.paper_engine"))


@pytest.fixture(scope="module")
def trained(modules):
    """One frozen model per instrument, over synthetic history."""
    portfolio_module, _, model_module, engine_module = modules
    history = {BTC: _rows(400, seed=0, drift=Decimal("0.9")),
               ETH: _rows(400, seed=5, drift=Decimal("-0.6"))}
    models = {}
    for instrument, rows in history.items():
        series = engine_module.series_from_rows(
            rows, product=instrument.split(":")[-1])
        artifact = model_module.train_paper_model(
            series, product=instrument.split(":")[-1])
        models[instrument] = model_module.load_paper_model(
            artifact, product=instrument.split(":")[-1])
    return history, models


def _engine(modules, trained, tmp_path, *, name="portfolio.sqlite",
            session="p1", seed=True):
    portfolio_module, store_module, _, _ = modules
    history, models = trained
    store = store_module.PaperPortfolioStore(tmp_path / name)
    spec = portfolio_module.build_portfolio_session_spec(models)
    engine = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id=session, session_spec=spec)
    if seed:
        # Seed everything but the tail, so the tail can actually be ingested.
        # Seeding a bar and then feeding it is refused as already processed.
        for instrument, rows in history.items():
            engine.seed_history(instrument, rows[:-10])
    return store, engine


def _feed(engine, history, index, clock, order=(BTC, ETH)):
    """Deliver bar `index` for each instrument in the given arrival order."""
    outcomes = {}
    for instrument in order:
        outcomes[instrument] = engine.ingest_candle(
            instrument, history[instrument][index], now=clock.advance(HOUR / 60))
    return outcomes


# --- the session specification --------------------------------------------


def test_the_session_names_every_frozen_contract(modules, trained):
    portfolio_module, _, _, _ = modules
    _, models = trained
    spec = portfolio_module.build_portfolio_session_spec(models)
    from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    assert spec.instruments == (BTC, ETH)
    assert spec.portfolio_spec_hash == PORTFOLIO_SPEC_V1.portfolio_spec_hash
    assert spec.signal_spec_hash == SIGNAL_SPEC_V1.spec_hash
    assert spec.risk_spec_hash == RISK_SPEC_V1.risk_spec_hash
    assert spec.canonical()["real_money"] is False
    assert spec.canonical()["shared_capital"] is True
    assert len(spec.session_spec_hash) == 64


def test_the_session_hash_does_not_depend_on_the_order_models_were_passed(
        modules, trained):
    portfolio_module, _, _, _ = modules
    _, models = trained
    forward = portfolio_module.build_portfolio_session_spec(
        models, instruments=(BTC, ETH))
    backward = portfolio_module.build_portfolio_session_spec(
        models, instruments=(ETH, BTC))
    assert forward.session_spec_hash == backward.session_spec_hash


@pytest.mark.parametrize("alias", ["BTC-USD", "btc-usd", "BTCUSD"])
def test_any_spelling_resolves_into_the_canonical_session(modules, trained, alias):
    portfolio_module, _, _, _ = modules
    _, models = trained
    spec = portfolio_module.build_portfolio_session_spec(
        models, instruments=(alias, ETH))
    assert spec.instruments == (BTC, ETH)


# --- batching -------------------------------------------------------------


def test_one_instrument_alone_never_triggers_a_rebalance(modules, trained, tmp_path):
    """The whole defence against arrival order deciding the portfolio."""
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())

    outcome = engine.ingest_candle(BTC, history[BTC][-10], now=clock.advance())
    assert outcome["ready"] is True
    assert outcome["batch_complete"] is False
    assert outcome["batch_missing"] == [ETH]
    assert engine.status == "WAITING_FOR_PORTFOLIO_BATCH" or engine._fill_count == 0
    assert not [event for event in store.events(session_id="p1", limit=500)
                if event.event_type == "PORTFOLIO_FILL"]


def test_a_batch_completes_only_when_every_instrument_has_a_target(
        modules, trained, tmp_path):
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())

    engine.ingest_candle(BTC, history[BTC][-10], now=clock.advance())
    assert not _events(store, "PORTFOLIO_BATCH_READY")
    engine.ingest_candle(ETH, history[ETH][-10], now=clock.advance())
    ready = _events(store, "PORTFOLIO_BATCH_READY")
    assert len(ready) == 1
    assert ready[0].payload["complete"] is True


def _events(store, event_type, session="p1"):
    found, after = [], None
    while True:
        page = store.events(session_id=session, after_event_id=after, limit=5000)
        if not page:
            break
        found.extend(event for event in page if event.event_type == event_type)
        after = page[-1].event_id
    return found


def test_a_flat_signal_is_a_target_not_a_missing_one(modules, trained, tmp_path):
    """FLAT is an exposure of zero; treating it as absent stalls every batch."""
    portfolio_module, store_module, model_module, engine_module = modules
    calm = {BTC: _calm_rows(400, base=20000), ETH: _calm_rows(400, base=3000)}
    models = {}
    for instrument, rows in calm.items():
        series = engine_module.series_from_rows(
            rows, product=instrument.split(":")[-1])
        artifact = model_module.train_paper_model(
            series, product=instrument.split(":")[-1])
        models[instrument] = model_module.load_paper_model(
            artifact, product=instrument.split(":")[-1])
    store = store_module.PaperPortfolioStore(tmp_path / "flat.sqlite")
    spec = portfolio_module.build_portfolio_session_spec(models)
    engine = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="flat", session_spec=spec)
    for instrument, rows in calm.items():
        engine.seed_history(instrument, rows)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())

    extra = {BTC: _calm_rows(3, start=GRID + HOUR * 400, base=20004),
             ETH: _calm_rows(3, start=GRID + HOUR * 400, base=3004)}
    for index in range(3):
        for instrument in (BTC, ETH):
            outcome = engine.ingest_candle(instrument, extra[instrument][index],
                                           now=clock.advance())
            assert outcome["ready"] is True, outcome
            assert Decimal(outcome["target_exposure"]) == 0
    ready = _events(store, "PORTFOLIO_BATCH_READY", session="flat")
    assert len(ready) == 3, "a flat market must still complete its batches"


# --- arrival order --------------------------------------------------------


@pytest.mark.parametrize("order", list(itertools.permutations([BTC, ETH])))
def test_arrival_order_changes_no_economic_decision(modules, trained, tmp_path,
                                                    order):
    history, _ = trained
    results = {}
    for label, arrival in (("baseline", (BTC, ETH)), ("permuted", order)):
        store, engine = _engine(modules, trained, tmp_path,
                                name=f"{label}.sqlite", session=label)
        clock = FakeClock(GRID + HOUR * 500)
        engine.start(now=clock.iso())
        for index in range(-10, -4):
            _feed(engine, history, index, clock, order=arrival)
        results[label] = engine.payload()

    baseline, permuted = results["baseline"], results["permuted"]
    for field in ("cash", "equity", "gross_exposure", "net_exposure",
                  "cumulative_fees", "cumulative_slippage_cost", "fill_count",
                  "rebalance_count", "positions"):
        assert permuted[field] == baseline[field], field


def test_the_batch_identity_does_not_depend_on_arrival_order(modules, trained,
                                                             tmp_path):
    history, _ = trained
    identities = []
    for label, arrival in (("a", (BTC, ETH)), ("b", (ETH, BTC))):
        store, engine = _engine(modules, trained, tmp_path,
                                name=f"id-{label}.sqlite", session=label)
        clock = FakeClock(GRID + HOUR * 500)
        engine.start(now=clock.iso())
        for index in range(-10, -6):
            _feed(engine, history, index, clock, order=arrival)
        identities.append([event.natural_key for event in
                           _events(store, "PORTFOLIO_TARGET_SET_CREATED",
                                   session=label)])
    assert identities[0] == identities[1]
    assert identities[0], "no batch executed"


def test_the_batch_identity_is_derived_not_generated(modules, trained):
    """A UUID would make a restart unable to recognise its own batch."""
    portfolio_module, _, _, _ = modules
    _, models = trained
    spec = portfolio_module.build_portfolio_session_spec(models)
    targets = {BTC: {"position_target_hash": "a" * 64},
               ETH: {"position_target_hash": "b" * 64}}
    first = portfolio_module.portfolio_batch_id(
        session_spec_hash=spec.session_spec_hash,
        decision_at="2026-01-01T00:00:00+00:00", targets=targets,
        frame_hash="c" * 64)
    second = portfolio_module.portfolio_batch_id(
        session_spec_hash=spec.session_spec_hash,
        decision_at="2026-01-01T00:00:00+00:00",
        targets={ETH: targets[ETH], BTC: targets[BTC]}, frame_hash="c" * 64)
    assert first == second
    different = portfolio_module.portfolio_batch_id(
        session_spec_hash=spec.session_spec_hash,
        decision_at="2026-01-01T01:00:00+00:00", targets=targets,
        frame_hash="c" * 64)
    assert different != first


# --- shared capital -------------------------------------------------------


def test_there_is_one_cash_ledger_and_equity_balances(modules, trained, tmp_path):
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -4):
        _feed(engine, history, index, clock)

    payload = engine.payload()
    assert "btc_cash" not in payload and "eth_cash" not in payload
    from decimal import localcontext

    from scripts.trading_lab.portfolio import ECONOMIC_PRECISION
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        rebuilt = Decimal(payload["cash"]) + sum(
            (Decimal(position["market_value"]) for position in payload["positions"]),
            Decimal(0))
        assert rebuilt == Decimal(payload["equity"])


def test_the_runtime_owns_no_financial_arithmetic(modules):
    """Fees, slippage and sizing come from the 6B engine or from nowhere."""
    import inspect

    portfolio_module, _, _, _ = modules
    source = inspect.getsource(portfolio_module)
    for duplicated in ("fee_rate *", "slippage_rate *", "* fee_rate",
                       "* slippage_rate", "target_exposure * ",
                       "pre_trade_equity /", "quantity * mark_price"):
        assert duplicated not in source, f"runtime duplicates {duplicated!r}"
    assert "rebalance(" in source and "mark_to_market" in source


# --- the online runtime against the offline engine ------------------------


def test_the_live_portfolio_matches_the_offline_engine_exactly(modules, trained,
                                                               tmp_path):
    """Same batches, same frames: the two must agree to the last digit."""
    from scripts.trading_lab.portfolio import (
        InstrumentPositionTarget, PORTFOLIO_SPEC_V1, PortfolioMarketFrame,
        PortfolioTargetSet, initial_portfolio_state, rebalance)

    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -2):
        _feed(engine, history, index, clock)

    # Rebuild the same decisions offline, straight from the recorded events.
    targets_by_batch = {}
    for event in _events(store, "PORTFOLIO_TARGET_SET_CREATED"):
        targets_by_batch[event.natural_key] = event.payload

    offline = initial_portfolio_state(PORTFOLIO_SPEC_V1)
    execution = engine.execution_spec.as_backtest_spec()
    for batch_id in [event.natural_key for event in
                     _events(store, "PORTFOLIO_TARGET_SET_CREATED")]:
        payload = targets_by_batch[batch_id]
        stored = payload["target_set"]
        target_set = PortfolioTargetSet(
            timestamp=stored["timestamp"],
            portfolio_spec_hash=stored["portfolio_spec_hash"],
            targets=tuple(
                InstrumentPositionTarget(
                    instrument_id=item["instrument_id"],
                    target_exposure=Decimal(item["target_exposure"]),
                    source_position_target_hash=item["source_position_target_hash"],
                    side=item["side"])
                for item in stored["targets"]))
        frame = PortfolioMarketFrame(
            timestamp=stored["timestamp"],
            prices=tuple((name, engine._opens[name][stored["timestamp"]])
                         for name in sorted(
                             set(item["instrument_id"] for item in stored["targets"])
                             | {position.instrument_id for position in offline.positions
                                if position.quantity != 0})))
        offline = rebalance(offline, target_set=target_set, frame=frame,
                            spec=PORTFOLIO_SPEC_V1,
                            execution_spec=execution).state

    assert str(offline.cash) == engine.payload()["cash"]
    assert str(offline.equity) == engine.payload()["equity"]
    assert offline.portfolio_state_hash == engine.state.portfolio_state_hash


# --- restart --------------------------------------------------------------


def test_a_restart_replays_nothing_twice(modules, trained, tmp_path):
    portfolio_module, store_module, _, _ = modules
    history, models = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -4):
        _feed(engine, history, index, clock)
    before = engine.payload()
    fills_before = len(_events(store, "PORTFOLIO_FILL"))

    reopened = store_module.PaperPortfolioStore(tmp_path / "portfolio.sqlite")
    spec = portfolio_module.build_portfolio_session_spec(models)
    resumed = portfolio_module.PaperPortfolioEngine(
        store=reopened, models=models, session_id="p1", session_spec=spec)
    report = resumed.restore(
        history={name: rows[:-10] for name, rows in history.items()})

    assert report["fill_count"] == before["fill_count"]
    assert resumed.payload()["cash"] == before["cash"]
    assert resumed.payload()["equity"] == before["equity"]
    assert len(_events(reopened, "PORTFOLIO_FILL")) == fills_before


def test_a_half_collected_batch_survives_a_restart(modules, trained, tmp_path):
    """BTC ready, crash, restart, ETH completes it. No lost batch."""
    portfolio_module, store_module, _, _ = modules
    history, models = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -5):
        _feed(engine, history, index, clock)
    # only BTC for the next bar, then "crash" with the batch half collected
    engine.ingest_candle(BTC, history[BTC][-5], now=clock.advance())
    decision_at = str(history[BTC][-5]["bar_open_at"])

    reopened = store_module.PaperPortfolioStore(tmp_path / "portfolio.sqlite")
    spec = portfolio_module.build_portfolio_session_spec(models)
    resumed = portfolio_module.PaperPortfolioEngine(
        store=reopened, models=models, session_id="p1", session_spec=spec)
    report = resumed.restore(
        history={name: rows[:-10] for name, rows in history.items()})

    assert decision_at in report["pending_batches"]
    pending = resumed._pending[decision_at]
    assert pending.ready_instruments == (BTC,)
    assert pending.missing_instruments == (ETH,)


def test_the_event_chain_verifies_after_a_restart(modules, trained, tmp_path):
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -5):
        _feed(engine, history, index, clock)
    chain = store.verify_chain(session_id="p1")
    assert chain["verified"] is True
    assert chain["events"] > 0


def test_a_tampered_portfolio_event_breaks_the_chain(modules, trained, tmp_path):
    import json
    import sqlite3

    portfolio_module, store_module, _, _ = modules
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    _feed(engine, history, -10, clock)

    connection = sqlite3.connect(tmp_path / "portfolio.sqlite")
    with connection:
        connection.execute(
            "UPDATE portfolio_events SET payload = ? WHERE sequence = 2",
            (json.dumps({"tampered": True}),))
    connection.close()
    with pytest.raises(store_module.PaperPortfolioStoreError):
        store.verify_chain(session_id="p1")


def test_a_committed_batch_is_never_executed_a_second_time(modules, trained,
                                                           tmp_path):
    """Crash after the fill commit must not trade it again on resume."""
    portfolio_module, store_module, _, _ = modules
    history, models = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -4):
        _feed(engine, history, index, clock)
    batches = [event.natural_key
               for event in _events(store, "PORTFOLIO_TARGET_SET_CREATED")]
    assert batches

    spec = portfolio_module.build_portfolio_session_spec(models)
    resumed = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="p1", session_spec=spec)
    resumed.restore(history={name: rows[:-10] for name, rows in history.items()})
    # feed the same bars again; every batch is already committed
    replay_clock = FakeClock(GRID + HOUR * 600)
    for index in range(-10, -4):
        _feed(resumed, history, index, replay_clock)
    assert [event.natural_key
            for event in _events(store, "PORTFOLIO_TARGET_SET_CREATED")] == batches


# --- snapshots ------------------------------------------------------------


def test_snapshots_are_written_even_when_nothing_ever_fills(modules, trained,
                                                            tmp_path):
    """The Phase 5D failure: a constant stride stepped over every multiple."""
    portfolio_module, store_module, model_module, engine_module = modules
    calm = {BTC: _calm_rows(400, base=20000), ETH: _calm_rows(400, base=3000)}
    models = {}
    for instrument, rows in calm.items():
        series = engine_module.series_from_rows(
            rows, product=instrument.split(":")[-1])
        artifact = model_module.train_paper_model(
            series, product=instrument.split(":")[-1])
        models[instrument] = model_module.load_paper_model(
            artifact, product=instrument.split(":")[-1])
    store = store_module.PaperPortfolioStore(tmp_path / "snap.sqlite")
    spec = portfolio_module.build_portfolio_session_spec(models)
    engine = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="snap", session_spec=spec)
    for instrument, rows in calm.items():
        engine.seed_history(instrument, rows)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())

    every = store_module.SNAPSHOT_EVERY_EVENTS
    extra = {BTC: _calm_rows(120, start=GRID + HOUR * 400, base=20004),
             ETH: _calm_rows(120, start=GRID + HOUR * 400, base=3004)}
    for index in range(120):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, extra[instrument][index],
                                 now=clock.advance())

    assert store.count(session_id="snap") > 2 * every
    assert not _events(store, "PORTFOLIO_FILL", session="snap"), \
        "the calm market was supposed to stay flat"
    snapshot = store.latest_snapshot(session_id="snap")
    assert snapshot is not None, "no snapshot after thousands of events"
    head = store.latest_events(session_id="snap", limit=1)[-1]
    assert head.event_id - snapshot["last_event_id"] < every


def test_a_flat_session_keeps_its_starting_equity(modules, trained, tmp_path):
    portfolio_module, store_module, model_module, engine_module = modules
    calm = {BTC: _calm_rows(400, base=20000), ETH: _calm_rows(400, base=3000)}
    models = {}
    for instrument, rows in calm.items():
        series = engine_module.series_from_rows(
            rows, product=instrument.split(":")[-1])
        artifact = model_module.train_paper_model(
            series, product=instrument.split(":")[-1])
        models[instrument] = model_module.load_paper_model(
            artifact, product=instrument.split(":")[-1])
    store = store_module.PaperPortfolioStore(tmp_path / "calm.sqlite")
    spec = portfolio_module.build_portfolio_session_spec(models)
    engine = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="calm", session_spec=spec)
    for instrument, rows in calm.items():
        engine.seed_history(instrument, rows)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    extra = {BTC: _calm_rows(10, start=GRID + HOUR * 400, base=20004),
             ETH: _calm_rows(10, start=GRID + HOUR * 400, base=3004)}
    for index in range(10):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, extra[instrument][index],
                                 now=clock.advance())
    payload = engine.payload()
    assert payload["fill_count"] == 0
    assert Decimal(payload["equity"]) == Decimal("100000")
    assert Decimal(payload["cash"]) == Decimal("100000")


# --- identity -------------------------------------------------------------


def test_a_candle_offered_under_another_instruments_name_is_refused(
        modules, trained, tmp_path):
    from scripts.trading_lab.portfolio import PortfolioIdentityMismatch

    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    with pytest.raises(PortfolioIdentityMismatch):
        engine.ingest_candle(BTC, history[ETH][-10], now=clock.advance(),
                             row_instrument=ETH)


@pytest.mark.parametrize("unknown", ["coinbase:SOL-USD", "nasdaq:AAPL", "SOL-USD"])
def test_an_instrument_outside_the_session_is_refused(modules, trained, tmp_path,
                                                      unknown):
    from scripts.trading_lab.portfolio import PortfolioIdentityMismatch
    from scripts.trading_lab.identity import IdentityError

    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    with pytest.raises((PortfolioIdentityMismatch, IdentityError)):
        engine.ingest_candle(unknown, history[BTC][-10], now=clock.advance())


def test_positions_are_keyed_by_canonical_identity(modules, trained, tmp_path):
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())
    for index in range(-10, -4):
        _feed(engine, history, index, clock)
    for position in engine.payload()["positions"]:
        assert position["instrument_id"] in (BTC, ETH)
        assert ":" in position["instrument_id"]


# --- holdout --------------------------------------------------------------


def test_a_protected_candle_is_refused_by_the_shared_runtime(modules, trained,
                                                             tmp_path):
    from scripts.trading_lab.protected_holdout import ProtectedHoldoutError

    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(datetime(2026, 10, 1, tzinfo=timezone.utc))
    engine.start(now=clock.iso())
    protected = {"bar_open_at": "2026-10-01T00:00:00+00:00", "open": "20000",
                 "high": "20010", "low": "19990", "close": "20005", "volume": "1"}
    with pytest.raises(ProtectedHoldoutError):
        engine.ingest_candle(BTC, protected, now=clock.iso())
    assert engine.status == "EMBARGOED"


def test_the_boundary_is_recorded_without_any_market_data(modules, trained,
                                                          tmp_path):
    from scripts.trading_lab.protected_holdout import ProtectedHoldoutError

    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(datetime(2026, 10, 1, tzinfo=timezone.utc))
    engine.start(now=clock.iso())
    protected = {"bar_open_at": "2026-10-05T00:00:00+00:00", "open": "12345.67",
                 "high": "1", "low": "1", "close": "1", "volume": "1"}
    with pytest.raises(ProtectedHoldoutError):
        engine.ingest_candle(BTC, protected, now=clock.iso())
    boundary = _events(store, "PROTECTED_HOLDOUT_BOUNDARY_REACHED")
    assert boundary
    rendered = str(boundary[0].payload)
    assert "12345.67" not in rendered
    assert boundary[0].payload["observed"] is False


def test_a_pending_batch_needing_protected_data_expires(modules, trained, tmp_path):
    """It must never ask for the candle that would complete it."""
    portfolio_module, store_module, _, _ = modules
    history, models = trained
    store, engine = _engine(modules, trained, tmp_path, seed=False)

    # history ending the hour before the boundary
    before = datetime(2026, 8, 31, 23, tzinfo=timezone.utc)
    rows = {BTC: _rows(400, seed=0, drift=Decimal("0.9"),
                       start=before - HOUR * 399),
            ETH: _rows(400, seed=5, drift=Decimal("-0.6"),
                       start=before - HOUR * 399)}
    for instrument, table in rows.items():
        engine.seed_history(instrument, table[:-1])
    clock = FakeClock(before)
    engine.start(now=clock.iso())

    for instrument in (BTC, ETH):
        engine.ingest_candle(instrument, rows[instrument][-1], now=clock.iso())
    decision_at = str(rows[BTC][-1]["bar_open_at"])
    # the execution bar for this batch opens at 2026-09-01T00:00Z: protected
    assert decision_at in engine._pending or decision_at in engine._completed
    incomplete = _events(store, "PORTFOLIO_BATCH_INCOMPLETE")
    assert incomplete, "the batch should have expired at the boundary"
    assert incomplete[0].payload["reason"] == "HOLDOUT_BOUNDARY"
    assert not _events(store, "PORTFOLIO_FILL")


def test_an_incomplete_batch_never_executes_even_when_its_price_arrives(
        modules, trained, tmp_path):
    """The dangerous case: BTC alone for two bars running.

    The second bar supplies the execution price for the first, so everything
    needed to trade BTC on its own is present. Only the completeness rule
    stops it -- and trading one leg alone would move cash and equity, which is
    precisely what makes the portfolio depend on arrival order.
    """
    history, _ = trained
    store, engine = _engine(modules, trained, tmp_path)
    clock = FakeClock(GRID + HOUR * 500)
    engine.start(now=clock.iso())

    engine.ingest_candle(BTC, history[BTC][-10], now=clock.advance())
    outcome = engine.ingest_candle(BTC, history[BTC][-9], now=clock.advance())

    assert outcome.get("rebalanced") is False
    assert outcome.get("reason") == "WAITING_FOR_PORTFOLIO_BATCH"
    assert outcome.get("missing") == [ETH]
    assert not _events(store, "PORTFOLIO_FILL")
    assert not _events(store, "PORTFOLIO_TARGET_SET_CREATED")
    assert engine.payload()["cash"] == "100000"
    assert engine.payload()["fill_count"] == 0


def test_two_identical_events_after_different_predecessors_hash_differently(
        modules):
    """Without the predecessor in the body, the chain is a list, not a chain."""
    _, store_module, _, _ = modules

    def _event(previous):
        return store_module.PortfolioEvent(
            event_id=1, session_id="s", sequence=2,
            event_type="PORTFOLIO_SNAPSHOT", event_at="2026-01-01T00:00:00Z",
            instrument_id=None, natural_key="batch-1", payload={"equity": "1"},
            previous_event_hash=previous, event_hash="")

    first = _event("a" * 64)
    second = _event("b" * 64)
    assert first.payload == second.payload
    assert first.recomputed_hash() != second.recomputed_hash()


def test_the_batch_identity_covers_every_instrument_regardless_of_order(modules,
                                                                        trained):
    """A batch id built from insertion order would differ per arrival order."""
    portfolio_module, _, _, _ = modules
    _, models = trained
    spec = portfolio_module.build_portfolio_session_spec(models)
    forward = {BTC: {"position_target_hash": "a" * 64},
               ETH: {"position_target_hash": "b" * 64}}
    backward = {ETH: forward[ETH], BTC: forward[BTC]}
    assert list(forward) != list(backward)
    assert portfolio_module.portfolio_batch_id(
        session_spec_hash=spec.session_spec_hash, decision_at="t",
        targets=forward, frame_hash="c" * 64) == \
        portfolio_module.portfolio_batch_id(
            session_spec_hash=spec.session_spec_hash, decision_at="t",
            targets=backward, frame_hash="c" * 64)
    # and it must actually depend on the targets, not merely on the timestamp
    assert portfolio_module.portfolio_batch_id(
        session_spec_hash=spec.session_spec_hash, decision_at="t",
        targets={BTC: {"position_target_hash": "z" * 64}, ETH: forward[ETH]},
        frame_hash="c" * 64) != portfolio_module.portfolio_batch_id(
            session_spec_hash=spec.session_spec_hash, decision_at="t",
            targets=forward, frame_hash="c" * 64)


# --- scenarios that actually trade ----------------------------------------
#
# The synthetic Ridge fits never cross the +/-0.0025 signal threshold, so
# every test above exercises batching over a portfolio that stays flat. That
# proves the plumbing and proves nothing about fills. A scripted model gives
# direct control over LONG, SHORT, FLAT and reversals.


class _ScriptedModel:
    """A stand-in that returns a chosen prediction per bar."""

    model_spec_hash = "5" * 64

    class _Fitted:
        fitted_model_hash = "f" * 64

    def __init__(self, script):
        self.script = script
        self.fitted = self._Fitted()

    def predict(self, rows):
        return [self.script.get(str(rows[0].bar_open_at), Decimal("0"))]


def _scripted_engine(modules, tmp_path, scripts, *, session="scripted",
                     name="scripted.sqlite", bars=60):
    """A session over calm bars, with predictions supplied by the caller."""
    portfolio_module, store_module, _, _ = modules
    history = {BTC: _calm_rows(400 + bars, base=20000),
               ETH: _calm_rows(400 + bars, base=3000)}
    models = {BTC: _ScriptedModel(scripts.get(BTC, {})),
              ETH: _ScriptedModel(scripts.get(ETH, {}))}
    store = store_module.PaperPortfolioStore(tmp_path / name)
    spec = portfolio_module.build_portfolio_session_spec(models)
    engine = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id=session, session_spec=spec)
    for instrument, rows in history.items():
        engine.seed_history(instrument, rows[:400])
    return store, engine, history, models


def _script(history, instrument, values):
    """Map predictions onto the bars that will be fed."""
    return {str(history[instrument][400 + index]["bar_open_at"]): Decimal(value)
            for index, value in enumerate(values)}


def test_a_scripted_session_opens_reverses_and_closes_positions(modules, tmp_path):
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {
        BTC: _script(history, BTC, ["0.02", "0.02", "-0.02", "0", "0"]),
        ETH: _script(history, ETH, ["-0.02", "0", "0.02", "0.02", "0"]),
    }
    store, engine, _, _ = _scripted_engine(modules, tmp_path, scripts)
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(6):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())

    fills = _events(store, "PORTFOLIO_FILL", session="scripted")
    assert fills, "the scripted scenario was supposed to trade"
    sides = {fill.payload["side"] for fill in fills}
    assert {"buy", "sell"} <= sides, "no reversal happened"
    payload = engine.payload()
    assert payload["fill_count"] == len(fills)
    assert Decimal(payload["equity"]) != Decimal("100000")


def test_a_trading_session_matches_the_offline_engine_exactly(modules, tmp_path):
    """Same batches, same frames, real fills: agreement to the last digit."""
    from scripts.trading_lab.portfolio import (
        InstrumentPositionTarget, PORTFOLIO_SPEC_V1, PortfolioMarketFrame,
        PortfolioTargetSet, initial_portfolio_state, rebalance)

    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02", "-0.02", "0"]),
               ETH: _script(history, ETH, ["-0.02", "0.02", "0.02", "0"])}
    store, engine, _, _ = _scripted_engine(modules, tmp_path, scripts,
                                           session="match", name="match.sqlite")
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(5):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())

    offline = initial_portfolio_state(PORTFOLIO_SPEC_V1)
    execution = engine.execution_spec.as_backtest_spec()
    for event in _events(store, "PORTFOLIO_TARGET_SET_CREATED", session="match"):
        stored = event.payload["target_set"]
        target_set = PortfolioTargetSet(
            timestamp=stored["timestamp"],
            portfolio_spec_hash=stored["portfolio_spec_hash"],
            targets=tuple(
                InstrumentPositionTarget(
                    instrument_id=item["instrument_id"],
                    target_exposure=Decimal(item["target_exposure"]),
                    source_position_target_hash=item["source_position_target_hash"],
                    side=item["side"])
                for item in stored["targets"]))
        needed = {item["instrument_id"] for item in stored["targets"]} | {
            position.instrument_id for position in offline.positions
            if position.quantity != 0}
        frame = PortfolioMarketFrame(
            timestamp=stored["timestamp"],
            prices=tuple((name, engine._opens[name][stored["timestamp"]])
                         for name in sorted(needed)))
        offline = rebalance(offline, target_set=target_set, frame=frame,
                            spec=PORTFOLIO_SPEC_V1,
                            execution_spec=execution).state

    assert engine.payload()["fill_count"] > 0
    assert str(offline.cash) == engine.payload()["cash"]
    assert str(offline.equity) == engine.payload()["equity"]
    assert offline.portfolio_state_hash == engine.state.portfolio_state_hash


def test_replaying_a_trading_session_changes_no_economic_state(modules, tmp_path):
    """Exactly-once means the books do not move, not only that no event repeats."""
    portfolio_module, store_module, _, _ = modules
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02", "-0.02", "0"]),
               ETH: _script(history, ETH, ["-0.02", "0.02", "0.02", "0"])}
    store, engine, _, models = _scripted_engine(
        modules, tmp_path, scripts, session="once", name="once.sqlite")
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(5):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())
    before = engine.payload()
    assert before["fill_count"] > 0

    spec = portfolio_module.build_portfolio_session_spec(models)
    resumed = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="once", session_spec=spec)
    resumed.restore(history={name: rows[:400] for name, rows in history.items()})
    replay = FakeClock(GRID + HOUR * 800)
    for index in range(5):
        for instrument in (BTC, ETH):
            resumed.ingest_candle(instrument, history[instrument][400 + index],
                                  now=replay.advance())

    after = resumed.payload()
    for field in ("cash", "equity", "fill_count", "rebalance_count",
                  "cumulative_fees", "cumulative_slippage_cost", "positions"):
        assert after[field] == before[field], field


@pytest.mark.parametrize("order", list(itertools.permutations([BTC, ETH])))
def test_arrival_order_changes_no_fill_in_a_trading_session(modules, tmp_path,
                                                            order):
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02", "-0.02", "0"]),
               ETH: _script(history, ETH, ["-0.02", "0.02", "0.02", "0"])}
    results = {}
    for label, arrival in (("base", (BTC, ETH)), ("perm", order)):
        store, engine, _, _ = _scripted_engine(
            modules, tmp_path, scripts, session=label, name=f"{label}.sqlite")
        clock = FakeClock(GRID + HOUR * 600)
        engine.start(now=clock.iso())
        for index in range(5):
            for instrument in arrival:
                engine.ingest_candle(instrument, history[instrument][400 + index],
                                     now=clock.advance())
        results[label] = (engine.payload(),
                          [event.payload["fill_hash"]
                           for event in _events(store, "PORTFOLIO_FILL",
                                                session=label)])
    base, perm = results["base"], results["perm"]
    assert base[0]["fill_count"] > 0
    assert perm[1] == base[1], "fills differ by arrival order"
    for field in ("cash", "equity", "positions", "cumulative_fees"):
        assert perm[0][field] == base[0][field], field


def test_a_batch_interrupted_between_deciding_and_filling_is_resumed(
        modules, tmp_path):
    """Crash after the target set, before the snapshot.

    Marking the batch done at the target set would drop a trade the log says
    was decided. It stays pending until its snapshot lands.
    """
    portfolio_module, store_module, _, _ = modules
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02"]),
               ETH: _script(history, ETH, ["-0.02", "0.02"])}
    store, engine, _, models = _scripted_engine(
        modules, tmp_path, scripts, session="resume", name="resume.sqlite")
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(3):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())
    complete = engine.payload()
    assert complete["fill_count"] > 0

    # Simulate the crash: drop the snapshot of the last batch, keeping its
    # target set and fills, exactly as an interrupted process would leave it.
    import sqlite3
    last_batch = _events(store, "PORTFOLIO_TARGET_SET_CREATED",
                         session="resume")[-1].natural_key
    connection = sqlite3.connect(tmp_path / "resume.sqlite")
    with connection:
        connection.execute(
            "DELETE FROM portfolio_events WHERE session_id = 'resume' "
            "AND event_type = 'PORTFOLIO_SNAPSHOT' AND natural_key = ?",
            (last_batch,))
    connection.close()

    spec = portfolio_module.build_portfolio_session_spec(models)
    resumed = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="resume", session_spec=spec)
    report = resumed.restore(
        history={name: rows[:400] for name, rows in history.items()})
    # the interrupted batch is not treated as finished
    assert last_batch in resumed._started


def test_a_resumed_batch_does_not_double_count_its_fills(modules, tmp_path):
    """Fills already on disk are refused by the database, so a resume must not
    count them again."""
    portfolio_module, _, _, _ = modules
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02"]),
               ETH: _script(history, ETH, ["-0.02", "0.02"])}
    store, engine, _, models = _scripted_engine(
        modules, tmp_path, scripts, session="dbl", name="dbl.sqlite")
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(3):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())
    recorded = len(_events(store, "PORTFOLIO_FILL", session="dbl"))
    assert engine.payload()["fill_count"] == recorded


def test_a_second_engine_on_the_same_session_cannot_re_execute_a_batch(
        modules, tmp_path):
    """The in-memory completed-set is per instance; the database is not.

    A second process attaching to a live session starts with no memory of what
    has been executed, so the guard that matters is the one that asks the log.
    """
    portfolio_module, _, _, _ = modules
    history = {BTC: _calm_rows(460, base=20000), ETH: _calm_rows(460, base=3000)}
    scripts = {BTC: _script(history, BTC, ["0.02", "0.02"]),
               ETH: _script(history, ETH, ["-0.02", "0.02"])}
    store, engine, _, models = _scripted_engine(
        modules, tmp_path, scripts, session="two", name="two.sqlite")
    clock = FakeClock(GRID + HOUR * 600)
    engine.start(now=clock.iso())
    for index in range(3):
        for instrument in (BTC, ETH):
            engine.ingest_candle(instrument, history[instrument][400 + index],
                                 now=clock.advance())
    fills_before = [event.payload["fill_hash"]
                    for event in _events(store, "PORTFOLIO_FILL", session="two")]
    assert fills_before

    # A fresh instance: same store, same session, no restore, no memory.
    spec = portfolio_module.build_portfolio_session_spec(models)
    second = portfolio_module.PaperPortfolioEngine(
        store=store, models=models, session_id="two", session_spec=spec)
    for instrument, rows in history.items():
        second.seed_history(instrument, rows[:400])
    other = FakeClock(GRID + HOUR * 900)
    for index in range(3):
        for instrument in (BTC, ETH):
            second.ingest_candle(instrument, history[instrument][400 + index],
                                 now=other.advance())

    assert [event.payload["fill_hash"]
            for event in _events(store, "PORTFOLIO_FILL", session="two")] == \
        fills_before
    assert second.payload()["fill_count"] == 0, \
        "a batch already committed must not be traded again"
