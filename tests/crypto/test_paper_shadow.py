"""Shadow trading: event store, model artefacts, pipeline, restarts, embargo.

No network, no real clock, no money. A FakeClock drives every time-dependent
path so the boundary behaviour can be tested without waiting for September.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import json
import pathlib

import pytest

pytestmark = pytest.mark.ml     # the shadow model is a Ridge; sklearn is required

HOUR = timedelta(hours=1)
GRID = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.paper_event_store")


@pytest.fixture
def engine_module():
    return importlib.import_module("scripts.trading_lab.paper_engine")


@pytest.fixture
def model_module():
    return importlib.import_module("scripts.trading_lab.paper_model")


@pytest.fixture
def live_module():
    return importlib.import_module("scripts.trading_lab.live_market")


@pytest.fixture
def holdout():
    return importlib.import_module("scripts.trading_lab.protected_holdout")


class FakeClock:
    """Time under test control. No test ever waits for a real hour."""

    def __init__(self, start: datetime):
        self.now = start

    def iso(self) -> str:
        return self.now.astimezone(timezone.utc).isoformat()

    def advance(self, delta: timedelta) -> str:
        self.now = self.now + delta
        return self.iso()


def _rows(count, *, seed=0, start=GRID, skip=()):
    out = []
    for index in range(count):
        if index in skip:
            continue
        close = Decimal(200 + (index * (17 + seed)) % 53 - (index % 9) * 2)
        out.append({"bar_open_at": (start + HOUR * index).isoformat(),
                    "open": str(close), "high": str(close + 2),
                    "low": str(close - 2), "close": str(close), "volume": "1.0"})
    return out


@pytest.fixture
def trained(engine_module, model_module):
    history = _rows(400)
    series = engine_module.series_from_rows(history, product="BTC-USD")
    artifact = model_module.train_paper_model(series, product="BTC-USD")
    return history, artifact, model_module.load_paper_model(artifact)


def _engine(engine_module, store_module, tmp_path, model, *, name="paper.sqlite",
            session="s1"):
    store = store_module.PaperEventStore(tmp_path / name)
    spec = engine_module.build_session_spec({"BTC-USD": model}, products=("BTC-USD",))
    return store, engine_module.PaperEngine(
        store=store, models={"BTC-USD": model}, session_id=session, session_spec=spec)


# --- event store -----------------------------------------------------------


def test_the_event_log_is_append_only_and_chained(store_module, tmp_path):
    store = store_module.PaperEventStore(tmp_path / "chain.sqlite")
    first = store.append(session_id="s", event_type="SESSION_STARTED",
                         event_at="2026-08-11T00:00:00+00:00", natural_key="session")
    second = store.append(session_id="s", event_type="CANDLE_INGESTED",
                          event_at="2026-08-11T01:00:00+00:00", product="BTC-USD",
                          natural_key="bar-1", payload={"bar": {"x": 1}})
    assert first.previous_event_hash == store_module.GENESIS_HASH
    assert second.previous_event_hash == first.event_hash
    assert second.sequence == 2
    assert store.verify_chain(session_id="s")["verified"] is True
    assert store.verify_chain(session_id="s")["head_hash"] == second.event_hash


def test_a_duplicate_candle_cannot_produce_a_second_event(store_module, tmp_path):
    store = store_module.PaperEventStore(tmp_path / "dup.sqlite")
    store.append(session_id="s", event_type="CANDLE_INGESTED", event_at="t",
                 product="BTC-USD", natural_key="bar-1")
    with pytest.raises(store_module.PaperEventStoreError, match="already exists"):
        store.append(session_id="s", event_type="CANDLE_INGESTED", event_at="t",
                     product="BTC-USD", natural_key="bar-1")
    # the same key in another session is a different run, and is allowed
    store.append(session_id="s2", event_type="CANDLE_INGESTED", event_at="t",
                 product="BTC-USD", natural_key="bar-1")


def test_a_tampered_event_breaks_the_chain(store_module, tmp_path):
    import sqlite3
    path = tmp_path / "tamper.sqlite"
    store = store_module.PaperEventStore(path)
    for index in range(4):
        store.append(session_id="s", event_type="CANDLE_INGESTED", event_at="t",
                     product="BTC-USD", natural_key=f"bar-{index}",
                     payload={"n": index})
    assert store.verify_chain(session_id="s")["verified"] is True
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("UPDATE paper_events SET payload = ? WHERE sequence = 2",
                           (json.dumps({"n": 999}),))
    connection.close()
    with pytest.raises(store_module.PaperEventStoreError, match="own hash"):
        store.verify_chain(session_id="s")


def test_a_deleted_event_breaks_the_chain(store_module, tmp_path):
    import sqlite3
    path = tmp_path / "gap.sqlite"
    store = store_module.PaperEventStore(path)
    for index in range(4):
        store.append(session_id="s", event_type="CANDLE_INGESTED", event_at="t",
                     product="BTC-USD", natural_key=f"bar-{index}")
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("DELETE FROM paper_events WHERE sequence = 2")
    connection.close()
    with pytest.raises(store_module.PaperEventStoreError, match="not contiguous"):
        store.verify_chain(session_id="s")


def test_an_in_memory_store_is_refused(store_module):
    with pytest.raises(store_module.PaperEventStoreError, match="survive a restart"):
        store_module.PaperEventStore(":memory:")


def test_a_snapshot_verifies_its_own_state(store_module, tmp_path):
    store = store_module.PaperEventStore(tmp_path / "snap.sqlite")
    event = store.append(session_id="s", event_type="PORTFOLIO_SNAPSHOT", event_at="t",
                         product="BTC-USD", natural_key="bar-1")
    store.write_snapshot(session_id="s", product="BTC-USD", last_event_id=event.event_id,
                         last_event_hash=event.event_hash, state={"cash": "100000"})
    restored = store.latest_snapshot(session_id="s", product="BTC-USD")
    assert restored["state"] == {"cash": "100000"}
    assert restored["last_event_hash"] == event.event_hash


# --- the shadow model ------------------------------------------------------


def test_the_shadow_model_is_deterministic_and_reloadable(trained, model_module,
                                                           engine_module):
    history, artifact, model = trained
    again = model_module.train_paper_model(
        engine_module.series_from_rows(history, product="BTC-USD"), product="BTC-USD")
    assert again["artifact_hash"] == artifact["artifact_hash"]
    assert again["fitted_hash"] == artifact["fitted_hash"]
    assert model.fitted.fitted_model_hash == artifact["fitted_hash"]


def test_the_shadow_model_uses_the_exact_v2_feature_set(model_module):
    v2 = importlib.import_module("scripts.trading_lab.real_benchmark_v2")
    spec = model_module.PAPER_MODEL_SPEC_V1.canonical()
    assert spec["feature_columns"] == list(v2.FEATURE_COLUMNS_V2)
    assert spec["feature_columns"] == ["return_1", "return_4", "return_12",
                                       "ema_spread_12_26", "rsi_14", "atr_pct_14"]
    assert spec["label"] == {"name": "forward_return", "horizon": 4}
    assert spec["model"] == "ridge_regression"
    assert spec["ridge_alpha"] == "1.0"
    assert spec["optimized"] is False and spec["research_evidence"] is False
    assert spec["shadow_only"] is True
    assert "xgboost" not in json.dumps(spec).lower()


def test_training_refuses_data_after_the_frozen_window(trained, model_module,
                                                        engine_module):
    """August 2026 must never reach the fit: the model predates the live run."""
    history, _, _ = trained
    late = _rows(40, start=datetime(2026, 8, 5, tzinfo=timezone.utc))
    series = engine_module.series_from_rows(history + late, product="BTC-USD")
    with pytest.raises(model_module.PaperModelError, match="frozen training end"):
        model_module.train_paper_model(series, product="BTC-USD")


def test_a_tampered_or_foreign_artifact_is_refused(trained, model_module):
    _, artifact, _ = trained
    broken = dict(artifact, fitted_hash="0" * 64)
    with pytest.raises(model_module.PaperModelError, match="own hash"):
        model_module.load_paper_model(broken)
    resigned = dict(artifact)
    resigned["feature_columns"] = list(reversed(artifact["feature_columns"]))
    resigned.pop("artifact_hash")
    resigned["artifact_hash"] = model_module._sha256(resigned)
    with pytest.raises(model_module.PaperModelError, match="feature schema"):
        model_module.load_paper_model(resigned)


# --- the live pipeline -----------------------------------------------------


def test_only_closed_candles_are_considered(live_module):
    """A candle still forming has a moving close; it may never be used."""
    assert live_module.latest_closed_bar_open("2026-08-11T03:19:00+00:00") == \
        datetime(2026, 8, 11, 2, tzinfo=timezone.utc)
    # within the settle delay the just-closed bar is not yet trusted
    assert live_module.latest_closed_bar_open("2026-08-11T03:00:02+00:00") == \
        datetime(2026, 8, 11, 1, tzinfo=timezone.utc)
    assert live_module.latest_closed_bar_open("2026-08-11T03:00:10+00:00") == \
        datetime(2026, 8, 11, 2, tzinfo=timezone.utc)


def test_the_pipeline_produces_one_decision_per_candle(trained, engine_module,
                                                       store_module, tmp_path):
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-4])
    engine.start(now=clock.iso())
    for row in history[-4:]:
        engine.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
    types = [event.event_type for event in
             store.events(session_id="s1", limit=1000)]
    assert types.count("PREDICTION_CREATED") == 4
    assert types.count("SIGNAL_CREATED") == 4
    assert types.count("POSITION_TARGET_CREATED") == 4
    assert types.count("CANDLE_INGESTED") == 4
    assert store.verify_chain(session_id="s1")["verified"] is True


def test_a_repeated_candle_is_ignored_rather_than_reprocessed(trained, engine_module,
                                                              store_module, tmp_path):
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-2])
    engine.start(now=clock.iso())
    first = engine.ingest_candle("BTC-USD", history[-2], now=clock.advance(HOUR))
    repeat = engine.ingest_candle("BTC-USD", history[-2], now=clock.advance(HOUR))
    assert first["skipped"] is False and repeat["skipped"] is True
    types = [e.event_type for e in store.events(session_id="s1", limit=1000)]
    assert types.count("PREDICTION_CREATED") == 1


def test_a_fill_is_priced_at_the_next_open_and_recorded_when_it_closes(
    trained, engine_module, store_module, tmp_path
):
    """Nothing is backdated; only the recording is late, and it says so."""
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-3])
    engine.start(now=clock.iso())
    for row in history[-3:]:
        engine.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
    fills = [e for e in store.events(session_id="s1", limit=1000)
             if e.event_type == "SIMULATED_FILL"]
    assert fills
    for event in fills:
        fill = event.payload["fill"]
        decided = event.payload["decided_at"]
        assert datetime.fromisoformat(fill["timestamp"]) == \
            datetime.fromisoformat(decided) + HOUR
        # the price is the open of the bar the fill lands on
        bar = next(row for row in history if row["bar_open_at"] == fill["timestamp"])
        assert Decimal(fill["reference_price"]) == Decimal(bar["open"])
        assert event.payload["observation_policy"] == \
            "recorded-when-the-fill-bar-closes-v1"


def test_a_gap_expires_the_pending_target_instead_of_filling_late(
    trained, engine_module, store_module, tmp_path
):
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-4])
    engine.start(now=clock.iso())
    engine.ingest_candle("BTC-USD", history[-4], now=clock.advance(HOUR))
    # the very next bar never arrives: skip straight to the one after
    engine.ingest_candle("BTC-USD", history[-2], now=clock.advance(HOUR * 2))
    gaps = [e for e in store.events(session_id="s1", limit=1000)
            if e.event_type == "GAP_DETECTED"]
    assert any("expired" in (e.payload.get("reason") or "") for e in gaps)
    fills = [e for e in store.events(session_id="s1", limit=1000)
             if e.event_type == "SIMULATED_FILL"]
    assert fills == []


def test_the_paper_engine_reuses_the_committed_fill_accounting(engine_module):
    """One economic engine, not two: the formulas live in economic_backtest."""
    source = pathlib.Path(engine_module.__file__).read_text()
    assert "apply_position_target" in source
    # ban the ARITHMETIC, not the vocabulary: the spec legitimately names its
    # own rates, it just must never compute a fill from them.
    for formula in ("* execution_spec.fee_rate", "cash -= delta",
                    "abs(delta) * fill_price", "* (_ONE + ", "* (_ONE - ",
                    "pre_trade_equity"):
        assert formula not in source, formula


# --- restarts --------------------------------------------------------------


@pytest.mark.parametrize("stop_after", [0, 1, 2, 3])
def test_a_restart_at_any_point_replays_nothing_twice(trained, engine_module,
                                                      store_module, tmp_path,
                                                      stop_after):
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    tail = history[-4:]
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-4])
    engine.start(now=clock.iso())
    for row in tail[:stop_after]:
        engine.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
    before = store.count(session_id="s1")

    # a fresh process picks the same log back up
    reopened = store_module.PaperEventStore(tmp_path / "paper.sqlite")
    spec = engine_module.build_session_spec({"BTC-USD": model}, products=("BTC-USD",))
    resumed = engine_module.PaperEngine(store=reopened, models={"BTC-USD": model},
                                        session_id="s1", session_spec=spec)
    resumed.restore(history={"BTC-USD": history[:-4]})
    for row in tail:
        try:
            resumed.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
        except Exception as error:  # a replayed candle must be skipped, not fatal
            pytest.fail(f"restart failed on {row['bar_open_at']}: {error}")
    types = [e.event_type for e in reopened.events(session_id="s1", limit=2000)]
    assert types.count("PREDICTION_CREATED") == 4
    assert types.count("CANDLE_INGESTED") == 4
    assert reopened.count(session_id="s1") >= before
    assert reopened.verify_chain(session_id="s1")["verified"] is True


def test_a_restart_restores_the_portfolio_exactly(trained, engine_module,
                                                  store_module, tmp_path):
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-5])
    engine.start(now=clock.iso())
    for row in history[-5:]:
        engine.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
    original = engine.state("BTC-USD")

    reopened = store_module.PaperEventStore(tmp_path / "paper.sqlite")
    spec = engine_module.build_session_spec({"BTC-USD": model}, products=("BTC-USD",))
    resumed = engine_module.PaperEngine(store=reopened, models={"BTC-USD": model},
                                        session_id="s1", session_spec=spec)
    resumed.restore(history={"BTC-USD": history[:-5]})
    restored = resumed.state("BTC-USD")
    assert restored.portfolio.cash == original.portfolio.cash
    assert restored.portfolio.quantity == original.portfolio.quantity
    assert restored.fill_count == original.fill_count
    assert restored.equity() == original.equity()


# --- the embargo, inside the running engine --------------------------------


def test_a_session_running_into_september_embargoes_itself(trained, engine_module,
                                                           store_module, tmp_path,
                                                           holdout):
    """The engine stands down by itself, and says why."""
    history, _, model = trained
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-1])
    engine.start(now="2026-08-31T23:59:59+00:00")
    assert engine.status("BTC-USD") == "RUNNING"

    protected = dict(history[-1], bar_open_at="2026-09-01T00:00:00+00:00")
    with pytest.raises(holdout.ProtectedHoldoutError):
        engine.ingest_candle("BTC-USD", protected, now="2026-09-01T00:00:05+00:00")
    assert engine.status("BTC-USD") == "EMBARGOED"

    events = store.events(session_id="s1", limit=1000)
    boundary = [e for e in events if
                e.event_type == "PROTECTED_HOLDOUT_BOUNDARY_REACHED"]
    assert len(boundary) == 1
    # the boundary event records the window, never a protected price
    text = json.dumps(boundary[0].payload)
    for forbidden in ("open", "high", "low", "close", "volume"):
        assert f'"{forbidden}"' not in text
    assert not any(e.event_type == "CANDLE_INGESTED"
                   and e.natural_key == "2026-09-01T00:00:00+00:00" for e in events)


def test_a_session_started_during_the_embargo_never_runs(trained, engine_module,
                                                         store_module, tmp_path):
    history, _, model = trained
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history)
    engine.start(now="2026-10-01T00:00:00+00:00")
    assert engine.status("BTC-USD") == "EMBARGOED"
    assert any(e.event_type == "PROTECTED_HOLDOUT_BOUNDARY_REACHED"
               for e in store.events(session_id="s1", limit=100))


def test_seeding_history_refuses_a_protected_bar(trained, engine_module,
                                                 store_module, tmp_path, holdout):
    history, _, model = trained
    _, engine = _engine(engine_module, store_module, tmp_path, model)
    poisoned = list(history)
    poisoned.append(dict(history[-1], bar_open_at="2026-09-15T00:00:00+00:00"))
    with pytest.raises(holdout.ProtectedHoldoutError):
        engine.seed_history("BTC-USD", poisoned)


def test_a_source_returning_protected_rows_is_refused_before_they_are_kept(
    live_module, holdout
):
    """The venue is not trusted to respect the range it was given."""
    import json as _json
    protected_open = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
    allowed_open = int(datetime(2026, 8, 31, 23, tzinfo=timezone.utc).timestamp())
    page = _json.dumps([
        [protected_open, "100", "104", "101", "103", "1.0"],
        [allowed_open, "100", "104", "101", "103", "1.0"],
    ]).encode()
    with pytest.raises(holdout.ProtectedHoldoutError):
        live_module.fetch_closed_candles(
            "BTC-USD", start="2026-08-31T22:00:00+00:00",
            end="2026-08-31T23:00:00+00:00", fetch=lambda url: page)


def test_a_poll_never_requests_protected_data(live_module, holdout):
    asked = []

    def fetch(url):
        asked.append(url)
        raise AssertionError("the poll should not have reached the network")

    with pytest.raises(holdout.ProtectedHoldoutError):
        live_module.poll_closed_candles("BTC-USD", now="2026-10-01T00:00:00+00:00",
                                        fetch=fetch)
    assert asked == []


def test_a_poll_right_before_the_boundary_only_asks_for_allowed_data(live_module):
    """The last poll of August reaches for August, and stops there."""
    captured = {}

    def fetch(url):
        captured["url"] = url
        import json as _json
        stamp = int(datetime(2026, 8, 31, 22, tzinfo=timezone.utc).timestamp())
        return _json.dumps([[stamp, "100", "104", "101", "103", "1.0"]]).encode()

    rows = live_module.poll_closed_candles(
        "BTC-USD", now="2026-08-31T23:30:00+00:00",
        since="2026-08-31T21:00:00+00:00", fetch=fetch)
    assert "2026-09" not in captured["url"]
    assert rows and all(row["bar_open_at"] < "2026-09-01" for row in rows)


def test_once_the_window_opens_the_product_stops_polling_entirely(live_module,
                                                                  holdout):
    """Not "trim the request" -- stand down. The product is embargoed."""
    def fetch(url):
        raise AssertionError("no request may be made during the embargo")

    for now in ("2026-09-01T00:00:00+00:00", "2026-09-01T00:30:00+00:00",
                "2026-11-30T23:59:59+00:00"):
        with pytest.raises(holdout.ProtectedHoldoutError):
            live_module.poll_closed_candles("BTC-USD", now=now,
                                            since="2026-08-31T21:00:00+00:00",
                                            fetch=fetch)


def test_a_restart_between_the_target_and_its_fill_still_fills_exactly_once(
    trained, engine_module, store_module, tmp_path
):
    """The pending target must survive a crash, or a decision is silently lost."""
    history, _, model = trained
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    tail = history[-3:]
    store, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-3])
    engine.start(now=clock.iso())
    engine.ingest_candle("BTC-USD", tail[0], now=clock.advance(HOUR))
    pending = engine.state("BTC-USD").pending_target
    assert pending is not None and pending["timestamp"] == tail[0]["bar_open_at"]

    # the process dies here, between the decision and the bar that fills it
    reopened = store_module.PaperEventStore(tmp_path / "paper.sqlite")
    spec = engine_module.build_session_spec({"BTC-USD": model}, products=("BTC-USD",))
    resumed = engine_module.PaperEngine(store=reopened, models={"BTC-USD": model},
                                        session_id="s1", session_spec=spec)
    resumed.restore(history={"BTC-USD": history[:-3]})
    restored = resumed.state("BTC-USD").pending_target
    assert restored is not None, "the pending target was lost across the restart"
    assert restored["timestamp"] == pending["timestamp"]
    assert restored["target_exposure"] == pending["target_exposure"]
    assert restored["position_target_hash"] == pending["position_target_hash"]

    resumed.ingest_candle("BTC-USD", tail[1], now=clock.advance(HOUR))
    fills = [e for e in reopened.events(session_id="s1", limit=2000)
             if e.event_type == "SIMULATED_FILL"]
    assert len(fills) == 1
    assert fills[0].payload["decided_at"] == pending["timestamp"]
    assert fills[0].payload["fill"]["timestamp"] == tail[1]["bar_open_at"]


def test_two_identical_events_after_different_predecessors_hash_differently(
    store_module, tmp_path
):
    """The chain is what makes truncation and reordering detectable.

    A hash that ignored its predecessor would still verify against itself, so
    the property has to be checked across sessions with identical content.
    """
    # Two events identical in EVERY field except the predecessor. Only the
    # hash function itself can isolate this; a stored pair would also differ by
    # session or sequence and the comparison would prove nothing.
    fields = dict(event_id=1, session_id="s", sequence=2,
                  event_type="CANDLE_INGESTED", event_at="t1", product="BTC-USD",
                  natural_key="bar", payload={"k": 2}, event_hash="")
    first = store_module.PaperEvent(previous_event_hash="a" * 64, **fields)
    second = store_module.PaperEvent(previous_event_hash="b" * 64, **fields)
    assert first.body()["payload"] == second.body()["payload"]
    assert first.recomputed_hash() != second.recomputed_hash(), (
        "the event hash does not depend on its predecessor; the chain is decorative")

    # and the stored chain really does link, end to end
    store = store_module.PaperEventStore(tmp_path / "link.sqlite")
    genesis = store.append(session_id="s", event_type="SESSION_STARTED", event_at="t0",
                           natural_key="session", payload={"k": 1})
    follower = store.append(session_id="s", event_type="CANDLE_INGESTED", event_at="t1",
                            product="BTC-USD", natural_key="bar", payload={"k": 2})
    assert follower.previous_event_hash == genesis.event_hash
    assert follower.recomputed_hash() == follower.event_hash


def test_the_engine_never_retrains_the_shadow_model(trained, engine_module,
                                                    store_module, tmp_path,
                                                    monkeypatch, model_module):
    """V1 freezes the model before the first live bar and leaves it alone.

    Retraining online would quietly turn a fixed shadow model into a rolling
    research loop, on data nobody agreed to spend.
    """
    history, _, model = trained

    def detonate(*args, **kwargs):
        raise AssertionError("the shadow engine retrained a model")

    monkeypatch.setattr(model_module, "train_paper_model", detonate)
    clock = FakeClock(datetime(2026, 8, 11, 3, tzinfo=timezone.utc))
    _, engine = _engine(engine_module, store_module, tmp_path, model)
    engine.seed_history("BTC-USD", history[:-3])
    engine.start(now=clock.iso())
    for row in history[-3:]:
        engine.ingest_candle("BTC-USD", row, now=clock.advance(HOUR))
    # the same fitted state throughout: one model, frozen before the session
    assert engine.models["BTC-USD"].fitted.fitted_model_hash == \
        model.fitted.fitted_model_hash
