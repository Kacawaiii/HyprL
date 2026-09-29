"""Store read cost of derivations and snapshots: one consistent in-memory view per operation, fed
incrementally, instead of re-reading the store per record or per item. The answers at every (T, H)
are those of direct store reads; the read counts are deterministic and do not grow with the store."""

from __future__ import annotations

from datetime import timedelta

import pytest

from scripts.trading_lab.fomc import processing, snapshot, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.store import FomcStore

from tests.crypto.fomc_support import P1, Env, statement_item

P2 = syn.statement_path("20260729")


def _env(root, minutes):
    env = Env(root)
    env.feed([statement_item(), statement_item(P2, guid="g2")])
    env.provider.routes[P1] = syn.page_response()
    env.provider.routes[P2] = syn.page_response()
    env.drive(minutes * 60, idle=30)
    return env


@pytest.fixture(scope="module")
def envs(tmp_path_factory):
    built = {m: _env(tmp_path_factory.mktemp(f"m{m}"), m) for m in (20, 40)}
    yield built
    for env in built.values():
        env.provider.close()


class _Direct:
    """Every read goes to SQLite, bounded by a horizon: the pre-view read path."""

    def __init__(self, store, upto):
        self.store, self.upto = store, upto

    def _limit(self, upto):
        return self.upto if upto is None else min(upto, self.upto)

    def horizon(self):
        return self.upto

    def view(self, upto=None):
        return _Direct(self.store, self._limit(upto))

    def rows(self, kind=None, *, key=None, upto=None):
        return FomcStore.rows(self.store, kind, key=key, upto=self._limit(upto))

    def select(self, kind, field, value, *, upto=None):
        return [r for r in self.rows(kind, upto=upto) if r.body.get(field) == value]

    def txns(self, *, upto=None):
        return FomcStore.txns(self.store, upto=self._limit(upto))

    def row_at(self, kind, seq):
        return next((r for r in self.rows(kind, upto=seq) if r.seq == seq), None)

    def read_raw(self, digest):
        return self.store.read_raw(digest)


def _reads(store, fn):
    store.reads.update(queries=0, rows=0)
    fn()
    return dict(store.reads)


def _resolved_instants(store, H):
    table = state.availability(store, H)
    return sorted({e.avail - timedelta(seconds=1) for e in table if e.resolved} |
                  {e.avail for e in table if e.resolved})[::3]


def test_view_answers_equal_direct_reads_at_every_t_h(envs, monkeypatch):
    store = envs[20].store
    horizons = [store.horizon() // 3, 2 * store.horizon() // 3, store.horizon()]
    cases = [(T, H) for H in horizons for T in _resolved_instants(store, H)]
    via_view = [snapshot.events_as_of(store, T, H) for T, H in cases]
    outstanding_view = [state.outstanding(store, upto=H) for H in horizons]
    monkeypatch.setattr(FomcStore, "view", lambda self, upto=None: _Direct(self, self.horizon() if upto is None else min(upto, self.horizon())))
    via_sql = [snapshot.events_as_of(store, T, H) for T, H in cases]
    assert len(cases) > 20
    assert via_view == via_sql
    assert {s["read_state"] for s in via_view} == {"FOMC_RESOLVED", "FOMC_CAUSAL_VISIBILITY_UNRESOLVED"}
    assert outstanding_view == [state.outstanding(store, upto=H) for H in horizons]


@pytest.mark.parametrize("minutes", [20, 40])
def test_a_cold_snapshot_loads_the_store_once(envs, minutes):
    env = envs[minutes]
    H = env.store.horizon()
    T = _resolved_instants(env.store, H)[-1]
    reader = FomcStore(env.root, wall_clock=env.clock.wall)  # a separate reader, empty mirror
    try:
        cost = _reads(reader, lambda: snapshot.events_as_of(reader, T, H))
        assert cost == {"queries": 3, "rows": len(reader.rows(upto=H)) + len(reader.txns(upto=H))}  # horizon, rows, txns
        again = _reads(reader, lambda: (snapshot.events_as_of(reader, T, H), state.outstanding(reader, upto=H)))
        assert again == {"queries": 2, "rows": 0}  # one horizon check per operation, no re-read
    finally:
        reader.close()


def test_derivation_and_capture_reads_do_not_grow_with_the_store(envs):
    costs = {}
    for minutes, env in envs.items():
        store = env.store
        H = store.horizon()
        feed = state.feed_responses(store)[-1]
        derivations = _reads(store, lambda: (state.outstanding(store, upto=H),
                                             processing.cycle_conclusion(store, feed, [], None),
                                             env.collector.eligible_work(), env.collector.derive_terminals(),
                                             env.collector.process_pending()))
        before_rows, before_txns = len(store.rows()), len(store.txns())
        capture = _reads(store, lambda: env.drive(300, idle=30))
        added = len(store.rows()) - before_rows + len(store.txns()) - before_txns
        costs[minutes] = (derivations, capture, added)
    (d20, c20, a20), (d40, c40, a40) = costs[20], costs[40]
    assert d20 == d40 and d20["rows"] == 0  # the same derivations cost the same on a store twice as large
    assert a20 == a40 and c20 == c40  # the same capture work reads the same rows, whatever the store size
    assert c20["rows"] <= 3 * a20  # new rows are read about once, keyed lookups aside
