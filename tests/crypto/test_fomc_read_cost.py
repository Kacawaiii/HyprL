"""Store read cost and in-memory work of derivations and snapshots: one consistent view per operation,
fed incrementally, and incremental aggregates instead of rescans per record or per item. The answers
at every (T, H), the selection sequence and replay equal those of direct SQL reads with full scans;
the counts are deterministic. In-memory work is counted as the rows a view hands out."""

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
    """Every read goes to SQLite, bounded by a horizon, and every derivation rescans: the pre-view path
    (never at the head, so the head aggregates are never used; aggregates rebuilt from scratch)."""

    def __init__(self, store, upto):
        self.store, self.upto = store, upto

    def at_head(self):
        return False

    def read_aggregate(self, name, factory, read, *, head_only=False):
        if head_only:
            return None
        agg = factory()
        for row in self.rows():
            agg.add(row)
        return read(agg)

    def memo(self, key, compute):
        return compute()

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
    store.reads.update(queries=0, rows=0, view_rows=0)
    fn()
    return dict(store.reads)


def _direct(monkeypatch):
    monkeypatch.setattr(FomcStore, "view", lambda self, upto=None: _Direct(self, self.horizon() if upto is None else min(upto, self.horizon())))


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
    _direct(monkeypatch)
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
        expected = len(reader.rows(upto=H)) + len(reader.txns(upto=H))
        assert (cost["queries"], cost["rows"]) == (3, expected)  # horizon, rows, txns
        again = _reads(reader, lambda: (snapshot.events_as_of(reader, T, H), state.outstanding(reader, upto=H)))
        assert (again["queries"], again["rows"]) == (2, 0)  # one horizon check per operation, no re-read
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
    sql = ("queries", "rows")
    assert [d20[k] for k in sql] == [d40[k] for k in sql] and d20["rows"] == 0  # same SQLite cost on a larger store
    assert a20 == a40 and [c20[k] for k in sql] == [c40[k] for k in sql]  # the same capture work reads the same rows
    assert c20["rows"] <= 3 * a20  # new rows are read about once, keyed lookups aside


# ------------------------------------------------------------------ in-memory work ---------------
PATHS = [syn.statement_path(f"202601{d:02d}") for d in range(1, 13)]


@pytest.fixture(scope="module")
def items_envs(tmp_path_factory):
    built = {}
    for minutes in (20, 40):
        env = Env(tmp_path_factory.mktemp(f"items{minutes}"))
        env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate(PATHS)])
        for p in PATHS:
            env.provider.routes[p] = syn.page_response()
        env.drive(minutes * 60, idle=30)
        built[minutes] = env
    yield built
    for env in built.values():
        env.provider.close()


def _work(env):
    store, collector = env.store, env.collector
    H = store.horizon()
    T = _resolved_instants(store, H)[-1]
    feed = state.feed_responses(store)[-1]
    ops = {"events_as_of": lambda: snapshot.events_as_of(store, T, H),
           "outstanding": lambda: state.outstanding(store, upto=H),
           "cycle_conclusion": lambda: processing.cycle_conclusion(store, feed, [], None),
           "open_episodes": collector.open_episodes, "eligible_work": collector.eligible_work,
           "derive_terminals": collector.derive_terminals, "process_pending": collector.process_pending,
           "feed_poll_due": collector.feed_poll_due}
    return {name: _reads(store, op)["view_rows"] for name, op in ops.items()}, len(store.rows()) + len(store.txns())


def test_in_memory_work_has_no_items_times_records_term(items_envs):
    """12 items; the 40-min store holds the same items and more feed cycles. Before this change the
    40-min store cost eligible_work 1627 rows instead of 1187 (+440 for +140 rows: one scan of every
    response per item) and open_episodes 900 instead of 660."""
    (w20, n20), (w40, n40) = _work(items_envs[20]), _work(items_envs[40])
    for name in ("outstanding", "cycle_conclusion", "open_episodes", "derive_terminals", "process_pending", "feed_poll_due"):
        assert w20[name] == w40[name], name  # per-item work only: independent of the store's history
    for name in ("events_as_of", "eligible_work"):  # the availability table: linear, once, never per item
        assert 0 <= w40[name] - w20[name] <= n40 - n20, name
    assert w20["process_pending"] == w20["feed_poll_due"] == 0  # answered by the head aggregates


def test_selection_snapshots_and_replay_equal_the_full_scan_path(tmp_path, monkeypatch):
    def run(root):
        env = Env(root)
        try:
            env.feed([statement_item(p, guid=f"g{i}") for i, p in enumerate(PATHS[:6])])
            for p in PATHS[:6]:
                env.provider.routes[p] = syn.page_response()
            del env.provider.routes[PATHS[0]]  # one failing item: retries and backoff
            env.drive(1500, idle=30)
            H = env.store.horizon()
            T = _resolved_instants(env.store, H)[-1]
            invoked = [(t.seq, t.body["class"], t.body["kind"], t.body.get("sid"), t.body.get("episode_key"), t.body["grant_mono"])
                       for t in env.store.rows("TRANSPORT_INVOKED")]
            return invoked, snapshot.events_as_of(env.store, T, H), snapshot.replay(env.store, T, H)
        finally:
            env.provider.close()
    fast = run(tmp_path / "fast")
    _direct(monkeypatch)
    slow = run(tmp_path / "slow")
    assert fast[0] == slow[0] and len(fast[0]) > 30  # the same selection decisions at the same instants
    assert fast[1] == slow[1] and fast[2] == slow[2] == fast[1]
