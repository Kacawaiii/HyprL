from datetime import timedelta
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench import labels, marketdata as md, snapshot as sn
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.replay_seal_forward import fills, forward, score, spec


@pytest.mark.parametrize('day,opening', [('2026-03-09', '13:30:30'), ('2026-11-02', '14:30:30')])
def test_dst_labels_and_crypto_follow_decision_and_session(day, opening):
    T = forward.decision_time(day)
    anchors = SimpleNamespace(stock=lambda d, at, names: {s: {'p': 100} for s in names}, crypto=lambda at: {})
    calls = []
    def stocks(d, at, names):
        calls.append(at)
        return {s: {'p': 100} for s in names}
    anchors.stock = stocks
    labels.label_universe_day(day, {}, anchors, spec.spec()['costs'], decision_at=T, last_day=day)
    assert calls[0] == opening
    assert fills.entry(day, T, False) == instant(day + 'T' + opening + 'Z')
    assert fills.entry(day, T, True) == T + timedelta(minutes=15)
    assert T < fills.entry(day, T, True) < fills.entry(day, T, False)
    assert forward.decision_time('2026-11-02').hour == 14
    assert forward.decision_time('2026-03-09').hour == 13


def test_outcomes_refuses_entry_at_or_before_T_even_for_cached_result():
    outcomes = score.Outcomes.__new__(score.Outcomes)
    outcomes.cache = {('ab', '2026-11-02', 'AAPL', 1): {'rel': 1}}
    outcomes.decision_time = lambda day: fills.stock_entry(day)
    with pytest.raises(ReplayError, match='ENTRY_NOT_AFTER_DECISION'):
        outcomes.get('ab', '2026-11-02', 'AAPL', 1)


class Vendor:
    def __init__(self):
        self.calls = []
    def get(self, origin, path, params):
        self.calls.append((path, params))
        names = params['symbols'].split(',')
        if 'trades' in path:
            return {'trades': {s: [{'t': params['start'], 'p': 100 if s == 'SPY' else 110}] for s in names}}, {}
        if path == '/v2/stocks/bars':
            rows = [{'t': iso(s.open_at), 'o': 100, 'c': 121 if s.session_date >= '2026-11-03' else 100,
                     'h': 125, 'l': 95, 'v': 1e6} for s in md.sessions('2026-05-01', '2026-11-06')]
            return {'bars': {n: [dict(r, c=100) if n == 'SPY' else r for r in rows] for n in names}}, {}
        return {'bars': {}}, {}


def test_news_forward_multiday_uses_actual_30s_fill_and_no_replay_end_date():
    vendor = Vendor()
    outcomes = score.Outcomes(vendor, '2026-11-02', '2026-11-06', spec.spec()['costs'], decision_time=forward.decision_time)
    result = outcomes.get('cd', '2026-11-02', 'SMLL', 2)
    assert result['entry_at'] == '2026-11-02T14:30:30Z'
    assert result['rel'] == pytest.approx(.1)  # 121 / 110; the daily open was 100.
    assert sum(result['gross_path'].values()) == pytest.approx(.1)
    assert result['exit_day'] == '2026-11-03'
    assert result['corporate_action'] is False
    assert any(p['start'] == result['entry_at'] for path, p in vendor.calls if 'trades' in path)


def test_hedge_leg_borrow_hard_to_borrow_and_diagnostic_labels():
    costs = spec.spec()['costs']
    flags = {'shortable': True, 'easy_to_borrow': True}
    outcome = {'rel': .04, 'vol': .02, 'cost': .006, 'gross_path': {'2026-07-17': .01, '2026-07-20': .03},
               'cluster': 'smallmid', 'entry_day': '2026-07-17', 'exit_day': '2026-07-20', 'corporate_action': False}
    (result, path), basis = score.signed_outcome(outcome, 'UP', 2, costs, flags, flags)
    expected = .006 + fills.hedge_cost(costs) + .03 * 3 / 365
    assert result['cost'] == pytest.approx(expected) and basis == 'NET_FULL_COSTS'
    assert result['net'] == pytest.approx(.04 - expected) == pytest.approx(sum(path.values()))
    assert score.signed_outcome(outcome, 'DOWN', 2, costs, dict(flags, easy_to_borrow=False), flags) == (None, 'HARD_TO_BORROW')
    assert score.signed_outcome(outcome, 'UP', 2, costs, flags, dict(flags, shortable=False)) == (None, 'HARD_TO_BORROW')
    assert score.signed_outcome(dict(outcome, corporate_action=True), 'UP', 2, costs, flags, flags) == (None, 'CORPORATE_ACTION_WINDOW')
    assert score.signed_outcome(dict(outcome, corporate_action=None), 'UP', 2, costs, flags, flags)[1] == 'GROSS_DIAGNOSTIC'
    assert score.signed_outcome(outcome, 'UP', 2, costs, flags)[1] == 'GROSS_DIAGNOSTIC'


@pytest.mark.parametrize('ratio', [1.01, 2.0])
def test_split_and_dividend_windows_are_detected(ratio):
    assert fills.corporate_action({'2026-07-15': 1, '2026-07-16': ratio}, '2026-07-15', '2026-07-16')
    assert fills.corporate_action({'2026-07-15': 1}, '2026-07-15', '2026-07-16') is None


def test_missing_required_outcome_cache_is_shared_exclusion_but_hash_mismatch_blocks():
    class Missing:
        def get(self, *args):
            raise ReplayError('OFFLINE_CACHE_MISS')
    outcomes = score.Outcomes(Missing(), '2026-07-01', '2026-08-31', spec.spec()['costs'])
    for asset in ('AAPL', 'SOL-USD'):
        assert outcomes.get('ab', '2026-07-15', asset, 1) is None
        assert outcomes.exclusion_reasons[('ab', '2026-07-15', asset, 1)] == 'OFFLINE_CACHE_MISS'
    class Corrupt:
        def get(self, *args):
            raise ReplayError('CACHE_HASH_MISMATCH')
    with pytest.raises(ReplayError, match='CACHE_HASH_MISMATCH'):
        score.Outcomes(Corrupt(), '2026-07-01', '2026-08-31', spec.spec()['costs'])


def test_missing_sealed_request_is_data_quality_but_bound_object_damage_is_integrity(tmp_path):
    import hashlib
    from scripts.trading_lab.replay_seal_forward.quality import DataTransport
    from scripts.trading_lab.replay_bench.grant import Transport, Budget
    transport = Transport(SimpleNamespace(payload={'scope': {}}, check=lambda: None), Budget(tmp_path),
                          tmp_path/'cache', offline=True)
    transport.allowed_cache = {}
    source = DataTransport(transport)
    origin, path = md.ALPACA, '/v2/stocks/bars'
    with pytest.raises(ReplayError, match='SEALED_SOURCE_DATA_MISSING'):
        source.get(origin, path)
    # A once-bound object that vanished is an integrity failure; no exclusion hides it.
    transport.allowed_cache[hashlib.sha256((origin+path).encode()).hexdigest()] = 'a'*64
    with pytest.raises(ReplayError, match='PREDICTION_CACHE_NOT_SEALED'):
        source.get(origin, path)
