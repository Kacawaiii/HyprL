"""Synthetic boundary tests: leakage, selection, trial accounting and chance."""
from copy import deepcopy
from datetime import date, timedelta
import json
import sqlite3

import pytest

from scripts.trading_lab.research_loop.core import (
    TrialLedger, audit, evaluate_trial, promotion_gate,
)
from scripts.trading_lab.research_loop.ingest import book_inventory, v1_paper, v1_predictions, v2_closed
from scripts.trading_lab.research_loop.stats import (
    binomial_tail, brier_null, paired_test, random_entry, required_decisions,
)


def decision():
    return {'decision_id': 'synthetic-1', 'decision_hash': 'a' * 64, 'source_hash': 'b' * 64,
            'strategy': 'synthetic', 'symbol': 'SYNTH', 'decision_at': '2025-01-02T12:00:00Z',
            'entry_at': '2025-01-02T14:30:00Z', 'exit_at': '2025-01-03T21:00:00Z',
            'action': 'UP', 'weight': .05,
            'facts': {'earnings_at': {'value': '2025-01-03T08:00:00Z',
                                     'available_at': '2025-01-01T00:00:00Z',
                                     'observed_at': '2025-01-01T00:00:00Z'}},
            'outcome': {'available_at': '2025-01-03T21:05:00Z', 'net_contribution': -.01,
                        'cost_basis': 'NET_FULL_COSTS'}}


def test_loss_does_not_prove_cause_and_late_facts_never_become_known():
    row = decision()
    row['facts']['market_cap_usd'] = {'value': 1e9, 'available_at': '2025-01-02T12:00:01Z',
                                    'observed_at': '2025-01-01T00:00:00Z'}
    before = deepcopy(row)
    result = audit(row)
    assert result['known_at_T'] == {'earnings_at': '2025-01-03T08:00:00Z'}
    assert result['not_available_at_T'] == ['market_cap_usd']
    assert result['error_classes'] == ['unidentified_requires_evidence']
    assert row == before


@pytest.mark.parametrize('symbol', ['BTC-USD', 'ETH-USD', 'BTC/USD', 'ETHUSD'])
def test_holdout_is_rejected_before_outcome(symbol):
    row = decision()
    row.update(symbol=symbol, entry_at='2026-08-31T14:30:00Z', exit_at='2026-09-01T21:00:00Z')
    # No outcome field can be accessed to evade the guard.
    row['outcome'] = object()
    with pytest.raises(ValueError, match='PROTECTED_HOLDOUT'):
        audit(row)


def test_validation_is_never_used_as_development_and_blocked_trials_count(tmp_path):
    row = decision()
    row.update(entry_at='2026-07-01T14:30:00Z', exit_at='2026-07-02T21:00:00Z')
    ledger = TrialLedger(tmp_path / 'trials.sqlite')
    try:
        a = evaluate_trial([row], 'H01', ledger)
        b = evaluate_trial([], 'H02', ledger)
        assert a['status'] == b['status'] == 'BLOCKED'
        assert a['metric'] is None and not a['passes_development']
        assert a['exclusions'] == {'OUTSIDE_DEVELOPMENT': 1}
        assert len(ledger.reservations()) == 2
        assert b['family_size'] == 8
        ledger.verify()
    finally:
        ledger.db.close()


def test_missing_feature_blocks_whole_trial_instead_of_deleting_a_loser(tmp_path):
    row = decision()
    row['decision_id'] = 'synthetic-2'
    del row['facts']['earnings_at']
    ledger = TrialLedger(tmp_path / 'trial.sqlite')
    try:
        result = evaluate_trial([decision(), row], 'H01', ledger)
        assert result['status'] == 'BLOCKED'
        assert result['eligible_dev'] == 2
        assert result['metric'] is None
    finally:
        ledger.db.close()


def test_trial_metric_keeps_cash_decisions_and_counts_every_reserved_variant(tmp_path):
    ledger = TrialLedger(tmp_path / 'trials.sqlite')
    try:
        for _ in range(8):
            ledger.reserve('H01', 'c' * 64)
        result = evaluate_trial([decision()], 'H01', ledger)
        assert result['metric']['n'] == 1
        assert result['metric']['mean_delta'] == .01
        assert result['family_size'] == 9
        assert result['metric']['p_adjusted'] >= result['metric']['p_raw']
        assert not result['passes_development']
        with pytest.raises(sqlite3.IntegrityError, match='append only'):
            ledger.db.execute("UPDATE events SET body='{}' WHERE seq=1")
    finally:
        ledger.db.close()


def test_validation_reservation_is_single_use_and_touched_is_explicit(tmp_path):
    ledger = TrialLedger(tmp_path / 'trials.sqlite')
    try:
        with pytest.raises(ValueError, match='ALREADY_TOUCHED'):
            ledger.reserve_validation('r', 'd')
        ledger.reserve_validation('r', 'd', acknowledge_touched=True)
        with pytest.raises(ValueError, match='SINGLE_USE_EXHAUSTED'):
            ledger.reserve_validation('r', 'd', acknowledge_touched=True)
    finally:
        ledger.db.close()


def test_coin_tail_and_power_are_independent_worked_examples():
    # Binomial(10, .5): C(10,8)+C(10,9)+C(10,10) = 56 out of 1024.
    assert binomial_tail(8, 10) == pytest.approx(.0546875)
    assert binomial_tail(0, 10) == 1
    assert required_decisions() == 783
    assert required_decisions(family=8) > required_decisions()


def test_zero_effect_cannot_pass_even_after_many_tries():
    result = paired_test([('2025-01-02', 0.0)] * 100, family=100)
    assert result['sessions'] == 1
    assert result['p_adjusted'] == 1
    result = brier_null([.5] * 10, [0, 1] * 5)
    assert result['brier'] == .25 and result['p_null'] == 1


def test_random_entry_uses_10000_draws_and_refuses_survivor_only_panel():
    slot = {'entry_at': '2025-01-02T14:30:00Z', 'exit_at': '2025-01-03T21:00:00Z',
            'decision_at': '2025-01-02T12:00:00Z',
            'eligibility_known_at': '2025-01-01T00:00:00Z', 'universe_hash': 'a' * 64,
            'eligible_symbols': ['SYNTH_A', 'SYNTH_B'], 'real_symbol': 'SYNTH_A',
            'candidate_net_returns': {'SYNTH_A': .01, 'SYNTH_B': -.01}, 'real_net_return': .01}
    result = random_entry([slot])
    assert result['status'] == 'COMPUTED' and result['draws'] == 10000
    assert result['percentile'] == 100
    assert .47 < result['p_null'] < .53
    del slot['candidate_net_returns']['SYNTH_B']
    assert random_entry([slot])['status'] == 'BLOCKED'


def test_protected_trial_attempt_is_counted_without_reading_its_outcome(tmp_path):
    row = decision()
    row.update(symbol='BTCUSD', entry_at='2026-09-01T00:00:00Z', exit_at='2026-09-02T00:00:00Z')
    row['outcome'] = object()  # not serializable: hashing it would violate the boundary
    ledger = TrialLedger(tmp_path / 'trials.sqlite')
    try:
        result = evaluate_trial([row], 'H01', ledger)
        assert result['reason'] == 'PROTECTED_HOLDOUT'
        assert len(ledger.reservations()) == 1
    finally:
        ledger.db.close()


def test_random_baseline_never_samples_holdout_from_the_universe():
    result = random_entry([{'entry_at': '2026-09-01T00:00:00Z', 'exit_at': '2026-09-02T00:00:00Z',
                            'eligible_symbols': ['SYNTH', 'BTCUSD'], 'candidate_net_returns': object()}])
    assert result['reason'] == 'PROTECTED_HOLDOUT' and result['draws'] == 0


def test_duplicate_opportunity_cannot_inflate_development_sample(tmp_path):
    ledger = TrialLedger(tmp_path / 'trials.sqlite')
    try:
        result = evaluate_trial([decision(), decision()], 'H01', ledger)
        assert result['reason'] == 'DUPLICATE_DECISION_IDENTITIES'
    finally:
        ledger.db.close()


def test_old_implementation_trial_binding_stays_rejected(tmp_path, monkeypatch):
    from scripts.trading_lab.research_loop import core
    path = tmp_path / 'trials.sqlite'
    ledger = TrialLedger(path)
    ledger.db.close()
    monkeypatch.setattr(core, 'implementation_hash', lambda: 'changed')
    with pytest.raises(ValueError, match='OLD_TRIAL_BINDING_REJECTED'):
        TrialLedger(path)


def test_acknowledged_book_close_is_not_a_fill(tmp_path):
    journal = tmp_path / 'synthetic.jsonl'
    journal.write_text(json.dumps({'action': 'closed', 'symbol': 'SYNTH', 'order_id': 'fake'}) + '\n')
    result = book_inventory(journal)
    assert result['events']['closed'] == 1
    assert result['proven_closed_fills'] == 0
    assert result['status'].startswith('BLOCKED')


def test_cumulative_broker_exit_is_counted_once_per_lot(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE records(sequence INTEGER PRIMARY KEY, payload TEXT)')
    base = {'account': 'ia_actions', 'event': 'trade_outcome', 'lot_id': 'lot',
            'prediction_hash': 'a' * 64, 'symbol': 'SYNTH'}
    events = [dict(base, purpose='entry', state='FILLED', entry_at='2025-01-02T14:30:00Z'),
              dict(base, purpose='exit', state='CLOSED', at='2025-01-03T21:01:00Z',
                   exit_at='2025-01-03T21:00:00Z', raw_return=.01, lot_pnl_before_broker_fees=10)]
    for r in events + events[-1:]:
        db.execute('INSERT INTO records(payload) VALUES(?)', (json.dumps(r),))
    db.commit(); db.close()
    rows, issues = v1_paper(path, [decision()])
    assert len(rows) == 1 and not issues
    assert rows[0]['outcome']['net_contribution'] is None


def test_forward_promotion_needs_40_distinct_sessions_and_stress_survival():
    pairs = []
    for i in range(40):
        pairs.append({'decision_id': str(i), 'session': str(date(2025, 1, 1) + timedelta(days=i)),
                      'mode': 'SHADOW_AB', 'mature': True, 'risk_ok': True,
                      'current_net': 0., 'candidate_net': .01, 'candidate_stressed_net': .005,
                      'rule_hash': 'frozen-rule', 'registered_at': '2024-12-31T00:00:00Z',
                      'decision_at': '2025-01-01T12:00:00Z'})
    receipt = {'completed': True, 'role': 'EXPLORATORY_TOUCHED_VALIDATION', 'rule_hash': 'frozen-rule'}
    assert not promotion_gate(pairs[:39], preregistered=True, validation_receipt=receipt)
    assert promotion_gate(pairs, preregistered=True, validation_receipt=receipt)
    pairs[0]['candidate_stressed_net'] = -10
    assert not promotion_gate(pairs, preregistered=True, validation_receipt=receipt)


def test_v2_closed_shadow_joins_original_approval_and_verifies_chain(tmp_path):
    import hashlib
    path = tmp_path / 'synthetic.sqlite'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY, at TEXT, kind TEXT, body TEXT, previous TEXT, hash TEXT)')
    plan = {'plan_id': 'plan', 'symbol': 'SYNTH', 'playbook': 'primary_news', 'layer': 'news',
            'snapshot_hash': 'a' * 64, 'holding_days': 2}
    records = [('2025-01-01T00:00:00Z', 'binding', {'mode': 'SHADOW_FORWARD', 'config': {'initial_equity': 100000}}),
               ('2025-01-02T12:00:00Z', 'approved', plan),
               ('2025-01-02T14:30:00Z', 'entry_fill', dict(plan, filled_at='2025-01-02T14:30:00Z', qty=10, entry=100)),
               ('2025-01-03T21:00:00Z', 'exit_fill', dict(plan, closed_at='2025-01-03T21:00:00Z', pnl=20, r=1))]
    previous = '0' * 64
    for seq, (at, kind, body) in enumerate(records, 1):
        row = {'seq': seq, 'at': at, 'kind': kind, 'body': body, 'previous': previous}
        hashed = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)', (seq, at, kind, json.dumps(body), previous, hashed))
        previous = hashed
    db.commit()
    rows, meta = v2_closed(path)
    assert len(rows) == 1 and meta['closed'] == 1
    assert rows[0]['outcome']['net_contribution'] == .0002
    assert audit(rows[0])['known_at_T']['playbook'] == 'primary_news'
    db.execute("UPDATE events SET body='{}' WHERE seq=4")
    db.commit(); db.close()
    with pytest.raises(ValueError, match='SOURCE_CHAIN_CORRUPT'):
        v2_closed(path)


def test_v1_crypto_uses_original_raw_target_with_null_spy_label(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE records(identity TEXT, kind TEXT, payload TEXT)')
    p = {'synthetic': False, 'product': 'SOL-USD', 'decision_at': '2025-01-02T12:00:00Z',
         'model_id': 'synthetic-reader', 'proposed_position': {'weight': -.05},
         'snapshot_hash': 's', 'features_hash': 'f',
         'signal': {'label_definition': {'entry_at': '2025-01-02T14:30:00Z', 'exit_at': '2025-01-03T21:00:00Z',
                                         'horizon': '1d', 'targets': ['raw']},
                    'view': {'view': 'DOWN', 'p_outperform': .4}}}
    label = {'prediction_hash': 'p', 'product': 'SOL-USD', 'available_at': '2025-01-03T21:05:00Z',
             'value': {'net_unit_pnl': -.012, 'raw_return': .01, 'spy_relative_return': None, 'cost_roundtrip': .002}}
    for identity, kind, payload in [('p', 'prediction', p), ('l', 'label', label)]:
        db.execute('INSERT INTO records VALUES(?,?,?)', (identity, kind, json.dumps(payload)))
    db.commit(); db.close()
    row, = v1_predictions(path)
    assert row['target'] == 'raw' and row['outcome']['y'] == 1
    assert row['outcome']['net_contribution'] == pytest.approx(-.0006)
