from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from scripts.trading_lab.research.contracts import Hypothesis, Trial
from scripts.trading_lab.research.proposals import propose, comparison_hypothesis, comparison_diagnostic, pair_decisions
from scripts.trading_lab.comparison_protocol import build_v2
from scripts.trading_lab.research.store import ResearchStore, IntegrityError
from tests.research.conftest import AT, RECORDED


def test_local_proposals_are_deterministic_bounded_complete_and_exploratory(dataset):
    one, two = propose(dataset), propose(dataset)
    assert [p['hypothesis'].identity for p in one] == [p['hypothesis'].identity for p in two]
    assert len(one) == 2 and all(p['hypothesis'].synthetic for p in one)
    for p in one:
        h, e = p['hypothesis'], p['prepared']
        assert Hypothesis.from_dict(h.to_dict()).identity == h.identity
        assert h.scope == 'EXPLORATORY' and h.population['studied_window']
        assert h.transformations['fit_on'] == 'TRAIN_ONLY'
        assert h.splits['purge_seconds'] == 14400 and h.splits['embargo_seconds'] == 3600
        assert h.decision_criteria == e.decision_criteria and h.costs == e.costs
        with pytest.raises(TypeError):
            h.features['columns'][0] = 'mutated'
    with pytest.raises(ValueError):
        propose(dataset, limit=3)
    with pytest.raises(ValueError):
        propose(dataset, max_trials=9)


@pytest.mark.parametrize('field,value', [('statement', ''), ('falsification', ''), ('mechanism', ''),
    ('transformations', {'fit_on': 'ALL'}), ('budgets', {'max_trials': 9, 'max_rows': 600, 'wall_seconds': 120}),
    ('scope', 'CONFIRMATORY_RESULT'), ('horizon_seconds', 0)])
def test_invalid_plans_refused(dataset, field, value):
    h = propose(dataset, limit=1)[0]['hypothesis']
    with pytest.raises(ValueError):
        replace(h, **{field: value})


def test_protocol_registry_consumes_exact_v2_and_cannot_execute(store):
    h, p = comparison_hypothesis(), build_v2()
    store.register(h, recorded_at=AT)
    assert h.protocol_hash == p['protocol_hash']
    assert h.to_dict()['decision_criteria'] == p['evaluation']
    assert h.to_dict()['features'] == p['pairing']['features']
    assert h.to_dict()['splits'] == p['calendar']['split']
    assert h.to_dict()['costs'] == p['pairing']['frozen_reused']
    assert h.population['studied_windows'] == 'EXPLORATORY'
    assert h.budgets['execution_enabled'] is False
    d = comparison_diagnostic()
    assert d['state'] == 'WAITING_DATA' and not d['execution_enabled']
    assert d['paired_decisions'] == 0 and len(d['actions']) == 4


@pytest.mark.parametrize('outcome,state', [('NULL', 'COMPLETE'), ('NEGATIVE', 'COMPLETE'),
                                         ('ABANDONED', 'ABANDONED'), ('ERROR', 'FAILED')])
def test_null_negative_abandoned_and_failed_trials_retained(store, dataset, outcome, state):
    p = propose(dataset, limit=1)[0]
    h, e = p['hypothesis'], p['prepared']
    store.register(h, recorded_at=AT)
    first = store.start_trial(h, e, trial_id='synthetic-trial', recorded_at=AT)
    store.observe_trial('synthetic-trial', state='RUNNING', outcome='PENDING', evidence={}, recorded_at=AT)
    store.observe_trial('synthetic-trial', state=state, outcome=outcome, evidence={'sample': 23}, recorded_at=RECORDED)
    history = store.records('trial', object_id='synthetic-trial')
    assert history[0]['identity'] == first
    assert [x['payload']['state'] for x in history] == ['PREPARED', 'RUNNING', state]
    assert all(x['payload']['criteria_hash'] == h.criteria_hash for x in history)
    assert store.get(first)['payload']['outcome'] == 'PENDING'
    with pytest.raises(ValueError, match='terminal'):
        store.observe_trial('synthetic-trial', state='COMPLETE', outcome='POSITIVE', evidence={})
    assert store.verify()['verified']


def test_frozen_plans_criteria_and_trial_budgets_cannot_be_changed(store, dataset):
    proposal = propose(dataset, limit=1, max_trials=1)[0]
    h, e = proposal['hypothesis'], proposal['prepared']
    store.register(h)
    with pytest.raises(ValueError, match='immutable'):
        store.register(replace(h, decision_criteria={'rule': 'easier'}))
    with pytest.raises(ValueError, match='criteria'):
        store.start_trial(h, replace(e, decision_criteria={'rule': 'easier'}), trial_id='bad')
    with pytest.raises(ValueError, match='configuration'):
        store.start_trial(h, replace(e, costs={'fee_rate': 0}), trial_id='bad')
    store.start_trial(h, e, trial_id='first')
    reopened = ResearchStore(store.root)
    with pytest.raises(ValueError, match='budget'):
        reopened.start_trial(h, e, trial_id='second')


def test_concurrent_trial_reservations_respect_persistent_budget(store, dataset):
    proposal = propose(dataset, limit=1, max_trials=1)[0]
    h, e = proposal['hypothesis'], proposal['prepared']
    store.register(h)
    def reserve(i):
        try:
            store.start_trial(h, e, trial_id='synthetic-' + str(i))
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sum(workers.map(reserve, range(2))) == 1
    assert len(store.records('trial')) == 1


def test_pairing_requires_exact_decisions_labels_costs_and_splits():
    row = {'product': 'BTC-USD', 'decision_at': AT, 'label': '.1', 'label_available_at': RECORDED,
           'split': 'test', 'costs_hash': 'a' * 64, 'admissible': True}
    assert pair_decisions([row], [row])['decisions'] == 1
    for altered in ([], [{**row, 'product': 'ETH-USD'}], [row, row]):
        with pytest.raises(ValueError):
            pair_decisions([row], altered)
    for key, value in (('label', '.2'), ('split', 'train'), ('costs_hash', 'b' * 64),
                       ('label_available_at', AT), ('admissible', False)):
        with pytest.raises(ValueError):
            pair_decisions([row], [{**row, key: value}])
