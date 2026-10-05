from dataclasses import replace
import sqlite3

import pytest

from scripts.trading_lab.research.contracts import ExecutionObservation
from scripts.trading_lab.research.store import ResearchStore, IntegrityError
from tests.research.conftest import issue, label, AT, RECORDED, H


def test_pending_labels_arrive_and_corrections_append_without_rewriting_prediction(store):
    p, e = issue(store)
    original = store.get(p.identity)
    store.append_label(label(p, available='2026-06-01T05:00:00+00:00', recorded='2026-06-01T06:00:00+00:00'))
    assert store.prediction_view(p.identity, as_of='2026-06-01T05:59:00Z')['label_state'] == 'PENDING'
    available = store.prediction_view(p.identity, as_of='2026-06-01T06:00:00Z')
    assert available['label_state'] == 'AVAILABLE' and len(available['labels']) == 1
    corrected = label(p, available='2026-06-01T07:00:00+00:00', recorded='2026-06-01T08:00:00+00:00', value='-.01', version='2')
    store.append_label(corrected)
    later = store.prediction_view(p.identity, as_of=RECORDED)
    assert [x['value'] for x in later['labels']] == ['.02', '-.01']
    assert store.get(p.identity) == original
    assert store.prediction_view(p.identity, as_of='2026-06-01T06:00:00Z') == available
    assert later['prediction']['uncertainty'] is None


@pytest.mark.parametrize('change', [{'product': 'ETH-USD'}, {'prediction_hash': 'b' * 64},
    {'horizon_seconds': 3600}, {'realized_at': AT}])
def test_mismatched_or_early_labels_refused(store, change):
    p, _ = issue(store)
    with pytest.raises((KeyError, ValueError)):
        store.append_label(replace(label(p), **change))
    assert not store.records('label')


def test_recorded_clock_and_exact_feature_snapshot_and_uncertainty_bindings(store):
    p, e = issue(store)
    with pytest.raises(ValueError, match='immutable'):
        store.issue(replace(p, outputs={**dict(p.outputs), 'return': '.2'}), replace(e,
            prediction_hash=replace(p, outputs={**dict(p.outputs), 'return': '.2'}).identity))
    with pytest.raises(ValueError, match='feature'):
        store.issue(p, replace(e, features=(('return_4', '2'),)))
    with pytest.raises(ValueError, match='snapshot'):
        store.issue(p, replace(e, snapshot={'schema': 'modified'}))
    with pytest.raises(ValueError, match='before'):
        store.issue(p, replace(e, recorded_at='2026-05-01T00:00:00Z'))
    uncertain = replace(p, prediction_id='synthetic-uncertain', uncertainty={'confidence': '.9'})
    with pytest.raises(ValueError, match='method'):
        store.issue(uncertain, replace(e, prediction_id=uncertain.prediction_id, prediction_hash=uncertain.identity))
    store.issue(p, e)
    assert len(store.records('prediction')) == 1
    with pytest.raises(KeyError):
        store.prediction_view(p.identity, as_of='2026-05-01T00:00:00Z')


def test_execution_arrivals_preserve_pending_and_proposal_execution_gaps(store):
    p, e = issue(store)
    observation = ExecutionObservation(observation_id='synthetic-fill', prediction_id=p.prediction_id,
        prediction_hash=p.identity, available_at='2026-06-01T01:00:00Z', recorded_at='2026-06-01T02:00:00Z',
        state='FILLED', executed_position={'quantity': '1'}, costs={'fee': '.1', 'slippage': '.05'},
        proposal_gap={'target_exposure': '.1', 'realized_exposure': '.09', 'method': 'synthetic-marked-exposure-v1'},
        errors=(), provenance={'synthetic': True})
    store.append_execution(observation)
    assert store.prediction_view(p.identity, as_of='2026-06-01T01:30:00Z')['execution_state'] == 'PENDING'
    view = store.prediction_view(p.identity, as_of=RECORDED)
    assert view['execution_state'] == 'FILLED' and view['prediction']['execution'] is None
    assert view['executions'][0]['payload']['proposal_gap']['realized_exposure'] == '.09'
    with pytest.raises(ValueError):
        store.append_execution(replace(observation, prediction_id='other'))
    with pytest.raises(ValueError):
        replace(observation, costs={'fee': '-1'})


def test_read_only_store_hash_checks_append_only_triggers_and_schema_binding(store):
    p, e = issue(store)
    reader = ResearchStore(store.root, read_only=True)
    assert reader.get(p.identity) == store.get(p.identity)
    with reader.connect() as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute('DELETE FROM records')
    with store.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('UPDATE records SET payload=?', ('{}',))
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('DELETE FROM records')
        db.execute('DROP TRIGGER no_record_update')
        db.execute("UPDATE records SET payload='{}' WHERE identity=?", (p.identity,))
    with pytest.raises(IntegrityError):
        reader.get(p.identity)
    with pytest.raises(IntegrityError):
        reader.verify()
    with store.connect() as db:
        db.execute("UPDATE metadata SET schema='research-store-old'")
    with pytest.raises(IntegrityError, match='schema'):
        ResearchStore(store.root, read_only=True)


def test_same_label_version_cannot_change_and_input_evidence_is_immutable(store):
    p, e = issue(store)
    one = label(p)
    store.append_label(one)
    with pytest.raises(ValueError, match='version immutable'):
        store.append_label(replace(one, value='.03'))
    with pytest.raises(ValueError, match='input evidence immutable'):
        store.issue(p, replace(e, baselines={'ZERO': '1'}))
    assert store.verify()['verified']


def test_equal_recording_clocks_keep_append_order_for_corrections(store):
    p, _ = issue(store)
    store.append_label(label(p, recorded=RECORDED))
    store.append_label(label(p, value='-.1', version='2', recorded=RECORDED))
    assert store.prediction_view(p.identity, as_of=RECORDED)['labels'][-1]['value'] == '-.1'


def test_count_byte_and_record_budgets_are_persistent_and_issue_is_atomic(store, monkeypatch):
    import scripts.trading_lab.research.store as module
    monkeypatch.setattr(module, 'MAX_RECORDS', 1)
    with pytest.raises(ValueError, match='budget'):
        issue(store)
    assert store.verify()['records'] == 0
    monkeypatch.setattr(module, 'MAX_RECORDS', 100)
    monkeypatch.setattr(module, 'MAX_BYTES', 10)
    with pytest.raises(ValueError, match='budget'):
        issue(store)
    assert store.verify()['records'] == 0
    monkeypatch.setattr(module, 'MAX_BYTES', 1000000)
    monkeypatch.setattr(module, 'MAX_RECORD_BYTES', 10)
    with pytest.raises(ValueError, match='size'):
        issue(store)
    assert store.verify()['records'] == 0
