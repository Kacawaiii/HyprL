from dataclasses import replace
import copy
from pathlib import Path

import pytest

from scripts.trading_lab.platform.jobs import JobRunner
from scripts.trading_lab.platform.model_lab_demo import wait
from scripts.trading_lab.research.importers import import_model_lab, paper_events, import_authorized_replay
from scripts.trading_lab.research.proposals import propose
from scripts.trading_lab.research.store import IntegrityError
from scripts.trading_lab.research.monitoring import monitor
from tests.research.conftest import RECORDED


@pytest.fixture(scope='module')
def lab_result(tmp_path_factory, dataset):
    root = tmp_path_factory.mktemp('synthetic-lab-observability')
    with JobRunner(root) as runner:
        runner.store.put_artifact('dataset', dataset, identity=dataset['fingerprint'])
        prepared = propose(dataset, limit=2)[1]['prepared']
        identifier = runner.store.submit('experiment', {'prepared': prepared.to_dict()})
        result = wait(runner.store, identifier, 60)
        yield result, root / (identifier + '-synthetic-shadow.sqlite')


def test_model_lab_import_preserves_exact_predictions_features_snapshots_and_late_labels(store, dataset, lab_result):
    result, shadow = lab_result
    imported = import_model_lab(store, dataset, result, shadow_path=shadow, recorded_at='2026-06-10T00:00:00Z')
    assert imported['synthetic'] and imported['predictions'] == len(result['predictions'])
    predictions = store.records('prediction')
    assert {p['identity'] for p in predictions} == {p['fingerprint'] for p in result['predictions']}
    assert store.verify()['verified']
    for p in predictions:
        view = store.prediction_view(p['identity'], as_of='2026-06-10T00:00:00Z')
        assert view['inputs']['features'] and view['inputs']['snapshot']['schema'] == 'information-snapshot-v1'
        assert view['label_state'] == 'AVAILABLE'
        assert view['inputs']['baselines']['ZERO'] == '0'
        assert 'TRAIN_MEAN' in view['inputs']['baselines']
    assert store.records('decision') and store.records('execution')
    assert any(p['payload']['state'] == 'FILLED' and p['payload']['proposal_gap']['method'] == 'observed-fill-marked-exposure-v1' for p in store.records('execution'))
    view = monitor(store, as_of='2026-06-10T00:00:00Z', product='BTC-USD', model_id='local-momentum-v1', split='test')
    assert view['performance']['sample'] > 10
    assert view['inference']['availability'] is None
    assert 'TRAIN_MEAN' in view['performance']['edge']


@pytest.mark.parametrize('mutation', ['prediction', 'artifact', 'manifest', 'fingerprint'])
def test_corrupted_model_lab_bindings_fail_closed(store, dataset, lab_result, mutation):
    result, _ = lab_result
    altered = copy.deepcopy(result)
    if mutation == 'prediction':
        altered['predictions'][0]['record']['features_hash'] = 'b' * 64
    elif mutation == 'artifact':
        altered['models']['BTC-USD']['synthetic'] = False
    elif mutation == 'manifest':
        altered['manifest']['artifacts']['prepared_hash'] = 'b' * 64
    else:
        altered['fingerprint'] = 'b' * 64
    with pytest.raises(IntegrityError):
        import_model_lab(store, dataset, altered)
    assert not store.records('prediction')


def test_read_only_shadow_chain_refuses_wrong_head_and_uses_no_writes(lab_result):
    result, path = lab_result
    before = path.read_bytes()
    events = paper_events(path, session_id=result['prepared']['experiment_id'], expected_head=result['shadow']['chain']['head_hash'])
    assert events and path.read_bytes() == before
    with pytest.raises(IntegrityError):
        paper_events(path, session_id=result['prepared']['experiment_id'], expected_head='0' * 64)


def test_authorized_replay_refuses_outside_worktree_before_any_replay(store, tmp_path, monkeypatch):
    import scripts.trading_lab.paper_replay as replay
    def forbidden(*args, **kwargs):
        raise AssertionError('replay was accessed')
    monkeypatch.setattr(replay, 'load_replay_corpus', forbidden)
    with pytest.raises(ValueError, match='worktree'):
        import_authorized_replay(store, runtime_root=tmp_path)


def test_engine_no_fill_and_expiry_observations_preserve_terminal_pending(store):
    from scripts.trading_lab.paper_event_store import PaperEvent
    from scripts.trading_lab.research.importers import _append_engine_observations
    from tests.research.conftest import issue, AT
    p, _ = issue(store, value='0')
    opening = '2026-05-31T23:00:00+00:00'
    def event(kind, natural_key, payload, index):
        return PaperEvent(event_id=index, session_id='synthetic-observation', sequence=index, event_type=kind,
            event_at='2026-06-01T01:00:00+00:00', product=p.product, natural_key=natural_key,
            payload=payload, previous_event_hash='0' * 64, event_hash=str(index) * 64)
    from decimal import Decimal
    from scripts.trading_lab.signal_engine import generate_signal
    from scripts.trading_lab.risk_engine import generate_position_target
    signal = generate_signal(timestamp=opening, prediction=Decimal('0'), model_spec_hash=p.model_contract_hash,
        fitted_hash=p.artifact_hash, benchmark_spec_hash='a' * 64)
    target = generate_position_target(signal=signal)
    events = [event('SIGNAL_CREATED', opening, {'signal': signal.canonical()}, 1),
        event('POSITION_TARGET_CREATED', opening, {'target': target.canonical()}, 2),
        event('PORTFOLIO_SNAPSHOT', AT, {'timestamp': AT, 'position_quantity': '0', 'equity': '100'}, 3)]
    _append_engine_observations(store, {(p.product, opening): p}, events, recorded_at='2026-06-01T02:00:00Z',
                                provenance={'synthetic': True})
    assert store.prediction_view(p.identity, as_of='2026-06-01T02:00:00Z')['execution_state'] == 'NO_FILL'
    other, _ = issue(store, identifier='synthetic-expired')
    expired = event('GAP_DETECTED', 'expired', {'target_timestamp': opening}, 4)
    _append_engine_observations(store, {(p.product, opening): other}, [expired], recorded_at='2026-06-01T02:00:00Z',
                                provenance={'synthetic': True})
    assert store.prediction_view(other.identity, as_of='2026-06-01T02:00:00Z')['execution_state'] == 'EXPIRED'
    terminal, _ = issue(store, identifier='synthetic-terminal')
    assert store.prediction_view(terminal.identity, as_of='2026-06-01T02:00:00Z')['execution_state'] == 'PENDING'
