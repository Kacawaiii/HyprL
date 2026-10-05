"""Read verified Model Lab artifacts and authorized replay; preserve missing evidence."""
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sqlite3

from scripts.trading_lab.platform.contracts import ExperimentManifest, PredictionRecord, LabelRecord, OUTPUTS, timestamp
from scripts.trading_lab.research.contracts import PredictionEvidence, ExecutionObservation, DecisionObservation, due_at
from scripts.trading_lab.research.store import now, IntegrityError
from scripts.trading_lab.sources.canonical import sha256_canonical


def paper_events(path, *, session_id, expected_head):
    """Read-only access to the NEW private replay/shadow DB, verifying every event."""
    from scripts.trading_lab.paper_event_store import PaperEvent
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    events, previous, sequence = [], '0' * 64, 0
    try:
        for row in db.execute('SELECT * FROM paper_events WHERE session_id=? ORDER BY sequence', (session_id,)):
            event = PaperEvent(**{**dict(row), 'payload': json.loads(row['payload'])})
            sequence += 1
            if event.sequence != sequence or event.previous_event_hash != previous or event.recomputed_hash() != event.event_hash:
                raise IntegrityError('paper input event chain mismatch')
            previous = event.event_hash
            events.append(event)
    finally:
        db.close()
    if previous != expected_head:
        raise IntegrityError('paper input chain head mismatch')
    return events


def _append_engine_observations(store, predictions, events, *, recorded_at, provenance):
    """Correlate using product + original decision opening, never fill observation time."""
    signals = {}
    targets = {}
    observed_execution = set()
    for event in events:
        if event.event_type == 'PREDICTION_CREATED':
            p = predictions.get((event.product, event.natural_key))
            if p and (Decimal(event.payload['prediction']) != Decimal(p.outputs['return']) or
                      event.payload['feature_vector_hash'] != p.features_hash):
                raise IntegrityError('paper observations differ from the issued prediction')
        elif event.event_type == 'SIGNAL_CREATED':
            signals[(event.product, event.natural_key)] = event.payload['signal']
        elif event.event_type == 'POSITION_TARGET_CREATED':
            key = (event.product, event.natural_key)
            target = event.payload['target']
            targets[key] = target
            p = predictions.get(key)
            if p:
                decision = DecisionObservation(observation_id=event.event_hash, prediction_id=p.prediction_id,
                    prediction_hash=p.identity, available_at=event.event_at, recorded_at=recorded_at,
                    signal=signals[key], risk={'policy_hash': target['risk_spec_hash'], 'target': target},
                    proposed_position=target, provenance={**provenance, 'event_hash': event.event_hash,
                        'method': 'original-paper-engine-decision-v1'})
                store.append_decision(decision)
        elif event.event_type == 'SIMULATED_FILL':
            key = (event.product, event.payload['decided_at'])
            p = predictions.get(key)
            if p:
                fill = event.payload['fill']
                target = targets.get(key)
                position = {'quantity': fill['position_after'], 'reference_price': fill['reference_price'],
                            'equity': fill['equity_after']}
                with localcontext() as context:
                    context.prec = 34
                    realized = Decimal(fill['position_after']) * Decimal(fill['reference_price']) / Decimal(fill['equity_after'])
                gap = {'target_exposure': target['target_exposure'] if target else None,
                       'realized_exposure': str(realized),
                       'exposure_difference': str(realized - Decimal(target['target_exposure'])) if target else None,
                       'decision_to_observation_seconds': (datetime.fromisoformat(timestamp(event.event_at)) -
                           datetime.fromisoformat(p.decision_at)).total_seconds(),
                       'method': 'observed-fill-marked-exposure-v1'}
                store.append_execution(ExecutionObservation(observation_id=event.event_hash, prediction_id=p.prediction_id,
                    prediction_hash=p.identity, available_at=event.event_at, recorded_at=recorded_at, state='FILLED',
                    executed_position=position, costs={'fee': fill['fee'], 'slippage': fill['slippage_cost']},
                    proposal_gap=gap, errors=(), provenance={**provenance, 'fill_hash': event.payload['fill_hash']}))
                observed_execution.add(key)
        elif event.event_type == 'GAP_DETECTED' and 'target_timestamp' in event.payload:
            key = (event.product, event.payload['target_timestamp'])
            p = predictions.get(key)
            if p:
                store.append_execution(ExecutionObservation(observation_id=event.event_hash, prediction_id=p.prediction_id,
                    prediction_hash=p.identity, available_at=event.event_at, recorded_at=recorded_at, state='EXPIRED',
                    executed_position=None, costs=None, proposal_gap=None, errors=('MISSING_NEXT_BAR',), provenance=provenance))
                observed_execution.add(key)
        elif event.event_type == 'PORTFOLIO_SNAPSHOT':
            # The engine checks the previous target before it emits this snapshot.
            opening = datetime.fromisoformat(event.payload['timestamp']) - timedelta(hours=1)
            key = (event.product, opening.isoformat())
            p = predictions.get(key)
            if p and key in targets and key not in observed_execution:
                payload = event.payload
                store.append_execution(ExecutionObservation(observation_id=event.event_hash, prediction_id=p.prediction_id,
                    prediction_hash=p.identity, available_at=event.event_at, recorded_at=recorded_at, state='NO_FILL',
                    executed_position={'quantity': payload['position_quantity'], 'equity': payload['equity']},
                    costs={'fee': '0', 'slippage': '0'}, proposal_gap=None, errors=(),
                    provenance={**provenance, 'method': 'contiguous-next-bar-no-fill-v1'}))
                observed_execution.add(key)


def import_model_lab(store, dataset, result, *, shadow_path=None, recorded_at=None):
    # An idle writer connection keeps WAL open across small durable append transactions.
    with store.connect():
        return _import_model_lab(store, dataset, result, shadow_path=shadow_path, recorded_at=recorded_at)


def _import_model_lab(store, dataset, result, *, shadow_path=None, recorded_at=None):
    """Do not refit. Exact emitted prediction hashes and original snapshots survive."""
    from scripts.trading_lab.platform.datasets import verify_dataset
    from scripts.trading_lab.platform.experiments import temporal_splits
    manifest = verify_dataset(dataset)
    prepared = ExperimentManifest.from_dict(result['prepared'])
    complete = ExperimentManifest.from_dict(result['manifest'])
    if (not manifest.synthetic or not complete.synthetic or complete.status != 'COMPLETE'
            or prepared.status != 'PREPARED' or complete.artifacts['prepared_hash'] != prepared.identity
            or complete.dataset_hash != manifest.identity or result['fingerprint'] != complete.identity
            or complete.artifacts['prediction_hash'] != sha256_canonical(result['predictions'])
            or complete.artifacts['backtest_hash'] != sha256_canonical(result['backtests'])
            or complete.artifacts['shadow_hash'] != sha256_canonical(result['shadow'])
            or replace(prepared, status='COMPLETE', artifacts=complete.artifacts).identity != complete.identity):
        raise IntegrityError('Model Lab result bindings mismatch')
    for product, artifact in result['models'].items():
        if sha256_canonical(artifact) != complete.artifacts['model_hashes'][product]:
            raise IntegrityError('Model Lab model artifact mismatch')
    at = timestamp(recorded_at or now())
    rows = {(r['product'], timestamp(r['decision_at'])): r for r in dataset['rows']}
    blocks, _ = temporal_splits(dataset['rows'], horizon_seconds=manifest.horizon_seconds,
                               embargo_seconds=prepared.splits['embargo_seconds'])
    means = {}
    with localcontext() as context:
        context.prec = 34
        for product in manifest.products:
            labels = [Decimal(r['label']) for r in blocks['train'] if r['product'] == product]
            means[product] = str(sum(labels) / len(labels))
    store.append('experiment', complete, object_id=complete.experiment_id, recorded_at=at)
    predictions = {}
    for item in result['predictions']:
        p = PredictionRecord.from_dict(item['record'])
        row = rows[(p.product, p.decision_at)]
        if (p.synthetic is not True or p.model_id != prepared.parameters['model_id']
                or item['fingerprint'] != p.identity or p.model_contract_hash != prepared.model_contract_hash
                or p.artifact_hash != complete.artifacts['model_hashes'][p.product]
                or p.features_hash != row['features_hash'] or p.snapshot_hash != row['snapshot_hash']
                or list(p.event_ids) != row['event_ids'] or p.horizon_seconds != manifest.horizon_seconds):
            raise IntegrityError('Model Lab prediction input mismatch')
        evidence = PredictionEvidence(prediction_id=p.prediction_id, prediction_hash=p.identity, recorded_at=at,
            features=tuple(row['features']), snapshot=dataset['snapshots'][p.snapshot_hash],
            baselines={'ZERO': '0', 'TRAIN_MEAN': means[p.product]},
            input_quality={'state': 'VALID', 'gaps': 0, 'source_states': row['source_states'],
                           'scope': 'selected price inputs; unselected event sources retain unknown states'},
            split=item['split'], provenance={'method': 'model-lab-artifact-import-v1', 'dataset_hash': manifest.identity,
                'experiment_hash': complete.identity, 'cadence_seconds': 3600, 'synthetic': True})
        store.issue(p, evidence)
        if timestamp(row['label_available_at']) <= at:
            label = LabelRecord(label_id='lab-label-' + p.prediction_id, prediction_id=p.prediction_id,
                prediction_hash=p.identity, product=p.product, horizon_seconds=p.horizon_seconds,
                realized_at=row['label_end'], available_at=row['label_available_at'], recorded_at=at,
                target='forward_return', value=row['label'], version='1', provenance={
                    'method': 'dataset-label-after-horizon-v1', 'dataset_hash': manifest.identity, 'synthetic': True})
            store.append_label(label)
        predictions[(p.product, row['bar_open_at'])] = p
    if shadow_path:
        events = paper_events(shadow_path, session_id=prepared.experiment_id, expected_head=result['shadow']['chain']['head_hash'])
        _append_engine_observations(store, predictions, events, recorded_at=at,
            provenance={'mode': 'synthetic_shadow', 'synthetic': True, 'session_hash': result['shadow']['session_hash']})
    return {'synthetic': True, 'predictions': len(predictions), 'prepared_hash': prepared.identity,
            'experiment_hash': complete.identity, 'recorded_at': at, 'model_id': prepared.parameters['model_id']}


def reconstruct_replay_features(rows_by_product):
    """Rebuild the exact frozen feature algorithm, with the replay's actual seed prefix."""
    from scripts.trading_lab.market_dataset import build_dataset
    from scripts.trading_lab.paper_engine import series_from_rows
    from scripts.trading_lab.paper_replay import REPLAY_START, REPLAY_END, SEED_BARS
    from scripts.trading_lab.real_benchmark_v2 import DATASET_CONFIG_V2
    reconstructed = {}
    with localcontext() as context:
        context.prec = 34
        for product, rows in rows_by_product.items():
            seed = [r for r in rows if r['bar_open_at'] < REPLAY_START][-SEED_BARS:]
            window = [r for r in rows if REPLAY_START <= r['bar_open_at'] <= REPLAY_END]
            dataset = build_dataset(series_from_rows(seed + window, product=product), config=DATASET_CONFIG_V2)
            for row in dataset.rows:
                if row.bar_open_at >= REPLAY_START and all(value is not None for _, value in row.features):
                    reconstructed[(product, row.bar_open_at)] = [[name, str(value)] for name, value in row.features]
    return reconstructed


def verified_replay_features(reconstructed, *, product, bar_open_at, captured_hash):
    features = reconstructed.get((product, bar_open_at))
    if features is None or sha256_canonical(features) != captured_hash:
        raise IntegrityError('reconstructed features do not match the captured prediction hash')
    return features


def import_authorized_replay(store, *, runtime_root, recorded_at=None, replay_database=None):
    with store.connect():
        return _import_authorized_replay(store, runtime_root=runtime_root, recorded_at=recorded_at,
                                         replay_database=replay_database)


def _import_authorized_replay(store, *, runtime_root, recorded_at=None, replay_database=None):
    """One NEW replay of exactly the authorized frozen May-July corpus; no training."""
    from scripts.trading_lab.app_api.paper_replay import PaperReplayViews
    from scripts.trading_lab.paper_replay import (load_replay_corpus, load_v2_artifacts, run_replay, SESSION_ID,
        REPLAY_START, REPLAY_END, READ_CUTOFF)
    from scripts.trading_lab.paper_engine import series_from_rows
    from scripts.trading_lab.market_dataset import forward_return_labels
    from scripts.trading_lab.platform.adapters import FrozenPaperAdapter
    root = Path(runtime_root).resolve()
    if not root.is_relative_to(Path.cwd().resolve() / 'var'):
        raise ValueError('new replay root must stay in this worktree var directory')
    frozen = PaperReplayViews(Path('data/crypto'))
    manifest = frozen._manifest()
    frozen_results = {p: frozen._result(p, manifest) for p in ('BTC-USD', 'ETH-USD')}
    corpus, rows = load_replay_corpus()
    artifacts = load_v2_artifacts()
    if replay_database is None:
        path = root / 'observability-paper-replay.sqlite'
        result = run_replay(rows, artifacts, database_path=path)
        if any(result['products'][p]['result_hash'] != frozen_results[p]['result_hash'] for p in frozen_results):
            raise IntegrityError('new replay differs from frozen reference results')
    else:
        path = Path(replay_database).resolve()
        if not path.is_relative_to(Path.cwd().resolve() / 'var'):
            raise ValueError('replay evidence must be a private worktree runtime, never an archive or live store')
        result = {'products': frozen_results, 'chain': {
            'head_hash': manifest['determinism']['first_chain_head_hash']}}
    at = timestamp(recorded_at or now())
    events = paper_events(path, session_id=SESSION_ID, expected_head=result['chain']['head_hash'])
    if len(events) != manifest['determinism']['events']:
        raise IntegrityError('authorized replay event count mismatch')
    labels = {}
    for product, product_rows in rows.items():
        window = [r for r in product_rows if REPLAY_START <= r['bar_open_at'] <= REPLAY_END]
        with localcontext() as context:
            context.prec = 34
            actual = forward_return_labels(series_from_rows(window, product=product), horizon=4)
        labels[product] = dict(zip((r['bar_open_at'] for r in window), actual))
    reconstructed = reconstruct_replay_features(rows)
    predictions = {}
    for event in events:
        if event.event_type != 'PREDICTION_CREATED':
            continue
        old = event.payload
        features = verified_replay_features(reconstructed, product=event.product, bar_open_at=old['bar_open_at'],
                                             captured_hash=old['feature_vector_hash'])
        certificate = {'schema': 'legacy-paper-input-v1', 'decision_at': old['decision_available_at'],
            'product': event.product, 'feature_hash': old['feature_vector_hash'], 'event_ids': [],
            'prediction_event_hash': event.event_hash, 'corpus_hash': corpus['corpus_content_hash'],
            'availability': 'assumed bar-close; no historical exchange revision attestation',
            'feature_values': 'NOT_RECORDED', 'information_snapshot': 'NOT_RECORDED'}
        p = PredictionRecord(prediction_id='paper-v2-' + event.event_hash, model_id='paper-ridge-v2',
            model_contract_hash=FrozenPaperAdapter('2').contract.identity,
            artifact_hash=artifacts[event.product]['artifact_hash'], product=event.product,
            decision_at=old['decision_available_at'], horizon_seconds=14400,
            snapshot_hash=sha256_canonical(certificate), features_hash=old['feature_vector_hash'], event_ids=(),
            outputs={k: old['prediction'] if k == 'return' else None for k in OUTPUTS}, synthetic=False)
        evidence = PredictionEvidence(prediction_id=p.prediction_id, prediction_hash=p.identity, recorded_at=at,
            features=features, snapshot=certificate, baselines={'ZERO': '0'},
            input_quality={'state': 'PARTIAL', 'gaps': None, 'reasons': ['PRICE_CLOCK_ASSUMED'], 'feature_state': 'HASH_VERIFIED_RECONSTRUCTION'},
            split='spent_oos', provenance={'method': 'frozen-authorized-paper-replay-import-v1',
                'cadence_seconds': 3600, 'synthetic': False, 'cost_model': 'synthetic',
                'original_prediction_hash': old['prediction_hash'], 'legacy_snapshot_certificate': True,
                'feature_reconstruction': {'method': 'frozen-paper-dataset-features-v1', 'decimal_precision': 34,
                    'seed_bars': 200, 'captured_hash_verified': True, 'historical_availability_attested': False}})
        store.issue(p, evidence)
        value = labels[p.product][old['bar_open_at']]
        if value is not None:
            realization = due_at(p)
            if realization <= timestamp(READ_CUTOFF):
                store.append_label(LabelRecord(label_id='replay-label-' + p.prediction_id, prediction_id=p.prediction_id,
                    prediction_hash=p.identity, product=p.product, horizon_seconds=p.horizon_seconds,
                    realized_at=realization, available_at=realization, recorded_at=at, target='forward_return',
                    value=str(value), version='1', provenance={'method': 'frozen-replay-close-forward-return-v1',
                        'source_result_hash': result['products'][p.product]['result_hash'],
                        'price_availability': 'assumed bar-close; corpus acquired later', 'decimal_precision': 34, 'synthetic': False}))
        predictions[(p.product, old['bar_open_at'])] = p
    _append_engine_observations(store, predictions, events, recorded_at=at,
        provenance={'mode': 'authorized_frozen_replay', 'synthetic': False, 'cost_model': 'synthetic',
                    'chain_head': result['chain']['head_hash']})
    # Keep only counts/identities for the historical summary, never copy raw price rows.
    summary = {'schema': 'authorized-replay-observability-v1', 'synthetic': False, 'scope': 'EXPLORATORY',
        'predictions': len(predictions), 'result_hashes': {p: r['result_hash'] for p, r in result['products'].items()},
        'manifest_hash': manifest['manifest_hash'], 'chain_head': result['chain']['head_hash'],
        'limitations': manifest['limitations'], 'new_real_training': False, 'network_requests': 0}
    store.append('replay-summary', summary, object_id=manifest['manifest_hash'], recorded_at=at)
    return summary
