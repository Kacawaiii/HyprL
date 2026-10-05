"""Offline registry -> Model Lab worker -> immutable ledger -> monitoring demo."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from scripts.trading_lab.platform.contracts import PredictionRecord, LabelRecord, OUTPUTS
from scripts.trading_lab.research.contracts import PredictionEvidence, InferenceObservation
from scripts.trading_lab.research.store import ResearchStore, now
from scripts.trading_lab.research.monitoring import make_reference, monitor
from scripts.trading_lab.research.proposals import propose, comparison_hypothesis, comparison_diagnostic
from scripts.trading_lab.sources.canonical import sha256_canonical


def synthetic_scenarios(store, *, at=None):
    """Injected synthetic drift, errors, latency, gaps and performance loss."""
    at = at or now()
    end = datetime.fromisoformat(at)
    h = sha256_canonical(['injected-synthetic-scenarios-v1'])
    for i in range(48):
        if i == 31:  # explicit missing hourly decision
            continue
        current = i >= 24
        decision = end - timedelta(hours=48 - i)
        features = [['return_4', '.03' if current else '.001'], ['atr_pct_14', '.04' if current else '.01']]
        snapshot = {'schema': 'synthetic-monitoring-input-v1', 'synthetic': True, 'as_of': decision.isoformat(),
                    'features': features, 'event_ids': []}
        p = PredictionRecord(prediction_id='injected-synthetic-' + str(i), model_id='injected-synthetic-monitor-v1',
            model_contract_hash=h, artifact_hash=h, product='SYNTHETIC-DEMO', decision_at=decision.isoformat(),
            horizon_seconds=3600, snapshot_hash=sha256_canonical(snapshot), features_hash=sha256_canonical(features),
            event_ids=(), outputs={k: ('.03' if current else '.001') if k == 'return' else None for k in OUTPUTS}, synthetic=True)
        e = PredictionEvidence(prediction_id=p.prediction_id, prediction_hash=p.identity, recorded_at=at,
            features=features, snapshot=snapshot, baselines={'ZERO': '0'}, input_quality={'state': 'VALID', 'gaps': 0},
            split='current' if current else 'reference', provenance={'synthetic': True, 'cadence_seconds': 3600,
                'method': 'injected-synthetic-monitoring-scenario-v1'})
        store.issue(p, e)
        store.append_label(LabelRecord(label_id='injected-label-' + str(i), prediction_id=p.prediction_id,
            prediction_hash=p.identity, product=p.product, horizon_seconds=3600,
            realized_at=(decision + timedelta(hours=1)).isoformat(),
            available_at=(decision + timedelta(hours=1)).isoformat(), recorded_at=at,
            target='forward_return', value='.001', provenance={'synthetic': True, 'method': 'injected-known-label-v1'}, version='1'))
        store.append('inference', InferenceObservation(observation_id='injected-attempt-' + str(i),
            model_id=p.model_id, model_contract_hash=p.model_contract_hash, artifact_hash=p.artifact_hash,
            horizon_seconds=p.horizon_seconds, product=p.product, decision_at=p.decision_at, recorded_at=at,
            status='ERROR' if current and i % 6 == 0 else 'SUCCESS',
            latency_ms=1500 if current else 1, errors=('INJECTED_ERROR',) if current and i % 6 == 0 else (), synthetic=True))
    reference = make_reference(store, reference_id='injected-reference-v1', as_of=at,
        product='SYNTHETIC-DEMO', model_id='injected-synthetic-monitor-v1', split='reference')
    result = monitor(store, as_of=at, product='SYNTHETIC-DEMO', model_id='injected-synthetic-monitor-v1',
                     split='current', reference_hash=reference)
    required = {'MISSING_DATA', 'TECHNICAL_DEGRADATION', 'DRIFT', 'PERFORMANCE_DROP'}
    if {c['category'] for c in result['classification']} != required:
        raise RuntimeError('synthetic monitoring classifications failed')
    return {'synthetic': True, 'reference_hash': reference, 'sample': result['sample'],
            'classifications': sorted(required), 'monitoring_hash': sha256_canonical(result)}


def replay_monitoring(store, *, as_of=None):
    """Explicit descriptive May references and June-July observations on spent data."""
    at = as_of or now()
    result = {}
    for product in ('BTC-USD', 'ETH-USD'):
        reference = make_reference(store, reference_id='spent-replay-v2-may-reference-' + product,
            as_of=at, product=product, model_id='paper-ridge-v2',
            start='2026-05-01T00:00:00Z', end='2026-06-01T00:00:00Z')
        view = monitor(store, as_of=at, product=product, model_id='paper-ridge-v2',
            start='2026-06-01T00:00:00Z', end='2026-08-01T00:00:00Z', reference_hash=reference)
        result[product] = {'reference_hash': reference, 'sample': view['sample'],
            'available_labels': view['performance']['sample'], 'pending_labels': view['performance']['pending'],
            'monitoring_hash': sha256_canonical(view), 'scope': 'EXPLORATORY',
            'method_hash': view['method_hash'], 'feature_values': 'HASH_VERIFIED_RECONSTRUCTION', 'inference_telemetry': 'NOT_OBSERVED'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='var/trading_lab/research-observability-demo')
    parser.add_argument('--bars', type=int, default=120)
    parser.add_argument('--paper-replay', action='store_true', help='read and replay the authorized frozen May-July corpus; no fit')
    parser.add_argument('--replay-database', default=None, help='reuse a private replay under this worktree var, verified against the frozen chain')
    args = parser.parse_args(argv)
    if args.replay_database and not args.paper_replay:
        parser.error('--replay-database requires the explicit --paper-replay scope')
    root = Path(args.root).resolve()
    if not root.is_relative_to(Path.cwd().resolve() / 'var'):
        parser.error('demo root must stay beneath this worktree ignored var directory')
    if root.exists():
        parser.error('demo root must be new; previous evidence is preserved')
    from scripts.trading_lab.platform.jobs import JobRunner
    from scripts.trading_lab.platform.model_lab_demo import wait
    from scripts.trading_lab.research.importers import import_model_lab, import_authorized_replay
    store = ResearchStore(root / 'registry')
    store.register(comparison_hypothesis())
    reports = []
    with JobRunner(root / 'lab') as runner:
        job = runner.store.submit('dataset', {'products': ['BTC-USD', 'ETH-USD'], 'bars': args.bars})
        data = wait(runner.store, job)
        dataset = runner.store.artifact(data['dataset_hash'], kind='dataset')
        for proposal in propose(dataset):
            plan, prepared = proposal['hypothesis'], proposal['prepared']
            store.register(plan)
            trial_id = prepared.experiment_id
            store.start_trial(plan, prepared, trial_id=trial_id)
            job = runner.store.submit('experiment', {'prepared': prepared.to_dict()})
            store.observe_trial(trial_id, state='RUNNING', outcome='PENDING', evidence={'job_id': job, 'synthetic': True})
            try:
                result = wait(runner.store, job)
                imported = import_model_lab(store, dataset, result,
                    shadow_path=runner.store.root / (job + '-synthetic-shadow.sqlite'))
                met = all(result['criteria_met'].values())
                references = {}
                for product in dataset['manifest']['products']:
                    at = now()
                    reference = make_reference(store, reference_id=prepared.experiment_id + '-' + product + '-validation',
                        as_of=at, product=product, model_id=prepared.parameters['model_id'], split='validation')
                    view = monitor(store, as_of=at, product=product, model_id=prepared.parameters['model_id'],
                                   split='test', reference_hash=reference)
                    references[product] = {'reference_hash': reference, 'sample': view['sample'],
                                          'monitoring_hash': sha256_canonical(view)}
                store.observe_trial(trial_id, state='COMPLETE', outcome='POSITIVE' if met else 'NEGATIVE',
                    evidence={'result_hash': runner.store.status(job)['result_hash'], 'criteria_met': result['criteria_met'], 'metrics': result['metrics'],
                              'synthetic': True, 'scientific_claim': False})
                reports.append({**imported, 'criteria_met': result['criteria_met'], 'monitoring': references})
            except Exception:
                store.observe_trial(trial_id, state='FAILED', outcome='ERROR', evidence={'code': 'DEMO_WORKLOAD_FAILED'})
                raise
    scenarios = synthetic_scenarios(store)
    replay = import_authorized_replay(store, runtime_root=root / 'replay', replay_database=args.replay_database) if args.paper_replay else None
    real_monitoring = replay_monitoring(store) if replay else None
    print(json.dumps({'schema': 'research-observability-demo-v1', 'model_lab': reports,
        'synthetic_scenarios': scenarios, 'authorized_replay': replay, 'real_monitoring': real_monitoring, 'comparison': comparison_diagnostic(),
        'store': store.verify(), 'network_requests': 0, 'new_real_training': False,
        'limitations': ['synthetic outcomes prove infrastructure only', 'spent real replay remains exploratory',
                        'future labels and absent telemetry remain pending or unknown']}, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
