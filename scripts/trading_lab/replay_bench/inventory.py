"""Counts and digests of inherited trials, without exporting headlines, responses or outcomes."""
from collections import Counter
from pathlib import Path
import json

from scripts.trading_lab.sources.canonical import sha256_canonical
from . import run, spec


def inventory(root):
    root = Path(root)
    registry = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_trials_v1.json').read_text())
    trials = []
    opus_dates = set()
    for trial in registry['trials']:
        folder = root / 'store' / trial['period'] / trial['config_hash'][:16] / trial['group']
        paths = sorted(folder.glob('*.json.gz'))
        counts = Counter()
        calls = Counter()
        for path in paths:
            record = run.read_json(path)
            for role in ('analyst', 'reviewer'):
                for call in (record.get(role) or {}).get('calls', []):
                    calls[call['family']] += 1
                    if call['family'] == 'opus':
                        opus_dates.add(record['session'])
            for variant, views in record['views'].items():
                for view in views:
                    counts[f"{variant}:{view.get('horizon', 'multiday')}:{view['view']}"] += 1
        trials.append({'trial_id': trial['trial_id'], 'period': trial['period'], 'group': trial['group'],
                       'config_hash': trial['config_hash'], 'spec_hash': trial['spec_hash'],
                       'stored_days': len(paths), 'view_counts': dict(counts), 'recorded_model_calls': dict(calls),
                       'records_digest': sha256_canonical({p.name: spec.file_hash(p) for p in paths}),
                       'admissible_post_cutoff': False})
    return {'schema': 'replay-legacy-inventory-v1', 'trials': trials, 'trial_count': len(trials),
            'opus_replay_dates': sorted(opus_dates), 'opus_replay_day_count': len(opus_dates),
            'test_trial_count': sum(t['period'] == 'test' for t in registry['trials']),
            'legacy_registry_hash': sha256_canonical(registry),
            'reason': 'exploratory only; the prespecified forced-memory rule was relaxed after seeing results'}
