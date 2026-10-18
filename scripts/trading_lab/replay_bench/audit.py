"""Recompute cutoff evidence from original private answers; never trust an editable summary."""
from collections import Counter
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .grant import ReplayError
from . import models, probe, spec
from .storage import load_lines


def checked_rows(path, events, *, forced):
    rows = load_lines(path)
    expected = {(family, event['id']): event for family in ('sonnet', 'opus') for event in events}
    if Counter((r['family'], r['id']) for r in rows) != Counter({key: 1 for key in expected}):
        raise ReplayError('PROBE_INCOMPLETE_OR_DUPLICATE', Path(path).name)
    checked = []
    for row in rows:
        event = expected[(row['family'], row['id'])]
        if ('error' in row or row['month'] != event['month'] or
                row.get('model') != models.MODELS[row['family']] or
                row.get('reported') != [models.MODELS[row['family']]] or
                bool(row.get('forced', False)) != forced):
            raise ReplayError('PROBE_IDENTITY_OR_RESPONSE_INVALID', row['id'])
        correct = bool((forced or row['knows']) and probe.correct(event, row['answer']))
        if correct != row['correct']:
            raise ReplayError('PROBE_GRADE_MISMATCH', row['id'])
        checked.append(dict(row, correct=correct))
    return checked


def evidence(root):
    root = Path(root)
    rule = spec.spec()['probe']
    main_set = json.loads((root / 'probe_set.json').read_text())
    control = json.loads((root / 'probe_control_set.json').read_text())
    if probe.probe_set_digest(main_set) != rule['set_sha256']:
        raise ReplayError('PROBE_SET_HASH_MISMATCH')
    if probe.probe_set_digest(control) != spec.CONTROL_SET_SHA256:
        raise ReplayError('PROBE_CONTROL_SET_HASH_MISMATCH')
    if Counter(e['month'] for e in main_set['events']) != Counter({m: rule['events_per_month'] for m in rule['months']}):
        raise ReplayError('PROBE_MONTH_COVERAGE_INVALID')
    tables = {}
    for kind, name, events, forced in (
        ('main', 'probe_results.jsonl', main_set['events'], False),
        ('controls', 'probe_control.jsonl', control['events'], False),
        ('forced', 'probe_supplementary.jsonl', control['events'] + main_set['events'], True),
    ):
        tables[kind] = probe.summarize(checked_rows(root / name, events, forced=forced), rule)
    floors = {}
    for family in ('sonnet', 'opus'):
        floors[family] = max(tables[k][family]['replay_floor'] or rule['months'][0] + '-01' for k in tables)
    return {'spec_hash': spec.SPEC_HASH, 'floor': floors, 'tables': tables,
            'evidence_hash': sha256_canonical({name: load_lines(root / name) for name in (
                'probe_results.jsonl', 'probe_control.jsonl', 'probe_supplementary.jsonl')})}


def floor(root, families):
    audit = evidence(root)
    return max(audit['floor'][f] for f in families)
