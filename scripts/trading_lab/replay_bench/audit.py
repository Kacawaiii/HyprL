"""Recompute cutoff evidence from original private answers; never trust an editable summary."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
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


def legacy_evidence(root):
    root = Path(root)
    rule = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_spec_v3.json').read_text())['probe']
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


def forced_summary(questions, answers, rule):
    months = sorted({q['month'] for q in questions})
    counts = {m: sum(answers[q['id']] == ('up' if q['up'] else 'down')
                     for q in questions if q['month'] == m) for m in months}
    pooled = [q for q in questions if q['month'] in rule['forced_pool_months']]
    correct = sum(counts[m] for m in rule['forced_pool_months'])
    return {'correct_by_month': counts, 'pooled_correct': correct, 'pooled_n': len(pooled),
            'always_up_by_month': {m: sum(q['up'] for q in questions if q['month'] == m) for m in months},
            'always_up_pooled': sum(q['up'] for q in pooled),
            'binomial_one_sided_p_vs_half': sum(math.comb(len(pooled), k) for k in range(correct, len(pooled) + 1)) / 2**len(pooled),
            'flagged': correct >= rule['forced_pool_threshold'] or max(counts.values()) >= rule['forced_month_threshold']}


def evidence(root):
    """Regrade pinned operator probe files offline; the operator attests pre-call hashing/isolation."""
    rule = spec.spec()['probe']
    directory = Path(os.environ.get('REPLAY_PROBE_ROOT', str(Path(root) / 'memory-probe')))
    inputs = {}
    for name, expected in rule['files'].items():
        file = directory / name
        if not file.is_file():
            raise ReplayError('OPERATOR_PROBE_MISSING', name)
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ReplayError('OPERATOR_PROBE_HASH_MISMATCH', name)
        inputs[name] = json.loads(raw)
    sets = [inputs[name]['questions'] for name in (
        'probe-preregistration.json', 'probe2-preregistration.json', 'probe3-preregistration.json')]
    if len(sets[0]) != 35 or len(sets[1]) != 60 or sets[1] != sets[2]:
        raise ReplayError('OPERATOR_PROBE_COVERAGE_INVALID')
    if Counter(q['month'] for q in sets[2]) != Counter({f'2026-{m:02}': 12 for m in range(5, 10)}):
        raise ReplayError('OPERATOR_PROBE_COVERAGE_INVALID')
    tables = {}
    for family in rule['families']:
        answers = []
        for index, questions in enumerate(sets):
            prefix = ('', 'probe2-', 'probe3-')[index]
            row = inputs[f'{prefix}answers-{family}.json']
            response = row['answers']
            if (len({q['id'] for q in questions}) != len(questions) or
                    set(response) != {q['id'] for q in questions} or json.loads(row['raw']) != response):
                raise ReplayError('OPERATOR_PROBE_RESPONSE_INVALID')
            if index and not set(response.values()) <= ({'up', 'down'} if index == 2 else {'up', 'down', 'unknown'}):
                raise ReplayError('OPERATOR_PROBE_RESPONSE_INVALID')
            answers.append(response)
        exact = 0
        for q in sets[0]:
            try:
                difference = abs(Decimal(str(answers[0][q['id']])) - Decimal(str(q['eps'])))
                exact += difference.is_finite() and difference <= Decimal(rule['exact_eps_tolerance'])
            except InvalidOperation:
                if answers[0][q['id']] != 'unknown':
                    raise ReplayError('OPERATOR_PROBE_RESPONSE_INVALID') from None
        tables[family] = {'exact_eps': {'correct': exact, 'n': len(sets[0])},
                          'optional_direction': {'unknown': sum(a == 'unknown' for a in answers[1].values()),
                                                 'n': len(sets[1])},
                          'forced_direction': forced_summary(sets[2], answers[2], rule)}
    if tables['opus']['forced_direction']['flagged'] or not tables['sonnet']['forced_direction']['flagged']:
        raise ReplayError('OPERATOR_PROBE_DECISION_MISMATCH')
    return {'spec_hash': spec.SPEC_HASH, 'floor': rule['floor'], 'excluded_families': rule['excluded_families'],
            'tables': tables, 'file_digests': rule['files'], 'evidence_hash': sha256_canonical(rule['files']),
            'limitations': rule['limitations']}


def floor(root, families):
    audit = evidence(root)
    return max(audit['floor'][f] for f in families)
