"""Regrade the same operator probes; publish digests/counts, never raw answers."""
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path

from scripts.trading_lab.replay_bench import audit as parent
from scripts.trading_lab.replay_bench.grant import ReplayError
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import spec


def evidence(root):
    inherited = parent.evidence(root)
    rule = spec.spec()['probe']
    directory = Path(os.environ.get('REPLAY_PROBE_ROOT', str(Path(root) / 'memory-probe')))
    file = directory / 'probe-gpt-answers.json'
    if not file.is_file() or spec.file_hash(file) != rule['gpt_answers_sha256']:
        raise ReplayError('GPT_PROBE_HASH_MISMATCH')
    value = json.loads(file.read_text())
    if value.get('model') != 'gpt-6.1-sol':
        raise ReplayError('GPT_PROBE_MODEL_MISMATCH')
    sets = [json.loads((directory / name).read_text())['questions'] for name in (
        'probe-preregistration.json', 'probe3-preregistration.json')]
    for key, questions in zip(('probe1', 'probe3'), sets):
        if set(value[key]) != {q['id'] for q in questions}:
            raise ReplayError('GPT_PROBE_COVERAGE_INVALID')
    if not set(value['probe3'].values()) <= {'up', 'down'}:
        raise ReplayError('GPT_PROBE_RESPONSE_INVALID')
    exact = 0
    for q in sets[0]:
        try:
            difference = abs(Decimal(str(value['probe1'][q['id']])) - Decimal(str(q['eps'])))
            exact += difference.is_finite() and difference <= Decimal('0.02')
        except InvalidOperation:
            if value['probe1'][q['id']] != 'unknown':
                raise ReplayError('GPT_PROBE_RESPONSE_INVALID') from None
    forced = parent.forced_summary(sets[1], value['probe3'], rule)
    if exact != 0 or forced['pooled_correct'] != 18 or forced['flagged']:
        raise ReplayError('GPT_PROBE_DECISION_MISMATCH')
    tables = dict(inherited['tables'], gpt={'exact_eps': {'correct': exact, 'n': 35}, 'forced_direction': forced})
    return {'tables': tables, 'floor': {'opus': '2026-07-01', 'gpt': '2026-07-01'},
            'file_digests': dict(inherited['file_digests'], **{'probe-gpt-answers.json': rule['gpt_answers_sha256']}),
            'evidence_hash': sha256_canonical([inherited['evidence_hash'], rule['gpt_answers_sha256']]),
            'limitations': inherited['limitations'] + ' GPT clean-session setup is operator-attested; no new calls.'}
