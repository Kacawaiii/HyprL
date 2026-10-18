"""Model-memory probe: dated, verifiable events asked with no context; recall per month sets the replay floor."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from .grant import ReplayError

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['knows', 'answer'],
          'properties': {'knows': {'type': 'boolean'}, 'answer': {'type': 'string', 'maxLength': 600}}}
PROMPT = ('Answer from your own memory only; you have no tools and no context. If you do not actually know the '
          'specific fact, set knows=false and answer "UNKNOWN". Do not guess. Return JSON matching the schema.\n'
          'Question: ')
# Supplementary sensitivity mode: a refusal-free best guess, graded on the answer alone.
FORCED = ('Answer from your own memory only; you have no tools and no context. Give your single best specific guess '
          'even if you are unsure (set knows=true only if you are confident). Return JSON matching the schema.\n'
          'Question: ')


def correct(event, answer):
    """Every group of `answer_key_all_of_any` must match at least one of its patterns (case-insensitive)."""
    text = answer.lower()
    return all(any(re.search(pattern, text) for pattern in group) for group in event['answer_key_all_of_any'])


def probe_set_digest(probe_set):
    return hashlib.sha256(json.dumps(probe_set, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run(claude, probe_set, families, out_path, *, workers=1, forced=False):
    if workers != 1:
        raise ReplayError('SERIAL_DISPATCH_REQUIRED')
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done[(row['family'], row['id'])] = row
    jobs = [(f, e) for f in families for e in probe_set['events'] if (f, e['id']) not in done]

    def one(job):
        family, event = job
        try:
            output, meta = claude.call(family, (FORCED if forced else PROMPT) + event['question'], SCHEMA,
                                       tag=('forced-' if forced else 'probe-') + event['id'], timeout=300)
        except ReplayError as error:
            if error.code in ('MODEL_USAGE_LIMIT', 'BUDGET_EXHAUSTED'):
                raise
            return {'family': family, 'id': event['id'], 'month': event['month'], 'error': error.code}
        return {'family': family, 'id': event['id'], 'month': event['month'], 'knows': output['knows'],
                'answer': output['answer'], 'forced': forced,
                'correct': bool((forced or output['knows']) and correct(event, output['answer'])),
                'model': meta['model'], 'reported': meta['reported']}

    with out_path.open('a') as sink:
        for job in jobs:
            row = one(job)
            sink.write(json.dumps(row) + '\n')
            sink.flush()
            done[(row['family'], row['id'])] = row
    return list(done.values())


def summarize(rows, rule):
    """Per family and month: n, claimed knowledge, correct. A month is KNOWN when correct >= rule['known_min_correct']."""
    table = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'claimed': 0, 'correct': 0, 'errors': 0}))
    for row in rows:
        cell = table[row['family']][row['month']]
        cell['n'] += 1
        if 'error' in row:
            cell['errors'] += 1
            continue
        cell['claimed'] += bool(row['knows'])
        cell['correct'] += bool(row['correct'])
    result = {}
    for family, months in table.items():
        known = [m for m, c in sorted(months.items()) if c['correct'] >= rule['known_min_correct']]
        incomplete = [m for m, c in months.items() if c['errors'] or c['n'] < rule['events_per_month']]
        last = max(known) if known else None
        if last:
            year, month = map(int, last.split('-'))
            floor = f'{year + month // 12}-{month % 12 + 1:02d}-01'
        else:
            floor = None
        result[family] = {'months': {m: dict(c) for m, c in sorted(months.items())}, 'known_months': known,
                          'last_known_month': last, 'replay_floor': floor, 'incomplete_months': sorted(incomplete)}
    return result
