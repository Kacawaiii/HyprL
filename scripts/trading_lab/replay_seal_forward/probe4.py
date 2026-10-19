"""Probe 4: do Opus or GPT know how fixed-universe names moved between 2026-10-01 and 2026-10-16?

Forced up/down on close-to-close moves over two windows (2026-09-30 -> 10-08 and 10-08 -> 10-16),
balanced so always-up scores exactly 50 %. The preregistration (questions, answers, rule, prompt)
is written privately and its SHA-256 is committed before any model call. A flagged model's test
period is cut at 2026-09-30.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.replay_bench.grant import Budget, ReplayError, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import models, spec

WINDOWS = (('2026-09-30', '2026-10-08'), ('2026-10-08', '2026-10-16'))
PREREGISTRATION = 'probe4-preregistration.json'
ARTIFACT = spec.ROOT / 'docs/artifacts/replay_memory_probe4_v1.json'
CUT = '2026-09-30'
ALPHA_POOLED, ALPHA_WINDOW = .05, .025


def threshold(n, alpha):
    """Smallest k with one-sided P(X >= k | n, 1/2) <= alpha."""
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n <= alpha:
            return k
    return n + 1


def questions(closes):
    """closes: {symbol: {date: close}}. Balanced per window: equal ups and downs, chosen by symbol order."""
    out = []
    for index, (first, last) in enumerate(WINDOWS):
        moves = {s: c[last] > c[first] for s, c in sorted(closes.items())
                 if first in c and last in c and c[last] != c[first]}
        ups = [s for s, up in moves.items() if up]
        downs = [s for s, up in moves.items() if not up]
        k = min(len(ups), len(downs))
        chosen = sorted([(s, True) for s in ups[:k]] + [(s, False) for s in downs[:k]])
        for symbol, up in chosen:
            out.append({'id': f'W{index + 1}-{symbol}', 'window': f'{first}..{last}', 'start': first, 'end': last,
                        'symbol': symbol, 'up': up})
    return out


def rule(items):
    n = len(items)
    per = {w: sum(q['window'] == w for q in items) for w in {q['window'] for q in items}}
    return {'text': ('Forced choice up/down, no unknown. Balanced design: always-up scores 50%. Flagged if pooled correct '
                     f'>= {threshold(n, ALPHA_POOLED)}/{n} (one-sided binomial p <= {ALPHA_POOLED}) or any window correct >= its '
                     f'threshold (p <= {ALPHA_WINDOW}). A flagged model has its test period cut at {CUT}. Fixed before any model call.'),
            'pooled_threshold': threshold(n, ALPHA_POOLED), 'n': n,
            'window_thresholds': {w: threshold(m, ALPHA_WINDOW) for w, m in sorted(per.items())}, 'cut': CUT}


def prompt(items):
    lines = ['Answer from your own knowledge only; no tools. For each id, say whether the stock or ETF closing price '
             '(regular session, unadjusted) on the end date was higher ("up") or lower ("down") than on the start date. '
             'You must answer "up" or "down" for every id, guessing if unsure. Return only the JSON object.']
    lines += [f"{q['id']}: {q['symbol']} close {q['start']} -> close {q['end']}" for q in items]
    return '\n'.join(lines)


def schema(items):
    return {'type': 'object', 'additionalProperties': False, 'required': [q['id'] for q in items],
            'properties': {q['id']: {'type': 'string', 'enum': ['up', 'down']} for q in items}}


def grade(prereg, answers):
    items = prereg['questions']
    if set(answers) != {q['id'] for q in items} or not set(answers.values()) <= {'up', 'down'}:
        raise ReplayError('PROBE4_RESPONSE_INVALID')
    r = prereg['criteria']
    if r != rule(items):
        raise ReplayError('PROBE4_RULE_CHANGED')
    correct = {q['id']: answers[q['id']] == ('up' if q['up'] else 'down') for q in items}
    by_window = {w: sum(correct[q['id']] for q in items if q['window'] == w) for w in r['window_thresholds']}
    pooled = sum(correct.values())
    flagged = pooled >= r['pooled_threshold'] or any(by_window[w] >= t for w, t in r['window_thresholds'].items())
    return {'pooled_correct': pooled, 'n': len(items), 'correct_by_window': by_window,
            'always_up': sum(q['up'] for q in items),
            'binomial_one_sided_p': sum(math.comb(len(items), k) for k in range(pooled, len(items) + 1)) / 2 ** len(items),
            'flagged': flagged, 'test_end': CUT if flagged else spec.spec()['periods']['test'][1]}


def preregister(directory, collector):
    target = Path(directory) / PREREGISTRATION
    if target.exists():
        raise ReplayError('PROBE4_ALREADY_PREREGISTERED')
    symbols = list(sn.STOCKS + sn.ETFS)
    bars, _ = md.stock_bars(collector, symbols, '2026-09-29T00:00:00Z', '2026-10-17T00:00:00Z')
    closes = {s: {b['t'][:10]: b['c'] for b in rows} for s, rows in bars.items()}
    items = questions(closes)
    if any(sum(q['window'] == f'{a}..{b}' for q in items) < 10 for a, b in WINDOWS):
        raise ReplayError('PROBE4_INSUFFICIENT_QUESTIONS')
    value = {'questions': items, 'rule': rule(items)['text'], 'criteria': rule(items),
             'prompt_sha256': hashlib.sha256(prompt(items).encode()).hexdigest(),
             'models': spec.spec()['models']['identities'], 'created_at': iso(now()),
             'source': 'Alpaca SIP raw daily closes (grant replay-data-collection-v1)'}
    atomic_json(target, value)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    public = {'schema': 'replay-memory-probe4-v1', 'preregistration_sha256': digest, 'n_questions': len(items),
              'windows': [f'{a}..{b}' for a, b in WINDOWS], 'rule': value['rule'], 'criteria': value['criteria'],
              'prompt_sha256': value['prompt_sha256'],
              'models': value['models'], 'note': 'Digest only; questions and answers stay private until graded.'}
    atomic_json(ARTIFACT, public)
    return public


def committed_preregistration(directory):
    """The private file must match the digest committed in Git (HEAD), before any call."""
    public = json.loads(subprocess.check_output(['git', '-C', str(spec.ROOT), 'show', 'HEAD:' + str(ARTIFACT.relative_to(spec.ROOT))]))
    file = Path(directory) / PREREGISTRATION
    if hashlib.sha256(file.read_bytes()).hexdigest() != public['preregistration_sha256']:
        raise ReplayError('PROBE4_PREREGISTRATION_HASH_MISMATCH')
    return json.loads(file.read_text()), public


def ask(directory, family, budget, *, runner=subprocess.run, clock=now):
    prereg, _ = committed_preregistration(directory)
    target = Path(directory) / f'probe4-answers-{family}.json'
    if target.exists() or target.with_suffix('.intent').exists():
        raise ReplayError('PROBE4_ALREADY_ASKED')
    text, shape = prompt(prereg['questions']), schema(prereg['questions'])
    if hashlib.sha256(text.encode()).hexdigest() != prereg['prompt_sha256']:
        raise ReplayError('PROBE4_PROMPT_HASH_MISMATCH')
    at = clock()
    budget.reserve_probe(family, 150, at)
    atomic_json(target.with_suffix('.intent'), {'at': iso(at), 'family': family})
    with tempfile.TemporaryDirectory(prefix='probe4-') as private:
        cwd = Path(private) / 'empty'
        cwd.mkdir()
        schema_file = Path(private) / 'schema.json'
        atomic_json(schema_file, shape)
        env = {k: v for k, v in os.environ.items() if k not in (
            'CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CODEX_THREAD_ID', 'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'OPENAI_BASE_URL', 'ANTHROPIC_BASE_URL')}
        env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY='1', CLAUDE_CODE_DISABLE_CLAUDE_MDS='1', CLAUDE_CODE_SIMPLE='1')
        try:
            result = runner(models.argv(family, shape, schema_file), input=text, text=True, capture_output=True,
                            cwd=str(cwd), env=env, timeout=900)
        except (subprocess.TimeoutExpired, OSError):
            raise ReplayError('CALL_OUTCOME_UNKNOWN') from None
    # Preserve an unsuccessful/uncertain call too. No second probe after inspecting a response.
    atomic_json(target.with_suffix('.response'), {'raw': result.stdout or '', 'err': result.stderr or '',
                                                 'returncode': result.returncode, 'at': iso(at)})
    try:
        output, reported = models.decode(result.stdout or '', result.stderr or '', result.returncode, family, shape)
    except ReplayError as error:
        if error.code == 'MODEL_USAGE_LIMIT':
            budget.stop_model(family, at)
        elif error.code == 'API_CREDIT_EXHAUSTED':
            budget.credit_stop(at)
        raise
    value = {'model': spec.spec()['models']['identities'][family], 'reported': reported, 'at': iso(at),
             'prompt_sha256': prereg['prompt_sha256'], 'answers': output, 'raw': result.stdout}
    atomic_json(target, value)
    target.with_suffix('.intent').unlink()
    return grade(prereg, output)


def evidence(directory):
    prereg, public = committed_preregistration(directory)
    out = {'preregistration_sha256': public['preregistration_sha256'], 'families': {}}
    for family in ('opus', 'gpt'):
        file = Path(directory) / f'probe4-answers-{family}.json'
        if not file.is_file():
            raise ReplayError('PROBE4_ANSWERS_MISSING', family)
        value = json.loads(file.read_text())
        if value['model'] != spec.spec()['models']['identities'][family] or value['prompt_sha256'] != prereg['prompt_sha256']:
            raise ReplayError('PROBE4_IDENTITY_MISMATCH', family)
        out['families'][family] = dict(grade(prereg, value['answers']),
                                       answers_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    return out


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog='replay_seal_forward.probe4')
    parser.add_argument('--probe-root', default=os.environ.get('REPLAY_PROBE_ROOT'), required=True)
    parser.add_argument('--data-root', default=os.environ.get('REPLAY_DATA_ROOT'), required=True)
    parser.add_argument('command', choices=('preregister', 'ask-opus', 'ask-gpt', 'evidence'))
    parser.add_argument('--collection-grant')
    parser.add_argument('--authorization')
    parser.add_argument('--alpaca-env')
    args = parser.parse_args(argv)
    try:
        from scripts.trading_lab.replay_bench.grant import Grant
        Grant(args.authorization or os.environ.get('REPLAY_AUTHORIZATION')).check()
        if args.command == 'preregister':
            from .collect import Collector, Grants
            collector = Collector(Grants(args.collection_grant, args.authorization), Budget(Path(args.data_root) / 'collector'),
                                  Path(args.data_root) / 'cache', alpaca_env=args.alpaca_env)
            result = preregister(args.probe_root, collector)
        elif args.command == 'evidence':
            result = evidence(args.probe_root)
        else:
            result = ask(args.probe_root, args.command.split('-')[1], models.DuelBudget(args.data_root))
    except ReplayError as error:
        result = {'status': 'BLOCKED', 'error': error.code, 'evidence': str(error.evidence)}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2 if result.get('status') == 'BLOCKED' else 0


if __name__ == '__main__':
    sys.exit(main())
