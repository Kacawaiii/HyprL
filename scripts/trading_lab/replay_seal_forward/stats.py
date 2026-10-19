"""The one declared primary endpoint: a:1d Brier skill vs the frozen dev base rate.

Brier skill = 1 - mean model Brier / mean base-rate Brier over the same forecasts. 95 % CIs
come from a moving-block bootstrap over decision sessions (blocks of 5 consecutive sessions,
10,000 resamples, fixed seed), per model and for the paired Opus-minus-GPT difference on the
same resampled sessions. "Winner" wording is allowed only when the paired CI excludes 0.
"""
from collections import defaultdict
import random

from scripts.trading_lab.replay_bench import marketdata as md
from scripts.trading_lab.replay_bench.grant import ReplayError

PRIMARY = {'panel': 'a_v1_as_live:1d', 'metric': 'brier_skill_vs_frozen_dev_base_rate',
           'ci': 'moving-block bootstrap over sessions, block 5, 10000 resamples, seed 20261018, percentile 95%',
           'paired': 'opus minus gpt on identical resampled sessions and forecasts',
           'decision_rule': 'winner wording only if the paired 95% CI excludes 0; otherwise NO_WINNER',
           'minimum_sessions_for_ci': 10}
BLOCK, RESAMPLES, SEED = 5, 10000, 20261018
# Fewer than two blocks gives a degenerate interval: no CI and no verdict below this.
MIN_SESSIONS = 2 * BLOCK


def _per_day(rows, base_p):
    days = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        y = 1.0 if r['outperform'] else 0.0
        cell = days[r['session']]
        cell[0] += (r['p'] - y) ** 2
        cell[1] += (base_p - y) ** 2
        cell[2] += 1
    return days


def skill(cells):
    model = sum(c[0] for c in cells)
    base = sum(c[1] for c in cells)
    return None if not base else 1 - model / base


def _ci(values):
    values = sorted(v for v in values if v is not None)
    if len(values) < 100:
        return None
    return [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]


def primary_endpoint(opus_rows, gpt_rows, base_p, *, panel=PRIMARY['panel']):
    """Rows: dicts with panel, session, asset, horizon, p, outperform. Only paired keys count."""
    if base_p is None:
        return dict(PRIMARY, state='BASE_RATE_UNAVAILABLE', verdict='NO_WINNER')
    key = lambda r: (r['session'], r['asset'], r['horizon'])
    left = {key(r): r for r in opus_rows if r['panel'] == panel}
    right = {key(r): r for r in gpt_rows if r['panel'] == panel}
    common = sorted(set(left) & set(right))
    if any(left[k]['outperform'] != right[k]['outperform'] for k in common):
        raise ReplayError('PRIMARY_TARGET_MISMATCH')
    own = {'opus': _per_day(left.values(), base_p), 'gpt': _per_day(right.values(), base_p)}
    own_point = {f: skill(d.values()) for f, d in own.items()}
    own_ci = {f: _bootstrap(d) for f, d in own.items()}
    a = _per_day([left[k] for k in common], base_p)
    b = _per_day([right[k] for k in common], base_p)
    sessions = sorted(a)
    out = dict(PRIMARY, base_rate=base_p, n_pairs=len(common), n_sessions=len(sessions),
               unpaired_dropped={'opus': len(left) - len(common), 'gpt': len(right) - len(common)})
    out.update(brier_skill=own_point, per_model_ci95=own_ci,
               per_model_sessions={f: len(d) for f, d in own.items()})
    if not sessions:
        return dict(out, state='NO_PAIRED_FORECASTS', ci95=dict(own_ci, difference=None), verdict='NO_WINNER')
    point = {'opus': skill(a.values()), 'gpt': skill(b.values())}
    difference = None if None in point.values() else point['opus'] - point['gpt']
    if len(sessions) < MIN_SESSIONS:
        return dict(out, state='TOO_FEW_SESSIONS_FOR_CI', paired_brier_skill=point, difference_opus_minus_gpt=difference,
                    ci95=dict(own_ci, difference=None), verdict='NO_WINNER')
    paired_ci = _bootstrap(a, b)
    verdict = 'NO_WINNER'
    if paired_ci and (paired_ci[0] > 0 or paired_ci[1] < 0):
        verdict = 'OPUS_BETTER' if paired_ci[0] > 0 else 'GPT_BETTER'
    return dict(out, state='COMPUTED', paired_brier_skill=point, difference_opus_minus_gpt=difference,
                ci95=dict(own_ci, difference=paired_ci), verdict=verdict)


def _bootstrap(a, b=None):
    if len(a) < MIN_SESSIONS:
        return None
    # Missing sessions retain their place in each block; a failure never makes
    # two distant exchange sessions appear consecutive.
    sessions = [s.session_date for s in md.sessions(min(a), max(a))]
    rng = random.Random(SEED)
    starts = max(1, len(sessions) - BLOCK + 1)
    blocks = -(-len(sessions) // BLOCK)
    draws = []
    for _ in range(RESAMPLES):
        chosen = []
        for _ in range(blocks):
            s = rng.randrange(starts)
            chosen += sessions[s:s + BLOCK]
        chosen = chosen[:len(sessions)]
        x = skill([a.get(d, (0, 0, 0)) for d in chosen])
        if b is None:
            draws.append(x)
        else:
            y = skill([b.get(d, (0, 0, 0)) for d in chosen])
            draws.append(None if x is None or y is None else x - y)
    return _ci(draws)


def wording(endpoint):
    """The only sentence a report may use about superiority."""
    if endpoint.get('verdict') == 'OPUS_BETTER':
        return 'Opus wins on the primary endpoint (paired 95% CI excludes 0).'
    if endpoint.get('verdict') == 'GPT_BETTER':
        return 'GPT wins on the primary endpoint (paired 95% CI excludes 0).'
    return 'No winner: the paired 95% CI of the primary endpoint includes 0 or is not yet estimable.'
