"""Proper scores on every forecast; conservative paired inference over calendar blocks."""
from collections import defaultdict
import itertools
import math
import statistics

from scripts.trading_lab.replay_bench.metrics import summarize, icc_by_day

EPSILON = 1e-6


def loss(p, y):
    q = min(1 - EPSILON, max(EPSILON, p))
    return {'brier': (p - y) ** 2, 'log_loss': -(y * math.log(q) + (1 - y) * math.log1p(-q))}


def proper(rows, base_probabilities):
    if not rows:
        return {'n_forecasts': 0, 'brier': None, 'log_loss': None}
    scored = [loss(r['p'], r['outperform']) for r in rows]
    base = [loss(p, r['outperform']) for r, p in zip(rows, base_probabilities)]
    if len(base) != len(rows):
        raise ValueError('baseline coverage differs')
    counts = defaultdict(int)
    by_day = defaultdict(list)
    for r, s in zip(rows, scored):
        counts[(r['session'], r['cluster'])] += 1
        by_day[r['session']].append(s['brier'])
    n = len(rows)
    icc = icc_by_day(by_day)
    m = n / len(by_day)
    out = {'n_forecasts': n, 'n_days_with_forecasts': len(by_day),
           'brier_half': .25, 'log_loss_half': math.log(2),
           'probability_ess_cluster_day_kish': n * n / sum(c * c for c in counts.values()),
           'probability_icc_day': icc, 'probability_ess_design_effect': n / (1 + (m - 1) * icc),
           'log_loss_clip': EPSILON}
    for key in ('brier', 'log_loss'):
        out[key] = statistics.mean(s[key] for s in scored)
        out[key + '_base_rate'] = statistics.mean(s[key] for s in base)
        out[key + '_improvement_vs_half'] = out[key + '_half'] - out[key]
        out[key + '_improvement_vs_base_rate'] = out[key + '_base_rate'] - out[key]
    return out


def trading(rows, dates, pnl, decisions):
    out = summarize(rows, dates, pnl_by_day=pnl, annualization=365)
    # Abstentions still create a full zero P&L series and zero drawdown.
    if not rows:
        out.update(hits=0, hit_rate=None, hit_ci95=None, expectancy_R=None, daily_pnl_sharpe=None,
                   max_drawdown=0.0, cumulative_return_sum=0.0, daily_pnl={d: 0.0 for d in dates},
                   ess_cluster_day_kish=0.0, ess_design_effect=0.0)
    n = len(decisions)
    abstain = sum(v == 'ABSTAIN' for v in decisions)
    out.update(n_decisions=n, n_abstentions=abstain, n_trade_decisions=n - abstain,
               abstention_rate=abstain / n if n else None, n_scored_trades=len(rows))
    return out


def sign_flip(rows, session_days, *, block_sessions=10):
    """Exact two-sided sign-flip of sums of daily means in nonoverlapping 10-session blocks.

    Equal weight to days (not to stocks). All horizons/instruments in a block move together.
    With 28 test sessions there are at most three blocks: coarse p-values are intentional.
    """
    if not rows:
        return {'n_pairs': 0, 'p_two_sided': None, 'mean_opus_minus_gpt': None, 'n_blocks': 0}
    per_day = defaultdict(list)
    for r in rows:
        per_day[r['session']].append(r['difference'])
    day_values = {d: statistics.mean(v) for d, v in per_day.items()}
    blocks = []
    for i in range(0, len(session_days), block_sessions):
        values = [day_values[d] for d in session_days[i:i + block_sessions] if d in day_values]
        if values:
            blocks.append(sum(values))
    observed = abs(sum(blocks))
    values = [abs(sum(s * b for s, b in zip(signs, blocks)))
              for signs in itertools.product((-1, 1), repeat=len(blocks))]
    return {'n_pairs': len(rows), 'n_days': len(day_values), 'n_blocks': len(blocks),
            'block_sessions': block_sessions, 'effective_independent_blocks': len(blocks),
            'mean_opus_minus_gpt': statistics.mean(day_values.values()),
            'p_two_sided': sum(v >= observed - 1e-12 for v in values) / len(values),
            'permutations': len(values), 'method': 'exact paired block sign-flip; equal-weight daily means',
            'assumption': 'exchangeable signs of block loss differences; adjacent blocks may remain dependent'}


def paired(opus, gpt, session_days):
    def keyed(rows):
        result = {(r['panel'], r['session'], r['asset'], r['horizon']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate paired forecast')
        return result
    left, right = keyed(opus), keyed(gpt)
    if left.keys() != right.keys():
        raise ValueError('paired forecast coverage differs')
    groups = defaultdict(list)
    for key, row in left.items():
        other = right[key]
        if row['outperform'] != other['outperform']:
            raise ValueError('paired outcome differs')
        for metric in ('brier', 'log_loss'):
            difference = loss(row['p'], row['outperform'])[metric] - loss(other['p'], row['outperform'])[metric]
            groups[row['panel'] + ':' + metric].append({'session': row['session'], 'difference': difference})
    result = {key: sign_flip(rows, session_days) for key, rows in groups.items()}
    ordered = sorted(result, key=lambda k: result[k]['p_two_sided'])
    running = 0.0
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, result[key]['p_two_sided'] * (len(ordered) - i)))
        result[key]['p_holm'] = running
    return result
