"""Sample-aware metrics: counts first, Wilson intervals, Brier, R expectancy, daily P&L, ESS."""
from collections import defaultdict
import math
import statistics


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def icc_by_day(values):
    """One-way ANOVA intra-class correlation of a per-view quantity grouped by day (0 when not estimable)."""
    groups = [v for v in values.values() if v]
    n = sum(len(g) for g in groups)
    k = len(groups)
    if k < 2 or n <= k:
        return 0.0
    grand = sum(sum(g) for g in groups) / n
    msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum((x - sum(g) / len(g)) ** 2 for g in groups for x in g) / (n - k)
    m0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    denominator = msb + (m0 - 1) * msw
    return max(0.0, (msb - msw) / denominator) if denominator > 0 else 0.0


def summarize(scored, days, *, pnl_by_day=None, annualization=252):
    """scored: list of (day, result) for directional views; days: every session day of the period."""
    n = len(scored)
    out = {'n_views': n, 'n_days': len(days), 'days_with_views': len({d for d, _ in scored})}
    if not n:
        return out
    hits = sum(r['hit'] for _, r in scored)
    gross = sum(r['hit_gross'] for _, r in scored)
    out.update(hits=hits, hit_rate=round(hits / n, 4), hit_ci95=wilson(hits, n), hits_gross=gross,
               mean_net_bp=round(1e4 * statistics.mean(r['net'] for _, r in scored), 2),
               mean_gross_bp=round(1e4 * statistics.mean(r['gross'] for _, r in scored), 2))
    rs = [r['R'] for _, r in scored if r.get('R') is not None]
    out['expectancy_R'] = round(statistics.mean(rs), 4) if rs else None
    probs = [(r['p'], r['outperform']) for _, r in scored if r.get('p') is not None]
    if probs:
        out['brier'] = round(statistics.mean((p - o) ** 2 for p, o in probs), 4)
        out['brier_n'] = len(probs)
    by_day = defaultdict(list)
    for d, r in scored:
        by_day[d].append(r['net'])
    pnl = ([pnl_by_day.get(d, 0.0) for d in days] if pnl_by_day is not None else
           [statistics.mean(by_day[d]) if by_day.get(d) else 0.0 for d in days])
    sd = statistics.stdev(pnl) if len(pnl) > 1 else 0
    out['daily_pnl_sharpe'] = round(statistics.mean(pnl) / sd * math.sqrt(annualization), 3) if sd else None
    out['daily_pnl'] = dict(zip(days, pnl))
    equity, peak, drawdown = 1.0, 1.0, 0.0
    for x in pnl:
        equity += x
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
    out['cumulative_return_sum'] = round(equity - 1, 5)
    out['max_drawdown'] = round(drawdown, 5)
    groups = {(d, r.get('cluster'), r.get('direction')) for d, r in scored}
    icc = icc_by_day({d: [r['net'] for r in v] for d, v in
                      ((d, [r for dd, r in scored if dd == d]) for d in by_day)})
    m = n / len(by_day)
    out['ess_cluster_day_direction'] = len(groups)
    cluster_counts = defaultdict(int)
    for d, r in scored:
        cluster_counts[(d, r.get('cluster'))] += 1
    out['ess_cluster_day_kish'] = round(n * n / sum(c * c for c in cluster_counts.values()), 2)
    out['wilson_interval_scope'] = 'descriptive only; assumes independence, ignores clustering'
    out['icc_day'] = round(icc, 4)
    out['ess_design_effect'] = round(n / (1 + (m - 1) * icc), 1)
    return out
