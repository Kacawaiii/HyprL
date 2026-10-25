"""Explicit null models; probabilities here are tail areas, never posteriors."""
from collections import defaultdict
import math
import random
from statistics import NormalDist


def binomial_tail(hits, n, p=0.5):
    """One-sided P(X >= hits), under independent Bernoulli(p)."""
    if not 0 <= hits <= n or not 0 < p < 1:
        raise ValueError('INVALID_BINOMIAL')
    if hits == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p) for k in range(hits, n + 1)]
    largest = max(logs)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(x - largest) for x in logs))


def required_decisions(*, p0=0.5, edge=0.05, alpha=0.05, power=0.8, family=1):
    """Normal-approximation planning, two-sided alpha, 80% power by default.

    Returns independent decisions/session clusters, not a guarantee of significance.
    Edge is fixed in advance, never the observed lucky estimate.
    """
    p1 = p0 + edge
    if not 0 < p0 < p1 < 1 or not 0 < alpha < 1 or not 0 < power < 1 or family < 1:
        raise ValueError('INVALID_POWER_PLAN')
    z = NormalDist()
    return math.ceil(((z.inv_cdf(1 - alpha / (2 * family)) * math.sqrt(p0 * (1 - p0))
                       + z.inv_cdf(power) * math.sqrt(p1 * (1 - p1))) / edge) ** 2)


def brier_null(probabilities, outcomes, *, p0=0.5, draws=10000, seed=1016):
    if (not probabilities or len(probabilities) != len(outcomes) or not 0 < p0 < 1
            or any(not 0 <= p <= 1 for p in probabilities) or any(y not in (0, 1) for y in outcomes)):
        raise ValueError('INVALID_BRIER_SAMPLE')
    n = len(probabilities)
    observed = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / n
    baseline = sum(p * p for p in probabilities)
    coefficients = [1 - 2 * p for p in probabilities]
    rng = random.Random(seed)
    count = 0
    for _ in range(draws):
        loss = (baseline + sum(w for w in coefficients if rng.random() < p0)) / n
        count += loss <= observed + 1e-14
    return {'brier': observed, 'p_null': (count + 1) / (draws + 1), 'draws': draws,
            'null': 'independent_Bernoulli_fixed_forecasts', 'base_rate': p0,
            'inferential_status': 'DIAGNOSTIC_ONLY_UNLESS_INDEPENDENCE_IS_JUSTIFIED'}


def paired_test(samples, *, family, draws=10000, seed=1016):
    """Session sign flips on mean contribution per ORIGINAL decision (cash included)."""
    groups = defaultdict(float)
    for session, delta in samples:
        if not math.isfinite(delta):
            raise ValueError('NONFINITE_DELTA')
        groups[session] += delta
    if not samples:
        return {'n': 0, 'sessions': 0, 'mean_delta': None, 'p_raw': None, 'p_adjusted': None}
    observed = sum(groups.values())
    rng = random.Random(seed)
    ge = sum(sum(value * rng.choice((-1, 1)) for value in groups.values()) >= observed - 1e-15
             for _ in range(draws))
    p = (1 + ge) / (draws + 1)
    return {'n': len(samples), 'sessions': len(groups), 'mean_delta': observed / len(samples),
            'p_raw': p, 'p_adjusted': min(1.0, p * family), 'family_size': family,
            'method': 'one_sided_session_sign_flip_Bonferroni', 'draws': draws,
            'assumption': 'independent_session_blocks_symmetric_null_deltas'}


def random_entry(slots, *, draws=10000, seed=1016):
    """Each slot binds its full point-in-time eligible universe and SAME endpoints.

    Caller supplies net returns (same side, weights and costs as real entry) for
    every eligible symbol. No missing-symbol deletion or date resampling allowed.
    Repeated identical exposures share the draw, preserving duplicate positions.
    """
    if not slots:
        return {'status': 'BLOCKED', 'reason': 'NO_CLOSED_TRADES', 'draws': 0}
    for slot in slots:
        members = slot.get('eligible_symbols', [])
        from .core import instant, protected
        try:
            touches_holdout = any(protected(symbol, slot['entry_at'], slot['exit_at'])
                                  for symbol in members + [slot.get('real_symbol', '')])
        except (KeyError, ValueError, TypeError, AttributeError):
            return {'status': 'BLOCKED', 'reason': 'INVALID_ENDPOINT_CLOCKS', 'draws': 0}
        if touches_holdout:
            return {'status': 'BLOCKED', 'reason': 'PROTECTED_HOLDOUT', 'draws': 0}
        try:
            clocks = [instant(slot[key])
                      for key in ['eligibility_known_at', 'decision_at', 'entry_at', 'exit_at']]
        except (KeyError, ValueError, TypeError, AttributeError):
            return {'status': 'BLOCKED', 'reason': 'INVALID_ENDPOINT_CLOCKS', 'draws': 0}
        candidates = slot.get('candidate_net_returns', {})
        if (not slot.get('universe_hash') or any(c.tzinfo is None for c in clocks)
                or not clocks[0] <= clocks[1] <= clocks[2] < clocks[3]
                or not members or len(members) != len(set(members)) or set(members) != set(candidates)
                or slot.get('real_symbol') not in members
                or not math.isfinite(slot['real_net_return'])
                or any(not math.isfinite(v) for v in candidates.values())
                or not math.isclose(slot['real_net_return'], candidates.get(slot.get('real_symbol'), math.nan))):
            return {'status': 'BLOCKED', 'reason': 'INCOMPLETE_MATCHED_UNIVERSE_OR_ENDPOINTS', 'draws': 0}
    rng = random.Random(seed)
    observed = sum(slot['real_net_return'] for slot in slots)
    ge = le = 0
    for _ in range(draws):
        selected = {}
        total = 0.0
        for slot in slots:
            key = (slot['entry_at'], slot['exit_at'], slot['universe_hash'], slot['real_symbol'])
            if key not in selected:
                selected[key] = rng.choice(sorted(slot['eligible_symbols']))
            total += slot['candidate_net_returns'][selected[key]]
        ge += total >= observed
        le += total <= observed
    return {'status': 'COMPUTED', 'draws': draws, 'n': len(slots), 'real_return_sum': observed,
            'percentile': 100 * le / draws, 'p_null': (ge + 1) / (draws + 1),
            'statistic': 'sum_net_portfolio_return_contributions',
            'limitations': 'conditional_symbol_randomization_only; portfolio/sector dependence not preserved; not posterior P(chance)'}
