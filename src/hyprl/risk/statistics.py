"""
Statistical corrections for backtesting.

Implements:
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Multiple testing correction for parameter search
- Probability of Backtest Overfitting (PBO)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz & Stegun)."""
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    # Rational approximation
    sign = 1.0 if p > 0.5 else -1.0
    p = p if p > 0.5 else 1.0 - p

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))


def sharpe_std_error(sharpe: float, n_obs: int) -> float:
    """
    Standard error of Sharpe ratio estimate.

    Based on Lo (2002) "The Statistics of Sharpe Ratios".

    Args:
        sharpe: Estimated Sharpe ratio
        n_obs: Number of observations (trades or return periods)

    Returns:
        Standard error of the Sharpe estimate
    """
    if n_obs <= 1:
        return float("inf")
    # SE(SR) = sqrt((1 + 0.5*SR²) / (n-1))
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / (n_obs - 1))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio accounting for multiple testing.

    From Bailey & López de Prado (2014) "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality".

    The DSR corrects for the fact that when testing N strategies, the best one
    will appear to have a higher Sharpe than it truly does, simply by chance.

    Args:
        sharpe: Observed Sharpe ratio of the selected strategy
        n_obs: Number of observations (trades or return periods)
        n_trials: Number of strategies/parameter combinations tested
        skewness: Skewness of returns (default 0 = normal)
        kurtosis: Kurtosis of returns (default 3 = normal)

    Returns:
        Deflated Sharpe Ratio (DSR). Values > 2 suggest genuine skill.
        DSR < 0 means the strategy is likely overfit.
    """
    if not math.isfinite(sharpe) or n_obs <= 1 or n_trials < 1:
        return float("nan")

    # Standard error of Sharpe (adjusted for non-normality)
    se_sr = sharpe_std_error(sharpe, n_obs)

    # Adjust SE for skewness and excess kurtosis
    # From Opdyke (2007)
    gamma3 = skewness
    gamma4 = kurtosis - 3.0  # Excess kurtosis
    adj = 1.0 + (gamma3 * sharpe / 4.0) + (gamma4 * sharpe * sharpe / 24.0)
    se_sr_adj = se_sr * math.sqrt(max(adj, 0.01))

    # Expected maximum Sharpe from N independent trials
    # E[max(SR)] ≈ sqrt(2*ln(N)) * SE(SR) for large N
    # More accurate: use order statistics
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        # Euler-Mascheroni constant
        gamma = 0.5772156649
        # Expected max from Gumbel distribution
        e_max_sr = se_sr_adj * ((1.0 - gamma) * _norm_ppf(1.0 - 1.0 / n_trials) +
                                 gamma * _norm_ppf(1.0 - 1.0 / (n_trials * math.e)))

    # Deflated Sharpe Ratio
    if se_sr_adj <= 0:
        return float("nan")

    dsr = (sharpe - e_max_sr) / se_sr_adj
    return dsr


def probability_of_backtest_overfitting(
    sharpe: float,
    n_obs: int,
    n_trials: int,
) -> float:
    """
    Probability that the observed Sharpe is due to overfitting.

    This is the probability that a randomly selected strategy would have
    achieved the observed Sharpe just by chance, given N trials.

    Args:
        sharpe: Observed Sharpe ratio
        n_obs: Number of observations
        n_trials: Number of strategies tested

    Returns:
        PBO in [0, 1]. Values > 0.5 suggest likely overfitting.
    """
    dsr = deflated_sharpe_ratio(sharpe, n_obs, n_trials)
    if not math.isfinite(dsr):
        return 0.5
    # PBO = 1 - Φ(DSR)
    return 1.0 - _norm_cdf(dsr)


def minimum_track_record_length(
    sharpe: float,
    target_confidence: float = 0.95,
) -> int:
    """
    Minimum track record length needed for statistical significance.

    From Bailey & López de Prado (2012).

    Args:
        sharpe: Annualized Sharpe ratio
        target_confidence: Desired confidence level (default 95%)

    Returns:
        Minimum number of years needed for significance
    """
    if sharpe <= 0:
        return 999
    z = _norm_ppf(target_confidence)
    # MinTRL = ((z / SR)² * (1 + 0.5*SR²)) years
    min_years = (z / sharpe) ** 2 * (1.0 + 0.5 * sharpe * sharpe)
    return max(1, int(math.ceil(min_years)))


@dataclass(slots=True)
class SharpeStatistics:
    """Statistical analysis of a Sharpe ratio estimate."""

    sharpe: float
    n_obs: int
    n_trials: int
    std_error: float
    deflated_sharpe: float
    pbo: float  # Probability of backtest overfitting
    min_track_record_years: int
    is_significant: bool  # DSR > 2 typically

    @classmethod
    def compute(
        cls,
        sharpe: float,
        n_obs: int,
        n_trials: int = 1,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        significance_threshold: float = 2.0,
    ) -> "SharpeStatistics":
        """Compute comprehensive Sharpe statistics."""
        se = sharpe_std_error(sharpe, n_obs)
        dsr = deflated_sharpe_ratio(sharpe, n_obs, n_trials, skewness, kurtosis)
        pbo = probability_of_backtest_overfitting(sharpe, n_obs, n_trials)
        min_trl = minimum_track_record_length(sharpe)
        is_sig = math.isfinite(dsr) and dsr >= significance_threshold

        return cls(
            sharpe=sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            std_error=se,
            deflated_sharpe=dsr,
            pbo=pbo,
            min_track_record_years=min_trl,
            is_significant=is_sig,
        )
