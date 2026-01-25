"""
Model Drift Detection for HyprL.
Detects feature drift, prediction drift, and performance degradation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from scipy import stats
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    drift_type: str  # "feature", "prediction", "performance"
    feature_name: Optional[str]
    severity: str  # "warning", "critical"
    psi_score: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov
    p_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "drift_type": self.drift_type,
            "feature_name": self.feature_name,
            "severity": self.severity,
            "psi_score": self.psi_score,
            "ks_statistic": self.ks_statistic,
            "p_value": self.p_value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class DriftDetector:
    """
    Detects drift in features and predictions.

    Uses Population Stability Index (PSI) and Kolmogorov-Smirnov test
    to identify when distributions have shifted significantly.
    """

    # PSI thresholds
    PSI_WARNING = 0.1   # Moderate drift
    PSI_CRITICAL = 0.25  # Severe drift

    # KS thresholds
    KS_WARNING = 0.1
    KS_CRITICAL = 0.2

    def __init__(
        self,
        baseline_data: Optional[dict[str, np.ndarray]] = None,
        baseline_path: Optional[Path] = None
    ):
        """
        Initialize drift detector.

        Args:
            baseline_data: Dict of {feature_name: array} from training set
            baseline_path: Path to saved baseline file
        """
        self.baseline: dict[str, np.ndarray] = {}
        self.baseline_stats: dict = {}
        self.alerts_history: list[DriftAlert] = []

        if baseline_data:
            self.set_baseline(baseline_data)
        elif baseline_path and baseline_path.exists():
            self.load_baseline(baseline_path)

    def set_baseline(self, baseline_data: dict[str, np.ndarray]):
        """Set baseline data for comparison."""
        self.baseline = baseline_data
        self.baseline_stats = self._compute_stats(baseline_data)
        logger.info(f"Baseline set with {len(baseline_data)} features")

    def save_baseline(self, path: Path):
        """Save baseline to file."""
        np.savez(path, **self.baseline)
        logger.info(f"Baseline saved to {path}")

    def load_baseline(self, path: Path):
        """Load baseline from file."""
        data = np.load(path)
        self.baseline = {k: data[k] for k in data.files}
        self.baseline_stats = self._compute_stats(self.baseline)
        logger.info(f"Baseline loaded from {path} with {len(self.baseline)} features")

    def _compute_stats(self, data: dict) -> dict:
        """Compute reference statistics."""
        computed_stats = {}
        for name, values in data.items():
            values = np.array(values)
            values = values[~np.isnan(values)]
            if len(values) > 0:
                computed_stats[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "quantiles": np.percentile(values, [10, 25, 50, 75, 90]).tolist()
                }
        return computed_stats

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change
        """
        expected = np.array(expected)
        actual = np.array(actual)

        # Remove NaN
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins from reference distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]

        # Avoid division by zero
        expected_percents = (expected_counts + 1) / (len(expected) + bins)
        actual_percents = (actual_counts + 1) / (len(actual) + bins)

        psi = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )

        return float(psi)

    def check_feature_drift(
        self,
        current_data: dict[str, np.ndarray]
    ) -> list[DriftAlert]:
        """
        Check drift for each feature.

        Args:
            current_data: Dict of {feature_name: array} for current period

        Returns:
            List of DriftAlert for features with detected drift
        """
        if not self.baseline:
            logger.warning("No baseline set, cannot check drift")
            return []

        alerts = []

        for feature_name, current_values in current_data.items():
            if feature_name not in self.baseline:
                continue

            baseline_values = self.baseline[feature_name]
            current_values = np.array(current_values)

            # Remove NaN
            baseline_clean = baseline_values[~np.isnan(baseline_values)]
            current_clean = current_values[~np.isnan(current_values)]

            if len(baseline_clean) < 10 or len(current_clean) < 10:
                continue

            # Calculate PSI
            psi = self.calculate_psi(baseline_clean, current_clean)

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_clean, current_clean)

            # Determine severity
            if psi > self.PSI_CRITICAL or ks_stat > self.KS_CRITICAL:
                severity = "critical"
            elif psi > self.PSI_WARNING or ks_stat > self.KS_WARNING:
                severity = "warning"
            else:
                continue  # No significant drift

            alert = DriftAlert(
                drift_type="feature",
                feature_name=feature_name,
                severity=severity,
                psi_score=psi,
                ks_statistic=ks_stat,
                p_value=p_value,
                message=f"Feature '{feature_name}' drift: PSI={psi:.3f}, KS={ks_stat:.3f}"
            )
            alerts.append(alert)

            if severity == "critical":
                logger.critical(f"DRIFT CRITICAL: {alert.message}")
            else:
                logger.warning(f"DRIFT WARNING: {alert.message}")

        self.alerts_history.extend(alerts)
        return alerts

    def check_prediction_drift(
        self,
        baseline_preds: np.ndarray,
        current_preds: np.ndarray
    ) -> Optional[DriftAlert]:
        """Check if prediction distribution has changed."""
        psi = self.calculate_psi(baseline_preds, current_preds)

        baseline_clean = baseline_preds[~np.isnan(baseline_preds)]
        current_clean = current_preds[~np.isnan(current_preds)]

        if len(baseline_clean) < 10 or len(current_clean) < 10:
            return None

        ks_stat, p_value = stats.ks_2samp(baseline_clean, current_clean)

        if psi > self.PSI_CRITICAL:
            severity = "critical"
        elif psi > self.PSI_WARNING:
            severity = "warning"
        else:
            return None

        alert = DriftAlert(
            drift_type="prediction",
            feature_name=None,
            severity=severity,
            psi_score=psi,
            ks_statistic=ks_stat,
            p_value=p_value,
            message=f"Prediction drift: PSI={psi:.3f}"
        )

        self.alerts_history.append(alert)
        return alert

    def check_performance_drift(
        self,
        recent_win_rate: float,
        baseline_win_rate: float = 0.54,
        n_trades: int = 20
    ) -> Optional[DriftAlert]:
        """
        Check if performance has degraded significantly.

        Uses binomial test to determine if recent win rate is
        significantly worse than baseline.
        """
        if n_trades < 10:
            return None

        wins = int(recent_win_rate * n_trades)

        # Binomial test (one-sided, testing if worse)
        # scipy binom_test is deprecated, use binomtest
        try:
            from scipy.stats import binomtest
            result = binomtest(wins, n_trades, baseline_win_rate, alternative='less')
            p_value = result.pvalue
        except ImportError:
            # Fallback for older scipy
            from scipy.stats import binom_test
            p_value = binom_test(wins, n_trades, baseline_win_rate, alternative='less')

        if p_value < 0.01:  # 99% confidence it's worse
            severity = "critical"
        elif p_value < 0.05:  # 95% confidence
            severity = "warning"
        else:
            return None

        alert = DriftAlert(
            drift_type="performance",
            feature_name=None,
            severity=severity,
            psi_score=0,
            ks_statistic=0,
            p_value=p_value,
            message=f"Performance degradation: Win rate {recent_win_rate:.1%} vs baseline {baseline_win_rate:.1%}"
        )

        self.alerts_history.append(alert)
        return alert

    def get_recent_alerts(self, hours: int = 24) -> list[DriftAlert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now().timestamp() - hours * 3600
        return [
            a for a in self.alerts_history
            if a.timestamp.timestamp() > cutoff
        ]

    def has_critical_drift(self, hours: int = 24) -> bool:
        """Check if there's any critical drift in recent period."""
        recent = self.get_recent_alerts(hours)
        return any(a.severity == "critical" for a in recent)

    def export_alerts(self, path: Path):
        """Export alerts to JSON file."""
        with open(path, 'w') as f:
            json.dump([a.to_dict() for a in self.alerts_history], f, indent=2)
