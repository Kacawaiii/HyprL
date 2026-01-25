"""
Data Quality Monitor for HyprL.
Validates incoming data before processing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report for data quality check."""
    symbol: str
    timestamp: datetime
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    staleness_seconds: int = 0
    missing_fields: list[str] = field(default_factory=list)
    outliers_detected: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "issues": self.issues,
            "staleness_seconds": self.staleness_seconds,
            "missing_fields": self.missing_fields,
            "outliers_detected": self.outliers_detected
        }


class DataQualityMonitor:
    """
    Validates data quality in real-time.

    Checks for:
    - Missing/invalid fields
    - OHLC consistency
    - Price outliers
    - Volume anomalies
    - Data staleness
    """

    # Thresholds
    MAX_STALENESS_SECONDS = 120  # Data too old
    MAX_PRICE_CHANGE_PCT = 0.15  # 15% in one candle = suspect
    MIN_VOLUME_RATIO = 0.1  # Volume < 10% of average = suspect
    MAX_SPREAD_PCT = 0.02  # Max 2% spread

    def __init__(self):
        self.last_prices: dict[str, float] = {}
        self.volume_history: dict[str, list[int]] = {}
        self.quality_history: list[DataQualityReport] = []

    def validate_candle(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: datetime
    ) -> DataQualityReport:
        """
        Validate a single candle.

        Args:
            symbol: Trading symbol
            open_, high, low, close: OHLC prices
            volume: Trading volume
            timestamp: Candle timestamp

        Returns:
            DataQualityReport with validation results
        """
        issues = []
        missing = []
        outliers = 0

        # 1. Check for missing/invalid fields
        if open_ <= 0 or np.isnan(open_):
            missing.append("open")
        if high <= 0 or np.isnan(high):
            missing.append("high")
        if low <= 0 or np.isnan(low):
            missing.append("low")
        if close <= 0 or np.isnan(close):
            missing.append("close")
        if volume < 0 or (isinstance(volume, float) and np.isnan(volume)):
            missing.append("volume")

        # 2. Check OHLC consistency
        if not missing:
            if high < low:
                issues.append(f"High ({high:.2f}) < Low ({low:.2f})")
            if high < open_ or high < close:
                issues.append("High is not the highest price")
            if low > open_ or low > close:
                issues.append("Low is not the lowest price")

        # 3. Check for price outliers
        if symbol in self.last_prices and close > 0:
            last_price = self.last_prices[symbol]
            if last_price > 0:
                change_pct = abs(close - last_price) / last_price
                if change_pct > self.MAX_PRICE_CHANGE_PCT:
                    issues.append(f"Price change {change_pct:.1%} exceeds {self.MAX_PRICE_CHANGE_PCT:.0%} threshold")
                    outliers += 1

        if close > 0:
            self.last_prices[symbol] = close

        # 4. Check volume
        if symbol in self.volume_history and len(self.volume_history[symbol]) >= 10:
            avg_volume = np.mean(self.volume_history[symbol][-20:])
            if avg_volume > 0 and volume < avg_volume * self.MIN_VOLUME_RATIO:
                issues.append(f"Volume {volume:,} < 10% of average {avg_volume:,.0f}")

        # Update volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        if volume > 0:
            self.volume_history[symbol].append(volume)
            if len(self.volume_history[symbol]) > 100:
                self.volume_history[symbol] = self.volume_history[symbol][-100:]

        # 5. Check staleness
        staleness = (datetime.now() - timestamp).total_seconds()
        if staleness > self.MAX_STALENESS_SECONDS:
            issues.append(f"Data is {staleness:.0f}s old (max: {self.MAX_STALENESS_SECONDS}s)")

        report = DataQualityReport(
            symbol=symbol,
            timestamp=timestamp,
            is_valid=len(missing) == 0 and len(issues) == 0,
            issues=issues,
            staleness_seconds=int(max(0, staleness)),
            missing_fields=missing,
            outliers_detected=outliers
        )

        self.quality_history.append(report)

        # Log issues
        if not report.is_valid:
            logger.warning(f"Data quality issue for {symbol}: {issues + missing}")

        return report

    def validate_features(
        self,
        features: dict[str, float],
        required_features: Optional[list[str]] = None
    ) -> tuple[bool, list[str]]:
        """
        Validate features before model prediction.

        Args:
            features: Dict of feature_name -> value
            required_features: List of required features

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if required_features is None:
            required_features = [
                'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',
                'atr_14', 'rsi_14', 'macd', 'bb_width'
            ]

        issues = []

        # Check required features
        for feat in required_features:
            if feat not in features:
                issues.append(f"Missing feature: {feat}")
            elif features[feat] is None:
                issues.append(f"Null value for: {feat}")
            elif np.isnan(features[feat]) or np.isinf(features[feat]):
                issues.append(f"Invalid value for {feat}: {features[feat]}")

        # Sanity checks for specific features
        if 'rsi_14' in features and features['rsi_14'] is not None:
            rsi = features['rsi_14']
            if not np.isnan(rsi) and (rsi < 0 or rsi > 100):
                issues.append(f"RSI out of range [0,100]: {rsi}")

        if 'atr_14' in features and features['atr_14'] is not None:
            atr = features['atr_14']
            if not np.isnan(atr) and atr < 0:
                issues.append(f"Negative ATR: {atr}")

        if 'bb_width' in features and features['bb_width'] is not None:
            bb = features['bb_width']
            if not np.isnan(bb) and bb < 0:
                issues.append(f"Negative Bollinger width: {bb}")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Feature validation failed: {issues}")

        return is_valid, issues

    def validate_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: datetime
    ) -> tuple[bool, list[str]]:
        """
        Validate a quote (bid/ask).

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        if bid <= 0 or np.isnan(bid):
            issues.append("Invalid bid price")
        if ask <= 0 or np.isnan(ask):
            issues.append("Invalid ask price")

        if bid > 0 and ask > 0:
            if bid > ask:
                issues.append(f"Bid ({bid}) > Ask ({ask})")

            spread_pct = (ask - bid) / bid
            if spread_pct > self.MAX_SPREAD_PCT:
                issues.append(f"Spread {spread_pct:.2%} exceeds {self.MAX_SPREAD_PCT:.0%}")

        staleness = (datetime.now() - timestamp).total_seconds()
        if staleness > 30:  # Quotes should be fresher than candles
            issues.append(f"Quote is {staleness:.0f}s old")

        return len(issues) == 0, issues

    def get_quality_summary(self, hours: int = 24) -> dict:
        """Get quality summary for recent period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.quality_history if r.timestamp > cutoff]

        if not recent:
            return {"total": 0, "valid": 0, "invalid": 0, "validity_rate": 1.0}

        valid = sum(1 for r in recent if r.is_valid)
        invalid = len(recent) - valid

        return {
            "total": len(recent),
            "valid": valid,
            "invalid": invalid,
            "validity_rate": valid / len(recent) if recent else 1.0,
            "common_issues": self._get_common_issues(recent)
        }

    def _get_common_issues(self, reports: list[DataQualityReport]) -> list[str]:
        """Get most common issues."""
        all_issues = []
        for r in reports:
            all_issues.extend(r.issues)
            all_issues.extend(r.missing_fields)

        if not all_issues:
            return []

        # Count occurrences
        from collections import Counter
        counts = Counter(all_issues)
        return [f"{issue}: {count}" for issue, count in counts.most_common(5)]

    def clear_history(self):
        """Clear quality history."""
        self.quality_history = []
