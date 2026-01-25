"""
Correlation Tracking Module - Monitor portfolio correlation risk.

Prevents overexposure to correlated assets (NVDA, MSFT, QQQ all tech).
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default correlation matrix (fallback if yfinance fails)
DEFAULT_CORRELATIONS = {
    ("NVDA", "MSFT"): 0.75,
    ("NVDA", "QQQ"): 0.85,
    ("MSFT", "QQQ"): 0.90,
    ("MSFT", "NVDA"): 0.75,
    ("QQQ", "NVDA"): 0.85,
    ("QQQ", "MSFT"): 0.90,
}


class CorrelationTracker:
    """Track and update correlation matrix for portfolio risk management."""

    def __init__(
        self,
        symbols: list[str],
        lookback_days: int = 60,
        max_correlated_notional: float = 15000,
        high_correlation_threshold: float = 0.7,
    ):
        """
        Args:
            symbols: List of symbols to track
            lookback_days: Days of history for correlation calculation
            max_correlated_notional: Max $ in highly correlated positions
            high_correlation_threshold: Correlation level considered "high"
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.max_correlated_notional = max_correlated_notional
        self.high_correlation_threshold = high_correlation_threshold
        self._matrix = None
        self._last_update: Optional[datetime] = None

    def update(self) -> dict:
        """Fetch fresh correlation matrix from yfinance."""
        try:
            import yfinance as yf
            import pandas as pd

            end = datetime.now()
            start = end - timedelta(days=self.lookback_days)

            # Download all symbols at once
            data = yf.download(
                self.symbols,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                logger.warning("No data returned from yfinance, using defaults")
                return DEFAULT_CORRELATIONS

            # Get closing prices and calculate returns
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                closes = data[["Close"]]
                closes.columns = self.symbols

            returns = closes.pct_change().dropna()
            corr_matrix = returns.corr()

            # Convert to dict
            self._matrix = {}
            for sym1 in self.symbols:
                for sym2 in self.symbols:
                    if sym1 != sym2:
                        try:
                            self._matrix[(sym1, sym2)] = corr_matrix.loc[sym1, sym2]
                        except KeyError:
                            self._matrix[(sym1, sym2)] = DEFAULT_CORRELATIONS.get(
                                (sym1, sym2), 0.5
                            )

            self._last_update = datetime.now()
            logger.info(f"Correlation matrix updated: {self._matrix}")
            return self._matrix

        except Exception as e:
            logger.warning(f"Failed to update correlation matrix: {e}")
            self._matrix = DEFAULT_CORRELATIONS.copy()
            self._last_update = datetime.now()
            return self._matrix

    def _is_stale(self) -> bool:
        """Check if matrix needs refresh (>6h old)."""
        if self._last_update is None or self._matrix is None:
            return True
        return (datetime.now() - self._last_update).seconds > 21600

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if self._is_stale():
            self.update()

        if sym1 == sym2:
            return 1.0

        # Try both orderings
        if self._matrix:
            if (sym1, sym2) in self._matrix:
                return self._matrix[(sym1, sym2)]
            if (sym2, sym1) in self._matrix:
                return self._matrix[(sym2, sym1)]

        # Fallback to defaults
        if (sym1, sym2) in DEFAULT_CORRELATIONS:
            return DEFAULT_CORRELATIONS[(sym1, sym2)]
        if (sym2, sym1) in DEFAULT_CORRELATIONS:
            return DEFAULT_CORRELATIONS[(sym2, sym1)]

        return 0.5  # Default moderate correlation

    def get_correlated_exposure(
        self, symbol: str, current_positions: list[dict]
    ) -> float:
        """
        Calculate total correlated exposure for a symbol.

        Args:
            symbol: Symbol to check
            current_positions: List of {"symbol": str, "notional": float}

        Returns:
            Total notional of highly correlated positions
        """
        correlated_exposure = 0.0

        for pos in current_positions:
            if pos["symbol"] == symbol:
                continue

            corr = self.get_correlation(symbol, pos["symbol"])
            if corr >= self.high_correlation_threshold:
                # Weight by correlation
                correlated_exposure += pos["notional"] * corr

        return correlated_exposure

    def check_correlation_limits(
        self, new_position: dict, current_positions: list[dict]
    ) -> tuple[bool, str, float]:
        """
        Check if new position respects correlation limits.

        Args:
            new_position: {"symbol": str, "notional": float}
            current_positions: List of {"symbol": str, "notional": float}

        Returns:
            (allowed: bool, reason: str, size_multiplier: float)
        """
        symbol = new_position["symbol"]
        notional = new_position["notional"]

        # Calculate correlated exposure
        correlated_exposure = self.get_correlated_exposure(symbol, current_positions)

        # Check if adding this position exceeds limit
        total_after = correlated_exposure + notional
        if total_after > self.max_correlated_notional:
            # Calculate how much we can add
            available = self.max_correlated_notional - correlated_exposure

            if available <= 0:
                return False, "correlation_limit_reached", 0.0

            # Reduce size
            multiplier = available / notional
            return True, "correlation_size_reduced", multiplier

        return True, "", 1.0

    def get_portfolio_risk(self, positions: list[dict]) -> dict:
        """
        Calculate portfolio-level risk metrics.

        Args:
            positions: [{"symbol": str, "notional": float}, ...]

        Returns:
            Dict with risk metrics
        """
        if not positions:
            return {
                "weighted_correlation": 0.0,
                "concentration_risk": 0.0,
                "num_positions": 0,
                "total_notional": 0.0,
                "max_single_exposure": 0.0,
            }

        total_notional = sum(p["notional"] for p in positions)
        if total_notional == 0:
            total_notional = 1  # Avoid division by zero

        # Weighted average correlation
        weighted_corr = 0.0
        pair_count = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                corr = self.get_correlation(p1["symbol"], p2["symbol"])
                weight = (p1["notional"] * p2["notional"]) / (total_notional ** 2)
                weighted_corr += corr * weight * 4  # Scale for visibility
                pair_count += 1

        # Concentration (Herfindahl-Hirschman Index)
        hhi = sum((p["notional"] / total_notional) ** 2 for p in positions)

        # Max single exposure
        max_exposure = max(p["notional"] for p in positions) / total_notional

        return {
            "weighted_correlation": round(weighted_corr, 3),
            "concentration_risk": round(hhi, 3),
            "num_positions": len(positions),
            "total_notional": total_notional,
            "max_single_exposure": round(max_exposure, 3),
        }


# Singleton
_tracker_instance: Optional[CorrelationTracker] = None


def get_correlation_tracker(
    symbols: list[str] = None,
) -> CorrelationTracker:
    """Get or create singleton CorrelationTracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CorrelationTracker(
            symbols=symbols or ["NVDA", "MSFT", "QQQ"]
        )
    return _tracker_instance
