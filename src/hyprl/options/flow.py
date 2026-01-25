"""
Options Flow Analysis Module - Use options data as leading indicator.

Analyzes put/call ratios, unusual activity, and IV to adjust trading signals.
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OptionsFlowAnalyzer:
    """Analyze options flow for trading signal adjustments."""

    # Thresholds for signal adjustment
    HIGH_PUT_CALL_RATIO = 1.5   # Bearish sentiment (contrarian bullish)
    LOW_PUT_CALL_RATIO = 0.7    # Bullish sentiment (contrarian bearish)
    UNUSUAL_VOLUME_MULT = 2.0   # Volume > 2x average = unusual

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 5 min)
        """
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[dict, datetime]] = {}

    def get_options_data(self, symbol: str, force_refresh: bool = False) -> dict:
        """
        Fetch and analyze options data for symbol.

        Args:
            symbol: Stock symbol
            force_refresh: Force cache refresh

        Returns:
            Dict with options metrics
        """
        # Check cache
        if not force_refresh and symbol in self._cache:
            data, cached_at = self._cache[symbol]
            if (datetime.now() - cached_at).seconds < self.cache_ttl:
                return data

        try:
            import yfinance as yf
            import pandas as pd

            ticker = yf.Ticker(symbol)

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options data for {symbol}")
                return self._default_data()

            # Use nearest expiration
            nearest = expirations[0]
            chain = ticker.option_chain(nearest)

            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return self._default_data()

            # Volume-based put/call ratio
            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0

            # Open interest ratio
            call_oi = calls["openInterest"].sum()
            put_oi = puts["openInterest"].sum()
            pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0

            # IV analysis
            avg_call_iv = calls["impliedVolatility"].mean()
            avg_put_iv = puts["impliedVolatility"].mean()
            avg_iv = (avg_call_iv + avg_put_iv) / 2
            iv_skew = avg_put_iv - avg_call_iv

            # Unusual activity detection
            call_vol_mean = calls["volume"].mean()
            put_vol_mean = puts["volume"].mean()

            unusual_calls = bool(
                (calls["volume"] > self.UNUSUAL_VOLUME_MULT * call_vol_mean).any()
            ) if call_vol_mean > 0 else False

            unusual_puts = bool(
                (puts["volume"] > self.UNUSUAL_VOLUME_MULT * put_vol_mean).any()
            ) if put_vol_mean > 0 else False

            # Max pain calculation
            max_pain = self._calculate_max_pain(calls, puts)

            # Get current stock price for context
            try:
                stock_price = ticker.info.get("currentPrice", 0) or ticker.info.get("regularMarketPrice", 0)
            except:
                stock_price = 0

            data = {
                "symbol": symbol,
                "expiration": nearest,
                "put_call_ratio": round(pc_ratio, 3),
                "put_call_oi_ratio": round(pc_oi_ratio, 3),
                "avg_iv": round(avg_iv, 4) if not pd.isna(avg_iv) else 0.3,
                "iv_skew": round(iv_skew, 4) if not pd.isna(iv_skew) else 0,
                "unusual_calls": unusual_calls,
                "unusual_puts": unusual_puts,
                "max_pain": round(max_pain, 2),
                "stock_price": stock_price,
                "total_call_volume": int(call_volume),
                "total_put_volume": int(put_volume),
                "total_call_oi": int(call_oi),
                "total_put_oi": int(put_oi),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache it
            self._cache[symbol] = (data, datetime.now())
            return data

        except Exception as e:
            logger.warning(f"Failed to fetch options data for {symbol}: {e}")
            return self._default_data()

    def _calculate_max_pain(self, calls, puts) -> float:
        """
        Calculate max pain price (where options expire worthless).

        Max pain is the strike price where option writers would pay
        the least to option holders.
        """
        try:
            import pandas as pd

            strikes = sorted(set(
                calls["strike"].tolist() + puts["strike"].tolist()
            ))

            if not strikes:
                return 0

            min_pain = float("inf")
            max_pain_strike = strikes[len(strikes) // 2]

            for strike in strikes:
                # Pain for call holders if price ends at this strike
                call_pain = 0
                for _, row in calls.iterrows():
                    if row["strike"] < strike:
                        call_pain += (strike - row["strike"]) * row["openInterest"]

                # Pain for put holders if price ends at this strike
                put_pain = 0
                for _, row in puts.iterrows():
                    if row["strike"] > strike:
                        put_pain += (row["strike"] - strike) * row["openInterest"]

                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike

            return max_pain_strike

        except Exception as e:
            logger.warning(f"Max pain calculation failed: {e}")
            return 0

    def _default_data(self) -> dict:
        """Return default values when data unavailable."""
        return {
            "symbol": "",
            "expiration": None,
            "put_call_ratio": 1.0,
            "put_call_oi_ratio": 1.0,
            "avg_iv": 0.3,
            "iv_skew": 0,
            "unusual_calls": False,
            "unusual_puts": False,
            "max_pain": 0,
            "stock_price": 0,
            "total_call_volume": 0,
            "total_put_volume": 0,
            "total_call_oi": 0,
            "total_put_oi": 0,
            "timestamp": datetime.now().isoformat(),
            "is_default": True,
        }

    def get_signal_adjustment(
        self,
        options_data: dict,
        decision: str,
    ) -> tuple[float, list[str]]:
        """
        Calculate signal adjustment based on options flow.

        Args:
            options_data: Options metrics from get_options_data
            decision: "long" or "short"

        Returns:
            (adjustment: float, reasons: list[str])

            Positive adjustment = more confident in signal
            Negative adjustment = less confident
        """
        adjustments = []
        reasons = []

        pc_ratio = options_data.get("put_call_ratio", 1.0)
        iv_skew = options_data.get("iv_skew", 0)
        unusual_calls = options_data.get("unusual_calls", False)
        unusual_puts = options_data.get("unusual_puts", False)

        if decision == "long":
            # High put/call = bearish sentiment = contrarian bullish
            if pc_ratio > self.HIGH_PUT_CALL_RATIO:
                adjustments.append(0.02)
                reasons.append("high_pc_contrarian_bullish")

            # Unusual calls = bullish flow
            if unusual_calls:
                adjustments.append(0.02)
                reasons.append("unusual_call_activity")

            # Unusual puts = bearish flow (bad for long)
            if unusual_puts:
                adjustments.append(-0.03)
                reasons.append("unusual_put_warning")

            # Put IV skew (puts more expensive = fear)
            if iv_skew > 0.1:
                adjustments.append(-0.01)
                reasons.append("put_skew_fear")

        elif decision == "short":
            # Low put/call = bullish sentiment = contrarian bearish
            if pc_ratio < self.LOW_PUT_CALL_RATIO:
                adjustments.append(0.02)
                reasons.append("low_pc_contrarian_bearish")

            # Unusual puts = bearish flow
            if unusual_puts:
                adjustments.append(0.02)
                reasons.append("unusual_put_activity")

            # Unusual calls = bullish flow (bad for short)
            if unusual_calls:
                adjustments.append(-0.03)
                reasons.append("unusual_call_warning")

            # Put IV skew (puts expensive = confirms fear)
            if iv_skew > 0.1:
                adjustments.append(0.01)
                reasons.append("put_skew_confirms")

        total_adjustment = sum(adjustments)

        # Cap adjustment to prevent over-reliance
        total_adjustment = max(-0.05, min(0.05, total_adjustment))

        return round(total_adjustment, 3), reasons

    def get_sentiment_summary(self, options_data: dict) -> str:
        """
        Get human-readable sentiment summary.

        Returns:
            Summary string like "Bearish (high put volume, unusual puts)"
        """
        pc_ratio = options_data.get("put_call_ratio", 1.0)
        unusual_calls = options_data.get("unusual_calls", False)
        unusual_puts = options_data.get("unusual_puts", False)

        signals = []

        if pc_ratio > self.HIGH_PUT_CALL_RATIO:
            sentiment = "Bearish"
            signals.append("high put/call")
        elif pc_ratio < self.LOW_PUT_CALL_RATIO:
            sentiment = "Bullish"
            signals.append("low put/call")
        else:
            sentiment = "Neutral"

        if unusual_calls:
            signals.append("unusual calls")
        if unusual_puts:
            signals.append("unusual puts")

        if signals:
            return f"{sentiment} ({', '.join(signals)})"
        return sentiment


# Singleton
_analyzer_instance: Optional[OptionsFlowAnalyzer] = None


def get_options_analyzer() -> OptionsFlowAnalyzer:
    """Get or create singleton OptionsFlowAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = OptionsFlowAnalyzer()
    return _analyzer_instance
