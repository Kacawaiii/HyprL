"""
Strategy Guards V2 - Enhanced trading guards with all new features.

Integrates:
- Event Calendar (earnings, FOMC, OpEx)
- Correlation Limits
- Macro Regime (VIX)
- Liquidity Sizing
- Options Flow

Usage in bridge:
    from hyprl.strategy.guards_v2 import StrategyGuardsV2
    guards = StrategyGuardsV2(symbols=["NVDA", "MSFT", "QQQ"])

    # Before placing order:
    result = guards.check_all(
        symbol="NVDA",
        decision="long",
        requested_qty=10,
        price=150.0,
        current_positions=[...],
    )

    if not result["allowed"]:
        # Skip trade
        log_event("guard_reject", **result)
        continue

    # Use adjusted values
    qty = result["adjusted_qty"]
    threshold_adj = result["threshold_adjustment"]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result from guard checks."""
    allowed: bool
    reason: str
    adjusted_qty: int
    adjusted_prob: float
    threshold_adjustment: float
    size_multiplier: float
    details: dict = field(default_factory=dict)


class StrategyGuardsV2:
    """
    Enhanced strategy guards combining all new risk features.

    Features:
    - Event Calendar: Block trades near earnings/FOMC/OpEx
    - Correlation: Limit correlated position exposure
    - Macro Regime: Adjust based on VIX level
    - Liquidity: Adjust size based on time/volume/spread
    - Options Flow: Adjust probability based on options sentiment
    """

    def __init__(
        self,
        symbols: list[str] = None,
        # Feature toggles
        enable_calendar: bool = True,
        enable_correlation: bool = True,
        enable_regime: bool = True,
        enable_liquidity: bool = True,
        enable_options: bool = True,
        # Calendar settings
        earnings_blackout_days: int = 3,
        # Correlation settings
        max_correlated_notional: float = 15000,
        high_correlation_threshold: float = 0.7,
        # Liquidity settings
        max_volume_pct: float = 0.01,
        min_position_value: float = 500,
    ):
        self.symbols = symbols or ["NVDA", "MSFT", "QQQ"]

        # Feature toggles
        self.enable_calendar = enable_calendar
        self.enable_correlation = enable_correlation
        self.enable_regime = enable_regime
        self.enable_liquidity = enable_liquidity
        self.enable_options = enable_options

        # Initialize components lazily
        self._calendar = None
        self._correlation = None
        self._regime = None
        self._liquidity = None
        self._options = None

        # Settings
        self.earnings_blackout_days = earnings_blackout_days
        self.max_correlated_notional = max_correlated_notional
        self.high_correlation_threshold = high_correlation_threshold
        self.max_volume_pct = max_volume_pct
        self.min_position_value = min_position_value

    @property
    def calendar(self):
        """Lazy load EventCalendar."""
        if self._calendar is None and self.enable_calendar:
            try:
                from hyprl.calendar.events import EventCalendar
                self._calendar = EventCalendar(
                    earnings_blackout_days=self.earnings_blackout_days
                )
            except ImportError as e:
                logger.warning(f"Calendar module not available: {e}")
                self.enable_calendar = False
        return self._calendar

    @property
    def correlation(self):
        """Lazy load CorrelationTracker."""
        if self._correlation is None and self.enable_correlation:
            try:
                from hyprl.risk.correlation import CorrelationTracker
                self._correlation = CorrelationTracker(
                    symbols=self.symbols,
                    max_correlated_notional=self.max_correlated_notional,
                    high_correlation_threshold=self.high_correlation_threshold,
                )
            except ImportError as e:
                logger.warning(f"Correlation module not available: {e}")
                self.enable_correlation = False
        return self._correlation

    @property
    def regime(self):
        """Lazy load RegimeDetector."""
        if self._regime is None and self.enable_regime:
            try:
                from hyprl.regime.macro import RegimeDetector
                self._regime = RegimeDetector()
            except ImportError as e:
                logger.warning(f"Regime module not available: {e}")
                self.enable_regime = False
        return self._regime

    @property
    def liquidity(self):
        """Lazy load LiquidityManager."""
        if self._liquidity is None and self.enable_liquidity:
            try:
                from hyprl.risk.liquidity import LiquidityManager
                self._liquidity = LiquidityManager(
                    max_volume_pct=self.max_volume_pct,
                    min_position_value=self.min_position_value,
                )
            except ImportError as e:
                logger.warning(f"Liquidity module not available: {e}")
                self.enable_liquidity = False
        return self._liquidity

    @property
    def options(self):
        """Lazy load OptionsFlowAnalyzer."""
        if self._options is None and self.enable_options:
            try:
                from hyprl.options.flow import OptionsFlowAnalyzer
                self._options = OptionsFlowAnalyzer()
            except ImportError as e:
                logger.warning(f"Options module not available: {e}")
                self.enable_options = False
        return self._options

    def check_calendar(self, symbol: str) -> tuple[bool, str, dict]:
        """
        Check event calendar restrictions.

        Returns:
            (allowed: bool, reason: str, event_data: dict)
        """
        if not self.enable_calendar or self.calendar is None:
            return True, "", {}

        try:
            event_data = self.calendar.get_event_data(symbol)
            skip, reason = self.calendar.should_skip_event_risk(symbol, event_data)
            return not skip, reason, event_data
        except Exception as e:
            logger.warning(f"Calendar check failed: {e}")
            return True, "", {}

    def check_correlation(
        self,
        symbol: str,
        notional: float,
        current_positions: list[dict],
    ) -> tuple[bool, str, float, dict]:
        """
        Check correlation limits.

        Returns:
            (allowed: bool, reason: str, size_multiplier: float, risk_data: dict)
        """
        if not self.enable_correlation or self.correlation is None:
            return True, "", 1.0, {}

        try:
            new_pos = {"symbol": symbol, "notional": notional}
            allowed, reason, multiplier = self.correlation.check_correlation_limits(
                new_pos, current_positions
            )
            risk_data = self.correlation.get_portfolio_risk(current_positions)
            return allowed, reason, multiplier, risk_data
        except Exception as e:
            logger.warning(f"Correlation check failed: {e}")
            return True, "", 1.0, {}

    def check_regime(self, decision: str) -> tuple[bool, str, float, float, dict]:
        """
        Check macro regime restrictions.

        Returns:
            (allowed: bool, reason: str, threshold_adj: float, size_mult: float, macro_data: dict)
        """
        if not self.enable_regime or self.regime is None:
            return True, "", 0.0, 1.0, {}

        try:
            from hyprl.regime.macro import REGIME_CONFIGS, MarketRegime

            regime, macro_data = self.regime.detect_regime()
            config = REGIME_CONFIGS[regime]

            # Check if trading allowed
            if regime == MarketRegime.CRISIS:
                return False, "crisis_mode", 0.0, 0.0, macro_data

            # Check if shorts allowed
            if decision == "short" and not config.allow_shorts:
                return False, f"shorts_blocked_{regime.value}", 0.0, 0.0, macro_data

            return (
                True,
                regime.value,
                config.threshold_adjustment,
                config.position_size_mult,
                macro_data,
            )
        except Exception as e:
            logger.warning(f"Regime check failed: {e}")
            return True, "", 0.0, 1.0, {}

    def check_liquidity(
        self,
        symbol: str,
        requested_qty: int,
        price: float,
        avg_daily_volume: int = 0,
        current_spread_pct: float = 0,
        avg_spread_pct: float = 0,
    ) -> tuple[bool, str, int, dict]:
        """
        Check liquidity and adjust size.

        Returns:
            (allowed: bool, reason: str, adjusted_qty: int, factors: dict)
        """
        if not self.enable_liquidity or self.liquidity is None:
            return True, "", requested_qty, {}

        try:
            adjusted_qty, factors = self.liquidity.adjust_position_size(
                requested_qty=requested_qty,
                symbol=symbol,
                price=price,
                avg_daily_volume=avg_daily_volume,
                current_spread_pct=current_spread_pct,
                avg_spread_pct=avg_spread_pct,
            )

            skip, reason = self.liquidity.should_skip_trade(adjusted_qty, price, factors)
            return not skip, reason, adjusted_qty, factors
        except Exception as e:
            logger.warning(f"Liquidity check failed: {e}")
            return True, "", requested_qty, {}

    def check_options(
        self, symbol: str, decision: str, probability: float
    ) -> tuple[float, float, list[str], dict]:
        """
        Check options flow and adjust probability.

        Returns:
            (adjusted_prob: float, adjustment: float, reasons: list, options_data: dict)
        """
        if not self.enable_options or self.options is None:
            return probability, 0.0, [], {}

        try:
            options_data = self.options.get_options_data(symbol)
            adjustment, reasons = self.options.get_signal_adjustment(options_data, decision)
            adjusted_prob = probability + adjustment
            return adjusted_prob, adjustment, reasons, options_data
        except Exception as e:
            logger.warning(f"Options check failed: {e}")
            return probability, 0.0, [], {}

    def check_all(
        self,
        symbol: str,
        decision: str,
        requested_qty: int,
        price: float,
        probability: float = 0.5,
        current_positions: list[dict] = None,
        avg_daily_volume: int = 0,
        current_spread_pct: float = 0,
        avg_spread_pct: float = 0,
    ) -> dict:
        """
        Run all guard checks and return combined result.

        Args:
            symbol: Stock symbol
            decision: "long", "short", or "flat"
            requested_qty: Requested position size
            price: Current price
            probability: Signal probability
            current_positions: List of {"symbol": str, "notional": float}
            avg_daily_volume: Average daily volume
            current_spread_pct: Current bid-ask spread %
            avg_spread_pct: Average spread %

        Returns:
            Dict with all check results and adjusted values
        """
        current_positions = current_positions or []
        result = {
            "symbol": symbol,
            "decision": decision,
            "original_qty": requested_qty,
            "original_prob": probability,
            "allowed": True,
            "reject_reason": "",
            "adjusted_qty": requested_qty,
            "adjusted_prob": probability,
            "threshold_adjustment": 0.0,
            "size_multiplier": 1.0,
            "checks": {},
        }

        # 1. Calendar check
        cal_allowed, cal_reason, event_data = self.check_calendar(symbol)
        result["checks"]["calendar"] = {
            "allowed": cal_allowed,
            "reason": cal_reason,
            "data": event_data,
        }
        if not cal_allowed:
            result["allowed"] = False
            result["reject_reason"] = f"calendar:{cal_reason}"
            return result

        # 2. Regime check
        regime_allowed, regime_reason, threshold_adj, regime_mult, macro_data = self.check_regime(decision)
        result["checks"]["regime"] = {
            "allowed": regime_allowed,
            "reason": regime_reason,
            "threshold_adjustment": threshold_adj,
            "size_multiplier": regime_mult,
            "data": macro_data,
        }
        if not regime_allowed:
            result["allowed"] = False
            result["reject_reason"] = f"regime:{regime_reason}"
            return result

        result["threshold_adjustment"] = threshold_adj
        result["size_multiplier"] *= regime_mult

        # 3. Correlation check
        notional = requested_qty * price
        corr_allowed, corr_reason, corr_mult, risk_data = self.check_correlation(
            symbol, notional, current_positions
        )
        result["checks"]["correlation"] = {
            "allowed": corr_allowed,
            "reason": corr_reason,
            "size_multiplier": corr_mult,
            "data": risk_data,
        }
        if not corr_allowed:
            result["allowed"] = False
            result["reject_reason"] = f"correlation:{corr_reason}"
            return result

        result["size_multiplier"] *= corr_mult

        # 4. Liquidity check
        # Apply previous multipliers first
        pre_liq_qty = int(requested_qty * result["size_multiplier"])
        liq_allowed, liq_reason, liq_qty, liq_factors = self.check_liquidity(
            symbol=symbol,
            requested_qty=pre_liq_qty,
            price=price,
            avg_daily_volume=avg_daily_volume,
            current_spread_pct=current_spread_pct,
            avg_spread_pct=avg_spread_pct,
        )
        result["checks"]["liquidity"] = {
            "allowed": liq_allowed,
            "reason": liq_reason,
            "adjusted_qty": liq_qty,
            "factors": liq_factors,
        }
        if not liq_allowed:
            result["allowed"] = False
            result["reject_reason"] = f"liquidity:{liq_reason}"
            return result

        result["adjusted_qty"] = liq_qty

        # 5. Options flow check (adjusts probability)
        adj_prob, opt_adj, opt_reasons, options_data = self.check_options(
            symbol, decision, probability
        )
        result["checks"]["options"] = {
            "adjustment": opt_adj,
            "reasons": opt_reasons,
            "data": options_data,
        }
        result["adjusted_prob"] = adj_prob

        return result


# Convenience function for simple integration
def create_guards(
    symbols: list[str] = None,
    enable_all: bool = True,
) -> StrategyGuardsV2:
    """Create StrategyGuardsV2 with default settings."""
    return StrategyGuardsV2(
        symbols=symbols or ["NVDA", "MSFT", "QQQ"],
        enable_calendar=enable_all,
        enable_correlation=enable_all,
        enable_regime=enable_all,
        enable_liquidity=enable_all,
        enable_options=enable_all,
    )
