"""Options Income Strategies.

Generate additional income through options on existing positions.
Strategies:
- Covered Calls: Sell calls on long stock positions
- Cash-Secured Puts: Sell puts to enter positions at lower price
- Collar: Protect downside while capping upside
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import math


class OptionStrategy(Enum):
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    COLLAR = "collar"
    PROTECTIVE_PUT = "protective_put"


@dataclass
class OptionLeg:
    """Single option leg."""
    symbol: str
    expiry: datetime
    strike: float
    option_type: str  # "call" or "put"
    action: str  # "buy" or "sell"
    qty: int
    premium: float  # Per share
    delta: float = 0.0
    iv: float = 0.0


@dataclass
class IncomeOpportunity:
    """An options income opportunity."""
    underlying: str
    strategy: OptionStrategy
    legs: list[OptionLeg]
    total_premium: float
    max_profit: float
    max_loss: float
    breakeven: float
    probability_profit: float
    annualized_return: float
    days_to_expiry: int
    risk_reward_ratio: float
    recommendation: str  # "strong", "moderate", "weak", "avoid"
    reason: str


@dataclass
class IncomeConfig:
    """Configuration for income strategies."""
    # Strike selection
    min_delta_call: float = 0.20  # Sell calls with delta <= 0.20 (OTM)
    max_delta_call: float = 0.35
    min_delta_put: float = -0.35
    max_delta_put: float = -0.20

    # Expiry preferences
    min_dte: int = 7  # Minimum days to expiry
    max_dte: int = 45  # Maximum days to expiry
    preferred_dte: int = 30  # Target ~30 DTE for theta decay

    # Premium requirements
    min_premium_pct: float = 0.5  # Min 0.5% premium
    min_annualized_return: float = 12.0  # Min 12% annualized

    # Risk management
    max_positions_pct: float = 0.50  # Max 50% of portfolio in options
    require_100_shares: bool = True  # Only sell covered calls on 100+ shares


class OptionsIncomeAnalyzer:
    """Analyzes and recommends options income strategies."""

    def __init__(self, config: Optional[IncomeConfig] = None):
        self.config = config or IncomeConfig()

    def estimate_option_price(
        self,
        stock_price: float,
        strike: float,
        days_to_expiry: int,
        volatility: float,
        option_type: str,
        risk_free_rate: float = 0.05,
    ) -> tuple[float, float]:
        """Estimate option price and delta using Black-Scholes approximation.

        Returns: (price, delta)
        """
        if days_to_expiry <= 0:
            # At expiry
            if option_type == "call":
                return max(0, stock_price - strike), 1.0 if stock_price > strike else 0.0
            else:
                return max(0, strike - stock_price), -1.0 if stock_price < strike else 0.0

        T = days_to_expiry / 365.0
        sigma = volatility

        # Black-Scholes
        d1 = (math.log(stock_price / strike) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if option_type == "call":
            price = stock_price * norm_cdf(d1) - strike * math.exp(-risk_free_rate * T) * norm_cdf(d2)
            delta = norm_cdf(d1)
        else:
            price = strike * math.exp(-risk_free_rate * T) * norm_cdf(-d2) - stock_price * norm_cdf(-d1)
            delta = norm_cdf(d1) - 1

        return max(0.01, price), delta

    def find_covered_call_strikes(
        self,
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
    ) -> list[dict]:
        """Find suitable strikes for covered calls."""
        strikes = []

        # Generate strikes from 2% to 15% OTM
        for otm_pct in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
            strike = round(stock_price * (1 + otm_pct), 0)
            price, delta = self.estimate_option_price(
                stock_price, strike, days_to_expiry, volatility, "call"
            )

            if self.config.min_delta_call <= delta <= self.config.max_delta_call:
                premium_pct = price / stock_price * 100
                annualized = premium_pct * (365 / days_to_expiry)

                strikes.append({
                    "strike": strike,
                    "premium": price,
                    "delta": delta,
                    "premium_pct": premium_pct,
                    "annualized_return": annualized,
                    "otm_pct": otm_pct * 100,
                    "prob_otm": 1 - delta,  # Approximate probability of staying OTM
                })

        return sorted(strikes, key=lambda x: x["annualized_return"], reverse=True)

    def analyze_covered_call(
        self,
        symbol: str,
        stock_price: float,
        shares: int,
        volatility: float,
        days_to_expiry: int = 30,
    ) -> Optional[IncomeOpportunity]:
        """Analyze covered call opportunity for a position."""

        if self.config.require_100_shares and shares < 100:
            return None

        # Number of contracts we can sell
        contracts = shares // 100

        # Find best strike
        strikes = self.find_covered_call_strikes(stock_price, volatility, days_to_expiry)

        if not strikes:
            return None

        # Pick strike with best risk/reward
        best = None
        for s in strikes:
            if s["annualized_return"] >= self.config.min_annualized_return:
                if s["premium_pct"] >= self.config.min_premium_pct:
                    if best is None or s["prob_otm"] > best["prob_otm"]:
                        best = s

        if not best:
            # Fall back to highest return
            best = strikes[0]

        strike = best["strike"]
        premium = best["premium"]

        total_premium = premium * contracts * 100
        max_profit = total_premium + (strike - stock_price) * contracts * 100
        max_loss = stock_price * contracts * 100 - total_premium  # If stock goes to 0
        breakeven = stock_price - premium

        # Recommendation logic
        if best["annualized_return"] >= 20 and best["prob_otm"] >= 0.70:
            recommendation = "strong"
            reason = f"High return ({best['annualized_return']:.1f}%) with good probability ({best['prob_otm']*100:.0f}%)"
        elif best["annualized_return"] >= 12 and best["prob_otm"] >= 0.60:
            recommendation = "moderate"
            reason = f"Decent return ({best['annualized_return']:.1f}%) with acceptable probability"
        elif best["annualized_return"] >= 8:
            recommendation = "weak"
            reason = f"Low return ({best['annualized_return']:.1f}%), consider waiting for better IV"
        else:
            recommendation = "avoid"
            reason = "Premium too low to justify risk"

        expiry = datetime.now(timezone.utc) + timedelta(days=days_to_expiry)

        return IncomeOpportunity(
            underlying=symbol,
            strategy=OptionStrategy.COVERED_CALL,
            legs=[
                OptionLeg(
                    symbol=f"{symbol}{expiry.strftime('%y%m%d')}C{int(strike)}",
                    expiry=expiry,
                    strike=strike,
                    option_type="call",
                    action="sell",
                    qty=contracts,
                    premium=premium,
                    delta=best["delta"],
                )
            ],
            total_premium=total_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            probability_profit=best["prob_otm"],
            annualized_return=best["annualized_return"],
            days_to_expiry=days_to_expiry,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            recommendation=recommendation,
            reason=reason,
        )

    def analyze_cash_secured_put(
        self,
        symbol: str,
        stock_price: float,
        cash_available: float,
        volatility: float,
        target_entry_discount: float = 0.05,  # Want to buy 5% lower
        days_to_expiry: int = 30,
    ) -> Optional[IncomeOpportunity]:
        """Analyze cash-secured put opportunity."""

        strike = round(stock_price * (1 - target_entry_discount), 0)
        cash_needed = strike * 100

        if cash_available < cash_needed:
            return None

        contracts = int(cash_available // cash_needed)
        if contracts < 1:
            return None

        price, delta = self.estimate_option_price(
            stock_price, strike, days_to_expiry, volatility, "put"
        )

        premium_pct = price / strike * 100
        annualized = premium_pct * (365 / days_to_expiry)

        total_premium = price * contracts * 100
        max_profit = total_premium
        max_loss = strike * contracts * 100 - total_premium  # If stock goes to 0
        breakeven = strike - price
        prob_profit = 1 + delta  # Delta is negative for puts

        if annualized >= 15 and prob_profit >= 0.70:
            recommendation = "strong"
            reason = f"Good entry point with {annualized:.1f}% annualized premium"
        elif annualized >= 10:
            recommendation = "moderate"
            reason = f"Acceptable premium, {target_entry_discount*100:.0f}% discount if assigned"
        else:
            recommendation = "weak"
            reason = "Consider lower strike or higher IV environment"

        expiry = datetime.now(timezone.utc) + timedelta(days=days_to_expiry)

        return IncomeOpportunity(
            underlying=symbol,
            strategy=OptionStrategy.CASH_SECURED_PUT,
            legs=[
                OptionLeg(
                    symbol=f"{symbol}{expiry.strftime('%y%m%d')}P{int(strike)}",
                    expiry=expiry,
                    strike=strike,
                    option_type="put",
                    action="sell",
                    qty=contracts,
                    premium=price,
                    delta=delta,
                )
            ],
            total_premium=total_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            probability_profit=prob_profit,
            annualized_return=annualized,
            days_to_expiry=days_to_expiry,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            recommendation=recommendation,
            reason=reason,
        )

    def scan_portfolio_opportunities(
        self,
        positions: list[dict],  # [{symbol, qty, current_price, avg_cost}]
        cash: float,
        default_iv: float = 0.30,
    ) -> list[IncomeOpportunity]:
        """Scan entire portfolio for income opportunities."""
        opportunities = []

        for pos in positions:
            symbol = pos["symbol"]
            qty = pos.get("qty", 0)
            price = pos.get("current_price", 0)
            iv = pos.get("iv", default_iv)

            if qty >= 100:
                # Covered call opportunity
                opp = self.analyze_covered_call(
                    symbol=symbol,
                    stock_price=price,
                    shares=qty,
                    volatility=iv,
                )
                if opp and opp.recommendation != "avoid":
                    opportunities.append(opp)

        # Cash-secured puts on watchlist (could be configured)
        # For now, skip

        return sorted(opportunities, key=lambda x: x.annualized_return, reverse=True)


def format_opportunity(opp: IncomeOpportunity) -> str:
    """Format opportunity as readable text."""
    emoji = {"strong": "💰", "moderate": "👍", "weak": "⚠️", "avoid": "❌"}

    lines = [
        f"{emoji.get(opp.recommendation, '?')} {opp.underlying} - {opp.strategy.value.upper()}",
        f"   Strike: ${opp.legs[0].strike:.0f} ({opp.days_to_expiry} DTE)",
        f"   Premium: ${opp.total_premium:.2f} ({opp.annualized_return:.1f}% annualized)",
        f"   Prob Profit: {opp.probability_profit*100:.0f}%",
        f"   Max Profit: ${opp.max_profit:.2f}",
        f"   Breakeven: ${opp.breakeven:.2f}",
        f"   {opp.reason}",
    ]
    return "\n".join(lines)
