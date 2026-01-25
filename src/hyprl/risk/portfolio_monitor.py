"""Portfolio Risk Monitor.

Real-time portfolio monitoring with risk metrics.
Works for any portfolio of stocks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
from pathlib import Path


@dataclass
class PositionMetrics:
    """Metrics for a single position."""
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    hold_time_hours: float
    is_long: bool


@dataclass
class PortfolioMetrics:
    """Aggregate portfolio metrics."""
    total_equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl_today: float
    daily_pnl: float
    daily_pnl_pct: float
    num_positions: int
    num_long: int
    num_short: int
    max_position_weight: float
    concentration_score: float  # Herfindahl index
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskMetrics:
    """Risk-specific metrics."""
    current_drawdown_pct: float
    max_drawdown_pct: float
    var_95_pct: float  # Value at Risk (estimated)
    beta_to_spy: float
    correlation_avg: float
    exposure_long: float
    exposure_short: float
    net_exposure: float
    gross_exposure: float


@dataclass
class PortfolioConfig:
    """Portfolio monitoring configuration."""
    # Risk limits
    max_position_weight: float = 0.40  # Max 40% in one position
    max_drawdown_alert: float = 0.05   # Alert at 5% drawdown
    max_correlation: float = 0.80      # Alert if positions too correlated

    # Position limits
    max_positions: int = 10
    max_hold_hours: int = 168  # 1 week

    # File paths (relative to base_dir)
    daily_report_file: str = "live/execution/alpaca/daily_report.json"
    orders_file: str = "live/execution/alpaca/orders.jsonl"
    state_file: str = "live/execution/alpaca/state.json"


class PortfolioMonitor:
    """Monitors portfolio risk in real-time."""

    def __init__(self, config: Optional[PortfolioConfig] = None, base_dir: str = "."):
        self.config = config or PortfolioConfig()
        self.base_dir = Path(base_dir)
        self._position_entry_times: dict[str, datetime] = {}

    def load_daily_report(self) -> Optional[dict]:
        """Load latest daily report."""
        path = self.base_dir / self.config.daily_report_file
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def get_position_metrics(self, report: dict) -> list[PositionMetrics]:
        """Extract position metrics from report."""
        positions = report.get("positions", [])
        total_value = report.get("equity_now", 100000)

        metrics = []
        for pos in positions:
            symbol = pos.get("symbol", "?")
            qty = float(pos.get("qty", 0) or 0)
            current_price = float(pos.get("current_price", 0) or 0)
            market_value = float(pos.get("market_value", 0) or 0)
            unrealized_pnl = float(pos.get("unrealized_pnl", 0) or 0)
            cost_basis = float(pos.get("cost_basis", 0) or 0)

            entry_price = cost_basis / qty if qty != 0 else current_price
            unrealized_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0
            weight = (abs(market_value) / total_value * 100) if total_value else 0

            # Estimate hold time (simplified)
            hold_hours = 24  # Default, would need order history for accurate

            metrics.append(PositionMetrics(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pct,
                weight_pct=weight,
                hold_time_hours=hold_hours,
                is_long=qty > 0,
            ))

        return metrics

    def calculate_portfolio_metrics(self, report: dict) -> PortfolioMetrics:
        """Calculate aggregate portfolio metrics."""
        account = report.get("account", {})
        positions = report.get("positions", [])

        equity = float(report.get("equity_now", 0) or 0)
        cash = float(account.get("cash", 0) or 0)
        positions_value = float(report.get("exposure_notional", 0) or 0)
        unrealized = float(report.get("unrealized_pnl", 0) or 0)
        realized = float(report.get("realized_pnl", 0) or 0)

        day_start = float(report.get("day_start_equity", equity) or equity)
        daily_pnl = equity - day_start if day_start else 0
        daily_pnl_pct = (daily_pnl / day_start * 100) if day_start else 0

        # Count positions
        num_long = sum(1 for p in positions if float(p.get("qty", 0) or 0) > 0)
        num_short = sum(1 for p in positions if float(p.get("qty", 0) or 0) < 0)

        # Calculate concentration (Herfindahl index)
        weights = []
        for pos in positions:
            mv = abs(float(pos.get("market_value", 0) or 0))
            if equity > 0:
                weights.append(mv / equity)

        concentration = sum(w**2 for w in weights) if weights else 0
        max_weight = max(weights) * 100 if weights else 0

        return PortfolioMetrics(
            total_equity=equity,
            cash=cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized,
            unrealized_pnl_pct=(unrealized / equity * 100) if equity else 0,
            realized_pnl_today=realized,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            num_positions=len(positions),
            num_long=num_long,
            num_short=num_short,
            max_position_weight=max_weight,
            concentration_score=concentration,
        )

    def calculate_risk_metrics(self, report: dict, positions: list[PositionMetrics]) -> RiskMetrics:
        """Calculate risk-specific metrics."""
        equity = float(report.get("equity_now", 100000) or 100000)
        drawdown = float(report.get("daily_drawdown_pct", 0) or 0)

        # Exposure calculations
        long_exposure = sum(p.market_value for p in positions if p.is_long)
        short_exposure = abs(sum(p.market_value for p in positions if not p.is_long))
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Simplified VaR estimate (assumes 2% daily vol)
        var_95 = gross_exposure * 0.02 * 1.65  # 95% confidence
        var_95_pct = (var_95 / equity * 100) if equity else 0

        return RiskMetrics(
            current_drawdown_pct=drawdown,
            max_drawdown_pct=drawdown,  # Would need history for true max
            var_95_pct=var_95_pct,
            beta_to_spy=1.0,  # Would need calculation
            correlation_avg=0.5,  # Would need calculation
            exposure_long=long_exposure,
            exposure_short=short_exposure,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
        )

    def check_risk_alerts(
        self,
        portfolio: PortfolioMetrics,
        risk: RiskMetrics,
        positions: list[PositionMetrics]
    ) -> list[dict]:
        """Check for risk limit breaches."""
        alerts = []

        # Drawdown alert
        if abs(risk.current_drawdown_pct) > self.config.max_drawdown_alert * 100:
            alerts.append({
                "type": "drawdown",
                "severity": "critical",
                "message": f"Drawdown {risk.current_drawdown_pct:.2f}% exceeds limit",
            })

        # Concentration alert
        if portfolio.max_position_weight > self.config.max_position_weight * 100:
            alerts.append({
                "type": "concentration",
                "severity": "warning",
                "message": f"Max position weight {portfolio.max_position_weight:.1f}% exceeds limit",
            })

        # Position count alert
        if portfolio.num_positions > self.config.max_positions:
            alerts.append({
                "type": "position_count",
                "severity": "warning",
                "message": f"{portfolio.num_positions} positions exceed max {self.config.max_positions}",
            })

        # Hold time alert
        for pos in positions:
            if pos.hold_time_hours > self.config.max_hold_hours:
                alerts.append({
                    "type": "hold_time",
                    "severity": "warning",
                    "message": f"{pos.symbol} held for {pos.hold_time_hours:.0f}h (max: {self.config.max_hold_hours}h)",
                })

        return alerts

    def get_full_report(self) -> Optional[dict]:
        """Get complete portfolio report."""
        report = self.load_daily_report()
        if not report:
            return None

        positions = self.get_position_metrics(report)
        portfolio = self.calculate_portfolio_metrics(report)
        risk = self.calculate_risk_metrics(report, positions)
        alerts = self.check_risk_alerts(portfolio, risk, positions)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": {
                "equity": portfolio.total_equity,
                "cash": portfolio.cash,
                "positions_value": portfolio.positions_value,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "daily_pnl": portfolio.daily_pnl,
                "daily_pnl_pct": portfolio.daily_pnl_pct,
                "num_positions": portfolio.num_positions,
                "concentration": portfolio.concentration_score,
            },
            "risk": {
                "drawdown_pct": risk.current_drawdown_pct,
                "var_95_pct": risk.var_95_pct,
                "net_exposure": risk.net_exposure,
                "gross_exposure": risk.gross_exposure,
            },
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "weight_pct": p.weight_pct,
                }
                for p in positions
            ],
            "alerts": alerts,
        }


def format_portfolio_summary(report: dict) -> str:
    """Format portfolio report as readable text."""
    if not report:
        return "No portfolio data available"

    lines = [
        "=" * 50,
        f"Portfolio Summary - {report['timestamp'][:19]}",
        "=" * 50,
        "",
        "PORTFOLIO",
        f"  Equity:        ${report['portfolio']['equity']:,.2f}",
        f"  Cash:          ${report['portfolio']['cash']:,.2f}",
        f"  Daily P&L:     ${report['portfolio']['daily_pnl']:,.2f} ({report['portfolio']['daily_pnl_pct']:.2f}%)",
        f"  Unrealized:    ${report['portfolio']['unrealized_pnl']:,.2f}",
        f"  Positions:     {report['portfolio']['num_positions']}",
        "",
        "RISK",
        f"  Drawdown:      {report['risk']['drawdown_pct']:.2f}%",
        f"  VaR (95%):     {report['risk']['var_95_pct']:.2f}%",
        f"  Net Exposure:  ${report['risk']['net_exposure']:,.2f}",
        "",
    ]

    if report["positions"]:
        lines.append("POSITIONS")
        for pos in report["positions"]:
            pnl_sign = "+" if pos["unrealized_pnl"] >= 0 else ""
            lines.append(
                f"  {pos['symbol']:6} {pos['qty']:>6.0f} @ ${pos['current_price']:.2f}  "
                f"{pnl_sign}${pos['unrealized_pnl']:.2f} ({pos['weight_pct']:.1f}%)"
            )
        lines.append("")

    if report["alerts"]:
        lines.append("ALERTS")
        for alert in report["alerts"]:
            icon = "🔴" if alert["severity"] == "critical" else "🟠"
            lines.append(f"  {icon} {alert['message']}")
        lines.append("")

    return "\n".join(lines)
