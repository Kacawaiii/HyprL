"""
HyprL Monitoring Module
=======================
Complete monitoring stack:
- Telegram notifications (trades, alerts, summaries)
- Health monitoring (heartbeat, crash detection)
- Risk monitoring (drawdown, daily loss, consecutive losses)
- Trade journaling (logging, analysis, export)
- Market regime detection (trending/ranging/volatile)
- Drift detection and data quality checks
"""

# Existing modules
from .drift_detector import DriftDetector, DriftAlert
from .data_quality import DataQualityMonitor, DataQualityReport

# Telegram notifications
from .telegram import (
    TelegramNotifier,
    get_notifier,
    send_trade_entry,
    send_trade_exit,
    send_risk_alert,
    send_daily_summary,
    send_heartbeat,
)

# Health monitoring
from .health import (
    HealthChecker,
    RiskMonitor,
    get_health_checker,
    get_risk_monitor,
)

# Trade journal
from .journal import (
    TradeJournal,
    TradeRecord,
    get_journal,
)

# Market regime
from .regime import (
    MarketRegime,
    RegimeState,
    RegimeDetector,
    get_regime_detector,
    detect_regime,
)

# Reporter
from .reporter import (
    HyprLReporter,
    SymbolStats,
    PortfolioStats,
    get_reporter,
)

# Discord notifications
from .discord_notifier import (
    HyprLDiscordNotifier,
    PaperAccount,
    DiscordConfig,
    get_discord_notifier,
)

__all__ = [
    # Existing
    "DriftDetector",
    "DriftAlert",
    "DataQualityMonitor",
    "DataQualityReport",
    # Telegram
    "TelegramNotifier",
    "get_notifier",
    "send_trade_entry",
    "send_trade_exit",
    "send_risk_alert",
    "send_daily_summary",
    "send_heartbeat",
    # Health
    "HealthChecker",
    "RiskMonitor",
    "get_health_checker",
    "get_risk_monitor",
    # Journal
    "TradeJournal",
    "TradeRecord",
    "get_journal",
    # Regime
    "MarketRegime",
    "RegimeState",
    "RegimeDetector",
    "get_regime_detector",
    "detect_regime",
    # Reporter
    "HyprLReporter",
    "SymbolStats",
    "PortfolioStats",
    "get_reporter",
    # Discord
    "HyprLDiscordNotifier",
    "PaperAccount",
    "DiscordConfig",
    "get_discord_notifier",
]
