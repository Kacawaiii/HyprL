"""Crypto Trading Module.

24/7 cryptocurrency trading using Alpaca Crypto API.
Supports BTC, ETH, and other major cryptocurrencies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import os
import json
from pathlib import Path

from .policy import PolicyConfig


class CryptoSymbol(Enum):
    """Supported cryptocurrency symbols."""
    BTC_USD = "BTC/USD"
    ETH_USD = "ETH/USD"
    LTC_USD = "LTC/USD"
    LINK_USD = "LINK/USD"
    UNI_USD = "UNI/USD"
    AAVE_USD = "AAVE/USD"
    AVAX_USD = "AVAX/USD"
    SOL_USD = "SOL/USD"


@dataclass
class CryptoConfig:
    """Configuration for crypto trading."""
    # Symbols to trade
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])

    # Timeframe settings
    timeframe: str = "1Hour"  # 1Min, 5Min, 15Min, 1Hour, 1Day
    lookback_days: int = 90

    # Model thresholds
    threshold_long: float = 0.58
    threshold_short: float = 0.42
    min_confidence: float = 0.55
    signal_mode: str = "ml"  # ml, policy, rule
    policy_symbols: list[str] = field(default_factory=list)
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)

    # Position sizing
    max_position_pct: float = 0.10  # Max 10% per crypto
    max_crypto_allocation: float = 0.20  # Max 20% of portfolio in crypto

    # Risk management
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    trailing_stop_pct: float = 0.02  # 2% trailing stop

    # Trading hours (crypto is 24/7 but we can limit)
    trade_weekends: bool = True
    trade_overnight: bool = True

    # Data paths
    models_dir: str = "models/crypto"
    signals_file: str = "live/logs/crypto_signals.jsonl"

    # API settings
    use_paper: bool = True


@dataclass
class CryptoPosition:
    """Current crypto position."""
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # "long" or "short"
    entry_time: datetime


@dataclass
class CryptoSignal:
    """Trading signal for crypto."""
    symbol: str
    direction: str  # "long", "short", "neutral"
    probability: float
    confidence: float
    size_pct: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CryptoTrader:
    """Handles crypto trading operations."""

    def __init__(self, config: Optional[CryptoConfig] = None, base_dir: str = "."):
        self.config = config or CryptoConfig()
        self.base_dir = Path(base_dir)
        self._client = None
        self._trading_client = None

    def _get_data_client(self):
        """Get Alpaca crypto data client."""
        if self._client is None:
            try:
                from alpaca.data.historical import CryptoHistoricalDataClient

                api_key = os.environ.get("APCA_API_KEY_ID")
                secret_key = os.environ.get("APCA_API_SECRET_KEY")

                if api_key and secret_key:
                    self._client = CryptoHistoricalDataClient(
                        api_key=api_key,
                        secret_key=secret_key
                    )
            except ImportError:
                pass
        return self._client

    def _get_trading_client(self):
        """Get Alpaca trading client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient

                api_key = os.environ.get("APCA_API_KEY_ID")
                secret_key = os.environ.get("APCA_API_SECRET_KEY")

                if api_key and secret_key:
                    self._trading_client = TradingClient(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper=self.config.use_paper
                    )
            except ImportError:
                pass
        return self._trading_client

    def fetch_bars(
        self,
        symbol: str,
        days: Optional[int] = None,
        timeframe: Optional[str] = None
    ) -> Optional[list[dict]]:
        """Fetch historical crypto bars."""
        client = self._get_data_client()
        if not client:
            return None

        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            days = days or self.config.lookback_days
            tf_str = timeframe or self.config.timeframe

            # Parse timeframe
            tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
            }
            tf = tf_map.get(tf_str, TimeFrame(1, TimeFrameUnit.Hour))

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )

            bars = client.get_crypto_bars(request)

            # Handle both BarSet (with .data) and dict-like access
            bar_data = bars.data if hasattr(bars, 'data') else bars
            if symbol in bar_data:
                return [
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "vwap": float(bar.vwap) if bar.vwap else None,
                    }
                    for bar in bar_data[symbol]
                ]
        except Exception as e:
            print(f"Error fetching crypto bars: {e}")

        return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest crypto price."""
        client = self._get_data_client()
        if not client:
            return None

        try:
            from alpaca.data.requests import CryptoLatestQuoteRequest

            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = client.get_crypto_latest_quote(request)

            if symbol in quotes:
                return float(quotes[symbol].ask_price)
        except Exception as e:
            print(f"Error getting latest price: {e}")

        return None

    def get_positions(self) -> list[CryptoPosition]:
        """Get current crypto positions."""
        client = self._get_trading_client()
        if not client:
            return []

        try:
            positions = client.get_all_positions()
            crypto_positions = []

            for pos in positions:
                # Filter for crypto symbols
                if "/" in pos.symbol or pos.symbol.endswith("USD"):
                    entry_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    qty = float(pos.qty)

                    crypto_positions.append(CryptoPosition(
                        symbol=pos.symbol,
                        qty=qty,
                        entry_price=entry_price,
                        current_price=current_price,
                        unrealized_pnl=float(pos.unrealized_pl),
                        unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                        side="long" if qty > 0 else "short",
                        entry_time=datetime.now(timezone.utc),  # Would need order history
                    ))

            return crypto_positions
        except Exception as e:
            print(f"Error getting positions: {e}")

        return []

    def get_account_equity(self) -> float:
        """Get account equity."""
        client = self._get_trading_client()
        if not client:
            return 0.0

        try:
            account = client.get_account()
            return float(account.equity)
        except Exception:
            return 0.0

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[str]:
        """Place a crypto order."""
        client = self._get_trading_client()
        if not client:
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            if order_type == "limit" and limit_price:
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                )

            order = client.submit_order(request)

            # TODO: Add bracket orders for stop/take profit

            return order.id
        except Exception as e:
            print(f"Error placing order: {e}")

        return None

    def close_position(self, symbol: str) -> bool:
        """Close a crypto position."""
        client = self._get_trading_client()
        if not client:
            return False

        try:
            client.close_position(symbol)
            return True
        except Exception as e:
            print(f"Error closing position: {e}")

        return False

    def log_signal(self, signal: CryptoSignal):
        """Log a crypto signal."""
        signals_path = self.base_dir / self.config.signals_file
        signals_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction,
            "probability": signal.probability,
            "confidence": signal.confidence,
            "size_pct": signal.size_pct,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reason": signal.reason,
        }

        with open(signals_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def format_crypto_signal(signal: CryptoSignal) -> str:
    """Format crypto signal as readable text."""
    emoji = {"long": "🟢", "short": "🔴", "neutral": "⚪"}

    lines = [
        f"{emoji.get(signal.direction, '?')} {signal.symbol} - {signal.direction.upper()}",
        f"   Probability: {signal.probability:.1%} (conf: {signal.confidence:.1%})",
        f"   Entry: ${signal.entry_price:,.2f}",
        f"   Stop Loss: ${signal.stop_loss:,.2f}",
        f"   Take Profit: ${signal.take_profit:,.2f}",
        f"   Size: {signal.size_pct:.1%} of portfolio",
        f"   Reason: {signal.reason}",
    ]
    return "\n".join(lines)
