"""Alpaca broker implementation.

Full lifecycle management for Alpaca Trading API with:
- Order submission/cancellation/tracking
- Position management with emergency close
- Idempotency via client_order_id
- Retry/backoff for transient errors
- Rate limit handling
- Market clock awareness

Requirements:
    pip install alpaca-py

Environment variables:
    ALPACA_API_KEY: API key ID
    ALPACA_SECRET_KEY: API secret key
    ALPACA_PAPER: "true" for paper trading (default), "false" for live

Usage:
    from hyprl.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    order = broker.submit_order("NVDA", 10, OrderSide.BUY)
    broker.wait_for_fill(order.id)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import (
    BrokerBase,
    Account,
    Position,
    Order,
    Fill,
    Clock,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds
RATE_LIMIT_RETRY_AFTER = 60  # seconds


def _with_retry(func):
    """Decorator for retry with exponential backoff."""

    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError as exc:
                logger.warning("Rate limit hit, waiting %ss", RATE_LIMIT_RETRY_AFTER)
                time.sleep(RATE_LIMIT_RETRY_AFTER)
                last_exception = exc
            except BrokerError as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Broker error (attempt %s), retrying in %ss: %s",
                        attempt + 1,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    last_exception = exc
                else:
                    raise
        if last_exception is not None:
            raise last_exception
        raise BrokerError("Unknown broker error")

    return wrapper


class AlpacaBroker(BrokerBase):
    """Alpaca Trading API broker implementation.

    Supports both paper and live trading with full order lifecycle management.

    Note: Paper trading uses IEX data only (not real-time).
    Use paper mode for technical validation, not performance testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (or ALPACA_API_KEY env var).
            secret_key: Alpaca secret key (or ALPACA_SECRET_KEY env var).
            paper: Use paper trading if True.
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
                GetOrdersRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaOrderSide,
                OrderType as AlpacaOrderType,
                TimeInForce as AlpacaTimeInForce,
                OrderStatus as AlpacaOrderStatus,
                QueryOrderStatus,
            )
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. "
                "Install with: pip install alpaca-py"
            ) from exc

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass to constructor."
            )

        # Check env override for paper mode
        env_paper = os.environ.get("ALPACA_PAPER", "").lower()
        if env_paper == "false":
            paper = False
        elif env_paper == "true":
            paper = True

        self.paper = paper
        self.client = TradingClient(
            self._api_key,
            self._secret_key,
            paper=paper,
        )

        # Store imports for later use
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._StopOrderRequest = StopOrderRequest
        self._StopLimitOrderRequest = StopLimitOrderRequest
        self._GetOrdersRequest = GetOrdersRequest
        self._AlpacaOrderSide = AlpacaOrderSide
        self._AlpacaOrderType = AlpacaOrderType
        self._AlpacaTimeInForce = AlpacaTimeInForce
        self._AlpacaOrderStatus = AlpacaOrderStatus
        self._QueryOrderStatus = QueryOrderStatus

        logger.info("AlpacaBroker initialized (paper=%s)", paper)

    # =========================================================================
    # Mapping helpers
    # =========================================================================

    def _map_order_side(self, side: OrderSide) -> "AlpacaOrderSide":
        return (
            self._AlpacaOrderSide.BUY if side == OrderSide.BUY else self._AlpacaOrderSide.SELL
        )

    def _map_order_side_back(self, side: "AlpacaOrderSide") -> OrderSide:
        return OrderSide.BUY if side == self._AlpacaOrderSide.BUY else OrderSide.SELL

    def _map_time_in_force(self, tif: TimeInForce) -> "AlpacaTimeInForce":
        mapping = {
            TimeInForce.DAY: self._AlpacaTimeInForce.DAY,
            TimeInForce.GTC: self._AlpacaTimeInForce.GTC,
            TimeInForce.IOC: self._AlpacaTimeInForce.IOC,
            TimeInForce.FOK: self._AlpacaTimeInForce.FOK,
        }
        return mapping.get(tif, self._AlpacaTimeInForce.DAY)

    def _map_order_type_back(self, ot: "AlpacaOrderType") -> OrderType:
        mapping = {
            self._AlpacaOrderType.MARKET: OrderType.MARKET,
            self._AlpacaOrderType.LIMIT: OrderType.LIMIT,
            self._AlpacaOrderType.STOP: OrderType.STOP,
            self._AlpacaOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
        }
        return mapping.get(ot, OrderType.MARKET)

    def _map_order_status_back(self, status: "AlpacaOrderStatus") -> OrderStatus:
        mapping = {
            self._AlpacaOrderStatus.NEW: OrderStatus.NEW,
            self._AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
            self._AlpacaOrderStatus.ACCEPTED: OrderStatus.ACCEPTED,
            self._AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            self._AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            self._AlpacaOrderStatus.CANCELED: OrderStatus.CANCELED,
            self._AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
            self._AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.NEW)

    def _map_time_in_force_back(self, tif: "AlpacaTimeInForce") -> TimeInForce:
        mapping = {
            self._AlpacaTimeInForce.DAY: TimeInForce.DAY,
            self._AlpacaTimeInForce.GTC: TimeInForce.GTC,
            self._AlpacaTimeInForce.IOC: TimeInForce.IOC,
            self._AlpacaTimeInForce.FOK: TimeInForce.FOK,
        }
        return mapping.get(tif, TimeInForce.DAY)

    def _parse_datetime(self, dt) -> Optional[datetime]:
        """Parse Alpaca datetime to Python datetime."""
        if dt is None:
            return None
        if isinstance(dt, datetime):
            return dt
        try:
            return datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _alpaca_order_to_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to our Order dataclass."""
        return Order(
            id=str(alpaca_order.id),
            client_order_id=alpaca_order.client_order_id or "",
            symbol=alpaca_order.symbol,
            qty=float(alpaca_order.qty or 0),
            filled_qty=float(alpaca_order.filled_qty or 0),
            side=self._map_order_side_back(alpaca_order.side),
            order_type=self._map_order_type_back(alpaca_order.order_type),
            status=self._map_order_status_back(alpaca_order.status),
            time_in_force=self._map_time_in_force_back(alpaca_order.time_in_force),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            created_at=self._parse_datetime(alpaca_order.created_at),
            updated_at=self._parse_datetime(alpaca_order.updated_at),
            filled_at=self._parse_datetime(alpaca_order.filled_at),
            canceled_at=self._parse_datetime(alpaca_order.canceled_at),
            failed_at=self._parse_datetime(alpaca_order.failed_at),
            asset_class=str(alpaca_order.asset_class) if alpaca_order.asset_class else "us_equity",
            raw={"alpaca_order_id": str(alpaca_order.id)},
        )

    def _alpaca_position_to_position(self, alpaca_pos) -> Position:
        """Convert Alpaca position to our Position dataclass."""
        qty = float(alpaca_pos.qty)
        return Position(
            symbol=alpaca_pos.symbol,
            qty=abs(qty),
            side="long" if qty > 0 else "short",
            entry_price=float(alpaca_pos.avg_entry_price),
            current_price=float(alpaca_pos.current_price),
            market_value=float(alpaca_pos.market_value),
            cost_basis=float(alpaca_pos.cost_basis),
            unrealized_pnl=float(alpaca_pos.unrealized_pl),
            unrealized_pnl_pct=float(alpaca_pos.unrealized_plpc) * 100,
            asset_class=str(alpaca_pos.asset_class) if alpaca_pos.asset_class else "us_equity",
            raw={"alpaca_asset_id": str(alpaca_pos.asset_id)},
        )

    # =========================================================================
    # Account & Portfolio
    # =========================================================================

    @_with_retry
    def get_account(self) -> Account:
        """Get current account information."""
        try:
            acct = self.client.get_account()
            return Account(
                equity=float(acct.equity),
                cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                currency=acct.currency or "USD",
                account_id=str(acct.id),
                status=str(acct.status),
                pattern_day_trader=bool(acct.pattern_day_trader),
                trading_blocked=bool(acct.trading_blocked),
                transfers_blocked=bool(acct.transfers_blocked),
                raw={"account_number": acct.account_number},
            )
        except Exception as exc:
            logger.error("Failed to get account: %s", exc)
            raise BrokerError(f"Failed to get account: {exc}") from exc

    @_with_retry
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            pos = self.client.get_open_position(symbol.upper())
            return self._alpaca_position_to_position(pos)
        except Exception as exc:
            if "position does not exist" in str(exc).lower():
                return None
            logger.warning("Error getting position for %s: %s", symbol, exc)
            return None

    @_with_retry
    def list_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            positions = self.client.get_all_positions()
            return [self._alpaca_position_to_position(p) for p in positions]
        except Exception as exc:
            logger.error("Failed to list positions: %s", exc)
            raise BrokerError(f"Failed to list positions: {exc}") from exc

    # =========================================================================
    # Order Management
    # =========================================================================

    @_with_retry
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Submit a new order."""
        symbol = symbol.upper()
        client_order_id = client_order_id or self.generate_client_order_id()

        existing = self.get_order_by_client_id(client_order_id)
        if existing:
            logger.info("Order with client_order_id=%s already exists", client_order_id)
            return existing

        alpaca_side = self._map_order_side(side)
        alpaca_tif = self._map_time_in_force(time_in_force)

        try:
            if order_type == OrderType.MARKET:
                request = self._MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    client_order_id=client_order_id,
                )
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("limit_price required for LIMIT orders")
                request = self._LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                )
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("stop_price required for STOP orders")
                request = self._StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    client_order_id=client_order_id,
                )
            elif order_type == OrderType.STOP_LIMIT:
                if stop_price is None or limit_price is None:
                    raise ValueError("stop_price and limit_price required for STOP_LIMIT orders")
                request = self._StopLimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            result = self.client.submit_order(request)
            order = self._alpaca_order_to_order(result)
            logger.info(
                "Order submitted: %s %s %s %s @ %s",
                order.id,
                symbol,
                side.value,
                qty,
                order_type.value,
            )
            return order

        except Exception as exc:
            error_str = str(exc).lower()
            if "insufficient" in error_str or "buying power" in error_str:
                raise InsufficientFundsError(f"Insufficient funds for order: {exc}") from exc
            if "rejected" in error_str:
                raise OrderRejectedError(f"Order rejected: {exc}") from exc
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
            logger.error("Failed to submit order: %s", exc)
            raise BrokerError(f"Failed to submit order: {exc}") from exc

    @_with_retry
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        try:
            result = self.client.get_order_by_id(order_id)
            return self._alpaca_order_to_order(result)
        except Exception as exc:
            if "not found" in str(exc).lower():
                return None
            logger.warning("Error getting order %s: %s", order_id, exc)
            return None

    @_with_retry
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        try:
            result = self.client.get_order_by_client_id(client_order_id)
            return self._alpaca_order_to_order(result)
        except Exception as exc:
            if "not found" in str(exc).lower():
                return None
            logger.warning("Error getting order by client_id %s: %s", client_order_id, exc)
            return None

    @_with_retry
    def list_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Order]:
        """List orders with optional filters."""
        try:
            query_status = None
            if status == "open":
                query_status = self._QueryOrderStatus.OPEN
            elif status == "closed":
                query_status = self._QueryOrderStatus.CLOSED
            elif status == "all":
                query_status = self._QueryOrderStatus.ALL

            request = self._GetOrdersRequest(
                status=query_status,
                limit=limit,
                after=after,
            )
            results = self.client.get_orders(request)
            return [self._alpaca_order_to_order(o) for o in results]
        except Exception as exc:
            logger.error("Failed to list orders: %s", exc)
            raise BrokerError(f"Failed to list orders: {exc}") from exc

    @_with_retry
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info("Order canceled: %s", order_id)
            return True
        except Exception as exc:
            if "cannot be canceled" in str(exc).lower():
                logger.warning("Order %s cannot be canceled (already terminal)", order_id)
                return False
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    @_with_retry
    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        try:
            results = self.client.cancel_orders()
            count = len(results) if results else 0
            logger.info("Canceled %s orders", count)
            return count
        except Exception as exc:
            logger.error("Failed to cancel all orders: %s", exc)
            raise BrokerError(f"Failed to cancel all orders: {exc}") from exc

    # =========================================================================
    # Position Management
    # =========================================================================

    @_with_retry
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position for a symbol."""
        symbol = symbol.upper()
        try:
            result = self.client.close_position(symbol)
            order = self._alpaca_order_to_order(result)
            logger.info("Position closed: %s -> order %s", symbol, order.id)
            return order
        except Exception as exc:
            if "position does not exist" in str(exc).lower():
                logger.info("No position to close for %s", symbol)
                return None
            logger.error("Failed to close position %s: %s", symbol, exc)
            raise BrokerError(f"Failed to close position {symbol}: {exc}") from exc

    @_with_retry
    def close_all_positions(self) -> List[Order]:
        """Emergency close all positions."""
        try:
            results = self.client.close_all_positions(cancel_orders=True)
            orders = []
            for res in results or []:
                if hasattr(res, "id"):
                    orders.append(self._alpaca_order_to_order(res))
            logger.warning("EMERGENCY: Closed all positions (%s orders)", len(orders))
            return orders
        except Exception as exc:
            logger.error("Failed to close all positions: %s", exc)
            raise BrokerError(f"Failed to close all positions: {exc}") from exc

    # =========================================================================
    # Market Clock
    # =========================================================================

    @_with_retry
    def get_clock(self) -> Clock:
        """Get current market clock."""
        try:
            clock = self.client.get_clock()
            return Clock(
                timestamp=self._parse_datetime(clock.timestamp) or datetime.now(timezone.utc),
                is_open=bool(clock.is_open),
                next_open=self._parse_datetime(clock.next_open) or datetime.now(timezone.utc),
                next_close=self._parse_datetime(clock.next_close) or datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.error("Failed to get clock: %s", exc)
            raise BrokerError(f"Failed to get clock: {exc}") from exc

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.get_clock()
            return clock.is_open
        except Exception:
            return False

    # =========================================================================
    # Alpaca-specific methods
    # =========================================================================

    def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
    ) -> Dict[str, Any]:
        """Get portfolio equity history.

        Args:
            period: Time period ("1D", "1W", "1M", "3M", "1A").
            timeframe: Data resolution ("1Min", "5Min", "15Min", "1H", "1D").

        Returns:
            Dict with timestamps, equity, profit_loss arrays.
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe,
            )
            history = self.client.get_portfolio_history(request)
            return {
                "timestamps": history.timestamp,
                "equity": history.equity,
                "profit_loss": history.profit_loss,
                "profit_loss_pct": history.profit_loss_pct,
            }
        except Exception as exc:
            logger.error("Failed to get portfolio history: %s", exc)
            return {}
