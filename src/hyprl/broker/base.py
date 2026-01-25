"""Broker abstraction layer — Contract definition.

This module defines the interface that all broker implementations must follow.
Designed for production use with proper lifecycle management, idempotency,
and emergency controls.

Usage:
    from hyprl.broker.base import BrokerBase
    from hyprl.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    account = broker.get_account()
    order = broker.submit_order("NVDA", 10, "buy")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good 'til canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class Account:
    """Broker account information."""

    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
    account_id: str = ""
    status: str = "active"
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Open position information."""

    symbol: str
    qty: float
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    asset_class: str = "us_equity"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Order information."""

    id: str
    client_order_id: str
    symbol: str
    qty: float
    filled_qty: float
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_avg_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    asset_class: str = "us_equity"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in (
            OrderStatus.NEW,
            OrderStatus.PENDING,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass
class Fill:
    """Trade fill/execution information."""

    id: str
    order_id: str
    symbol: str
    qty: float
    price: float
    side: OrderSide
    timestamp: datetime
    commission: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Clock:
    """Market clock information."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class BrokerError(Exception):
    """Base exception for broker errors."""


class OrderRejectedError(BrokerError):
    """Order was rejected by the broker."""


class InsufficientFundsError(BrokerError):
    """Insufficient buying power for order."""


class RateLimitError(BrokerError):
    """Rate limit exceeded."""


class BrokerBase(ABC):
    """Abstract base class for broker implementations.

    All broker implementations must implement these methods.
    The contract ensures consistent behavior across Paper, Alpaca, IBKR, etc.
    """

    # =========================================================================
    # Account & Portfolio
    # =========================================================================

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account information.

        Returns:
            Account with equity, cash, buying_power, status.

        Raises:
            BrokerError: If account fetch fails.
        """

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Ticker symbol (e.g., "NVDA").

        Returns:
            Position if exists, None otherwise.
        """

    @abstractmethod
    def list_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of Position objects.
        """

    # =========================================================================
    # Order Management
    # =========================================================================

    @abstractmethod
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
        """Submit a new order.

        Args:
            symbol: Ticker symbol.
            qty: Order quantity (supports fractional for some brokers).
            side: BUY or SELL.
            order_type: MARKET, LIMIT, STOP, STOP_LIMIT.
            time_in_force: DAY, GTC, IOC, FOK.
            limit_price: Required for LIMIT and STOP_LIMIT orders.
            stop_price: Required for STOP and STOP_LIMIT orders.
            client_order_id: Optional idempotency key (auto-generated if None).

        Returns:
            Order object with status.

        Raises:
            OrderRejectedError: If order is rejected.
            InsufficientFundsError: If not enough buying power.
            BrokerError: For other errors.
        """

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Broker order ID.

        Returns:
            Order if found, None otherwise.
        """

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID (idempotency key).

        Args:
            client_order_id: Client-provided order ID.

        Returns:
            Order if found, None otherwise.
        """

    @abstractmethod
    def list_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Order]:
        """List orders with optional filters.

        Args:
            status: Filter by status ("open", "closed", "all").
            limit: Maximum number of orders to return.
            after: Only orders after this timestamp.

        Returns:
            List of Order objects.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID.

        Args:
            order_id: Broker order ID.

        Returns:
            True if cancellation request accepted, False otherwise.
        """

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders canceled.
        """

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol.

        CRITICAL: Must be called before close_position() to avoid
        'insufficient qty available' errors when shares are held_for_orders.

        Args:
            symbol: Ticker symbol.

        Returns:
            Number of orders canceled.
        """
        # Default implementation: subclasses should override
        return 0

    # =========================================================================
    # Position Management
    # =========================================================================

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Closing order if position existed, None otherwise.
        """

    @abstractmethod
    def close_all_positions(self) -> List[Order]:
        """Emergency close all positions.

        Returns:
            List of closing orders.
        """

    # =========================================================================
    # Market Data & Clock
    # =========================================================================

    @abstractmethod
    def get_clock(self) -> Clock:
        """Get current market clock.

        Returns:
            Clock with market open/close times.
        """

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if market is open for trading.
        """

    # =========================================================================
    # Utility Methods (with default implementations)
    # =========================================================================

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Order:
        """Wait for order to reach terminal state.

        Args:
            order_id: Order ID to watch.
            timeout_seconds: Maximum wait time.
            poll_interval: Time between status checks.

        Returns:
            Order in terminal state.

        Raises:
            TimeoutError: If order doesn't fill within timeout.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_seconds:
            order = self.get_order(order_id)
            if order and order.is_terminal:
                return order
            time.sleep(poll_interval)
        raise TimeoutError(f"Order {order_id} did not fill within {timeout_seconds}s")

    def calculate_order_qty(
        self,
        symbol: str,
        notional: float,
        price: float,
        fractional: bool = False,
    ) -> float:
        """Calculate order quantity from notional value.

        Args:
            symbol: Ticker symbol (for min qty rules).
            notional: Dollar amount to invest.
            price: Current price.
            fractional: Allow fractional shares.

        Returns:
            Quantity to order.
        """
        qty = notional / price
        if not fractional:
            qty = int(qty)
        return max(qty, 0)

    def generate_client_order_id(self, prefix: str = "hyprl") -> str:
        """Generate unique client order ID for idempotency.

        Args:
            prefix: ID prefix.

        Returns:
            Unique client order ID.
        """
        import uuid

        return f"{prefix}_{uuid.uuid4().hex[:16]}"
