"""HyprL Broker abstraction layer.

Provides unified interface for order execution across different brokers.

Supported brokers:
    - PaperBroker: Local simulation (no real orders)
    - AlpacaBroker: Alpaca Trading API (paper + live)
    - (Future) IBKRBroker: Interactive Brokers

Usage:
    from hyprl.broker import AlpacaBroker, OrderSide

    broker = AlpacaBroker(paper=True)
    order = broker.submit_order("NVDA", 10, OrderSide.BUY)
    broker.wait_for_fill(order.id)
"""

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

__all__ = [
    # Base classes
    "BrokerBase",
    "Account",
    "Position",
    "Order",
    "Fill",
    "Clock",
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # Exceptions
    "BrokerError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "RateLimitError",
]


# Lazy imports for optional dependencies
def get_alpaca_broker():
    """Get AlpacaBroker class (requires alpaca-py)."""
    from .alpaca import AlpacaBroker

    return AlpacaBroker
