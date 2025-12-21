import sys
import types
from datetime import datetime, timezone

from hyprl.broker.base import OrderSide, OrderStatus


def _install_fake_alpaca(monkeypatch) -> None:
    alpaca = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    client = types.ModuleType("alpaca.trading.client")
    requests = types.ModuleType("alpaca.trading.requests")
    enums = types.ModuleType("alpaca.trading.enums")

    from enum import Enum

    class AlpacaOrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

    class AlpacaOrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"

    class AlpacaTimeInForce(Enum):
        DAY = "day"
        GTC = "gtc"
        IOC = "ioc"
        FOK = "fok"

    class AlpacaOrderStatus(Enum):
        NEW = "new"
        PENDING_NEW = "pending_new"
        ACCEPTED = "accepted"
        PARTIALLY_FILLED = "partially_filled"
        FILLED = "filled"
        CANCELED = "canceled"
        REJECTED = "rejected"
        EXPIRED = "expired"

    class QueryOrderStatus(Enum):
        OPEN = "open"
        CLOSED = "closed"
        ALL = "all"

    class _BaseRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MarketOrderRequest(_BaseRequest):
        pass

    class LimitOrderRequest(_BaseRequest):
        pass

    class StopOrderRequest(_BaseRequest):
        pass

    class StopLimitOrderRequest(_BaseRequest):
        pass

    class GetOrdersRequest(_BaseRequest):
        pass

    class TradingClient:
        def __init__(self, *args, **kwargs):
            self._now = datetime.now(timezone.utc)

        def get_account(self):
            return types.SimpleNamespace(
                equity="10000",
                cash="5000",
                buying_power="10000",
                currency="USD",
                id="acct_1",
                status="ACTIVE",
                pattern_day_trader=False,
                trading_blocked=False,
                transfers_blocked=False,
                account_number="ABC123",
            )

        def get_clock(self):
            return types.SimpleNamespace(
                timestamp=self._now,
                is_open=True,
                next_open=self._now,
                next_close=self._now,
            )

        def get_all_positions(self):
            return []

        def submit_order(self, request):
            return types.SimpleNamespace(
                id="oid_1",
                client_order_id=getattr(request, "client_order_id", ""),
                symbol=getattr(request, "symbol", "NVDA"),
                qty=getattr(request, "qty", 1),
                filled_qty=getattr(request, "qty", 1),
                side=getattr(request, "side", AlpacaOrderSide.BUY),
                order_type=getattr(request, "order_type", AlpacaOrderType.MARKET),
                status=AlpacaOrderStatus.ACCEPTED,
                time_in_force=getattr(request, "time_in_force", AlpacaTimeInForce.DAY),
                limit_price=getattr(request, "limit_price", None),
                stop_price=getattr(request, "stop_price", None),
                filled_avg_price=None,
                created_at=self._now,
                updated_at=self._now,
                filled_at=None,
                canceled_at=None,
                failed_at=None,
                asset_class="us_equity",
            )

        def cancel_order_by_id(self, order_id):
            return True

        def get_orders(self, request):
            return []

        def get_order_by_id(self, order_id):
            return None

        def get_order_by_client_id(self, client_order_id):
            return None

        def close_position(self, symbol):
            return self.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=1,
                    side=AlpacaOrderSide.SELL,
                    time_in_force=AlpacaTimeInForce.DAY,
                    client_order_id="close_1",
                )
            )

        def close_all_positions(self, cancel_orders=True):
            return []

        def get_portfolio_history(self, request):
            return types.SimpleNamespace(timestamp=[], equity=[], profit_loss=[], profit_loss_pct=[])

    client.TradingClient = TradingClient
    requests.MarketOrderRequest = MarketOrderRequest
    requests.LimitOrderRequest = LimitOrderRequest
    requests.StopOrderRequest = StopOrderRequest
    requests.StopLimitOrderRequest = StopLimitOrderRequest
    requests.GetOrdersRequest = GetOrdersRequest
    enums.OrderSide = AlpacaOrderSide
    enums.OrderType = AlpacaOrderType
    enums.TimeInForce = AlpacaTimeInForce
    enums.OrderStatus = AlpacaOrderStatus
    enums.QueryOrderStatus = QueryOrderStatus

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums)


def test_alpaca_broker_smoke_mock(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    _install_fake_alpaca(monkeypatch)

    from hyprl.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    account = broker.get_account()
    assert account.equity == 10000.0
    assert broker.is_market_open() is True

    order = broker.submit_order(symbol="NVDA", qty=1, side=OrderSide.BUY)
    assert order.symbol == "NVDA"
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.ACCEPTED
