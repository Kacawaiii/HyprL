from __future__ import annotations

import math

from hyprl.execution.broker import PaperBroker


def test_market_order_executes_and_updates_balance() -> None:
    broker = PaperBroker(initial_balance=10_000.0)
    broker.place_order("AAA", side="buy", quantity=10, price=100.0)
    positions = broker.get_positions()
    assert math.isclose(broker.get_balance(), 10_000.0 - 1_000.0)
    assert positions["AAA"]["quantity"] == 10
    assert positions["AAA"]["avg_price"] == 100.0


def test_round_trip_generates_realized_pnl() -> None:
    broker = PaperBroker(initial_balance=5_000.0)
    broker.place_order("AAA", side="buy", quantity=20, price=50.0)
    broker.place_order("AAA", side="sell", quantity=20, price=55.0)
    assert math.isclose(broker.get_balance(), 5_000.0 + 100.0)
    assert "AAA" not in broker.get_positions()
    assert broker.get_trade_log()[-1]["realized_pnl"] == 100.0


def test_short_position_supported() -> None:
    broker = PaperBroker(initial_balance=3_000.0)
    broker.place_order("AAA", side="sell", quantity=10, price=30.0)
    positions = broker.get_positions()["AAA"]
    assert positions["quantity"] == -10
    assert positions["avg_price"] == 30.0
    broker.place_order("AAA", side="buy", quantity=10, price=25.0)
    assert math.isclose(broker.get_balance(), 3_000.0 + 50.0)
