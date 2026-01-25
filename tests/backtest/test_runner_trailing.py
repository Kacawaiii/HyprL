import pytest
import pandas as pd
from hyprl.backtest.runner import _simulate_trade_python, _simulate_trade
from hyprl.risk.manager import RiskOutcome

def test_trailing_stop_moves_to_be_plus():
    # Setup: Long trade, Entry 100, Stop 90 (Risk 10).
    # Activation 1.0R (at 110), Distance 0.0R (Stop moves to High - 0 = High).
    # Wait, user example: "activation=1.0, distance=0.0 -> stop passe à entry après +1R"
    # If distance is 0.0, stop moves to High - 0. So if price hits 110, stop moves to 110.
    # If we want stop to move to ENTRY, distance should be 1.0R (10).
    # Let's follow the user's logic: "activation=1.0, distance=0.0 -> stop passe à entry après +1R"
    # If activation is 1.0R (price 110), and distance is 0.0R (0), then new stop = 110 - 0 = 110.
    # This locks in the profit immediately.
    # If the user meant "Breakeven", distance should be 1.0R (10). 110 - 10 = 100.
    # Let's assume the user wants to test the mechanism, so I'll use specific values.
    
    # Let's test: Activation at 110 (1R), Distance 5 (0.5R).
    # Entry 100, Stop 90. Risk 10.
    # Activation Price = 110. Distance Delta = 5.
    
    risk = RiskOutcome(
        direction="long",
        entry_price=100.0,
        position_size=1.0,
        stop_price=90.0,
        take_profit_price=200.0, # Far away
        risk_amount=10.0,
        rr_multiple=10.0,
        trailing_stop_activation_price=110.0,
        trailing_stop_distance_price=5.0
    )
    
    # Prices:
    # 0: 100 (Entry)
    # 1: 105 (No activation yet)
    # 2: 112 (Activation! High 112 >= 110. New Stop = 112 - 5 = 107)
    # 3: 106 (Dip. Low 106 <= Stop 107. Stopped out at 107)
    
    prices = pd.DataFrame({
        "high": [100, 105, 112, 108],
        "low":  [100, 102, 109, 106],
        "close":[100, 104, 111, 106]
    })
    
    exit_price, exit_idx, reason = _simulate_trade_python(prices, 0, risk)
    
    # Should exit at index 3 (Low 106 hits Stop 107)
    # Exit price should be the stop price: 107.
    assert exit_idx == 3
    assert exit_price == 107.0
    assert reason == "trailing_stop"

def test_trailing_stop_ne_recule_jamais():
    # Long trade. Entry 100, Stop 90.
    # Activation 110. Distance 5.
    
    risk = RiskOutcome(
        direction="long",
        entry_price=100.0,
        position_size=1.0,
        stop_price=90.0,
        take_profit_price=200.0,
        risk_amount=10.0,
        rr_multiple=10.0,
        trailing_stop_activation_price=110.0,
        trailing_stop_distance_price=5.0
    )
    
    # Prices:
    # 1: High 115. Activation! Stop = 115 - 5 = 110.
    # 2: High 112. (Lower high). New potential stop = 112 - 5 = 107.
    #    But current stop is 110. Should NOT lower to 107.
    # 3: Low 108. Should trigger stop at 110.
    
    prices = pd.DataFrame({
        "high": [100, 115, 112, 112],
        "low":  [100, 114, 111, 108],
        "close":[100, 114, 111, 108]
    })
    
    exit_price, exit_idx, reason = _simulate_trade_python(prices, 0, risk)
    
    assert exit_idx == 3
    assert exit_price == 110.0
    assert reason == "trailing_stop"

def test_trailing_short():
    # Short trade. Entry 100, Stop 110. Risk 10.
    # Activation 90 (1R). Distance 5 (0.5R).
    # Activation Price = 90. Distance Delta = 5.
    
    risk = RiskOutcome(
        direction="short",
        entry_price=100.0,
        position_size=1.0,
        stop_price=110.0,
        take_profit_price=0.0,
        risk_amount=10.0,
        rr_multiple=10.0,
        trailing_stop_activation_price=90.0,
        trailing_stop_distance_price=5.0
    )
    
    # Prices:
    # 0: Entry
    # 1: Low 88. Activation! (88 <= 90). New Stop = 88 + 5 = 93.
    #    High 95. Stop is 110 initially. 95 < 110. Safe.
    #    Wait, does it check stop BEFORE trailing update?
    #    In my code: 1. Update Trailing. 2. Check Exits.
    #    So at idx 1:
    #      Lowest = 88. Activation triggered. New Stop = 93.
    #      Check Exits: High 95 >= New Stop 93? YES.
    #      So it exits at idx 1.
    
    prices = pd.DataFrame({
        "high": [100, 95, 94],
        "low":  [100, 88, 92],
        "close":[100, 90, 93]
    })
    
    exit_price, exit_idx, reason = _simulate_trade_python(prices, 0, risk)
    
    # It exits at idx 1 because High 95 >= New Stop 93.
    assert exit_idx == 1
    assert exit_price == 93.0
    assert reason == "trailing_stop"


def test_simulate_trade_with_trailing_skips_accel(monkeypatch: pytest.MonkeyPatch) -> None:
    # When trailing parameters are present we should bypass the Rust accelerator to
    # ensure trailing stop logic and exit reasons match the Python implementation.
    def fake_accel(*_args, **_kwargs):  # pragma: no cover - should never run
        raise AssertionError("accelerator should be skipped for trailing trades")

    monkeypatch.setattr("hyprl.backtest.runner._simulate_trade_path_accel", fake_accel)

    risk = RiskOutcome(
        direction="long",
        entry_price=100.0,
        position_size=1.0,
        stop_price=90.0,
        take_profit_price=200.0,
        risk_amount=10.0,
        rr_multiple=10.0,
        trailing_stop_activation_price=110.0,
        trailing_stop_distance_price=5.0,
    )

    prices = pd.DataFrame(
        {
            "high": [100, 105, 112, 108],
            "low": [100, 102, 109, 106],
            "close": [100, 104, 111, 106],
        }
    )

    exit_price, exit_idx, reason = _simulate_trade(prices, 0, risk)

    assert exit_idx == 3
    assert exit_price == 107.0
    assert reason == "trailing_stop"
