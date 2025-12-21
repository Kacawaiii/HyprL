# HyprL Alpaca Broker Integration

**Status:** ✅ Implemented
**Module:** `src/hyprl/broker/alpaca.py`
**Dependency:** `alpaca-py>=0.30.0`

---

## Overview

AlpacaBroker provides full integration with Alpaca Trading API for both paper and live trading.

**Features:**
- ✅ Order submission/cancellation/tracking
- ✅ Position management with emergency close
- ✅ Idempotency via client_order_id
- ✅ Retry/backoff for transient errors
- ✅ Rate limit handling
- ✅ Market clock awareness
- ✅ Portfolio history tracking

---

## Setup

### 1. Install dependencies

```bash
pip install alpaca-py>=0.30.0
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Get Alpaca API credentials

1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Go to [Paper Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
3. Generate API keys (API Key ID + Secret Key)

### 3. Configure environment

```bash
# Copy template
cp .env.broker.alpaca.example .env.broker.alpaca

# Edit with your credentials
nano .env.broker.alpaca
```

**Required variables:**
```bash
ALPACA_API_KEY=your_api_key_id_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true  # true for paper, false for live
```

### 4. Source credentials

```bash
source .env.broker.alpaca
```

---

## Usage

### Basic Example

```python
from hyprl.broker.alpaca import AlpacaBroker
from hyprl.broker.base import OrderSide, OrderType

# Initialize (uses env vars)
broker = AlpacaBroker(paper=True)

# Get account info
account = broker.get_account()
print(f"Equity: ${account.equity:,.2f}")
print(f"Buying Power: ${account.buying_power:,.2f}")

# Check market hours
if broker.is_market_open():
    print("Market is open")
else:
    clock = broker.get_clock()
    print(f"Market closed. Next open: {clock.next_open}")

# Submit market order
order = broker.submit_order(
    symbol="NVDA",
    qty=10,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
)
print(f"Order submitted: {order.id}")

# Wait for fill (with timeout)
filled_order = broker.wait_for_fill(order.id, timeout=30)
if filled_order and filled_order.status.value == "filled":
    print(f"Filled at ${filled_order.filled_avg_price}")
else:
    print("Order not filled")

# Get position
position = broker.get_position("NVDA")
if position:
    print(f"Position: {position.qty} shares @ ${position.entry_price}")
    print(f"Unrealized P&L: ${position.unrealized_pnl:,.2f}")

# Close position
close_order = broker.close_position("NVDA")
print(f"Position closed: {close_order.id}")
```

### Advanced: Limit Orders with Stop Loss

```python
from hyprl.broker.alpaca import AlpacaBroker
from hyprl.broker.base import OrderSide, OrderType, TimeInForce

broker = AlpacaBroker(paper=True)

# Submit limit buy order
entry_order = broker.submit_order(
    symbol="MSFT",
    qty=5,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    limit_price=400.00,
    time_in_force=TimeInForce.DAY,
    client_order_id="msft_entry_20251221_001",
)

# Wait for fill
filled = broker.wait_for_fill(entry_order.id, timeout=60)

if filled:
    # Submit stop-loss order
    stop_order = broker.submit_order(
        symbol="MSFT",
        qty=5,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        stop_price=390.00,  # Stop at -$10/share
        time_in_force=TimeInForce.GTC,
        client_order_id="msft_stop_20251221_001",
    )
    print(f"Stop-loss placed: {stop_order.id}")
```

### Emergency Position Close

```python
# Close all positions immediately
orders = broker.close_all_positions()
print(f"Closed {len(orders)} positions")
```

---

## Integration with HyprL Live Runner

### Replace PaperBrokerImpl

**Before (paper broker):**
```python
from hyprl.broker.dryrun import PaperBrokerImpl

broker = PaperBrokerImpl(initial_cash=10000)
```

**After (Alpaca paper):**
```python
from hyprl.broker.alpaca import AlpacaBroker

broker = AlpacaBroker(paper=True)
```

### Full live_hour.py integration

```python
# scripts/run_live_hour.py
import os
from hyprl.broker.alpaca import AlpacaBroker
from hyprl.live.strategy_engine import StrategyEngine

# Load credentials
from dotenv import load_dotenv
load_dotenv('.env.broker.alpaca')

# Initialize broker
broker = AlpacaBroker(paper=True)

# Check market hours
if not broker.is_market_open():
    print("Market closed, skipping")
    exit(0)

# Initialize strategy engine
engine = StrategyEngine(
    config_path="configs/NVDA-1h_v3.yaml",
    broker=broker,
)

# Run bar
engine.on_new_bar()
```

---

## API Methods

### Account & Portfolio

| Method | Returns | Description |
|--------|---------|-------------|
| `get_account()` | `Account` | Account info (equity, cash, buying_power) |
| `get_position(symbol)` | `Position` | Position for ticker |
| `list_positions()` | `List[Position]` | All open positions |
| `get_portfolio_history(period, timeframe)` | `Dict` | Equity curve |

### Order Management

| Method | Returns | Description |
|--------|---------|-------------|
| `submit_order(...)` | `Order` | Submit new order |
| `get_order(order_id)` | `Order` | Get order by ID |
| `get_order_by_client_id(...)` | `Order` | Get order by client_order_id |
| `list_orders(status, limit)` | `List[Order]` | List orders (open/closed/all) |
| `cancel_order(order_id)` | `bool` | Cancel order |
| `cancel_all_orders()` | `int` | Cancel all open orders |
| `wait_for_fill(order_id, timeout)` | `Order` | Wait for order fill |

### Position Management

| Method | Returns | Description |
|--------|---------|-------------|
| `close_position(symbol)` | `Order` | Close position for ticker |
| `close_all_positions()` | `List[Order]` | Emergency close all |

### Market Clock

| Method | Returns | Description |
|--------|---------|-------------|
| `get_clock()` | `Clock` | Market clock (is_open, next_open) |
| `is_market_open()` | `bool` | True if market open |

---

## Error Handling

AlpacaBroker uses custom exceptions:

```python
from hyprl.broker.base import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
    RateLimitError,
)

try:
    order = broker.submit_order("NVDA", 1000, OrderSide.BUY)
except InsufficientFundsError:
    print("Not enough buying power")
except OrderRejectedError as e:
    print(f"Order rejected: {e}")
except RateLimitError:
    print("Rate limit hit, wait 60s")
except BrokerError as e:
    print(f"Broker error: {e}")
```

**Built-in retry logic:**
- Automatic retry with exponential backoff (3 attempts)
- Rate limit handling (60s wait)
- Transient error recovery

---

## Idempotency

Use `client_order_id` to prevent duplicate orders:

```python
# Same client_order_id won't create duplicate order
order1 = broker.submit_order(
    "NVDA", 10, OrderSide.BUY,
    client_order_id="nvda_entry_20251221_123456"
)

# This will return existing order (not create new one)
order2 = broker.submit_order(
    "NVDA", 10, OrderSide.BUY,
    client_order_id="nvda_entry_20251221_123456"
)

assert order1.id == order2.id
```

---

## Paper vs Live Trading

| Mode | Endpoint | Data | Purpose |
|------|----------|------|---------|
| **Paper** | `paper-api.alpaca.markets` | IEX (15min delayed) | Testing, validation |
| **Live** | `api.alpaca.markets` | Real-time SIP | Production trading |

**⚠️ Important:**
- Paper trading uses IEX data (free, delayed)
- Not suitable for performance testing (data lag)
- Use paper for technical validation only
- Live trading requires account approval + real capital

---

## Rate Limits

Alpaca enforces rate limits:
- **200 requests/minute** for most endpoints
- **Orders:** 200/min
- **Positions:** 200/min
- **Account:** 200/min

AlpacaBroker handles rate limits automatically with retry logic.

---

## Testing

```bash
# Test broker initialization
.venv/bin/python -c "
from hyprl.broker.alpaca import AlpacaBroker
broker = AlpacaBroker(paper=True)
print(broker.get_account())
"

# Run unit tests
pytest tests/broker/test_alpaca.py -v
```

---

## Troubleshooting

### "No module named 'alpaca'"

```bash
pip install alpaca-py>=0.30.0
```

### "Alpaca API credentials required"

Set environment variables:
```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

Or source .env file:
```bash
source .env.broker.alpaca
```

### "Rate limit exceeded"

Wait 60 seconds or reduce request frequency.

### "Market is closed"

Check market hours:
```python
clock = broker.get_clock()
print(f"Next open: {clock.next_open}")
```

---

## Next Steps

1. ✅ Setup Alpaca paper account
2. ✅ Configure .env.broker.alpaca
3. ✅ Test basic operations
4. ⏳ Integrate with live_hour.py
5. ⏳ Run live paper trading (Palier 1)
6. ⏳ Monitor trades + validate parity
7. ⏳ Request live trading approval (after paper validation)

---

## References

- [Alpaca API Docs](https://docs.alpaca.markets/)
- [alpaca-py SDK](https://github.com/alpacahq/alpaca-py)
- [Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
- HyprL Broker Base: `src/hyprl/broker/base.py`
