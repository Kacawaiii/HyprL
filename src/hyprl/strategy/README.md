# Strategy Module (Private)

This directory contains the proprietary trading logic:

- **Signal generation** - ML probability thresholds + technical filters
- **Position sizing** - Kelly criterion implementation
- **Entry/Exit logic** - Smart filters and guards
- **Risk management** - Circuit breakers, trailing stops

## Architecture

```
strategy/
├── core.py              # Main strategy orchestrator
├── signal_quality.py    # Signal scoring and filtering
├── sizing_v2.py         # Kelly-based position sizing
├── guards_v2.py         # Entry guards and filters
├── exits_v2.py          # Exit logic (stops, targets)
├── trailing_v2.py       # Trailing stop implementation
└── smart_filter.py      # ML-based trade filtering
```

## Why Private?

The strategy logic represents the trading edge. The infrastructure (broker integration, indicators, backtesting engine) is public to demonstrate technical competence.

**Want access?** See [hyprlcore.com](https://hyprlcore.com) for signal subscriptions.
