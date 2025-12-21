# HyprL Runbook (v0)

## Start procedures

1) Ensure env is loaded:
```bash
source .env.broker.alpaca
```

2) Start core signal engine:
```bash
.venv/bin/python scripts/run_live_hour.py \
  --config configs/NVDA-1h_v3.yaml \
  --trade-log live/logs/core_v3/trades_NVDA.csv \
  --summary-file live/logs/core_v3/summary_NVDA.json
```

3) Start Alpaca bridge:
```bash
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl \
  --out live/execution/alpaca/orders.jsonl \
  --state live/execution/alpaca/state.json \
  --paper \
  --symbols NVDA,MSFT,QQQ \
  --max-orders-per-day 10 \
  --max-notional-per-day 5000 \
  --kill-switch /tmp/hyprl_kill_switch
```

## Stop procedures

- Create kill switch:
```bash
touch /tmp/hyprl_kill_switch
```

- Stop processes with standard process manager or Ctrl+C.

## If signals stop

- Check core logs:
  - `live/logs/live_signals.jsonl`
  - `live/logs/core_v3/summary_NVDA.json`
- Verify market open via broker.

## If Alpaca API errors

- Check `live/execution/alpaca/orders.jsonl` for error events.
- Restart bridge after cooldown.
- Confirm credentials in `.env.broker.alpaca`.

## Emergency close

If required, use broker close via Alpaca console or dedicated ops script (if available).

## Logs

- Signals: `live/logs/live_signals.jsonl`
- Orders: `live/execution/alpaca/orders.jsonl`
- Track record: `docs/reports/track_record/`
