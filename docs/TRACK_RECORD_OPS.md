# Track Record Ops (v0)

This doc defines the paper and live micro procedures for the public track record.
Core v3 stays frozen. Execution uses the Alpaca bridge.

## Locations

- Snapshots: `docs/reports/track_record/snapshots/`
- Latest snapshot: `docs/reports/track_record/latest.json`
- Reports: `docs/reports/TRACK_RECORD.md` and `docs/reports/TRACK_RECORD.json`
- Report hash: `docs/reports/TRACK_RECORD.sha256`
- Bridge log (optional): `live/execution/alpaca/orders.jsonl`

## Paper procedure (minimum 2 weeks)

1) Start core signal engine (unchanged):
```bash
.venv/bin/python scripts/run_live_hour.py \
  --config configs/NVDA-1h_v3.yaml \
  --trade-log live/logs/core_v3/trades_NVDA.csv \
  --summary-file live/logs/core_v3/summary_NVDA.json
```

2) Start Alpaca bridge (paper):
```bash
. ./.env.broker.alpaca
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

3) Daily snapshot (paper):
```bash
. ./.env.broker.alpaca
.venv/bin/python scripts/ops/alpaca_track_record_snapshot.py --paper
```

4) Weekly report:
```bash
.venv/bin/python scripts/ops/make_track_record_report.py
```

## Live micro procedure (after paper stability)

- Reduce risk. Keep max orders/day and max notional/day low.
- Use kill switch file path in the bridge.
- Keep separate paper and live snapshots.

## Deterministic regeneration

1) Ensure snapshots exist.
2) Re-run report script:
```bash
.venv/bin/python scripts/ops/make_track_record_report.py \
  --snapshots-dir docs/reports/track_record/snapshots \
  --orders-log live/execution/alpaca/orders.jsonl \
  --out-dir docs/reports
```

3) Verify hashes:
```bash
sha256sum -c docs/reports/TRACK_RECORD.sha256
```

## Notes

- Snapshots are append-only by date.
- Do not store secrets in the repo.
- Use `--dry-run` and `--once` on the bridge for safe smoke tests.
