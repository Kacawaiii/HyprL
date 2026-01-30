# Final Ops Summary: Live Data + Bridge Fixes

Date: 2026-01-07 (UTC)

## Problem
- Live signals were produced from stale bars (yfinance fallback), leading to `SKIP_STALE_BAR` and/or trading on old data.
- Close/flip failed with Alpaca `held_for_orders` because open exit legs reserved qty.
- Exits were expiring at close (DAY orders), leaving positions unprotected overnight.
- Duplicate writers / mixed equity values polluted `live_signals.jsonl` and auditability.
- Caps were blocking orders instead of clipping size.

## Fixes Applied

### Data/Signals
- Switched live bars to Alpaca via `alpaca-py` (feed `IEX`), with yfinance fallback.
- Added `data_source`, `bar_start_time`, `bar_end_time`, and stale-bar guard (`HYPRL_MAX_BAR_AGE_SEC=5400`).
- Added bar completion buffer (`HYPRL_BAR_COMPLETE_BUFFER_SEC=10`) to avoid using an in-progress 1h bar.
- Fixed missing `timedelta` import in the live runner.
- Hourly wrapper now sources `/home/ubuntu/HyprL/.env.bridge` for correct envs.

### Bridge / Execution
- Added `cancel_open_orders_for_symbol()` before close/flip to prevent `held_for_orders`.
- Reattach exits with **OCO + GTC** (non-expiring), including rebase to `avg_entry_price`.
- Added integer qty enforcement for bracket shorts; logs `bracket_qty_floor` / `blocked_bracket_qty_lt_1`.
- Caps now **clip** size instead of hard-blocking, to keep trading within `max_notional_total` / `max_notional_symbol`.
- Slippage guard enabled: `--max-entry-slippage-pct 0.3` (logs `slippage_reject_long/short`).
- Skip entry if already past TP: `--skip-past-tp` (logs `skip_past_tp_long/short`).
- RTH polling: `--poll-seconds 60` off-hours, `--poll-seconds-rth 10` in-session.
- Limit-entry mode available but disabled: `--use-limit-orders --limit-buffer-pct 0.1`.

### Ops / Infra
- Rotated `live_signals.jsonl` and reset bridge `last_offset` for a clean audit stream.
- Hourly systemd timer uses `flock` to prevent overlap.
- Track-record daily service retained; baseline snapshot revalidated with start equity ~100k.

## Validation (Pass)
- Alpaca bars present for NVDA/MSFT/QQQ; last bar timestamp near current session.
- Stale-bar guard fires outside RTH (expected) and not during RTH when bars are fresh.
- Exit orders show `order_class=oco`, `time_in_force=gtc`, `legs=True`.
- `live_signals.jsonl` clean and `data_source=alpaca:iex` after rotation.

## Validation (Pending)
- Close/flip sequence post-patch still needs a live event to confirm:
  `cancel_open_orders -> close_position` with **no** `close_failed held_for_orders`.

## Remaining Checks (RTH)
- Run during US market hours (15:30–22:00 Paris):
  - `hyprl-core-v3-hourly.service` writes fresh signals.
  - `age_sec < HYPRL_MAX_BAR_AGE_SEC` and `bar_len_sec = 3600`.
- On next flip/flat:
  - confirm `cancel_open_orders` then `close_position` in `orders.jsonl`/journald.

## Optional Next Steps
- Add RTH-only guard in hourly wrapper to avoid nightly skip noise.
- Add entry-gap guard if execution slippage dominates (after >=20 closed entries).
