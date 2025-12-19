# HyprL SENSE — Full Spec (audit-only v0.2)

**Status:** Audit-only (neutralized). Core_v3 frozen; SENSE sidecar séparé.  
**Config:** `configs/sense.yaml` (risk_multiplier=1.0, threshold_tighten=0.0).  
**Code:** `src/hyprl_sense/*` (backfill, chrono, fear, fusion, runner, shadow, telemetry).  
**Ops:** `scripts/ops/run_sense_daily.sh` (60d window NVDA/QQQ), `scripts/run_sense_backfill.py`, `scripts/run_sense_shadow.py`.  
**Analytics:** `scripts/analysis/*` (rollup, weak-days, slot impact, events, trade impact).

## Pipeline
- Backfill OHLCV 1h (yfinance, cap ~730d), chunking via `backfill.py`.
- Features: chrono (gap_multiple, chrono_trust), fear (atr_spike proxy).
- Anti-lookahead: aligned `_a` columns (shifted by 1).
- Fusion: sense_trust = (chrono_trust_a + (1 - fear_index_a))/2; sense_filter default ALLOW; BLOCK only if fear_index_a>=0.90 (fallback); reason_code telemetry; risk_multiplier forced 1.0.
- Outputs: controls_{ticker}_1h.parquet + summary_{ticker}.json under `live/sense/<TICKER>/<RUN_ID>/`.

## Shadow harness
- Merge trades/controls with timestamp normalization (trade_ts/entry_ts/index; sense_ts/ts/index), tolerance default 90m, filters trades outside controls_min/max ± tol.
- Summary: match_rate, retention_on_window/over_all_trades, filtered_out_count, max_dd_baseline/shadow, filter_distribution, pnl_delta (scaled by risk_multiplier).

## Analytics (audit)
- Rollup controls: `scripts/analysis/sense_rollup_controls.py` → `live/sense/_analytics/<T>_controls_rollup_1h.parquet` + sources.csv (tz normalized, dedupe).
- Weak-days: `scripts/analysis/sense_weak_days_report.py` → weekday/hour stats + REPORT.md.
- Slot impact trades: `scripts/analysis/sense_trade_slot_impact.py`.
- Events: CPI/NFP via `scripts/analysis/build_bls_events.py`; FOMC ICS via `scripts/analysis/build_fomc_events.py`; events stored in `live/sense/_analytics/events/events.csv`.
- Event flag impact (bars): `scripts/analysis/sense_event_flag_impact.py` (filters out-of-range events, anchors to first bar ≥ event, post-event windows 1h/4h/1d, intraday ret_fwd_1h, slot-matched diffs).
- Trade event impact: `scripts/analysis/sense_trade_event_impact.py` (post-event windows only).
- Outputs per ticker: `live/sense/_analytics/reports/<TICKER>/` (REPORT_events.md, event_flag_impact.csv, event_slot_matched_impact_*.csv, weekday stats).

## Ops/cron
- Daily capture (audit-only): `scripts/ops/run_sense_daily.sh` (run_id `sense_live_<date>_<time>`, NVDA+QQQ 60d).
- Logs: `live/sense/_ops/sense_daily.log`.

## Constraints
- No sizing/threshold impact (risk_multiplier=1.0, threshold_tighten=0.0); BLOCK only fear>=0.90 fallback.
- Yahoo cap ~730d; use chunk-days in backfill.

## Build/test
- Sense tests: `pytest tests/hyprl_sense -q`.
- Merge robustness: `tests/hyprl_sense/test_shadow_merge.py` (ts/DatetimeIndex cases, window filtering).
