# HyprL Ascendant v2 – Copilot Brief

Use this as the default system/context prompt for coding assistance. Scope is maintenance/ops/reporting; **do not change core strategy/risk/config baselines.**

## Golden Rules (do not touch)
- Core risk/strategy logic: `src/hyprl/risk/{kelly,guards,sizing}.py`, `src/hyprl/execution/algos.py` (TWAP), feature presets (`src/hyprl/features/nvda_v2.py`, `equity_v2` stack), `src/hyprl/data/market.py` windowing.
- Configs v2: `configs/*-1h_v2.yaml`, `configs/portfolio_ascendant_v2.yaml`; Ascendant v2 weights = NVDA 0.30, MSFT 0.27, AMD 0.27, META 0.09, QQQ 0.07, SPY 0.00.
- Replay-orch + aggregate remain the portfolio baseline; do not redesign the old multi-ticker live runner.

## Baselines (keep within same order of magnitude)
- NVDA 1h v2 1y BT (2024-11-27→2025-11-26, Kelly/TWAP OFF): PF≈3.26, MaxDD≈8.03 %, Trades≈337. Replay parity: PF≈3.19, 0 mismatches vs BT.
- Per-ticker W0 replay-orch (2025-01-01→2025-03-01, Kelly/TWAP OFF): NVDA PF≈2.516 (43 trades), META≈2.674 (27), MSFT≈4.122 (37), AMD≈2.172 (36), QQQ≈1.590 (42), SPY≈0.878 (39).
- Ascendant v2 portfolio (replay-orch): W0 PF≈6.856 DD≈3.21 %; W1 PF≈1.303 DD≈2.01 %; W2 PF≈3.732 DD≈3.34 %.
- NVDA live-like paper W0 (run_live_hour, Kelly+guards ON, TWAP OFF): PF≈1.566, MaxDD≈10.46 %, Sharpe≈7.10, Trades≈176. Rolling ~120 trades: PF≈3.719, MaxDD≈11.84 %, Sharpe≈15.33.

## Allowed scope
- Maintenance/hygiene: ops/reporting/tools/scripts, docs, tests; small validation/logging/docstring tweaks.
- Keep reporter timezone-aware (`datetime.now(timezone.utc)`), HTML gated on `markdown` presence.
- No model/risk/feature changes, no config tuning.

## Key scripts/CLIs
- Backtest/Replay/Diff: `scripts/run_backtest.py`, `scripts/run_live_replay.py`, `scripts/diff_signal_logs.py`.
- Ops runners: `scripts/ops/run_live_single_ticker_hourly.py`, `scripts/ops/run_live_multi_ticker_hourly.py`.
- Concat: `scripts/tools/concat_trades_live_generic.py` (pattern optional, mtime-ordered, dedupe trade_id→fallback tuple, empty CSV with headers).
- Monitor: `scripts/monitor/monitor_portfolio_health.py`; wrapper `scripts/ops/run_portfolio_monitor_live.py` (Asc v2 weights default, skips zero-weight, parses status).
- Reporting: `scripts/reporting/export_report.py` (md/html, weight coverage check, weight-sum warning in console/notes).
- Replay-orch: `scripts/run_portfolio_replay_orch_cli.py` (subprocess run_live_replay), aggregator `scripts/aggregate_portfolio_replay.py`.

## Targeted tests (avoid full pytest in constrained env)
Run:
```
.venv/bin/pytest \
  tests/scripts/test_concat_trades_live_generic.py \
  tests/scripts/test_export_report.py \
  tests/scripts/test_compute_portfolio_metrics.py \
  tests/features/test_equity_v2_presets.py \
  tests/risk/test_kelly.py \
  tests/risk/test_live_risk_caps.py \
  tests/risk/test_risk_guards.py \
  tests/execution/test_twap.py -q
```
Address only failures directly related to the touched code.

## Ops pipelines (Palier 1/2)
- NVDA hourly: `run_live_single_ticker_hourly.py --config configs/NVDA-1h_v2.yaml --log-root live/logs [--backfill --start ... --end ...]`.
- Palier 2 multi-ticker: `run_live_multi_ticker_hourly.py --symbols NVDA,MSFT,AMD` (or `--configs ...`) → nightly `concat_trades_live_generic.py` per ticker → `run_portfolio_monitor_live.py --log-root live/logs --summary-out live/logs/portfolio_live/health_asc_v2.json`.
- Reports: `export_report.py --trade-logs <csvs> --weights <weights> --initial-equity 10000 --annualization 1638 --output <md/html> --format md|html`.

## Docs alignment
- Keep README + docs references consistent with current CLIs; Asc v2 is official, SPY=0, TWAP optional. Update numbers only if rerun shows small drift; no conceptual rewrites.
