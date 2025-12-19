# HyprL `core_v3` — Freeze Spec

**Status:** FROZEN (baseline production/research)  
**Frozen tag:** `portfolio_core_1h_v3_gate2_oos_v1`  
**Primary configs:** `configs/NVDA-1h_v3.yaml`, `configs/MSFT-1h_v3.yaml`, `configs/QQQ-1h_v3.yaml`, `configs/portfolio_core_1h_v3.yaml`  
**Native engine:** `native/hyprl_supercalc` + wrapper `src/hyprl/native/supercalc.py`  
**Factsheet:** `docs/reports/portfolio_core_1h_v3_factsheet.md`

## Frozen surface
- Ticker presets: `NVDA-1h_v3.yaml`, `MSFT-1h_v3.yaml`, `QQQ-1h_v3.yaml` (feature_columns=19, model artifacts `models/*_1h_xgb_v3.joblib`).
- Portfolio preset: `configs/portfolio_core_1h_v3.yaml` (weights 0.4/0.3/0.3, annualization 1638).
- Native interface: `native/hyprl_supercalc/`, `src/hyprl/native/supercalc.py`, parity tests `tests/test_supercalc_*.py`, `tests/backtest/test_supercalc_native.py`.
- Ops entrypoints: `scripts/run_portfolio_backtest_v3.py`, `scripts/ops/run_core_v3_hourly_batch.py`, `scripts/ops/run_portfolio_monitor_live.py`, `scripts/ops/push_core_v3_discord.py`.

## Data/feature/model contract
- OHLCV 1h (yfinance cap ~730d), UTC index.
- Feature set (19 cols, preset `nvda_v2` in configs): `ret_1h, ret_3h, ret_6h, ret_24h, atr_14, atr_72, atr_14_norm, atr_72_norm, rsi_7, rsi_14, rsi_21, vol_ratio_10_30, vol_regime_high, volume_zscore_24, volume_surge, range_pct, true_range, ret_skew_20, ret_kurt_20`.
- Models: XGBoost, calibration none, artifacts in `models/*_1h_xgb_v3.joblib`, scaler n_features_in_=19.

## Signal/risk (per configs)
- Thresholds: NVDA long 0.53/short 0.47; MSFT long 0.63/short 0.55; QQQ long 0.73/short 0.53.
- Trend filter on (long_min/short_min per config).
- Risk: risk_pct 0.01, atr_multiplier 1.0, reward_multiple 1.5, min_position_size 5.
- Trailing: enabled, stop_activation 1.0, stop_distance 0.04.
- Throttle (QQQ): max_daily_trades=1, min_bars_between_entries=6; (MSFT) max_daily_trades=3, min_bars_between_entries=6.

## Backtest/portfolio pipeline
- Runner: `scripts/run_portfolio_backtest_v3.py` (engine native preferred, costs applied: commission_pct=slippage_pct=0.0005 per side).
- Portfolio weights: NVDA 0.4 / MSFT 0.3 / QQQ 0.3.
- Factsheet snapshot (net costs base): PF_net 3.01, Sharpe_net 7.32, MaxDD_net 4.53%, Trades 1582; OOS 2024-03-01→2024-12-01 PF_net 7.79, Sharpe_net 17.91, MaxDD_net 0.82% (Trades 390).

## Live/replay ops
- Live hour runner: `scripts/run_live_hour.py`; replay: `scripts/run_live_replay.py` (uses BacktestConfig built from config; PaperBrokerImpl costs same as backtest).
- Portfolio ops wrappers: `scripts/ops/run_live_multi_ticker_hourly.py`, `scripts/ops/run_portfolio_monitor_live.py`, `scripts/ops/run_core_v3_hourly_batch.py`.
- Discord poster: `scripts/ops/push_core_v3_discord.py`.

## Build/test
- Build native: `bash scripts/build_supercalc.sh`.
- Parity tests: `pytest tests/test_supercalc_*.py tests/backtest/test_supercalc_native.py -q`.
- Gate1 smoke: `pytest tests/backtest/test_qqq_v3_gate1.py -q` (present).

## Change protocol
Any change to frozen items = new version tag, new factsheet, log in PROJECT_BRAIN + manifest/hashes.
