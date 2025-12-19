# README_repro.md — core_v3 repro/audit snapshot (90+)

1) HEAD=9a9b8304ae9c07d7c49ca58561afddb5255b52a7 (factsheet core_v3)
2) Configs v3: configs/NVDA-1h_v3.yaml, configs/MSFT-1h_v3.yaml, configs/QQQ-1h_v3.yaml, configs/portfolio_core_1h_v3.yaml
3) Modèles v3 (19 feats): models/nvda_1h_xgb_v3.joblib, models/msft_1h_xgb_v3.joblib, models/qqq_1h_xgb_v3.joblib
4) Factsheet: docs/reports/portfolio_core_1h_v3_factsheet.md (PF_net=3.01, Sharpe_net=7.32, MaxDD_net=4.53%, Trades=1582)
5) Build native: bash scripts/build_supercalc.sh
6) Backtest portfolio: python scripts/run_portfolio_backtest_v3.py --config configs/portfolio_core_1h_v3.yaml
7) Replay NVDA W1/W0/W2: python scripts/run_live_replay.py --config configs/NVDA-1h_v3.yaml --start <YYYY-MM-DD> --end <YYYY-MM-DD> --trade-log <OUT>/NVDA/W*/trades.csv
8) Replay QQQ W1/W0/W2: python scripts/run_live_replay.py --config configs/QQQ-1h_v3.yaml --start <YYYY-MM-DD> --end <YYYY-MM-DD> --trade-log <OUT>/QQQ/W*/trades.csv
9) Hashes: sha256sum <OUT>/**/trades.csv > <OUT>/hashes.sha256
10) Scorecards (coûts additionnels; trades déjà net coûts base): python scripts/exp/make_scorecard.py --fee-bps 1.0 --slippage-bps 2.0 --out <OUT>/scorecard_<TICKER>_real.md
11) NVDA baseline (live/exp/NVDA_windows_2025-12-19_1730): W1 PF=7.77 (n=46, MaxDD≈-394), W0 PF=3.54 (n=49, MaxDD≈-293), W2 PF=2.35 (n=48, MaxDD≈-1230)
12) QQQ baseline (live/exp/QQQ_windows_2025-12-19_1737): W1 PF=2.57 (n=44, MaxDD≈-483), W0 PF=3.87 (n=49, MaxDD≈-189), W2 PF=2.26 (n=39, MaxDD≈-800)
13) Coûts base: PaperBrokerImpl net (commission_pct=slippage_pct=0.0005/side); scorecards = stress (incrémental)
14) SENSE audit-only: configs/sense.yaml (risk_multiplier=1.0); analytics live/sense/_analytics/; events.csv (macro)
15) Smoke tests: pytest -q tests/hyprl_sense -p no:cacheprovider; pytest -q tests/test_supercalc_*.py tests/backtest/test_supercalc_native.py -p no:cacheprovider
