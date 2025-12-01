# Reproduce Core Results – HyprL Ascendant v2

## 0) Setup (résumé)
- Créer un venv, installer les dépendances:  
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt  # si présent
  pip install -e .                 # dev install
  ```
- Assumer que les données/caches sont disponibles localement (prix 1h via MarketDataFetcher).  
- Exécuter les commandes depuis la racine du repo (préfixer par `.venv/bin/python` si besoin).

## 1) NVDA 1h v2 — Backtest 1 an + Replay + Diff (parité)
Commandes:
```bash
.venv/bin/python scripts/run_backtest.py \
  --config configs/NVDA-1h_v2.yaml \
  --start 2024-11-27 --end 2025-11-26 \
  --signal-log data/signals/nvda_bt_v2_repro.csv

.venv/bin/python scripts/run_live_replay.py \
  --config configs/NVDA-1h_v2.yaml \
  --start 2024-11-27 --end 2025-11-26 \
  --parity-mode \
  --signal-log data/signals/nvda_replay_v2_repro.csv

.venv/bin/python scripts/diff_signal_logs.py \
  --backtest-log data/signals/nvda_bt_v2_repro.csv \
  --replay-log data/signals/nvda_replay_v2_repro.csv \
  --output data/signals/nvda_diff_v2_repro.csv
```
Attendu (tolérance: ±0.05 PF, ±1% DD, ±5 trades max si warmup diffère):
- Backtest NVDA v2 (2024-11-27 → 2025-11-26): PF≈3.26, Sharpe≈9.31, MaxDD≈8.03%, Trades≈337, Win≈75.07%, Final balance≈256,994.99 (+2,469.95%), Exit mix: trailing≈74.8%, stop_loss≈24.9%, end_of_data≈0.3%.
- Replay (mêmes dates): Trades=337, PF≈3.192, Win≈75.07%, Total PnL≈640,896.36, Final balance≈650,896.36, Exit mix identique.
- Diff (nvda_diff_v2_repro.csv): Rows compared=342, BT-only=0, Replay-only=0, Matching decisions=342.
Si PF ou DD trop différents: vérifier période/config/cache et présence de `--parity-mode`.

## 2) W0 (2025-01-01 → 2025-03-01) — per-ticker replays (orchestrateur)
Exemple commande:
```bash
.venv/bin/python scripts/run_portfolio_replay_orch_cli.py \
  --configs configs/NVDA-1h_v2.yaml \
           configs/META-1h_v2.yaml \
           configs/MSFT-1h_v2.yaml \
           configs/AMD-1h_v2.yaml \
           configs/QQQ-1h_v2.yaml \
           configs/SPY-1h_v2.yaml \
  --start 2025-01-01 --end 2025-03-01 \
  --out-dir live/logs/portfolio/orch_cli_W0_repro
```
Attendus par ticker (ordres de grandeur):
- NVDA: Trades 43, PF 2.516, Win 72.09%, PnL 4,384.78, Final bal 14,384.78, Exit mix stop_loss 27.9%, trailing 72.1%.
- META: Trades 27, PF 2.674, Win 74.07%, PnL 1,342.87, Final bal 11,342.87, Exit mix end_of_data 3.7%, stop_loss 22.2%, trailing 74.1%.
- MSFT: Trades 37, PF 4.122, Win 81.08%, PnL 2,482.91, Final bal 12,482.91, Exit mix end_of_data 2.7%, stop_loss 16.2%, trailing 81.1%.
- AMD: Trades 36, PF 2.172, Win 63.89%, PnL 1,852.36, Final bal 11,852.36, Exit mix end_of_data 2.8%, stop_loss 33.3%, trailing 63.9%.
- QQQ: Trades 42, PF 1.590, Win 69.05%, PnL 831.76, Final bal 10,831.76, Exit mix stop_loss 31.0%, trailing 69.0%.
- SPY (legacy, poids 0 en v2): Trades 39, PF 0.878, Win 61.54%, PnL -191.41, Final bal 9,808.59, Exit mix end_of_data 2.6%, stop_loss 35.9%, trailing 61.5%.
Sanity check: si PF par ticker est dans ces bornes et le volume de trades est proche, la repro W0 est OK.

## 3) Portefeuille Ascendant v2 — W0/W1/W2 (agrégation)
Commande type:
```bash
.venv/bin/python scripts/aggregate_portfolio_replay.py \
  --trade-logs \
    live/logs/portfolio/orch_cli_W0_repro/trades_NVDA_orch.csv \
    live/logs/portfolio/orch_cli_W0_repro/trades_META_orch.csv \
    live/logs/portfolio/orch_cli_W0_repro/trades_MSFT_orch.csv \
    live/logs/portfolio/orch_cli_W0_repro/trades_AMD_orch.csv \
    live/logs/portfolio/orch_cli_W0_repro/trades_QQQ_orch.csv \
    live/logs/portfolio/orch_cli_W0_repro/trades_SPY_orch.csv \
  --weights NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00 \
  --initial-equity 10000 \
  --summary-out live/logs/portfolio/orch_cli_W0_repro/summary_portfolio_v2_repro.json \
  --equity-out live/logs/portfolio/orch_cli_W0_repro/equity_portfolio_v2_repro.csv
```
Attendus (tolérances légères):
- W0: PF ≈ 6.86, MaxDD ≈ 3.2%, Trades ≈ 360–390.
- W1 (2024-09-01 → 2024-11-30): PF ≈ 1.30, MaxDD ≈ 2.0%, Trades ≈ 340–380.
- W2 (2025-03-01 → 2025-05-31): PF ≈ 3.73, MaxDD ≈ 3.3%, Trades ≈ 350–400.

## 4) NVDA “live-like paper” (optionnel)
Commande:
```bash
.venv/bin/python scripts/run_live_hour.py \
  --config configs/NVDA-1h_v2.yaml \
  --backfill \
  --start 2025-01-01 --end 2025-03-31 \
  --trade-log live/logs/live_nvda/trades_NVDA_paper_W0_repro.csv \
  --summary-file live/logs/live_nvda/summary_NVDA_paper_W0_repro.json

.venv/bin/python scripts/monitor/monitor_portfolio_health.py \
  --trade-logs live/logs/live_nvda/trades_NVDA_paper_W0_repro.csv \
  --weights NVDA=1.0 \
  --initial-equity 10000 \
  --annualization 1638 \
  --summary-out live/logs/live_nvda/health_NVDA_paper_W0_repro.json
```
Attendus: PF ≈ 1.55–1.60, MaxDD ≈ 10–11%, Sharpe (ann 1638) ≈ 7.0, Trades ≈ 170–190.

## 5) Expected vs observed / troubleshooting
- PF/DD divergents: vérifier période, configs YAML, caches de prix, présence de `--parity-mode` en replay.
- Diff parity NVDA non nulle: vérifier que les deux signal logs proviennent de la même période/config et que le warmup est suffisant.
- Mismatch de trades: s’assurer que start/end sont respectés (caches anciens peuvent biaiser), warmup suffisant (≥77 bars pour nvda_v2).
- Scripts introuvables: vérifier chemins `scripts/` et `configs/`, exécuter depuis la racine du repo, préfixer avec `.venv/bin/python` si nécessaire.
