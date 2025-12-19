#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=./src

OUT="live/exp/repro_$(date +%F_%H%M)"

echo "[1] Build supercalc"
bash scripts/build_supercalc.sh

echo "[2] Portfolio backtest v3"
if [ -f scripts/run_portfolio_backtest_v3.py ]; then
  python scripts/run_portfolio_backtest_v3.py --config configs/portfolio_core_1h_v3.yaml
elif [ -f scripts/run_portfolio_replay.py ]; then
  echo "INFO: portfolio backtest runner missing; using replay aggregator"
  python scripts/run_portfolio_replay.py \
    --configs configs/NVDA-1h_v3.yaml configs/MSFT-1h_v3.yaml configs/QQQ-1h_v3.yaml \
    --weights 0.4 0.3 0.3 \
    --start 2024-03-01 --end 2024-12-01 \
    --out-dir "$OUT/portfolio" \
    --summary-out "$OUT/portfolio/summary.json"
else
  echo "WARN: no portfolio runner found; skipping portfolio step"
fi

echo "[3] Replays W0/W1/W2 (NVDA/QQQ)"
mkdir -p "$OUT"/{NVDA,QQQ}/W{0,1,2}

python scripts/run_live_replay.py --config configs/NVDA-1h_v3.yaml --start 2024-09-01 --end 2024-11-30 --trade-log "$OUT/NVDA/W1/trades.csv"
python scripts/run_live_replay.py --config configs/NVDA-1h_v3.yaml --start 2025-01-01 --end 2025-03-31 --trade-log "$OUT/NVDA/W0/trades.csv"
python scripts/run_live_replay.py --config configs/NVDA-1h_v3.yaml --start 2025-03-01 --end 2025-05-31 --trade-log "$OUT/NVDA/W2/trades.csv"

if [ -f models/qqq_1h_xgb_v3.joblib ]; then
  python scripts/run_live_replay.py --config configs/QQQ-1h_v3.yaml --start 2024-09-01 --end 2024-11-30 --trade-log "$OUT/QQQ/W1/trades.csv"
  python scripts/run_live_replay.py --config configs/QQQ-1h_v3.yaml --start 2025-01-01 --end 2025-03-31 --trade-log "$OUT/QQQ/W0/trades.csv"
  python scripts/run_live_replay.py --config configs/QQQ-1h_v3.yaml --start 2025-03-01 --end 2025-05-31 --trade-log "$OUT/QQQ/W2/trades.csv"
else
  echo "WARN: models/qqq_1h_xgb_v3.joblib missing; skipping QQQ replays"
fi

echo "[4] Hashes"
find "$OUT" -name "*.csv" -type f -print0 | sort -z | xargs -0 sha256sum | tee "$OUT/hashes.sha256"

echo "[DONE] Outputs at $OUT"
