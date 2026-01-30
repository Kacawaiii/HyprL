#!/usr/bin/env bash
set -euo pipefail

TICKERS=(AAPL IWM AMD META AMZN)

# Quick gate pair (Platt calibrated).
TH_LONG=0.55
TH_SHORT=0.45

# Test window (last 180d).
START=$(date -d "180 days ago" +%Y-%m-%d)
END=$(date +%Y-%m-%d)

# Sweep grid (only if gate OK).
LONGS=(0.50 0.53 0.55 0.58 0.60 0.63)
SHORTS=(0.37 0.40 0.42 0.45 0.48 0.50)

export POLARS_MAX_THREADS=1

# Gate thresholds.
MIN_TRADES=120
MIN_PF=1.10
MAX_DD=20.0
MIN_SHARPE=0.0

extract_metrics() {
  python - "$1" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="ignore").read()

def grab(pattern: str) -> float:
    match = re.search(pattern, text, re.I)
    if not match:
        return float("nan")
    try:
        return float(match.group(1))
    except Exception:
        return float("nan")

trades = grab(r"Trades:\s*([0-9]+(?:\.[0-9]+)?)")
pf = grab(r"Profit factor:\s*([0-9]+(?:\.[0-9]+)?)")
maxdd = grab(r"Max drawdown:\s*([0-9]+(?:\.[0-9]+)?)")
sharpe = grab(r"Sharpe ratio:\s*([\-0-9]+(?:\.[0-9]+)?)")
print(f"{trades} {pf} {maxdd} {sharpe}")
PY
}

for T in "${TICKERS[@]}"; do
  RUN_ID="$(date +%Y-%m-%d)_${T}"
  RUN_DIR="runs/discovery/${RUN_ID}"
  mkdir -p "$RUN_DIR" models/discovery configs/discovery

  echo "==================== ${T} ===================="

  # 1) Dataset (730d 1h).
  python scripts/prepare_discovery_dataset.py \
    --ticker "$T" --period 730d --interval 1h \
    --output "${RUN_DIR}/${T,,}_1h_features_v3.parquet" \
    --min-rows 2500 --min-span-ratio 0.85 \
    | tee "${RUN_DIR}/dataset.log"

  # 2) Train + holdout NPZ.
  python scripts/train_model_nvda_1h_v3.py \
    --dataset "${RUN_DIR}/${T,,}_1h_features_v3.parquet" \
    --artifact "models/discovery/${T,,}_1h_xgb_v3.joblib" \
    --feature-list "models/discovery/${T,,}_1h_xgb_v3_features.json" \
    --val-preds-out "${RUN_DIR}/val_preds_${T,,}.npz" \
    --horizon 6 --threshold-pct 0.003 --val-frac 0.2 \
    | tee "${RUN_DIR}/train.log"

  # 3) Calibrate (Platt).
  python scripts/calibrate_from_npz.py \
    --npz "${RUN_DIR}/val_preds_${T,,}.npz" \
    --method platt \
    --out "models/discovery/${T,,}_1h_cal_platt.joblib" \
    --plot "${RUN_DIR}/reliability_platt.png" \
    | tee "${RUN_DIR}/calibration.log"

  # 4) Config (Platt calibration artifact).
  cat > "configs/discovery/${T}-1h_v3_platt.yaml" <<YAML
ticker: ${T}
interval: 1h
period: 730d
model:
  type: xgboost
  artifact: models/discovery/${T,,}_1h_xgb_v3.joblib
  preset: discovery_v3
  feature_list: models/discovery/${T,,}_1h_xgb_v3_features.json
  calibration:
    artifact: models/discovery/${T,,}_1h_cal_platt.joblib
thresholds:
  long: ${TH_LONG}
  short: ${TH_SHORT}
risk:
  risk_pct: 0.01
  atr_multiplier: 1.5
  reward_multiple: 2.0
YAML

  # 5) Quick gate backtest (180d).
  python scripts/run_backtest.py \
    --config "configs/discovery/${T}-1h_v3_platt.yaml" \
    --start "${START}" --end "${END}" \
    --long-threshold "${TH_LONG}" --short-threshold "${TH_SHORT}" \
    --engine native --fast \
    --export-trades "${RUN_DIR}/quick_trades_L${TH_LONG}_S${TH_SHORT}.csv" \
    | tee "${RUN_DIR}/quick_metrics.log"

  if [ -s "${RUN_DIR}/quick_trades_L${TH_LONG}_S${TH_SHORT}.csv" ]; then
    python scripts/analyze_trades.py \
      --trades "${RUN_DIR}/quick_trades_L${TH_LONG}_S${TH_SHORT}.csv" \
      > "${RUN_DIR}/quick_analysis.txt"
  else
    echo "[WARN] quick trades file missing -> skipping analyze_trades." | tee -a "${RUN_DIR}/quick_metrics.log"
  fi

  read TRADES PF MAXDD SHARPE < <(extract_metrics "${RUN_DIR}/quick_metrics.log" || echo "nan nan nan nan")
  echo "[GATE] trades=${TRADES} pf=${PF} maxdd=${MAXDD} sharpe=${SHARPE}" | tee -a "${RUN_DIR}/quick_metrics.log"
  if [[ "${SHARPE}" == "nan" ]]; then
    echo "[GATE] sharpe missing -> ignoring sharpe gate." | tee -a "${RUN_DIR}/quick_metrics.log"
  fi

  if python - <<PY
import math
import sys

trades = float("${TRADES}")
pf = float("${PF}")
maxdd = float("${MAXDD}")
sharpe = float("${SHARPE}")
ok = (
    not math.isnan(trades) and trades >= ${MIN_TRADES}
    and not math.isnan(pf) and pf >= ${MIN_PF}
    and not math.isnan(maxdd) and maxdd <= ${MAX_DD}
    and (math.isnan(sharpe) or sharpe > ${MIN_SHARPE})
)
sys.exit(0 if ok else 1)
PY
  then
    :
  else
    echo "[SKIP] ${T} fails gate -> no sweep." | tee -a "${RUN_DIR}/quick_metrics.log"
    continue
  fi

  echo "[SWEEP] ${T} passed gate -> running 2D threshold sweep..." | tee -a "${RUN_DIR}/quick_metrics.log"

  for long in "${LONGS[@]}"; do
    for short in "${SHORTS[@]}"; do
      if python - <<PY
l = float("$long")
s = float("$short")
raise SystemExit(0 if l > s else 1)
PY
      then
        :
      else
        continue
      fi

      TAG="L${long}_S${short}"
      python scripts/run_backtest.py \
        --config "configs/discovery/${T}-1h_v3_platt.yaml" \
        --start "${START}" --end "${END}" \
        --long-threshold "${long}" --short-threshold "${short}" \
        --engine native --fast \
        --export-trades "${RUN_DIR}/trades_${TAG}.csv" \
        | tee "${RUN_DIR}/metrics_${TAG}.log"

      if [ -s "${RUN_DIR}/trades_${TAG}.csv" ]; then
        python scripts/analyze_trades.py \
          --trades "${RUN_DIR}/trades_${TAG}.csv" \
          > "${RUN_DIR}/analysis_${TAG}.txt"
      else
        echo "[WARN] trades file missing for ${TAG} -> skipping analyze_trades." | tee -a "${RUN_DIR}/metrics_${TAG}.log"
      fi
    done
  done
done
