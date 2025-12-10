#!/usr/bin/env bash
set -euo pipefail

# Resolve WEBHOOK from env (.env.ops) if not passed explicitly
WEBHOOK="${WEBHOOK:-${OPS_ALERT_WEBHOOK:-${SLACK_WEBHOOK_URL:-}}}"
if [[ -z "${WEBHOOK}" ]]; then
  echo "[ERROR] No WEBHOOK/OPS_ALERT_WEBHOOK set" >&2
  exit 1
fi
export WEBHOOK

# Palier 2 hourly monitor + alert chain (run after hourly runners).
# Usage:
#   SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..." ./scripts/ops/palier2_hourly.sh

ROOT="${ROOT:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-live/logs}"
HEALTH_PATH="${HEALTH_PATH:-${LOG_ROOT}/portfolio_live/health_asc_v2.json}"
PF_ALERT="${PF_ALERT:-1.3}"
DD_ALERT="${DD_ALERT:-20}"
SHARPE_ALERT="${SHARPE_ALERT:-1.5}"
HEARTBEAT_SYMBOLS="${HEARTBEAT_SYMBOLS:-NVDA,MSFT,AMD,META,QQQ}"
MAX_AGE_MIN="${MAX_AGE_MIN:-90}"

cd "${ROOT}"

echo "[palier2_hourly] monitor portfolio -> ${HEALTH_PATH}"
python scripts/ops/run_portfolio_monitor_live.py --log-root "${LOG_ROOT}" --summary-out "${HEALTH_PATH}"

echo "[palier2_hourly] alert portfolio health"
if [[ -n "${WEBHOOK}" ]]; then
  python scripts/ops/alert_portfolio_health.py \
    --health "${HEALTH_PATH}" \
    --pf-alert "${PF_ALERT}" \
    --dd-alert "${DD_ALERT}" \
    --sharpe-alert "${SHARPE_ALERT}" \
    --webhook "${WEBHOOK}"
else
  python scripts/ops/alert_portfolio_health.py \
    --health "${HEALTH_PATH}" \
    --pf-alert "${PF_ALERT}" \
    --dd-alert "${DD_ALERT}" \
    --sharpe-alert "${SHARPE_ALERT}"
fi

echo "[palier2_hourly] check heartbeats"
if [[ -n "${WEBHOOK}" ]]; then
  python scripts/ops/check_heartbeat.py \
    --root "${LOG_ROOT}" \
    --max-age-min "${MAX_AGE_MIN}" \
    --symbols "${HEARTBEAT_SYMBOLS}" \
    --webhook "${WEBHOOK}"
else
  python scripts/ops/check_heartbeat.py \
    --root "${LOG_ROOT}" \
    --max-age-min "${MAX_AGE_MIN}" \
    --symbols "${HEARTBEAT_SYMBOLS}"
fi

echo "[palier2_hourly] done"
