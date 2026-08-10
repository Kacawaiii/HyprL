#!/usr/bin/env bash
# Start the HyprL cockpit: read-only API on loopback + Vite dev server.
#
# Both children are killed on exit, including on Ctrl-C, so a stopped session
# never leaves a port held by an orphan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${HYPRL_API_PORT:-8787}"
DATA_ROOT="${HYPRL_DATA_ROOT:-$ROOT/data/crypto}"

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [[ ! -d "$ROOT/apps/web/node_modules" ]]; then
  echo "[hyprl] installing frontend packages from the lockfile…"
  (cd "$ROOT/apps/web" && npm ci --no-audit --no-fund)
fi

echo "[hyprl] API    http://127.0.0.1:$API_PORT/api/v1/health"
(cd "$ROOT" && python -m scripts.trading_lab.app_api.server \
   --data-root "$DATA_ROOT" --port "$API_PORT") &
API_PID=$!

echo "[hyprl] cockpit http://127.0.0.1:5173"
(cd "$ROOT/apps/web" && npm run dev -- --port 5173) &
WEB_PID=$!

wait -n "$API_PID" "$WEB_PID"
