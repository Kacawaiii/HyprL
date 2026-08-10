#!/usr/bin/env bash
# Start the HyprL cockpit: read-only API on loopback + Vite dev server.
#
# Both children are killed on exit, including on Ctrl-C, so a stopped session
# never leaves a port held by an orphan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${HYPRL_API_PORT:-8787}"
DATA_ROOT="${HYPRL_DATA_ROOT:-$ROOT/data/crypto}"

# `npm run dev` spawns vite as a grandchild, so signalling only the PID we
# hold is not guaranteed to reach the server that actually holds the port.
# Each child therefore runs in its own process group and is signalled as a
# group (negative PID).
stop_group() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  stop_group "${API_PID:-}"
  stop_group "${WEB_PID:-}"
  sleep 1
  kill -KILL -- "-${API_PID:-0}" 2>/dev/null || true
  kill -KILL -- "-${WEB_PID:-0}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [[ ! -d "$ROOT/apps/web/node_modules" ]]; then
  echo "[hyprl] installing frontend packages from the lockfile…"
  (cd "$ROOT/apps/web" && npm ci --no-audit --no-fund)
fi

echo "[hyprl] API    http://127.0.0.1:$API_PORT/api/v1/health"
setsid python -m scripts.trading_lab.app_api.server \
  --data-root "$DATA_ROOT" --port "$API_PORT" &
API_PID=$!

echo "[hyprl] cockpit http://127.0.0.1:5173"
(cd "$ROOT/apps/web" && setsid npm run dev -- --port 5173) &
WEB_PID=$!

wait -n "$API_PID" "$WEB_PID"
