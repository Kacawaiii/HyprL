#!/usr/bin/env bash
# HyprL shadow trading control. No real money, no broker, no exchange key.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m scripts.trading_lab.paper_shadow_cli "$@"
