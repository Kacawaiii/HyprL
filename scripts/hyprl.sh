#!/usr/bin/env bash
# HyprL local application. No real money, no broker, no exchange API key.
#
# One entry point: start, stop, restart, status, doctor, logs, build,
# paper <start|stop|restart|status>, export, verify-export, import,
# support-bundle, settings.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m scripts.trading_lab.hyprl_cli "$@"
