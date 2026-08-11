#!/usr/bin/env bash
# Build a reproducible local release bundle into dist/hyprl-local/.
# No real money, no broker, no exchange API key.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m scripts.trading_lab.hyprl_cli release "$@"
