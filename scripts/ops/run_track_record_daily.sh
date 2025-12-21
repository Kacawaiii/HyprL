#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env.broker.alpaca ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.broker.alpaca
  set +a
fi

ASOF_DATE="$(date -u +%F)"
OUT_DIR="docs/reports/track_record"

.venv/bin/python scripts/ops/alpaca_track_record_snapshot.py \
  --paper \
  --out-dir "$OUT_DIR" \
  --asof-date "$ASOF_DATE"

.venv/bin/python scripts/ops/make_track_record_report.py \
  --in-dir "$OUT_DIR" \
  --out-dir "$OUT_DIR"
