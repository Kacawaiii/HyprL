#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
CRATE_DIR="$ROOT_DIR/native/hyprl_supercalc"

if [ ! -d "$VENV_PATH" ]; then
  echo "Environnement virtuel introuvable (${VENV_PATH})." >&2
  echo "Créez-le via 'python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt'." >&2
  exit 1
fi

if [ ! -d "$CRATE_DIR" ]; then
  echo "Crate Rust absente (${CRATE_DIR})." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$CRATE_DIR"
if ! command -v maturin >/dev/null 2>&1; then
  echo "maturin n'est pas installé dans la venv. Lancez 'pip install maturin'." >&2
  exit 1
fi
maturin develop --release
