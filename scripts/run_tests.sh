#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Environnement virtuel manquant: $VENV_PATH" >&2
  echo "Créez-le via 'python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .'." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

if [ $# -eq 0 ]; then
  set -- pytest -q
fi

exec "$@"
