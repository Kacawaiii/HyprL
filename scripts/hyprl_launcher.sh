#!/usr/bin/env bash
# HyprL launcher helper for Linux/macOS. Provides quick access to GUIs and core CLIs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  cat <<'EOF'
Environnement virtuel manquant (.venv). Créez-le avant d'utiliser le launcher :
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
EOF
  exit 1
fi

# shellcheck disable=SC1090
source "$ROOT_DIR/.venv/bin/activate"

prompt_with_default() {
  local label="$1"
  local default="$2"
  local value
  read -r -p "$label [$default]: " value
  if [[ -z "$value" ]]; then
    echo "$default"
  else
    echo "$value"
  fi
}

confirm_yes() {
  local label="$1"
  local default="$2"
  local value
  read -r -p "$label [$default]: " value
  value="${value:-$default}"
  case "${value,,}" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

run_command() {
  echo ">>> $*"
  (cd "$ROOT_DIR" && "$@")
}

launch_gui() {
  run_command streamlit run scripts/hyprl_gui.py
}

launch_replay_gui() {
  run_command streamlit run scripts/hyprl_replay_gui.py
}

run_analysis() {
  local ticker period interval model calibration
  ticker="$(prompt_with_default "Ticker" "AAPL")"
  period="$(prompt_with_default "Période (yfinance format)" "5d")"
  interval="$(prompt_with_default "Intervalle" "1h")"
  model="$(prompt_with_default "Model type (logistic/random_forest/xgboost)" "logistic")"
  calibration="$(prompt_with_default "Calibration (none/platt/isotonic)" "none")"

  local cmd=(python scripts/run_analysis.py --ticker "$ticker" --period "$period")
  [[ -n "$interval" ]] && cmd+=(--interval "$interval")
  [[ -n "$model" && "$model" != "logistic" ]] && cmd+=(--model-type "$model")
  if [[ -n "$calibration" && "$calibration" != "none" ]]; then
    cmd+=(--calibration "$calibration")
  fi
  run_command "${cmd[@]}"
}

run_backtest() {
  local ticker period interval initial_balance seed long_th short_th adaptive export_csv model calibration
  ticker="$(prompt_with_default "Ticker" "AAPL")"
  period="$(prompt_with_default "Période (ex: 1y, 6mo)" "1y")"
  interval="$(prompt_with_default "Intervalle" "1h")"
  initial_balance="$(prompt_with_default "Capital initial" "10000")"
  seed="$(prompt_with_default "Seed" "42")"
  long_th="$(prompt_with_default "Long threshold" "0.55")"
  short_th="$(prompt_with_default "Short threshold" "0.40")"
  model="$(prompt_with_default "Model type (logistic/random_forest/xgboost)" "logistic")"
  calibration="$(prompt_with_default "Calibration (none/platt/isotonic)" "platt")"
  if confirm_yes "Activer le mode adaptatif ? (y/n)" "n"; then
    adaptive="yes"
  else
    adaptive="no"
  fi
  if confirm_yes "Exporter les trades en CSV ? (y/n)" "y"; then
    export_csv="$(prompt_with_default "Chemin CSV" "data/trades_${ticker}_${period}.csv")"
  else
    export_csv=""
  fi

  local cmd=(python scripts/run_backtest.py
    --ticker "$ticker"
    --period "$period"
    --initial-balance "$initial_balance"
    --seed "$seed"
    --long-threshold "$long_th"
    --short-threshold "$short_th")

  [[ -n "$interval" ]] && cmd+=(--interval "$interval")
  [[ -n "$model" && "$model" != "logistic" ]] && cmd+=(--model-type "$model")
  if [[ -n "$calibration" && "$calibration" != "none" ]]; then
    cmd+=(--calibration "$calibration")
  fi
  [[ "$adaptive" == "yes" ]] && cmd+=(--adaptive)
  [[ -n "$export_csv" ]] && cmd+=(--export-trades "$export_csv")

  run_command "${cmd[@]}"
}

run_tests() {
  run_command pytest -q
}

run_custom() {
  local custom_cmd
  read -r -p "Commande (ex: python scripts/run_threshold_sweep.py --ticker AAPL): " custom_cmd
  if [[ -z "$custom_cmd" ]]; then
    echo "Commande vide, retour au menu."
    return
  fi
  run_command bash -lc "$custom_cmd"
}

show_menu() {
  cat <<'EOF'
HyprL Launcher
--------------
1) Lancer GUI principal (Streamlit)
2) Lancer GUI Replay
3) Run Analysis (CLI)
4) Run Backtest (CLI)
5) Lancer pytest -q
6) Commande personnalisée
0) Quitter
EOF
}

while true; do
  show_menu
  read -r -p "Choix: " choice
  case "$choice" in
    1) launch_gui ;;
    2) launch_replay_gui ;;
    3) run_analysis ;;
    4) run_backtest ;;
    5) run_tests ;;
    6) run_custom ;;
    0) exit 0 ;;
    *) echo "Choix invalide."; sleep 0.5 ;;
  esac
done
