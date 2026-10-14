#!/bin/bash -l
# Synthetic tomorrow decisions/labels in a fresh private runtime. Paper probes
# inspect today's broker with GET only; synthetic time never simulates fills.
set -euo pipefail
umask 077
drill_day=${1:-$(date -u -d tomorrow +%F)}
[[ "$drill_day" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || exit 2
drill_root="$HOME/drill/$drill_day"
mkdir -p "$drill_root"
# Keep each invocation separate; never delete or copy a production journal.
drill_run=$(mktemp -d "$drill_root/synthetic-XXXXXXXX")
drill_repo=$(systemctl --user show hyprl-trader-run.service -p WorkingDirectory --value)
cd "$drill_repo"
drill_python="$HOME/venv-gate/bin/python"
drill_module=scripts.trading_lab.trader_agent.cli
drill_common=(--authorization "$HOME/authorizations/trader-agent-v1.json" --runtime "$drill_run/runtime" --dry-run)
drill_paper=(--authorization "$HOME/authorizations/trader-agent-v1.json" --runtime "$HOME/trader-agent-v1"
  --paper-authorization "$HOME/authorizations/trader-alpaca-paper-v1.json"
  --data-authorization "$HOME/authorizations/trader-alpaca-data-v1.json" --dry-run)
drill_failures=()
printf '| step | rc | time | verified top-level state and evidence |\n|---|---|---|---|\n' > "$drill_run/summary.md"
step() {
  local name=$1 kind=$2 started rc=0 summary
  shift 2
  started=$(date +%s)
  timeout 600 "$@" > "$drill_run/$name.out" 2>&1 || rc=$?
  if (( rc == 0 )); then
    summary=$("$drill_python" -m scripts.trading_lab.trader_agent.drill_summary "$kind" "$drill_run/$name.out") || rc=$?
  else
    summary=COMMAND_FAILED
  fi
  printf '| %s | %s | %ss | %s |\n' "$name" "$rc" "$(( $(date +%s) - started ))" "$summary" >> "$drill_run/summary.md"
  if (( rc != 0 )); then drill_failures+=("$name"); fi
}
step run-am run "$drill_python" -m "$drill_module" run "${drill_common[@]}" --defer-synthetic-labels --synthetic-time "${drill_day}T12:00:00Z"
step recover-am recover "$drill_python" -m "$drill_module" recover "${drill_common[@]}" --synthetic-time "${drill_day}T12:30:00Z"
step run-pm run "$drill_python" -m "$drill_module" run "${drill_common[@]}" --slot pm --defer-synthetic-labels --synthetic-time "${drill_day}T16:00:00Z"
step recover-pm recover "$drill_python" -m "$drill_module" recover "${drill_common[@]}" --slot pm --synthetic-time "${drill_day}T16:30:00Z"
step label label "$drill_python" -m "$drill_module" label "${drill_common[@]}" --synthetic-time "${drill_day}T21:30:00Z"
drill_matured=$(date -u -d "$drill_day + 10 days" +%F)
step label-matured matured "$drill_python" -m "$drill_module" label "${drill_common[@]}" --synthetic-time "${drill_matured}T21:30:00Z"
step current-execute-crypto paper "$drill_python" -m "$drill_module" paper-execute "${drill_paper[@]}" --paper-account ia_crypto
step current-execute-equity paper "$drill_python" -m "$drill_module" paper-execute "${drill_paper[@]}" --paper-account ia_actions
step current-exit-crypto paper "$drill_python" -m "$drill_module" paper-exit "${drill_paper[@]}" --paper-account ia_crypto
step current-exit-equity paper "$drill_python" -m "$drill_module" paper-exit "${drill_paper[@]}" --paper-account ia_actions
step current-paper-audit paper "$drill_python" -m "$drill_module" paper-audit "${drill_paper[@]}"
{
  if (( ${#drill_failures[@]} == 0 )); then printf 'STATUS: DONE\n'; else printf 'STATUS: BLOCKED\n'; fi
  printf '\n# Drill for %s (generated %s)\n\n' "$drill_day" "$(date -u +%FT%TZ)"
  printf 'Tomorrow: native-shape synthetic prices, deterministic analyst/reviewer fixtures, pinned calendar, isolated budgets and research journal. No production runtime copy.\n\n'
  printf 'Label checks cover tomorrow\047s matured equities and a separate future checkpoint for all remaining horizons.\n\n'
  printf 'Paper compatibility: current broker/journal, GET only, zero production writes. Future acceptance, fills and tomorrow\047s paper decisions remain unproven.\n\n'
  cat "$drill_run/summary.md"
} > "$drill_run/report.md"
cp "$drill_run/report.md" "$HOME/reports/drill-$drill_day.md"
if (( ${#drill_failures[@]} > 0 )) && [[ ${HYPRL_DRILL_NOTIFY:-1} == 1 ]]; then
  # Preserve the existing unattended failure notification. A manual proof can
  # disable delivery without changing verification or its failure exit status.
  "$drill_python" - "$HOME/private/telegram.env" "$drill_day" "${drill_failures[*]}" <<'PY'
import sys
from scripts.trading_lab.trader_agent.rehearsal import telegram
delivery = telegram(sys.argv[1])('nightly-drill:' + sys.argv[2],
    'HyprL nightly drill ' + sys.argv[2] + ' failed: ' + sys.argv[3] + '. See the private daily drill report.')
print('drill failure notification: ' + delivery)
PY
fi
printf 'drill %s failures: %s\n' "$drill_day" "${drill_failures[*]:-none}"
(( ${#drill_failures[@]} == 0 ))
