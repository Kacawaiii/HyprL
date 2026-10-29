# Daily v2 shadow inputs

`scripts.agent_v2_refresh.daily` is a data-only sidecar. It reuses the activated
engine's registry selection, causal radar projection, producer validators and
planner. The engine folders, specs, runtime binding and Tuesday evidence remain
unchanged. It never dispatches a model or a broker order and never reruns a missed
decision cycle.

The timer runs at 10:15 UTC on weekdays. The sidecar checks the NYSE calendar
before any request and stops acquisition at 12:25 UTC. Its failure unit sends a
fixed alert through the operator notifier. An explicit `--repair-today` run can
perform a current-time planner preflight after that window; its receipts retain
their actual timestamps and cannot establish that the missed cycle would have
accepted information unavailable at its original cutoff.

Calendar acquisition retains the existing one-pass scope, response bounds,
receipts, source hashes and conservative civil-date markers. A complete annual
FOMC schedule is required for every year in the rolling evaluation horizon. An
unparseable schedule or missing BLS weekday cell blocks publication. Missing
earnings dates still exclude instruments; no absence or release hour is invented.

Registry requests use the full activated journal for per-day source budgets,
spacing, halt records and append-only reservations. Only the builder's duplicate
lookup is narrowed to the current UTC day, permitting a new day's legitimate
submissions request without retrying a failed request on the same day. No budget
or journal is reset or migrated.

Each date owns one private output directory, reserved before acquisition. A failed
attempt is not retried automatically. All four inputs and the real producer and
planner must pass before the complete generation is published through one atomic
`current-config.json` replacement. Cycle/watch units can use that pointer while
continuing to execute the exact activated engine checkout. Existing stale-input
guards remain authoritative if refresh fails.

Run synthetic validation with:

```sh
python -m pytest tests/agent_v2_quality/test_daily_refresh.py -q
```

Settings identify the activated runtime/code SHA, pinned sidecar SHA, original
base config and registry, calendar grant, current radar report directory, private
output root, and byte hashes of unchanged input grants/seed files. Keep settings,
receipts, bodies, keys and generated inputs outside Git. Deployment must preserve
the cycle/watch engine directory and commands, changing only their config path.
