# Local supervision v1

This slice reuses the existing application server, isolated `platform.jobs.JobRunner`, EDGAR
service and offline EDGAR closure. It operates only in the current worktree's ignored `var/`.
It does not manage system services or other users' processes. No capture, real training, remote
model call or broker is enabled by the default commands.

## Private configuration

Create an operator-owned JSON file **outside the checkout**, with mode `0600`. This is a documented
template, not an install command. Replace the placeholders locally; never commit the resulting file.

```json
{
  "schema": "hyprl-ops-v1",
  "runtime_root": "var/trading_lab/ops-local",
  "port": 8790,
  "fomc_store": "<operator-supplied read-only FOMC archive directory>",
  "edgar_archive": "<operator-supplied read-only EDGAR closure directory>",
  "research_root": "<optional private research demo registry directory>"
}
```

The three read-only roots are optional and must be separate from the operations runtime.
Unknown keys, arbitrary executables, non-local runtime roots and symlinks escaping `var/` are
refused. Configuration values never appear in API responses or the controller's diagnostics.
The listener is always `127.0.0.1`; configuration cannot enable a public bind.

With the repository's Python environment active, set `HYPRL_OPS_CONFIG` to that private file:

```bash
python -m scripts.ops.hyprl_ops verify --config "$HYPRL_OPS_CONFIG"
python -m scripts.ops.hyprl_ops start --config "$HYPRL_OPS_CONFIG"
python -m scripts.ops.hyprl_ops stop --config "$HYPRL_OPS_CONFIG"
python -m scripts.ops.hyprl_ops resume --config "$HYPRL_OPS_CONFIG"
```

`start`/`resume all` starts one worker supervisor and the read-only API, in separate sessions.
`--service app` and `--service workers` control them independently. This operations listener
does not enable Model Lab POST controls; the existing authenticated lab/B2B services retain their
own admission surfaces. A developer can enqueue synthetic workloads using `JobStore(runtime/lab)`.
Use one runner per lab root; its existing locks fence both a second runner and orphan workers.
Synthetic workers retain their existing CPU, memory, wall, queue and lifetime budgets.

The process record binds UID, Linux boot identity, start ticks and exact argv. Stop pins a Linux
pidfd and rechecks identity before sending SIGTERM to the owner. It never signals an arbitrary
process group or escalates to SIGKILL. A reused/foreign PID, unavailable pidfd, or incomplete
45-second drain is BLOCKED. EDGAR keeps its in-flight completion and persistence deadlines.
Repeated start is idempotent. Queued jobs survive worker stops; interrupted running jobs retain
their existing FAILED/interruption state and require an explicit new job.

## Read-only health API

`GET` and `HEAD /api/v1/ops/health` return `hyprl-ops-health-v1`. Other verbs return 405;
query parameters are refused. A normally started app can also read fixed telemetry with
`--ops-root var/...`. Server SHA, implementation digest and spec hashes are captured at startup;
later worktree commits do not relabel the running process.

The response contains:

| Field | Meaning |
|---|---|
| `running_versions` | API process Git SHA, implementation digest, API version and verified FOMC/EDGAR spec bindings |
| `services` | Configured process identities, state, start instant, running version records and RSS |
| `sources` | Read-only source state, own horizon, last durable operation, attested cutoff, age and counts |
| `last_operations` | Last 20 commands, terminal state and stable error codes; private journal retains 100 |
| `errors` | Stable workload, identity, telemetry and storage diagnostics |
| `workers` | Running isolated worker PIDs, progress, resource limits, queue, lifetime job and artifact budgets |
| `edgar_service` | Identity-bound heartbeat (5-second freshness bound), suspended grants, incidents and original-store authorization budget/expiry |
| `resources` | API process RSS/CPU/threads and free bytes on the configured runtime volume |

NOT_CONFIGURED, NOT_OBSERVED, STALE, FOREIGN and INTEGRITY_ERROR stay explicit. OBSERVED is a
telemetry observation, not a live-provider availability guarantee. Archive age measures elapsed
time from its own attestation; it never implies that a new source check occurred. Missing telemetry
does not become zero errors or fresh data. Reads use the shared private DB/WAL reader, including
for worker state, so opening a read-only database cannot create archive side files. Historical
job failures remain visible. No raw body, contact, configuration, command line or log is returned.

`verify` checks spec bindings, configured source replay and readable telemetry without a lock, mkdir, request,
training or runtime mutation. It returns BLOCKED and exit 2 for observed errors; absent services
remain STOPPED/unknown in its payload. This is a preflight, not a substitute for tests, source
replay qualification or a scientific verdict.

## Backup and restore

Stop the configured services first. Targets stay in this worktree's `var/` and must be new,
separate directories. External archives, configuration and logs are excluded.

```bash
python -m scripts.ops.hyprl_ops backup --config "$HYPRL_OPS_CONFIG" --target var/trading_lab/backup-001
# Use a second private config naming a fresh runtime root. No existing evidence is overwritten.
python -m scripts.ops.hyprl_ops restore --config "$HYPRL_OPS_RESTORE_CONFIG" --target var/trading_lab/backup-001
python -m scripts.ops.hyprl_ops resume --config "$HYPRL_OPS_RESTORE_CONFIG"
```

Backup fences independent runner/execution locks, copies SQLite through a consistent read-only
backup (including WAL), refuses symlinks and enforces 10,000-file/512-MiB limits. A manifest binds
every file's name, size and digest. Restore checks the complete inventory before its first data
write; changed, extra or out-of-scope files fail closed. Partial failed targets are preserved and
must not be reused. Store schemas/specs are never migrated.

When a writable, authorized EDGAR runtime exists, backup reuses `closure._owner_guard`,
`consistent_copy` and `verify_copy`: raw digests, reopen/replay, health and qualification must pass.
Its copy is restored as **edgar-evidence**, never as the original accounting store. Resume cannot
rewind the authorization budget by restoring a backup. Restore is for app/workers and offline
EDGAR evidence; resuming capture requires the original durable store.

## Future operator-authorized EDGAR activation

This campaign does not execute these commands. The controller requires a separately named
`--service edgar --allow-capture`, a private `edgar_authorization` path under the operator's
`~/authorizations/`, and the **original** accounting store at `runtime_root/edgar`. The underlying
service rechecks scope, spec, CIKs, expiry, consumed reservations and throttle termination at
startup and effective dispatch. Default `all` never launches EDGAR. New stores must be provisioned
under an operator's scoped authorization outside this slice. Never copy a grant to another store
or change its contents to reset spend; global accounting across independent stores remains an
operator-controlled limitation of the existing service.

```bash
python -m scripts.ops.hyprl_ops start --config "$HYPRL_OPS_CONFIG" --service edgar --allow-capture
python -m scripts.ops.hyprl_ops stop --config "$HYPRL_OPS_CONFIG" --service edgar
python -m scripts.ops.hyprl_ops resume --config "$HYPRL_OPS_CONFIG" --service edgar --allow-capture
```

Offline qualification: `python -m pytest tests/ops_supervision -q`. Existing supervised EDGAR
dispatch/deadline/storage/closure tests remain authoritative. No real restart, new capture,
production alert delivery or cross-store budget qualification is claimed by these tests.
