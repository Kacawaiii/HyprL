# HyprL local operations (Phase 5E)

Running HyprL used to mean knowing that the API is a Python module, that the
cockpit is a Vite project, that both must be running, and which port each
expects. This phase replaces that with one command and one URL.

No real money, no broker, no exchange API key. Nothing in this document
changes a trading contract.

## Running it

```
./scripts/hyprl.sh start        # build if needed, serve, open the browser
./scripts/hyprl.sh status
./scripts/hyprl.sh stop
./scripts/hyprl.sh restart
./scripts/hyprl.sh doctor       # offline diagnosis, changes nothing
./scripts/hyprl.sh logs --limit 200
./scripts/hyprl.sh build        # production frontend only
```

Shadow trading keeps its own verbs and its Phase 5D behaviour:

```
./scripts/hyprl.sh paper start|stop|restart|status
```

Runtime archives and diagnostics:

```
./scripts/hyprl.sh export <archive.zip> [--include-logs]
./scripts/hyprl.sh verify-export <archive.zip>
./scripts/hyprl.sh import <archive.zip> --destination <empty-dir>
./scripts/hyprl.sh support-bundle <bundle.json>
./scripts/hyprl.sh settings [--set field=value]
```

Development is unchanged: `./scripts/dev_app.sh` still runs Vite with hot
reload on 5173 against the API on 8787.

## Single origin

In production one server answers everything on `127.0.0.1:8787`:

| Path        | Served by                                        |
|-------------|--------------------------------------------------|
| `/api/v1/*` | the read-only application API                    |
| `/*`        | the frontend build, with SPA fallback            |

API paths are matched **before** the static layer and never fall through to
`index.html`. A mistyped endpoint returning a page of HTML with a 200 would
send every caller looking for the bug in the wrong place.

The bind address is `127.0.0.1` and there is no configuration that changes
that by default. A non-loopback bind must be passed explicitly and prints a
warning. CORS keeps its explicit dev-origin allowlist; production needs no
cross-origin rule at all because there is only one origin.

## Static file safety

The dist root is resolved once at startup. Every request is decoded **exactly
once** and the resulting real path must live under that root, or it is
refused with a 403.

Decoding once is the point: `%2e%2e` becomes `..` and is caught by the
containment check, while `%252e%252e` becomes the literal text `%2e%2e`,
which is a filename that does not exist. Decoding twice is what would create
the vulnerability.

Also refused: absolute paths (joined relative to the root, never used as a
root), backslashes, null bytes, and symlinks that resolve outside the build.
Nothing outside `apps/web/dist` is reachable — not the repository, not
`.git`, not `var/`, not `data/`.

A missing **asset** returns 404. A missing **route** returns the document, so
deep links survive a reload.

## Cache policy

| Response                      | Cache-Control                          |
|-------------------------------|----------------------------------------|
| Hashed assets (`/assets/…`)   | `public, max-age=31536000, immutable`  |
| `index.html`                  | `no-cache`                             |
| Other files in dist           | `public, max-age=300`                  |
| Every API response            | `no-store`                             |

Vite writes content-hashed filenames, so a changed file is a new name and can
be cached indefinitely. The entry document is not hashed and must be
revalidated, or a rebuilt app is invisible to a browser that already has it.
API responses describe a runtime that changes; a cached copy is a lie.

## Process lifecycle

A PID file is a claim, not a fact. Before any signal is sent, three things
must hold:

1. the process exists (a zombie does not count — `kill(pid, 0)` succeeds on
   one, and treating that as alive makes stop wait out its whole grace period
   and then SIGKILL a process that already exited);
2. its start time from `/proc` matches what was recorded — this is what
   actually defeats PID reuse;
3. its command line still carries the `hyprl-local-app` marker.

Any mismatch means the file is stale or the PID now belongs to somebody else.
It is removed and reported, never signalled. **HyprL will not terminate a
process it cannot prove is its own.**

Stop sends SIGTERM to the application's own process group, waits, re-verifies
identity, and only then escalates to SIGKILL. `killpg(0)` and `kill(-1)` are
explicitly refused. Children are reaped, so no zombies accumulate.

Starting twice is a no-op. A port held by something else refuses the start
rather than racing it.

## Runtime directory

Everything written at runtime lives under `var/trading_lab/`, which is git
ignored:

```
var/trading_lab/
  paper_v1.sqlite      the shadow event log (Phase 5D location, unchanged)
  paper_session.json   the active session marker
  runtime/             ops.sqlite, settings.json, pid file, lifecycle marker
  logs/                hyprl.jsonl and its rotated generations
  exports/  support/  tmp/
```

The event log keeps its Phase 5D path: migrating a hash-chained audit
database for a tidier tree is not a trade worth making. Directories are
created `0700` and files `0600`. Nothing here is secret today — there are no
keys anywhere in this project — but a runtime directory is exactly where a
future secret would land.

## Logging

JSONL, one object per line: `timestamp`, `level`, `component`, `event`,
`message`, and optionally `session_id`, `product`, `error_code`, `context`.

Redaction is **structural**. Keys matching `authorization`, `cookie`,
`api_key`, `token`, `secret`, `password`, `passphrase`, `credential`,
`bearer`, `signature` and friends are replaced at any nesting depth, and
home-directory paths are rewritten to `<home>` before anything is written.
There is no way for a caller to opt out. This holds even though no such value
exists in the project today — relying on that would mean the first component
to acquire one also acquires a leak.

Rotation is size-based: 10 MiB per file, 5 generations, a 50 MiB ceiling.
The `log_retention_preset` setting picks 2, 5 or 10 generations.

## Health

One vocabulary: `HEALTHY`, `DEGRADED`, `ERROR`, `STOPPED`, `EMBARGOED`.

| Situation                | State       | Why |
|--------------------------|-------------|-----|
| Research embargo active  | `EMBARGOED` | the guard working is not a failure |
| Network unreachable      | `DEGRADED`  | says nothing about recorded data |
| Hash chain broken        | `ERROR`     | the audit trail is the product |

Components: `app_api`, `paper_engine`, `event_store`, `market_ingestion`,
`model`, `holdout_guard`. History lives in `runtime/ops.sqlite`, capped at
10 000 records — telemetry may be pruned, which is precisely why it does not
share a file with the append-only paper log.

## Doctor

Offline and read-only. It never modifies and never opens a socket: a
diagnostic that talks to an exchange fails when the exchange is down, blames
the local machine, and fetches market data outside the engine's guard rails.
It may read the *specification* of the protected window; it has no code path
that could read one candle inside it.

Exit codes: `0` all passed, `1` warnings only, `2` at least one failure.
`--ignore-warnings` collapses 1 into 0 for scripts.

A core install without the optional `[ml]` extra still runs every check that
does not need it, including the holdout check.

## Recovery

Two separate facts, reported separately:

* **`last_shutdown_clean`** — an unclean shutdown is ordinary. A laptop lid, a
  SIGKILL, a power cut. WAL plus an append-only log is built for it.
* **`event_chain_verified`** — a broken chain is not ordinary.

Successful recovery is stated calmly, not as an alarm. Painting it red trains
people to dismiss red. Repair is deliberately absent: a tool that silently
"fixes" a hash chain produces a log that verifies and means nothing.

While the app is running, `last_shutdown_clean` refers to the shutdown
*before* this run — otherwise a healthy app would show a recovery banner for
its entire uptime.

## Snapshot monitoring

The Phase 5D live smoke found a snapshot trigger that could never fire, and
nothing noticed because nothing was watching. Now `events_since_last_snapshot`
and `snapshot_due` are exposed per product, and falling more than two
intervals behind is `DEGRADED` — never corruption, because a missing snapshot
costs replay time on restart, not integrity.

The measure is **per product**. A global event-id difference made whichever
product stopped receiving candles first look thousands of events behind while
nothing was wrong.

## Settings

Allowed: `theme`, `sidebar_collapsed`, `default_product`,
`default_chart_window`, `time_display`, `log_retention_preset`,
`launch_browser`, `paper_auto_start`.

Refused, by name and with an explanation: every signal threshold, risk cap,
fee, slippage rate, model hyper-parameter, feature set, execution policy and
holdout date. `SignalSpec V1`, `RiskSpec V1`, `ExecutionSpec V1`,
`PaperModelSpec V1` and the protected research window are immutable in this
build, and their hashes appear in committed results — the moment one becomes
a setting, every recorded result stops describing the software that ran.

Unknown fields are refused rather than ignored: a key that silently did
nothing would be believed to have taken effect. A corrupt settings file falls
back to defaults with a warning and cannot reach a trading contract, because
trading contracts are not represented in this file at all.

## Export and import

Export uses SQLite's backup API, not a file copy. With WAL enabled a plain
copy is a torn snapshot — committed transactions live in `-wal` until a
checkpoint, so the copy can open cleanly and be missing the last hour.

An archive contains the database, a `manifest.json` and `SHA256SUMS`. The
manifest's identity is its **content hash**, which excludes `created_at`: two
exports of the same runtime state must agree, and a wall clock in the identity
would make that impossible. Logs are excluded unless asked for.

Import treats every archive as hostile, including our own. Refused before
anything is written: absolute paths, `..` in any segment, backslashes,
symlinks, non-regular files, duplicate names, more than 10 000 members, and
anything expanding past 2 GiB.

**Import never targets the live runtime.** Merging an exported session into a
running one means reconciling two hash chains, and there is no honest way to
do that — the result would verify while describing a history that never
happened. Import writes to a separate empty directory and refuses to
overwrite.

## Support bundle

Built from an allowlist, never by walking a structure looking for things to
keep. A denylist fails the first time a new field appears, silently and
permanently, because nobody re-reads a file they already trust.

Included: version and commit, capabilities, spec hashes, health summary,
recent error **codes**, chain status, storage sizes, build metadata, embargo
status.

Excluded and tested for: market history, predictions, fills, hostname,
username, home directory, environment variables, and absolute paths of any
kind.

## Operations API

`GET /api/v1/ops/health-history`, `/ops/runtime`, `/ops/recovery`,
`/ops/storage`, `/ops/settings`. All bounded, all read-only.

There is no `POST /restart`, no `POST /paper/start`, no delete. Lifecycle
stays on the command line: a cockpit that can start a trading process is one
cross-site request away from doing it without being asked.

## Data retention

Paper events, fills and predictions are **never** pruned automatically. They
are the audit trail. A large database produces a warning suggesting export and
archival — deleting evidence to save disk is not a trade this project makes.
Logs and health history are bounded, because they are telemetry.
