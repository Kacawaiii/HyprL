# Official event sources — entry point

What was officially known at instant T, as far as a store had committed by horizon H, the same answer every
time, with its provenance, and an explicit "not known" when it is not. Two sources today: Federal Reserve FOMC
statements and SEC EDGAR 8-K submissions. Binding rules for agents: `AGENTS.md`. Acceptance criteria and their
executable check: `docs/MVP_ACCEPTANCE.md` (PROPOSED until the operator accepts it). Per-source detail and the
invariant → code → test tables: `docs/FOMC_V1_OFFLINE_SLICE.md`, `docs/EDGAR_V1_OFFLINE_SLICE.md`.

## Architecture

```
official source ──(only the owner process, only with an authorization)──► collector ──► store (append-only)
                                                                                           │ read-only opening
                                    cockpit (apps/web, Events page) ◄── read-only API ◄────┘  snapshot(T, H) · replay
```

| layer | where | role |
|---|---|---|
| shared primitives | `scripts/trading_lab/sources/` | append-only record store with content-addressed raws and a read-only opening (`store.py`), CAUSAL_AVAILABILITY_V3 (`causal.py`), strict HTTP `Date`/`Age` clock evidence (`httpclock.py`), canonical hashing (`canonical.py`), rolling rate limiter (`limiter.py`) |
| FOMC | `scripts/trading_lab/fomc/` | spec revision 25: feed discovery, statement acquisition, content identity, snapshots, health, supervised service, pilot tooling |
| EDGAR | `scripts/trading_lab/edgar/` | spec revision 1: submissions listing of a closed CIK watchlist, filings/revisions/absences, snapshots, bounded runner (`service.py`) |
| read-only API | `scripts/trading_lab/app_api/` (`sources.py`, `server.py`) | `/api/v1/sources/{fomc,edgar}` (status), `/snapshot`, `/replay`, `/items/{sid}` (FOMC), `/filings/{accession}` (EDGAR); store fixed at start |
| cockpit | `apps/web/` (`EventsPage.tsx`, `EdgarPanel.tsx`) | the Events page reads the API only |
| acceptance | `scripts/trading_lab/mvp_check.py` | runs the journey over two stores and reports PASS / FAIL / BLOCKED |

There is no timeline layer on this base.

The specs `docs/artifacts/fomc_capture_spec_v1.json` and `docs/artifacts/edgar_capture_spec_v1.json` are
authoritative; the code pins their hashes (`spec.py`, `verify_spec_binding`).

## Guarantees in plain words

- **Raw first.** The bytes received are stored, immutable and content-addressed, before anything is derived
  from them. Every read re-verifies the raws it depends on.
- **Causal availability.** A fact is available from the moment a *verified server response* attested it
  (HTTP `Date`), never from a date the source claims inside the payload. A read at T under horizon H shows
  only what was committed by H and attested by T.
- **Provenance is never availability.** Values such as EDGAR's `acceptanceDateTime` are kept verbatim as
  provenance and never used as an instant, an availability or an order.
- **Unknown stays unknown.** When T is not covered by an attested response the read is
  `*_CAUSAL_VISIBILITY_UNRESOLVED` with no items and no health: never a guess, never a zero.
- **Read-only opening.** The API and the checks open stores with `read_only=True`: no DDL, pragma, side file
  or write; every write is refused. A store of another schema or spec hash is rejected, not migrated or
  repaired.
- **Replay.** `/replay` re-derives a read from the records alone (raws re-verified, clock verdicts and
  processing re-derived) and fails if it differs from the stored derivation.
- **Fail closed.** Unknown shape, corrupt raw, replay mismatch, unverified clock: a refusal or an
  `UNRESOLVED`/`CLOCK_UNVERIFIED` state, never a silent fix.
- **No churn, corrections are new revisions.** Old revisions and old reads never change; absence is an
  observation, not a deletion.
- **No request without an authorization.** Nothing reaches an official source without an operator
  authorization naming scope, budget and expiry (EDGAR runner) or the supervised owner (FOMC service).

## State of each source

| source | state | proven | not proven |
|---|---|---|---|
| FOMC (spec rev 25, `fomc-store-v5`) | **closed for the scope its real pilot exercised**: 90-minute rev-25 pilot VALIDATED (2026-10-02) | discovery from the feed, statement acquisition, content identity over real Cloudflare variations, revisions/observations, causal snapshots (resolved and UNRESOLVED) with replay, source health, rate-limit journal, +300 s and +3600 s rechecks | 24 h and 7-day rechecks, a new real publication, a real redirect, a real storage incident/restart/crash |
| EDGAR (spec rev 1, `edgar-store-v1`) | offline slice + **one real fixture trial** (2026-10-04, 2 CIKs, 8 responses, 161 filings), spec revision 1 kept | UV1 columns, UV5 `Date` present and valid; every read identical after reopening and replay | UV2 time-zone semantics, UV3 removals, UV4 response to a rate violation, UV6 universality; a new 8-K, a correction, an absence or a failure against the real source (offline only); continuous collection and supervision |

Details and numbers: the two slice documents. Everything beyond these scopes needs a new operator
authorization.

## Operator runbooks

The commands below were run offline on this repository; `python` is the project's interpreter (venv).

### Start the read-only API on stores

```
python -m scripts.trading_lab.app_api.server --port 8787 \
    --fomc-store /path/to/fomc-store --edgar-store /path/to/edgar-store
# add --dist-root apps/web/dist (after `cd apps/web && npm ci && npm run build`) to serve the cockpit from the same origin
```

Either flag may be omitted (that source answers `NOT_CONFIGURED`). The store path is fixed at start; a client
never supplies a path. Defaults: host `127.0.0.1`, port 8787; a non-loopback host prints a warning, do not
use one. **IPv6 loopback:** where `127.0.0.1` does not accept connections use `--host ::1` (the server then
binds AF_INET6 and is reachable at `http://[::1]:PORT/` only, not at `127.0.0.1`). Reading:

```
curl http://127.0.0.1:8787/api/v1/sources/edgar                     # status: store identity, horizon, suggested_as_of
curl -G --data-urlencode "as_of=<suggested_as_of>" http://127.0.0.1:8787/api/v1/sources/edgar/snapshot
curl -G --data-urlencode "as_of=<suggested_as_of>" http://127.0.0.1:8787/api/v1/sources/edgar/replay
```

`as_of` is an ISO-8601 instant with an offset (required; a naive or missing one is a 400); `horizon` is
optional (default: the store's horizon; beyond it, 400). `suggested_as_of` is the store's attested lower bound
on "now": a read after it is UNRESOLVED by design. The FOMC routes are identical under `/sources/fomc`.

### Open a store read-only (Python)

```python
from pathlib import Path
from datetime import datetime
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.edgar import snapshot

store = EdgarStore(Path("/path/to/edgar-store"), wall_clock=None, read_only=True)   # FomcStore likewise
snap = snapshot.filings_as_of(store, datetime.fromisoformat("2026-10-04T02:04:55+00:00"), 26)
print(snap["read_state"], len(snap["filings"]), snap["identity"])
snapshot.replay(store, datetime.fromisoformat("2026-10-04T02:04:55+00:00"), 26)       # raises on any mismatch
```

Never open a real store without `read_only=True`; work on a closure copy.

### Verify a store or a closure

```
python -m scripts.trading_lab.mvp_check --fomc-store DIR --edgar-store DIR \
    --fomc-reads snapshots.jsonl --web --output report.json
```

It opens both stores read-only, serves the API in-process, checks status, reads, identity stability, replay,
provenance, health, routes, the FOMC recorded reads and that no store file changed, then the cockpit
(`--web`: typecheck, lint, vitest, build; needs `npm ci`). Exit 0 all PASS, 1 any FAIL, 2 some BLOCKED (never
counted as PASS: without `--web` the four cockpit criteria are BLOCKED). Run offline on the two real
closure copies without `--web`: 19 PASS, 0 FAIL, 4 BLOCKED (cockpit), exit 2. Criteria: `docs/MVP_ACCEPTANCE.md`.
To compare two copies of a store, hash every file (`sha256sum`), before and after any work.

### Run a bounded EDGAR trial (sends real requests: only with an authorization)

1. The operator writes an **authorization file** outside Git (JSON):

   | field | rule |
   |---|---|
   | `authorizes` | `sec_edgar_submissions_v1` |
   | `spec_hash` | the EDGAR spec hash (`edgar/spec.py`, currently `98828c55…5ce`) |
   | `ciks` | 1–10 CIKs |
   | `max_requests` | integer 1–50, counting failures |
   | `not_after` | ISO-8601 expiry with offset, in the future |
   | `user_agent` | organization and contact e-mail (kept outside Git) |
   | `granted_by`, `granted_at` | who and when |

2. Preflight, no network, no write, no store created:

   ```
   python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE --check
   ```

   It prints the spec binding, the CIKs, budget and expiry (exit 0), or `EDGAR capture refused: …` listing
   every problem (exit 2).
3. Run, without `--check`. The runner paces requests (10 s global spacing, at most 6 per 60 s, one CIK at
   least every 600 s) and stops at the first of: the request budget, the expiry, SIGINT/SIGTERM, or a
   403/429 (`SOURCE_THROTTLED`: no retry, no further request). A restart marks an unfinished attempt
   INTERRUPTED and re-derives; the budget counts per run.
4. Verify the new store offline as above, then qualify it (see the fixture trial in the EDGAR document).

FOMC capture is a supervised service (`python -m scripts.trading_lab.fomc.service --store DIR --check` for its
preflight); its pilot protocol and closure are in `docs/FOMC_V1_OFFLINE_SLICE.md`. Without an explicit
operator authorization for the scope, do not start either.

### Where evidence lives

- **In Git:** code, synthetic fixtures and tests, specs, and *evidence as digests, URLs, counts and
  identities* (for example `docs/artifacts/edgar_fixture_qualification_v1.json`).
- **Never in Git:** stores, raw bodies, logs, authorization files, contact e-mails, tokens, private paths,
  captured fixtures, `.env` content.
- Real stores on the shared server are under `/srv/hyprl-stores/` (read-only closure copies; the rev-25 FOMC
  copy comes with its recorded reads `*.snapshots.jsonl`). Reports go to `~/reports/`.

## Checks before a push

See `AGENTS.md` ("Checks before any push"); anything that cannot run is reported BLOCKED with its cause.
