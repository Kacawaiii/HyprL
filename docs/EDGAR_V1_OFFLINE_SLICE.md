# SEC EDGAR V1 — slice and first real fixture

> Entry point for both official sources (architecture, guarantees, runbooks): `docs/OFFICIAL_EVENT_SOURCES.md`.

Implements `docs/artifacts/edgar_capture_spec_v1.json` **revision 1**
(`98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce`; pinned in
`scripts/trading_lab/edgar/spec.py`, checked by `verify_spec_binding`). The JSON is authoritative. This is an
**offline slice** (synthetic listings, in-process fake fetcher) completed on 2026-10-04 by **one bounded real
fixture trial** under an explicit operator authorization (see "Real fixture trial"); the spec itself
authorizes no capture (`authorizes_capture: false`) and no continuous collection is authorized.

```
python -m pytest tests/crypto/test_edgar_slice.py tests/crypto/test_app_api_edgar.py   # 39 + 3 tests
python -m scripts.trading_lab.edgar.demo                  # new 8-K, correction, 8-K/A, absence, reappearance, replay
python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE --check   # no network, no write
python -m scripts.trading_lab.edgar.closure close --store DIR --copy NEW_DIR --report FILE --authorization FILE
python -m pytest tests/crypto/test_edgar_closure.py         # synthetic operational supervision and closure
python -m pytest tests/crypto/test_edgar_supervised.py      # observed wire format, restart fences, SIGTERM
python -m scripts.trading_lab.app_api.server --edgar-store DIR   # read-only API + cockpit Events page
```

## Scope (frozen in the spec)

- **Surface:** `GET https://data.sec.gov/submissions/CIK{cik10}.json` only, for a closed watchlist of 1–10
  CIKs given once by the operator manifest.
- **Forms:** `8-K` and `8-K/A`; every other form is counted per listing, never normalized.
- **Window:** `filings.recent` only; `filings.files` (older pages) is recorded as present, never fetched.
- **Not fetched:** primary documents and exhibits, XBRL APIs, full-text search, RSS, daily/full indexes.

## Contract check against the official documentation (2026-10-03)

Verified and quoted in the spec (`verified_facts`, sources: SEC "EDGAR Application Programming Interfaces",
"Accessing EDGAR Data", Webmaster FAQ): the submissions URL format; `recent` holds at least one year or
1,000 filings (whichever is more) and `files` lists older pages; the APIs are updated in real time as
filings are disseminated (typical delay under a second for submissions, longer at peaks); no
authentication, no CORS; fair access at most 10 requests/second with a declared User-Agent, the SEC may
limit rates; the accession number is a unique identifier assigned to an accepted submission; archive paths
with and without dashes and `-index` files; SEC staff may authorize post-acceptance corrections and
removals (wrong filer, duplicate, unreadable, sensitive information), and later removals are not
reflected in earlier daily indexes; submissions begun after 5:30 p.m. ET may be disseminated the next
business day; EDGAR hours 6:00 a.m.–10:00 p.m. ET.

**Not documented, frozen fail-closed** (`unverified`): the column names of `filings.recent` (UV1: any
other shape is PARSER_FAILED), the format and time zone of `acceptanceDateTime` (UV2: verbatim text,
never parsed), whether and when a removed filing leaves the listing (UV3: only `ABSENT_FROM_LISTING`),
the response to a rate violation (UV4: 403/429 = SOURCE_THROTTLED, 600 s pause), the presence of a valid
HTTP `Date` (UV5: otherwise CLOCK_UNVERIFIED, attests nothing), any amendment link in the listing (UV6:
none inferred). A real run must first settle these with a fixture.

## Guarantees: invariant → code → test

| guarantee | code | tests |
|---|---|---|
| E1 one filing per accession, amendments separate (FOMC: source item identity) | `listing.source_item_id`, `collector.derive` (FILING_REVISION `amends: null`, `amendment_link: NOT_PROVIDED_BY_SOURCE`) | EDGAR04 |
| E2 raw first, immutable, verified on every read (FOMC: RAW_INTEGRITY_EVERYWHERE) | `collector.poll` (`put_raw` before RESPONSE), `sources.store` | EDGAR12 |
| E3 no churn; corrections are new revisions, old ones stay | `listing.filing_identity` (EDGAR_FILING_METADATA_V1), unique FILING_REVISION | EDGAR02, EDGAR03 |
| E4 absence is an observation, inside the documented window only, never a deletion | `collector.derive` (FILING_ABSENCE when filing_date > the listing's oldest filingDate), `snapshot._filing` | EDGAR05, EDGAR06, EDGAR07 |
| E5 acceptanceDateTime is provenance only | `listing.parse_listing` (text verbatim), `snapshot._filing` (`provenance.acceptance_datetime_text`; availability from V3) | EDGAR09 |
| E6 causal reads (FOMC: CAUSAL_AVAILABILITY_V3, reused unchanged) | `sources.causal`, `snapshot.filings_as_of` | EDGAR01, EDGAR11, EDGAR03 (an old read is unchanged) |
| E7 fail closed: unknown shape, corrupt raw, replay mismatch | `listing.ListingRejected`, `snapshot.SnapshotFailed/ReplayFailed`, `snapshot.replay` (processing, rows and health re-derived) | EDGAR08, EDGAR12, EDGAR15 |
| E8 bounded access: watchlist only, submissions surface only, paced, declared User-Agent | `collector.poll` (watchlist), `sources.limiter` (10 s spacing, ≤ 6/60 s, 60 s embargo), `transport.HttpsFetcher` | EDGAR14, EDGAR16, watchlist test |
| store opening rule, read-only opening (FOMC: STORE_OPENING_RULE) | `sources.store.admit_existing`, `RecordStore(read_only=True)`, `edgar-store-v1` | EDGAR13 |
| restart: an attempt of an earlier epoch without outcome is INTERRUPTED, a saved record without processing is processed (FOMC: reconciliation) | `collector.reconcile` | restart test (replay identical, INTERRUPTED health re-derived) |
| one 30 s deadline over connect, request, headers and body | `transport.HttpsFetcher.fetch` | deadline test |
| a store operation stalled for 10 s suspends new requests; ended incidents commit before requests resume; stop/expiry are rechecked after a slow invocation commit | `service.StorageWatch`, `collector.poll(request_allowed=...)` | `test_edgar_closure.py`: blocked raw write and invocation commit, expiry/stop while blocked, one durable incident |
| stop confirmed, owner excluded during backup, every raw verified, exact causal reads identical after reopen/replay, health replay, copy unchanged; integrity and authorized completion separate | `closure.close/consistent_copy/verify_copy/run_completion` | `test_edgar_closure.py`: budget/expiry, interrupted run, corrupt raw, altered health, recorded identity, owner/stop refusal, WAL backup |
| persistent user timer closure at expiry, rounded up to the second; template generation never installs or starts units | `closure.closure_units/write_closure_units` | `test_edgar_closure.py`: whole/fractional seconds, templates written with no systemctl call |
| no request without an operator authorization (EDGAR spec, CIKs, budget <= 50, expiry, declared User-Agent); stops at budget, expiry or signal | `service.load_authorization/check/run` | 12 refusal cases (nothing sent, no store created), budget and expiry |
| read-only API and cockpit journey: status, read at (as_of, horizon), filing detail (revisions, observations, absences, provenance), replay | `app_api.sources.EdgarViews`, `/api/v1/sources/edgar[/snapshot|/replay|/filings/{accession}]`, `apps/web/src/pages/EdgarPanel.tsx` | `test_app_api_edgar.py`, `apps/web/src/test/edgar.test.tsx` |

The demo reads after each step and shows, for example, the corrected 8-K with two revisions seen, the
8-K/A as its own filing, the 8-K `ABSENT_FROM_LISTING` and then `PRESENT` again, availability taken from
the attesting response (never from `acceptanceDateTime`), and every read identical after reopening the
store read-only and replaying it.

## Real fixture trial (2026-10-04)

**Authorization and bounds.** One trial authorized by the operator in session: submissions endpoint only,
CIKs 320193 (Apple Inc.) and 789019 (MICROSOFT CORP, both in the instrument catalog), at most 8 physical
requests including failures, at least 600 s between two requests of one CIK, the global limiter (10 s
spacing, 60 s embargo), 45 minutes, stop on 403/429, no retry, User-Agent with the operator's declared
contact (kept outside Git). Frozen code `9dd955893412e82a4663a95b746b8f4565cd6e9f` (detached worktree), new
`edgar-store-v1` store, authorization file, store, raws and logs outside Git. `--check` passed (no file
created) before the launch.

**What happened.** Started 01:34:45Z as the user unit `edgar-fixture-trial`; 8 requests, 4 rounds of AAPL
then MSFT (01:35:46/56, 01:45:56, 01:46:06, 01:56:07/17, 02:06:17/27), **8 × HTTP 200**, all
CLOCK_VERIFIED (Date − receipt 0.01–0.66 s), grant-to-end 0.24–0.49 s; ended 02:16:27Z, "request budget
spent", exit 0. No 403/429, no failure, no retry. Each CIK served the same bytes four times: **2 distinct
raws** for 8 records. Listings: Apple 1001 rows (one older page), Microsoft 1002 rows (two older pages);
in scope **102 + 56 8-K and 2 + 1 8-K/A = 161 filings, 161 revisions, 644 observations, 0 absences** (no
churn on re-reads).

**Verified offline** (consistent backup-API copy, read-only): `integrity_check` ok; 6 causal reads
resolved (104, then 161 filings, all PRESENT) and the reads at "now" UNRESOLVED with nothing shown; every
read identical after reopening and identical in verified replay (8/8); the copy unchanged by the reads; the
API on the copy: same snapshot identity, filing detail with revisions and observations, health without
failure, replay identical. Evidence: `docs/artifacts/edgar_fixture_qualification_v1.json` (digests, URLs,
timings, counts, matrix); local: the run directory's `verification.json`.

**UV1–UV6 qualification** (documentation / this fixture / still unknown):

| point | documentation | fixture (2 CIKs, 8 responses) | verdict | rule kept |
|---|---|---|---|---|
| UV1 columns and types of `filings.recent` | not documented | the 5 required and 9 optional columns present in every listing with the frozen types; 2 more columns (`core_type`, `isXBRLNumeric`) ignored by normalization, kept in the raws | OBSERVED_COMPATIBLE | any other shape stays PARSER_FAILED |
| UV2 acceptanceDateTime format and time zone | not documented | 8012 values, all `YYYY-MM-DDTHH:MM:SS.000Z` | FORMAT_OBSERVED_SEMANTICS_UNKNOWN | verbatim text, never an instant, an availability or an order |
| UV3 removed filings and the listing | removals and corrections exist (VF10) | none provoked, 0 absences | UNKNOWN_PERSISTS | only ABSENT_FROM_LISTING inside the window |
| UV4 response to a rate violation | 10 req/s, the SEC may limit (VF6) | 8 × 200, none provoked | UNKNOWN_PERSISTS | 403/429 THROTTLED, trial stops, ≥ 600 s pause |
| UV5 HTTP Date | not documented | present and valid in 8/8, within 0.66 s | OBSERVED_PRESENT_AND_WITHIN_TOLERANCE | no valid Date = CLOCK_UNVERIFIED |
| UV6 amendment link in the listing | not documented | 180 `/A` rows of all forms, no linking column | NO_LINK_COLUMN_OBSERVED_UNIVERSALITY_UNKNOWN | no parent inferred |

Response headers seen: `access-control-allow-origin`, `connection`, `content-encoding` (gzip, decoded),
`content-length`, `content-type` (application/json), `date`, `strict-transport-security`, `vary`, three
`x-amz*` gateway identifiers; no `Age`.

**Spec decision.** Revision 1 is kept unchanged: nothing observed contradicts it and every UV rule stays as
conservative as written. A revision would bind new stores to another hash and refuse this fixture store
(STORE_OPENING_RULE, no implicit migration); the observed compatibility is recorded in the qualification
artifact, bound to the spec hash and the code SHA.

## Shared primitives (extracted from the FOMC slice)

`scripts/trading_lab/sources/`: the append-only record store with content-addressed raws and a read-only
opening, CAUSAL_AVAILABILITY_V3 and its prefix, strict HTTP `Date`/`Age` clock evidence, canonical hashing
and the rolling limiter. FOMC binds them to its spec unchanged (its 198 tests pass); EDGAR binds them to its
own spec, schema and unique kinds.

## Bounded-run supervision and offline closure

The runner remains single-threaded for capture. A separate monitor reads the shared store's operation
markers without its lock, including when the runner's own commit or raw write is stalled. At 10 s it
suspends the limiter and emits a storage alert. Once writes resume, the runner commits a
`STORAGE_INCIDENT` before opening the request gate again. The gate rechecks stop and expiry immediately
before transport, including after an invocation commit that blocked. A reserved invocation that never
sends is recorded as `INTERRUPTED` and remains counted conservatively.

Operational `RUN_STARTED`/`RUN_ENDED` rows record public authorization bounds, an authorization
digest and the stop reason. Each service attempt also records that digest: canonical JSON of the
complete grant, with CIKs normalized to ten digits. File names, JSON whitespace and key order do not
change it. Schema and spec revision stay unchanged. The declared contact is never persisted in these
rows. All reservations under the same grant count across epochs, including attempts that never sent
or were interrupted. `--check` refuses an exhausted, expired or terminated grant without writing.
Restart using the same authorization and the same durable store; a new empty store cannot prove a
previous launch's accounting. Never reuse a grant with a new store or edit it to recreate budget.

A 403/429 commits `AUTHORIZATION_TERMINATED` in the same transaction as its outcome and health row,
so a crash before `RUN_ENDED` cannot reopen that grant. An earlier throttled outcome also fences an
older run without that termination row. Even a different grant retains the source-wide 600 s pause.
Restarts restore per-CIK cooldowns from durable reservation times, conservatively allowing the full
30 s start uncertainty; a fresh limiter epoch cannot shorten the frozen 600 s cadence. Stop and expiry
remain cancellable during these waits. The runner checks bounds again after the last response and
limits the between-round sleep by expiry.

Production HTTPS runs in a disposable Linux worker process. The owner enforces the 30 s grant-to-body
deadline independently of DNS, headers, read, gzip decoding, connection close and result delivery;
it kills and reaps a hung worker before recording an unavailable outcome. Late bytes cannot enter the
store. SIGTERM goes to the owner: an in-flight body completes within its existing deadline, is committed
and processed, and then the owner releases its lock. A supervisor must allow that drain; the prepared
user-unit launch below uses `KillMode=mixed` so its initial stop targets only the owner.

`STORE_DIR/service-status.json` is atomically replaced without taking the SQLite lock. It names the
PID, epoch, update instant, run state/reason, grant suspension, active/pending incidents and threshold.
The monitor refreshes it roughly once per second and immediately on incident transitions; alerts
`STORAGE_INCIDENT_STARTED`/`STORAGE_INCIDENT_ENDED` also appear on stdout. A supervisor reads this file,
checks freshness and process identity, and treats a stale `running` value as lost supervision. An
ended status is written after the monitor stops, so a late heartbeat cannot overwrite it.
Closure retains the stricter single-epoch completion criterion: a resumed capture can finish its
authorization budget while its interrupted original run remains NOT_ACCOMPLISHED.

`closure.close` confirms an explicitly named capture process or user unit has stopped, takes the
existing owner lock, and holds it through a consistent SQLite backup from a read-only connection and
the immutable raw copy. A published read-only closure directory without an owner file is also
admissible: the directory cannot admit a collector that needs to create that file. A writable store
without its owner file is refused. No source file is created by closure. The target must be new and
separate from the source, and the JSON report must be outside the source.

Verification opens only the copy read-only: `integrity_check`, every RESPONSE raw's digest and length,
reads at server attesting instants plus the frozen causal bound, a read at now, reopening with identical
snapshots, verified replay at each exact `(T, H)`, full-horizon replay including health and the unresolved
tail, UV1–UV6 qualification, and a before/after digest of every copied file. Optional recorded reads
(`--snapshots JSONL`, entries with `T`, `H`, `identity`) must match their recorded identities too.
Rejected source listings remain valid evidence if their rejection replays; inability to qualify such a
listing is explicit. Qualification observations never relax a UV rule.

The JSON report separates `integrity: VALID/INVALID` from `run: ACCOMPLISHED/NOT_ACCOMPLISHED`. A stopped,
throttled or interrupted run can have VALID integrity. Accomplishment requires the authorized budget
or a witnessed expiry, with compatible scope and one owner epoch; closing an already interrupted run
after its expiry proves no duration. Legacy stores can prove a spent budget using `--authorization`
because they predate durable run bounds. Without supplied or durable bounds, completion stays
NOT_ACCOMPLISHED. Missing stop/owner/copy verification stays INVALID, with the failed check recorded.
When the complete private authorization is supplied, its canonical identity must also match the
recorded grant; identical public bounds from a different grant cannot claim accomplishment.

Generate a persistent closure timer and one-shot service into a new template directory with:

```
python -m scripts.trading_lab.edgar.closure units --unit edgar-trial --code CODE_DIR --run RUN_DIR \
  --authorization AUTHORIZATION_FILE --close-at OFFSET_AWARE_EXPIRY --out NEW_TEMPLATE_DIR
```

The templates close `RUN_DIR/store` into `RUN_DIR/closure-copy` and write
`RUN_DIR/closure-report.json`. `Persistent=true` covers a missed expiry; `OnCalendar` is UTC, rounded
up to the next second when necessary. Template generation performs no installation or systemd action.
The operator must install them before an authorized run. Tests only write templates and mock capture
unit stopping; no units are installed.

## Next bounded capture: operator preparation only

This template grants nothing until the operator fills and approves every placeholder in a private
file under the host's authorization directory. Keep the contact, authorization, store, logs and
closure reports outside Git. Proposed scope: submissions only, these two CIKs, at most eight total
requests including failures and crash reservations, a 45-minute expiry, stop durably on 403/429,
no continuous collection, training or trading. Retain the same file and store on restart.

```json
{
  "authorizes": "sec_edgar_submissions_v1",
  "spec_hash": "98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce",
  "ciks": ["0000320193", "0000789019"],
  "max_requests": 8,
  "not_after": "<OFFSET_AWARE_EXPIRY_45_MINUTES_AFTER_GRANTED_AT>",
  "user_agent": "<ORGANIZATION> <OPERATOR_DECLARED_CONTACT>",
  "granted_by": "<OPERATOR>",
  "granted_at": "<OFFSET_AWARE_GRANT_INSTANT>"
}
```

Operator commands, from this checked-out branch and with the chosen Python environment active.
Set `EDGAR_EXPIRY` to the exact `not_after` value. Preflight and template generation are offline:

```bash
EDGAR_CODE_DIR="$PWD"
EDGAR_RUN_DIR="$EDGAR_CODE_DIR/var/edgar-next-bounded"
EDGAR_AUTHORIZATION="$HOME/authorizations/edgar-next-bounded.json"
EDGAR_PYTHON="$(command -v python)"
EDGAR_EXPIRY='<EXACT_NOT_AFTER_FROM_AUTHORIZATION>'
mkdir -p "$EDGAR_RUN_DIR"
chmod 700 "$EDGAR_RUN_DIR"
"$EDGAR_PYTHON" -m scripts.trading_lab.edgar.service \
  --store "$EDGAR_RUN_DIR/store" --authorization "$EDGAR_AUTHORIZATION" --check
"$EDGAR_PYTHON" -m scripts.trading_lab.edgar.closure units \
  --unit edgar-next-bounded --code "$EDGAR_CODE_DIR" --run "$EDGAR_RUN_DIR" \
  --authorization "$EDGAR_AUTHORIZATION" --close-at "$EDGAR_EXPIRY" \
  --out "$EDGAR_RUN_DIR/closure-templates"
```

The operator reviews and installs the generated closure timer/service as **user** units and enables
the persistent timer before launch. The following launch/stop commands are prepared, never executed
as part of offline qualification. They name only this new capture unit; no system units are involved.

```bash
systemd-run --user --unit=edgar-next-bounded \
  --property="WorkingDirectory=$EDGAR_CODE_DIR" \
  --property=KillMode=mixed --property=TimeoutStopSec=180 \
  "$EDGAR_PYTHON" -m scripts.trading_lab.edgar.service \
  --store "$EDGAR_RUN_DIR/store" --authorization "$EDGAR_AUTHORIZATION"

# Read supervisor status; stop only the named capture.
cat "$EDGAR_RUN_DIR/store/service-status.json"
systemctl --user stop edgar-next-bounded.service

# One closure only; the copy directory must be new. The timer invokes these same paths.
"$EDGAR_PYTHON" -m scripts.trading_lab.edgar.closure close \
  --unit edgar-next-bounded --store "$EDGAR_RUN_DIR/store" \
  --copy "$EDGAR_RUN_DIR/closure-copy" --report "$EDGAR_RUN_DIR/closure-report.json" \
  --authorization "$EDGAR_AUTHORIZATION"
```

For an already stopped capture, omit `--unit` and use fresh copy/report paths if a closure already
exists. Optional `--snapshots JSONL` verifies recorded `(T, H, identity)` reads. Closure exit 0 requires
both VALID integrity and ACCOMPLISHED run; exit 1 can still contain VALID captured evidence. Inspect
the separate verdicts, qualification and unresolved tail. No launch, unit installation or capture is
authorized by this registry entry.

## Supervised offline qualification (2026-10-05)

The supervision branch was merged with phase5 hardening and the real digit-string CIK fix; both
parents' tests remain. Synthetic listings now carry the sixteen observed columns, including ignored
`core_type` and integer/null `isXBRLNumeric`. Qualification tests send gzip through fake connections
with every observed header name and IMF-fixdate Date values. Spec revision/hash remain unchanged.

In addition, the fixture's **exact decoded listing bytes** and **exact recorded header lines** were
replayed locally through fake connections, re-encoding the bytes as gzip. The captured Content-Length
is preserved as header evidence; it is not a length claim about this newly compressed fake wire body.
The local harness prohibited socket connections. The fixture was opened read-only and its full tree
digest was identical before and after. Private inputs and outputs remain ignored; only digests,
counts and verdicts are in `docs/artifacts/edgar_supervised_offline_qualification_v1.json`.

| local case | fake requests / responses | result | closure integrity / run |
|---|---|---|---|
| full bounded run | 8 / 8 | 161 revisions, all CLOCK_VERIFIED; 8 raws verified, 9/9 reopen and replay reads | VALID / ACCOMPLISHED |
| restart mid-budget | 2 + 2 / 4 | one four-request budget across epochs; exhausted relaunch sends zero; all CLOCK_VERIFIED | VALID / NOT_ACCOMPLISHED (multiple epochs) |
| stop during limiter wait | 0 / 0 | stopped before transport | VALID / NOT_ACCOMPLISHED |
| 429 | 1 / 0 | termination durable; same-grant restart refused | VALID / NOT_ACCOMPLISHED |
| 30 s deadline | 1 / 0 | unavailable outcome, no raw admitted; failed request consumes budget | VALID / ACCOMPLISHED (budget, no listing qualification) |
| store stall >10 s | 4 / 4 | zero requests while stalled, one persisted incident, 161 revisions | VALID / ACCOMPLISHED |

Every case reopened/replayed identically, replayed health, and left the closure copy unchanged.
Cases with no admitted responses cannot qualify listing or clock compatibility; their valid integrity
and any accomplished request budget remain separate from that absence of evidence.

| required behaviour | executable offline evidence |
|---|---|
| flock, epoch, crash reconciliation and saved-record processing | `test_watchlist_ownership_and_spec_binding`, `test_a_restart_interrupts_open_attempts_and_processes_saved_records`, `test_owner_initialization_failure_releases_flock` |
| SIGTERM drains and processes an in-flight body, releases ownership | `test_sigterm_finishes_inflight_body_processes_it_and_releases_ownership` (real signal, fenced fake connection) |
| alerts and supervisor status during a blocked store | `test_status_is_readable_during_a_storage_stall_and_cleared_after_recovery`; closure tests for raw-write and invocation-commit stalls |
| grant identity on every attempt, budget across epochs, exhausted/expired refusal | `test_observed_wire_format_full_bound_and_closure`, `test_restart_mid_budget_counts_all_epochs_and_reservations`, `test_canonical_authorization_identity_survives_filename_and_json_order_changes`; slice authorization-refusal tests |
| stop/expiry during limiter waits and after a slow commit | slice embargo/slow-commit tests, `test_stop_during_spacing_wait_keeps_only_the_finished_attempt`, closure stopped/expired-stall tests |
| durable 403/429 stop, including crash before run termination | `test_throttle_termination_is_atomic_and_refuses_restart_even_without_run_ended` (both statuses), `test_a_new_grant_keeps_the_previous_throttle_pause_for_all_ciks` |
| physical deadline; hung transport fenced; no late bytes | `test_hung_transport_is_killed_reaped_and_never_returns_late_bytes` (connect, headers, read, decode, close, delivery), `test_deadline_includes_decode`, `test_a_slow_commit_consumes_the_physical_deadline_without_sending` |
| offline copy/integrity/raws/causal reads/reopen/replay/health/qualification; separate verdicts | `test_budget_closure_verifies_raws_causal_reads_reopen_replay_health_and_qualification`, closure rejection/tamper/WAL/owner tests, `test_closure_requires_the_same_grant_identity_even_with_identical_public_bounds` |


## Remaining limits

- One real trial only (2 CIKs, 8 responses, 33 minutes): UV2 semantics, UV3, UV4 and the universality of
  UV6 stay unknown, and nothing here covers a new 8-K appearing during a capture, a correction, an absence
  or a failure against the real source (those are proved offline only).
- The runner is single-threaded and step-driven, with storage supervision and persistent closure
  templates. An operation that never resumes cannot durably record its incident; a still-held owner
  lock prevents closure verification. No real storage fault or timer/reboot deployment has been
  exercised; those behaviors are verified synthetically only.
- Absence detection depends on the documented window (UV3); a filing removed and replaced by the SEC with
  the same accession would show as a correction, not as a removal.
- Any further capture needs a new operator authorization (budget, expiry, declared contact).
  Continuous collection remains outside the authorization of this bounded runner.
