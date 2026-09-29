# FOMC V1 — offline integrated slice

Implements `docs/artifacts/fomc_capture_spec_v1.json` **revision 22**
(`ba6a01e5f12e810ecde89711304c278862d147de298c63e7602a8359fa18e678`, pinned in
`scripts/trading_lab/fomc/spec.py` and checked by `verify_spec_binding`). The JSON stays
authoritative. This is an **offline slice**: every source is synthetic and served by a local
Unix-socket provider; no Federal Reserve request, fixture capture or live run is part of it.

**Capture verdict: NOT READY. One capture blocker remains (`COMMIT_FSYNC_120S`, below).** The
autonomous service, the TLS connector, the store opening rule and a prolonged crash/restart run are
delivered and verified; real capture stays refused by the code (`service.CAPTURE_BLOCKERS`).

```
python -m scripts.trading_lab.fomc.demo                 # the executable path, end to end
python -m pytest tests/crypto/test_fomc_*.py            # 138 passed + 2 strict xfail (the blocker)
python -m scripts.trading_lab.fomc.soak --hours 26      # prolonged run with crashes, verified (~95 s)
python -m scripts.trading_lab.fomc.service --store DIR  # real capture: refused while blocked (exit 3)
```

`feed -> durable raw -> classification -> primary acquisition -> revision + observation link ->
cycle -> events_as_of(T, H) -> reopen -> offline replay`

| module | role |
|---|---|
| `spec.py` | revision-22 constants, canonical hashing, spec binding |
| `identity.py` | URL admission (scheme, host, userinfo, port, percent, query, fragment), `source_item_id`, durable keys |
| `clock.py` | strict `Date`/`Age`, per-response clock verdict, canonical timestamps |
| `store.py` | SQLite WAL/FULL, `commit_seq`, append-only rows, unique keys, immutable content-addressed raw (published once with `os.link`, never overwritten) with digest checks; an append-only in-memory mirror refreshed with only the rows committed since its last refresh, served as consistent `StoreView`s bounded by a horizon, with incremental aggregates registered by derivations; read and in-memory work accounting |
| `ledger.py` | epochs (fencing), atomic `TRANSPORT_INVOKED`, one outcome per attempt, `LATE_EVIDENCE`, episodes |
| `limiter.py` | FIX15 rolling window, spacing, embargo |
| `transport.py` | manual redirect loop, per-hop admission and grant, 60 s deadline from each grant over DNS/TCP/TLS/write/headers/body/decoding (stage timeouts, watchdog, late-result checks), bounded decoded body |
| `parsing.py` | Content-Type gate, strict UTF-8, secure RSS (expat), token-bounded HTML anchors, grammars |
| `processing.py` | one terminal outcome per PROCESSABLE record, feed classification + cycle, primary classification + revision/link (27 normalized fields), fenced runs, poison guard, the record's source-health result |
| `health.py` | source health per surface: mappings, precedence, the persisted row, the exposed state |
| `state.py` | LIVE_ELIGIBLE, server-attested `avail`, `NOW_LB`, anchors, obligations, episodes, item conclusion, RESOLVE validity |
| `collector.py` | single owner (flock + epoch), fetch/commit, reconciliation, episodes, class rotation and per-class ordering (`select`), operator actions, manifests, integrity diagnostics |
| `snapshot.py` | P(T) under horizon H, read state, 9-step selection, discovery state, source health, identity, read-time raw dependency checks, verified replay (health re-derived) |
| `service.py` | the autonomous owner: tick loop, task closure at the 60/120/600 s bounds, worker threads, restart; the real-capture entry point, refused while `CAPTURE_BLOCKERS` is non-empty |
| `soak.py` | the prolonged synthetic run with faults, crashes and restarts, and its verification |
| `synthetic.py`, `demo.py` | simulated clock, local provider, fixtures, the demo |

## Proven guarantees: invariant → code → test

Each row is exercised by the named tests (offline, synthetic sources). Anything not proven by a test is listed under "Remaining limits".

| rule | code | tests |
|---|---|---|
| F08 raw first, F72 corruption fails closed | `collector.commit`, `store.put_raw/read_raw`, `processing.process_record`, `collector.verify_integrity/_diagnose`, `snapshot._verify_dependencies/replay` | persistence `raw_is_content_addressed…`; snapshot `test_7`; hardening: corrupt or missing raw fails `events_as_of` without prior verification (old `(T, H)` too, revisions and discovery cycles), `put_raw` never overwrites, corrupt slot → the new attempt is `LOCAL_PERSISTENCE_FAILED` (no RESPONSE, no anchor, no recheck satisfaction, no LIVE availability; backfill, LIVE acquisition and LIVE recheck covered) + integrity diagnostics on older records, file never repaired |
| F09/F10/F74 revisions keyed (item, hash), mode-neutral links | `processing.classify_primary` | snapshot `test_1`, `test_2`, `test_7` (A-B-A) |
| `normalized_minimum_fields` (27): 21 immutable properties of the revision, 4 per observation (`observed_at`, `source_observation_id`, `raw_artifact_identities_and_hashes`, `rss_guid_if_available`), `observation_mode` and `ingested_at` on both (on the revision: creation provenance and the avail of its creating transaction) | `spec.NORMALIZED_KEYS/REVISION_FIELDS/OBSERVATION_FIELDS`, `processing.classify_primary` (REVISION row written once, LINK row per observation), `snapshot._item_state` (`ingested_at` = avail within P(T)) | normalized: the split covers exactly the 27 spec names; nulls where prescribed (`content_source_available_at`, `source_updated_at`, `declared_release_at` for immediate or unparsed release, `observed_at` when unverified, GUID when none was listed); backfill → LIVE (one unchanged row, backfill `observed_at` = the actual collection instant, never the release time), content correction D1 → D2 (new revision, D1 row intact, same item), A-B-A (V_A reused, one link per observation) |
| source health per surface (`source_health`, `failure_classification_v1.precedence`), durable before visible | `health.for_attempt_outcome/for_record/for_integrity_diagnostic/row`; committed in the transaction of the determining record: the attempt outcome (`ledger.commit_attempt_outcome`), the processing outcome (`processing.finish_run/poison`) or the integrity diagnostic (`collector._diagnose`) | health: the six primary cases (success, healthy negative, CLOCK_UNVERIFIED, PARSER_FAILED, PARSER_FAILED over CLOCK_UNVERIFIED with the verdict kept, SOURCE_UNAVAILABLE), each in its outcome's transaction, never on the feed surface; feed zero; LOCAL_PERSISTENCE_FAILURE, INTERRUPTED, RAW_CORRUPTION before processing → `NO_PROVIDER_HEALTH_STATE`; cancellation → no result |
| health exposed only under FOMC_RESOLVED, within P(T), bound to identity | `health.exposed`, `snapshot.events_as_of` | health: SOURCE_NOT_CHECKED before any primary check in P(T), latest check within P(T), absent when unresolved or RETROSPECTIVE_SOURCE, identity = hash of the payload with health, a copy with altered health → another identity |
| corruption found after the terminal outcome: outcome kept, diagnostic and `NO_PROVIDER_HEALTH_STATE` (RAW_CORRUPTION) on the record's surface in one transaction, once per record, no refetch | `collector._diagnose/verify_integrity`, `INTEGRITY_DIAGNOSTIC` unique per record | health `corruption_found_after_the_outcome…`: one transaction with exactly the two rows, outcome unchanged, no request and no `TRANSPORT_INVOKED`, a second scan adds nothing; a resolved snapshot after it shows the step-3 corruption barrier and `primary_statement` = RAW_CORRUPTION for the same record, feed health untouched; the older (T, H) that used that raw fails closed (`SnapshotFailed`); verified replay fails closed |
| health replay: every field of every health row re-derives from the durable data of its own transaction (key, provider_id, surface, check_at, result_state, reason, outcome, attempt, record, sid, diagnostics); one row per determining transaction, none elsewhere; a diagnosed raw must still fail its digest | `snapshot.verify_health` (called by `replay`); `check_at` = the record's `wall_at_receipt` or the transaction's `wall_at_commit` (`store.append(wall_at_commit=…)`) | health `health_replay_checks_every_durable_field…`: 33 isolated edits (11 per row kind: attempt outcome, processed record, integrity diagnostic) on store copies, each fails closed while the untouched store passes; `…binds_check_at_to_durable_provenance…`: edited `wall_at_commit` (attempt, diagnostic), edited `wall_at_receipt`, deleted health row, deleted or edited diagnostic, diagnosed raw restored → each fails closed |
| `live_watermark` | `snapshot.prefix` | snapshot `live_watermark…`: at every probed T, resolved iff a resolved W with avail > T and no earlier UNRESOLVED exists; nothing FOMC is exposed otherwise; resolved later by the next verified responses; replay at the old (T, H) stays unresolved. No `SNAPSHOT_WATERMARK` transaction is written (see below) |
| F26–F30 identity and URL admission | `identity.admit_url`, `source_item_id` | persistence scheme/port/percent cases (FOMC31–38) |
| local save deadline: 120 s monotonic after the network end, then `LOCAL_PERSISTENCE_FAILED`, no request | `collector.network_ended` keeps the attempt's deadline from the network end; `collector._commit` (retry every `SAVE_RETRY_S` = 5 s, an implementation choice) admits the record only through `ledger.commit_response(admit=…)`, whose check runs inside the write transaction, after `BEGIN IMMEDIATE` and before the rows are inserted and SQLite's `COMMIT` (at 120 s, expired: no RESPONSE, no LATE_EVIDENCE); `collector.reconcile` writes `LOCAL_PERSISTENCE_FAILED` at or after the deadline when no outcome exists, even while the saving task is blocked (first outcome wins). What this does **not** bound is stated under "Not in this slice" | local_deadlines: 23 failures → saved at +115 s, 24 → `LOCAL_PERSISTENCE_FAILED` at +120 s; a slow write that succeeds at +119.9 s → RESPONSE, at +120.0 s and +120.1 s → `LOCAL_PERSISTENCE_FAILED` without any record; `fetch()` → wait → `reconcile()` at +120 s without `commit()` → `LOCAL_PERSISTENCE_FAILED`, the late bytes stay out; each: one provider request, no false zero, next attempt is #2; bytes held past it are never persisted |
| 600 s absolute attempt deadline, INTERRUPTED, late results | `collector.reconcile` (active tasks tracked, never dead before the deadline), `collector._commit` (closes first), `ledger.commit_response` (LATE_EVIDENCE) | local_deadlines: alive at 599.5 s, INTERRUPTED at 600 s, late bytes → LATE_EVIDENCE; restart interrupts at once and keeps the budget; a held feed poll blocks polls without a cycle or zero |
| 600 s processing-run deadline, DEAD runs, poison | `processing.start_run/finish_run` (fence rejects results at or after the deadline), `collector.process_pending/_mark_dead` | local_deadlines: no second run and no DEAD before 600 s, replaced at 600 s, 599.9 s admitted vs 600 s discarded, overrunning feed runs poisoned with a NOT_ZERO conclusion |
| F31 decoded body cap | `transport._admit_200` | components `oversized…` (FOMC39) |
| F60 60 s physical deadline from each grant, per hop | `transport.Deadline`, `_resolve/_connect/_hop/_admit_200`, watchdog | hardening: late DNS without TCP, late TLS, blocked/dripped headers, dripped body, per-redirect deadline, real-socket watchdog (FOMC127–130, FOMC154) |
| F32 strict UTF-8, F33/F34 anchors, F36 XML security | `parsing` | components decoding/anchors/XML cases (FOMC41–62) |
| F38–F40 final 200, redirects ≤ 3 | `transport.fetch` | processing `redirects…` (FOMC69/70/84) |
| F44–F49 limiter | `limiter.Limiter` | components `limiter…` (FOMC90/91/97) |
| F50–F56 anchor, obligations, satisfaction | `state.anchor/obligations/work_satisfied` | processing `resolve_cutoff…` (O300), snapshot `test_1`, `test_7` |
| F57/F65/F66 six attempts, 120 bound, no extra traffic | `ledger.transport_invoked`, `collector._ready` | persistence budget/concurrency/fencing; processing `dead_page…` (24 requests) |
| ELIGIBLE_CLASS_ALTERNATION_V1: durable rotation, per-class order (backfill: manifest commit_seq, next eligible instant, sid; recheck: instant, sid, offset, a first attempt eligible from due_at + 92 s; acquisition: instant, sid), continuations inside their logical fetch | `collector.select/_ready/_opening_instant/_order`, `transport.fetch` | scheduling: FOMC126 (24 items, 5 permanently failing: strict alternation whenever both classes are eligible, healthy +300 s and +1 h rechecks served, failing episodes SUSPENDED after ≤ 6 attempts, feed gap ≤ 90 s, limiter respected), FOMC126 same choice after restart (both rotation directions), backfill across two manifests (manifest 1 first although every manifest-2 sid is smaller, a retry taken as soon as eligible ahead of manifest 2, manifest 2 in sid order, a redirect continuation = next physical request with no extra `TRANSPORT_INVOKED`, same choice and same first backfill entry after restart); the old ordering fails this test |
| MANUAL_RETRY of a backfill entry never listed by the feed: URL and manifest from the validated entry, mode HISTORICAL_BACKFILL, manifest-first rank, own budget | `collector.manual_retry/_manifest_entry/work_of`, `collector._order`, `state.work_satisfied` (the entry's own key or its manual retry satisfies it) | scheduling `manual_retry_of_a_backfill_entry…`: the autonomous entry SUSPENDED after 6 × 404; the retry ranks ahead of 40 manifest-2 entries with smaller sids (same first choice after restart); its first attempt dies with the owner (INTERRUPTED), the second is taken as soon as eligible ahead of manifest 2's remaining entries, which keep sid order; budgets: autonomous key 6 attempts and still SUSPENDED, manual key 2, 8 requests for that URL, 1 per manifest-2 entry; record mode HISTORICAL_BACKFILL, work MANUAL_RETRY, entry URL, no LIVE anchor; an unknown item or an item in two manifests without a named manifest is refused |
| F61/F79 zero and its exposure | `processing.cycle_conclusion`, `snapshot._discovery` | processing `new_statement…`; snapshot `test_4`, FOMC245 |
| F63 healthy negative | `classify_primary` | components `healthy_negative…` |
| F64 selection within P(T) | `snapshot._item_state` | snapshot `test_1`, `test_3`, `test_7`, FOMC244 |
| F70/F71 server-attested availability, PROCESSABLE vs LIVE_ELIGIBLE | `state.availability/live_eligible` | processing `clock_unverified…`, `unverified_malformed…`; snapshot `test_3` |
| F73 RESOLVE cutoff, FOMC247 OPENABLE | `collector.resolve`, `state.resolve_validity` | processing `resolve_cutoff…`, `fomc247…`; snapshot `test_4`; components unidentifiable |
| F75 single owner, fenced runs, poison | `collector.process_pending`, `processing.process_record/poison` | processing `poison_guard…`, crash test |
| F76 clock headers | `clock` | persistence `clock_check_cases` |
| F77 GUID provenance | `processing.classify_feed` | components `guid…` (FOMC181/182) |
| F78 one outcome, LATE_EVIDENCE | `ledger.commit_response` | persistence `one_outcome…`; processing crash/late test; snapshot `test_5` |
| replay == live at (T, H) | `snapshot.events_as_of/replay` | snapshot `test_8`, FOMC244; demo step 8 |
| no quadratic store re-reads in derivations and snapshots, answers unchanged | `store.FomcStore.view/_Mirror/StoreView`; every `state` derivation, `processing.cycle_conclusion/classify_feed/earlier_feed_unterminated`, `collector.open_episodes/eligible_work/derive_terminals/process_pending`, `snapshot.events_as_of/replay` read one view | read_cost: `events_as_of` and `outstanding` through the view equal direct SQL reads at > 20 (T, H) pairs (resolved and unresolved); a cold reader loads the store once (3 queries, rows + txns read once), then 1 query and 0 rows per operation; the same derivations (6 queries, 0 rows) and the same 300 s of capture (210 queries, 75 rows for 50 new rows + txns) cost exactly the same on a 20-min and a 40-min store. Before (11f56bc, 30-/60-min stores): a snapshot cost 95/125 queries and 742/1342 rows, a cycle conclusion 101/161 queries and 341/611 rows, `eligible_work` 65/95 queries and 631/1171 rows |
| no items × records in-memory work, answers unchanged | incremental aggregates fed by the mirror: `state.NowLb` (running maximum, exact at any horizon), `state.OpenWork` (attempts without outcome, records without processing outcome, non-terminal LIVE feed records; used only by a view at the head, older views rescan); per-view memo of LIVE_ELIGIBLE FEED_CLASSIFIED records; failed feeds found through the outcome index | read_cost `in_memory_work…` (12 items, 20- vs 40-min stores, work counted as rows handed out by the view): outstanding, cycle conclusion, open_episodes, derive_terminals, process_pending, feed_poll_due cost exactly the same on both sizes, the last two nothing; events_as_of and eligible_work grow by at most the added rows + txns (the availability table, once). Before: eligible_work 1187 → 1627 rows (+440 for +140 rows) and open_episodes 660 → 900. `selection_snapshots_and_replay_equal_the_full_scan_path`: a 25-min run with a failing item gives the same `TRANSPORT_INVOKED` sequence (class, kind, item, episode, grant instant), snapshot and replay with the aggregates as with SQL reads and full rescans |

| autonomous service: every tick closes what reached its bound (LOCAL_PERSISTENCE_FAILED 120 s after the network end while the save task is still blocked, INTERRUPTED 600 s after TRANSPORT_INVOKED, DEAD run 600 s after its start, replacement, poison) and hands work to worker threads; the owner never runs or waits on a task; late results are fenced | `service.FomcService.tick/fetch_in_flight`, `collector.inline/dispatch_run`, `collector.fetch` (TRANSPORT_INVOKED and task registration under one lock) | service: autonomous capture (anchor, O300, zero); hung save: nothing before +120 s, LPF at the first tick after it while the task is still parked, >= 24 owner ticks during the hang, polling and attempt #2 continue around the parked task, its release writes no RESPONSE or LATE_EVIDENCE; task stuck after TRANSPORT_INVOKED: INTERRUPTED at +600 s, attempt #2 counted after it, late result fenced; hung processing run: DEAD and replaced at +600 s, INTERNAL_PROCESSING_ERROR at +1200 s, feed polled throughout, never a zero, one outcome; network stall cut by the physical deadline (scaled to 0.5 s) while the owner ticked >= 10 times |
| restart: a new owner interrupts every earlier-epoch attempt at once; no budget or key is recreated | `collector.reconcile` (epoch), unique keys and durable attempt counts | service `restart_interrupts…`: the dead owner's parked attempt INTERRUPTED (non-current epoch) when the new owner starts, attempt #2 in the same episode, keys unique; soak: two crashes |
| production TLS connector: SNI, chain and hostname verification; an invalid certificate is SOURCE_UNAVAILABLE, never an exception out of the fetch | `transport.HttpsConnector.context/wrap`, `transport._connect` (TLS failures mapped) | tls (local CA and certificates, server on a Unix socket, production `connect` and `wrap`): valid certificate: 200 with SNI `www.federalreserve.gov`, TLS >= 1.2; wrong name, expired, self-signed: SOURCE_UNAVAILABLE with the verification error, no request sent; the system trust store rejects the local CA; the context requires CERT_REQUIRED and hostname checking. Found and fixed: a certificate failure used to escape the transport as an exception |
| STORE_OPENING_RULE: an existing store with another schema version (including the unversioned stores of earlier checkpoints) or another spec hash is rejected before any write; no migration | `store._admit_existing` (read-only; `immutable` when there are no WAL frames), `SCHEMA_VERSION = fomc-store-v2` | store_opening: unversioned, later version, other spec: `StoreRejected`, every byte of the directory unchanged (also with pending WAL frames), no owner lock taken; a current store reopens and a new store is versioned |
| prolonged run with faults, two owner crashes and restarts, verified after reopening | `soak.run/verify` | soak (8 h in the suite; 26 h by CLI, ~95 s): every hourly snapshot re-reads identically at its (T, H), verified replay of a subset equals them, `verify_health` passes, one processing outcome per record, <= 1 open attempt at the end, unique keys, <= 6 attempts per key, <= 120 requests per LIVE item, limiter spacing and window across restarts, every zero cycle LIVE_ELIGIBLE with nothing outstanding and every earlier feed record terminal before B, LPF without a late record, every recheck due long enough ago served. 26 h result: 3 boots, 6325 transactions, 1577 attempts and requests, 1557 cycles (1547 zero), 6 revisions, 5 rechecks served, 25 snapshots re-read, 6 replayed |

## Capture blocker: COMMIT_FSYNC_120S

- **Rule at stake.** A local save that starts before +120 s (after the network end) and finishes at or
  after it must never create a RESPONSE; at +120 s the owner commits LOCAL_PERSISTENCE_FAILED.
- **What holds.** The admission check runs inside the write transaction; a save blocked anywhere
  *before* the COMMIT (raw write, `fsync` of the raw file, retries) is closed by the owner at +120 s
  and its late bytes never become RESPONSE or LATE_EVIDENCE (tested, also through the service).
- **What does not.** SQLite's `COMMIT` (its `fsync` under `synchronous=FULL`) runs after the check and
  can be neither bounded nor cancelled: a COMMIT stalled in `fsync` past +120 s still makes the
  RESPONSE durable, and while it stalls it holds the store's write lock, so the owner can neither tick
  nor write LOCAL_PERSISTENCE_FAILED. Reproduced by two strict-xfail tests (a COMMIT that returns 130 s
  after the check; a COMMIT blocked while the owner tries to tick):
  `python -m pytest tests/crypto/test_fomc_service.py -k capture_blocker --runxfail`.
- **Why no fix here.** Every durable decision is itself a COMMIT that can stall the same way. The only
  design that proves "durable before +120 s" is two-phase (write the record, then a confirmation whose
  recorded check instant is before +120 s). It contradicts the spec's PROCESSABLE rule (fixed by the
  record's own transaction), and the owner would still wait behind the single SQLite writer. Killing
  the process does not undo WAL frames already written. The resolution is therefore a spec decision:
  either amend the bound to the admission-check instant and treat a stalled `fsync` as a storage fault
  with an operational alarm, or adopt two-phase admission with bounded store waits for the owner.
  `CAPTURE_BLOCKERS` stays non-empty until then.

## Capture protocol (for the day the blocker is lifted)

Preconditions: the blocker's resolution merged, its two tests passing as ordinary tests and
`CAPTURE_BLOCKERS = ()`; an NTP-disciplined host (`causal_predicates.production_requirement`); a new,
empty store directory (a store is never reused across schema versions: STORE_OPENING_RULE).

```
cd ~/HyprL && git fetch origin && git checkout <reviewed commit> && git status --short   # clean tree
python -c "from scripts.trading_lab.fomc import spec; print(spec.verify_spec_binding())"
# expected: ba6a01e5f12e810ecde89711304c278862d147de298c63e7602a8359fa18e678
python -m pytest tests/crypto/test_fomc_*.py -q                  # everything passes, no xfail left
python -m scripts.trading_lab.fomc.soak --hours 26              # ends with "soak verified"
timedatectl show -p NTPSynchronized --value                      # yes
STORE=/srv/hyprl/fomc/store-$(date -u +%Y%m%dT%H%M%SZ)           # a new directory, never an old store
nohup python -m scripts.trading_lab.fomc.service --store "$STORE" --tick 1 >> "$STORE.log" 2>&1 &
```

During capture, read through a separate process that never starts a collector (opening a `FomcStore`
adds no row; a second owner of the same store is refused):

```
python - "$STORE" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from scripts.trading_lab.fomc import snapshot
from scripts.trading_lab.fomc.store import FomcStore
store = FomcStore(Path(sys.argv[1]), wall_clock=lambda: datetime.now(timezone.utc))
snap = snapshot.events_as_of(store, datetime.now(timezone.utc))
print(snap["read_state"], snap.get("discovery"), snap.get("health"), snap["H"], snap["identity"])
PY
```

Stop with `kill -INT <pid>` (a kill or a crash is also safe: the next start interrupts the old epoch's
attempts). Afterwards, offline and on a copy of the store, verify a recorded `(T, H)` with
`snapshot.replay(store, T, H)` and `snapshot.verify_health(store, H)`.

## Remaining limits

Not proven by this slice, or outside it:

- The capture blocker above (`COMMIT_FSYNC_120S`).
- No real network run (by mandate): the TLS connector is proven against a local server and local
  certificates over a Unix socket; public DNS, TCP to port 443 and the provider's real certificate
  chain are first exercised on capture day.
- Spec deviation, liveness only: the service keeps one logical fetch in flight at a time, whereas
  `selection.work_conserving` forbids serializing logical fetches below FIX15 and the one-in-flight
  rules. A slow fetch (up to 60 s per physical attempt) or a task stuck until its 600 s bound delays
  every other fetch, feed polls included. It never creates a zero (no cycle without a poll), an extra
  request or a budget. To decide before acceptance: concurrent logical fetches, or accept the deviation.
- Bounds are closed at the first owner tick at or after them (1 s in production, never before). A hung
  task keeps its thread until the process exits; it is fenced, not killed. `settle_s` (waiting for
  workers between ticks) exists only for simulated clocks.
- The step-driven path (`Collector.step`, used by most tests and the demo) still runs tasks inline in
  the caller's thread; only the service gives the non-blocking guarantees.
- `SNAPSHOT_WATERMARK`: not written, because it is not needed to satisfy `live_watermark`. The last
  transaction of a store is always UNRESOLVED (its V must come later), so a read at T is resolved
  exactly when some later transaction has already been resolved by a later verified response with
  avail > T. A watermark transaction would itself wait for that same verified response, so it could
  not make any read resolvable earlier; reads past the last resolved avail stay
  `FOMC_CAUSAL_VISIBILITY_UNRESOLVED` (never a zero) until capture resolves them.
- Source health, interpretation choices: `SOURCE_NOT_CHECKED` is exposed only when a surface has no
  check within P(T) (the spec names no staleness interval for reads, so none is derived); the optional
  `RATE_LIMITED` diagnostic is not written; a successful check is a row with `result_state` null (no
  failure state), its success record being the processing outcome or cycle conclusion of the same
  transaction.
- Health replay proves consistency, not truth: `check_at` repeats a local wall reading (the record's
  `wall_at_receipt`, or the transaction's `wall_at_commit`). Replay proves that each row repeats its
  durable source exactly; it cannot prove the reading was correct, and a consistent rewrite of a
  reading together with every copy of it (or of the whole store) is not detectable: the store is not
  signed. `wall_at_receipt` of a CLOCK_VERIFIED record is additionally tied to its Date/Age within the
  90 s clock-check tolerance; an unverified record's reading has no such tie.
- A store holding an integrity diagnostic can never pass verified replay (its raw stays corrupt and is
  never repaired); `verify_health` checks its health rows on their own.
- Normalized fields, interpretation choices (the spec fixes the vocabulary, not these values except
  `UNKNOWN` for an immediate release): `timestamp_semantics` describes the declared release
  (`EXACT_INSTANT` when parsed, `UNKNOWN` otherwise); `declared_release_trust_verdict` is
  `TRUSTED_EXACT` for a parsed EST/EDT claim and `UNKNOWN` otherwise; `timestamp_trust_verdict`, for
  content availability, is `UNTRUSTED` (the design's effect: `source_available_at` null, only
  `observed_at` and avail bound it); `revision_id` is the `(source_item_id, content hash)` key.
- In-memory work that still grows with the store, once per derivation and never per item:
  `availability` (the causal table, O(transactions)) inside `events_as_of` and `eligible_work`, the
  discovery lookup of the latest feed record, `open_episodes` walking every manifest entry and
  operator record, `derive_terminals` every episode. The head aggregates serve only views at the head;
  an older view (a snapshot at an old H) rescans, which is exact but linear.
- No store migration, by rule (STORE_OPENING_RULE): stores of earlier checkpoints are rejected, not
  upgraded; the slice has no production store.
- Official historical fixtures and real capture are separate, later steps.
