# FOMC V1 — offline integrated slice

Implements `docs/artifacts/fomc_capture_spec_v1.json` **revision 22**
(`ba6a01e5f12e810ecde89711304c278862d147de298c63e7602a8359fa18e678`, pinned in
`scripts/trading_lab/fomc/spec.py` and checked by `verify_spec_binding`). The JSON stays
authoritative. This is an **offline slice**: every source is synthetic and served by a local
Unix-socket provider; no Federal Reserve request, fixture capture or live run is part of it.

```
python -m scripts.trading_lab.fomc.demo          # the executable path, end to end
python -m pytest tests/crypto/test_fomc_*.py     # 113 tests
```

`feed -> durable raw -> classification -> primary acquisition -> revision + observation link ->
cycle -> events_as_of(T, H) -> reopen -> offline replay`

| module | role |
|---|---|
| `spec.py` | revision-22 constants, canonical hashing, spec binding |
| `identity.py` | URL admission (scheme, host, userinfo, port, percent, query, fragment), `source_item_id`, durable keys |
| `clock.py` | strict `Date`/`Age`, per-response clock verdict, canonical timestamps |
| `store.py` | SQLite WAL/FULL, `commit_seq`, append-only rows, unique keys, immutable content-addressed raw (published once with `os.link`, never overwritten) with digest checks; an append-only in-memory mirror refreshed with only the rows committed since its last refresh, served as consistent `StoreView`s bounded by a horizon; read accounting |
| `ledger.py` | epochs (fencing), atomic `TRANSPORT_INVOKED`, one outcome per attempt, `LATE_EVIDENCE`, episodes |
| `limiter.py` | FIX15 rolling window, spacing, embargo |
| `transport.py` | manual redirect loop, per-hop admission and grant, 60 s deadline from each grant over DNS/TCP/TLS/write/headers/body/decoding (stage timeouts, watchdog, late-result checks), bounded decoded body |
| `parsing.py` | Content-Type gate, strict UTF-8, secure RSS (expat), token-bounded HTML anchors, grammars |
| `processing.py` | one terminal outcome per PROCESSABLE record, feed classification + cycle, primary classification + revision/link (27 normalized fields), fenced runs, poison guard, the record's source-health result |
| `health.py` | source health per surface: mappings, precedence, the persisted row, the exposed state |
| `state.py` | LIVE_ELIGIBLE, server-attested `avail`, `NOW_LB`, anchors, obligations, episodes, item conclusion, RESOLVE validity |
| `collector.py` | single owner (flock + epoch), fetch/commit, reconciliation, episodes, class rotation and per-class ordering (`select`), operator actions, manifests, integrity diagnostics |
| `snapshot.py` | P(T) under horizon H, read state, 9-step selection, discovery state, source health, identity, read-time raw dependency checks, verified replay (health re-derived) |
| `synthetic.py`, `demo.py` | simulated clock, local provider, fixtures, the demo |

## Invariant → code → test

| rule | code | tests |
|---|---|---|
| F08 raw first, F72 corruption fails closed | `collector.commit`, `store.put_raw/read_raw`, `processing.process_record`, `collector.verify_integrity/_diagnose`, `snapshot._verify_dependencies/replay` | persistence `raw_is_content_addressed…`; snapshot `test_7`; hardening: corrupt or missing raw fails `events_as_of` without prior verification (old `(T, H)` too, revisions and discovery cycles), `put_raw` never overwrites, corrupt slot → the new attempt is `LOCAL_PERSISTENCE_FAILED` (no RESPONSE, no anchor, no recheck satisfaction, no LIVE availability; backfill, LIVE acquisition and LIVE recheck covered) + integrity diagnostics on older records, file never repaired |
| F09/F10/F74 revisions keyed (item, hash), mode-neutral links | `processing.classify_primary` | snapshot `test_1`, `test_2`, `test_7` (A-B-A) |
| `normalized_minimum_fields` (27): 21 immutable properties of the revision, 4 per observation (`observed_at`, `source_observation_id`, `raw_artifact_identities_and_hashes`, `rss_guid_if_available`), `observation_mode` and `ingested_at` on both (on the revision: creation provenance and the avail of its creating transaction) | `spec.NORMALIZED_KEYS/REVISION_FIELDS/OBSERVATION_FIELDS`, `processing.classify_primary` (REVISION row written once, LINK row per observation), `snapshot._item_state` (`ingested_at` = avail within P(T)) | normalized: the split covers exactly the 27 spec names; nulls where prescribed (`content_source_available_at`, `source_updated_at`, `declared_release_at` for immediate or unparsed release, `observed_at` when unverified, GUID when none was listed); backfill → LIVE (one unchanged row, backfill `observed_at` = the actual collection instant, never the release time), content correction D1 → D2 (new revision, D1 row intact, same item), A-B-A (V_A reused, one link per observation) |
| source health per surface (`source_health`, `failure_classification_v1.precedence`), durable before visible | `health.for_attempt_outcome/for_record/row`; committed with its outcome by `ledger.commit_attempt_outcome` and `processing.finish_run/poison` | health: the six primary cases (success, healthy negative, CLOCK_UNVERIFIED, PARSER_FAILED, PARSER_FAILED over CLOCK_UNVERIFIED with the verdict kept, SOURCE_UNAVAILABLE), each in its outcome's transaction, never on the feed surface; feed zero; LOCAL_PERSISTENCE_FAILURE, INTERRUPTED, RAW_CORRUPTION → `NO_PROVIDER_HEALTH_STATE`; cancellation → no result |
| health exposed only under FOMC_RESOLVED, within P(T), bound to identity, replayed | `health.exposed`, `snapshot.events_as_of`, `snapshot._replay_health` | health: SOURCE_NOT_CHECKED before any primary check in P(T), latest check within P(T), absent when unresolved or RETROSPECTIVE_SOURCE, identity = hash of the payload with health, replay equal at the same (T, H), a copy with altered health → other identity and `ReplayFailed` |
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

## Not in this slice

- No real TLS/HTTPS run: `HttpsConnector` exists but is never exercised (no network by mandate).
- No background daemon or periodic tick: the collector is step-driven. How each bound is imposed:
  - 60 s physical deadline: during I/O (stage timeouts, socket watchdog, late-result checks).
  - 120 s save deadline: by the admission check inside the write transaction and by `reconcile()`, on
    the injectable monotonic clock. **Exact scope of that check:** it runs after `BEGIN IMMEDIATE` and
    before the rows are inserted and before SQLite's `COMMIT`. The `COMMIT` itself (its `fsync` under
    `synchronous=FULL`) and the `fsync` calls of `put_raw` are not bounded by this code: a `COMMIT`
    that starts before +120 s and blocks can make the RESPONSE durable after +120 s, and `reconcile()`
    in the same process waits for the same store lock, so it cannot write `LOCAL_PERSISTENCE_FAILED`
    meanwhile. Tests exercise slow operations *before* the check (+119.9 / +120.0 / +120.1 s); no test
    shows a blocking `fsync` bounded, and none is claimed.
  - 600 s attempt and run deadlines: at the next trigger (`step`, `reconcile`, `process_pending`,
    `commit`) and by fences that discard late results. They are **not** imposed by a timer that
    interrupts a running task: a processing run that hangs inside the owner's own thread blocks that
    owner until it returns (its late result is then fenced and the run marked DEAD), and an attempt
    whose task dies silently is only interrupted when a trigger runs after its deadline.
  - Liveness of the in-process task registry assumes one owner process; tasks of an earlier process
    are detected by epoch (immediately), not by the deadline.
- `SNAPSHOT_WATERMARK`: not written, because it is not needed to satisfy `live_watermark`. The last
  transaction of a store is always UNRESOLVED (its V must come later), so a read at T is resolved
  exactly when some later transaction has already been resolved by a later verified response with
  avail > T. A watermark transaction would itself wait for that same verified response, so it could
  not make any read resolvable earlier; reads past the last resolved avail stay
  `FOMC_CAUSAL_VISIBILITY_UNRESOLVED` (never a zero) until capture resolves them.
- Source health, interpretation choices: `SOURCE_NOT_CHECKED` is exposed only when a surface has no
  check within P(T) (the spec names no staleness interval for reads, so none is derived); the optional
  `RATE_LIMITED` diagnostic is not written; `check_at` is a local wall reading kept as provenance
  (never a causal input); a successful check is a row with `result_state` null (no failure state),
  its success record being the processing outcome or cycle conclusion of the same transaction.
- Normalized fields, interpretation choices (the spec fixes the vocabulary, not these values except
  `UNKNOWN` for an immediate release): `timestamp_semantics` describes the declared release
  (`EXACT_INSTANT` when parsed, `UNKNOWN` otherwise); `declared_release_trust_verdict` is
  `TRUSTED_EXACT` for a parsed EST/EDT claim and `UNKNOWN` otherwise; `timestamp_trust_verdict`, for
  content availability, is `UNTRUSTED` (the design's effect: `source_available_at` null, only
  `observed_at` and avail bound it); `revision_id` is the `(source_item_id, content hash)` key.
- Manual retry of `HISTORICAL_BACKFILL` work: `manual_retry` resolves the URL through the item's
  CANDIDATE, which a backfill-only item does not have, and its order key uses manifest 0. Not fixed here.
- Read cost: views remove repeated store reads; some derivations still rescan the view in memory per
  item (for example `NOW_LB` inside obligations), i.e. CPU of order items × records per step.
- No store migration: rows written by earlier checkpoints (LINK `mode`, REVISION without the new
  fields, no health rows) are not upgraded; the slice has no production store.
- Official historical fixtures and real capture are separate, later steps.
