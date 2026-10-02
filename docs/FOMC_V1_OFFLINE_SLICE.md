# FOMC V1 — offline integrated slice

Implements `docs/artifacts/fomc_capture_spec_v1.json` **revision 25**
(`b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece`, superseding revision 24
`235e474cfc9a50daa00a0835d513af12d7a5e7485af26775b589d573ce6fdb36`; pinned in
`scripts/trading_lab/fomc/spec.py` and checked by `verify_spec_binding`). The JSON stays
authoritative. This is an **offline slice**: every source is synthetic and served by a local
Unix-socket provider; no Federal Reserve request, fixture capture or live run is part of it.

**Status (2026-10-02):** runtime revision 25 finished and verified offline (HTML-context content identity
with separate CANONICAL/RAW_FALLBACK domains, FIX15 grant journal; below). The revision-23 real pilot was
**interrupted after 1 h 23 min** by an abrupt termination of the WSL VM (2026-10-01 ~19:44 UTC, see "Pilot
capture"); its store is kept unchanged and its closure stays at 2026-10-02T18:31:14Z (re-armed, frozen
closure code). Its duration will therefore be NOT_ACCOMPLISHED, and the gated revision-25 successor
(90 min) is programmed but will refuse to start under the authorized conditions. **Runtime verdict: READY.** The two runtime blockers are closed. `COMMIT_FSYNC_120S` is closed by spec
revision 23 (the 120 s bound governs admission, a stalled store is a storage incident) and its
implementation. Serialized fetches are replaced by central grant dispatch with concurrent logical
fetches (`work_conserving`). `service.CAPTURE_BLOCKERS` is empty. Nothing has been captured: the first
real run is the capture protocol below; what remains unproven offline is listed under "Remaining limits".

```
python -m scripts.trading_lab.fomc.demo                         # the executable path, end to end
python -m pytest tests/crypto/test_fomc_*.py                    # 196 tests (one private proof skips without the local fixtures)
python -m scripts.trading_lab.fomc.soak --hours 26              # prolonged run, verified (~70 s)
python -m scripts.trading_lab.fomc.service --store DIR --check  # capture preflight, no network
```

`feed -> durable raw -> classification -> primary acquisition -> revision + observation link ->
cycle -> events_as_of(T, H) -> reopen -> offline replay`

| module | role |
|---|---|
| `spec.py` | revision-22 constants, canonical hashing, spec binding |
| `identity.py` | URL admission (scheme, host, userinfo, port, percent, query, fragment), `source_item_id`, durable keys |
| `clock.py` | strict `Date`/`Age`, per-response clock verdict, canonical timestamps |
| `store.py` | SQLite WAL/FULL, `commit_seq`, append-only rows, unique keys, immutable content-addressed raw (published once with `os.link`, never overwritten) with digest checks; an append-only in-memory mirror refreshed with only the rows committed since its last refresh, served as consistent `StoreView`s bounded by a horizon, with incremental aggregates registered by derivations; read and in-memory work accounting |
| `ledger.py` | epochs (fencing, limiter epoch start), atomic `TRANSPORT_INVOKED` with its initial GRANT, the FIX15 grant journal, one outcome per attempt, `LATE_EVIDENCE`, episodes |
| `limiter.py` | FIX15 rolling window, spacing, embargo |
| `transport.py` | manual redirect loop, per-hop admission and grant, 60 s deadline from each grant over DNS/TCP/TLS/write/headers/body/decoding (stage timeouts, watchdog, late-result checks), bounded decoded body |
| `parsing.py` | Content-Type gate, strict UTF-8, secure RSS (expat), token-bounded HTML anchors, grammars |
| `processing.py` | one terminal outcome per PROCESSABLE record, feed classification + cycle, primary classification on the original document + revision keyed by content identity / link with its own raw (27 normalized fields), fenced runs, poison guard, the record's source-health result |
| `canon.py` | FOMC_CANON_V2 / FOMC_CONTENT_IDENTITY_V2: a strict position-keeping HTML tokenizer; the three Cloudflare spans neutralized only as real tokens in their real context, effective addresses kept, anything else RAW_FALLBACK; a domain-separated identity |
| `health.py` | source health per surface: mappings, precedence, the persisted row, the exposed state |
| `state.py` | LIVE_ELIGIBLE, server-attested `avail`, `NOW_LB`, anchors, obligations, episodes, item conclusion, RESOLVE validity |
| `collector.py` | single owner (flock + epoch), fetch/commit, reconciliation, episodes, class rotation and per-class ordering (`select`), operator actions, manifests, integrity diagnostics |
| `snapshot.py` | P(T) under horizon H, read state, 9-step selection, discovery state, source health, identity, read-time raw dependency checks, verified replay (health re-derived) |
| `pilot.py` | pilot tooling, no provider request: snapshot reads, fixture export, supervision, closure (copy, re-read and replay of every recorded read, health replay, audit, FIX15 proved over the grant journal, recheck status, separate integrity and duration verdicts, short-pilot criteria), the gated successor launch |
| `service.py` | the autonomous owner: storage watch and incidents, task closure at the 60/120/600 s bounds, central grant dispatch to concurrent fetch workers, processing workers, clean stop, restart; the capture entry point and its `--check` preflight |
| `soak.py` | the prolonged synthetic run (statement pages with per-response Cloudflare bytes) with faults, a storage incident, crashes, restarts and a clean stop, and its verification |
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
| autonomous service: every tick closes what reached its bound (LOCAL_PERSISTENCE_FAILED at the 120 s admission bound while the save task is still blocked, INTERRUPTED 600 s after TRANSPORT_INVOKED, DEAD run 600 s after its start, replacement, poison); the owner never runs or waits on a task and waits for the store at most `owner_wait_s`; late results are fenced | `service.FomcService.tick`, `collector.close_attempts/inline/dispatch_run`, `store.locked/set_wait_bound` (StoreBusy) | service: autonomous capture (anchor, O300, zero); hung save: nothing before +120 s, LPF at the first tick after it while the task is parked, the feed polled around it, attempt #2 after it, its release admits nothing; task stuck after TRANSPORT_INVOKED: INTERRUPTED at +600 s, the item has one fetch in flight meanwhile, late result fenced; hung processing run: DEAD and replaced at +600 s, INTERNAL_PROCESSING_ERROR at +1200 s, polls throughout, never a zero; network stall cut by the physical deadline (scaled to 0.5 s) while the owner ticked and polled |
| concurrent logical fetches, `work_conserving` and `selection.grant_dispatch` (rev 23): at each instant FIX15 admits a start, a waiting validated continuation (smallest TRANSPORT_INVOKED) first, otherwise one selection whose TRANSPORT_INVOKED commits before the start is consumed; one feed poll and one fetch per item in flight, no other cap | `service._dispatch/_next_continuation/_continuation`, `collector.begin/complete`, `limiter.try_grant/register/suspended`, `transport.fetch(started, continuation)` | service `slow_fetch_lets_other_items_and_polls_start…`: while one item's fetch is held, the two other items are acquired and >= 4 polls start, starts exactly 10 s apart, never a second fetch of the held item; `continuations_rotation_limiter_budgets_and_late_results…` (5 items: slow, redirect, failing, two normal; 30 min): one fetch per item and one poll at a time, the redirect hop is the next physical request with one TRANSPORT_INVOKED, every decision follows the class rotation from durable records, FIX15 holds over every physical start, <= 6 attempts per key, the failing item blocks nobody, the slow item's attempt INTERRUPTED at 600 s, its late bytes LATE_EVIDENCE that never anchor, anchor from its attempt #2 |
| task registries safe against late workers: a worker only removes its own entry; an old run can neither erase, kill nor replace a newer run | `collector._registry/_forget_run/_mark_dead` (compare-and-delete, DEAD once), `service._spawn` (own task only), `store.read_aggregate` (aggregates copied under the lock), `_Mirror.field_index` (published complete) | service `old_runs_worker_never_erases…`: run 1 hung, DEAD at 600 s, run 2 started; run 1's late return leaves run 2 registered and alive, one RUN_DEAD (run 1), run 2 commits the outcome. With the former unconditional removal the test fails (run 2 killed, poisoned) |
| LOCAL_SAVE_ADMISSION_BOUND_V1 (rev 23, F80, FOMC248/249): admission checked inside the transaction before its insertions, at 120 s expired; an admitted COMMIT may become durable later and is valid; observed_at and avail untouched | `ledger.commit_response(admit)`, `collector._commit` | local_deadlines: +119.9 s admitted, +120.0 and +120.1 s nothing; service `admitted_commit_stalled_past_120_s…` (the old blocker's physical delay, now the contract): the COMMIT returns 130 s after the check, the record is the outcome, `observed_at` = receipt reading (before the stall), available only after the release (avail > release instant), one request, no LPF |
| STORAGE_INCIDENT_V1 (rev 23, F81, FOMC250): detection without the store, alert, no grant, no durable decision while the store cannot write, first-outcome reconciliation, no repair request, no budget recreated | `store.stalled` (operation markers outside the lock), `service.watch_storage/_alert/_record_incidents`, monitor thread, `limiter.suspended` | service: COMMIT stalled 130 s: incident from 10 s, one STARTED and one ENDED alert, no start during it, the owner keeps ticking with StoreBusy, STORAGE_INCIDENT record; `stalled_store_stops_grants…`: a COMMIT stalls 300 s while another item's admission bound passes: no grant and no decision during it, then the stalled record stands and the other attempt gets LPF at the first commit, no repair request, attempt #2 in the same episode, keys unique; a stalled raw write is an incident too, LPF still committed at 120 s since SQLite can write; the owner's own COMMIT stalling is detected by the monitor without the owner |
| clean stop and restart: a clean stop leaves no attempt without outcome and grants nothing more; a new owner interrupts an earlier epoch's attempts at once; no budget or key recreated | `service.stop`, `collector.close_attempts` (epoch), unique keys and durable attempt counts | service `clean_stop…` (nothing left, nothing interrupted by the next owner), `restart_interrupts…`; soak: two crashes and a clean stop |
| FOMC_CONTENT_IDENTITY_V1 (rev 24, F82, FOMC251-256): `raw_sha256` stays the integrity of the received bytes; a revision is keyed by the SHA-256 of FOMC_CANON_V1 canonical bytes; admission parses the original document; reads that rely on a content identity verify the raws it comes from; replay re-derives every link's identity | `canon.canonicalize`, `processing.classify_primary` (REVISION `content_hash`/`first_raw_sha256`/`content_identity`, LINK `content_sha256`/`canonicalization`/own raw), `snapshot.content_identity/_item_state`, `snapshot.replay` | content_identity: only the three spans change and the re-keyed address stays; text, rate, date, title, release line, effective address or share target changed: a new identity; a marker in the text, a link outside `<a>`, an altered span, an altered or misplaced challenge script, invalid UTF-8, odd or non-printable values, a bound exceeded: REFUSED, raw identity, no merge; per-response Cloudflare bytes: one revision, one record, raw and link each (FOMC251); A-B-A reuses A (FOMC254); backfill then LIVE: one revision, LIVE availability only from the LIVE link (FOMC255); a page rejected on its original document stays rejected; a corrupt newest raw with an equal identity fails the read and replay closed (FOMC256); same (T, H) after reopening and replay, identical; an altered stored identity fails replay. Private proof on the official fixtures (local only); soak 26 h with Cloudflare pages keeps 6 revisions |
| HTML CONTEXT AND IDENTITY DOMAINS (rev 25, F82, FOMC257-258): FOMC_CANON_V2 recognizes a span only as a real token at its byte position (the only double-quoted `href` of an `<a>`, a `<span>` with exactly `class`/`data-cfemail` and its exact text, the parameters inside the digest-checked attribute-less `<script>` right before `</body>`); a marker in a comment, in another attribute's value or in a raw-text element (`textarea`, `title`, `script`, `style`, ...) and any unreadable structure is RAW_FALLBACK; nothing is reserialized; the identity hashes {identity, canonicalizer, domain, bytes_sha256} | `canon._tokens/_spans/canonicalize/identity`, `processing.classify_primary` (REVISION `content_identity` with domain and bytes digest; LINK `canonicalization`) | content_identity: the two reported counterexamples (`<a title='href="/cdn-cgi/l/email-protection#H"'>` and a cf_email span inside `<textarea>`) and a commented-out link, each with two XOR keys: RAW_FALLBACK, two identities, the same raw twice meets again; end to end through classification and the snapshot: separate revisions, the read follows the newest observation's own revision; re-canonicalizing canonical bytes: RAW_FALLBACK, equal bytes digest, different identity; real Cloudflare variations still merge (FOMC251, A-B-A FOMC254, backfill then LIVE FOMC255, corruption FOMC256, replay). On the 33 real raws of the rev23 pilot: all CANONICAL (65/32/33 spans), one identity per URL, admission parse unchanged; the three official fixtures CANONICAL |
| FIX15 GRANT JOURNAL (rev 25, F83, FOMC259-260): every consumed grant has a durable GRANT row (epoch, order, monotonic instant, attempt, item or FEED, hop, admitted URL), the initial one in its TRANSPORT_INVOKED transaction, a continuation committed before the dispatcher registers it and hands it over, a grant abandoned before any TRANSPORT_INVOKED with its reason; each fetch records its grant count; the EPOCH carries the limiter epoch start | `ledger.grant_row/journal_grant/transport_invoked(grant=)`, `collector.journal_grant/_invoke/_invoke_or_journal`, `service._dispatch` (journal, then `register`, then release), `transport.fetch(continuation=, journal=)`, `pilot.fix15_journal` | grant_journal: a 3-hop redirect chain (hops 0-3, URLs equal to the chain, orders consecutive, PROVEN); a failure after a redirect (2 grants, count 2); a grant cancelled after TRANSPORT_INVOKED and one abandoned before it (fenced epoch); a lost continuation row, a lost initial row, a disagreeing count: NOT_PROVEN; two grants 1 s apart: FAIL; a continuation whose journal row cannot commit is never granted and sends nothing, then proceeds once the store writes; a stop while a continuation waits consumes no grant; a restart: two epochs, orders from 1 in each, embargo per epoch. Soak 26 h: PROVEN over 1576 grants and 3 epochs. Deadlines (from the grant), no refund, continuations first and single use are unchanged |
| pilot closure states what it proves (rev23-compatible, commit `909399fa`) | `pilot.audit/_in_flight_overlaps/fix15_coverage/recheck_status/verify_copy/pilot_duration/close` | pilot: overlaps per source item across classes, the feed apart; FIX15 over redirect hops NOT_PROVEN, never PASS; a premature stop keeps integrity VALID but duration NOT_ACCOMPLISHED; unresolved reads re-read and replayed; recheck status from durable records (SATISFIED, IN_PROGRESS, SUSPENDED, PENDING_DUE, NOT_YET_DUE by NOW_LB) |
| production TLS connector: SNI, chain and hostname verification; an invalid certificate is SOURCE_UNAVAILABLE, never an exception out of the fetch | `transport.HttpsConnector.context/wrap`, `transport._connect` (TLS failures mapped) | tls (local CA and certificates, server on a Unix socket, production `connect` and `wrap`): valid certificate: 200 with SNI `www.federalreserve.gov`, TLS >= 1.2; wrong name, expired, self-signed: SOURCE_UNAVAILABLE with the verification error, no request sent; the system trust store rejects the local CA; the context requires CERT_REQUIRED and hostname checking. Found and fixed: a certificate failure used to escape the transport as an exception |
| STORE_OPENING_RULE: an existing store with another schema version (unversioned, `fomc-store-v2` of the previous checkpoint, or later) or another spec hash (including revision 22) is rejected before any write; no migration | `store._admit_existing` (read-only; `immutable` when there are no WAL frames), `SCHEMA_VERSION = fomc-store-v5` | store_opening: unversioned, v2, v3 (the rev23 pilot's), v4 (revision 24), v6, revision-22 spec, other spec: `StoreRejected`, every byte of the directory unchanged (also with pending WAL frames), no owner lock taken; a current store reopens and a new store is versioned; service `capture_preflight_refuses_an_incompatible_store` |
| prolonged run with faults, a storage incident, two owner crashes and restarts and a clean stop, verified after reopening | `soak.run/verify` | soak (8 h in the suite; 26 h by CLI, ~75 s): every hourly snapshot re-reads identically at its (T, H), verified replay of a subset equals them, `verify_health` passes, one processing outcome per record, no attempt left without outcome, one fetch per item and one poll in flight, <= 6 attempts per key, <= 120 requests per LIVE item, FIX15 across restarts, no start during the incident, every zero cycle LIVE_ELIGIBLE with nothing outstanding and every earlier feed record terminal before B, LPF without a late record, every recheck due long enough ago served. 26 h: 3 boots, 6322 transactions, 1576 attempts and requests, 1556 cycles (1544 zero), 6 revisions, 5 rechecks served, one incident (210 s) with its two alerts, 25 snapshots re-read, 6 replayed |

## Spec revision 25: HTML context, identity domains, grant journal

**Two counterexamples to revision 24.** (1) FOMC_CANON_V1 matched markers by pattern, not by HTML
context: `<a title='href="/cdn-cgi/l/email-protection#H"'>texte</a>` and
`<textarea><span class="__cf_email__" data-cfemail="H">[email&#160;protected]</span></textarea>` were
canonicalized, so two XOR keys of the same address merged although neither is a link or a span. (2) The
identity was the bare SHA-256 of the canonical bytes: a refused record kept its raw digest, so a raw equal
to another record's canonical bytes (r and t empty) shared that record's identity.

**Decision.** FOMC_CANON_V2 reads the raw with a strict tokenizer that keeps byte positions in the original
and never reserializes (comments, declarations, start and end tags with quoted or unquoted attributes,
raw-text elements up to their own end tag); a span is replaced only inside the region of a real token of
its kind; any marker elsewhere, or any structure the tokenizer cannot read unambiguously (unterminated
comment, tag, attribute value or raw-text element, `<!--` inside a script), is RAW_FALLBACK with its reason.
FOMC_CONTENT_IDENTITY_V2 = SHA-256 of the canonical serialization of `{identity, canonicalizer, domain,
bytes_sha256}`, domain CANONICAL or RAW_FALLBACK, used for revision keys, links, content comparison,
snapshots and replay; `raw_sha256` stays the integrity of the received bytes. FIX15_GRANT_JOURNAL_V1
journals every consumed grant before it is used (below, and the table above).

**Binding.** Revision 25 hash `b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece`, superseding `235e474c…` (revision 24); 83 invariants (F83), 260
cases (FOMC257-FOMC260); `FOMC_CANON_V2`, `FOMC_CONTENT_IDENTITY_V2`, `FIX15_GRANT_JOURNAL_V1`; stores
`fomc-store-v5`. Stores v4 (revision 24) and v3 (the revision-23 pilot's) are rejected; nothing migrates.

## Spec revision 24: raw integrity and content identity

**Evidence, from bytes already acquired (no new download).** In the revision-23 pilot store, 33 statement
responses of 17 URLs: every line that differed between responses of the same URL held one of three
Cloudflare spans, and nothing else differed:
- `href="/cdn-cgi/l/email-protection#H"` in `<a>` start tags (the share link and the media contact);
- `<span class="__cf_email__" data-cfemail="H">[email&#160;protected]</span>`;
- `window.__CF$cv$params={r:'<16 hex>',t:'<base64 of a Unix time>'}` in one challenge script, always the
  same 906 bytes once r and t are removed (SHA-256 `f69b046a…`), always right before `</body>`.
`H` is lowercase hex of even length: a key byte then the address XOR the key. Every value decodes to
printable ASCII: `media@frb.gov` (32 spans and 32 links) and `?body=<the page's own URL>` (33 share links).
Only the key and r/t change between responses.

**Decision (FOMC_CANON_V1, `revision_policy.content_identity`).** Re-key both e-mail forms to key 0 (the
effective address stays in the identity), empty r and t inside the digest-checked challenge script, keep
every other byte; refuse (raw identity, no merge) on any marker outside these exact structures, invalid
UTF-8, a value that does not decode to printable ASCII, an altered script, or more than 16/16/1 spans.
Under it the 33 real responses are all CANONICAL (65 links, 32 spans, 33 parameter pairs neutralized), 16
raw identities merge into one content identity per URL, and the admission parse of every canonical
document equals that of its original. The three official fixtures are CANONICAL with the expected spans.

**Binding.** Revision 24 hash `235e474cfc9a50daa00a0835d513af12d7a5e7485af26775b589d573ce6fdb36`,
superseding `3fc2f9a7…` (revision 23); 82 invariants (F82), 256 cases (FOMC251-FOMC256); canonicalizer
`FOMC_CANON_V1`, content identity `FOMC_CONTENT_IDENTITY_V1`; stores `fomc-store-v4`. A revision-23 store
(the running pilot's) is rejected by this code, and nothing migrates it.

## Spec revision 23: the guarantee abandoned and the guarantee obtained

- **Decision** (`retry_policy.local_failure.admission_bound`, `LOCAL_SAVE_ADMISSION_BOUND_V1`): the
  120 s local bound governs the *admission* of a response record, checked atomically inside its
  transaction before its insertions. A COMMIT begun after that check may become durable later. A
  stalled COMMIT or fsync is a storage incident (`storage_incident`, `STORAGE_INCIDENT_V1`).
- **Abandoned.** The reading of revision 22 under which a local save finishing at or after +120 s
  could never create a record. It is not enforceable: a COMMIT can be neither bounded nor cancelled,
  and every durable decision is itself a COMMIT. The physical duration of durability is not guaranteed.
- **Obtained.** No record exists whose admission check ran at or after +120 s; every attempt has
  exactly one outcome; `observed_at` stays the receipt reading; a record made durable late is only
  available later (`causal_predicates.avail` from its real `commit_seq`), never earlier; nothing is
  rewritten to hide the delay. During a storage incident no grant is attributed and no durable decision
  is promised; afterwards the first committed outcome of each attempt stands, without a repair request
  or a recreated budget. The physical delay of the old blocker (a COMMIT returning 130 s after the
  check) is reproduced as an ordinary test of this contract.
- **Binding.** Revision 23 hash `3fc2f9a705d99e964208c10425c6016db4ddb375c630cbba10cbe34c4abe9ee9`
  (`spec.SPEC_HASH`, `verify_spec_binding`), superseding `ba6a01e5…` (`spec.SUPERSEDED_SPEC_HASH`);
  81 invariants (F80, F81 added), 250 cases (FOMC248-FOMC250 added); stores are `fomc-store-v3` and bound
  to the revision-23 hash, so a revision-22 store is rejected (STORE_OPENING_RULE).

## Capture protocol (prepared, not executed)

Preconditions: an NTP-disciplined host (`causal_predicates.production_requirement`); a new, empty
store directory on a local disk (a store is never reused across schema versions); the service's
stderr routed to the operator (storage alerts are `FOMC-ALERT {json}` lines).

```
cd ~/HyprL && git fetch origin && git checkout <reviewed commit> && git status --short   # clean tree
python -c "from scripts.trading_lab.fomc import spec; print(spec.SPEC_REVISION, spec.verify_spec_binding())"
# expected: 23 3fc2f9a705d99e964208c10425c6016db4ddb375c630cbba10cbe34c4abe9ee9
python -m pytest tests/crypto/test_fomc_*.py -q                 # 149 passed, no xfail
python -m scripts.trading_lab.fomc.soak --hours 26             # ends with "soak verified"
timedatectl show -p NTPSynchronized --value                     # yes
STORE=/srv/hyprl/fomc/store-$(date -u +%Y%m%dT%H%M%SZ)          # a new directory, never an old store
python -m scripts.trading_lab.fomc.service --store "$STORE" --check     # preflight ok, nothing created
nohup python -m scripts.trading_lab.fomc.service --store "$STORE" --tick 1 >> "$STORE.log" 2>&1 &
echo $! > "$STORE.pid"
```

During capture: `grep FOMC-ALERT "$STORE.log"` shows storage incidents (no request is granted while one
lasts); read snapshots through a separate process that never starts a collector (opening a `FomcStore`
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

Stop: `kill -TERM $(cat "$STORE.pid")` (clean stop: no new grant, fetches in flight finish within their
bounds and are committed, ownership released; a kill -9 or a crash is also safe, the next start
interrupts the old epoch's attempts). Afterwards, offline and on a copy of the store, verify a recorded
`(T, H)` with `snapshot.replay(store, T, H)` and `snapshot.verify_health(store, H)`.

## Pilot capture (real, revision 23: interrupted after 1 h 23 min)

The first real run of the service against `www.federalreserve.gov`, authorized for 24 h. Store, raws,
logs and fixtures stay outside Git (`redistribution.raw_storage = LOCAL_RESTRICTED`); only URLs,
digests and provenance are recorded here.

**Preconditions checked:** code `36b2e307d6b73fce5f2c430051f176347f60044f` (tests at that SHA: 150
passed), spec revision 23 `3fc2f9a7…` (`verify_spec_binding`), NTP synchronized, 868 GB free, no other
FOMC emitter (no FOMC process or unit; `services/crypto_news` declares a Fed source but nothing runs
it; the cron entry `run_sense_daily.sh` points to a missing file), working tree and stash preserved,
a new `fomc-store-v3` store outside the repository, preflight `--check` ok (nothing created).

**Launch.** `2026-10-01T18:21:14Z`, systemd user unit `fomc-pilot` (PID 56842), tick 1 s, run directory
`/home/kyo/fomc-pilot/run-20261001T182106Z/` (`store/`, `service.log`, `snapshots.jsonl`,
`alerts.log`, `pid`, `started_utc`, `code_sha`, `spec_binding`, `ntp`, `preflight`):

```
RUN=/home/kyo/fomc-pilot/run-20261001T182106Z
systemd-run --user --unit=fomc-pilot --working-directory=/home/kyo/HyprL-phase5 \
  --property=TimeoutStopSec=180 --property=KillMode=mixed \
  --property=StandardOutput=append:$RUN/service.log --property=StandardError=append:$RUN/service.log \
  -- /usr/bin/python3 -m scripts.trading_lab.fomc.service --store $RUN/store --tick 1 \
  --manifest $RUN/fixtures-manifest.json
systemd-run --user --unit=fomc-pilot-supervisor --working-directory=/home/kyo/HyprL-phase5 \
  -- /usr/bin/python3 -m scripts.trading_lab.fomc.pilot supervise --store $RUN/store --unit fomc-pilot \
  --service-log $RUN/service.log --alerts $RUN/alerts.log --snapshots $RUN/snapshots.jsonl \
  --copy $RUN/closure-copy --report $RUN/closure-report.json --close-at 2026-10-02T18:31:14+00:00
```

**Official fixtures** (acquired by the service itself as a `HISTORICAL_BACKFILL` manifest, through the
transport and the FIX15 limiter, single owner; exported to `/home/kyo/fomc-pilot/fixtures-v1/` with
bytes, headers, URLs, digests, provenance and processed fields; all checks green):

| fixture | URL | raw SHA-256 | bytes | Date | processed |
|---|---|---|---|---|---|
| summer EDT | `…/newsevents/pressreleases/monetary20260617a.htm` | `91a8ba0316f43d41cc7278fea0ce449a411b7536149c18ddd21147189586e948` | 81083 | Thu, 01 Oct 2026 18:22:25 GMT | CLOCK_VERIFIED; 2026-06-17; "For release at 2:00 p.m. EDT" → `2026-06-17T18:00:00+00:00` (EXACT) |
| winter EST | `…/newsevents/pressreleases/monetary20260128a.htm` | `49fa733577b8415b77d953b347b04e489ba389c2db54e39ccca544d6a201dbf0` | 82109 | Thu, 01 Oct 2026 18:23:05 GMT | CLOCK_VERIFIED; 2026-01-28; "For release at 2:00 p.m. EST" → `2026-01-28T19:00:00+00:00` (EXACT) |
| immediate release | `…/newsevents/pressreleases/monetary20150318a.htm` | `90e3bbea33bf00946aa4393e13e8a25de32f37cf4bb0205054b0699048c556d5` | 83666 | Thu, 01 Oct 2026 18:23:00 GMT | CLOCK_VERIFIED; 2015-03-18; "For immediate release" → declared null (IMMEDIATE) |

**First real results (first 11 minutes).** DNS, TCP, TLS (production `HttpsConnector`, system trust
store) and HTTP 200 on the feed and statement pages; 28 of 28 responses CLOCK_VERIFIED (the feed carries
`Age`, statement pages do not); 15 FAMILY candidates from the feed; LIVE primary acquisitions of the four
admissible 2026 statements (04-29, 06-17, 07-29, 09-16: NORMALIZED_REVISION_COMMITTED, anchored, LIVE
available); 10 family-path items DEFINITELY_OUT_OF_SCOPE (minutes and other releases); the first O300
recheck started; first cycles NOT_ZERO (new candidates, then an outstanding item), then
EVENTS_OBSERVED_ZERO. Resolved snapshot at `T = 2026-10-01T18:32:15.552066+00:00`, `H = 158` (`P = 144`),
identity `8adbc765c6c805d8cdc4a7709f2005a0b994f7c2b64e46e19505bacce2e71638`: discovery NOT_ZERO
(cycle 137), 4 CURRENT_REVISION LIVE-available, 2 CURRENT_REVISION backfill-only, 11 NOT_IN_V1_SCOPE,
health without failure on both surfaces. A read at T = now right after launch was resolved but its
P(T) held only the pre-capture transactions: availability needs the next verified response + 92 s, and
no rule was changed.

**Findings from real traffic.**
- *Revision churn (spec decision needed, not a code defect).* Statement pages carry bytes that change on
  every response (Cloudflare e-mail obfuscation links and the `__CF$cv$params` challenge script). The
  spec keys revisions by the raw body hash (`satisfaction.same_content` / `changed_content`), so every
  observation of an unchanged statement is a new EventRevision (the 06-17 statement fetched minutes
  apart by backfill and LIVE gave two). The implementation conforms to revision 23; no guarantee is
  weakened, but "content revision" no longer means a change of the statement. Resolving it needs a spec
  revision (a revision identity over the normalized statement content or a frozen canonicalization of
  the volatile spans); it was not changed during the pilot.
- *Transient DNS timeouts.* Six of the first 44 attempts ended `SOURCE_UNAVAILABLE` "DNS resolution
  failed: gaierror", in two bursts (18:22–18:24 and 18:33:25–18:33:45 UTC), each after about 20 s
  (resolver timeouts; `/etc/resolv.conf` points straight at 1.1.1.1/1.0.0.1 without a local cache; 0
  failures in 40 lookups of a neutral host). Classified, counted against the episode budget and
  retried per spec; not a code defect. A local caching resolver on the capture host would cut this
  budget consumption; it is an operational choice, not made during the pilot.
- *Rechecks of every anchored item.* By the spec, the anchor is the first LIVE_ELIGIBLE primary record
  whatever its classification, so the 13 LIVE-acquired family items (the 4 statements and 9 other
  releases) each carry O300/O3600/O86400/O604800 obligations; 11 O300 rechecks ran within 13 minutes.

**Monitoring.** The supervisor unit `fomc-pilot-supervisor` (PID 57012) routes every `FOMC-ALERT`
line and any service stop before the closure (`FOMC-SERVICE-DOWN`) to the system journal (identifier
`fomc-pilot`, priority crit; `journalctl -t fomc-pilot -p crit -f`) and to `alerts.log`, and records a
snapshot read every hour in `snapshots.jsonl`. Route checked with a test notice at launch.

**Closure, programmed for 2026-10-02T18:31:14Z** (24 h 10 min, so that the O86400 recheck of the first
anchors is due before it): `systemctl --user stop fomc-pilot` (clean stop), owner-lock check, a copy
through the SQLite backup API from a read-only connection (WAL frames included, never the database
file alone) plus the immutable raws, then offline on the copy: every recorded resolved snapshot
re-read and replayed at its (T, H), `verify_health`, and the audit (no attempt without outcome, <= 6
attempts per key, one fetch per item and one poll in flight, FIX15 per epoch, <= 120 requests per LIVE
item, one processing outcome per record, unique keys, no false zero). Report:
`$RUN/closure-report.json`; result notice in the journal. To run it by hand:
`python -m scripts.trading_lab.fomc.pilot close --store $RUN/store --unit fomc-pilot --copy $RUN/closure-copy-manual --snapshots $RUN/snapshots.jsonl --report $RUN/closure-report-manual.json`.

**Closure tool corrected and redeployed without touching the capture.** The first closure tool checked
in-flight overlaps per episode key, reported FIX15 as passed from the initial grants alone, gave one
verdict for integrity and duration, and re-read only resolved reads. Commit `909399fa` (rev23-compatible)
fixes all four and reports recheck status from durable records. It runs from its own detached worktree
`/home/kyo/fomc-pilot/closure-code-909399f`, so later commits cannot change it. At 18:51 UTC the original
supervisor was stopped (it holds no closure state) and `fomc-pilot-supervisor-v2` (PID 70712) started
with the same closure instant `2026-10-02T18:31:14Z`, `--planned-start 2026-10-01T18:21:14+00:00` and
`--log-offset end`. The service (PID 56842) was not restarted. A dry run on a backup-API copy of the live
store, with no service stop: copy intact, the recorded reads re-read and replayed identically, health
replay and audit green, no in-flight overlap, FIX15 initial grants PASS (71) but all physical starts
NOT_PROVEN (12 failed attempts have no hop record), 15 O300 rechecks SATISFIED, the later ones NOT_YET_DUE.
The revision-23 store keeps raw-hash revisions by design: its revision churn is expected and is not
re-keyed.

**Interruption (2026-10-01, established from the system journal).** The pilot ran in boot `f217ce34`
(17:39:05–21:43:41 CEST). From 21:32 CEST that boot logs repeated `systemd-resolved: Clock change detected`
and NTP UDP timeouts to `ntp.ubuntu.com`; its last entry is 21:43:41 CEST and it ends with **no shutdown
sequence**: the WSL VM was terminated from the host side, killing the service and the supervisor without a
stop. The store's last write (WAL) is 2026-10-01T19:44:09Z: about 1 h 23 min of capture. Four short boots
followed (21:46, 21:55, 22:07, 22:11 CEST; 2026-10-02 14:01 CEST), none running the transient units; the
current boot started 2026-10-02 14:21:35 CEST. The store, raws and frozen code were not touched. The
closure was re-armed unchanged in content: a persistent user timer `fomc-rev23-closure.timer`
(`OnCalendar=2026-10-02 18:31:14 UTC`, `Persistent=true`, so a missed instant runs at the next boot) runs
`pilot close` from `/home/kyo/fomc-pilot/closure-code-909399f` with the arguments the supervisor would
have used (`--unit fomc-pilot`, planned start 18:21:14Z, planned end 18:31:14Z), writing
`closure-report.json` and `closure.log`; the reason is recorded in `$RUN/closure_rearmed`. No
reconstruction: the interrupted attempts stay as recorded, FIX15 continuations stay NOT_PROVEN, and the
duration verdict will be NOT_ACCOMPLISHED (the service is not running at the planned end).

**DNS failures: cause.** 15 of 88 attempts were `SOURCE_UNAVAILABLE` "DNS resolution failed", each
after about 20 s (glibc stub: 5 s timeout x 2 attempts x 2 servers, no local cache). A raw UDP probe of
neutral names only (never the provider's), one query per server per second with a 2 s timeout, during the
pilot (19:22–19:42 UTC, 794 queries): 47 % timeouts (1.1.1.1 56 %, 1.0.0.1 39 %), every name alike, rcode 0
whenever an answer came, answered latencies median 42 ms, p90 ~330 ms, max 1.5 s; in 62 of 227 back-to-back
pairs both servers failed. The same boot logged NTP UDP timeouts. On 2026-10-02 (another network attachment,
192.168.36.0/24): UDP 27–54 % and TCP-53 20–27 % failures within 2 s, and from the Windows host itself ICMP
loss of 92 % to the local gateway, 96 % to 1.1.1.1 and 0 % to 1.0.0.1 with 41–321 ms jitter. **Cause:**
packet loss and jitter on the host's own network path, upstream of WSL and present on Windows, worse
towards 1.1.1.1; not the Federal Reserve, not the resolver software, not UDP alone. A resolver change would
not remove the loss (a local cache would only reduce how often lookups are exposed to it; putting 1.0.0.1
first would reduce the timeouts measured here): it is an operational proposal for the operator, not
applied. Failures keep their budgets and delays.

**Status, kept distinct.**
- Runtime: finished, revision 25 (offline proofs above).
- Capture: the revision-23 pilot ran 2026-10-01T18:21:14Z to ~19:44Z, then was killed with its VM; closure
  re-armed for 2026-10-02T18:31:14Z.
- Capture validated: no. At best integrity VALID with duration NOT_ACCOMPLISHED; FIX15 over redirect hops
  NOT_PROVEN for this store, never reconstructed.
- Not exercised by this pilot: the O604800 (7-day) recheck; a real storage incident, restart or crash;
  MANUAL_RETRY, RESOLVE and suspension against the real provider; a real redirect, PARSER_FAILED or
  clock-unverified response (none seen so far); a new FOMC statement released during capture (the next
  meeting is after the window).

## Revision-25 successor (programmed, gated: 90-minute real pilot)

Authorized only after the revision-23 closure and only if its report shows integrity VALID **and**
duration ACCOMPLISHED, the old service stopped and its store's ownership free. `pilot successor` checks,
in this order, and records every unmet condition in `/home/kyo/fomc-pilot/successor-decision.json` (and a
journal notice) instead of starting anything: the closure report (waited for until `--wait-until`), its
integrity and duration verdicts, `fomc-pilot` not active, the old store's owner lock free, no other
`fomc-pilot*` service active (never a second emitter), NTP synchronized, the revision-25 spec binding, the
`--check` preflight on a new store. When every condition holds it starts `fomc-pilot-rev25` from frozen
code (a detached worktree at the pushed SHA) on a new `fomc-store-v5` store with the fixtures manifest,
and `fomc-pilot-rev25-supervisor`, whose closure 90 minutes later stops the service cleanly, takes a
consistent copy and verifies it offline. The short pilot's criteria (`closure-report.json` →
`criteria`): one content identity per item over re-reads whose raw bytes differ, every observation its own
record, raw and link, snapshots re-read and replayed identically, FIX15 journal PROVEN, durable state of
the +300 s and +1 h rechecks (served when due long enough). Programmed with a transient timer
(`fomc-successor`, 2026-10-02 18:33 UTC, waiting for the report until 19:30 UTC); a reboot before then
cancels it rather than starting a pilot at an unplanned time.

**Expected outcome, stated in advance:** the revision-23 duration will be NOT_ACCOMPLISHED (interrupted
pilot), so the successor will record NOT_LAUNCHED. Running the 90-minute pilot anyway needs a new explicit
authorization; the rule is not relaxed to make it start. Offline rehearsal of the criteria:
`test_a_short_pilot_rehearsal_meets_the_successor_criteria` (MET); gate:
`test_the_successor_is_launched_only_behind_its_gate`.

Manual equivalent of the launch, from a detached worktree at the pushed SHA:

```
SHA=<pushed feat/phase5 SHA>
git -C /home/kyo/HyprL-phase5 worktree add --detach /home/kyo/fomc-pilot/code-rev25-${SHA:0:9} $SHA
CODE=/home/kyo/fomc-pilot/code-rev25-${SHA:0:9}; cd $CODE
python -m pytest tests/crypto/test_fomc_*.py -q && python -m scripts.trading_lab.fomc.soak --hours 26
python -c "from scripts.trading_lab.fomc import spec; print(spec.SPEC_REVISION, spec.verify_spec_binding())"
test "$(systemctl --user is-active fomc-pilot)" != active && timedatectl show -p NTPSynchronized --value
RUN=/home/kyo/fomc-pilot/run-rev25-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $RUN
cp /home/kyo/fomc-pilot/run-20261001T182106Z/fixtures-manifest.json $RUN/
python -m scripts.trading_lab.fomc.service --store $RUN/store --check
systemd-run --user --unit=fomc-pilot-rev25 --working-directory=$CODE --property=TimeoutStopSec=180 \
  --property=KillMode=mixed --property=StandardOutput=append:$RUN/service.log \
  --property=StandardError=append:$RUN/service.log -- /usr/bin/python3 -m scripts.trading_lab.fomc.service \
  --store $RUN/store --tick 1 --manifest $RUN/fixtures-manifest.json
START=$(date -u +%Y-%m-%dT%H:%M:%S+00:00); CLOSE=$(date -u -d '+90 minutes' +%Y-%m-%dT%H:%M:%S+00:00)
systemd-run --user --unit=fomc-pilot-rev25-supervisor --working-directory=$CODE -- /usr/bin/python3 \
  -m scripts.trading_lab.fomc.pilot supervise --store $RUN/store --unit fomc-pilot-rev25 \
  --service-log $RUN/service.log --alerts $RUN/alerts.log --snapshots $RUN/snapshots.jsonl \
  --copy $RUN/closure-copy --report $RUN/closure-report.json --close-at $CLOSE --planned-start $START
```

## Remaining limits

Not proven by this slice, or outside it:

- The physical duration of durability is not bounded, by spec (revision 23): a record admitted before
  +120 s may become durable later; it is then valid and only later available. A storage incident is
  detected after 10 s (`storage_incident.threshold_seconds`); its durable record is written only once
  the store writes again; while the owner's own COMMIT stalls, only the monitor thread runs (alert and
  grant suspension). Alerts go to stderr (`FOMC-ALERT`); routing them to a pager is the operator's.
- FIX15 over all physical starts is proved from the revision-25 grant journal only. Revision-23 and -24
  stores have no journal: their continuation grants stay NOT_PROVEN and are never reconstructed. The
  journal proves what the owner granted and recorded; a grant consumed by the limiter whose row could not
  commit sends nothing (dispatcher: never registered; step-driven path: abandoned before I/O) and is
  therefore absent from the journal by design. Not exercised against the real provider: a real redirect.
- FOMC_CANON_V2 knows only the Cloudflare spans observed in 2026-10 and a strict subset of HTML; if
  Cloudflare changes its markup or a page becomes ambiguous to the tokenizer, records fall back to
  RAW_FALLBACK (more revisions, never a wrong merge) until a new canonicalizer version is specified.
  Under V2 the stable identity over real re-reads is proved on the 33 stored rev23 raws only; the
  successor pilot that would show it live has not run.
- Bounds are closed at the first owner tick at or after them (1 s in production, never before). A hung
  task keeps its thread until the process exits; it is fenced, not killed. `settle_s` (waiting for
  workers between ticks) exists only for simulated clocks.
- The step-driven path (`Collector.step`, used by most tests and the demo) still runs one fetch at a
  time, inline in the caller's thread; only the service gives the concurrency and non-blocking
  guarantees. Worker threads are created per task (no pool); a task hung forever keeps its thread.
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
