# FOMC V1 — offline integrated slice

Implements `docs/artifacts/fomc_capture_spec_v1.json` **revision 22**
(`ba6a01e5f12e810ecde89711304c278862d147de298c63e7602a8359fa18e678`, pinned in
`scripts/trading_lab/fomc/spec.py` and checked by `verify_spec_binding`). The JSON stays
authoritative. This is an **offline slice**: every source is synthetic and served by a local
Unix-socket provider; no Federal Reserve request, fixture capture or live run is part of it.

```
python -m scripts.trading_lab.fomc.demo          # the executable path, end to end
python -m pytest tests/crypto/test_fomc_*.py     # 74 tests
```

`feed -> durable raw -> classification -> primary acquisition -> revision + observation link ->
cycle -> events_as_of(T, H) -> reopen -> offline replay`

| module | role |
|---|---|
| `spec.py` | revision-22 constants, canonical hashing, spec binding |
| `identity.py` | URL admission (scheme, host, userinfo, port, percent, query, fragment), `source_item_id`, durable keys |
| `clock.py` | strict `Date`/`Age`, per-response clock verdict, canonical timestamps |
| `store.py` | SQLite WAL/FULL, `commit_seq`, append-only rows, unique keys, immutable content-addressed raw (published once with `os.link`, never overwritten) with digest checks |
| `ledger.py` | epochs (fencing), atomic `TRANSPORT_INVOKED`, one outcome per attempt, `LATE_EVIDENCE`, episodes |
| `limiter.py` | FIX15 rolling window, spacing, embargo |
| `transport.py` | manual redirect loop, per-hop admission and grant, 60 s deadline from each grant over DNS/TCP/TLS/write/headers/body/decoding (stage timeouts, watchdog, late-result checks), bounded decoded body |
| `parsing.py` | Content-Type gate, strict UTF-8, secure RSS (expat), token-bounded HTML anchors, grammars |
| `processing.py` | one terminal outcome per PROCESSABLE record, feed classification + cycle, primary classification + revision/link, fenced runs, poison guard |
| `state.py` | LIVE_ELIGIBLE, server-attested `avail`, `NOW_LB`, anchors, obligations, episodes, item conclusion, RESOLVE validity |
| `collector.py` | single owner (flock + epoch), fetch/commit, reconciliation, episodes, class rotation, operator actions, manifests, integrity diagnostics |
| `snapshot.py` | P(T) under horizon H, read state, 9-step selection, discovery state, identity, read-time raw dependency checks, verified replay |
| `synthetic.py`, `demo.py` | simulated clock, local provider, fixtures, the demo |

## Invariant → code → test

| rule | code | tests |
|---|---|---|
| F08 raw first, F72 corruption fails closed | `collector.commit`, `store.put_raw/read_raw`, `processing.process_record`, `collector.verify_integrity/_diagnose`, `snapshot._verify_dependencies/replay` | persistence `raw_is_content_addressed…`; snapshot `test_7`; hardening: corrupt or missing raw fails `events_as_of` without prior verification (old `(T, H)` too, revisions and discovery cycles), `put_raw` never overwrites, corrupt slot → `CORRUPTION_FAIL_CLOSED` + diagnostic |
| F09/F10/F74 revisions keyed (item, hash), mode-neutral links | `processing.classify_primary` | snapshot `test_1`, `test_2`, `test_7` (A-B-A) |
| F26–F30 identity and URL admission | `identity.admit_url`, `source_item_id` | persistence scheme/port/percent cases (FOMC31–38) |
| F31 decoded body cap | `transport._admit_200` | components `oversized…` (FOMC39) |
| F60 60 s physical deadline from each grant, per hop | `transport.Deadline`, `_resolve/_connect/_hop/_admit_200`, watchdog | hardening: late DNS without TCP, late TLS, blocked/dripped headers, dripped body, per-redirect deadline, real-socket watchdog (FOMC127–130, FOMC154) |
| F32 strict UTF-8, F33/F34 anchors, F36 XML security | `parsing` | components decoding/anchors/XML cases (FOMC41–62) |
| F38–F40 final 200, redirects ≤ 3 | `transport.fetch` | processing `redirects…` (FOMC69/70/84) |
| F44–F49 limiter | `limiter.Limiter` | components `limiter…` (FOMC90/91/97) |
| F50–F56 anchor, obligations, satisfaction | `state.anchor/obligations/work_satisfied` | processing `resolve_cutoff…` (O300), snapshot `test_1`, `test_7` |
| F57/F65/F66 six attempts, 120 bound, no extra traffic | `ledger.transport_invoked`, `collector._ready` | persistence budget/concurrency/fencing; processing `dead_page…` (24 requests) |
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

## Not in this slice

- No real TLS/HTTPS run: `HttpsConnector` exists but is never exercised (no network by mandate).
- No background daemon: the collector is step-driven. The 60 s physical-attempt deadline is
  enforced; the 600 s run deadline, the 600 s absolute attempt deadline and the 120 s save deadline
  are not time-enforced, and ended tasks are detected at reconcile under a single-threaded owner
  assumption.
- Snapshots do not yet expose or bind the FOMC source-health state; no `SNAPSHOT_WATERMARK` write.
- Selection implements class rotation and (next eligible instant, source_item_id) ordering, without
  the manifest-first ordering of HISTORICAL_BACKFILL and without a dedicated FOMC126 test.
- Normalized revision fields are a subset of `normalized_minimum_fields`.
- Derivations rescan the store (quadratic); fine for tests, not for years of capture.
- Official historical fixtures and real capture are separate, later steps.
