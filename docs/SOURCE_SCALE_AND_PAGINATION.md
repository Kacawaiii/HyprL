# Offline source read scale and pagination

The source API keeps the canonical complete snapshot identity, policy, spec binding,
T, H and P unchanged. The spec files and store schemas are unchanged.

Snapshot, detail and timeline routes accept `limit` (default 200, maximum 1000)
and `cursor`. Existing list fields contain the requested page. The optional
`pagination` object contains `limit`, per-list `totals`, and `next_cursor` (null at
the end). Cursors bind the complete snapshot identity, endpoint and entity. Supply
the first response's H on every subsequent request; a cursor from another read,
source, endpoint or entity fails closed. Page size can change between requests.

For a detail response, one cursor pages all history lists together: revisions,
observations, EDGAR absences, and FOMC `item.links`. A shorter history is empty on
later pages. Concatenating each list independently reproduces the complete history.
Item/filing summaries, provenance, health and identity continue to describe the
complete read, including when a history page has no revisions.

`/api/v1/sources/{fomc,edgar}/timeline` returns causal durable activity through P,
in commit and intra-transaction order. It includes processing outcomes, health,
cycle conclusions, revisions, observations/links and absences. An unresolved read
has no timeline rows. Every exposed history raw is checked, including first raws
of old revisions and FOMC link artifact digests.

Each configured source keeps one admitted read-only reader, one complete snapshot,
one timeline and at most one selected entity's history. A DB replacement or change
without a newer commit invalidates these caches. Appends to the same admitted DB
retain immutable prefixes and decode only added rows; changing (T, H) replaces
the cached read. Every refresh re-admits schema/spec. Requests serialize per
source, and server shutdown closes the readers. Nothing is written to the source.
Every cache hit re-hashes all snapshot raw dependencies; exposed detail/timeline
history raws are also re-hashed. Cache hits never hide physical raw corruption.

Cold identity construction reads the complete store once and uses linear mirror
memory; ordering filings adds the usual sort cost. The existing identity binds
the complete read. Timeline and entity history construction
are each linear once per cached read. Entity lookup uses a per-read index; even
details of the last filing do not walk the cached list of other filings.
Subsequent pages use list slices, with zero
SQL-decoded rows and at most one indexed view lookup per exposed historical row
that names a raw by its response record number.
Raw verification remains proportional to the complete snapshot's distinct raw
dependencies, plus the exposed history page; it is deliberately never cached.
Switching reads, entities or a changing DB can require rebuilding these caches.
These bounds describe a frozen offline read, not a constant-time cold read.

EDGAR replay uses indexed transaction slices and versioned latest events per
accession. It does not rescan every filing observation for every processing record.
Source status uses indexed counts and transaction endpoints; EDGAR's server lower
bound uses a versioned aggregate instead of a response-history scan.

The cockpit requests 200 rows at a time, displays one page, offers next/previous
controls, and pins H after the first response. Timeline loading is on demand.
It never automatically fetches the entire history. Pagination metadata is optional
in the TypeScript contracts so older responses with small lists still render.

## Reproduce and verify

```sh
python -m scripts.trading_lab.sources.scale --polls 24 --fomc-hours 72
python -m scripts.trading_lab.sources.scale --source edgar --polls 24
```

Only temporary synthetic stores are created. EDGAR uses ten CIKs, 1000 filings per
listing, repeated polls and a later clock attestation; FOMC reuses the soak's local
provider, clock, faults, restarts and offline verification. JSON output records
elapsed milliseconds, `store.reads` (queries, decoded rows, handed-out view rows),
compact payload bytes, counts and identities. Opening the private read-only DB
copy is measured separately from snapshot construction. This is application-route
work and JSON serialization size, excluding HTTP transfer/browser rendering.

The pagination regressions are in `test_app_api_source_pagination.py`, which the
required `tests/crypto/test_app_api*.py` check includes. They cover a 10,000-filing
store, more than 1000 observations of one filing, full concatenation, nested links,
historical identities, appended commits at fixed H, endpoint/entity/read cursor
refusals, corruption on cache hits, schema/spec refusal, concurrent readers,
bounded warm row costs and linear replay work. Cockpit tests exercise on-demand
pages, pinned horizons and lazy timeline loading.

## Measured offline results

On the shared server, 2026-10-04; milliseconds are individual samples, not latency
guarantees. Rows are SQL-decoded rows / view rows handed out. Payload sizes are
compact JSON bytes. The EDGAR fixture has ten CIKs, 1000 filings per listing, 24
polls per CIK plus two later attestations (H=728). Its complete read has 10,000
filings and identity `75572c3a430e6d83066cc36b298460bab3254e75e26fbe44c2041e0b82c0e187`.

| EDGAR operation | ms | rows / view rows | payload bytes |
| --- | ---: | ---: | ---: |
| `filings_as_of`, cold | 8253.773 | 253940 / 733211 | 15357763 |
| API snapshot, cold identity | 8833.566 | 253940 / 733211 | 86516 |
| API snapshot, next 200 filings | 7.535 | 0 / 0 | 86516 |
| API timeline, first 200 rows | 273.734 | 0 / 251303 | 145863 |
| API timeline, next 200 rows | 7.951 | 0 / 100 | 146255 |
| API filing detail, first 5 rows | 577.547 | 0 / 26 | 4758 |
| API filing detail, next 5 rows | 7.790 | 0 / 0 | 4119 |
| API status, first aggregate | 60.348 | 0 / 3 | 702 |
| API verified replay | 18632.266 | 0 / 1450421 | 669 |

EDGAR private DB opening took 340.470 ms separately. Full replay's view work is
5.71 times the stored row/transaction count, with no SQL row re-read; its returned
identity matched the plain complete snapshot. First status initializes its lower
bound aggregate once. First detail initializes its history field index once.

The FOMC soak ran 72 simulated hours, reached H=17368 with 4332 responses, and
passed historical snapshot re-reads, verified replay, health and capture invariants.
Its final read identity is
`6eed4c29c06f77c322a4d405c1b34a4e445a43030493274cf78bc63f3a4570b5`.

| FOMC operation | ms | rows / view rows | payload bytes |
| --- | ---: | ---: | ---: |
| `events_as_of`, cold | 1405.435 | 52088 / 39151 | 19777 |
| API snapshot, cold identity | 1117.622 | 52088 / 39151 | 4695 |
| API snapshot, warm | 1.246 | 0 / 0 | 4695 |
| API item detail, warm | 1.025 | 0 / 0 | 9532 |

FOMC private DB opening took 18.950 ms separately. Final regressions also bound
FOMC timeline continuation work by page size, including historical raw checks.
The 1005-observation EDGAR regression ensures histories beyond the 1000-row API
bound remain accessible through pages, without SQL re-reads on continuation.

Required Python check: 462 passed, two conditional skips (local-only official
fixtures and the separate `hyprl_api` service absent from this branch). After the
final entity lookup index, all 89 API tests passed, including the added last-filing
cost regression. Web: `npm ci`, typecheck and lint passed; all 128 Vitest tests
passed with one worker for the shared server's memory budget. Missing optional
Python test dependencies were installed in temporary worktree-local directories
and removed after validation; the shared environment was not changed.
