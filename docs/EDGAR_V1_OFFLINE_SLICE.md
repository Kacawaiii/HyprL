# SEC EDGAR V1 — slice and first real fixture

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

## Remaining limits

- One real trial only (2 CIKs, 8 responses, 33 minutes): UV2 semantics, UV3, UV4 and the universality of
  UV6 stay unknown, and nothing here covers a new 8-K appearing during a capture, a correction, an absence
  or a failure against the real source (those are proved offline only).
- The runner is single-threaded and step-driven: no concurrent fetches, no storage-incident detection and no
  persistent supervisor/closure (the FOMC service has them); it is bounded instead by its authorization's
  request budget and expiry.
- Absence detection depends on the documented window (UV3); a filing removed and replaced by the SEC with
  the same accession would show as a correction, not as a removal.
- Any further capture needs a new operator authorization (budget, expiry, declared contact); continuous
  collection would also need the supervision the FOMC service has.
