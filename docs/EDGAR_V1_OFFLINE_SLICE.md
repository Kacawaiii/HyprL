# SEC EDGAR V1 — offline slice

Implements `docs/artifacts/edgar_capture_spec_v1.json` **revision 1**
(`98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce`; pinned in
`scripts/trading_lab/edgar/spec.py`, checked by `verify_spec_binding`). The JSON is authoritative. This is an
**offline slice**: every listing is synthetic, served by an in-process fake fetcher; no request is made to
the SEC, and the spec authorizes none (`authorizes_capture: false`).

```
python -m pytest tests/crypto/test_edgar_slice.py        # 24 tests, cases EDGAR01-EDGAR16
python -m scripts.trading_lab.edgar.demo                  # new 8-K, correction, 8-K/A, absence, reappearance, replay
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

The demo reads after each step and shows, for example, the corrected 8-K with two revisions seen, the
8-K/A as its own filing, the 8-K `ABSENT_FROM_LISTING` and then `PRESENT` again, availability taken from
the attesting response (never from `acceptanceDateTime`), and every read identical after reopening the
store read-only and replaying it.

## Shared primitives (extracted from the FOMC slice)

`scripts/trading_lab/sources/`: the append-only record store with content-addressed raws and a read-only
opening, CAUSAL_AVAILABILITY_V3 and its prefix, strict HTTP `Date`/`Age` clock evidence, canonical hashing
and the rolling limiter. FOMC binds them to its spec unchanged (its 198 tests pass); EDGAR binds them to its
own spec, schema and unique kinds.

## Remaining limits

- No real capture: the production fetcher (`HttpsFetcher`) is tested against an in-process fake connection
  only; real TLS, real headers, the real column set and the real `Date` behaviour are unverified (UV1–UV6).
- Step-driven, single-threaded collector; no service, no supervisor, no storage-incident detection and no
  reconciliation of an attempt interrupted by a crash (the FOMC service has them; EDGAR does not yet).
- The per-request socket timeout is 30 s per operation, not a total deadline from the grant.
- Absence detection depends on the documented window (UV3); a filing removed and replaced by the SEC with
  the same accession would show as a correction, not as a removal.
- Not in the API or cockpit yet (the FOMC journey is).
