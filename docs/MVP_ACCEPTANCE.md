# First MVP — official events at an instant, reproducible: acceptance

**Status: PROPOSED.** Written by an agent; it becomes the acceptance only when the operator accepts it.

The MVP: for two official sources (Federal Reserve FOMC statements, SEC EDGAR 8-K submissions) a user can ask
"what was officially known at instant T, as far as the store had committed by horizon H", get the same answer
every time, see where each fact came from, and see when the answer is **not** known. Everything is offline and
read-only over stores that were captured earlier.

## How to run it

```
python -m scripts.trading_lab.mvp_check --fomc-store DIR --edgar-store DIR \
    [--fomc-reads snapshots.jsonl] [--web] [--output FILE]
python -m pytest tests/crypto/test_mvp_check.py -q          # the check itself, on synthetic stores
```

The script serves the read-only API in-process on the loopback address that accepts connections (`127.0.0.1`,
else `::1`), calls it over HTTP, and prints a JSON report: `PASS`, `FAIL` or `BLOCKED` per criterion. `BLOCKED`
means "could not be checked" (missing input, dependency or loopback) with its cause and **never counts as
PASS**. Exit status: 0 all PASS, 1 any FAIL, 2 none failed but some BLOCKED. Without `--web` the four cockpit
criteria are BLOCKED; without `--fomc-reads`, FOMC-08 is. The report of a real run holds identities, counts and
digests only, and stays outside Git.

## Criteria

`S` is `FOMC` or `EDGAR`; each criterion runs for both sources and has the id `S-NN`.

| id | criterion | checked by |
|---|---|---|
| S-00 | the store directory exists and is fingerprinted (path, size, mtime, SHA-256 of every file) | `mvp_check` (`check_present`) |
| S-01 | **opened read-only, status AVAILABLE**: `read_only` true, spec revision/hash and schema version reported, horizon > 0, a `suggested_as_of` (the store's attested lower bound on now), responses recorded; a refused store (wrong schema) is `REJECTED`, not repaired | `mvp_check` `S-01`; `test_app_api_fomc.py::test_an_incompatible_or_corrupt_store_is_refused_not_repaired`, `test_app_api_edgar.py::test_requests_outside_the_contract_and_refused_stores` |
| S-02 | **reads at instants**: at `suggested_as_of` the read is `*_RESOLVED` with identity, H = horizon, health and items; after that instant (+1 day) and before any activity (horizon 1) it is `*_CAUSAL_VISIBILITY_UNRESOLVED` with **no items and no health** — never a guess | `mvp_check` `S-02`; `test_app_api_fomc.py::test_a_read_at_an_earlier_horizon_or_after_now_lb_shows_what_was_known` |
| S-03 | **identity stable**: the same (T, H) gives the same identity on a second read and after a brand-new server reopens the store, for a RESOLVED and an UNRESOLVED read | `mvp_check` `S-03`; EDGAR/FOMC slice tests (identity after reopening) |
| S-04 | **verified replay identical**: `/replay` re-derives the read from the records; `identical` true, no error, `replay_identity` = read identity, at the suggested instant, after it, and (FOMC) at every recorded read | `mvp_check` `S-04`; `test_app_api_fomc.py::test_status_snapshot_item_and_replay_over_a_read_only_store`, `test_app_api_edgar.py::test_status_reads_filing_detail_and_replay` |
| S-05 | **item / filing detail with provenance**: every row of the snapshot opens; the detail belongs to the same read identity; revisions agree with the item state (a revision exists exactly when the item has one; out-of-scope FOMC items have none) and at least one item has one; every observation carries a 64-hex raw digest and an instant; FOMC: request URL and byte length, canonical source URL; EDGAR: filing index URL | `mvp_check` `S-05`; the two `test_app_api_*` journeys |
| S-06 | **source health**: FOMC exposes `discovery_feed` and `primary_statement`; EDGAR one entry per watched CIK; health exists only under a RESOLVED read (S-02) | `mvp_check` `S-06`; `FOMC_V1_OFFLINE_SLICE.md` health row |
| S-07 | **every route** `/api/v1/sources/{fomc,edgar}`, `/snapshot`, `/replay`, `/items/{sid}` or `/filings/{accession}` answers 200 over HTTP; outside the contract: no `as_of` 400, naive `as_of` 400, horizon beyond the store 400, malformed id 400, unknown route 400 (JSON, never a page), POST 405 | `mvp_check` `S-07`; `tests/crypto/test_app_api_*.py`, `test_ops_server.py` |
| S-09 | **store left unchanged**: the fingerprint of S-00 is identical after all reads, replays and the reopening | `mvp_check` `S-09`; the `_fingerprint` assertions of the two journeys |
| FOMC-08 | **recorded reads reproduce**: every line of the pilot's `snapshots.jsonl` re-read at its own (T, H) gives the recorded identity and read state (RESOLVED and UNRESOLVED alike) | `mvp_check --fomc-reads` |
| COCKPIT-01..04 | **cockpit journey**: `npm run typecheck`, `npm run lint`, `npx vitest run`, `npm run build` in `apps/web` (after `npm ci`) | `mvp_check --web`; or run the commands by hand |

(There is no S-08: the number is kept free for the FOMC recorded reads, which exist for FOMC only.)

Whole-repository gate, unchanged by this document (AGENTS.md "Checks before any push"):
`python -m pytest tests/crypto/test_fomc_*.py tests/crypto/test_sources_primitives.py tests/crypto/test_edgar_slice.py tests/crypto/test_app_api*.py tests/crypto/test_ops_server.py tests/crypto/test_equity_api.py tests/api -q`.

## Verdict

The MVP is accepted for a pair of stores when `mvp_check` over them reports **PASS for every criterion** (zero
FAIL, zero BLOCKED), and the repository gate above passes on the same commit. A BLOCKED criterion is reported as
not accepted until it is run somewhere it can be.

## Explicitly out of scope

- The 24 h and 7-day rechecks of a captured publication.
- A new real publication, or any new capture: no request to the Federal Reserve, SEC EDGAR or any other source.
- Real redirects (the stores hold none that this check could exercise).
- Continuous collection, scheduling, supervision of a running collector.
- Trading, paper or live; models; holdout data.
- Correctness of the sources' content beyond what the specs freeze (the check proves reproducibility,
  provenance and refusal to guess, not that an agency's statement is right).
- A timeline view or the Events page layout (other work); only their API/cockpit commands are in the gate.
