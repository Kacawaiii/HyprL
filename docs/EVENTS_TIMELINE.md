# Official events timeline

`GET /api/v1/events/timeline?as_of=2026-06-17T19:00:00Z` reads the configured FOMC and SEC EDGAR stores
at one instant. It opens them read-only and makes no source request. The cockpit's Events page has a
Timeline tab with the same read and optional, independent FOMC and EDGAR horizons.

`as_of` requires an ISO-8601 instant with an explicit offset. Optional `fomc_horizon` and
`edgar_horizon` are integer commit sequences, defaulting to the respective store's head; configured
stores reject values outside their own head. Malformed or negative horizons are rejected even when
a source is unavailable. The combined result is bounded by `MAX_SOURCE_ITEMS` (1,000); exceeding
the bound is an error, never silent truncation.

Each source is read through its existing causal snapshot (`events_as_of` or `filings_as_of`). The
response has `T`, `sources`, `rows`, and `identity`, together with the API version and `read_only`.
Each source entry contains its `read_state`, snapshot `identity`, horizon `H` and prefix `P`.
Unconfigured sources have `NOT_CONFIGURED`; refused store openings have `REJECTED`; a snapshot or
availability projection that fails closed has `REFUSED`. Those entries have a null snapshot identity
and an explanatory `reason`. Causally unresolved snapshots retain their source's original read state
and identity. They contribute no rows. Rows from another resolved source can still be shown; this
is a partial timeline, with no claim that unavailable sources are empty.

Rows contain `source`, `id`, `title` or `form`, `state`, `revision`, `content_identity`, `available_at`,
and `provenance`. Only FOMC `CURRENT_REVISION` items are included. Their availability is taken from
`state.availability` at the transaction that first recorded the selected revision. An A-B-A return
to an existing revision keeps that revision's original availability. EDGAR includes `PRESENT` and
`ABSENT_FROM_LISTING` filings and uses their existing `first_available_at`, including after corrections
or absences.

The order is the instant `available_at`, then `source`, then `id`, ascending. FOMC's
`declared_release_at` and `declared_release_text` and EDGAR's verbatim `acceptance_datetime_text`
are nested under `provenance` and explicitly labeled in the cockpit. They never supply availability
or affect order.

The timeline identity is `sha256_canonical` of exactly:

```python
{
    "T": T,
    "sources": {
        "fomc": {"identity": fomc_snapshot_identity, "read_state": fomc_read_state},
        "edgar": {"identity": edgar_snapshot_identity, "read_state": edgar_read_state},
    },
    "rows": ordered_rows,
}
```

It binds both independent source identities and read states, even if their selected rows are unchanged.
Horizon and prefix annotations are already bound through each source snapshot identity. Reasons and
API metadata do not enter the identity. Existing source specifications, snapshots and API payloads
are unchanged. The offline tests are `tests/crypto/test_app_api_timeline.py` and
`apps/web/src/test/timeline.test.tsx`; real-store verification evidence belongs only in the local report.
