# Event features V2

V2 sits next to [V1](EVENT_FEATURES_V1.md); V1 is untouched (module behaviour, `docs/artifacts/event_features_matrix_v1.json`
with identity `a5382f91…`, the demo). Implementation: `scripts/trading_lab/event_features/v2.py`, policy
`ATTESTED_EVENT_FEATURES_V2`, classification `ATTESTED_OBSERVATION_CLASSES_V1`. Tests: `tests/crypto/test_event_features_v2.py`.
It reads no prices, acquires nothing and trains nothing.

## Why

V1 counts every attested event. The first valid listing of an issuer onboards its whole existing inventory (104 and 57
filings in the real EDGAR fixture trial), so a V1 count of 104 describes what the store learned, not what happened.
V2 separates onboarding from news. On the two real closure stores every attested observation is INITIAL_INVENTORY
(6 FOMC, 161 EDGAR): there is no newly observed event in them yet.

## Classes

Computed per reader at its pinned H, from the store's own rows, never from filing dates or declared release times.

| Class | EDGAR | FOMC |
|---|---|---|
| INITIAL_INVENTORY | accession first observed in the first valid listing (`LISTING_CLASSIFIED`) of its issuer in the store | statement whose first verified observation is a `HISTORICAL_BACKFILL`, or whose candidate was created by the first valid feed read (`FEED_CLASSIFIED`) |
| NEWLY_OBSERVED | accession first observed in a later read | statement first observed otherwise (a later feed, or a live acquisition) |
| REVISION | a later observation of a known accession with changed metadata (new revision) | a later verified observation of a known statement with changed content |

The same content seen again is not an event. An observation whose transaction has no resolved availability is never emitted, but
still advances the per-item state. Each observation carries the attested availability of its transaction
(CAUSAL_AVAILABILITY_V3), which orders every window and is the only condition of use. A backfill that is acquired after a
statement already appeared as a new feed candidate is classified by its first verified observation (backfill = inventory): this
edge is stated, not hidden.

## Features

States are V1's (NOT_OBSERVED, UNRESOLVED, RESOLVED, NOT_CONFIGURED, INTEGRITY_ERROR, NOT_APPLICABLE, UNKNOWN_MAPPING, and the
unwatched-issuer NOT_OBSERVED); every non-RESOLVED state keeps null values. Windows are (T−7d, T] and (T−30d, T], open on
the left, closed on the right.

| Source | Column | Meaning |
|---|---|---|
| FOMC | `new_statements_attested_7d` / `_30d` | NEWLY_OBSERVED statements attested in the window |
| FOMC | `statement_revisions_attested_7d` / `_30d` | REVISIONs attested in the window |
| FOMC | `hours_since_last_new_statement` | hours since the attested availability of the last NEWLY_OBSERVED statement; null if none |
| FOMC | `new_statement_attested_within_24h` | that last availability is at most 24 h before T (inclusive) |
| EDGAR (8-K, 8-K/A of the mapped issuer) | `new_accessions_attested_7d` / `_30d` | NEWLY_OBSERVED accessions attested in the window |
| EDGAR | `accession_revisions_attested_7d` / `_30d` | REVISIONs attested in the window |
| EDGAR | `hours_since_last_new_accession` | as above |
| EDGAR | `new_accession_item_2_02_7d`, `_5_02_`, `_7_01_`, `_8_01_` | exact item token among the NEWLY_OBSERVED accessions of the 7-day window; unknown items give null, never false |

Inventory sizes are not features. Each RESOLVED source carries `inventory`: attested inventory observations at T, the availability
of the last of them, and the availability of the first valid read (only if it is at or before T). They feed the warm-up rule of the
[comparison protocol](COMPARISON_PROTOCOL_V1.md). Each row also carries `protection`: the protected intervals of the product that contain T
and those touched by the 30-day event window (V1 flagged the crypto holdout only).

```python
from scripts.trading_lab.event_features.v2 import EventFeaturesV2

with EventFeaturesV2(fomc_store="FOMC_DIR", edgar_store="EDGAR_DIR") as features:
    rows = features.rows([("AAPL", "2027-04-01T20:00:00Z")])
```

## Verified

Inventory burst not counted as new (and V1 still counts it); a later new accession counted, including a filing dated years
earlier; revision separated and an unchanged re-listing ignored; FOMC feed inventory, later statement, correction and backfill;
window edges (T−7d/T−30d excluded, T and T−24h included, ±1 µs); null states; protection flags at both boundaries; V1 matrix
identity and V1 outputs unchanged; determinism and byte-identical stores. The real closure stores were read read-only: only counts above are recorded.
