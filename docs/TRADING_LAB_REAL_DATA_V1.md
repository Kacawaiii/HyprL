# Trading Lab — real market corpus V1

Phases 1 to 3 were proved against synthetic fixtures. This corpus replaces the
input with something real, captured once and frozen so that any later result can
be reproduced from the repository alone.

**THE REAL BENCHMARK HAS NOT BEEN RUN YET.** Phase 4A is a data gate: no model
was fitted on this corpus, no `rank_ic`, MAE or RMSE was computed on it, and no
configuration was chosen after looking at anything. Freezing the data before
freezing the protocol is the entire point — it is what stops the fold geometry
from being tuned to flatter the answer.

## Source

| | |
|---|---|
| provider | `coinbase_exchange_rest` (`api.exchange.coinbase.com`, public candles endpoint) |
| products | `BTC-USD`, `ETH-USD` |
| timeframe | `1h` |
| range (inclusive openings) | `2025-08-01T00:00:00Z` → `2026-07-31T23:00:00Z` |
| expected openings | 8760 per product (365 days × 24) |
| capture protocol | `coinbase-exchange-rest-batched-v1` |
| captured | `2026-08-10T12:40:52.793452Z` → `2026-08-10T12:41:27.998875Z` |

The range was fixed before the first request and before any model ever touched
the data. It is not to be shifted because a period later looks inconvenient.

## Capture protocol

The plan is a pure function of `(product, range, timeframe, batch_size)`: 30
non-overlapping windows of 300 openings each — 300 being `MAX_CANDLES_PER_RESPONSE`,
the limit the Phase 1 adapter already enforces — for 60 requests in total. Retries
are bounded (4 attempts, fixed backoff) and never move the requested
window. Request pacing is a constant, not a random sleep.

## Raw and canonical

Both levels are kept. Discarding the raw responses would make a canonicalisation
bug unrecoverable.

* `{PRODUCT}/raw/batch_NNNN.json` — the response bytes exactly as received.
* `{PRODUCT}/canonical.jsonl` — one canonical JSON object per line, ascending by
  opening, derived through the Phase 1 adapter so the values obey the MarketBar
  contract. JSON numbers are parsed straight into `Decimal` from their textual
  token; nothing passes through a binary float.

No secret, header, cookie, token or machine detail is stored — the endpoint is
public and the artefacts contain only market data and hashes.

## Identity

| hash | meaning |
|---|---|
| `corpus_spec_hash` | `7a9de4d8331ece3856408636ad650a8dff44625a83c1a64fa9c134bc7627cdbd` — what was **asked for** |
| `corpus_content_hash` | `688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b` — the bytes that **came back** |
| `manifest_content_sha256` | `7513bb23a20c140602b5dbd54aef9082d1bd8330970da318bd0ad0c44dd44d03` — the manifest body, excluding this field |

The split matters for audit: re-running the same request against a source that has
changed its mind yields the same spec hash and a different content hash.

## What was captured

| product | rows | first opening | last opening | missing | identical duplicates |
|---|---|---|---|---|---|
| BTC-USD | 8750 | 2025-08-01T00:00:00+00:00 | 2026-07-31T23:00:00+00:00 | 10 | 0 |
| ETH-USD | 8750 | 2025-08-01T00:00:00+00:00 | 2026-07-31T23:00:00+00:00 | 10 | 0 |

### Gap policy

Gaps are declared, never filled. There is no forward-fill and no interpolation;
Phase 2 already breaks indicator warm-up and label windows at a gap.

Identical missing intervals were observed in both captured products:

* `2025-10-25T16:00Z` → `2025-10-25T20:00Z`
* `2026-05-08T02:00Z` → `2026-05-08T06:00Z`

That is the observation, and it is all the corpus establishes. A common source
interruption is a plausible reading of two five-hour windows shared by two
products, but nothing here demonstrates it: no exchange status record was
consulted, and this document does not assert one. The openings are simply
declared absent and left absent.

### Duplicate policy

An opening returned twice with a byte-identical payload is deduplicated and
counted. An opening returned twice with **different** values inside one capture
fails closed: picking "the last one" would bury a source contradicting itself.

## Offline verify and replay

Neither command touches the network, and the tests replace the HTTP function with
one that raises, so a regression reaching for the exchange fails loudly.

`verify` recomputes, from the files alone: the manifest hash, the spec hash, the
content hash, every raw SHA-256 and byte size, every canonical SHA-256, the
canonical file against what the raw responses re-derive, ordering, duplicate
counts, gap lists, OHLC invariants and range boundaries. A single flipped byte in
any artefact is detected.

`replay` rebuilds `canonical rows → MarketDataStore → causal snapshot →
replay_snapshot → MarketSeries` into a temporary database. No database is
committed.

### Three clocks, kept apart

| clock | meaning |
|---|---|
| bar time | `bar_open_at` / `bar_close_at` — when the market moved |
| actual capture time | `capture_started_at`, `capture_completed_at`, per-batch `captured_at` — when these bytes were really obtained |
| synthetic replay marker | the `available_at` / `ingested_at` fed to Phase 1 during canonicalisation and replay |

The third is a **synthetic deterministic replay marker**. It exists only to
satisfy Contract A's `available_at >= bar_close_at` rule and to make replay
reproducible: the same corpus replayed twice yields the *same* snapshot identity
instead of a fresh one per run. It is never a statement about when a value was
actually known to the world.

## Honest limitation: not a point-in-time revision history

```
historical_candle_corpus                  = true
point_in_time_exchange_revision_history   = false
```

Every OHLCV value here is the value the exchange served **at capture time in
2026**, not the value it would have served at some earlier instant.

The future benchmark will prevent lookahead across bars: a feature at T reads only
openings ≤ T, and Phase 1 enforces that. What it cannot do is prove that a later
correction to a candle was not already folded into the value we captured. If a
candle was ever revised, the revised value is what we hold, at every point in the
series.

That is a limitation of the **data**, not a causal defect in the engine. No
revision history has been reconstructed, and no stronger claim should be made.

## Before the real benchmark can run

Frozen already: model configurations (Ridge α=1.0, XGBoost V1), the selection rule
`validation-rank-ic-mae-rmse-v1`, its three-observation validation floor, and the
refit policy.

Still to be decided **and written down before any score is computed**: the official
feature set, the exact `DatasetConfig`, and the real `WalkForwardConfig`. Those have
only ever existed inside test fixtures. Choosing them after seeing performance would
undo everything this corpus was frozen to protect.
