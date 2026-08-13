# US Equity Corpus V1 — contract

**STATUS: CONTRACT FROZEN — NO DATA CAPTURED YET**

**NO PREDICTIONS**

**NO BACKTEST RESULTS**

**NO COMMERCIAL EDGE CLAIM**

**NOT STRICT POINT-IN-TIME REVISION HISTORY**

---

This document describes what Corpus V1 asks for and what it will accept. It is
written and committed *before* the first request, and that ordering is the
whole point: a range chosen after looking at the data is a curated range, and a
validation rule relaxed after a capture failed is not a validation rule.

When a capture completes, a results section is appended below with the counts,
hashes and gap audit. Until then this file describes a contract and nothing
else.

## 1. What is requested

| | |
| --- | --- |
| corpus id | `massive_us_equity_v1` |
| instruments | `xnas:AAPL`, `xnas:MSFT`, `xnas:NVDA`, `xnas:QQQ` |
| timeframe | `30m` |
| session | US regular session only |
| adjustment | `SPLIT_ADJUSTED` |
| provider | `massive-stocks-historical-v1` |
| calendar | `US_EQUITY_REGULAR` |
| calendar dependency | `pandas_market_calendars==5.4.0` |
| calendar spec hash | `1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314` |
| requested range | `2024-08-01` → `2026-07-31` (exchange sessions) |
| **corpus spec hash** | `93cfdb1a749bfa1de5c69c5dced2908413cfb8bba7f3f663fd9c747151b1b5ed` |

Derived from the calendar, not from arithmetic:

```
sessions in range                 501
expected 30m bars per instrument  6483      (13 × 501 − 6 × 5 early closes)
expected bars, all four           25932
early closes in range               5       2024-11-29, 2024-12-24,
                                            2025-07-03, 2025-11-28, 2025-12-24
```

Nothing after `2026-07-31` belongs in V1. A different range is a different
corpus and needs an `EquityCorpusSpec V2`, not a quiet extension after the fact.

### The QQQ venue was resolved, not assumed

`xnas:QQQ` comes from the contractual reference metadata recorded in
`tests/fixtures/crypto/massive_reference_tickers.json` (`primary_exchange:
XNAS`, `type: ETF`), which the registry is asserted against. On top of that,
the capture runner re-checks the venue, the asset class **and the returned
symbol** against the live reference endpoint before storing a single bar, and
stops if any disagrees. A guessed venue names a different instrument under a
plausible id, and every artefact referencing it would be wrong in a way no
price check would catch.

## 2. What "SPLIT_ADJUSTED" means here

Historical prices restated in current share terms. Splits are reflected;
**dividends are not applied**.

This is **not** total return. A price return computed from this corpus
understates the total return by every dividend paid in the window. Corpus V1
does not claim otherwise, and `TOTAL_RETURN` is refused rather than silently
served as split adjustment.

HyprL does not recompute adjusted prices. The provider serves the adjusted
series; the split records captured alongside exist for **provenance and audit**
— so a later phase can check a price discontinuity against a real corporate
event instead of guessing — and are never used to re-derive a price.

## 3. Point-in-time limitation

> US Equity Corpus V1 is a frozen historical provider-data benchmark. It is
> **not** a strict point-in-time exchange revision dataset unless the provider
> contract explicitly supplies revision history and HyprL records it.

Two separate reasons, and both matter:

1. If the provider ever revised a bar, the capture recorded the revised value,
   not the value that would have been served on the original date.
2. Split adjustment restates history **by definition**. A price shown for
   2024-08-01 in a split-adjusted series is not the price that was printed on
   2024-08-01.

Causal backtesting still holds — a feature at T only ever reads openings ≤ T —
but no future backtest on this corpus may be called **strict PIT** without an
additional layer that reconstructs revisions. `manifest.json` records
`point_in_time_exchange_revision_history: false` so the claim cannot be lost.

## 4. What is accepted

Every bar must sit on `ExpectedBarGrid(US_EQUITY_REGULAR, session, 30m)`.
Refused, always:

- pre-market and after-hours
- weekends and holidays
- any bar after an early close
- any opening not on the 30-minute session grid

A provider returning more data than was asked for is normal. Accepting it
because it exists is not.

Arithmetic invariants, fail-closed: every price positive, `high ≥ max(open,
close)`, `low ≤ min(open, close)`, `high ≥ low`, `volume ≥ 0`, UTC timestamps,
`bar_close > bar_open`. Prices arrive as strings and become `Decimal` — a float
price has already lost digits before it reaches the validator.

## 5. Gaps

```
expected bar openings − observed bar openings = MISSING_EXPECTED_BAR
```

That is the only classification. Overnight, weekends, holidays and the hours
after an early close are **not gaps**: no bar was ever expected there. Nothing
is filled and nothing is interpolated — a gap is reported, never repaired.

When several instruments are missing the same interval, the manifest records it
as `overlapping_missing_intervals`. It is **not** called a provider outage:
that is a conclusion the data does not support, and the same discipline the
Coinbase capture used in Phase 4A.

## 6. Identity: three hashes, deliberately separate

| hash | covers | moves when |
| --- | --- | --- |
| `corpus_spec_hash` | instruments, provider, calendar + its dependency version, timeframe, session type, adjustment policy, requested range, canonical schema, gap policy, corporate-action policy | the *request* changes |
| `instrument_content_hash` | one instrument's canonical bars in canonical order | that instrument's *data* changes |
| `corpus_content_hash` | all four, in canonical order | any data changes |

Re-running the same request against a provider that has changed its mind gives
the same spec hash and a different content hash. That difference is the entire
signal an audit is looking for, and it exists only because the two are computed
from different things.

Content hashes are independent of filesystem order, JSON key order and the
caller's ambient `Decimal` context.

## 7. Network boundary

The Phase 6D `NoNetworkTransport` is **unchanged** and remains the default: the
provider registered in `PROVIDERS_V1` still cannot make a request. Real HTTP
lives in a separate class, `MassiveHTTPTransport`, injected at exactly one call
site.

- GET only. No verb that could change anything on the other end exists.
- Host allowlist: `api.massive.com`. Not a scheme check — `https://` says
  nothing about where the bytes are going.
- Bounded retries on **transport** failures only: timeouts, resets, 5xx, 429.
  A 4xx or a non-JSON body is an *answer*; retrying it produces the same wrong
  answer more slowly while hiding a bug behind a delay.
- Rate limits are typed (`RateLimitedError`), counted, and honour `Retry-After`
  up to 60s; beyond that the run stops rather than sleeping indefinitely.
- Pagination is bounded (`MAX_PAGES_PER_REQUEST`), cycle-detected, and a
  repeated cursor stops the run rather than producing a corpus that looks whole.
- No websocket, no background daemon, no parallel request storm.

## 8. Credential

`HYPRL_MASSIVE_API_KEY`, read at request time, wrapped in `Secret`, sent in an
`Authorization` header, and dropped.

Never committed · never logged · never in the manifest · never in a raw
artifact · never in an exception · never in a repr · never in a query string
(the transport refuses a key-shaped one outright, because a query string is
copied into every access log on the path).

The manifest records `credential_present: true|false` — state, never value. A
`401`/`403` is reported without echoing the response body, since a vendor may
quote the header it just rejected.

**Missing key is a valid state.** The runner stops cleanly *before* any
request, writes nothing, creates no directory and no manifest, and reports
`capture_blocked_missing_credential`. There is no partial corpus, because a
half-corpus that looks whole is worse than none.

## 9. Reproducibility

- Raw responses are stored byte-for-byte **before** anything parses them. A
  canonicalisation bug is recoverable only if the source survives.
- Request identity is deterministic — provider, instrument, timeframe,
  adjustment, range, page, cursor — and excludes the wall clock. Capture time
  is metadata, never identity.
- A repeated request identity is refused unless the bytes are identical; two
  different answers to one question cannot both be the answer.
- `verify` recomputes every row and hash from raw + spec + calendar, offline. A
  test replaces `urlopen` with something that raises to prove it never reaches
  for a socket.
- `rebuild` deletes the canonical files, regenerates them from raw and asserts
  **byte-identity**.
- A capture refuses to run into a non-empty directory, so two attempts can
  never be blended invisibly.

## 10. Storage

```
data/equities/massive_us_equity_v1/
  spec.json
  manifest.json
  raw/<venue>_<symbol>/<symbol>_page_NNNN_<identity>.json
  canonical/<venue>_<symbol>.jsonl
  corporate_actions/<venue>_<symbol>.json
```

JSONL for canonical rows, one row per line, fixed key order, `Decimal` values
serialised as strings — a JSON number would be read back as a float by most
readers and the hash would then depend on the reader. The same layout the
Coinbase corpus uses.

## 11. What this phase does not do

No prediction, no feature, no model, no benchmark, no backtest, no paper
trading, no equity in the tradable registry, no P&L. The equities remain
catalogue-only. No crypto contract is touched, no BTC/ETH request is made, and
the protected holdout is neither observed nor spent.

`commercial_edge_established = false`.

---

## Results

*No capture has completed. This section is appended by the data commit when one
does, with row counts, first/last bars, gap audit, split records, sizes and
hashes.*
