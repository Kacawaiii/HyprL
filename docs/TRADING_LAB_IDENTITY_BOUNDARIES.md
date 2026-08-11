# Identity boundaries

Phase 6A found that the protected-holdout guard compared a raw external
product string against a canonical internal one: `product in self.products`.
Twelve of thirteen spellings of a reserved instrument reported *not
protected*. Nothing in the codebase happened to send those spellings, so two
phases of tests never noticed.

This document is the rule that prevents the next one. It is the source of
truth for Phase 6B and beyond: before adding a boundary that compares two
identities, find its row here.

## The five classes

| Class | What | Rule |
|---|---|---|
| **A — semantic** | instrument, venue, provider, timeframe | parse → canonicalise → compare. Aliases welcome. |
| **B — opaque** | sha256 digests, protocol and schema versions | byte-exact. Never normalised. |
| **C — display** | labels, descriptions | never compared for identity at all |
| **D — legacy** | `BTC-USD` in artefacts and the event log | immutable; reached through an adapter |
| **E — runtime** | session ids | validated, never canonicalised, never a path |

The asymmetry is the point. Being liberal about *recognising* an instrument
is what makes the holdout guard safe. Being liberal about recognising a
digest is what makes an integrity check meaningless.

## The matrix

| Identity | Canonical form | Aliases accepted? | Parser | Matching | Legacy form | On failure |
|---|---|---|---|---|---|---|
| **InstrumentId** | `coinbase:BTC-USD` | yes — case, `/ _ .`, no separator, whitespace, NFKC, venue prefix | `instruments.InstrumentId.parse` / `normalize_symbol` | semantic, via registry | `BTC-USD` | `IdentityError`, no fallback |
| **Venue** | `coinbase` (lower) | case, whitespace | `instruments.normalize_venue` | semantic | — | `InstrumentError` |
| **ProviderId** | `coinbase-public-v1` | **no** | registry lookup | exact, closed set | — | `RegistryError` → 404 |
| **Timeframe** | `1h` | `1H`, ` 1h `, `60m`→`1h` | `instruments.Timeframe.parse` | semantic | `1h`, `1d` | `InstrumentError`, never falls back to `1h` |
| **ProtocolVersion** | `trading-lab.signal-engine.v1` | **no** | — | exact | — | `IdentityError` |
| **SpecHash** | 64 lowercase hex | **no** | `identity.require_exact_digest` | byte-exact | — | `IdentityError` |
| **ResultHash** | 64 lowercase hex | **no** | same | byte-exact | — | `IdentityError` |
| **EventHash** | 64 lowercase hex | **no** | store recomputes | byte-exact | — | chain reports unverified |
| **SessionId** | `paper-YYYYMMDDTHHMMSSZ` | **no** | `identity.require_session_id` | exact | — | `IdentityError` |
| **Cursor** | opaque base64 | **no** | `pagination.decode_cursor` | exact, bound to endpoint + product + query digest | — | `AppApiError` |

### Notes that matter

**Timeframe folds minutes into hours, not hours into days.** `60m` and `1h`
are one duration on every calendar, so they are one identity. `24h` and `1d`
are equal only on a market that never closes; folding them would bake a
crypto assumption into the identity layer — the exact mistake the trading
calendar exists to prevent.

**Venue is not provider.** `coinbase` the venue and `coinbase-public-v1` the
market-data provider are different identities with different registries. The
same instrument could one day be served by a second provider.

**The API requires canonical spelling.** `/api/v1/markets/BTC-USD` works;
`/api/v1/markets/btc-usd` is a 404. This is the narrower of the two valid
choices under class A. Accepting aliases would be *safe* — everything
downstream receives the resolved id — but it widens a read-only API's input
surface for no caller that needs it, and one resource under many URLs is a
caching and logging nuisance. The registry still decides, so the rule is
stated once instead of inferred from a membership test.

**Digests are validated, never repaired.** A caller holding an uppercase or
padded digest has a bug upstream; quietly fixing it hides that bug behind a
check that then proves nothing. Note that `^…$` is *not* a safe anchor in
Python — `$` also matches before a final newline, so `"<digest>\n"` passes.
Use `\Z`.

## Cross-product guards

The failure this whole document is about: a BTC candle and an ETH candle are
the same six numbers, a BTC target stream and an ETH target stream are the
same dataclass, and a model fitted on one predicts happily from the other's
features. Nothing raises a type error. The result is plausible and wrong.

| Boundary | Guard | Enforced by |
|---|---|---|
| model artefact × inference product | `load_paper_model(artifact, product=…)` | `paper_model` |
| targets × market bars | `run_economic_backtest(…, bars_product=…)` — **required, no default** | `economic_backtest` |
| candle × session product | `ingest_candle(…, row_product=…)` + slot membership | `paper_engine` |
| bar × series | `InstrumentBar.require_instrument`, `require_single_instrument` | `market_providers` |
| cursor × product | endpoint + product + query digest | `app_api.pagination` |
| any product × reserved window | canonical, venue-blind, fail-closed on unreadable input | `protected_holdout` |

## Rules for new code

1. Never compare a value that arrived from outside with an internal identity
   using `==` or `in`. Route it through `scripts/trading_lab/identity.py`.
2. Unknown is never equal to anything — including another unknown. There is
   no fallback to a default product, the first registry entry, or Coinbase.
3. A function that takes two things belonging to the same instrument must
   take that instrument's identity too, without a default. A default of
   `None` reopens the hole for every caller that forgets.
4. Do not normalise a digest, a protocol version, or a session id.
5. Do not rewrite a legacy identity. Committed artefacts and the hash-chained
   event log are keyed by `BTC-USD`, and the hashes exist so that cannot
   change. Resolve down to it with `resolve_legacy_product`.
6. The frontend never canonicalises. It receives `instrument_id` from the
   backend and passes it back opaquely; symbols are for display only.

`tests/crypto/test_identity_boundaries.py` enforces 1 with an AST scan and a
documented allowlist. A new raw product comparison must be fixed or written
down there with a reason — and a stale allowlist entry fails too, so the list
cannot rot into a comment that no longer describes anything.
