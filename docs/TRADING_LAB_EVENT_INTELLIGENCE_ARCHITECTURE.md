# Event Intelligence — causal architecture (design only)

```
STATUS: DESIGN ONLY

NO SCRAPER · NO PROVIDER · NO NETWORK · NO LLM CALL

NO EVENT CORPUS · NO MODEL · NO SIGNAL · NO BACKTEST

DESIGN SPEC HASH
8d3f9b151ffc204b93de58c7802475dc2d23bd9e54ded58be42481507cd7724f
```

Canonical spec: `docs/artifacts/event_intelligence_design_v1.json`. The hash is
`sha256` of that file's canonical JSON and is reproducible with:

```bash
python -c "import json,hashlib;print(hashlib.sha256(json.dumps(json.load(open('docs/artifacts/event_intelligence_design_v1.json')),sort_keys=True,separators=(',',':')).encode()).hexdigest())"
```

---

## 0. Why this shape

Phase 6F established that six technical price features carry no measurable
out-of-sample signal at a 5-session horizon. The honest reading is not "use
more indicators" — it is that price history alone is a weak description of
what a trader actually reacts to. A trader reacts to *information arriving*,
against *expectations already held*, in a *market context*.

So the object this system needs is not a better feature. It is a defensible
answer to one question:

> **What could HyprL have known at time T, and nothing more?**

Everything below exists to make that question answerable, auditable, and
replayable offline. None of it is implemented here.

## 1. What HyprL knows at T

`InformationSnapshot(as_of=T)` is the only object a future model may consume.
It contains, and can contain nothing else:

- event revisions whose `available_at <= T`, one per logical event — the
  latest such revision, chosen deterministically;
- the causal cluster state as it stood at T;
- market data already available at T;
- enrichments computed **only** from inputs available at T.

It is deterministic and hashed. Two rebuilds from the same raw artefacts
produce the same snapshot hash, with no network.

This is not a new invention. Phase 1C already implements exactly this shape
for market bars — `as_of` as a cutoff over a declared availability timestamp,
deterministic revision selection, fail-closed ties, immutable manifests,
bounded reads, replay. Event Intelligence extends that machinery to a second
record type rather than building a second causal store beside it.

## 2. Which timestamp gates visibility

Five timestamps, one gate.

| field | meaning |
|---|---|
| `published_at` | the instant the **source declares** as first publication |
| `observed_at` | the first instant **our collector** saw the resource |
| `ingested_at` | the instant the revision became durable in our store |
| `effective_at` | the **economic** instant the event applies to, when different |
| `source_updated_at` | the source's own last-modified, when exposed |

The gate is:

```
available_at = max(published_at_trusted, observed_at)
visible to as_of(T)  ⟺  available_at <= T
```

**Why the max, and not either alone.** `published_at` alone proves only when a
source *says* it published; it is self-reported, frequently wrong, sometimes
backdated, and for a scraped page often just a CMS field. `observed_at` alone
is honest but throws away real information: an official CPI release genuinely
was public at 08:30:00 ET even if our poller noticed at 08:30:04. The maximum
is never earlier than the moment we could actually have known, which is the
only property a causal gate needs.

`published_at` is *trusted* — allowed to win the max — only when the source is
TIER_1_OFFICIAL or TIER_2_PRIMARY, clock precision is minute or better, and
the observation was LIVE. Otherwise it falls back to `observed_at`. A media
article claiming 14:00 while we first saw it at 15:40 becomes available at
15:40.

`effective_at` never gates visibility. A CPI release at 08:30 concerns the
prior month; the month is `effective_at`, the release instant is what a trader
could act on.

The repository already carries this pattern: `market_data_store` stores
`available_at` beside `ingested_at` and combines bounds conservatively with
`_later_timestamp`. This design generalises a rule that already exists here.

### Clock precision is a field, not an assumption

`second | minute | hour | day | unknown`. A date-only historical item is
**not** placed at 00:00 and treated as known from midnight — that would hand a
model most of a day of free lookahead. It becomes available at the **end** of
its day in the source's timezone. `unknown` precision is excluded from causal
queries entirely: fail closed.

## 3. Revisions: what we knew at 14:05 vs 14:10

Nothing is ever overwritten. A logical event has a stable id; each observed
version is an immutable revision with its own `available_at`.

```
14:00  revision 1  headline A                  available_at 14:00
14:07  revision 2  article updated             available_at 14:07
14:15  revision 3  correction                  available_at 14:15
```

`event_revisions(event_id, as_of=14:05)` returns revision 1. At 14:10,
revision 2. The 14:15 correction cannot retroactively change what a 14:05
snapshot contains — which is precisely the failure mode that makes naive news
backtests worthless.

Selection is `max(available_at) <= as_of`. Two revisions at the identical
maximal timestamp are a **conflict, not a coin flip** — the same fail-closed
choice Phase 1C already makes for market receipts.

## 4. Live vs historical backfill

These are different epistemic objects and must be labelled as such.

| | `observation_mode=LIVE` | `HISTORICAL_BACKFILL` |
|---|---|---|
| `observed_at` | genuinely recorded | **must not be fabricated** |
| `available_at` | `max(published_at_trusted, observed_at)` | derived from `published_at` alone |
| causal strength | strong | weaker, and marked |

A backfill can never claim to know when we would have observed something in
2024. Pretending otherwise is the single easiest way to manufacture a fake
edge. A future benchmark may therefore require a minimum causal quality grade
(A–E) and refuse to mix a date-only D item with an exact A release without
saying so.

## 5. Twenty articles, one fact

Two layers, and the lower one is never destroyed.

**SourceObservation** — every publication actually observed. Immutable,
raw-backed, one per source item revision. Twenty articles are twenty
observations, because twenty things really were published.

**CanonicalEvent** — a cluster of observations describing the same fact. This
is what "one event" means for analysis: a rate decision covered by Reuters,
Bloomberg and the Fed's own site is one event with three observations.

Clustering is a *derived, revisable opinion*. If the clustering algorithm
improves next year, it is recomputed — and the observations and raw artefacts
underneath are untouched, so every historical snapshot can be rebuilt under
either clustering version.

### Keeping clustering causal

This is the subtle leak. Consider:

```
14:00  a small ambiguous wire item
15:00  a detailed story revealing the item was a major announcement
```

A cluster built today knows both. A snapshot at 14:10 must not. So cluster
state is a **function of `as_of`**: `cluster_state(as_of=T)` is computed from
observations with `available_at <= T` only, and carries its own hash. The
enriched 15:00 cluster exists only for `T >= 15:00`.

This costs recomputation. It is the difference between measuring foresight and
measuring hindsight.

## 6. Official macro releases and surprise

Scheduled releases get a specialised record, because the interesting quantity
is not the number.

```
indicator · period · actual · consensus · previous · revised_previous
unit · release_timestamp
```

**Consensus is causal data in its own right**, not an attribute of the
release. It has a value, an `available_at`, a provider, and its own revision
history — because expectations move in the days before a print, and the
consensus that matters is the one that existed *before* the release.

```
surprise_raw    = actual − consensus
surprise_pct
surprise_zscore = surprise_raw / σ(prior surprises available before this release)
```

Two rules that are easy to get wrong and fatal when wrong:

1. A consensus figure published **after** the release may never serve as the
   pre-release expectation.
2. The `σ` normalising a z-score uses only releases available before this one.
   Computing σ over the whole sample leaks the future into every observation.

Revisions matter here too: `previous` may be restated later. The restatement
is a new revision with its own `available_at`, never an edit.

## 7. Entities

Canonical namespaces, reusing the identity discipline already in the repo:

```
asset:coinbase:BTC-USD      instrument:xnas:NVDA
org:federal-reserve         country:US        currency:USD
```

Every link carries **method, confidence and provenance**. "Apple" in a story
about fruit tariffs is not `instrument:xnas:AAPL`. An ambiguous mention links
to no instrument — fail closed, the same rule the instrument registry already
applies to `yahoo:AAPL`.

## 8. Where an LLM may and may not be used

```
fetch → persist raw → deterministic normalization → [optional] enrichment
```

Everything left of the bracket runs without a model and is the source of
truth. An LLM may propose `sentiment`, `importance`, `novelty`, `summary`,
`entity candidates`, `embedding`.

An LLM may **never** produce `published_at`, `observed_at`, source identity,
or raw content. Provenance is not a judgement call.

Every enrichment record binds `model_id`, `model_version`,
`prompt_spec_hash`, `computed_at`, `input_content_hash`. Re-analysis under a
new model **appends** a revision; it never overwrites. Model drift then
becomes visible and datable instead of silently rewriting history.

If no model is available, ingestion still works and events remain valid —
degraded, not broken.

### Sentiment is entity-conditioned

A single event is positive for USD, negative for BTC and neutral for AAPL. A
scalar "positive/negative headline" is the wrong shape. The record is an
opinion per entity — direction, confidence, horizon hint — and it is a
**derived feature, never ground truth**.

## 9. Market context vs outcome

The same price series serves two roles that must never touch.

| | window | may enter a snapshot |
|---|---|---|
| **context** | strictly ≤ `available_at` | yes |
| **outcome** | strictly > `available_at` | **never** |

Context: returns over 5m/1h/24h before T, volatility, volume, trend and
correlation regime. Outcome: returns at +5m/+30m/+1h/+6h/+1d/+5 sessions, max
favourable and adverse excursion, volatility expansion.

They live in **separate artefacts**, joined only by `event_id` at analysis
time. No object a causal query returns has an outcome field reachable from it.
That is a structural guarantee, not a convention — the 6F leakage audit showed
how cheap it is to check a structural rule and how expensive an implicit one
is to trust.

`LargeMoveDefinitionSpec` (asset, horizon, threshold, direction, and a
volatility-normalised alternative) is deliberately **not defined now**.
Choosing a threshold after looking at recent moves is exactly how a
research question gets selected by its answer. It must be frozen before use, like every
other spec in this repository.

## 10. No news vs no collector

Four distinct states, and none of them is zero:

```
NO_EVENTS_OBSERVED · SOURCE_UNAVAILABLE · SOURCE_NOT_CHECKED
RATE_LIMITED · PARSER_FAILED
```

"Quiet market" and "our collector was down" produce identical zeros under a
naive encoding, and a model will happily learn that outages predict calm. Each
poll records its outcome, so absence is always attributable.

## 11. Storage and replay

**Decision: reuse the Phase 1C architecture — SQLite append-only receipts plus
immutable manifests — rather than build a second causal store.**

The requirements list (point-in-time query, revision history, deterministic
replay, corruption detection, bounded reads, durability) is the list Phase 1C
already satisfies, and a second implementation of causal storage is a second
thing to keep correct. Raw payloads are content-addressed files beside the
database, exactly as the Yahoo corpus stores raw beside canonical.

Identities are derived, not random: `SourceObservationId` from
(provider, source_item_id, content_hash); `RevisionId` from
(observation, available_at); `EnrichmentId` from
(input_content_hash, model_id, model_version, prompt_spec_hash).
`CanonicalEventId` is the exception — a cluster is an opinion, so it is
versioned by clustering algorithm and `as_of`, never presented as an intrinsic
property of the world.

Replay rebuilds observations, normalization and deterministic state from raw
with no network. Enrichments replay only under an identical model version;
otherwise they are retained as versioned artefacts and their absence is
explicit.

Corruption is fail-closed, following the local corpus: hash the raw, hash the
normalized observation, bind both to a manifest, and never silently re-fetch
from the network to repair a read.

## 12. Redistribution

Article text is very often `redistribution_permitted=false`, and normalizing
it does not launder it. Four tiers:

| tier | example | may be committed |
|---|---|---|
| restricted raw | source HTML/JSON | never |
| normalized text | cleaned article body | never, when the source restricts |
| derived metadata | taxonomy label, entity ids, timestamps, hashes | yes |
| aggregate results | counts, metrics | yes |

The existing `assert_no_restricted_data` guard already refuses to ship a
dataset whose manifest forbids redistribution; event artefacts declare the
same flag and inherit that protection.

## 13. Provider shortlist for a future 6G-A

Ordered, deliberately short. **Every property below is a design assumption and
carries `REQUIRES WEB VERIFICATION BEFORE FREEZE`** — no network call was made
in this phase, so none of it is confirmed.

| # | provider / category | purpose | tier | timestamp quality | historical | live | credentials | redistribution | complexity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | US macro official (BLS, BEA) | CPI, payrolls, GDP | TIER_1 | exact release time expected | expected good | scheduled poll | likely none | likely permissive (US gov) | low–medium |
| 2 | Federal Reserve | FOMC statements, minutes | TIER_1 | exact | expected good | scheduled poll | likely none | likely permissive | low |
| 3 | SEC EDGAR | filings, 8-K, earnings | TIER_1 | exact acceptance time | good | poll | likely none | likely permissive | medium |
| 4 | Crypto regulatory / exchange official | ETF decisions, incidents | TIER_1/2 | mixed | uneven | poll | varies | varies | medium |
| 5 | One general news/RSS family | breadth, cross-source dedup | TIER_3 | minute at best | poor | poll | varies | **often forbidden** | medium |

The ordering is deliberate: sources 1–3 are the ones where an exact,
trustworthy `published_at` exists, which is what makes causal grade A possible
at all. The news layer comes last because it is the weakest causally and the
most restricted legally — it earns its place by enabling dedup and coverage,
not by being a primary timestamp.

## 14. The smallest useful V1

**6G-A should implement the EventStore plus exactly one provider.**

Not five. The thing that must be proven first is not coverage but causality:
that a real source can be fetched raw-first, revised without overwriting,
queried `as_of(T)`, and replayed offline to an identical snapshot hash. One
official provider with exact timestamps — the Federal Reserve or a single BLS
release — exercises every invariant with the least surface.

Assets initially observed: BTC, ETH, QQQ, AAPL, MSFT, NVDA — the six already
in the catalogue, so entity linking is testable against a registry that
already exists and already fails closed.

Sequence, each with an independent counter-review:

```
6G-DESIGN  (this)
  → 6G-A   EventStore + one provider
  → 6G-B   multi-provider ingestion + dedup
  → 6G-C   historical/live event corpus
  → only then: event features, and a frozen predictive experiment
```

No model is built until the corpus is capturable and replayable. 6F's value
was that its protocol was frozen before its result existed; the same ordering
applies here.

## 15. Invariants

| # | invariant |
|---|---|
| E1 | No event revision is visible to `as_of(T)` unless `available_at <= T`. |
| E2 | A revision never overwrites a previous one; all history is append-only. |
| E3 | `HISTORICAL_BACKFILL` is never labelled live-observed, and never fabricates `observed_at`. |
| E4 | Cluster state `as_of(T)` uses only observations with `available_at <= T`. |
| E5 | No LLM-derived value may replace a provenance timestamp or source identity. |
| E6 | Source outage, not-checked, rate-limited, parser-failed and zero-events are distinct states; none is encoded as 0. |
| E7 | Restricted raw or normalized text never enters a release, archive or support bundle. |
| E8 | `InformationSnapshot(T)` is deterministic, hashed, and rebuildable offline. |
| E9 | No post-event market data is reachable from any object a causal query returns. |
| E10 | A consensus used for surprise must have been available before the release. |
| E11 | A z-score normalisation uses only observations available before the observation it normalises. |
| E12 | Raw bytes are durable and hashed before any parse produces an observation. |
| E13 | Two revisions tied at the maximal `available_at` are a conflict, not a silent pick. |
| E14 | A date-only item becomes available at end-of-day, never 00:00. |
| E15 | `unknown` clock precision is excluded from causal queries. |
| E16 | An ambiguous entity mention links to no instrument. |
| E17 | Every entity link carries method, confidence and provenance. |
| E18 | Every enrichment binds model id, version, prompt hash and input content hash. |
| E19 | Re-analysis appends an enrichment revision; it never mutates an existing one. |
| E20 | `effective_at` never gates visibility; only `available_at` does. |
| E21 | No query parameter can bypass the `available_at <= as_of` gate. |
| E22 | A source observation is never deleted or merged away by clustering. |

## 16. Threat model

| threat | required behaviour |
|---|---|
| wrong/backdated `published_at` | untrusted tier falls back to `observed_at`; max rule bounds the damage |
| source revises content in place | new revision; old one still returned for earlier `as_of` |
| duplicate news flood | many observations, one canonical event; observation count kept as a feature, not as event count |
| same event across sources | clustering, with observations preserved |
| malicious HTML / oversized payload | bounded read, raw stored as bytes, parser sandboxed from the store |
| hostile redirect | reuse `safe_http`: allowlist, no credential forwarding, revalidate final URL |
| credential leak | never logged, never in raw artefacts, never in support bundle — the existing rule |
| source outage | `SOURCE_UNAVAILABLE`, never zero events |
| parser drift | `parser_version` on every observation; reparse appends |
| timezone ambiguity | store tz-aware instants plus precision; refuse naive local times |
| future enrichment leakage | enrichment `computed_at` and inputs gated by `as_of` |
| backfill posing as live | `observation_mode` is mandatory and audited |
| article edited retroactively | revisions; earlier snapshots unchanged |
| consensus revised after release | consensus revision history with its own `available_at` |
| LLM model drift | versioned enrichment revisions; old snapshots reproducible |
| redistribution leak | four-tier policy + existing release guard |
| corrupt local artefact | hash mismatch → fail closed, no network repair |

## 17. Carried-forward findings from 6F

Two LOW items from the 6F counter-review, recorded here rather than patched:

- **A.** Add a permanent test proving a genuinely missing equity session breaks
  the indicator segment. The behaviour was verified correct during the
  counter-review, but no shipped test constructs a discontinuous series.
- **B.** Strengthen the scaler-provenance test to assert the *identity* of the
  252 training rows, not only `len(X) == 252`. Content was verified
  independently during the counter-review; the repo test pins cardinality only.

## 18. What this design does not establish

No scraper exists. No event has been captured. No provider property has been
verified against the web. There is no event corpus, no enrichment, no model,
no signal, no backtest.

`commercial_edge_established = false`, and this phase does not move it.
