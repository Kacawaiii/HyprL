# Event Intelligence — causal architecture (design only)

```
STATUS: DESIGN ONLY

NO SCRAPER · NO PROVIDER · NO NETWORK · NO LLM CALL

NO EVENT CORPUS · NO MODEL · NO SIGNAL · NO BACKTEST

DESIGN SPEC HASH (revision 4 — authoritative)
4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689

supersedes  8d3f9b15… (rev 1) → 25a8839f… (rev 2) → 9ebd1972… (rev 3)
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

## 1. What HyprL knows at T — and in which sense

There are two honest answers to "what was known at T", and conflating them is
the single most expensive mistake this architecture can make. So the mode is
**mandatory**, part of the snapshot's identity, and has no default:

```
InformationSnapshot(as_of=T, visibility_mode=DURABLE_OBSERVED)
InformationSnapshot(as_of=T, visibility_mode=RETROSPECTIVE_SOURCE,
                    minimum_causal_quality=...)
```

**DURABLE_OBSERVED** answers *what had HyprL actually observed and made durable
by T*. It gates on `durable_available_at <= T`. This is the reference for live
operation, exact replay, and auditing a decision that was really taken.

**RETROSPECTIVE_SOURCE** answers *what the world could have known at T
according to source timestamps recovered later*. It gates on
`source_available_at <= T` and requires an explicit minimum causal quality. It
is for historical research only and may **never** be presented as
`LIVE_OBSERVED` or as "HyprL knew this at T".

Either way a snapshot contains only: event revisions visible under that mode,
the cluster state resolved at T, enrichments visible under that mode, market
data available at T, and the durably recorded source-health state. It is
deterministic and hashed, and the two modes produce **distinct identities**.

This is not a new invention. Phase 1C already implements this shape for market
bars — `as_of` as a cutoff over a declared availability timestamp,
deterministic revision selection, fail-closed ties, immutable manifests,
bounded reads, replay. Event Intelligence extends that machinery to a second
record type rather than building a second causal store beside it.

## 2. Which timestamp gates visibility

Three timestamps are recorded independently, and **none is ever rewritten from
another**.

| field | meaning |
|---|---|
| `source_available_at` | conservative instant the **source** claims it became public — `null` when the source timestamp fails the trust policy |
| `observed_at` | the instant **our collector** actually obtained the bytes |
| `ingested_at` | the instant the normalized revision became **durable**, and therefore replayable |

Two bounds are derived from them:

```
live_observed_available_at = max(source_available_at if trusted, observed_at)
durable_available_at       = max(live_observed_available_at, ingested_at)
```

`live_observed_available_at` measures the earliest defensible *collector*
knowledge. It is a diagnostic — a latency measure — and it **does not gate**
the snapshot.

`durable_available_at` measures the earliest instant the *system* could
durably reproduce the revision, and **it is the gate** for DURABLE_OBSERVED.

### Why durability, and not observation

A collector can observe at 14:00:04 and crash before the durable write, which
only lands at 14:03:

```
published 14:00:00 · observed 14:00:04 · ingested 14:03:00
durable_available_at = 14:03:00

DURABLE_OBSERVED snapshot at 14:01  →  ABSENT
DURABLE_OBSERVED snapshot at 14:04  →  VISIBLE
```

Gating on the observation would let a replay at 14:01 assert knowledge the
durable store did not hold — a window exactly as wide as a retry or an
incident, which is when markets move. A replay can only honour what was
durable, so durability gates the snapshot. The observation bound is kept
because it is the honest measure of collector latency; it is simply not the
thing a replay can promise.

Within the max, `source_available_at` may win only when the timestamp is
**trusted** — and trust is a versioned decision, not a judgement made at query
time.

### TimestampTrustPolicyV1

Evaluated **at ingestion**, recorded on the revision, and versioned. Inputs:
provider class, source tier, provenance, timestamp precision, timezone
confidence, source timestamp semantics, observation mode. Verdicts:

```
TRUSTED_EXACT · TRUSTED_CONSERVATIVE · UNTRUSTED · UNKNOWN
```

Trust requires TIER_1_OFFICIAL or TIER_2_PRIMARY, precision of a minute or
better, and a known timezone. `UNTRUSTED` sets `source_available_at = null`, so
only `observed_at` can bound the revision. A media article claiming 14:00 that
we first saw at 15:40 becomes collector-available at 15:40, and durable when it
is written.

### Clock anomalies

Source clock ahead, source clock behind, collector clock anomaly, timezone
parse anomaly: all three timestamps are retained separately and none is
silently corrected. An anomaly produces a diagnostic flag, a causal-quality
degradation, or a fail-closed refusal depending on severity. Where both
timestamps are usable, the `max` is already the conservative resolution. No NTP
machinery is implied.

### Interval timestamps resolve to the start of the next unit

A precision is not a semantics. `precision = minute` does not by itself mean
"published somewhere inside that minute" — a provider contract may define a
minute timestamp as an exact instant truncated for display. So the trust policy
binds **precision and timestamp semantics together**, and the resolution
follows the provider's declared semantics:

```
EXACT_INSTANT · INTERVAL_START · INTERVAL_UNKNOWN · DATE_ONLY · UNKNOWN
```

When the timestamp identifies an **interval**, `source_available_at` is the
**start of the next unit** in source-local time:

| precision | example (Europe/Paris) | `source_available_at` |
|---|---|---|
| EXACT_INSTANT | 14:32:07 declared exact | **14:32:07** — no artificial +1 unit |
| second (interval) | 14:32:07 | 14:32:08 → UTC |
| minute | 14:32 | **14:33:00** → UTC |
| hour | 14h | **15:00:00** → UTC |
| day | 2026-09-05 | **2026-09-06 00:00:00** → UTC |
| unknown | — | **unavailable** for strict causal use |

The reasoning is the same at every scale: we do not claim knowledge inside an
interval we cannot place. An official release whose contract guarantees an
exact publication instant is used at that instant — adding a second there would
be false conservatism, and the trust policy is what distinguishes the two
cases. Which semantics a given provider actually offers is a question for the
web-verification phase, not an assumption made here.

Timezone comes from the frozen provider contract and DST is resolved by the
timezone database. An **unknown timezone yields no `source_available_at`** and
never silently defaults to UTC.

### `effective_at` never gates anything

It describes the economic period an event concerns. July CPI published in
August is visible in August and never in July.

## 2b. What "durable" means

Revision 2 made `ingested_at` the gate and then used the word *durable*
intuitively. That is not good enough for the quantity a whole visibility rule
rests on, so it is defined here against the only event that can carry it.

> A normalized `EventRevision` is **durable** only after the transaction that
> makes it visible in the authoritative EventStore has **committed
> successfully**.

Four things therefore **do not exist** for any snapshot: an uncommitted
attempt, an in-flight transaction, a rolled-back transaction, and a failed
commit. Not "exist but invisible" — they have no `ingested_at` at all, and
`durable_available_at` is undefined. There is no fallback to `observed_at`.

**Two authorities, no competition.** The raw artefact is authoritative for
source bytes and provenance. The EventStore commit is authoritative for the
visibility of the normalized revision. A durable raw file alone never makes a
revision visible.

### How `ingested_at` is assigned

Requiring a wall clock read *after* commit and written back would need a second
transaction, so the semantics are stated instead of a fictional API:

> **Transaction-assigned logical commit bound.** The value is assigned inside
> the transaction and becomes authoritative **only if that transaction
> commits**. On rollback or failure it is discarded and never was an
> `ingested_at`.

A pre-commit wall clock may be retained as a `prepared_at` diagnostic. It is
never durable evidence and never affects visibility.

```
attempt 1  prepared_at 14:02:59.900 · commit FAILS   → no ingested_at
attempt 2  commit succeeds                            → ingested_at = 14:07
```

14:02:59.900 is never resurrected, even when the content bytes are identical.
A failed attempt cannot donate its timestamp to a later success.

### The one case that is not a failure

A commit can succeed and the acknowledgement be lost — the process dies after
the write. That revision **is** durable and authoritative. Deterministic
identity plus a uniqueness constraint let the retry *rediscover* it rather than
write a duplicate or a later `ingested_at`. Phase 1C already does exactly this:
an existing snapshot id is verified field by field and entry by entry rather
than trusted.

The distinction matters and is easy to get backwards:

| | outcome |
|---|---|
| commit **failed** | retry gets a **new** durable boundary |
| commit **succeeded**, ack lost | the **original** committed boundary stands; no duplicate |

### Concurrency

V1 assumes a single authoritative SQLite EventStore with serialized write
transactions; commit ordering at the store boundary resolves durable ingestion
order. No distributed consensus is implied. Two collectors racing on the same
content produce exactly one revision; two genuinely different revisions may
both commit, each with its own boundary, and normal causal selection applies.

Store commit order may disambiguate *storage* chronology. It never overrides
the equal-source-timestamp conflict rule, and a rowid is never business truth.

### Honest durability

"Durable" means committed under the store's frozen durability configuration —
the Phase 1C contract is `journal_mode=WAL` with `synchronous=FULL` — not a
metaphysical claim that no hardware failure could ever lose it. `ingested_at`
carries a time value, but its authority is ordering and durability, not
absolute clock accuracy: a clock skew never makes an uncommitted row durable,
and source or observed timestamps are never rewritten to fit it.

### The crash matrix

| state | raw | normalized | `ingested_at` | event visible | retry |
|---|---|---|---|---|---|
| **A** response received, nothing written | no | no | — | **no** | refetch |
| **B** raw durable, normalized absent | yes | no | — | **no** | reparse from raw, no network |
| **C** normalized transaction in flight | yes | no | — | **no** | re-attempt commit |
| **D** normalized commit failed / rolled back | yes | no | discarded | **no** | new attempt, **new** boundary |
| **E** normalized commit succeeded | yes | yes | valid | **yes**, once `durable_available_at <= T` | — |
| **F** commit succeeded, ack lost | yes | yes | the **original** boundary | **yes** | idempotent rediscovery, no duplicate |
| **G** enrichment computed, not persisted | — | — | — | event **yes**, enrichment **no** | re-persist |
| **H** enrichment committed | — | — | — | enrichment **yes** once its own boundary ≤ T | — |
| **I** cluster merge computed, not committed | — | — | — | **prior** cluster state stands | re-attempt |
| **J** cluster merge committed | — | — | — | new `ClusterStateRevision` visible | — |

The commit rule applies identically to event revisions, enrichments, cluster
state revisions and **source-health records**. A failed write of
`SOURCE_UNAVAILABLE` is not durable knowledge of an outage; it is nothing.

### One consistent read

A snapshot is built from **one consistent bounded store view**. It may never
combine events read before a concurrent commit with clusters or source-health
read after it — that would describe a state the store never occupied. Phase 1C
takes `BEGIN IMMEDIATE` before its selection for precisely this reason.

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

Different epistemic objects, labelled as such and never relabelled.

| | `LIVE` | `HISTORICAL_BACKFILL` |
|---|---|---|
| `observed_at` | genuinely recorded | the instant of **the backfill**, never reconstructed |
| `ingested_at` | when the revision became durable | the backfill's durable write |
| DURABLE_OBSERVED at historical T | as computed | **effectively absent** — `ingested_at` is the backfill instant |
| RETROSPECTIVE_SOURCE at historical T | n/a | admissible **only** if `source_available_at` is determinable, trusted, precision known, and causal quality meets the study's minimum |

An article published in 2024 and backfilled in 2027 carries
`observed_at = 2027`. Writing 2024 there would be a fabrication, and it is the
easiest way to manufacture a fake edge. It is therefore invisible to any
DURABLE_OBSERVED snapshot of 2024, and visible to a RETROSPECTIVE_SOURCE
snapshot only under an explicit quality floor.

Items that fail the floor are still **stored** — they are simply inadmissible
for strict causal research, rather than quietly mixed in.

### Causal quality grades

| grade | criteria |
|---|---|
| A | official/primary, exact timestamp, LIVE, strong provenance |
| B | reliable source, minute precision, trusted timestamp |
| C | weaker timestamp semantics, media tier |
| D | date-only, conservative next-day-start availability |
| E | unknown timing or unknown timezone — excluded from strict snapshots |

A future study states `minimum_causal_quality >= B` and the policy decides,
with no ad-hoc human interpretation at query time.

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
14:30  a second source: "the announcement concerns ETF approvals"
15:00  official confirmation
```

A cluster built today knows all three. A snapshot at 14:10 must not. So cluster
state is a **function of `as_of` and of the visibility mode**: it uses only
observations visible under that mode at T, and carries its own hash. The
enriched cluster exists only for `T >= 14:30`.

This costs recomputation. It is the difference between measuring foresight and
measuring hindsight.

### Merge, split, and lineage

Clustering is a **derived state versioned in time**, with an append-only
lineage rather than a mutable label:

```
ClusterStateRevision · ClusterLineageId · parent_cluster_ids[]
operation ∈ { CREATE, MERGE, SPLIT, RECLASSIFY }
```

**Merge** — X and Y look separate at T1 and are recognised as one event at T2.
Snapshot T1 still shows two; snapshot T2 shows the merged state.

**Split** — X looks single at T1 and is recognised as two events X1 and X2 at
T2. Snapshot T1 still shows one; snapshot T2 shows the split.

Neither operation rewrites a prior cluster state or a prior snapshot.
`CanonicalEventId` is therefore resolved **through the lineage as of T** — it is
never a final identity computed with future observations, which would make
every historical snapshot depend on today's opinion.

### Syndication: a hostname is not a source

A Reuters dispatch republished by fifteen domains is not fifteen independent
confirmations. Observations carry a source lineage — `publisher`,
`origin_publisher`, `syndication_parent`, `wire_service`, `source_lineage_id`,
`provenance_confidence` — and the counts stay distinct:

```
observation_count = 15   publisher_count = 15   syndicated_copy_count = 14
independent_source_count = 1
```

When provenance is unavailable, `independent_source_count` is **unknown or
low-confidence — never the hostname count**. A flood of copies is a real
attention signal, and attention is not confirmation; keeping
`observation_count`, `publisher_count`, `independent_source_count` and
`velocity` as separate metrics is what stops one from being read as the other.

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
   pre-release expectation. Under DURABLE_OBSERVED the consensus must have been
   **durably** available before the release; under RETROSPECTIVE_SOURCE its
   `source_available_at` must precede the release and meet the quality floor.
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
new model **appends** a revision; it never overwrites. Model drift then becomes
visible and datable instead of silently rewriting history.

If no model is available, ingestion still works and events remain valid —
degraded, not broken.

### An enrichment has its own availability

An enrichment is not available merely because its input was. It carries
`input_available_at`, `computed_at` and its own `ingested_at`, and under
DURABLE_OBSERVED it is visible only when its **own** `durable_available_at <= T`:

```
article available 14:00 · enrichment computed 18:00

DURABLE_OBSERVED snapshot 15:00  →  article YES, enrichment NO
DURABLE_OBSERVED snapshot 19:00  →  article YES, enrichment YES
```

Letting an 18:00 sentiment appear in a 15:00 snapshot would be a lookahead
dressed as a feature.

### Retrospective enrichment is a separate layer

Re-analysing a 2024 archive with a 2026 model is legitimate research, and it is
**not** a live-known feature. It is an explicit `RETROSPECTIVE_ENRICHMENT`
layer; the snapshot identity binds the visibility mode, the enrichment policy
and the model/spec version, so a retrospective feature can never be mistaken
for something the system knew at the time.

The same rule governs entity linking. Deterministic linking performed at
ingestion may be durable; a later relink with a better model is a versioned
enrichment, never a retroactive improvement of an old snapshot.

### Sentiment is entity-conditioned

A single event is positive for USD, negative for BTC and neutral for AAPL. A
scalar "positive/negative headline" is the wrong shape. The record is an
opinion per entity — direction, confidence, horizon hint — and it is a
**derived feature, never ground truth**.

## 9. Market context vs outcome

The same price series serves two roles that must never touch — and, revision 4
adds, market data has a **causal barrier of its own**.

### Two gates, combined with AND

Revisions 1–3 constrained market context by the bar's own time and stopped
there. That is not enough. A bar can be economically old and epistemically
new — corrected, revised or backfilled long after the moment it describes. So:

```
a market row may enter InformationSnapshot(as_of=T) only if

    market event time  <= T          (economic gate)
AND
    the selected revision is available under
    Phase 1C declared-ingested as_of(T)   (ingestion gate)
```

Never OR. A bar whose event time is *after* T stays forbidden even if a
corrupted record claims an earlier ingestion; a bar whose declared
`ingested_at` is after T is invisible even though its bar time precedes T.

### Phase 1C is the market authority

Market context is **consumed** from an exact Phase 1C `MarketSnapshot` taken at
the same `as_of`, not re-derived here:

```
InformationSnapshot(T)
├── event side   — event causal policy (DURABLE_OBSERVED | RETROSPECTIVE_SOURCE)
└── market side  — MarketSnapshot(as_of=T), PHASE1C_DECLARED_INGESTED_ASOF_V1
```

Reading canonical or final bars filtered only by market timestamp is
**forbidden** as a construction of verified causal context. Phase 1C already
selects deterministically over declared `ingested_at` and already fails closed
when two contradictory contents sit at the same maximal ingestion time — it
refuses to arbitrate by comparing hashes, because a hash is not an arbiter
between two contradictory OHLCV declarations.

`MarketSnapshot.as_of` **equals** `InformationSnapshot.as_of`. Not the latest
snapshot, not the nearest, not one taken after T.

### What Phase 1C does and does not prove

Stated plainly, because overselling it here would undo the point. Phase 1C's
`ingested_at` is the **declared** historical ingestion timeline persisted in
the store. It provides replay causality *according to that stored timeline*.
Its own contract says it is "never a live wall-clock, never a promise of
lookahead-free real-time knowledge".

So the market policy is named `PHASE1C_DECLARED_INGESTED_ASOF_V1` and is
**never** labelled `LIVE_OBSERVED`. The event side and the market side are
distinct causal policies, both bound into identity, and the composite claims no
more than the weaker of the two: *events are DURABLE_OBSERVED while market data
follows PHASE1C_DECLARED_INGESTED_ASOF_V1*.

A future provider supplying historical candles does not thereby prove those
candles were observed live at the time; a future market provider must declare
its own timestamp and ingestion semantics.

### Retrospective research does not license final prices

A `RETROSPECTIVE_SOURCE` event snapshot still uses a causal market context.
Retrospective mode is about *source* timestamps for events; it never licenses
"load the final historical prices". If research later needs
`FINAL_HISTORICAL_MARKET_DATA`, that must be a distinct, named, hashed policy —
never conflated with causal context. It is not created now.

### Derived market features

Returns over 5m/1h/24h, volatility, volume, trend and correlation regime may be
computed **only** from rows the bound market causality policy admits at T —
never from a final full-history dataset. And the same distinction the
enrichment layer already makes applies here:

| | |
|---|---|
| **causal market input data** | rows admitted at T |
| **derived market feature availability** | whether the *computed artefact* was itself available at T |

A feature computed later from inputs that were causally available at T is valid
**research reconstruction**. It may not be presented as something HyprL held at
T unless its own durable availability supports that claim.

### Outcome stays on the other side

Outcome — returns at +5m/+30m/+1h/+6h/+1d/+5 sessions, max favourable and
adverse excursion, volatility expansion — lives in **separate artefacts**,
joined only by `event_id` at analysis time. No object a causal query returns has
an outcome field reachable from it.

There are now **two protected directions**, and the new gate completes the old
one rather than replacing it:

| | forbidden in causal context |
|---|---|
| **A** (E9) | market data from *after* the event |
| **B** (E38) | market data from *before* the event but **ingested after** `as_of` |

### Failure and composition

MarketSnapshot corruption, missing required data or an ambiguous revision
selection makes the composite verified replay **fail closed** — no
final-history fallback, in either direction: never "causal events + final market
data", never "causal market + latest events".

The composite does **not** claim a distributed transaction across two stores,
because none exists and pretending otherwise would be the same kind of
overselling this section exists to prevent. Each component is independently
exact and verified under its own causal store at the same declared `as_of`, and
both component identities are frozen into the composite identity. Determinism
comes from identities, not from a cross-database lock.

`LargeMoveDefinitionSpec` (asset, horizon, threshold, direction, and a
volatility-normalised alternative) is deliberately **not defined now**.
Choosing a threshold after looking at recent moves is exactly how a
research question gets selected by its answer. It must be frozen before use,
like every other spec in this repository.

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

Concretely, the mapping a future 6G-A inherits rather than invents:

| Phase 1C primitive | reuse for events |
|---|---|
| append-only receipt rows keyed by logical id | **as-is** — event revisions per `source_item_id` |
| `as_of` cutoff over a declared availability timestamp | **generalised** — the gating column becomes mode-dependent (`durable_available_at` or `source_available_at`) |
| deterministic revision selection, fail-closed on ties | **as-is** |
| immutable snapshot manifests + bounded reads | **generalised** — manifest identity additionally binds `visibility_mode`, the trust policy id/version, and the quality policy |
| `BEGIN IMMEDIATE` before selection, idempotence proven field-by-field | **pattern reused directly** — it is the consistent-read rule (E34) and the lost-acknowledgement rule (E33) |
| `journal_mode=WAL`, `synchronous=FULL` durability contract | **inherited as the meaning of "durable"**, not as a metaphysical guarantee |
| `MarketSnapshot` builder — `as_of` over declared `ingested_at`, deterministic revision selection, `SnapshotSelectionConflict` on ties, immutable identity, bounded read, offline replay | **reusable as-is, and mandatory** — it *is* the market context authority (E38). Required properties: `ingested_at <= as_of`, deterministic selection, ambiguity fail-closed, immutable snapshot identity, offline replay, bounded read. **Forbidden bypass:** reading canonical or final bars filtered only by market timestamp. |
| replay that rebuilds rather than trusts a stored hash | **as-is** |
| `_later_timestamp` conservative combiner | **generalised, not borrowed blindly** — it combines two market bounds; events need a three-way rule over source/observation/ingestion, so the pattern is reused and the helper is not |

Raw payloads are content-addressed files beside the database, exactly as the
Yahoo corpus stores raw beside canonical. A same-URL fetch whose bytes differ
produces a **new** raw artefact, a new content hash and a new revision; the
earlier raw is never overwritten.

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

Every normalized revision **binds the content hash of the raw artefact it came
from**. That makes one case explicit which revision 2 left open: if the
required raw artefact is missing, hash-mismatched or corrupt while the
normalized row is intact, verified replay of any snapshot depending on that
revision **fails closed** — diagnostic `NORMALIZED_PRESENT_RAW_CORRUPT`. A
normalized row that looks fine is not evidence that the bytes behind it were;
the snapshot asserts a provenance that can no longer be verified. Nothing is
re-fetched. If a forensic inspection mode is ever offered it must be named
`NON_VERIFIED` and can never produce a verified snapshot.

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

## 15. Ten cases the design must answer without argument

| # | scenario | DURABLE_OBSERVED | RETROSPECTIVE_SOURCE |
|---|---|---|---|
| C1 | published 14:00, observed 14:00:04, ingested 14:03 — snapshot **14:01** | **ABSENT** | — |
| C2 | same event — snapshot **14:04** | **VISIBLE** | — |
| C3 | article published 2024, backfilled 2027 — snapshot at 2024 | **ABSENT** (`ingested_at` = 2027) | visible **only if** the source timestamp is trusted and meets the quality floor |
| C4 | date-only, timezone `America/New_York` | available at start of the **next** local day, in UTC | same |
| C5 | date-only, timezone unknown | no `source_available_at`; grade E | **inadmissible** to strict causal research |
| C6 | article available 14:00, enrichment computed 18:00 — snapshot **15:00** | article **YES**, enrichment **NO** | enrichment only as `RETROSPECTIVE_ENRICHMENT`, explicitly marked |
| C7 | two clusters merge at T2 — snapshot **T1** | unchanged: still two | unchanged |
| C8 | one cluster splits at T2 — snapshot **T1** | unchanged: still one | unchanged |
| C9 | 15 syndicated copies of one dispatch | `observation_count = 15`, `independent_source_count = 1` (or unknown) — **never 15** | same |
| C10 | `effective_at` precedes publication (July CPI, August release) | visible in **August** | visible in **August** |

Every row is decided by a stated rule, not by interpretation.

And the market-causality cases:

| # | scenario | outcome |
|---|---|---|
| MKT1 | bar open 13:00, declared ingested 18:00, snapshot 14:00 | **ABSENT** — 13:00 ≤ 14:00 but 18:00 > `as_of` |
| MKT2 | bar V1 ingested 13:05, correction V2 ingested 18:00, snapshot 14:00 | **V1** |
| MKT3 | same data, snapshot 19:00 | **V2**, if normal Phase 1C selection admits it |
| MKT4 | bar event time 15:00, ingested 13:00, snapshot 14:00 | **ABSENT** — market event time is in the future |
| MKT5 | backfilled bar, event time years before T, ingested after T | **ABSENT** |
| MKT6 | final canonical history returns a corrected bar, Phase 1C returns an older revision | the snapshot **must** use the Phase 1C result |
| MKT7 | same visible rows under two market causality policies | **different** snapshot identities |
| MKT8 | a derived return uses one row not admitted by the causal MarketSnapshot | **invalid**, fail closed |
| MKT9 | event side verified, MarketSnapshot corrupt | composite verified replay **fails** |
| MKT10 | events DURABLE_OBSERVED + market PHASE1C_DECLARED_INGESTED_ASOF_V1 | valid composite, but it may **not** claim market live-observation semantics the market policy does not prove |

And the transaction-boundary cases, decided the same way:

| # | scenario | outcome |
|---|---|---|
| T1 | pre-transaction timestamp candidate, commit fails | revision **invisible**; the candidate never was an `ingested_at` |
| T2 | failed at 14:03, retry commits 14:07 | durable visibility **≥ 14:07**; never 14:03 |
| T3 | commit succeeds 14:03, crash before ack, retry at 14:07 | **original** revision authoritative, boundary 14:03 preserved, no duplicate |
| T4 | raw durable, normalized parse crashes | raw exists; **no** EventRevision visible |
| T5 | revision committed, required raw later corrupt | verified replay **fails closed** |
| T6 | same visible rows, TrustPolicyV1 vs V2 | **different** snapshot identities |
| T7 | minute-precision interval 14:32 | invisible before the provider-resolved boundary (14:33 if INTERVAL) |
| T8 | hour-precision 14h | invisible before the provider-resolved boundary (15:00 if INTERVAL) |
| T9 | event committed, enrichment persistence fails | event **yes**, enrichment **absent** |
| T10 | MERGE computed, transaction fails | **prior** cluster state remains |
| T11 | two concurrent identical-content ingestions | exactly **one** revision |
| T12 | snapshot read concurrent with new commits | one consistent store view, no mixed state |

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
| **E23** | **Durable visibility.** A revision, enrichment, cluster-state revision or source-health record is invisible in `DURABLE_OBSERVED` until the authoritative transaction persisting that exact immutable object has **committed successfully**; visibility then requires `durable_available_at = max(live_observed_available_at, ingested_at) <= T`. Failed, rolled-back and in-flight transactions have no durable boundary. |
| **E24** | **Backfill honesty.** `HISTORICAL_BACKFILL` observations are never represented as live-observed system knowledge; retrospective visibility requires the explicit `RETROSPECTIVE_SOURCE` mode. |
| **E25** | **Date-only determinism.** A date-only source timestamp resolves to the start of the next local calendar day in the source timezone; an unknown timezone never defaults to UTC and yields no `source_available_at`. |
| **E26** | **Enrichment availability.** No enrichment computed or persisted after T may appear in a `DURABLE_OBSERVED` snapshot at T. |
| **E27** | **Cluster lineage.** A later MERGE, SPLIT or RECLASSIFY never rewrites a prior cluster state or a prior snapshot. |
| **E28** | **Source independence.** Distinct hostnames are never automatically independent sources; without lineage, `independent_source_count` stays unknown. |
| **E29** | **Effective time.** `effective_at` never advances causal visibility under any mode. |
| **E30** | **Replay durability.** A verified `DURABLE_OBSERVED` snapshot is reconstructible only from objects whose authoritative commits succeeded by their durable boundaries **and** whose bound raw/provenance artefacts pass integrity verification. |
| **E32** | **Commit authority.** No pre-commit timestamp, prepared object, in-memory object or failed transaction can satisfy durable visibility. |
| **E33** | **Retry honesty.** A failed attempt never donates its timestamp to a later successful retry. A lost acknowledgement after a successful commit reuses the existing committed revision rather than creating a duplicate or a later boundary. |
| **E34** | **Consistent snapshot read.** An `InformationSnapshot` is built from one consistent bounded store view and cannot combine causal objects from mutually inconsistent commit states. |
| **E35** | **Raw integrity.** A verified snapshot depending on a normalized revision requires the bound raw/provenance hashes to verify; missing or corrupt required raw causes fail-closed replay. |
| **E36** | **Trust policy identity.** Snapshot identity binds the exact timestamp trust policy id and version used. |
| **E37** | **Precision conservatism.** An interval-valued timestamp cannot become causally visible before the conservatively resolved end of its uncertainty interval, per the frozen provider timestamp semantics. |
| **E38** | **Market ingestion causality.** Any market data entering an `InformationSnapshot` must satisfy the frozen market causality policy at the snapshot's `as_of`. A market revision whose declared ingestion availability is later than `as_of` is invisible even when its market event time is earlier. A direct event-time-only market read is forbidden for verified causal context. |
| **E39** | **Market causal policy identity.** Snapshot identity binds the exact market causality policy id/version and the `MarketSnapshot` identity and `as_of` used. Different market causality semantics produce different snapshot identities even when the visible rows happen to be identical. |
| **E40** | **Market derivation causality.** Derived market context may depend only on market inputs admitted by the bound market causality policy; a retrospectively computed market feature is not represented as historically durable unless its own availability supports that claim. |
| **E31** | **Snapshot mode is explicit.** No snapshot may be built without a stated `visibility_mode`; the mode participates in the snapshot identity. |

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
| collector observed then crashed before durable ingest | E23 — the revision is invisible until `ingested_at`; no snapshot claims it earlier |
| replay overclaims pre-ingestion knowledge | E23, E30 — durability is the gate, and replay is bounded by it |
| date-only timezone ambiguity | E25 — next-day-start in the source timezone; unknown timezone yields no availability |
| cluster split retroactivity | E27 — split and merge are lineage operations, prior snapshots unchanged |
| syndication inflation | E28 — hostname count is never independence |
| future-computed enrichment appearing historically | E26 — an enrichment needs its own durable availability |
| pre-commit timestamp used as durability evidence | E32 — only a successful commit carries a boundary |
| failed attempt donating its timestamp to a retry | E33 — a retry gets a new boundary |
| duplicate revision after a lost acknowledgement | E33 — deterministic identity, idempotent rediscovery |
| snapshot mixing pre- and post-commit reads | E34 — one consistent bounded store view |
| normalized row intact but raw corrupt | E35 — verified replay fails closed |
| trust policy swapped without changing identity | E36 — policy id/version is bound |
| interval timestamp treated as an exact instant | E37 — provider semantics decide the conservative boundary |
| late market correction leaking into an earlier snapshot | E38 — the ingestion gate; the snapshot sees the revision selected at T |
| historical market backfill presented as available at an old T | E38 — declared ingestion after `as_of` is invisible |
| final canonical market history bypassing Phase 1C | E38 — an event-time-only read is forbidden for verified context |
| market causality policy changed without changing snapshot identity | E39 — policy id/version is bound |
| derived market feature consuming future or late-ingested input | E40 — derivation may only consume admitted rows |

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
