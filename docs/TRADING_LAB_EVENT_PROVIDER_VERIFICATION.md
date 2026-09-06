# Event provider verification — official source contracts

```
STATUS: VERIFICATION ONLY

NO CAPTURE SPEC FROZEN · NO SCRAPER · NO EVENT STORE · NO CORPUS

DESIGN HASH VERIFIED AGAINST
4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689  (revision 4)

VERIFICATION ARTEFACT HASH (revision 2 — authoritative)
ed6f322fee5845da9ffaf80b6724df56c92bf233df587e74333346f4a7a13b82

supersedes 39c1cdeb… (rev 1)
```

Evidence: `docs/artifacts/event_provider_verification_v1.json` — 53 items, each
with an official URL, verdict and confidence.

**Method.** Only primary official domains count as contractual evidence:
`sec.gov`, `federalreserve.gov`, `bls.gov`, `bea.gov`, `apps.bea.gov`. Search
was used only to *locate* official pages. No credential was used, no
registration performed, no bulk dataset downloaded. Where a property is not
officially documented it is recorded **UNKNOWN** — never converted to false.

---

## 1. SEC timestamps — what is true, and what revision 1 got wrong

Revision 1 claimed SEC could reach live causal grade **A** "subject to confirming
acceptanceDateTime precision". Independent counter-review showed that framing was
materially wrong: **precision was never the obstacle.**

### What is now established

`acceptanceDateTime` exists on the Submissions surface and is precise:

```
format     YYYY-MM-DDTHH:MM:SS.sssZ
precision  millisecond
timezone   UTC (trailing Z)
```

The earlier UNKNOWN was an *environment limitation* — `data.sec.gov` was
unreachable from the fetch tool, not unavailable as a service.

### What it actually means

SEC's own field definition (Financial Statement and Notes Data Sets):

> **accepted** — "The **acceptance** date and time of the registrant's filing
> with the Commission. Filings accepted after 5:30pm EST are considered filed on
> the following business day."

It marks **acceptance by the Commission**, not public dissemination. And SEC's
filing-status page states that submissions after 5:30 p.m. ET "will **not be
disseminated** by EDGAR until the next business day".

So for post-cutoff filings, **acceptance precedes public availability by
hours**. Populating `source_available_at` from `acceptanceDateTime` would make
those events visible before the public could see them — a lookahead in the exact
direction the design forbids.

**No per-filing public-dissemination timestamp was verified on the Submissions
surface.** Its field inventory is `acceptanceDateTime, accessionNumber, act,
core_type, fileNumber, filingDate, filmNumber, form, isInlineXBRL, isXBRL,
isXBRLNumeric, items, primaryDocDescription, primaryDocument, reportDate, size`.
That is a statement about *this* surface, not about every SEC system.

### The three cautions that replace it

1. `filingDate` is an **assigned business date**, not an exact publication
   instant, and can differ from the acceptance date.
2. `acceptanceDateTime` is precise but is **not proven equivalent to
   dissemination**.
3. `source_available_at` must therefore not be populated from either as an exact
   instant. When no dissemination timestamp exists, the safe causal boundary is
   the collector's `observed_at`.

No derived "06:00 ET next business day" timestamp is adopted. That bound appears
in an EDGAR rule, but its exact domain is not established and revision 1 of a
verification is not the place to invent one.

### Why SEC remains safe despite this

A collector **cannot observe a filing before EDGAR disseminates it**. So
`observed_at` naturally prevents the lookahead, and `DURABLE_OBSERVED` gates on
`max(observed_at, ingested_at)` — which design rev4 explicitly supports when
`source_available_at` is absent (`"source_available_at = null; only observed_at
can bound"`). **No design change is required.**

### Corrected causal grades

| | revision 1 | revision 2 |
|---|---|---|
| live, source-timestamp axis | A (conditional) | **E** — no verified publication instant |
| live, effective | — | **SAFE** via `observed_at` + commit-bound `ingested_at`, but not a source-declared instant, so not A |
| backfill | B | **D** — no historical `observed_at`, acceptance must not substitute for dissemination, only date-only `filingDate` remains |

### `prevrpt` — a causal trap worth naming

The same SEC dataset documents `prevrpt`: "TRUE indicates that the submission
information was **subsequently amended** prior to the end cutoff date of the data
set." Its value depends on events *after* the filing and on the dataset cutoff.
It is future-aware by construction, it is **not** an amendment parent pointer,
and it must never enter a causal record or feature at the filing's historical
timestamp.

## 2. Provider summaries

### SEC EDGAR — recommended

| property | verdict |
|---|---|
| authentication | **none** — "These APIs do not require any authentication or API keys" |
| latency | submissions API "typical processing delay of **less than a second**" — surface update guidance, **not an SLA and not a publication timestamp** |
| rate policy | **10 requests/second**, declared User-Agent `Company Name AdminContact@domain.com` |
| identity | **accession number** `##########-YY-SSSSSS`, assigned automatically by EDGAR on acceptance |
| revisions | amendments (10-K/A, 10-Q/A, 8-K/A) are **separate filings with their own accession numbers** — append-only by construction |
| archive | `companyfacts.zip` / `submissions.zip` "recompiled nightly"; `/Archives` permanent |
| raw capture | static JSON and documents, no JS rendering |

**Timestamp:** see §1 — precise but acceptance-only; no verified dissemination
instant; live capture relies on collector `observed_at`.

**Unknowns:** redistribution of filer-submitted content (filings are third-party
submissions and may embed filer-owned or third-party copyrighted exhibits — no
blanket public-domain assumption is made), and the field linking an amendment to
its original.

### Federal Reserve — runner-up, strongest timestamp

The primary document itself carries **"For release at 2:00 p.m. EDT"** — minute
precision *with* an explicit timezone. Separately, the page carries
`Last Update: <date>`, date-only, which under the design's §7 rule is site
maintenance metadata and must never be read as publication time. The FOMC
*calendar* page states no times at all; the time lives on each statement.

Official RSS feeds exist (`/feeds/press_monetary.xml`), and minutes follow a
stated rule — "released three weeks after the date of the policy decision".

**Unknowns:** feed update cadence and item timestamp semantics, correction
behaviour, and any rate/fair-access policy. Identity is a date-based URL
convention (`monetary{YYYYMMDD}{a|b}{n}`), stable but not an agency-assigned
opaque id.

### BLS — richest macro relevance, heavier first integration

The **news release document** states "Transmission of material in this release
is embargoed until **8:30 a.m. (ET)**" — time with timezone. The **schedule
page** states `Release Time: 08:30 AM` with **no timezone**, so the release
document, not the schedule, is the causal surface. Releases carry a `USDL-YY-NNNN`
number and are preserved at dated archive paths
(`/news.release/archives/cpi_08122025.pdf`), which is what makes originally
published values recoverable.

API: v1.0 keyless (25 queries/day), v2.0 registered (500/day, 50 series, 20 years).

**Unknowns — and they are the important ones:** the FAQ documents no per-value
publication timestamp, no historical vintage exposure, and no
preliminary/revised marker. A value fetched today from the API is not
necessarily the value published then; the causal surface is therefore the dated
release archive, not the series API.

### BEA — clean limits, weakest causal surface

Registration required (36-character UserID on every request). Limits documented
precisely: **100 requests/min, 100 MB/min, 30 errors/min**. JSON or XML.

GDP **Advance / Second / Third Estimate** appear as separately scheduled,
separately titled events — naturally append-only rather than a mutated value.

**Unknowns:** the schedule lists `8:30 AM` / `10:00 AM` with **no timezone**;
no verified release identity; vintage exposure undocumented.

## 3. Scorecard

| | SEC | Fed | BLS | BEA |
|---|---|---|---|---|
| timestamp quality | **WEAK** | **EXCELLENT** | GOOD | ACCEPTABLE |
| identity quality | **EXCELLENT** | ACCEPTABLE | GOOD | UNKNOWN |
| revision model | **EXCELLENT** | UNKNOWN | WEAK | GOOD |
| historical archive | **EXCELLENT** | GOOD | GOOD | ACCEPTABLE |
| live capture | **EXCELLENT** | GOOD | ACCEPTABLE | ACCEPTABLE |
| raw preservation | EXCELLENT | EXCELLENT | EXCELLENT | EXCELLENT |
| credential friction | **EXCELLENT** | EXCELLENT | GOOD | ACCEPTABLE |
| rate-limit clarity | **EXCELLENT** | UNKNOWN | GOOD | EXCELLENT |
| redistribution clarity | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| causal backfill | **WEAK** | GOOD | ACCEPTABLE | WEAK |
| implementation complexity | LOW | LOW | MEDIUM | MEDIUM |
| market relevance | GOOD | **EXCELLENT** | **EXCELLENT** | GOOD |
| **overall** | **SUITABLE_FOR_6G_A** | SUITABLE_FOR_6G_A | SUITABLE_LATER | SUITABLE_LATER |

## 4. Recommendation

**`sec_edgar`**, confidence **MEDIUM** — lowered from MEDIUM_HIGH, because the
earlier recommendation partly rested on a timestamp claim that did not survive
review.

SEC still wins, but **explicitly not on its source timestamp — it has none that
is verified.** It wins on the axes 6G-A actually has to prove: an agency-assigned
deterministic identity rather than a URL convention; append-only submissions
where each accession is an independent immutable observation; a documented access
contract (10 req/s, declared User-Agent); no credentials; nightly bulk archives
for deterministic offline fixtures; and enough event volume to exercise the
store.

There is also a positive reason its weakness helps. With no dissemination
instant, SEC **forces 6G-A to exercise the `source_available_at = null` path**
with `observed_at` bounding — a path design rev4 explicitly supports and which
must be proven to work. A provider with a perfect source timestamp would leave
that path untested.

**The Fed has the better timestamp by a clear margin** and is now strictly ahead
of SEC on that axis. It is the right *second* provider, precisely because it
exercises the source-declared-instant path. It loses first place on identity
(date-based URL convention), on UNKNOWN revision and correction semantics, on an
UNKNOWN rate policy, and on volume — roughly eight FOMC decisions a year is thin
for shaking out an EventStore.

Verdict versus revision 1: **CONFIRMED_WITH_LOWER_CONFIDENCE.**

### Causal grades

| provider | max live | max backfill |
|---|---|---|
| SEC | **E on the source axis**; safe in practice via collector `observed_at` | **D** |
| Fed | A | B |
| BLS | A on the release surface | B via dated archive; **D/E** for the series API alone |
| BEA | B | C |

No provider reaches grade A on backfill: a source timestamp recovered later can
establish *publication* time, never our historical *observation* time.

### The consensus gap

**No official market-consensus product was found at any of the four agencies.**
Official statistical agencies publish the actual, the previous and revisions —
not what the market expected. Consensus will require a separate, independently
verified provider before any surprise metric can be computed.

This does **not** block 6G-A, whose objective is ingestion and replay, not
surprise.

## 5. Contract fit against the frozen design

| design element | SEC fit |
|---|---|
| `TimestampTrustPolicyV1` | testable: official/primary tier, but the precision and semantics of `acceptanceDateTime` must be established before a verdict can be assigned |
| SourceObservation identity | accession number — satisfies "never URL alone" |
| Revision identity | amendments are separate accessions — append-only |
| raw-first | static bytes, trivially preservable |
| `DURABLE_OBSERVED` | supported: our collector supplies `observed_at` and `ingested_at`; SEC supplies `source_available_at` |
| `RETROSPECTIVE_SOURCE` | supported via `/Archives` + bulk zips, at backfill grade |
| redistribution | **UNKNOWN** — must be verified before any corpus is published |
| offline replay | bulk archives make deterministic fixtures straightforward |

## 6. Blockers, reclassified

| | status | effect on a capture spec |
|---|---|---|
| **B1a** `acceptanceDateTime` format/precision/timezone | **RESOLVED** — ISO-8601, UTC, millisecond | none |
| **B1b** per-filing public dissemination timestamp | **NOT_EXPOSED_ON_SELECTED_SURFACE** | **NON_BLOCKING_IF_CONSERVATIVE_POLICY_USED** — rev4 permits `source_available_at = null`; it does cap the causal grade |
| **B2** redistribution of filer-submitted raw content | **UNKNOWN** | **NON_BLOCKING_IF_CONSERVATIVE_POLICY_USED** — raw stays `LOCAL_RESTRICTED`, `redistribution_permitted=false`, release guard applies |
| **B3** amendment parent pointer | **UNKNOWN** | **NOT_REQUIRED_FOR_6G_A_MVP** — each accession is an independent immutable observation; the spec must declare `amendment_parent_lineage_available=false` rather than invent one |

### Proposed safe SEC V1 semantics — *proposal only, not a frozen capture spec*

```
identity              accessionNumber
source_available_at   null  (unless a verified dissemination timestamp appears)
acceptance_at         acceptanceDateTime — provenance only, NOT an availability gate
filing_date           date metadata only, NOT an availability gate
observed_at           collector's first successful acquisition of the public resource
ingested_at           EventStore successful-commit boundary (design rev4)
amendment_parent      unavailable — not claimed
prevrpt               retrospective only — forbidden in causal records and features
raw                   locally preserved, LOCAL_RESTRICTED, redistribution false by default
```

Rights are kept in four distinct categories — SEC-produced API metadata,
filer-submitted filing content, HyprL normalized metadata, and derived aggregate
outputs — with no blanket verdict across EDGAR.

## 7. What this does not establish

Provider suitability only. No predictive edge, no causal market impact, no
economic significance, no useful sentiment or surprise.

`commercial_edge_established = false`.
