# Event provider verification — official source contracts

```
STATUS: VERIFICATION ONLY

NO CAPTURE SPEC FROZEN · NO SCRAPER · NO EVENT STORE · NO CORPUS

DESIGN HASH VERIFIED AGAINST
4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689  (revision 4)

VERIFICATION ARTEFACT HASH (revision 3 — authoritative)
e854ddb9cc8b5bc2634fe6ca6fc793289bafd6c94bb49dde71109e1d17341034

supersedes 39c1cdeb… (rev 1) → ed6f322f… (rev 2)

RECOMMENDED FIRST PROVIDER: federal_reserve  (was sec_edgar — REVERSED)
```

Evidence: `docs/artifacts/event_provider_verification_v1.json` — 65 items, each
with an official URL, verdict and confidence: SEC 29, Fed 16, BLS 11, BEA 9.

**Revision 3 rebuilds the recommendation layer only.** The SEC timestamp
findings of revision 2 are preserved unchanged and are not reopened. What changed
is the *selection*: independent counter-review found the ranking rested on an
argument the protocol forbids, on an aesthetic penalty against the Fed, and on
one overstated SEC property. Corrected, the ranking reverses. See §4.

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

The earlier UNKNOWN was an *environment limitation*, and revision 3 states it
precisely: the **WebFetch tool** fails DNS/timeout for `data.sec.gov` and
`www.sec.gov` in this environment, while narrow official-only `curl` GETs
succeed with HTTP 200. That is a tool limitation and implies **no provider
outage**.

### What it actually means

SEC's own field definition (Financial Statement and Notes Data Sets):

> **accepted** — "The **acceptance** date and time of the registrant's filing
> with the Commission. Filings accepted after 5:30pm EST are considered filed on
> the following business day."

**Provenance guard (revision 3).** That definition documents the *Financial
Statement Data Sets* field, a **different SEC product** from the Submissions API,
so it is no longer relied on as sole proof for
`filings.recent.acceptanceDateTime`; the selected-surface semantics are recorded
conservatively as PARTIALLY_VERIFIED. The conclusion does not depend on it. EDGAR
operational documentation separates acceptance from dissemination directly:

> "EDGAR accepts new filer applications, new filings, and changes to filer data
> each business day, Monday through Friday, from 6:00 a.m. to 10:00 p.m., ET…
> Some filing submissions that begin after **5:30 p.m. ET** — or **10:00 p.m. for
> Ownership forms 3, 4, 5** — will be **disseminated the next business day**,
> showing up in the following business day's index."

Note the exact scope: *some* submissions, keyed to when transmission **begins**,
with a **form-conditional** cutoff. It is not generalised further.

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

**Surface scope (revision 3):** `prevrpt` is **not** a `filings.recent` field —
the Submissions inventory above does not contain it. It belongs to the Financial
Statement Data Sets, which is where the prohibition applies. Recorded to prevent
cross-surface schema contamination: the trap is real, on that dataset.

## 2. Provider summaries

### SEC EDGAR — runner-up, strongest identity and archive

| property | verdict |
|---|---|
| authentication | **none** — "These APIs do not require any authentication or API keys" |
| latency | submissions API "typical processing delay of **less than a second**" — surface update guidance, **not an SLA and not a publication timestamp** |
| rate policy | **10 requests/second**, declared User-Agent `Company Name AdminContact@domain.com` |
| identity | **accession number** `##########-YY-SSSSSS`, assigned automatically by EDGAR on acceptance |
| revisions | amendments (10-K/A, 10-Q/A, 8-K/A) are **separate filings with their own accession numbers**, so amendment lineage is additive — **but the public surface is not immutable**, see below |
| discovery | **entity/CIK-scoped**, not a global feed: `data.sec.gov/submissions/CIK##########.json`; older history paginated via `filings.files[]` |
| entity mapping | official `company_tickers.json`, 10 415 entries — **B5 resolved** |
| archive | `companyfacts.zip` / `submissions.zip` "recompiled nightly"; `/Archives` permanent |
| raw capture | static JSON and documents, no JS rendering |

**Timestamp:** see §1 — precise but acceptance-only; no verified dissemination
instant; live capture relies on collector `observed_at`.

**The surface is not immutable.** SEC documents that *"filings are sometimes
authorized by SEC staff for removal or correction … removals processed on
subsequent business days will not be reflected in any previous daily, feed, or
oldload index."* Revision 2 called SEC *"append-only"* and each accession an
*"immutable observation"*; that was an overclaim. The precise position is:
accession gives deterministic filing identity, amendments are separate filings,
**but a provider implementation must preserve each observation/retrieval state
rather than assume the public surface itself is immutable.** HyprL's own
raw-first history stays append-only regardless — a later divergent or absent
fetch is a *new* observation state, never an overwrite.

This is not a mark against SEC. Documented correction behaviour is *better*
evidence than the Fed's UNKNOWN. It is simply not immutability.

**Entity semantics are not uniform.** QQQ (CIK 1067839) has `entityType`
`"investment"` with empty SIC, category and state of incorporation, and files
497 / 485BPOS / NPORT-P / 24F-2NT / N-30B-2 — investment-company forms, not the
8-K/10-K population of an operating issuer. A mixed equity+ETF universe is not
one filer semantics.

**Unknowns:** redistribution of filer-submitted content (filings are third-party
submissions and may embed filer-owned or third-party copyrighted exhibits — no
blanket public-domain assumption is made), and the field linking an amendment to
its original.

### Federal Reserve — recommended first provider

The primary document itself carries **"For release at 2:00 p.m. EDT"** — minute
precision *with* an explicit timezone. Separately, the page carries
`Last Update: <date>`, date-only, which under the design's §7 rule is site
maintenance metadata and must never be read as publication time. The FOMC
*calendar* page states no times at all; the time lives on each statement.

That release line is the **authoritative source-time evidence**, verified on two
independent historical statements (2026-06-17, 2026-07-29). Under
`TimestampTrustPolicyV1` it satisfies `trusted_requires` — TIER_1_OFFICIAL,
precision ≤ minute, known timezone — so **`source_available_at` can actually be
populated.**

Separately and *only as corroboration*: the official monetary-policy feed carries
an RFC-822 `pubDate` with explicit timezone, and on all four FOMC statements in
the feed it reads `18:00:00 GMT` = 14:00 EDT, matching the documents. It is not a
constant — another item in the same feed carries `19:00:00 GMT`. But the feeds
page documents no `pubDate` semantics and every sampled item falls inside EDT, so
**DST behaviour is unverified and `pubDate` is not claimed as publication
authority.** The feed serves discovery; the document serves the timestamp.

**Identity — corrected.** Every item carries a declared `<guid>` (15/15
inspected), valued as the canonical document URL under the stable dated pattern
`monetary{YYYYMMDD}{a|b}{n}`. Revision 2 scored this down for not being *"an
agency-assigned opaque id."* That penalty is **withdrawn as aesthetic**: the
design requires deterministic identity, not opacity, and an official
feed-declared GUID is deterministic. Re-scored **GOOD**.

**Discovery is one fetch.** `/feeds/press_monetary.xml` — one officially declared
~9.6 KB topic feed, listed among 31 on the feeds page — enumerates items with
title, link, guid, category and pubDate. No entity index, no per-issuer fan-out,
no credential.

**Unknowns:** correction/revision behaviour (**F1**); automated-access and rate
contract (**F2**) — `/robots.txt` returns the site's "Page not Found" document,
and *absence of a declared restriction is not permission*; and the release-line
parsing and EST/EDT rule (**F4**).

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

Scale: EXCELLENT · GOOD · ACCEPTABLE · WEAK · UNKNOWN · BLOCKING.
`implementation complexity` is scored as **simplicity** (EXCELLENT = simplest).

| | SEC | Fed | BLS | BEA |
|---|---|---|---|---|
| timestamp quality | **WEAK** | **EXCELLENT** | GOOD | ACCEPTABLE |
| identity quality | **EXCELLENT** | GOOD ↑ | GOOD | UNKNOWN |
| discovery quality ★ | ACCEPTABLE | **EXCELLENT** | ACCEPTABLE | ACCEPTABLE |
| revision model | GOOD ↓ | UNKNOWN | WEAK | GOOD |
| historical archive | **EXCELLENT** | GOOD | GOOD | ACCEPTABLE |
| live capture | **EXCELLENT** | GOOD | ACCEPTABLE | ACCEPTABLE |
| raw preservation | EXCELLENT | EXCELLENT | EXCELLENT | EXCELLENT |
| credential friction | EXCELLENT | EXCELLENT | GOOD | ACCEPTABLE |
| rate/access clarity | **EXCELLENT** | UNKNOWN | GOOD | EXCELLENT |
| redistribution clarity | UNKNOWN | ACCEPTABLE | UNKNOWN | UNKNOWN |
| causal backfill | **WEAK** | GOOD | ACCEPTABLE | WEAK |
| implementation simplicity | ACCEPTABLE | **EXCELLENT** | ACCEPTABLE | ACCEPTABLE |
| market relevance | GOOD | **EXCELLENT** | **EXCELLENT** | GOOD |
| **overall** | SUITABLE_FOR_6G_A — *second* | **SUITABLE_FOR_6G_A — first** | SUITABLE_LATER | SUITABLE_LATER |

↑ raised on evidence · ↓ downgraded on evidence

★ **`discovery quality` is a new operational supporting axis**, added because
CaptureSpec feasibility depends materially on it. It is **not** inserted into the
frozen ten-criterion priority ordering below; it informs implementation
simplicity, live-capture feasibility and freeze readiness.

## 4. Recommendation

**`federal_reserve`**, confidence **MEDIUM**, runner-up `sec_edgar`.
Verdict: **FED_PREFERRED** — a reversal of revision 2.

### What the counter-review found

Revision 2 recommended SEC on three supports, and independent review removed or
weakened all three.

1. **An argument the protocol forbids.** Revision 2 argued SEC was preferable
   because *"with no dissemination instant, SEC forces 6G-A to exercise the
   `source_available_at = null` path."* Ranking a provider by how many edge
   branches it exercises is not a selection criterion. It is now classified
   **SECONDARY_SYNTHETIC_TEST_COVERAGE_BENEFIT** with **ZERO ranking weight** —
   those paths are testable with deterministic synthetic fixtures, and in any
   case `TimestampTrustPolicyV1` has four verdicts that *no* single real provider
   exercises, so branch coverage never comes from provider choice.
2. **An aesthetic penalty.** The Fed was marked down for lacking "an
   agency-assigned opaque id" while in fact exposing an official feed-declared
   `<guid>` on every inspected item. Determinism is the requirement; opacity is
   not. Re-scored GOOD.
3. **An overstated SEC property.** "Append-only", "immutable observation" — SEC
   documents post-acceptance correction and removal. Re-scored GOOD.

### The ordered decision

The original infrastructure-first ordering, preserved and unrenumbered:

| # | criterion | SEC | Fed | winner |
|---|---|---|---|---|
| 1 | source authority | TIER_1 primary | TIER_1 primary | — |
| 2 | **timestamp quality** | **WEAK** | **EXCELLENT** | **FED** |
| 3 | deterministic identity | EXCELLENT | GOOD | SEC |
| 4 | revision semantics | GOOD (documented) | UNKNOWN | SEC |
| 5 | raw capture simplicity | EXCELLENT | EXCELLENT | — |
| 6 | legal clarity | third-party filer content | agency-authored | **FED** |
| 7 | credential friction | none | none | — |
| 8 | historical fixtures | EXCELLENT | GOOD | SEC |
| 9 | relevance | GOOD | EXCELLENT | FED |
| 10 | implementation simplicity | ACCEPTABLE | EXCELLENT | **FED** |

These are **priorities, not a lexicographic algorithm**: winning criterion 2 does
not automatically end the argument, so SEC's later advantages are weighed for
materiality rather than discarded.

**Criterion 1 ties, so the decision falls to criterion 2 — and there the gap is
categorical, not marginal.** The Fed publishes an explicit source release instant
on the primary document. SEC has no verified per-filing publication instant at
all.

That difference is not cosmetic, and this is the part that is *not* a
test-coverage argument: **`RETROSPECTIVE_SOURCE` is one of the two mandatory
snapshot modes**, and its visibility rule is `source_available_at <= T`. A
SEC-first 6G-A would leave that mode without real source evidence — null live,
date-only (grade D) in backfill. The Fed populates both modes.

| mode | SEC | Fed |
|---|---|---|
| `DURABLE_OBSERVED` | fully supported | fully supported |
| `RETROSPECTIVE_SOURCE` | **WEAK** — gate unpopulated | **SUPPORTED** — declared release instant |

The Fed also wins legal clarity (agency-authored, no third-party filer-rights
layer), relevance, and implementation simplicity by a wide margin — one declared
feed plus one static page against an entity index, per-CIK pagination and
heterogeneous issuer semantics.

**SEC's three wins are real, and none is categorical.** Identity is EXCELLENT
against a GOOD that is still fully deterministic. Revision semantics are
documented — but richer and *more complex*, which is a burden for a first
integration, not a benefit. The archive advantage is volume, and 6G-A needs
determinism, not volume, to prove ingestion and replay.

Confidence is **MEDIUM, not HIGH** — about remaining unknowns, not about the
ranking: F1 and F2 must be resolved before anything is frozen.

### SEC remains the right second provider

Its research is retained in full. It contributes the strongest deterministic
identity, the largest offline fixture corpus, a documented access contract, and
— as a *consequence* of integrating it, never as a reason for choosing it — the
real-world `source_available_at = null` path.

### Proposed narrow Fed V1 — *proposal only, not a frozen capture spec*

```
scope                 FOMC monetary-policy statements ONLY — not all Fed content
discovery             official monetary-policy feed (endpoint not frozen)
fetch                 official primary FOMC statement page
identity              official feed-declared <guid> / deterministic canonical identity
source_available_at   the release instant declared on the primary statement
                      — never a schedule assumption, never pubDate alone
observed_at           collector acquisition, independent of the declared instant
ingested_at           EventStore successful-commit boundary (design rev4)
raw                   feed bytes + primary page bytes, locally preserved
```

### Causal grades

| provider | live | backfill |
|---|---|---|
| SEC | **E on the source-timestamp axis**; safe in practice via collector `observed_at` | **D** |
| Fed | **A** — official/primary, LIVE, declared release instant at minute precision | **B** |
| BLS | A on the release surface | B via dated archive; **D/E** for the series API alone |
| BEA | B | C |

No provider reaches grade A on backfill: a source timestamp recovered later can
establish *publication* time, never our historical *observation* time.

### The consensus gap

**No official market-consensus product was verified on the evaluated official
surfaces.** Official statistical agencies publish the actual, the previous and
revisions — not what the market expected. Consensus will require a separate,
independently verified provider before any surprise metric can be computed. This
does **not** block 6G-A, whose objective is ingestion and replay, not surprise.

## 5. Contract fit against the frozen design

| design element | Fed fit (selected) | SEC fit (second) |
|---|---|---|
| `TimestampTrustPolicyV1` | release line satisfies `trusted_requires` — TIER_1_OFFICIAL, ≤ minute, known tz | `acceptanceDateTime` cannot earn public-availability trust: it is not a publication instant |
| SourceObservation identity | official feed-declared `<guid>` — deterministic, never URL-alone-by-accident | accession number — agency-assigned |
| Revision identity | **UNKNOWN** (F1) — detectable by raw hashing, but the policy is unverified | separate accessions, plus documented upstream correction/removal |
| raw-first | static feed + page bytes | static JSON and documents |
| `DURABLE_OBSERVED` | supported — `observed_at` + commit-bound `ingested_at` | supported — same |
| `RETROSPECTIVE_SOURCE` | **supported** — declared release instant populates the gate | **weak** — gate null live, date-only in backfill |
| redistribution | agency-authored; no reuse-terms page verified → conservative restricted | **UNKNOWN** — filer-submitted third-party content |
| offline replay | static archived statements | bulk archives make deterministic fixtures straightforward |

**No design change is required by either provider.** `design_change_required =
false`; design rev4 remains at
`4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689`.

## 6. Blockers

### Federal Reserve — selected provider

| | status | effect on a capture spec |
|---|---|---|
| **F1** revision/correction semantics | **UNKNOWN** | **STILL_BLOCKING_CAPTURE_SPEC** — V1 change-detection policy undefined |
| **F2** automated-access / rate contract | **UNKNOWN** — no `robots.txt` served, no ceiling located | **STILL_BLOCKING_CAPTURE_SPEC** — a conservative self-imposed cadence must be chosen and recorded |
| **F3** RSS `pubDate` semantics | not formally documented | **NON_BLOCKING** — not needed: the primary statement carries its own release line, so the feed serves discovery while `source_available_at` comes from the document |
| **F4** release-line parsing / EST-EDT resolution | unspecified; all sampled items fall inside EDT | **STILL_BLOCKING_CAPTURE_SPEC** |
| **F5** pre-2021 archive depth | unverified | **NOT_REQUIRED_FOR_MVP** |

### SEC EDGAR — second provider

| | status | effect on a capture spec |
|---|---|---|
| **B1a** `acceptanceDateTime` format/precision/timezone | **RESOLVED** — ISO-8601, UTC, millisecond | none |
| **B1b** per-filing dissemination timestamp | **NOT_EXPOSED_ON_SELECTED_SURFACE** | **NON_BLOCKING_WITH_CONSERVATIVE_POLICY** — rev4 permits `source_available_at = null`; it does cap the causal grade |
| **B2** redistribution of filer-submitted raw content | **UNKNOWN** | **NON_BLOCKING_WITH_CONSERVATIVE_POLICY** — raw stays `LOCAL_RESTRICTED`, `redistribution_permitted=false`, release guard applies |
| **B3** amendment parent pointer | **UNKNOWN** | **NOT_REQUIRED_FOR_MVP** — the spec must declare `amendment_parent_lineage_available=false` rather than invent one |
| **B4** discovery mechanism / polling scope | **undecided** | **STILL_BLOCKING_CAPTURE_SPEC** — a future spec must choose among official mechanisms (known-CIK polling, an official recent-filings feed/index, or another verified surface). This does *not* make SEC unsuitable. |
| **B5** ticker/CIK entity mapping | **RESOLVED** — official `company_tickers.json` | none |
| **B6** post-acceptance correction/removal handling | **documented upstream, unspecified in V1** | **STILL_BLOCKING_CAPTURE_SPEC** |

### Proposed safe SEC V1 semantics — *proposal only, retained for 6G-B*

```
identity              accessionNumber
source_available_at   null  (unless a verified dissemination timestamp appears)
acceptance_at         acceptanceDateTime — provenance only, NOT an availability gate
filing_date           date metadata only, NOT an availability gate
observed_at           collector's first successful acquisition of the public resource
ingested_at           EventStore successful-commit boundary (design rev4)
amendment_parent      unavailable — not claimed
prevrpt               retrospective only — forbidden in causal records and features
                      (and NOT a filings.recent field: it belongs to the
                       Financial Statement Data Sets, a different SEC product)
discovery             UNDECIDED — blocker B4
upstream change       UNDECIDED — blocker B6
raw                   locally preserved, LOCAL_RESTRICTED, redistribution false by default
```

Rights are kept in four distinct categories — SEC-produced API metadata,
filer-submitted filing content, HyprL normalized metadata, and derived aggregate
outputs — with no blanket verdict across EDGAR.

### Capture spec readiness

**`ready_for_capture_spec_freeze = false`**, for the Fed as much as for SEC.
Provider selection and freeze readiness are separate gates, and it is correct for
`provider_selected_for_6ga = true` to hold while freeze remains false. What the
Fed still lacks: an access/rate policy (F2), revision/change-detection semantics
(F1), a release-line parsing and DST rule (F4), a chosen poll cadence, and a
frozen host allowlist and endpoint.

## 7. What this does not establish

Provider suitability only. No predictive edge, no causal market impact, no
economic significance, no useful sentiment or surprise.

`commercial_edge_established = false`.
