# Event provider verification — official source contracts

```
STATUS: VERIFICATION ONLY

NO CAPTURE SPEC FROZEN · NO SCRAPER · NO EVENT STORE · NO CORPUS

DESIGN HASH VERIFIED AGAINST
4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689  (revision 4)

VERIFICATION ARTEFACT HASH
39c1cdeb47ea69426bece2876f770484c03ebd202c0ecab17f0548d370339541
```

Evidence: `docs/artifacts/event_provider_verification_v1.json` — 42 items, each
with an official URL, verdict and confidence.

**Method.** Only primary official domains count as contractual evidence:
`sec.gov`, `federalreserve.gov`, `bls.gov`, `bea.gov`, `apps.bea.gov`. Search
was used only to *locate* official pages. No credential was used, no
registration performed, no bulk dataset downloaded. Where a property is not
officially documented it is recorded **UNKNOWN** — never converted to false.

---

## 1. The finding that matters most

**SEC officially documents the exact distinction the design froze.**

> "Most live submissions that begin transmission after 5:30 p.m. ET … will
> receive a filing date of **6:00 a.m. ET the next business day** and will not
> be **disseminated** by EDGAR until the next business day."

So `filingDate` is an *assigned business date*, not a publication instant — a
filing transmitted at 18:00 on Monday carries Tuesday's filing date and becomes
public on Tuesday. Using `filingDate` as `source_available_at` would place an
event up to a day before it existed publicly.

This is the first candidate whose own contract forces the three-way separation
the design spent four revisions building. It exercises `source_available_at` on
a real, documented rule instead of a hypothetical one.

## 2. Provider summaries

### SEC EDGAR — recommended

| property | verdict |
|---|---|
| authentication | **none** — "These APIs do not require any authentication or API keys" |
| latency | submissions API "typical processing delay of **less than a second**"; JSON updated "in real time, as submissions are disseminated" |
| rate policy | **10 requests/second**, declared User-Agent `Company Name AdminContact@domain.com` |
| identity | **accession number** `##########-YY-SSSSSS`, assigned automatically by EDGAR on acceptance |
| revisions | amendments (10-K/A, 10-Q/A, 8-K/A) are **separate filings with their own accession numbers** — append-only by construction |
| archive | `companyfacts.zip` / `submissions.zip` "recompiled nightly"; `/Archives` permanent |
| raw capture | static JSON and documents, no JS rendering |

**Unknowns:** the literal format, precision and timezone of `acceptanceDateTime`
could not be confirmed — `data.sec.gov` was unreachable from this environment.
Redistribution terms were not verified from an official terms page, and the
field linking an amendment to its original was not confirmed.

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
| timestamp quality | GOOD | **EXCELLENT** | GOOD | ACCEPTABLE |
| identity quality | **EXCELLENT** | ACCEPTABLE | GOOD | UNKNOWN |
| revision model | **EXCELLENT** | UNKNOWN | WEAK | GOOD |
| historical archive | **EXCELLENT** | GOOD | GOOD | ACCEPTABLE |
| live capture | **EXCELLENT** | GOOD | ACCEPTABLE | ACCEPTABLE |
| raw preservation | EXCELLENT | EXCELLENT | EXCELLENT | EXCELLENT |
| credential friction | **EXCELLENT** | EXCELLENT | GOOD | ACCEPTABLE |
| rate-limit clarity | **EXCELLENT** | UNKNOWN | GOOD | EXCELLENT |
| redistribution clarity | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| causal backfill | GOOD | GOOD | ACCEPTABLE | WEAK |
| implementation complexity | LOW | LOW | MEDIUM | MEDIUM |
| market relevance | GOOD | **EXCELLENT** | **EXCELLENT** | GOOD |
| **overall** | **SUITABLE_FOR_6G_A** | SUITABLE_FOR_6G_A | SUITABLE_LATER | SUITABLE_LATER |

## 4. Recommendation

**`sec_edgar`**, on infrastructure grounds, not predictive usefulness — which is
the ordering §17 of the mandate requires and the whole point of 6G-A.

It is the only candidate that combines no credentials, an agency-*assigned*
deterministic identity, append-only amendment semantics, a documented
sub-second dissemination latency, an explicit rate and User-Agent policy, and
nightly bulk archives for deterministic offline fixtures. And its documented
transmission / filing-date / dissemination split is precisely the distinction
the design exists to enforce.

The Fed is a close runner-up and has the better timestamp; it loses on
deterministic identity and on three UNKNOWNs (feed semantics, corrections, rate
policy). If the acceptanceDateTime unknown below cannot be resolved, the Fed is
the fallback.

### Causal grades

| provider | max live | max backfill |
|---|---|---|
| SEC | A (pending `acceptanceDateTime` precision) | B |
| Fed | A | B |
| BLS | A on the release surface | B via dated archive; **D/E** for the series API alone |
| BEA | B | C |

No provider reaches grade A on backfill: a source timestamp recovered later can
establish *publication* time, never our historical *observation* time. That is
the design's `RETROSPECTIVE_SOURCE` mode, and it is why backfill is graded
separately.

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

## 6. Blocking unknowns before any capture-spec freeze

1. **`acceptanceDateTime` literal format, precision and timezone.**
   `data.sec.gov` was unreachable from this environment, so this was not
   confirmed against a live response or a field-level specification. This is the
   field the whole causal gate would rest on.
2. **Official redistribution / licence terms** for EDGAR content.
3. **The documented field linking an amendment to the filing it amends.**

## 7. What this does not establish

Provider suitability only. No predictive edge, no causal market impact, no
economic significance, no useful sentiment or surprise.

`commercial_edge_established = false`.
