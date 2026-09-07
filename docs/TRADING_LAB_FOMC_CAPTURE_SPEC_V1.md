# FOMC event capture spec V1 — Federal Reserve monetary-policy statements

```
CAPTURE SPEC ID       federal_reserve_fomc_capture_v1
VERSION               1
STATUS                FROZEN_PRE_IMPLEMENTATION

NOT implemented · NOT captured · NOT live · zero requests made in this phase

CAPTURE SPEC HASH  (revision 3 — authoritative)
d4242ea54afbf0b60604edc597b8bcfdcc733883b4b77ec5b0a471e52dd33d79

supersedes 74571bb0… (rev 1) → b852b560… (rev 2)

BINDS
  event intelligence design rev4
  4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689
  provider verification rev3
  e854ddb9cc8b5bc2634fe6ca6fc793289bafd6c94bb49dde71109e1d17341034
```

Canonical artefact: `docs/artifacts/fomc_capture_spec_v1.json`. **The JSON is
authoritative.** Every causal and security decision lives there; this document
explains why, and adds nothing the JSON does not already bind.

This spec was built entirely from provider evidence already audited and frozen
in revision 3. **No network request was made while writing it.**

---

## 1. What this captures, and what it deliberately does not

One event family: **FOMC monetary-policy statements**. Nothing else.

Not the minutes, not the SEP, not press conferences, not speeches, not the
Beige Book, not FRED, not the Federal Reserve's other press releases, and no
other central bank. The point of a first capture contract is not coverage — it
is to prove one deterministic, causally honest, offline-replayable path end to
end. Breadth is a later argument.

## 2. Two surfaces with different authority

The distinction that the whole spec turns on:

| surface | role |
|---|---|
| `/feeds/press_monetary.xml` | **discovery only** — how we learn a statement exists |
| the statement HTML page | **semantic authority** — content, date, release instant |

The feed never sets `source_available_at`. If the feed and the page disagree,
**the page governs and a diagnostic is recorded**; the primary timestamp is
never silently rewritten. If the *identity mapping* itself conflicts, the item
fails closed rather than being merged.

URLs are never constructed from a date. A statement URL must arrive from an
official discovery link — a feed item's `<link>` live, or an official FOMC
calendar/archive link in backfill. The path pattern in the spec is a
**validation** of an officially supplied link, never a recipe for building one.

Allowlist: `www.federalreserve.gov`, exactly. The apex domain is *not*
included, because no canonical redirect requirement was ever verified for it —
listing it would be a guess wearing the costume of a security control. HTTPS
only, port 443 only, no userinfo, redirects validated *before* they are
followed, capped at 3. This reuses `safe_http`, the repo's single shared
allowlist/redirect primitive, rather than growing a second copy of it.

## 3. Deciding what is an FOMC statement

### A positive classifier, not a verdict on everything else

A URL containing `monetary` is not evidence. The predicate is a **conjunction**
on verified official fields:

```
feed item title         == "Federal Reserve issues FOMC statement"
AND
primary page title      == "Federal Reserve issues FOMC statement"
```

after Unicode NFC, entity unescaping and whitespace collapsing. That exact
title was verified on the FOMC statement feed items and on statement pages
sampled across 2015–2026.

Revision 2 treated a non-match as *"NOT an FOMC event"* — a definitive
negative. Counter-review R2 showed where that leads: a successful poll could
skip a differently-titled official monetary action and then report
`EVENTS_OBSERVED_ZERO`, asserting an absence it had not established. The
predicate is now what it always actually was — a **positive classifier for the
V1 subset**, and nothing more.

### Three outcomes, not two

| state | meaning |
|---|---|
| `IN_SCOPE_V1` | the exact predicate succeeds on **both** surfaces |
| `DEFINITELY_OUT_OF_SCOPE` | excluded by a frozen deterministic rule needing **no** semantic guessing — deliberately narrow |
| `UNRESOLVED_CANDIDATE` | on the official monetary surface, fails the positive predicate, and cannot be proven outside the domain without guessing |

**A non-match defaults to `UNRESOLVED_CANDIDATE`.** Unknown relevance resolves
to unresolved, never to a negative — and there is no frozen safe negative rule
for a differently-titled official monetary item, so the default is what
actually applies in practice.

No revision is created either way. The difference is that an unresolved
candidate **blocks the zero assertion** for that cycle.

V1 is deliberately not widened with keyword lists, semantic similarity, an LLM,
regex over body text, or hardcoded emergency-action vocabulary. Honest
incompleteness beats speculative coverage.

### What the scope actually claims

```
taxonomy_event_family                          FOMC_MONETARY_POLICY_STATEMENT
capture_scope_id                               standard_fomc_statement_release_pattern_v1
coverage_claim                                 ONLY_EVENTS_CONFIRMED_BY_FROZEN_V1_PREDICATE
coverage_complete_for_all_fomc_actions         false
coverage_complete_for_all_monetary_feed_items  false
```

The frozen rev4 taxonomy type is kept; `capture_scope_id` records that V1
captures a deterministic **sub-family** of it, and is bound into every
normalized revision so no downstream consumer can confuse *provider family*
with *coverage completeness*.

### Zero means one specific thing

`EVENTS_OBSERVED_ZERO` requires **all seven**:

1. the discovery feed fetch succeeded
2. the raw feed artefact was durably persisted
3. the feed parse succeeded
4. every relevant feed item was classified **deterministically**
5. `unresolved_candidate_count = 0`
6. no primary-page fetch or parse remains unresolved for a candidate that could affect the result
7. newly admitted `IN_SCOPE_V1` events = 0

Fetch and parse success alone is **not** sufficient. What the state asserts is:

> no new events matching this spec's frozen V1 predicate were confirmed during
> this successful discovery cycle

and never *"zero FOMC events"*, *"no FOMC statement occurred"* or any
complete-world claim. Downstream, it **may not** be read as *"the Federal
Reserve produced no monetary-policy event"*, and no model or feature may treat
it as a complete-world event count without a future coverage contract — a
constraint that binds later InformationSnapshot composition.

An emergency inter-meeting action is the highest-impact event class and the one
most likely to carry a non-standard title. Under revision 3 it produces an
unresolved candidate and blocks the zero, rather than becoming evidence of
nothing having happened.

## 4. Time — the part worth getting right

### Two different questions

Revision 1 collapsed these into one, and counter-review R1 caught it. They are
separate, and V1 now keeps them apart by construction:

| question | answered by |
|---|---|
| when was the **logical statement** released? | `declared_release_at` — provider metadata |
| when did **these exact bytes** become public? | **unknown** — so `source_available_at = null` |

### The declared release time

The statement page carries a date line, a title and a release line in
adjacency:

```
January 28, 2026
Federal Reserve issues FOMC statement
For release at 2:00 p.m. EST
```

The strict grammar is unchanged: only the four explicit forms are accepted,
`EST → -05:00`, `EDT → -04:00`, the timezone is **read, never inferred** (`EST`
verified on 2025-12-10, 2026-01-28 and 2019-01-30; `EDT` across 2016–2026), and
a bare `ET` is not accepted at all.

What changed is its **role**. It composes into `declared_release_at` —

```
2026-01-28  +  14:00  +  EST(-05:00)   →   2026-01-28T19:00:00Z
```

— and that value is **non-gating provenance**. It never gates content
visibility, and it may never be written into `source_available_at`,
`source_updated_at`, `effective_at`, `observed_at` or `ingested_at`.

### Content availability: always null in V1

```
content revision source_available_at = null      ← every revision, both modes,
                                                   no exception
```

The Federal Reserve contract verified in provider revision 3 exposes **no
revision-specific source availability timestamp**. The release line dates the
*logical event*; it says nothing about when the byte sequence HyprL is holding
came into existence. Assigning it to a content revision would retroproject a
possibly-later body onto the original release instant — the HIGH that R1 found.

So a content revision is bounded by what HyprL can actually prove:

```
live_observed_available_at = observed_at
durable_available_at       = max(observed_at, ingested_at)
```

This needs **no design change**: rev4 already supports a null
`source_available_at` bounded by `observed_at`, and V1 simply always takes that
branch.

This applies even to the *first* live body. A statement declared at 14:00:00 and
first received at 14:00:47 cannot be proven byte-for-byte to have existed at
14:00:00 — so it is visible from observation and commit, not from 14:00. That
closes a subtle 47-second retroprojection.

### The timestamps that are not it — and one rejected suggestion

`Last Update`, HTTP `Last-Modified` and RSS `pubDate` are all barred from
`source_available_at` **and** from `source_updated_at`, and none is vintage
proof, revision chronology or an identity input.

R1's own remedy proposed capturing `Last Update` into `source_updated_at`. That
proposal is **rejected**. Provider revision 3 established `Last Update` as
page-maintenance metadata, not a verified content-revision publication
timestamp, and a wrong vintage signal is worse than an honestly declared absent
one. `source_updated_at` stays **null** in V1.

`pubDate` remains corroboration only — its semantics are undocumented, and
`19:00Z` is ambiguous between 2:00 p.m. EST and 3:00 p.m. EDT.

### When there is no clock

The 2015-03-18 statement reads **`For immediate release`**. That is a recognised
syntax, not a malformed one:

| release line | `declared_release_at` | content `source_available_at` |
|---|---|---|
| explicit `H:MM a.m./p.m. EST/EDT` | the composed instant | **null** |
| `For immediate release` | **null** | **null** |
| anything else | `PARSER_FAILED` | — |

A statement is not corrupt because the Fed did not print a clock on it.

### What retrospective mode can and cannot do

```
DURABLE_OBSERVED                        SUPPORTED
retrospective release-time metadata     SUPPORTED
retrospective exact-content visibility  NOT_SUPPORTED_V1
```

A backfilled body is **not** eligible to become source-visible at
`declared_release_at`; it is visible only on its actual observed/durable
timeline. And no trust verdict can substitute for missing vintage proof: a
`TRUSTED_EXACT` release timestamp means *"the source reliably declares the
logical release at T"*, **not** *"every currently fetched byte existed at T"*.

There is deliberately no blanket "retrospective supported" flag anywhere in the
spec.

The raw body hash is provenance for what HyprL actually observed and when. It
does not become proof of historical content existence merely because the same
record also carries a `declared_release_at` recovered from the page.

## 5. Identity without trusting a URL

```
logical item key = (provider_id, event_family,
                    official_statement_date, canonical_primary_statement_url)
source_item_id   = SHA-256(canonical JSON of that tuple)
```

which then feeds design rev4's `SourceObservationId` — *(provider,
source_item_id, content_hash)*, never a URL alone.

The RSS `<guid>` is stored as **discovery provenance**, and is neither business
identity nor revision identity nor proof that content is unchanged. Backfill
cannot depend on it at all, since historical statements fall outside current
feed depth. Provider-side uniqueness is not contractually documented, so
identity quality stays **GOOD**, not EXCELLENT — and both conflict directions
(one GUID → two items, one item → incompatible GUIDs) **fail closed**. Never
"latest wins."

## 6. Revisions, without assuming the Fed never edits

Upstream correction semantics are **UNKNOWN**, and V1 never converts that into
an immutability assumption. The client side does the work instead:

```
same item, same body hash        → no duplicate revision
same item, different body hash   → append a NEW immutable revision
overwrite                        → forbidden, raw and normalized alike
```

Re-observation is scheduled at **+5 min, +1 h, +24 h, +7 d** after the first
successful fetch — catching immediate, same-day, next-day and delayed
corrections for about four extra requests per statement, at roughly eight
statements a year.

After +7 days, **V1 promises nothing.** That is stated as a limitation rather
than disguised as immutability; extending the window is CaptureSpec V2.

## 7. Rate, honestly

```
provider_rate_limit                    UNKNOWN
provider_automated_access_prohibition  NOT_VERIFIED
client_max_requests_per_minute         6      ← HyprL policy
live_feed_poll_interval_seconds        60
minimum_request_spacing_seconds        10
```

The Federal Reserve has **not** granted a quota of 6 rpm or any other figure.
No published numeric limit was found, and no prohibition of automated access
was found either — two different facts, kept apart. The ceiling is a
conservative client policy covering the poll, occasional page fetches and
bounded re-checks, with no burst retries and no hidden retry-library behaviour;
on failure the collector records source health and waits for the next normal
cycle.

Polling is plain elapsed-time. V1 derives no schedule from the FOMC calendar,
the 2:00 p.m. tradition or `pubDate` — a predictive scheduler would be a second
source of truth about time, which is the one thing this design refuses.

## 8. Raw first, always

```
fetch bytes → durable raw + hash → parse → durable normalized revision → snapshot-eligible
```

on every feed poll, every statement fetch and every re-observation. A parser
never touches bytes that lack a durable raw identity. `raw_body_bytes` is the
entity body *after* standard HTTP content decoding, hashed with SHA-256
**before** any text decode, with `Content-Encoding` retained as metadata — so a
charset problem becomes `PARSER_FAILED` rather than a silently mangled hash.

Correctness must not depend on HTTP 304: a conditional request may never be the
reason a raw artefact is missing.

Security requirements are frozen too — XML external entity resolution
**disabled**, no JavaScript, no browser, no shelling out, bounded sizes
(2 MiB feed, 5 MiB statement), bounded timeouts (10 s connect, 30 s read,
matching the repo's existing convention), and **no network during replay**.

## 9. Redistribution — stricter than the evidence requires

R3 verified the Federal Reserve disclaimer: *"Unless otherwise indicated,
information on Board's website is in the public domain and may be copied and
distributed without permission."*

V1 is nonetheless `LOCAL_RESTRICTED`, redistribution `DISABLED`. Infrastructure
proof does not need to ship raw bytes, and a blanket refusal avoids doing
per-item *"unless otherwise indicated"* rights analysis on every exhibit and
image. The evidence layer records what is true; the capture policy stays
tighter than it.

## 10. What the capture layer may not contain

No sentiment, importance, novelty, LLM summary, asset impact, expected
reaction or target. No market data, returns or volatility — this phase does not
join Phase 1C at all. No consensus, expected move or probability. And no rate
decision extraction: target range, basis points, vote split, stance and QT
amounts are all deferred, because none of them is needed to *identify* the
statement family, and infrastructure is what 6G-A has to prove.

**No LLM anywhere in ingestion.** A model may not decide whether something is
an FOMC statement, nor its date, release time, identity, revision or source
authority. Those are deterministic parses or they are nothing.

## 11. Twenty-five cases, decided in advance

`FOMC01`–`FOMC25` in the JSON settle summer/winter releases, immediate release,
bare `ET`, a feed item whose page will not load, late observation, unchanged
and changed bytes under one GUID, GUID conflicts, `Last Update` drift, a 2027
backfill of a 2026 statement, local raw corruption, malformed XML, feed
outages, a genuinely empty poll, a failed commit after raw persistence, a lost
ACK, and an unchanged +7 d re-check.

Revision 2 added three that close the R1 HIGH, and revision 3 four more that
close the R2 MEDIUM:

| | case | outcome |
|---|---|---|
| `FOMC19` | body corrected upstream at 16:00, first backfilled in 2026 | at 14:30 the fetched body is **not visible**; no claim about the uncaptured original either |
| `FOMC20` | live correction, both bodies print the same release line | neither revision inherits 14:00 |
| `FOMC21` | first live fetch 47 s after the declared release | still `source_available_at = null` |
| `FOMC22` | unscheduled monetary action, different title | unresolved candidate; **zero prohibited** |
| `FOMC23` | feed title matches, primary title differs | conflict; no event, no zero, no silent downgrade |
| `FOMC24` | clean cycle, no matches, no unresolved | zero **permitted**, scoped to the V1 subset |
| `FOMC25` | positive candidate, primary fetch fails | unresolved; zero prohibited |

Invariants `F01`–`F24` state the same commitments in testable form. Revision 2
added `F19` (release time is not vintage proof), `F20` (backfill content
vintage) and `F21` (revision-specific availability); revision 3 adds `F22`
(an absence claim cannot exceed classifier coverage), `F23` (an unresolved
candidate blocks zero) and `F24` (discovery/primary conflict fails closed).

## 12. What this does not establish

Nothing has been captured, implemented or observed. No predictive edge, no
causal market impact, no economic significance.

`commercial_edge_established = false`.
