# FOMC event capture spec V1 — Federal Reserve monetary-policy statements

```
CAPTURE SPEC ID       federal_reserve_fomc_capture_v1
VERSION               1
STATUS                FROZEN_PRE_IMPLEMENTATION

NOT implemented · NOT captured · NOT live · zero requests made in this phase

CAPTURE SPEC HASH
74571bb01188fbf38f1a9e5f4d20d95b21885c62a23981c706f83ea7070bd3cd

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

An unscheduled or differently-titled FOMC action would **not** match, and would
be skipped rather than guessed at. That narrowness is deliberate: widening the
predicate is a spec revision, not a judgement call made at runtime.

## 4. Time — the part worth getting right

### The source instant

The statement page carries a date line, a title and a release line in
adjacency:

```
January 28, 2026
Federal Reserve issues FOMC statement
For release at 2:00 p.m. EST
```

So `source_available_at` composes as:

```
2026-01-28  +  14:00  +  EST(-05:00)   →   2026-01-28T19:00:00Z
```

**The timezone is read, never inferred.** The document writes the literal
abbreviation — `EST` verified on 2025-12-10, 2026-01-28 and 2019-01-30; `EDT`
verified across 2016–2026 samples — so V1 accepts only the four explicit forms
and maps `EST → -05:00`, `EDT → -04:00`. A bare `ET` is not accepted at all.

### The three timestamps that are not it

`Last Update` is site maintenance metadata and can never advance availability.
HTTP `Last-Modified` and `Date` are transport facts. And the RSS `pubDate` is
**corroboration only** — its semantics are undocumented, and a GMT value is
ambiguous on its own: `19:00Z` is *both* 2:00 p.m. EST and 3:00 p.m. EDT. That
ambiguity is precisely why the document, not the feed, holds the timestamp.

### When there is no clock

The 2015-03-18 statement reads **`For immediate release`**. That is a
recognised syntax, not a malformed one, and the distinction matters:

| release line | outcome |
|---|---|
| explicit `H:MM a.m./p.m. EST/EDT` | exact `source_available_at` |
| `For immediate release` | **valid event**, `source_available_at = null` |
| anything else | `PARSER_FAILED` |

A statement is not corrupt merely because the Fed did not print a clock on it.
It simply has no source instant, and `observed_at` bounds it alone — a path
design rev4 already supports.

This also closes the R3 LOW finding: **V1 claims nothing about "all archived
statements."** The parser is record-specific. Samples were verified; universal
coverage was not, and the spec says so.

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

## 11. Eighteen cases, decided in advance

`FOMC01`–`FOMC18` in the JSON settle summer/winter releases, immediate release,
bare `ET`, a feed item whose page will not load, late observation, unchanged
and changed bytes under one GUID, GUID conflicts, `Last Update` drift, a 2027
backfill of a 2026 statement, local raw corruption, malformed XML, feed
outages, a genuinely empty poll, a failed commit after raw persistence, a lost
ACK, and an unchanged +7 d re-check.

Invariants `F01`–`F18` state the same commitments in testable form.

## 12. What this does not establish

Nothing has been captured, implemented or observed. No predictive edge, no
causal market impact, no economic significance.

`commercial_edge_established = false`.
