# FOMC event capture spec V1 — Federal Reserve monetary-policy statements

```
CAPTURE SPEC ID       federal_reserve_fomc_capture_v1
VERSION               1
STATUS                FROZEN_PRE_IMPLEMENTATION

NOT implemented · NOT captured · NOT live · zero requests made in this phase

CAPTURE SPEC HASH  (revision 25 — authoritative)
b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece

supersedes 74571bb0 (rev 1) → b852b560 → d4242ea5 → f0b68307 → 8a39a39a → 6693a665
        → 3db09125 → c1d56f46 → 0757b1f5 → 3dcf0a60 → b9ad3c44 → b6772a2f
        → 67e2d7d5 → bbc29abb → 49f8050f → 69156776 → 5cec4f30
        → dd0cff22 → 83df6d2d → 83caecb0 (rev 20) → 12f929f4 (rev 21)
        → ba6a01e5 (rev 22) → 3fc2f9a7 (rev 23) → 235e474c (rev 24)

BINDS
  event intelligence design rev4
  4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689
  provider verification rev3
  e854ddb9cc8b5bc2634fe6ca6fc793289bafd6c94bb49dde71109e1d17341034
  FOMC DOM anchor evidence V1
  bd7d17d29c3f71efa24a503f9ffab75ff0ba2b051544dae25723daad745c6a3a
  commit 1707895794c31a7ca9425f4a3dce48a6590b845c
```

Canonical artefact: `docs/artifacts/fomc_capture_spec_v1.json`. **The JSON is
authoritative.** Every causal and security decision lives there; this document
explains why, and adds nothing the JSON does not already bind.

This spec was built from provider evidence already audited and frozen, plus the
committed DOM anchor evidence bound above. **No network request was made while
writing revision 25.** Its evidence is the local raw of the revision-23 pilot
(no new download).

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
official discovery link: a LIVE feed item's `<link>`, or an entry of a validated
backfill manifest (§6). The path grammar is a **validation** of an officially
supplied link, never a recipe for building one. Since revision 20 it is exact:
`/newsevents/pressreleases/monetary<YYYYMMDD><letter><digits>.htm`, with a real
calendar date. Statements and other monetary releases share this family, so the
suffix letter says nothing about relevance. The titles decide (§3).

Allowlist: `www.federalreserve.gov`, exactly. The apex domain is *not*
included, because no canonical redirect requirement was ever verified for it —
listing it would be a guess wearing the costume of a security control. HTTPS
only, port 443 only, no userinfo, redirects validated *before* they are
followed, capped at 3. From `safe_http` only `require_https_host` is reused. The
repo's `AllowlistedRedirectHandler` is **not** conforming: it lets urllib follow
up to ten redirects by itself inside one `urlopen`, with no per-hop limiter grant
or deadline. V1 requires a manual redirect loop (§7a), which Phase 6G-A must
still build.

## 3. Deciding what is an FOMC statement

### A positive classifier, not a verdict on everything else

A URL containing `monetary` is not evidence. The predicate is a **conjunction**
on verified official fields:

```
feed item title         == "Federal Reserve issues FOMC statement"
AND
primary page title      == "Federal Reserve issues FOMC statement"
```

Revision 2 treated a non-match as *"NOT an FOMC event"* — a definitive
negative. Counter-review R2 showed where that leads: a successful poll could
skip a differently-titled official monetary action and then report
`EVENTS_OBSERVED_ZERO`, asserting an absence it had not established. The
predicate is now what it always actually was — a **positive classifier for the
V1 subset**, and nothing more.

### Where those values are read from — the anchors

Revision 10 named three anchors and defined none of them: `"official statement
date"`, `"page title / event family"`, `"release line"`. R10 showed that is not
a specification. The document `<title>` and the visible heading carry *different
strings*, so one implementation would classify every statement and another would
classify none — both conforming.

Revision 11 freezes the anchors against committed structural evidence
(`bd7d17d2`, 5 pages, 2021–2026):

```
div#article
  └─ div[class token "heading"]        ← authoritative region
       ├─ p.article__time              → official_statement_date
       ├─ h3.title                     → primary_page_title
       └─ p.releaseTime                → release_line (start tag)
```

Each anchor must resolve to **exactly one** node, never first-wins, last-wins,
visible-wins or traversal order. Zero or two title or date anchors →
`PARSER_FAILED`. Zero or two release-line anchors only lose the release time
(`RELEASE_TIME_UNPARSED`, non-gating); the statement is kept.
Layout classes on the heading container (`col-xs-12 col-sm-8 col-md-8`) were
observed but are **not** required; only the `heading` token is.

**The document `<title>` is excluded on two independent grounds.** Its text is
`Federal Reserve Board - Federal Reserve issues FOMC statement` — prefixed, so it
fails the exact predicate. And the selector is **not unique**: every sampled page
carries a second `<title>Lock</title>` inside an inline SVG icon. It survives as
diagnostics only; it can neither satisfy nor break the classifier.

**`div#lastUpdate` is excluded structurally, not by value.** It sits outside
`div#article`. Its displayed value *equalled* the statement date on all five
samples — so a value check could never have separated them. Only structure can.

### The release line is token-bounded, not DOM-text-bounded

The evidence turned up something that would have broken a naive freeze:

```html
<p class="releaseTime">For release at 2:00 p.m. EST
<ul class="list-unstyled">        ← there is no </p>
```

**`p.releaseTime` is never closed.** A WHATWG parser implicitly closes the
paragraph at `<ul>`; a permissive parser does not, and swallows the share menu
and the entire statement body. That is not theoretical — it is why the evidence
run's own first analysis pass found *zero* release candidates on all five pages
while finding title and date instantly.

So `release_line` is defined at the **token level**, not by DOM text:

> after locating the unique `p.releaseTime` **start tag**, take the contiguous
> character data immediately following it and preceding the **first subsequent
> markup token**.

`element.textContent`, recursive descendant text, browser-rendered text and
recovered-DOM subtree text are all **forbidden as authority**. No parser library
is pinned; deterministic tokenization distinguishing start tag / character data /
character reference / next markup token is required. A parser that has already
repaired the DOM past that boundary may not substitute its repaired text.

Empty leading segment → `PARSER_FAILED`. Two independent declarations in the
segment → `PARSER_FAILED`. Never first-match.

### The normalization pipeline, in order

R10 also found the transformations were listed without an order, which changes
results. Now numbered:

```
1. extract text from the frozen field source, resolving HTML character references
2. Unicode NFC
3. collapse each run of ASCII whitespace (09, 0A, 0C, 0D, 20) to one U+0020
4. trim leading/trailing U+0020
5. exact compare, or strict grammar parse
```

**Entity resolution precedes NFC.** `&#101;&#769;` → `U+0065 U+0301` → NFC →
composed form. NFC-first would leave a reference-encoded decomposed sequence
unnormalized and make the outcome order-dependent.

**NBSP is preserved.** `U+00A0` is not ASCII whitespace, so `&nbsp;` survives
step 3 as `U+00A0`. No browser-layout equivalence is imported. Only NFC — never
NFD, NFKC, NFKD, case folding or accent removal.

On the evidence: `h3.title` and `p.article__time` were already clean ASCII on
5/5; only the release segment carried trailing LF and multiple spaces, so collapse
and trim are genuinely required there. NFC, NBSP and character references were
**not exercised** by the sample — the pipeline is frozen for determinism, not
because the data demanded it.

### These anchors are empirical, and fail closed

The Federal Reserve publishes no markup contract. The anchors hold across 5
pages, 2021–2026, one template family. Nothing is claimed for 2015, 2019, all Fed
history or all monetary pages, and no `For immediate release` page was captured —
so its DOM placement is not empirically exercised, though its **grammar** is
unchanged: same anchor, existing grammar; different structure, non-gating
`RELEASE_TIME_UNPARSED` (the statement is kept).

If upstream title or date markup drifts and cardinality breaks, V1 stops with
`PARSER_FAILED`.
Heuristic broadening and automatic alternative-node search are forbidden. That is
deliberate drift detection, not a limitation to be engineered around.

### Three outcomes, not two

Every **new** feed item in the monetary family is fetched once, whatever its
suffix or feed title. The titles are normalized first: parsed XML text, NFC,
ASCII whitespace collapsed and trimmed, NBSP kept, no case folding. Then the
primary page decides:

| feed title exact? | primary title exact? | outcome |
|---|---|---|
| yes | yes (and the date anchor succeeds) | `IN_SCOPE_V1` |
| no | no (title anchor extracted) | `DEFINITELY_OUT_OF_SCOPE`, a **healthy** negative for the V1 subset |
| one of them | the other not | `CLASSIFICATION_CONFLICT`, unresolved |
| title anchor fails | — | `PARSER_FAILED`, unresolved |

A negative needs positive evidence: a successfully parsed page whose title fails
the V1 predicate. URL shape is never negative evidence. A safe official link
outside the family is an **unsupported discovery shape**: it is unresolved and
never fetched, and only an operator resolution concludes it. A differently-titled
FOMC action with a differently-titled feed item is outside V1 and not captured.
Zero is a V1-subset assertion and never says that no FOMC action happened.

V1 is deliberately not widened with keyword lists, semantic similarity, an LLM,
regex over body text, or hardcoded emergency-action vocabulary.

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

Zero is a property of **one LIVE discovery cycle**, started by one durable feed
record that is `LIVE_ELIGIBLE` (defined below). Its boundary B is that record's
position in the store's commit order. The cycle can be zero only if:

1. the feed record is `LIVE_ELIGIBLE` and parses;
2. every earlier processable LIVE feed record was already **terminal** before B
   (classified, or ended in a terminal failure such as a channel-level parser
   failure, so a malformed feed never blocks polling). Every
   processable LIVE feed record is classified, even one whose clock check
   failed, and **strictly in commit order** against every earlier
   classification. So a listing seen during a clock fault is never skipped, and
   neither is a late response from an old or interrupted poll: it is classified
   as unverified evidence, so its exact title still raises a marker;
3. it lists **no new item**: nothing unidentifiable, no new feed candidate, no
   first sighting of an unsupported path, and no *new* GUID or feed-title value
   (each new value raises a diagnostic exactly once);
4. the durable state before B holds **no outstanding relevant item**, whether the
   feed still lists it or not.

Two terms are kept apart. A record is **processing-terminal**
(`RECORD_PROCESSING_TERMINAL`) once its local processing has a final outcome. An
item is **semantically concluded** (`ITEM_SEMANTICALLY_CONCLUDED`) by the rule
below. **An item may contribute to zero only if it is semantically concluded.**
The items are:

- LIVE feed candidates;
- unidentifiable feed items, which get an item-level record, never just a
  feed-level `PARSER_FAILED`;
- feeds whose own processing failed.

An item is **concluded** in exactly two ways:

1. **Naturally.** Every per-response record is processing-terminal, no LIVE
   episode or manual episode of LIVE work is open, no marker exists after the latest valid resolution, and the item's
   latest `LIVE_ELIGIBLE` record ended in a normalized in-scope outcome or a
   healthy `DEFINITELY_OUT_OF_SCOPE`. Every later processable LIVE record must
   have an outcome of the same class: a clock-ineligible record **may reopen an item, never
   conclude one**.
2. **By an operator resolution with a cutoff.** The resolution is valid only if
   every per-response record of the item is already processing-terminal, no LIVE
   episode or manual episode of LIVE work is open, and no acquisition could still
   open its episode (it has none while the item has no LIVE anchor) when it
   commits. An acquisition that can no longer open, because an operator retry
   already anchored the item, never blocks a resolution (`FOMC247`). It covers exactly
   the records up to its own commit sequence, and **any later record re-opens the
   item**. There is no same-content exemption; a resolution is a decision about
   the durable state at its cutoff, never a permanent permission.

Everything else is outstanding: a pending, retrying or suspended fetch, a parser
failure, a title conflict, an unsupported path, a corrupt raw, an internal error,
an item seen only through clock-ineligible observations, or anything newer than
the last resolution. GUIDs are provenance. A new GUID on a known URL, or a
known GUID on a new URL, raises a diagnostic once per value and makes that cycle
not zero. A missing GUID raises one diagnostic per item and never blocks zero.
None of these is a marker. Two markers remain, both cleared only by a resolution and only
up to its cutoff: a feed retitle to the exact statement title, and a missing or
invalid title when the item is first seen.

A channel-level malformed feed clears once a later feed parses. That leaves a
stated residual: an item that appears only in malformed feeds is lost. A local
processing defect in a feed needs an operator. Backfill observations never count
here, and operator retries of LIVE work are LIVE observations. A real statement
whose page will not parse therefore blocks zero until it is concluded, even after
it leaves the feed. That fail-safe cost is intended.

A routine non-statement makes its first cycle non-zero while it is being
investigated. It then concludes as a healthy negative and blocks nothing; no
`PARSER_FAILED` is recorded. That assumes its page carries the usual title
anchor, which is expected but not evidenced: the DOM evidence covers statements
only. A page without it stays outstanding until a later observation or an
operator concludes it. Backfill work never blocks a LIVE cycle. A
backfilled item seen by the LIVE feed for the first time is a new candidate
there.

The verdict is computed from the fixed pre-cycle state, written once, and never
rewritten. "Newly admitted" is reporting provenance: the items a cycle
discovered that were later admitted, by whatever observation.

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

Revision 19 freezes what happens when the printed label looks wrong for the
season (`LITERAL_ZONE_LABEL_V1`): **nothing is corrected.** `EST` is always
UTC−05:00 and `EDT` always UTC−04:00, whatever the date:

```
2026-07-15  +  2:00 p.m. EST   →   2026-07-15T19:00:00Z     (no rewrite to EDT)
2026-01-15  +  2:00 p.m. EDT   →   2026-01-15T18:00:00Z     (no rewrite to EST)
```

No America/New_York rule is consulted, so conversion needs no timezone
database and replays identically everywhere. An implementation may attach a
non-causal "seasonally unusual" diagnostic, but it is optional, never changes
the value and never rejects the page. Any other label — `ET`, `Eastern`,
`local time`, no zone, an NBSP or case variant, `noon`, a leading zero — matches
no grammar. The result is `RELEASE_TIME_UNPARSED`: `declared_release_at` is null,
no zone is guessed, a diagnostic is recorded, and **the statement is kept**.
Revision 20 restores the non-gating rule of revision 1. A missing or duplicated
release-line anchor is treated the same way. The title and date anchors still
fail closed.

The accepted clock is strict: the hour is 1–12 without a leading zero, minutes
are two digits, `12 a.m.` is 00 and `12 p.m.` is noon, so `12:30 p.m. EDT` on
2026-06-17 is `16:30:00Z`.

All FOMC timestamps use the repo's existing canonical primitive,
`market_data_store._canonical_timestamp`: a UTC `datetime.isoformat()` with
`+00:00`, at whatever resolution Python's datetime provides. Arithmetic and
ordering use parsed instants, never string comparison. The `Z` forms in the cases
are notation only.

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
live_observed_available_at = observed_at            (diagnostic only)
durable_available_at       = avail of the link's commit (server-attested, ≥ observed_at)
```

Worked example: a statement observed at 18:00:47 commits at once. The next
feed poll's request starts after that commit, and its verified response has
`observed_at` 18:01:40. So the statement's availability is 18:01:40 + 92 s =
**18:03:12**. Snapshots at 18:00:49 and 18:03:11 do not show it; the snapshot at
18:03:12 does.

This needs **no design change**: rev4 already supports a null
`source_available_at` bounded by `observed_at`, and V1 simply always takes that
branch.

This applies even to the *first* live body. A statement declared at 14:00:00 and
first received at 14:00:47 cannot be proven byte-for-byte to have existed at
14:00:00 — so it is visible from observation and durable commit, not from 14:00. That
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

## 5. Logical identity and discovery URL authority

```
logical item key = (provider_id, event_family, canonical_primary_statement_url)
source_item_id   = SHA-256(canonical JSON array of those three string values)

identity_url_authority = OFFICIAL_DISCOVERY_LINK_PRE_REDIRECT
canonical_primary_statement_url = normalize(official_discovery_link_before_redirect)
```

which then feeds design rev4's `SourceObservationId` — *(provider,
source_item_id, content_hash)*, never a URL alone.

The values are serialized in that order using the existing canonical JSON rule.
`official_statement_date` is required normalized metadata of the observed
content revision. It never participates in the logical item hash or identity
dedupe. Neither do runtime timestamps, `declared_release_at`, RSS `pubDate`,
`Last Update`, `source_updated_at`, `content_hash` or RSS GUID.

Revision 4 addresses R3/X6: a corrected date on the same canonical primary URL
must preserve the logical item. Revision 5 defines the authority of that URL:
the official discovery link **before primary redirects**, normalized using the
existing V1 transforms. If that URL cannot be established, the existing
fail-closed policy applies. Date-only, `(provider, event_family, date)` and
GUID+date fallback identities remain forbidden.

The RSS `<guid>` is stored as **discovery provenance**, and is neither business
identity nor revision identity nor proof that content is unchanged. Backfill
cannot depend on it at all, since historical statements fall outside current
feed depth. Provider-side uniqueness is not contractually documented, so
identity quality stays **GOOD**, not EXCELLENT. Since revision 21 the URL is the
identity and the GUID is provenance only. One GUID on two URLs never merges them,
and a changed GUID never splits an item. Each new value raises one diagnostic
that makes its cycle not zero, without withholding any event. Never "latest
wins."

A date correction alone is not an identity conflict. With the same URL and
GUID, it preserves `source_item_id`; historical backfill (manifest) observations
without a GUID do so as well. Existing GUID conflict rules still apply when
independently triggered.

The date remains required for normalization, under the existing primary-page
authority and date grammar. A date parse failure follows the existing
parser/source-health policy. Only its identity role changes.

### The discovery link is the identity input

For LIVE, use the official monetary-policy RSS item's `<link>`. For
HISTORICAL_BACKFILL, use the entry of the validated backfill manifest (§6). In
both cases the discovery artifact must already be durable
under existing raw-first and provenance rules. URLs are never synthesized from
dates, and historical identity requires no GUID.

The JSON freezes this conceptual order:

1. Fetch and durably persist the official discovery artifact.
2. Parse its official statement link `U_discovered`.
3. Validate that link using existing V1 URL/security rules.
4. Apply the existing frozen URL normalization to that link.
5. Bind the result as `canonical_primary_statement_url`.
6. Compute `source_item_id` from the unchanged three-field key.
7. Perform the primary request under existing HTTP policy.
8. Follow only redirects admitted by existing security rules.
9. Persist the redirect chain and final URL as transport provenance.
10. Never recompute `source_item_id` from the final URL.

Computing a candidate identity does not admit an event. Primary classification,
normalization, raw-first ordering and server-attested visibility still apply.
The existing normalized field `canonical_source_url` carries the identity URL;
no field is renamed and no normalized or raw schema is expanded.

### HTTPS scheme admission and serialization

Revision 6 fixes the scheme-casing ambiguity found at R5 point 8. Admission
accepts only HTTPS, comparing the parsed scheme to `https` **ASCII
case-insensitively**: map ASCII `A-Z` to `a-z` only, with no Unicode case folding
or scheme inference. `https`, `HTTPS`, `Https`, `hTtPs` and `HtTpS` all pass the
scheme check. `http`, `ftp`, `file`, `data`, `javascript` and an empty/missing
scheme are rejected. There is no silent HTTP-to-HTTPS upgrade, no identity
admission and no primary request for a rejected scheme.

For the scheme component, the order is: extract the official discovery URL,
parse it, validate HTTPS without ASCII case distinction, reject any other
scheme, serialize exactly lowercase ASCII `https`, then apply all other
existing URL normalization rules unchanged. Non-scheme validation remains
mandatory; the discovery URL still supplies identity before primary redirects.

FOMC31 freezes these exact strings:

```
input:     HTTPS://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm
canonical: https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm
```

The lowercase input and mixed-case HTTPS variants produce that same canonical
URL and, for the same provider and event family, the same `source_item_id`.
Scheme case alone cannot create a source item, identity conflict or content
revision. Original source spelling remains available in durable raw discovery
provenance; those bytes are never rewritten to canonicalize the derived
identity URL.

### HTTPS default-port admission and serialization

Revision 7 fixes only the port-serialization ambiguity found at R6 point 11.
For an otherwise valid HTTPS discovery URL, an absent port is accepted and an
explicit decimal port equal to 443 is accepted. Every other explicit port is
rejected, including `:80`, `:444`, `:8443`, `:1`, `:65535` and `:0`.

For the port component, parse the official discovery URL, distinguish absence
from an explicit token, validate that entire token, then admit only decimal
443 when present. The token must contain one or more ASCII digits `0-9`, parsed
in base 10 with value in `0..65535`; leading zeroes do not change its value
(`0443` is decimal 443). Signs, whitespace, hexadecimal notation, partial-token
parsing, non-numeric tokens, an empty explicit port and out-of-range values
fail closed. Malformed explicit syntax must never become "port absent" merely
because a parser supplies no numeric port. No library-specific exception class
is prescribed.

After successful validation, canonical identity serialization ALWAYS omits the
port component. The authority contains only the hostname normalized by the
existing rule, never `:443`. FOMC33 freezes the exact convergence:

```
input A:   https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm
input B:   https://www.federalreserve.gov:443/newsevents/pressreleases/monetary20260617a.htm
canonical: https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm
```

With the same provider and event family, both inputs yield byte-identical
identity URLs and the same `source_item_id`. Explicit default-port presence
alone cannot create a logical item, identity conflict or content revision.
Original `:443` spelling remains in durable raw discovery provenance only;
those bytes are never rewritten to canonicalize the derived identity URL.

Reject non-443 or malformed ports BEFORE identity derivation and BEFORE any
request. Never strip a custom port and continue, rewrite it to 443, or upgrade
HTTP to HTTPS. FOMC34 freezes `:444` rejection with no identity admission or
primary request. Scheme admission and all other validation remain mandatory.

The same port admission applies to each redirect target before follow under
the existing security policy. An admitted target with `:443` remains
non-identifying and cannot re-key the discovery-anchored item. Redirect limits,
host allowlisting and all other redirect semantics are unchanged.

### Percent-escape syntax admission, without decoding

Revision 8 fixes only the malformed-percent admission ambiguity found at R7
point 17. Every literal `%` in a V1-admitted URL must begin this exact grammar:

```
PCT_ENCODED := "%" HEXDIG HEXDIG
HEXDIG     := ASCII 0-9 / A-F / a-f
```

At each `%`, the next two characters must exist and be ASCII hex digits.
Consume that three-character escape and continue scanning the remaining URL
text; ordinary following characters are not part of that escape. No represented
octet is decoded, and decoded content is not rescanned.

`%`, `%2`, `%G0`, `%0G`, `%GG`, `%2Z`, `%%20`, `%2%` and `%u1234` are rejected.
`%20`, `%2F`, `%2f`, `%7E`, `%7e`, `%00` and `%FF` pass THIS syntax gate only.
Syntax success never overrides any existing component, security or primary-path
restriction and never by itself admits an identity or a request.

The relevant order is: extract the official discovery URL from durable
provenance; parse under existing V1 policy; validate existing structural
constraints; check every literal `%`; reject any malformed escape; only then
continue existing normalization/identity derivation and request admission.
The check covers every URL component before normalization can discard or
rewrite it. Query rejection, fragment removal and userinfo rejection remain
unchanged; none is a repair mechanism for malformed percent syntax.

A malformed escape cannot produce an admitted
`canonical_primary_statement_url` or `source_item_id` and cannot reach the safe
request layer. For example, the discovered link ending
`monetary20260617a%.htm` is rejected, not hashed literally. Never replace `%`
with `%25`, strip it, decode a partial escape, guess a missing nibble, change
an invalid character's case and retry, or use a literal-percent fallback.

The same gate applies to each redirect target before final admission and
follow: `https://www.federalreserve.gov/path%GG` is rejected before request.
Redirect targets remain non-identifying; all other redirect rules are unchanged.
Malformed source text may remain inside the durable raw discovery artifact.
Those bytes are not mutated: raw provenance is not URL admission.

For a valid escape already admitted under REV7, the existing path rule remains
`PRESERVE VERBATIM`. This fix adds no decoding or re-encoding and makes no new
decision about hex-case canonical equivalence (`%2f` versus `%2F`), `%2F` versus
`/`, `%7E` versus `~`, or `%2E` versus `.`. FOMC38's `/path%2Fsegment` remains
unchanged by this syntax gate; it does not assert full V1 primary-URL admission.

### Transport and HTML URLs do not re-key items

Here "canonical" means the normalized official discovery-link URL selected by
V1. The actual request URL at each hop, ordered redirect chain and final
allowlisted response URL are separate transport provenance. They cannot replace
the identity URL, merge or split items, or cause an identity conflict solely
because the redirect destination differs. Existing GUID conflicts still apply.

| observed provenance | identity URL |
|---|---|
| feed link or manifest entry U1 redirects to U2 | `normalize(U1)` |
| the same U1 later redirects to U3 | the same `normalize(U1)` |
| primary HTML declares canonical U4 | still `normalize(U1)` |
| manifest entry U1 redirects to U2, with no GUID | `normalize(U1)` |

An HTML `<link rel="canonical">` has no semantic or identity authority in V1
and is not a primary semantic anchor. If retained, it is optional diagnostic
provenance only; it authorizes no network retrieval. A mismatch alone cannot
change identity, merge or split items, or create an identity conflict.

Re-observations retain the stored U1-derived identity URL. A request may start
from that official identity URL; its final destination that day never re-keys
the item. With the same body, a transport change creates no content revision.
With a changed body, the existing revision policy appends a revision under the
same item. Recheck scheduling is unchanged.

### Alias equivalence remains outside V1

Different normalized discovery links remain distinct logical source items
unless an existing conflict rule independently blocks admission. V1 claims no
automatic alias equivalence or global deduplication across aliases. Body hash,
date, title, HTML canonical and final redirect destination cannot merge them;
alias equivalence needs a separately frozen policy.

A discovery link changing from U1 to U2 differs from U1's redirect target
changing. The existing item keeps its stored U1-derived identity; a newly
discovered U2 follows existing conflict/dedupe rules. There is no identity
rewrite or "latest URL wins" rule.

Revision 5 fixed the authority of the normalization input; revision 6 froze
HTTPS scheme admission and output casing; revision 7 made port admission and
default-port omission explicit. Revision 8 adds only malformed-percent syntax
rejection. All existing URL transforms and component restrictions are preserved,
including query rejection, fragment removal, userinfo rejection, hostname
lowercasing, scheme/port canonicalization and existing path semantics.
Trailing-dot hosts, IDNA/punycode, IP literals, backslashes, dot segments,
duplicate/trailing slashes and valid percent-escape equivalence are not changed
by this fix. The remaining independent counter-review is not performed here.
Redirect/final/HTML-canonical identity roles, raw/decoding/size rules, timeouts
and scheduling are preserved. Full independent counter-review remains pending;
no runtime or `safe_http.py` change is made here.

## 6. Revisions, without assuming the Fed never edits

Upstream correction semantics are **UNKNOWN**, and V1 never converts that into
an immutability assumption. The client side does the work instead:

```
same item, same body hash        → no duplicate revision
same item, different body hash   → append a NEW immutable revision
overwrite                        → forbidden, raw and normalized alike
```

For fixed provider P and event family E, FOMC26 freezes this history:

| observation | canonical URL | body hash | official date | logical item | content revision |
|---|---|---|---|---|---|
| O1 | U | H1 | D1 | S | R1 |
| later O2 | U | H2 | D2 | S | R2 appended |

Here H1 differs from H2 and D1 from D2. R1 retains D1 intact; R2 records D2.
Both are revisions of S. Downstream must not count the corrected date as a
second independent logical FOMC event.

For identical verified persisted primary body bytes under the **same
CaptureSpec hash**, the parsed date must be identical. A conflicting date is a
parser/normalizer determinism failure: fail closed, create no second content
revision and never silently mutate normalized state. Existing raw and
immutable revisions remain intact; this diagnostic adds no global health state.

Each revision still has `source_available_at = null`, `source_updated_at = null`
and availability carried per observation link (`observed_at` and server-attested `ingested_at`). Identity repair
never rewrites an existing revision's `observed_at`, `ingested_at` or
`declared_release_at`. D2 is metadata, not proof of when a correction occurred
or when its bytes became public. FIX1 visibility and FIX2 scope/zero semantics
remain unchanged.

### Raw integrity is not content identity (revision 24)

The first real pilot showed that a statement page is not byte-stable: Cloudflare
re-encodes the obfuscated e-mail addresses with a new key on every response and
injects a challenge script whose ray and timestamp change every time. Keyed by
the raw body hash, every recheck of an unchanged statement became a new revision.

Revision 24 keeps `raw_sha256` as the integrity of the bytes received — storage,
immutability, corruption handling and replay all still use it, and every
response keeps its own record, raw and observation link. A revision is now keyed
by a separate **content identity** (`FOMC_CONTENT_IDENTITY_V1`): the SHA-256 of
canonical bytes produced by `FOMC_CANON_V1`, which touches exactly three spans.
The two e-mail obfuscation forms are *re-keyed*, not removed: the hidden address
(`media@frb.gov`, or the share link's body) stays in the identity, only the
per-response key goes. The challenge parameters `r` and `t` are emptied, but
only inside the exact challenge script, recognized by its digest and its place
just before `</body>`. Every other byte is kept. Since revision 25 the spans count
only as real HTML tokens at their place in the original bytes — an `<a>`
start tag's `href` value, a `<span>` start tag with its exact content, the
content of the challenge `<script>` — so a marker in a comment, in another
attribute or inside a `<textarea>` is **refused**, as are an ambiguous
structure and invalid UTF-8. A refused record falls back to its raw bytes in a
separate RAW_FALLBACK identity domain: identical refused raws still meet, but a
refused raw never shares a revision with a canonicalized document. Admission still parses the original
document, and a read that relies on a content identity re-verifies the raws it
comes from.

### The recheck anchor is exactly the original primary observed_at

FIX16 replaces the ambiguous phrase "first successful primary statement fetch"
with a direct canonical field binding:

```
revision_policy.reobservation_anchor_timestamp_field = observed_at
revision_policy.reobservation_anchor.timestamp_field = observed_at
source_event = FIRST_QUALIFYING_DURABLE_PRIMARY_CONTENT_OBSERVATION
```

For each existing logical FOMC source item, the original anchor `A` is exactly
the persisted `observed_at` of the first qualifying durable primary content
observation. `observed_at` keeps its existing meaning: actual successful receipt
of the **complete admitted primary response body**. No timestamp alias or
choice of another capture milestone is permitted.

An observation qualifies only when all four conditions hold: final admitted HTTP
200 under FIX12, complete admitted body under FIX8/FIX12, its canonical
`observed_at`, and the existing durable raw/SourceObservation persistence
boundary. A consumed request-start unit, feed observation, non-admitted response
or durable failure diagnostic is insufficient. This introduces no new persistence
primitive or EventStore schema.

**Anchor value and anchor durability are distinct.** Durability makes the
supporting body and observation provenance available for deterministic
reconstruction; it preserves the earlier `observed_at` and its association with
the logical source item. Raw commit time never replaces that value.

| milestone | example UTC time | role |
|---|---|---|
| request start | 18:00:00 | not anchor |
| response headers | 18:00:02 | not anchor |
| complete admitted body received / `observed_at` | 18:00:05 | **anchor value A** |
| qualifying raw/observation becomes durable | 18:00:06 | anchor becomes persistently available/reconstructible; A stays 18:00:05 |
| normalized EventRevision commit | 18:00:07 | not required to establish anchor; A stays 18:00:05 |

`ingested_at` is the server-attested availability of the commit. It has no anchor
authority even if it happens to equal `observed_at`. Neither `declared_release_at`
nor restart time nor the monotonic limiter-start timestamp can replace A. A
declared release at 14:00 followed by the first qualifying primary observation
at 18:00 yields anchor 18:00, with no retroprojection.

### Qualification survives downstream failure only after raw durability

If a complete primary body is observed at 18:00:05 but a crash occurs before the
required raw/SourceObservation durability boundary, that volatile observation
leaves **no reconstructible anchor**. Do not fabricate its 18:00:05 timestamp or
an obligation from memory/log inference. A later qualifying durable observation
may establish the first persistent anchor using its own `observed_at`.

If that observation becomes durable at 18:00:06, a crash before normalized commit
leaves anchor **18:00:05 reconstructible from the durable record**. Do not shift
it to raw commit time, restart time or a future normalization commit.

Semantic parse success and normalized EventRevision commit are **not required**
for anchor qualification. A later `PARSER_FAILED` or normalization failure does
not erase or move an anchor established by an admitted complete, durably recorded
primary observation. This establishes no normalized event success and bypasses
none of the existing parser, classifier, source-health or zero gates.

A 404, 500, 206, 204, 304, redirect failure, incomplete body or oversized aborted
response cannot establish an anchor. DNS/connect/TLS failure, timeout or
connection reset without qualifying primary content cannot establish one either.
Existing request accounting still counts those admitted attempt starts.

### The original observation fixes all target timestamps

The unchanged offsets are **300, 3600, 86400 and 604800 seconds**, each added
directly to original A. There is no rounding, minute/bucket alignment or chaining
from a previous recheck's actual start or completion.

For A = `2026-09-22T18:00:05Z`, target timestamps are exactly:

```
A + 300     = 2026-09-22T18:05:05Z
A + 3600    = 2026-09-22T19:00:05Z
A + 86400   = 2026-09-23T18:00:05Z
A + 604800  = 2026-09-29T18:00:05Z
```

Timestamps follow the canonical primitive (§4): an `observed_at` of
`18:00:05.250Z` gives a +300 target of `18:05:05.250Z`, never truncated for
scheduler convenience.

A is immutable for the same logical source item. Later H1 → H2 content revisions,
raw/normalized commits and scheduled recheck observations retain their own
timestamps without creating another anchor or resetting the grid. A restart at
A+2 minutes leaves the first target A+300, never restart+300.

The due margin and FIX15 may delay a target A+300 well past it: in `FOMC107` the
first grant for O300 comes at 18:08:54 for A = 18:00:05. A and
the other targets A+3600, A+86400 and A+604800 remain unchanged. The limiter's
monotonic timestamp is for rate-duration comparisons; the schedule anchor is
canonical UTC `observed_at`. The entire REV16 `rate_limiter` contract is
unchanged.

Anchor metadata is operational observation provenance, **not source content-vintage
proof**. FOMC V1 content `source_available_at` stays null. The existing distinction
between observed and ingested times and raw-before-normalized visibility remains.

### A durable anchor entails four logical obligations

FIX17 defines `REOBSERVATION_OBLIGATION_SET_V1`. Once the original qualifying
FIX16 anchor A is durable for logical source item S, exactly these four logical
obligations exist:

```
O(S,A) = {(S,A,300), (S,A,3600), (S,A,86400), (S,A,604800)}
due_at(S,A,offset) = A + offset seconds
obligation_birth_event = DURABLE_REOBSERVATION_ANCHOR_EXISTS
```

No separate scheduler write, registration marker, parsing success or normalized
EventRevision commit gates their logical existence. The durable original anchor
and frozen offsets are the authority. A scheduler row may cache membership or
serve as an execution index; its absence cannot remove an obligation.

`REOBSERVATION_OBLIGATION_KEY_V1` is the ordered semantic tuple
`(source_item_id, original_reobservation_anchor_observed_at, offset_seconds)`.
The item identity is unchanged. A is exclusively that item's original FIX16
anchor, never a later observation or a cache-provided replacement. Existing
canonical timestamp precision is preserved. Physical row IDs or tuple encodings
are implementation choices. Content hash, EventRevision/SourceObservation IDs,
worker/restart IDs and CaptureSpec hash are not additional tuple components.

The four keys belong to one immutable grid across repeated same-body observations,
H1 → H2, parser recovery and restart. Two distinct source items with the same A
have eight distinct obligations. Four counts logical identities across statuses;
it does not mean four pending jobs, four rows or four immediate HTTP requests.

### Crash recovery derives membership from durable anchors

If A=`18:00:05` is durable at `18:00:06`, a crash at `18:00:06.001` before any
scheduler row still leaves all four logical obligations reconstructible. Their
targets remain `18:05:05`, `19:00:05`, next day `18:00:05` and +7d `18:00:05`.
If the observation is lost before anchor durability, no obligation set survives
from it; a later qualifying durable observation may establish the first grid.
Parser failure after qualifying raw durability leaves the four obligations intact
without implying normalized success.

On startup/recovery, every durable anchor governed by V1 must yield its complete
logical set, independently of scheduler rows or normalized revisions. On-demand
derivation or reconciliation/materialization is allowed. Missing, partial,
duplicate or conflicting cache values cannot change the authoritative set:

| materialization | logical membership with durable (S,A) |
|---|---|
| zero rows | all four obligations |
| rows for 300 and 3600 only | all four; 86400 and 604800 still exist |
| four correct rows | the same four |
| a repeated 300 row | one logical 300 member; four total |

Repeated, interrupted or concurrent reconstruction, including retry after lost
materialization-write ACK, yields the same keys. No atomic anchor-plus-four-rows
transaction or particular storage primitive is required. Rows without supporting
durable anchor provenance cannot create obligations. All four identities remain
derivable even after their due times; their status is derived as described
below.

The cache rule concerns membership, identity and target only. Reconstruction
by itself neither assigns nor resets status, proves satisfaction, nor
authorizes execution. Status comes from durable observation records, as frozen
next. No corruption-repair procedure is specified.

Derivation itself needs no network and creates no observation, revision,
accounting debit, limiter permit or retry permission. Keys and due times are
operational metadata; they do not supply causal timestamps or source-vintage
proof. Existing provenance binds the governing CaptureSpec semantics; future
implementation uses its authoritative final hash. FIX17 adds no runtime migration
or cross-version reconciliation policy and no pre-implementation identity churn.

### An obligation is pending until a real observation satisfies it

Revision 19 (`REOBSERVATION_SCHEDULER_POLICY_V1`) gives each obligation exactly
two logical states, `PENDING` and `SATISFIED`. `PENDING_NOT_DUE` and
`PENDING_DUE` are merely views of `PENDING` against the clock. There is no
`MISSED`, `EXPIRED` or `FAILED` obligation: a failed attempt is a health
outcome, not an obligation status, and passing `due_at` ends nothing.

The status is **derived, never remembered**:

```
(S, A, o) is SATISFIED  ⇔  some qualifying observation R of S has R.observed_at ≥ A + o
```

A qualifying observation is exactly what could have been an anchor: final
exact 200, complete admitted primary body, canonical `observed_at`, durable
per-response raw/observation record tied to S. Every admitted response keeps
its own per-response record, even when its bytes match an earlier one, so
satisfaction is keyed to that record — not to the content-derived observation
id, and not to an `EventRevision`.

Everything else follows from that one rule:

| situation | result |
|---|---|
| collector offline across O300's due time | still required; eligible after restart; the eventual observation keeps its real `observed_at` — nothing backdated |
| O300, O3600, O86400 all overdue | **one** real current fetch satisfies all three; O604800 waits for its own `due_at`; no catch-up GETs, no claim about the past |
| observation at A+4000 | satisfies O300 and O3600, not O86400 |
| observation at 18:30, O3600 due 19:00 | satisfies nothing for O3600 — no early credit |
| O300 due 18:05:05, `NOW_LB` passes due + 92 s at the 18:08:15 feed record, grant 18:08:24 | satisfied at A+500; lateness is not failure; no target moves |
| same bytes H1 again | satisfied; no new revision |
| changed bytes H2 | satisfied at raw durability; the new revision follows downstream |
| durable body, then parser failure | satisfied; health `PARSER_FAILED`; no normalized success |
| DNS/TCP/TLS failure, timeout, redirect failure, non-200, truncation, oversize | still `PENDING` |

Because offsets increase, the satisfied set is always a prefix of the grid. The
anchor observation itself (`observed_at = A`) satisfies nothing. Comparisons use
parsed canonical instants, so `18:05:05+00:00` does not satisfy a target of
`18:05:05.250000+00:00`. Any exported "satisfied" status carries the satisfying
observation and its `observed_at`, so lateness is always visible. Coalesced
satisfaction therefore never reads as "checked at +5 min".

Only a **LIVE** observation can anchor the grid. A backfill fetch never creates
the four rechecks: fetching a 2021 page today cannot tell what it said five
minutes after release.

**Every response checks the clock against its own server time.** A final 200
counts only if its strictly parsed `Date` (IMF-fixdate, exactly one line, no leap
second), plus a valid `Age` (at most one line, at most 86400 s), agrees with the
local wall clock within 90 s. Otherwise the response is `CLOCK_UNVERIFIED`. It is
persisted **and processed like any other response**, but it can never anchor,
satisfy a recheck, start a cycle, conclude an item or serve as availability
evidence. Only the final response's `Date`
counts, never a redirect's.

There is no trust state, bootstrap or baseline. A bogus clock step or a broken
header affects only the responses it touches, and a corrected clock verifies
again at once.

**Capture never stalls on clock uncertainty; causal visibility waits instead.**
The host must be NTP-disciplined, but that is defense in depth, not proof. Two
orthogonal predicates (`CAUSAL_PREDICATES_V2`) carry every clock rule:

- **`PROCESSABLE`**, fixed at commit: an admitted, complete final 200 whose raw
  bytes and digest committed. Every such record is parsed, classified and
  processed to a final local outcome, whatever the clock says. Corruption found
  later never changes it; it produces a diagnostic and a snapshot barrier.
- **`LIVE_ELIGIBLE`**: `PROCESSABLE`, LIVE mode, and verified by **its own**
  server `Date`/`Age`. Only it can anchor, start a cycle, satisfy a recheck,
  succeed an episode, conclude an item or prove LIVE availability.

No record inherits clock trust from another (`CLOCK_REFERENCE_MAX_AGE` = 0), so
no local elapsed clock enters a causal rule. That matters on this kind of host: a
VM or WSL2 pause freezes the wall, the monotonic and the suspend-aware clocks
alike, so none of them can prove elapsed time.

**One tolerance, derived once.** `Date` and `Age` are whole seconds, so a
response's true serve instant lies in [`Date`+`Age`, `Date`+`Age`+2). The check
allows 90 s between that and `observed_at`. So a verified `observed_at` is within
**92 s** of the true serve instant. That 92 is the only tolerance any causal rule
uses:

- **Availability.** A transaction becomes available at the `observed_at` + 92 s
  of the first verified response (by commit order) whose request began after it
  committed, and
  never earlier than its own `observed_at` or any earlier availability. That
  request was sent after the commit was durable, so this bound holds through
  suspends, pauses, slow commits and wall steps. Until that response arrives, the
  transaction is unresolved and visibility waits: with an accurate clock, about
  150–250 s after commit.
- **Server time.** `NOW_LB` is the largest verified `observed_at` − 92 s. It is
  never later than the true present.
- **Due and backoff** use `NOW_LB` only (see retrying).

Nothing is ever reordered or re-stamped. `ingested_at` is the derived
availability, and the binds declare this override of the design's `ingested_at`
assignment. The stated residual: a cache that serves a stale `Date` without
`Age` to a collector whose clock is equally slow can pass the check.

### Retrying, with durable episodes

Revision 21 (`BOUNDED_RETRY_EPISODE_V2`) gives every unit of autonomous work
exactly one durable **episode**, keyed without any clock value:

| work | episode key |
|---|---|
| first LIVE fetch of a new item | the acquisition |
| each recheck | `(item, anchor, offset)`, one per offset |
| each backfill entry | `(manifest, item)` |
| operator retry | the operator record |

An episode is opened by a durable record. It has 6 attempts, with backoffs of
60 / 300 / 900 / 3600 / 14400 s, and it ends only with a durable
`EPISODE_SUCCEEDED` or `EPISODE_SUSPENDED` record, which is final. An attempt is
counted when its `TRANSPORT_INVOKED` record commits. That is the last step before
the first transport call, after ownership, validation and the limiter grant.
**Once durable, it always counts**, even if the process dies before sending
anything. Selection, limiter waiting, ownership failure, validation and local
work count nothing.
Redirect hops belong to the same attempt.

Clock steps, restarts, relisting and recovery can neither open, close nor reopen
an episode. For each attempt, the first committed outcome is authoritative.
Rechecks of one item open one episode at a time, with the smallest due offset
first. A new episode never fires within 60 s of the item's last failure.
Transport must start within 1 s of the limiter grant. If `TRANSPORT_INVOKED` has
not committed by then, the grant is abandoned: nothing is sent and the work
waits 60 s. If that `TRANSPORT_INVOKED` still commits later, it counts. This 1 s
bound is what FIX15's "immediately initiate" means here: committing
`TRANSPORT_INVOKED` is the only step allowed between the grant and the transport
call, and an abandoned grant can never be kept or revived. The rule is conservative, because an attempt can
be counted without traffic but never the reverse. A suspended recheck stays suspended; a later offset's own episode
may still satisfy it by coalescing. A LIVE item therefore has at most five
autonomous episodes, one acquisition plus four offsets, giving an **absolute
ceiling of 5 × 6 × 4 = 120** physical requests. Backfill runs and operator
retries are counted separately.

**Timing uses server time only.**
- **Due.** A recheck is due when `NOW_LB` ≥ its due time + 92 s. Any verified
  response fetched after that point is late enough to satisfy it, so a wall-clock
  jump can never burn an episode early.
- **Backoff.** The next attempt waits until `NOW_LB` ≥ the failed outcome's
  availability + the backoff. Both sides are server-attested, so the rule reads
  the same after a restart, a new boot or a suspend. No local deadline or
  elapsed-time credit is persisted.
- **No verified responses.** Due and backoff decisions hold, so waiting never
  burns work; an attempt already eligible when a clock fault begins may still be
  spent unverified. Raw capture, feed polling and local processing continue.

An attempt counts only as satisfying or not; the episode suspends after its
sixth counted non-satisfying attempt.
Recovery writes that suspension itself if a crash interrupted it, so no episode
can stay open with nothing left to try.
The `TRANSPORT_INVOKED` commit itself enforces the budget. It commits only with
the current epoch's fencing token, an open episode, fewer than six attempts, and
no other attempt of the item still in flight. Episode keys are unique at commit.
The current limiter-epoch owner marks every older attempt without an outcome as
`INTERRUPTED`, and every own attempt whose task ended or that has no outcome
600 s after `TRANSPORT_INVOKED`. An attempt whose 120 s admission bound passed
without an outcome gets `LOCAL_PERSISTENCE_FAILED` instead. These commits happen
at the first trigger at which the store can commit. All of these still count.

**The 120 s bound is an admission bound** (`LOCAL_SAVE_ADMISSION_BOUND_V1`,
revision 23). A response record is admitted only by a check made inside its own
transaction, under the store's write lock and before any of its rows is
inserted: fewer than 120 s since the network phase ended — *at 120 s, expired*.
A check that fails admits nothing, neither a response nor `LATE_EVIDENCE`.

What the bound does **not** cover is how long the `COMMIT` that follows the
check takes. A `COMMIT` cannot be bounded or cancelled once begun, and every
durable decision is itself a `COMMIT`, so revision 23 abandons the reading of
revision 22 under which nothing could become durable after +120 s. An admitted
record that becomes durable late is a valid record and, if it commits first, its
attempt's outcome. Nothing is rewritten to hide the delay: `observed_at` stays
the receipt reading, and availability still comes from the first verified
response requested after the record committed — so a record that was durable
late is only available later, never earlier.

**A stalled store is a storage incident, never a decision**
(`STORAGE_INCIDENT_V1`). A store operation — a transaction including its
`COMMIT` and `fsync`, or a raw-body write — in progress for 10 s is an incident.
It is detected from in-process markers, without the store. It raises an
operational alert, and no FIX15 grant, initial or continuation, is attributed
while it lasts; requests already granted run to their own 60 s deadline. No
durable decision is promised while the store cannot write. When it writes again
the owner records the incident and reconciles from durable state: the first
committed outcome of each attempt stands, there is no repair request, no new
key and no refunded attempt.

**An attempt has exactly one outcome** (`ATTEMPT_OUTCOME_FENCE_V2`). A response
record is its attempt's outcome only while the attempt has none. A response that
arrives after its attempt was already marked, for example `INTERRUPTED`, is still
stored and processed, but as **`LATE_EVIDENCE`**, which every rule treats as
clock-unverified:
- **It can add fail-safe state.** It may create candidates, markers and
  diagnostics, and it may reopen an item.
- **It can never act causally.** It never concludes an item, anchors, satisfies
  a recheck, starts a cycle, makes content current or serves as a clock
  reference.

The attempt gets no further redirect grants. Attempt outcomes and eligible
records therefore commit in request order, and a late stale response can add
caution but never a conclusion. The bound assumes a durable store with
fsync-durable commits; a restored, re-initialized, replaced or forked store is
outside V1 and invalidates it. Nothing stays in flight. Bytes lost after
arrival count as one attempt, as a crash does. While the store cannot commit, no
attempt can start at all.

**Local processing is derived work.** Any committed response without a final
processing outcome, and any feed record without a cycle conclusion, is required
local work. It is reconciled at startup, on a **periodic tick of at most 60 s**, after
every per-response commit (including a late record from an old epoch), and
before every request. Only the current
epoch owner processes, with at most one run per record. Each run is `RUNNING`
with a 600 s deadline until it is terminal, or `DEAD` (epoch lost, task ended,
deadline passed, or four failed local commits). Outcome commits are fenced to the run that is `RUNNING`; the one exception is
the poison outcome below, which the current owner commits itself and re-derives
until committed. A record with two
`DEAD` runs
ends as `INTERNAL_PROCESSING_ERROR`: fail closed, outstanding, no crash loop, and
a feed record still gets its (not-zero) cycle conclusion. Healthy running work
never counts toward that limit, and local failures never refetch. No new feed
poll starts until every earlier feed record is terminal (classified, or ended in
a terminal failure) and every cycle-starting one has its conclusion. A feed response that is not
`LIVE_ELIGIBLE` ends its poll attempt with a `CLOCK_INELIGIBLE` diagnostic and no
cycle, and polling continues.

**Operator retries close.** A `MANUAL_RETRY` episode succeeds as soon as its work
is satisfied by any `LIVE_ELIGIBLE` observation, before sending anything. An
in-flight manual attempt keeps its observation. Manual episodes never touch
autonomous ones and sit outside the 120 bound.

### What a snapshot shows (`FOMC_CURRENT_CONTENT_SELECTION_V1`)

`events_as_of(T)` and `InformationSnapshot(T)` read FOMC items from the longest
commit-order prefix P(T) whose availability is resolved and ≤ T. An unresolved
transaction ends the prefix. **Every input is read only inside P(T)**: records,
outcomes, revisions, links and diagnostics. An outcome committed after the
prefix is still pending for the read. For each item, the read takes the latest
processable response in P(T) by commit order, of any mode and clock verdict.
- If that response is verified, its content hash selects the revision.
- If it is unverified, and the latest verified response has the same hash, that
  revision is current.
- A corrupt newest response, or any other mismatch, gives a barrier. Unverified
  raw is kept as unresolved source state:

| situation | snapshot state |
|---|---|
| H1 normalized | current = H1 |
| H1 normalized, H2 durable but unnormalized | `CURRENT_CONTENT_UNAVAILABLE_DUE_TO_NEWER_UNNORMALIZED_SOURCE`; H1 is history only |
| H1, then H2 normalized | current = H2 |
| H1 → H2 → H1 | current = the existing H1 revision (no duplicate) |
| clock-unverified H2 after H1 | barrier: newer bytes exist; they are never availability evidence |
| clock-unverified old H1 after verified H2 | barrier, never a return to H1 |
| backfill H2 after a LIVE H1 | unnormalized: barrier; normalized: current = H2, but not LIVE-available; LIVE zero unaffected |
| observed item with no revision (outstanding, backfill parse failure) | `SOURCE_ACTIVITY_NO_REVISION`, never absence |
| concluded out of scope | `NOT_IN_V1_SCOPE` |
| revision whose only link is unverified | barrier; current needs a verified link |
| newest record corrupted after commit | barrier, never the older revision |

Revisions are keyed by content and are **mode-neutral**. Each IN_SCOPE observation
links to its revision with its own mode and availability. Backfill H1 followed
by LIVE H1 is one revision with two links. LIVE availability starts at the LIVE
link, and strict causal research requires a LIVE link. Backfill never pre-empts
LIVE discovery, anchoring or availability.

Every read is evaluated over one store view whose **horizon H** (its largest
commit sequence) is stored with the snapshot. Availability, the prefix and every
input use only records up to H. So replaying at the same (T, H) reproduces the
live read exactly, including an unresolved one. The selection policy, H and each
FOMC item's selection state are bound into snapshot identity, so a store where V1 is current and one where a barrier sits
over V1 never share an identity. Every read also carries one FOMC-level state, bound into identity, so an unresolved read is never an empty resolved one. The FOMC part of a live snapshot at `T` is
admissible only when both hold:

- a transaction already has resolved availability later than `T`;
- nothing unresolved precedes that transaction.

Otherwise the FOMC part is `FOMC_CAUSAL_VISIBILITY_UNRESOLVED` as a whole: no
item state, zero conclusion or FOMC health is exposed. Other providers' content
is unaffected.

**Zero is only shown for the newest feed record.** A resolved read shows
discovery from the latest processable LIVE feed record in the prefix:
- **That record's own cycle conclusion.** ZERO or not zero, if the conclusion is
  inside the prefix.
- **`NO_CURRENT_CYCLE`.** When that record starts no cycle, for example because
  its clock check failed.
- **`DISCOVERY_PENDING`.** When that record, or its conclusion, is not yet
  inside the prefix.

An older zero is never presented as current. A consumer that requires LIVE
availability gets `CURRENT_REVISION_NOT_LIVE_AVAILABLE` unless the item's latest
`LIVE_ELIGIBLE` record carries the current content. A return to older content
seen only through backfill is never LIVE-available.

### Backfill is a manifest, not a crawl

The manifest is UTF-8 JSON without BOM: `{"version": 1, "urls": [...]}`, with at
most 1000 **raw** entries counted before dedupe. It is validated whole before
any request: one bad entry rejects everything, with zero requests. Its hash,
length, counts, canonical work list and operator provenance are stored before
the first fetch. Replay never reads an external file.

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
conservative client policy. Revision 14 named logical operations without
defining a counted unit: a fetch with three redirect follows could be counted
as either one request or four. **FIX14 freezes the physical unit and its shared
provider domain.**

### One authorized physical attempt consumes one unit

The canonical unit is **`PHYSICAL_REQUEST_ATTEMPT_START`**. Exactly one unit is
consumed when the FOMC V1 client is granted permission to begin one concrete
outbound HTTP attempt for one already validated request URL. Permission and
accounting occur **before any network side effect** of that attempt.

Once permission to begin is granted, the unit is consumed even if DNS, TCP,
connect, TLS, read or other transport processing fails, the connection resets,
the attempt times out, or the final response is non-200. **Failure never refunds
the unit.** A feed poll cancelled before network authorization consumes zero.

**A logical operation is not the accounting unit.** Feed polls, primary fetches,
scheduled rechecks, historical retrievals and redirecting logical fetches may
each cause one or more physical attempts. Each actual feed GET, primary GET,
redirect-follow GET, permitted later reattempt, scheduled recheck GET and
historical provider GET consumes one unit. Feed discovery followed by one
primary attempt therefore consumes at least two units.

DNS, TCP, TLS, HTTP request write and response processing within one admitted
attempt are transport substages, not additional requests. DNS + TCP + TLS + one
GET consumes one unit total; this rule defines no substage timeout semantics.

### One provider-wide domain, including every redirect hop

`accounting_scope = FEDERAL_RESERVE_PROVIDER_V1_GLOBAL`. All actual allowed
attempts under this CaptureSpec V1 provider client share **one logical limiter
and accounting gate**. Independent feed, primary, recheck, redirect, reattempt
or historical quota buckets are forbidden. Category-specific queues may exist
only if their aggregate physical starts pass through that same provider gate.

Any historical Fed GET executed under this same provider client/network policy
shares the domain with live traffic, one unit per physical attempt. There is no
separate live-plus-historical capacity. Unrelated external research tools are
outside this runtime scope. The host allowlist remains exactly
`www.federalreserve.gov`; no URL or host admission changes.

Every followed redirect target must first pass existing FIX13 URL validation,
then independently reacquire provider permission before its network attempt:

```
GET U0 → 302 → GET U1 → 301 → GET U2 → 200     3 units
GET U0 → GET U1 → GET U2 → GET U3              4 units
U3 proposes redirect #4 to U4                 U4 never requested; no fifth unit
U0 returns a rejected Location               U0 = 1; rejected target = 0 units
```

A foreign host, bad scheme, forbidden query or malformed percent escape rejected
before contact creates no target attempt and no additional unit. The request
that returned that redirect already consumed its own unit. The same zero applies
to the fourth redirect target that FIX12 forbids following.

**No intra-fetch exemption:** a logical fetch reserves no multi-request corridor.
U1 and U2 must each reacquire the same gate; HTTP library auto-follow cannot
bypass it. Existing retry semantics remain authoritative: no hidden library
retries, no tight loops, and later attempts only via the next normal discovery
cycle or an allowed scheduled pending-item retry under the same ceiling. Any
legitimately permitted later attempt consumes a new unit. A library must expose
each such attempt and pass it through the gate; accounting alone does not
authorize a retry. No retry is free because it belongs to an earlier fetch.

Every scheduled recheck attempt consumes one unit in this domain. A due recheck
cannot bypass unavailable limiter admission. FIX16 binds its anchor to the exact
**`observed_at` of the first qualifying durable primary content observation** in
§6, with unchanged +5 min, +1 h, +24 h and +7 d offsets. A limiter delay changes
neither the anchor nor those targets; late execution is frozen by revision 19
in §6.

### Exact rolling window and start-to-start spacing

Revision 15 left the temporal contract unresolved. FIX15 binds it in canonical
`rate_limiter`, while preserving FIX14's unit, shared domain and no-refund rule.
The algorithm is **`ROLLING_PHYSICAL_START_WINDOW_V1`**, with a 60-second window
and at most six physical starts. Fixed UTC/calendar-minute buckets, token
buckets, leaky buckets and calendar-minute resets are forbidden.

For candidate monotonic instant `t`, a prior admitted start `s` counts exactly
when **`0 < t - s < 60`**, in seconds. Thus prior history is restricted to
`(t-60s,t)`. The candidate is admitted by the window test only if:

```
count({s in history | 0 < t - s < 60}) <= 5
```

After adding the candidate, **for every admitted start `t`, `(t-60s,t]` contains
at most six admitted physical starts**. A prior start of age exactly 60 seconds
is excluded. Do not round a candidate across a threshold or add an early-start
tolerance.

```
prior starts: 0, 10, 20, 30, 40, 50
candidate 60:     active prior = 10, 20, 30, 40, 50; count 5; spacing 10 → ADMITTED
candidate 59.999: active prior = 0, 10, 20, 30, 40, 50; count 6; spacing 9.999 → WAIT
```

Unless an example explicitly uses epoch elapsed `e`, its `t=0` is an arbitrary
origin **after the current epoch's cold-start embargo**, with exclusive provider
ownership already established. The valid sequence `0,10,20,30,40,50,60` does not
waive the startup rule below.

Spacing is **`START_TO_START`**. With `p` the immediately previous admitted
provider start, admission requires **`t - p >= 10` seconds**. Exact equality
passes. No previous start in the current epoch means the spacing test passes;
ownership and embargo still apply. Completion time has no role: if A starts at
0 and completes at 45, B may start at 45 when the window permits. There is no
completion-plus-10 wait to 55.

### One event for accounting, window and spacing

`PHYSICAL_REQUEST_ATTEMPT_START` is the **atomic provider-limiter grant-and-consume
event immediately before transport invocation** for one validated request. It
occurs after URL/request validation and waiting for rate permission, and before
DNS, TCP, TLS or HTTP side effects. Accounting, window history and spacing use
this same event and timestamp. Socket write and response completion cannot
supply separate rate or spacing timestamps.

A permit cannot be reserved, stored, transferred, reused or banked for later
execution. Grant and consumption belong to one immediate attempt initiation.
If the task cannot proceed after grant, its authorization is abandoned and its
unit remains consumed; any future attempt needs fresh admission. Rescheduling
or a new epoch cannot revive a previous grant.

Cancellation after grant but before network I/O still consumes one unit. Its
`t` remains in window and spacing history under the normal age rules. There is
no refund and no successful observation implied. Conversely, failing admission
creates no start, consumes no unit and causes no network side effect.

### Monotonic, atomic admission across all provider emitters

Within one limiter epoch, rate decisions use **`MONOTONIC_ELAPSED_TIME`** in one
comparable domain. UTC corrections, NTP jumps, manual clock changes, timezone
and DST have no effect. Last start at monotonic 100 and candidate at 105 still
fail spacing after a five-minute backward UTC jump. Normal UTC request and
observation provenance remains preserved; monotonic rate timestamps never
become `observed_at`, `ingested_at`, `declared_release_at` or `source_available_at`.

All concurrent emitters must share an atomic admission decision. After checking
ownership and embargo, the limiter atomically inspects current history, tests
the rolling window and spacing, and, if both pass, grants, consumes and registers
`t` before another candidate can be granted. Only then may transport begin.

Two workers cannot both read five prior starts and receive permission from that
same stale state. One may be granted first; the other must re-evaluate updated
history. At the same instant its spacing delta is zero and it cannot start.
Although an equal-timestamp prior start is excluded by the strict prior-window
predicate, the mandatory spacing test rejects that second grant.

These semantics apply across processes as well as categories. An implementation
may centralize issuance or coordinate processes, but cannot use independent
process-local budgets. No mutex, async lock, database, file lock or IPC primitive
is prescribed. The first worker is not specified; queue ordering and fairness
remain outside FIX15.

### Every new limiter epoch begins with a 60-second embargo

**`LIMITER_EPOCH`** is one continuous provider runtime epoch with available,
comparable monotonic start history. Prior-epoch monotonic history is not reused
or reconstructed for admission, and durable per-request limiter history is not
required. Safety instead requires exclusive provider ownership plus a full
**60-second cold-start embargo**, measured in the new epoch's monotonic domain.

The new admission epoch begins only after exclusive provider ownership is
established and previous independent issuers can no longer initiate provider
attempts. Time spent waiting alongside an independently active old issuer cannot
count toward the new epoch's embargo. If exclusivity cannot be established, no
new provider network attempt may start; no source-health enum is introduced.

For every fresh epoch, the first admitted provider start must satisfy:

```
elapsed_since_limiter_epoch_begin >= 60 seconds
e=2 or e=59.999   → WAIT; no provider start
e=60.000         → embargo passes; then evaluate normal window and spacing
```

This applies to initial startup, crash recovery, clean restart and limiter
recreation whenever a fresh non-comparable epoch begins. Neither UTC history,
persisted request times nor apparent downtime may shorten the embargo. A new
process cannot simply wait 60 seconds while the old independent issuer remains
active and then open another budget. Ownership must exclude overlapping
independent emission epochs.

After the embargo, current-epoch history is empty before the first admitted
start, then updated by each subsequent admission. Exclusive handoff ensures all
unknown prior-epoch starts are at least 60 seconds old when the new epoch first
admits: they are outside the strict-left window, and the 10-second spacing is
also satisfied. No comparison between monotonic epochs is needed.

### All categories wait for the same limiter

Feed, primary, redirect follow, permitted later reattempt, scheduled recheck and
historical provider capture obey the same window, spacing, clock, atomic grant
and epoch rules. Feed start 100 followed by primary candidate 101 must wait;
earliest spacing eligibility is 110. Redirect U0 start 200, 302 received at 201,
follow candidate 201 must wait until at least 210. Both remain subject to the
rolling window and ownership/epoch gates.

Waiting for a full window, spacing or embargo is pending network work. It is
not by itself `SOURCE_UNAVAILABLE`, `PARSER_FAILED` or `EVENTS_OBSERVED_ZERO`.
No busy loop or new source-health state is required.

For pending work under exclusive ownership, the earliest legal start is
constrained together by epoch begin +60 seconds, previous start +10 seconds
when present, and the first instant with at most five active prior starts under
`0 < t - s < 60`. Conditions are evaluated against current state at admission;
eligibility grants neither a reservation nor a queue-service guarantee.

### Which work gets the next grant

FIX15 decides **when** a start may happen; `ELIGIBLE_CLASS_ALTERNATION_V1`
decides **which** work takes it:

| class | work |
|---|---|
| `FEED_DISCOVERY` | the LIVE feed poll; LIVE acquisition episodes |
| `REOBSERVATION` | recheck episodes |
| `HISTORICAL_BACKFILL` | manifest entry episodes |

At each grant, redirect hops of an in-flight attempt go first. Otherwise the next
class after the class of the latest durable `TRANSPORT_INVOKED` record is served,
skipping empty classes. Inside `FEED_DISCOVERY` the poll and acquisitions
alternate the same durable way. Within a class, episodes are ordered by
`(next eligible instant, source_item_id)`. Suspended episodes are never eligible.
At most one fetch per
item is in flight across all classes. Scheduling is work-conserving.

Logical fetches of different items and the feed poll run concurrently; there is
no other concurrency cap. One dispatcher attributes every grant at the instant
FIX15 would admit a start (`selection.grant_dispatch`, revision 23): a validated
redirect hop of the in-flight attempt with the smallest `TRANSPORT_INVOKED`
first, otherwise one selection decision whose `TRANSPORT_INVOKED` commits before
the next decision. Work is selected only when its grant can be attributed at
once, so nothing is selected twice and no grant is consumed for a
`TRANSPORT_INVOKED` that does not commit.

Provider requests ignore ambient proxy settings (`HTTPS_PROXY` and similar).

### Accounting does not redefine observations or source health

An accounting unit is not a `SourceObservation` row. An attempt can fail before
any successful response; consuming its unit implies no successful observation
or `EventRevision`. Their existing semantics remain unchanged.

For a successful content response, `observed_at` retains actual successful
response-observation semantics; for a revision it is receipt of the primary
statement raw bytes used for that revision. The accounting start event becomes
none of `observed_at`, `source_available_at`, `ingested_at` or
`declared_release_at`. Merely waiting for future limiter permission introduces
no failure state. An actual attempted request that later fails uses the existing
failure mapping; no health enum is added.

Polling is plain elapsed-time. V1 derives no schedule from the FOMC calendar,
the 2:00 p.m. tradition or `pubDate` — a predictive scheduler would be a second
source of truth about time, which is the one thing this design refuses.

## 7a. What a response must be before it counts

Revision 12 recorded the HTTP status as provenance and never said which status
admits a body. The raw-first order is status-agnostic —

```
fetch bytes → durable raw + hash → parse/normalize → durable revision → snapshot-eligible
```

— so a 404 whose body happened to be a well-formed statement page could be read
as something to hash, persist and parse. Two conforming implementations would
then disagree about whether an error page becomes an event.

```
admitted  ⟺  final_status_code == 200        (exact equality)
```

Not any 2xx. Not `status < 400`. Not "a body is present", and emphatically not
"the body looks valid". **Body appearance never overrides status.** Every other
final status is `SOURCE_UNAVAILABLE`: no parser input, no EventRevision, no clean
zero.

A 200 is a gate, not a verdict — the body still has to pass the decoded-size
bound, durable raw and hash, UTF-8, XML/HTML security, the anchors and the
classifier, and commit. Nothing is bypassed.

### Three statuses that look admissible and are not

| | policy | why it matters |
|---|---|---|
| `206` | UNSUPPORTED; Range and If-Range **forbidden** | arrives *complete at the transport layer* while being semantically partial |
| `204` | UNSUPPORTED | an empty body must not become a valid content hash |
| `304` | UNSUPPORTED; prior-body reuse **forbidden** | avoids introducing a second cache/replay semantics into V1 |

The `206` case is the sharp one. FIX8 already forbade treating an incomplete
prefix as a complete body — but that covered truncation the **client** causes by
aborting. A `206` is incompleteness the **server** declares, and it would have
walked through the door FIX8 thought it had closed. It is now shut from both
sides.

`304` is refused even when a verified prior body exists locally. That body stays
historical local data; it is not substituted into this HTTP observation, and no
revision arises from a 304. V1 also never *sends* `If-None-Match` or
`If-Modified-Since`, so a compliant run should never see one.

`ETag`, `Last-Modified` and HTTP `Date` may be kept as provenance and may never
become `source_available_at`, `source_updated_at`, `declared_release_at`,
`observed_at` or `ingested_at` — the same prohibition FIX1 froze for `Last
Update`, arriving by a different route.

### Redirects, counted exactly

```
allowed statuses            301 302 303 307 308  — and only these
max followed transitions    3
Location                    REQUIRED; never guessed or reconstructed
every target                security-validated BEFORE the request
fourth redirect target      NEVER requested
redirect bodies             non-semantic
identity effect             NONE
```

"Max 3" means three *followed transitions* after the initial request. If the
response after following redirect #3 is itself a redirect, that is #4 — its
target is not requested and the fetch ends `SOURCE_UNAVAILABLE`. A loop
terminates on that count at the latest. Every hop stays a GET.

A final 200 reached through a valid chain enters the pipeline normally, and
identity is still the FIX4 authority: the official discovery link **before**
redirects. Redirects never re-key `source_item_id`.

### Where a redirect target comes from

FIX12 required a `Location`, validated every target before contacting it and
bounded the transitions. It never said how a `Location` *value* becomes the
absolute URL being validated — and since the URL rules require HTTPS, that gap had
teeth:

```
GET  …/monetary20260916a.htm
301  Location: /newsevents/pressreleases/monetary20260916b.htm   ← the ordinary case
```

One reading validates that string as-is: no scheme, HTTPS check fails, item
permanently uncapturable. The other resolves it against the current URL, as every
HTTP library does, and follows it. Both conformed.

V1 now supports the full set of reference forms:

| form | example | supported |
|---|---|---|
| absolute URI | `https://host/foo.htm` | yes |
| absolute-path | `/a/b.htm` | yes |
| relative-path | `b.htm` | yes |
| scheme-relative | `//host/foo.htm` | yes — inherits `https` |
| fragment-only | `#section` | yes — still a transition |
| query-only | `?x=1` | resolvable, then refused by the query rule |

```
1. read Location
2. trim outer HTTP OWS — SP and HTAB only
3. reject empty
4. parse as a URI-reference
5. resolve against the CURRENT hop's actual request URL   (RFC 3986 semantics)
6. apply the existing V1 URL security and canonicalization rules
7. apply the existing fragment non-request rule
8. request only if admitted
```

**The base advances with each hop.** `U0` → `../b/two.htm` → `U1`; then `U1`
returning `three.htm` resolves against `U1`, not `U0`. `FOMC82` pins that down,
because resolving everything against the original URL is the natural bug.

**Resolution earns no trust.** Step 6 is unchanged — a resolved target satisfies
exactly the same constraints as any other URL. `//evil.example/foo` resolves
cleanly to HTTPS and is then refused by the host allowlist *before* the host is
contacted. `http://…` is refused rather than silently upgraded. A query survives
resolution and is then refused by the existing queryless rule — **never stripped**
to make the target admissible. Malformed percent escapes still fail under FIX7,
and resolution may not percent-decode ahead of that gate.

Only the outer `SP`/`HTAB` of the field value are trimmed. Embedded whitespace is
not repaired into `%20`, empty or whitespace-only `Location` is
`SOURCE_UNAVAILABLE` with no "same URL" inference, and no browser-style repair,
URL search or guessed scheme is permitted.

Identity is untouched: resolution derives only the next `request_url` for that
hop. `canonical_primary_statement_url` and `source_item_id` remain the
pre-redirect official discovery link, however many relative hops intervene.

### Raw-first, clarified rather than weakened

For an admitted 200 the durable raw body still precedes any normalized
visibility. For a non-admitted status, transport and status provenance may be
durable, but there is **no admitted complete content body** eligible for semantic
replay or parsing. Raw-first does not mean *parse every HTTP body regardless of
status* — and no successful content hash is ever created from a 204, 206 or 304
payload, an error body, or a redirect body.

FIX12 itself introduced no retry policy. A non-admitted response maps to
`SOURCE_UNAVAILABLE` *if it is the final response of the logical attempt*.
Existing `polling.retry_policy` remains authoritative; revision 19 binds its
timing (§6, §7). FIX14 binds every redirect follow and permitted later
reattempt to the physical unit and shared gate in §7, and FIX15 freezes their
rate admission there, without changing retry permissions or final-response
admission.

### Every physical attempt ends within 60 seconds

A connect timeout of 10 s and a read timeout of 30 s do not bound an attempt: a
server that sends one byte every 29 s never trips either. It never triggers the
body cap within any useful time either. Revision 19 adds
`PHYSICAL_ATTEMPT_TOTAL_DEADLINE_V1`:

```
total_deadline_seconds   60
clock                    monotonic (the limiter's epoch domain)
begins_at                PHYSICAL_REQUEST_ATTEMPT_START — the grant, after all limiter waiting
activity_independent     true — arriving bytes never reset it
covers                   DNS, TCP, TLS, request write, headers, body, content decoding
```

Connect 10 s still bounds each TCP connection attempt. Read 30 s is an
inactivity bound on every blocking socket read or write, including the TLS
handshake and the header wait. The earliest limit wins, and none extends the
total. DNS has no separate limit, but it is inside the 60 s. At expiry the
attempt aborts *logically*: a late DNS answer is discarded, no TCP connection is
started from it and no HTTP byte is sent. What the platform resolver keeps doing
internally afterwards is outside the contract. Local persistence, hashing and parsing come after
the network phase and are outside the deadline.

Each redirect follow is its own physical attempt with its own fresh 60 s,
starting at its own grant. Limiter waiting between hops counts against no
attempt. Resolved addresses are tried one at a time in resolver order — no
parallel racing — inside the same attempt and deadline, from one lookup covering
IPv4 and IPv6. With at most four attempts, a logical fetch is network-active for at
most 240 s.

On expiry the attempt is aborted, and nothing it received is admitted: no
complete-body hash, no parser, no revision, no anchor, no satisfied obligation,
no refunded unit. The surface is `SOURCE_UNAVAILABLE`, zero is blocked, and a
recheck stays pending.

### Redirects are followed by hand

The client must never follow a redirect itself (`MANUAL_REDIRECT_LOOP_V1`). For
every 3xx it reads the status, resolves `Location` under the frozen rules (with no
`URI` fallback and no repair), and validates the target *before* any contact. It
closes the redirect response and its connection without draining the body. It
then takes a new limiter grant, a new connection and a fresh 60 s deadline for
the next hop, up to 3 hops. `require_https_host` is kept as one necessary check
only, because it accepts some URLs V1 rejects. The
transport must enforce connect 10 s and read 30 s separately.

### Content-Type is checked exactly

`CONTENT_TYPE_GATE_V1` runs after raw persistence. The feed accepts
`application/rss+xml`, `application/xml` and `text/xml`; the statement page
accepts `text/html`. Type and parameter names are case-insensitive, whitespace
around them is allowed, and parameter order and non-charset parameters are
ignored after the duplicate check. A trailing `;` is allowed, and quoted
values are unescaped first. Charset must pass the UTF-8 rules. A missing header,
an unparseable value, a duplicated parameter name or two different media types
is `PARSER_FAILED`, with no sniffing.

### Every failure has one health state

Each failure maps to one rev4 state or to no provider health state. When several
conditions apply, this fixed precedence decides. The clock check is **not** a
stage that stops processing: a clock-unverified response keeps its diagnostic,
is still processed, and gets the health state of any later failure:

| precedence | condition | state |
|---|---|---|
| 1 | raw digest mismatch | no provider health state; `CORRUPTION_FAIL_CLOSED` |
| 2 | local persistence failure | no provider health state |
| 3 | DNS, TCP, TLS, timeouts, redirect failure, non-200, truncation, content coding | `SOURCE_UNAVAILABLE` |
| 4 | size, Content-Type, charset, UTF-8, XML security/syntax, feed structure, title/date anchors | `PARSER_FAILED` (a clock-unverified response also keeps its `CLOCK_UNVERIFIED` diagnostic) |
| 5 | response not `LIVE_ELIGIBLE` (fails its own clock check) and nothing above applies | no provider health state; `CLOCK_INELIGIBLE` diagnostic; still processed |
| 6 | everything else, including healthy negatives, conflicts, new items and release-clock diagnostics | no failure state |

A routine non-statement is a successful check, not a parser failure. A recheck
failure is recorded on `primary_statement` only.

## 7b. XML parsing has no expansion path

Revision 11 forbade XML **external** entity resolution. Counter-review R11 showed
that is not the same as being safe: a billion-laughs bomb declares only
**internal** entities and resolves nothing externally, so the clause never fires.
And the FIX8 decoded-body cap does not help either — its own basis says it counts
*"HTTP entity-body bytes … before any charset/text decoding"*, so it is satisfied
and finished before the parser starts. A ~1 KiB feed could pass the size cap,
pass strict UTF-8, be hashed and persisted, and only then expand to gigabytes.

The fix is the smallest one available: **forbid `DOCTYPE`**. That removes the
custom entity declaration machinery entirely, so there is no expansion to bound —
no entity-count, nesting-depth or expanded-byte caps are needed, because nothing
can be declared in the first place.

```
DOCTYPE                        FORBIDDEN  → PARSER_FAILED
internal DTD subset            FORBIDDEN  (follows from the above)
external DTD                   FORBIDDEN  (stated separately)
external general entities      FORBIDDEN
external parameter entities    FORBIDDEN
custom entity declarations     UNSUPPORTED
XInclude                       FORBIDDEN  → PARSER_FAILED
network / filesystem / URI / catalog resolution   FORBIDDEN
second-pass expansion (resolve_entities, xinclude, load_dtd …)  FORBIDDEN
```

Three details that matter more than the list:

**Rejection precedes resolution.** `<!DOCTYPE rss SYSTEM "https://evil.invalid/x.dtd">`
is rejected *without fetching that URI*, and `file:///etc/passwd` is never
opened. Detecting a construct by first resolving it would defeat the point.

**XInclude fails closed rather than being ignored.** Leaving it as an inert
unknown element would be library-dependent — which is the class of defect this
whole review series keeps finding.

**Detection is syntax-aware, not substring matching.** A literal `<!DOCTYPE`
inside an XML comment is *not* a prohibited construct: `<!-- <!DOCTYPE rss> -->`
must be accepted. `FOMC62` exists specifically to pin that down, so the policy
cannot be read as licence to grep the raw bytes. The five built-in entities
(`&amp; &lt; &gt; &apos; &quot;`) and numeric character references stay valid —
they are not DTD declarations.

Library defaults are **not** authoritative. No parser is pinned, but an
implementation must explicitly ensure every line above holds.

The `security_requirements` list now enumerates these separately and carries a
note that the decoded-body caps bound **parser input only**. "XXE disabled" is
not a sufficient summary — that shorthand is exactly what hid this gap.

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

### Decoded-body caps apply during production, not after unbounded expansion

Revision 9 changes only **when** the existing decoded-body cap is enforced.
The byte stage above stays unchanged: HTTP entity-body bytes after all
applicable transport/content decoding required by the frozen body semantics,
before charset/text decoding. Feed bodies remain capped at **2097152 bytes
(2 MiB)** and statement bodies at **5242880 bytes (5 MiB)**.

Decoded bytes MUST be counted incrementally from zero while they are produced.
Before appending each decoded chunk, compare `decoded_count + len(chunk)` with
the applicable cap. If it would exceed the cap, **abort body decoding
immediately**, do not append excess bytes, and do not continue consuming or
decompressing the response merely to learn its final decoded size. Otherwise,
append and update the counter. The accepted decoded-body accumulation must
never exceed the cap.

The decoder's output production must itself be bounded incrementally: an
unbounded `decompress_all` followed by a length check is forbidden, including
when its already-materialized result is then divided into chunks. Library,
bounded decoding API, buffer layout and chunk size are not prescribed. This is
not a promise of constant memory, exact RSS, total process-memory bounds or a
specific decompressor-internal-memory limit.

The cap applies to identity, gzip, brotli or other encodings **only insofar as
they are already admitted by V1**; this fix adds no encoding support.
`Content-Length`, compressed length and socket-byte length do not replace
decoded-byte counting. A missing, incorrect or misleadingly small
`Content-Length` cannot bypass the cap. Its role remains transport provenance
or an early rejection hint only where existing semantics permit that use.

Exactly the cap passes the **size gate only**, subject to all other existing
checks. One additional decoded byte aborts at attempted excess. Thus a 100 KiB
compressed statement whose full expansion would be 20 MiB must stop when
output would exceed 5 MiB, without producing/consuming the full 20 MiB first.
MiB remains 1048576 bytes, not decimal MB.

Oversize is terminal for that response's normal pipeline. It reuses the
existing **`PARSER_FAILED`** mapping, never `EVENTS_OBSERVED_ZERO`, and reaches
neither the charset/text decoder, XML/HTML parser nor normalizer. No
EventRevision is created; no oversized response is persisted or admitted as a
valid complete raw body artifact, and no successful complete-body SHA-256 is
manufactured from an incomplete prefix. No truncation-and-parse,
parse-first-N-MiB, best effort or partial-body replay is allowed.

Transport/provenance and failure metadata may remain only as already permitted
by the existing design; this fix defines no diagnostic hashes or partial-body
replay format. For complete admitted bodies, the existing hashed, persisted and
replay bytes and raw-first ordering are unchanged. These caps apply where a
body is consumed under current V1 policy; redirect identity, validation and
follow rules remain unchanged, with no new redirect-body parsing semantics.

### Text decoding: one codec, strict UTF-8

Revision 10 freezes `parser_policy.text_decoding_v1` for both RSS/XML discovery
and primary HTML. **UTF-8 is the only codec**, and the explicit default when
no encoding signal is present. This is a narrow V1 admission rule, not a claim
that all Federal Reserve documents use UTF-8.

There is no HTTP-versus-document authority competition. Every recognized
encoding signal is a **consistency constraint**: all must resolve to UTF-8.
Any unsupported, unknown, empty/malformed or conflicting declaration yields
`PARSER_FAILED`, even if the body happens to be valid UTF-8/ASCII.

For XML, recognize HTTP `Content-Type` charset, the UTF-8 BOM and the XML
encoding declaration. For HTML, recognize HTTP charset, the UTF-8 BOM,
`<meta charset=...>` and `<meta http-equiv="Content-Type" ... charset=...>`
through the `content` attribute. HTML attribute names and the `http-equiv`
value are compared ASCII case-insensitively for this purpose. All recognized
meta declarations must be checked; there is no first-wins or last-wins rule.
Every HTTP charset parameter must likewise pass, including multiple values.

Charset labels permit only syntax-level quote removal where the metadata
grammar allows it, trimming of surrounding ASCII whitespace permitted by that
grammar, and ASCII `A-Z` to `a-z`. The possible trim characters are U+0009,
U+000A, U+000C, U+000D and U+0020, restricted by the relevant metadata syntax.
No internal-character rewriting, Unicode whitespace trimming or Unicode case
folding is allowed. The entire normalized label must be exactly **`utf-8` or
`utf8`**. There is no platform codec-alias lookup.

After the unchanged bounded-body admission, durable persistence and byte hash,
inspect the leading BOM and HTTP declarations. A leading `EF BB BF` is allowed;
omit exactly one such signature from the **text-only view**, then strictly
decode the entire remaining body as UTF-8. Validate the document's encoding
declarations on that text before exposing it for semantic extraction or
normalization. No declaration may select another decoder or initiate a second
attempt. Invalid/truncated sequences, overlong encodings, encoded surrogates
and code points above U+10FFFF fail closed.

The UTF-8 signature is not exposed as semantic leading U+FEFF. Raw, hashed and
replay bytes still contain it, and it still counts toward the byte cap. Do not
strip another signature or later U+FEFF code point. Leading UTF-32 signatures
`00 00 FE FF` / `FF FE 00 00` and UTF-16 signatures `FE FF` / `FF FE` are
rejected, checking the four-byte forms before the overlapping two-byte forms.
Never switch to a UTF-16/32 decoder.

No charset heuristics, chardet, browser/locale/platform default, `errors=ignore`,
`errors=replace`, `surrogateescape`, latin-1/windows-1252 fallback or codec
guessing is allowed. R9's HTTP `windows-1252` / HTML `utf-8` / isolated byte
`E9` counterexample therefore fails; it cannot decode `E9` as `é` and continue.

Existing raw HTTP provenance must retain the exact relevant `Content-Type`
field value(s), including charset spelling and duplicates, or explicitly
record actual header absence. A missing provenance record is not equivalent
to an absent HTTP header: fail closed, never refetch. The same verified body
bytes, persisted relevant metadata and CaptureSpec hash must yield the same
Unicode text or `PARSER_FAILED` offline.

Charset failure retains the already-admitted raw body and its byte hash,
creates no EventRevision, and cannot establish `EVENTS_OBSERVED_ZERO` for the
affected feed/candidate. No new health state is introduced. This fix adds no
Unicode normalization or whitespace transformation beyond consuming the one
encoding signature; existing title/release transformations remain unchanged.
XML entities/DTD/expansion, malformed HTML recovery, HTTP status/304, retries,
rate windows, scheduling, restart/overdue handling and timeouts remain outside
this fix. No fixture or runtime implementation is created here.

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

## 11. Two hundred and sixty cases, decided in advance

`FOMC01`–`FOMC260` in the JSON settle summer/winter releases, immediate release,
bare `ET`, a feed item whose page will not load, late observation, unchanged
and changed bytes under one GUID, GUID conflicts, `Last Update` drift, a 2027
backfill of a 2026 statement, local raw corruption, malformed XML, feed
outages, a genuinely empty poll, a failed commit after raw persistence, a lost
ACK, and an unchanged +7 d re-check.

Revision 2 added three that close the R1 HIGH, revision 3 four more that
close the R2 MEDIUM, and revision 4 adds the R3 date-correction case.
Revision 5 adds four cases for R4's discovery-URL identity finding; revision 6
adds HTTPS scheme convergence and non-HTTPS rejection; revision 7 adds
default-port convergence and non-default-port rejection. Revision 8 adds three
malformed-percent rejection cases and one syntax-only valid-escape control.
Revision 9 adds bounded compression-bomb rejection and exact decoded-cap
boundaries. Revision 10 adds seven strict UTF-8 decoding cases:

| | case | outcome |
|---|---|---|
| `FOMC19` | body corrected upstream at 16:00, first backfilled in 2026 | at 14:30 the fetched body is **not visible**; no claim about the uncaptured original either |
| `FOMC20` | live correction, both bodies print the same release line | neither revision inherits 14:00 |
| `FOMC21` | first live fetch 47 s after the declared release | still `source_available_at = null` |
| `FOMC22` | unscheduled monetary action, different titles | healthy out-of-V1-scope after its page is read; first cycle not zero |
| `FOMC23` | feed title matches, primary title differs | conflict; no event, no zero, no silent downgrade |
| `FOMC24` | clean cycle, no matches, no unresolved | zero **permitted**, scoped to the V1 subset |
| `FOMC25` | positive candidate, primary fetch fails | unresolved; zero prohibited |
| `FOMC26` / R3-X6 | same canonical URL, D1/H1 corrected to D2/H2 | same logical item S; append R2, retain R1; valid with the same GUID or without any GUID |
| `FOMC27` | official discovery U1 redirects to U2 | identity from `normalize(U1)`; U2 is transport provenance |
| `FOMC28` | same U1 redirects first to U2, later to U3 | same item; unchanged body keeps revision, changed body appends revision |
| `FOMC29` | HTML canonical U4 differs from discovery U1 | U4 has no identity authority; no re-key or automatic merge |
| `FOMC30` | official archive U1 redirects to U2 without GUID | same pre-redirect discovery identity rule |
| `FOMC31` | uppercase/mixed-case HTTPS in the official discovery link | accepted scheme, exact lowercase `https` output, same identity as lowercase input |
| `FOMC32` | HTTP official-host URL | reject, no upgrade, no identity admission, no primary request |
| `FOMC33` | absent port versus explicit HTTPS `:443` | same portless canonical URL and source item; no conflict or revision from port spelling alone |
| `FOMC34` | HTTPS official-host URL with `:444` | reject before identity derivation or request; no stripping or port rewrite |
| `FOMC35` | statement link ending `monetary20260617a%.htm` | reject; no canonical identity, source item or primary request |
| `FOMC36` | `https://www.federalreserve.gov/path%2` | reject the short escape; no identity or request |
| `FOMC37` | `https://www.federalreserve.gov/path%GG` | reject non-hex escape; also reject before redirect follow |
| `FOMC38` | `https://www.federalreserve.gov/path%2Fsegment` | percent syntax passes, `%2F` unchanged; full URL admission not asserted |
| `FOMC39` | 100 KiB compressed statement with potential 20 MiB decoded output | stop at attempted excess over 5 MiB; `PARSER_FAILED`, no text decode, parse, complete body artifact/hash or revision |
| `FOMC40` | statement 5 MiB / feed 2 MiB, then each cap + 1 byte | exact cap passes size gate only; attempted extra byte aborts decoding |
| `FOMC41` | HTTP `windows-1252`, HTML meta `utf-8`, isolated `E9` | `PARSER_FAILED`; no fallback or EventRevision |
| `FOMC42` | HTML, no charset signals, valid UTF-8 | strict UTF-8 default; text gate passes only |
| `FOMC43` | leading UTF-8 BOM, other signals absent or compatible | text gate passes; one signature omitted from text, raw/hash bytes unchanged |
| `FOMC44` | XML `encoding="ISO-8859-1"` | `PARSER_FAILED`; no ISO-8859-1 decoding |
| `FOMC45` | recognized declaration of `x-unknown` | `PARSER_FAILED`; no platform alias lookup |
| `FOMC46` | UTF-8-compatible declarations, invalid UTF-8 bytes | `PARSER_FAILED`; raw retained, no lossy replacement |
| `FOMC47` | HTTP UTF-8, UTF-8 BOM, HTML meta utf-8, valid body | strict UTF-8 text gate passes; no competing authority |

FOMC26 also requires identical-body date determinism under the same spec hash:
a conflicting date fails closed without a new content revision. Future test
requirements explicitly include both the same-GUID case and historical
archive/backfill with no GUID. These are requirements only; no fixture is
created or captured here. FOMC27–FOMC30 are also explicit future test
requirements, including both body-hash outcomes for FOMC28. FOMC31/FOMC32 add
future requirements for HTTPS case variants and rejected schemes only; no
runtime or fixture is created here. FOMC33/FOMC34 add future requirements for
default-port convergence and non-443 rejection. Malformed-port controls include
`:abc`, `:443abc`, an empty explicit port, and out-of-range `:65536`, all rejected
rather than treated as absent/default. A leading-zero `0443` control must parse
as decimal 443 and be omitted from canonical identity. The same admission is
required on redirect targets without changing their non-identifying role.

FOMC35–FOMC38 add future offline requirements for malformed forms `%`, `%2`,
`%GG`, `%G0`, `%0G`, `%2Z`, `%%20`, `%2%` and `%u1234`, and valid syntax controls
`%20`, `%2F`, `%2f`, `%7E`, `%7e`, `%00` and `%FF`. They also require malformed
redirect-target rejection before follow, no repair or decoding, and unchanged
raw discovery provenance. These are specification requirements only; no fixture
or runtime is created or exercised here.

FOMC39/FOMC40 add future offline test requirements for identity below cap,
gzip exactly cap, cap + 1 and small-compressed/huge-decoded responses, with the
same controls for brotli or other encodings only if already admitted. Both feed
and statement boundaries, missing and misleadingly small `Content-Length`, and
all terminal oversize outcomes must be checked. An instrumented test decoder
must prove it stops at attempted excess without first producing/consuming the
full 20 MiB; testing only the final rejection verdict is insufficient. These
are requirements only: no fixture or implementation test is created or run here.

FOMC41–FOMC47 add future offline requirements for XML with no declaration,
UTF-8 or ISO-8859-1 declarations, accepted UTF-8 and rejected UTF-16/32 BOMs,
and agreeing/conflicting HTTP and XML signals. HTML requirements cover absent
charset, HTTP/meta signals separately and together, the `http-equiv` form,
multiple agreeing/conflicting metas, unknown labels, invalid UTF-8 and the
UTF-8 BOM. Label aliases/case/quoting/ASCII-space controls, exact BOM handling,
strict byte-error rejection, preserved raw hashes and metadata-only offline
replay must also be checked. No fixture or implementation test is created or
run here; passing the text gate never alone admits an EventRevision.

Invariants `F01`–`F53` state the same commitments in testable form. Revision 2
added `F19` (release time is not vintage proof), `F20` (backfill content
vintage) and `F21` (revision-specific availability); revision 3 adds `F22`
(an absence claim cannot exceed classifier coverage), `F23` (an unresolved
candidate blocks zero) and `F24` (discovery/primary conflict fails closed).
Revision 4 strengthens `F09`: metadata corrections preserve logical identity
at the same canonical URL, while changed bytes append immutable revisions.
New `F25` forbids mutable content metadata, including the official date, from
defining logical identity and binds the identical-body determinism requirement.
Revision 5 adds `F26` (the official discovery URL anchors logical identity) and
`F27` (transport redirection cannot re-key an item).
Revision 6 strengthens `F26` with exact lowercase ASCII `https` serialization
and adds `F28`: HTTPS scheme case cannot alter logical identity.
Revision 7 strengthens `F26` with existing hostname normalization and exact
default-port omission, and adds `F29`: explicit default HTTPS port cannot alter
logical identity. All earlier invariants retain their commitments.
Revision 8 strengthens `F26` with deterministic malformed-percent rejection
before identity and adds `F30`: malformed percent escapes are rejected before
identity, primary request or redirect follow, without repair or decoding.
Revision 9 adds `F31`: decoded-body size is enforced during bounded production,
with immediate abort at attempted excess and no oversized or truncated body
passed to parsing or admitted as a complete body artifact/hash or revision.
All previous invariants remain unchanged by this fix.
Revision 10 adds `F32`: text decoding is single-valued and strict UTF-8, with
all recognized declarations acting as consistency constraints, deterministic
signature removal in the text view only, and replay based on unchanged body
bytes plus persisted HTTP charset provenance. F01–F31 remain unchanged.
Revision 15 adds **F44 — PROVIDER RATE ACCOUNTING COUNTS PHYSICAL ATTEMPT STARTS,
NOT LOGICAL FETCHES**: each independently attempted HTTP request consumes one
unit before network side effects, with no refund on failure. Feed, primary,
every followed redirect, permitted later reattempt, scheduled recheck and
historical provider GET all count. Three redirect follows plus the initial
request consume four units.

Revision 15 also adds **F45 — ALL FOMC V1 NETWORK CATEGORIES SHARE ONE PROVIDER
ACCOUNTING DOMAIN**: no independent feed/primary/recheck/redirect/historical
quota buckets, and every physical attempt passes through the same gate.
Redirect/retry internals cannot bypass it. F01–F43 and FIX1–FIX13 retain their
commitments.

The canonical JSON adds these adversarial specification cases:

| case | expected accounting |
|---|---|
| `FOMC83` — U0 → 302, U1 → 301, U2 → 200 | 3 physical units, each through the same gate |
| `FOMC84` — U0 plus three followed transitions to U3 | 4 units; redirect #4 target U4 never requested, no fifth unit |
| `FOMC85` — one feed GET followed by one primary GET | 2 units in the same domain |
| `FOMC86` — U0 returns `302 Location: //evil.example/x` | U0 = 1 unit; rejected target = 0; total 1 |
| `FOMC87` — admitted start followed by DNS/connect/TLS/transport failure | 1 unit, no refund and no successful observation implied |
| `FOMC88` — failed attempt then legitimately permitted later reattempt | 1 + 1 units through the same gate; existing retry policy unchanged |
| `FOMC89` — recheck pending alongside feed/primary traffic | same provider domain; 1 unit if attempted; no separate quota or due-time bypass; ordering per `FOMC126` |

Revision 16 adds four invariants while preserving F01–F45:

| invariant | frozen requirement |
|---|---|
| `F46` — PROVIDER RATE CEILING USES A ROLLING 60-SECOND PHYSICAL-START WINDOW | prior starts count iff `0 < t - s < 60`; at most five before the candidate, at most six in `(t-60s,t]` after admission; exact t=60 excludes start 0 |
| `F47` — MINIMUM SPACING IS PROVIDER-WIDE START-TO-START | `t - previous_start >= 10`, equality allowed; same event for accounting/window/spacing; no category exemption, banked permits or post-grant refund |
| `F48` — RATE ADMISSION IS ATOMIC AND MONOTONIC WITHIN A LIMITER EPOCH | shared atomic inspect/test/grant/register across emitters; monotonic comparisons; no stale-state double grant or UTC-driven early start |
| `F49` — NEW LIMITER EPOCHS FAIL CONSERVATIVELY CLOSED FOR 60 SECONDS | exclusive ownership, full new-epoch monotonic embargo, exact 60 allowed, fresh history and no overlapping independent epochs |

The corresponding new canonical adversarial cases are:

| case | frozen verdict |
|---|---|
| `FOMC90` — prior starts 0,10,20,30,40,50; candidate 60 | 5 active prior starts, spacing 10; ADMITTED |
| `FOMC91` — same history; candidate 59.999 | 6 active prior starts, spacing 9.999; WAIT, no start/unit |
| `FOMC92` — previous start 100; candidate 110 | spacing passes at exact equality; other gates still apply |
| `FOMC93` — A starts 0, completes 45; B candidate 45 | spacing passes with delta 45; no wait to completion +10; window still applies |
| `FOMC94` — monotonic 100 → 105; UTC jumps backward five minutes | spacing fails with delta 5; wall-clock change has no rate effect |
| `FOMC95` — two simultaneous workers, five prior starts, spacing permits | at most one same-instant grant; the second sees registration and fails spacing |
| `FOMC96` — grant at t, cancellation before network I/O | one consumed unit; t remains in window/spacing history; no refund or reuse |
| `FOMC97` — fresh exclusive epoch, candidates e=2 / 59.999 / 60 | first two wait; exact 60 passes embargo and normal gates on empty new-epoch history |
| `FOMC98` — feed start 100, primary candidate 101 | WAIT; earliest spacing eligibility 110, subject to window |
| `FOMC99` — U0 start 200, 302 at 201, follow candidate 201 | WAIT; earliest spacing eligibility 210, subject to window |
| `FOMC100` — old independent issuer remains active | no new independent provider attempt; establish exclusivity, then a full new-epoch embargo |

Non-restart rate-limiter examples use the post-embargo time origin defined in §7.

Revision 17 adds **F50 — REOBSERVATION GRID IS ANCHORED TO THE FIRST DURABLE
PRIMARY OBSERVATION'S observed_at**: for each logical source item, exact original
`observed_at` supplies A and every offset sum. Request/header time, raw or
normalized commit, declared release, `ingested_at`, monotonic limiter start and
restart time cannot replace it. Later observations and revisions cannot reset A.

It also adds **F51 — DURABILITY DETERMINES ANCHOR SURVIVAL, NOT ANCHOR TIME**:
volatile observation lost before durability establishes no reconstructible
anchor; once durable, its original `observed_at` survives crash/restart and
downstream parsing failure without commit-relative or restart-relative reset.
F01–F49 retain their commitments.

| case | frozen anchor verdict |
|---|---|
| `FOMC101` — start/header/body/raw/normalized times 18:00:00/:02/:05/:06/:07 | A=18:00:05; targets 18:05:05, 19:00:05, next day 18:00:05 and +7d 18:00:05 |
| `FOMC102` — declared release 14:00, observed 18:00, raw durable 18:00:01 | A=18:00; no retroprojection or source availability inference |
| `FOMC103` — complete admitted primary body observed 18:00:05, raw durable 18:00:06, then parsing/normalization fails | A exists at 18:00:05; failure does not erase or replace it; no normalized success implied |
| `FOMC104` — transient observation 18:00:05, crash before raw/observation durability | no reconstructible anchor survives; a later qualifying durable observation may establish the first anchor |
| `FOMC105` — raw durable 18:00:06, crash before normalized commit, restart 18:02:05 | recover A=18:00:05; first target remains 18:05:05 |
| `FOMC106` — first H1 at A=18:00:05; later recheck H2 at 19:00:07 | same source item's A is unchanged; content revision does not reset future targets |
| `FOMC107` — due O300 recheck delayed by FIX15 spacing and class rotation to a grant at 18:08:54 | A and targets A+3600/A+86400/A+604800 stay fixed; satisfied by the 18:08:55 observation |

Revision 18 adds **F52 — DURABLE REOBSERVATION ANCHOR IMPLIES FOUR REQUIRED
LOGICAL OBLIGATIONS** and **F53 — REOBSERVATION OBLIGATION IDENTITY IS DERIVED
AND IDEMPOTENT**. F01–F51 and all earlier case verdicts retain their commitments.

| case | frozen logical-obligation verdict |
|---|---|
| `FOMC108` — anchor durable, crash before any scheduler row | exactly four keys and original A+offset targets reconstructed |
| `FOMC109` — only 300 and 3600 materialized | all four logical obligations exist |
| `FOMC110` — 300 materialized twice | one logical 300 obligation; four total |
| `FOMC111` — reconstruction performed three times | same four keys each time; repeated/concurrent recovery and lost materialization ACK add none |
| `FOMC112` — H1 followed by H2 | same original anchor and four keys; same-body reobservation and parser recovery likewise add no grid |
| `FOMC113` — qualifying primary raw durable, then parser failure | four obligations exist; no EventRevision success implied |
| `FOMC114` — crash before anchor durability | zero reconstructible obligations from that lost observation |
| `FOMC115` — S1 and S2 share A | four per item, eight total; no collision |

Revision 19 adds **F54–F62**, closing the remaining runtime semantics. They
cover past-due obligations, coalescing, the durable-observation satisfaction
boundary, failed attempts, crash/ACK reconstruction, class fairness, the 60 s
physical-attempt deadline, failure-blocks-zero and literal EST/EDT. F01–F53 and
every earlier verdict keep their commitments; `FOMC04`, `FOMC89`, `FOMC105` and
`FOMC107` now point to the rules that complete them.

| case | frozen verdict |
|---|---|
| `FOMC116` — offline across O300's due time, restart later | still `PENDING_DUE`; real later observation; no backdating |
| `FOMC117` — O300/O3600/O86400 overdue, one fetch at A+2d | one logical fetch satisfies all three; O604800 pending |
| `FOMC118` — earliest reachable O300: `NOW_LB` passes due + 92 s after the 18:08:15 feed record, grant 18:08:24 | satisfied at A+500; lateness visible; no drift |
| `FOMC119` — observation at A+1800 | O300 satisfied; O3600 stays pending |
| `FOMC120` — same H1 recheck | satisfied; no new revision |
| `FOMC121` — H2 durable, crash before revision commit | satisfied; revision appended idempotently after recovery |
| `FOMC122` — durable recheck, anchor absent in HTML | satisfied; `PARSER_FAILED`; no revision |
| `FOMC123` — 500, or deadline expiry | pending; `SOURCE_UNAVAILABLE`; eligible 60 s after the failure |
| `FOMC124` — crash after durable recheck, before marker | satisfied on recovery; no duplicate GET |
| `FOMC125` — lost ACK | same satisfied set; no GET from ACK loss |
| `FOMC126` — feed vs persistent recheck backlog | strict alternation; nothing starves |
| `FOMC127` — one byte every 29 s | aborted at 60 s; nothing admitted |
| `FOMC128` — DNS hang | aborted at 60 s; late answer discarded |
| `FOMC129` — TLS hang | inactivity 30 s or total 60 s; `SOURCE_UNAVAILABLE` |
| `FOMC130` — header hang | inactivity 30 s or total 60 s; `SOURCE_UNAVAILABLE` |
| `FOMC131` — 2 of 3 candidates succeed | 2 kept; no rollback; zero impossible |
| `FOMC132` — July, `EST` | 19:00Z; no seasonal rewrite |
| `FOMC133` — January, `EDT` | 18:00Z; no seasonal rewrite |
| `FOMC134` — `ET`, `Eastern`, `local time`, no zone | no guess; `RELEASE_TIME_UNPARSED`; statement kept (revision 20) |
| `FOMC135` — corrupt raw at replay | fail closed offline; no refetch; obligations unchanged |

Revision 20 repairs every HIGH and MEDIUM finding of the REV19 global review.
F57, F59–F62 are rewritten and **F63–F72** added. They cover:

- structural-then-semantic relevance
- the newer-content barrier
- finite, durable retry
- no provider traffic from local failures
- manual redirects
- the backfill manifest without a recheck grid
- acquisitions that survive feed changes
- canonical timestamps
- clock anomalies that fail safe
- corruption that fails closed everywhere

`FOMC04`, `FOMC15`, `FOMC22`–`FOMC25`, `FOMC123`, `FOMC126`, `FOMC128`, `FOMC131`
and `FOMC134` are updated to the new rules.

| case | frozen verdict |
|---|---|
| `FOMC136` — 4 concluded statements + 11 concluded non-statements, nothing outstanding | zero |
| `FOMC137` — new in-family item, non-statement title | fetched once; healthy negative; its cycle not zero, later cycles may be |
| `FOMC138` — only concluded items, nothing outstanding | zero |
| `FOMC139` — admitted item reappears | known, not newly admitted |
| `FOMC140` — new statement admitted after its cycle concluded | cycle not zero; admission attributed to it |
| `FOMC141` — H1 normalized, H2 durable but parser-failed | no H2 revision; newer-content barrier |
| `FOMC142` — anchored page 404 forever | one episode per offset key, 6 attempts each; then silence |
| `FOMC143` — restart after 3 failures | attempt 4 after the embargo and the persisted 900 s wait |
| `FOMC144` — 100 suspended items + feed + fresh recheck + backfill | nothing starves |
| `FOMC145` — body received, local store fails, bytes lost | one ordinary episode attempt, never an extra retry |
| `FOMC146` — urllib `AllowlistedRedirectHandler` | not conforming |
| `FOMC147` — manifest duplicates, other order | deduplicated, `source_item_id` order |
| `FOMC148` — backfill of a 2021 statement | no anchor, no rechecks |
| `FOMC149` — backfilled item appears LIVE | new LIVE candidate; not zero until concluded; revision reused with a LIVE link |
| `FOMC150` — old-epoch response commits after new epoch | normal before the owner's `INTERRUPTED`, otherwise `LATE_EVIDENCE`; same in live and replay |
| `FOMC151` — no Content-Type | `PARSER_FAILED` |
| `FOMC152` — `text/html` and `application/json` | `PARSER_FAILED` |
| `FOMC153` — `Text/HTML ; Charset="utf-8"; foo=bar` | accepted |
| `FOMC154` — DNS answer at 61 s | no TCP, no HTTP byte |
| `FOMC155` — due-time equality at sub-second precision | parsed instants decide |
| `FOMC156` — wall clock jumps a day forward | capture continues; nothing visible or due early |
| `FOMC157` — corrupt raw during processing | no parse, no refetch |
| `FOMC158` — `2:00&nbsp;p.m. ET` | null release time; statement kept |
| `FOMC159` — candidate vanishes from the next feed | acquisition still required; later cycles not zero |
| `FOMC160` — clean LIVE cycle, 300 backfill entries pending | zero if nothing LIVE is outstanding |
| `FOMC161` — H1 → H2 → H1 | H1's revision reused and current; no duplicate |
| `FOMC162` — known item's feed title changes | cycle not zero (diagnostic); classification unchanged; no traffic |
| `FOMC163` — NTP steps the clock back 3 s | still verified |
| `FOMC164` — crash after durable success, before outcome record | success; no duplicate GET |
| `FOMC165` — host clock 5 min fast | responses unverified but processed; no causal effect until corrected |
| `FOMC166` — bogus 280 s forward step | unverified against its own `Date`; nothing pre-satisfied |
| `FOMC167` — CDN-cached page, old `Date`, `Age: 1200` | server time = Date + Age; usable |

Revision 21 closes the REV20 final review. It rewrites F23, F24, F59, F61, F63–F66,
F68, F70 and F71, and adds F73–F77:

- outstanding items persist until concluded;
- revisions are mode-neutral, while availability is specific to each observation;
- local processing work derives from durable records;
- HTTP clock headers are parsed strictly;
- the GUID is provenance, the URL is identity.

Older cases whose verdicts changed are updated, in particular FOMC22–25,
FOMC136–138, FOMC142, FOMC149 and FOMC156–167.

| case | frozen verdict |
|---|---|
| `FOMC168` — statement parse fails, still listed | outstanding; not zero |
| `FOMC169` — unresolved acquisition omitted by next feed | not zero |
| `FOMC170` — H1 normalized, H2 unnormalized | snapshot does not return H1 as current |
| `FOMC171` — A → B → A | existing A revision current |
| `FOMC172` — H2 later normalized | barrier clears |
| `FOMC173` — backward wall step mid-episode | no close, no reopen |
| `FOMC174` — six failed O300 attempts | one final suspended episode |
| `FOMC175` — O3600 success after O300 suspended | O300 satisfied, episode not reopened |
| `FOMC176` — restart during a 3600 s backoff | wait recomputed from durable records (`NOW_LB` ≥ avail + 3600); no wall deadline kept |
| `FOMC177` — relist of a suspended acquisition | no new episode |
| `FOMC178` — routine non-statement | healthy negative |
| `FOMC179` — CDATA, entities, whitespace in RSS titles | deterministic normalization |
| `FOMC180` — unforeseen safe feed path | unresolved; zero blocked |
| `FOMC181` — one GUID, two URLs | no merge; diagnostic |
| `FOMC182` — one URL, new GUID | one item; diagnostic |
| `FOMC183` — backfill H1 then LIVE H1 | one revision, LIVE link |
| `FOMC184` — manifest with 1001 raw entries | rejected, 0 requests |
| `FOMC185` — one invalid entry in 1000 | rejected, 0 requests |
| `FOMC186` — late old-epoch record | processed on commit (as `LATE_EVIDENCE` if its attempt already has an outcome) |
| `FOMC187` — cycle conclusion commit fails | local retry, no refetch |
| `FOMC188` — host 1 day off, invalid `Date` | unverified; no admission |
| `FOMC189` — duplicate `Date` | reference unusable |
| `FOMC190` — no `Age` | Age = 0 |
| `FOMC191` — `Age: 90000` | reference unusable; unverified |
| `FOMC192` — only the redirect has a valid `Date` | no clock evidence |
| `FOMC193` — response without `Date` | unverified |
| `FOMC194` — raw commit during backward step | available at max(next verified `observed_at` + 92 s, own `observed_at`) |
| `FOMC195` — `HTTPS_PROXY` set | ignored |
| `FOMC196` — processing always throws | `INTERNAL_PROCESSING_ERROR`, no loop |
| `FOMC197` — out-of-scope item rechecked with the statement title | re-opened; not zero |
| `FOMC198` — backfill H2 fails to parse on a LIVE-admitted item | snapshot barrier; zero unaffected |
| `FOMC199` — unsupported path, later operator-resolved | first cycle not zero; then non-blocking |
| `FOMC200` — crash between feed commit and its conclusion | reconciliation concludes it; no deadlock |
| `FOMC201` — host suspends 8 h between two transactions | each available from the next verified response; nothing early |
| `FOMC202` — feed retitled to statement title, recheck says out of scope | still outstanding |
| `FOMC203` — resolution attempted while an observation is unprocessed | invalid; concludes nothing |
| `FOMC204` — backfill fetch of a LIVE-anchored item | never satisfies LIVE rechecks |
| `FOMC205` — unidentifiable statement item rotates out of the feed | durable record; not zero until an operator resolves it |
| `FOMC206` — O300 before `PENDING_DUE` | dispatch refused; no attempt burned; a verified recheck can never be early |
| `FOMC207` — `TRANSPORT_INVOKED` ACK lost, no transport | counts; closed as interrupted; nothing in flight |
| `FOMC208` — parser-failed backfill-only item | `SOURCE_ACTIVITY_NO_REVISION` |
| `FOMC209` — unverified newer H2 after V1 | barrier, never evidence |
| `FOMC210` — feed processing always crashes, item rotates out | feed record outstanding; not zero |
| `FOMC211` — resolved item, then any new LIVE observation | re-opened; not zero |
| `FOMC212` — two spellings, titles and GUIDs in one feed record | one item; one diagnostic per new value; not zero |
| `FOMC213` — crash before writing the 6th-attempt suspension | recovery writes it |
| `FOMC214` — `TRANSPORT_INVOKED` not committed within 1 s | nothing sent; not counted |
| `FOMC215` — unverified old H1 after verified H2 | barrier, never H1 |
| `FOMC216` — legitimate NTP step of +120 s | committed; available from the next verified response |
| `FOMC217` — exact-title item with missing `<link>`, rotated out | item-level record; not zero until resolved |
| `FOMC218` — commits after a boot before any verified response | unresolved, then bounded by the first verified response |
| `FOMC219` — operator retry anchors while acquisition in backoff | acquisition episode succeeds; nothing left open |
| `FOMC220` — `TRANSPORT_INVOKED` commits late, repeatedly | each counts; may suspend after six; bound kept |
| `FOMC221` — valid statement with unverified clock | processed and revisioned; no anchor; item outstanding until eligible or validly resolved |
| `FOMC222` — feed response fails its clock check | classified; poll ends, no cycle, `CLOCK_INELIGIBLE`; polling continues |
| `FOMC223` — last verified response 20 days old, clock 600 s slow | nothing eligible, nothing early; due work held |
| `FOMC224` — first primary response unverified, a later one verified | anchor on the first eligible record; no new keys |
| `FOMC225` — two processing runs for one record | one running run; every dead run counts, running ones never |
| `FOMC226` — manual retry open when autonomous work succeeds | manual episode succeeds before dispatch |
| `FOMC227` — restart in a new boot mid-backoff, clock off a year | waits on server time only; bounded |
| `FOMC228` — verified record 90 s ahead of true time | available no earlier than its `observed_at` |
| `FOMC229` — VM paused an hour inside a fetch | eligible if committed before its attempt's outcome; available only after the next verified response |
| `FOMC230` — suspend between a transaction's wall reading and its commit | available from a response sent after the durable commit |
| `FOMC231` — late old-epoch feed response after a newer poll | `LATE_EVIDENCE`, classified next, no cycle; its new items become candidates |
| `FOMC232` — item listed only during a clock fault, then dropped | candidate created; later cycles not zero until concluded |
| `FOMC233` — concluded negative, later unverified recheck shows the statement title | reopened |
| `FOMC234` — quiet period, poll task dies after `TRANSPORT_INVOKED` | interrupted at the next tick; polling resumes |
| `FOMC235` — wall clock +7 days while O604800 pending | not opened; no attempt burned |
| `FOMC236` — reboots every 30 min during a 14400 s backoff | attempt 6 still becomes eligible |
| `FOMC237` — live read before the newest outcome commits; later replay | identical (inputs only from P(T)) |
| `FOMC238` — newest primary record corrupted after commit | barrier; never the older revision |
| `FOMC239` — late old-epoch feed with the exact title after a retitling feed | `LATE_EVIDENCE` classified after the newer record; marker; outstanding |
| `FOMC240` — V1 current vs barrier over V1 | different snapshot identities |
| `FOMC241` — attempt interrupted at 600 s, its stale response arrives after a newer conflict | `LATE_EVIDENCE`, processed as unverified; item stays outstanding; snapshot barrier |
| `FOMC242` — CDN serves a 200 HTML maintenance page as the feed (also when clock-unverified) | terminal parser failure; `PARSER_FAILED` health plus clock diagnostic; polling continues |
| `FOMC243` — live read with everything unresolved and no FOMC item yet | FOMC-level unresolved, never an empty resolved part |
| `FOMC244` — unresolved live read, later resolved; replay at the same horizon | unresolved again, same identity |
| `FOMC245` — newest feed record has no conclusion yet, an older cycle was zero | `DISCOVERY_PENDING`; the older zero is not shown |
| `FOMC246` — A (LIVE), B (LIVE), then A via backfill | current A, but not LIVE-available |
| `FOMC247` — operator retry anchors an item before its acquisition opens; conflict | acquisition can no longer open; RESOLVE valid; without an anchor it stays invalid |
| `FOMC248` — admission check passes at +40 s, the `COMMIT` returns at +170 s | valid record and outcome; `observed_at` unchanged; available only after it was durable; storage incident, no grant meanwhile |
| `FOMC249` — a local write hangs, the admission check runs at +120.0 s | nothing admitted; `LOCAL_PERSISTENCE_FAILED`; at +119.9 s the record is admitted |
| `FOMC250` — a `COMMIT` stalls 300 s while a poll and a recheck are due | alert; no grant, no decision; afterwards first-outcome reconciliation, no repair request, budgets unchanged |
| `FOMC251` — two responses differ only in Cloudflare keys and challenge parameters | two records and raws, one revision |
| `FOMC252` — text, rate, date, title, release line or effective e-mail address changes | a new revision each time |
| `FOMC253` — a Cloudflare marker in the text, an ambiguous structure, invalid UTF-8 | canonicalization refused; raw identity; no merge |
| `FOMC254` — A, B, then A with other Cloudflare bytes | A's revision reused |
| `FOMC255` — backfill then LIVE with different Cloudflare bytes | one revision; LIVE availability only from the LIVE link |
| `FOMC256` — the newest same-content record's raw is corrupt | the read and replay fail closed |
| `FOMC257` — a Cloudflare marker in a `title` attribute, a `<textarea>` or a comment | refused (RAW_FALLBACK); no merge across keys |
| `FOMC258` — a document equal to another record's canonical bytes | RAW_FALLBACK; a different identity |
| `FOMC259` — redirects, a failure after a redirect, an abandoned grant, a restart | every grant journaled; FIX15 proved |
| `FOMC260` — a lost or inconsistent grant record | FIX15 NOT_PROVEN; an unjournaled grant is never used |

These are specification cases only; no fixture, capture or runtime is created
or exercised here. Revision 22 (FIX21) closes the three blockers of the REV21
global review: the clock verdict no longer stops processing (D1), a resolution
is blocked only by an acquisition that can still open (D2, `FOMC247`), and
`FOMC05`, `FOMC107`, `FOMC118`, `FOMC176` and `FOMC206` now match the active
rules (D3). Revision 23 (FIX22) records one decision — the 120 s local bound
governs admission, not durability, and a stalled store is a storage incident
(`FOMC248`–`FOMC250`, invariants F80–F81) — and states how grants are dispatched
to concurrent fetches. Revision 24 (FIX23) separates raw integrity from content
identity (`FOMC251`–`FOMC256`, invariant F82). Revision 25 (FIX24) reads the
Cloudflare spans only as real HTML tokens, separates the CANONICAL and
RAW_FALLBACK identity domains, and journals every FIX15 grant (`FOMC257`–
`FOMC260`, invariant F83). This document authorizes no capture.

## 12. What this does not establish

Nothing has been captured, implemented or observed. No predictive edge, no
causal market impact, no economic significance.

`commercial_edge_established = false`.
