# FOMC event capture spec V1 — Federal Reserve monetary-policy statements

```
CAPTURE SPEC ID       federal_reserve_fomc_capture_v1
VERSION               1
STATUS                FROZEN_PRE_IMPLEMENTATION

NOT implemented · NOT captured · NOT live · zero requests made in this phase

CAPTURE SPEC HASH  (revision 14 — authoritative)
bbc29abba992cfa4edeecc69c3b9beeb51f5955fbc6b87f0f863d8ca5d0f9050

supersedes 74571bb0 (rev 1) → b852b560 → d4242ea5 → f0b68307 → 8a39a39a → 6693a665
        → 3db09125 → c1d56f46 → 0757b1f5 → 3dcf0a60 → b9ad3c44 → b6772a2f
        → 67e2d7d5 (rev 13)

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
writing revision 11.**

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

Each anchor must resolve to **exactly one** node. Zero or two →
`PARSER_FAILED`, never first-wins, last-wins, visible-wins or traversal order.
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
unchanged: same anchor, existing grammar; different structure, fail closed.

If upstream markup drifts and cardinality breaks, V1 stops with `PARSER_FAILED`.
Heuristic broadening and automatic alternative-node search are forbidden. That is
deliberate drift detection, not a limitation to be engineered around.

### Three outcomes, not two
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
identity quality stays **GOOD**, not EXCELLENT — and both conflict directions
(one GUID → two items, one item → incompatible GUIDs) **fail closed**. Never
"latest wins."

A date correction alone is not an identity conflict. With the same URL and
GUID, it preserves `source_item_id`; historical archive/backfill observations
without a GUID do so as well. Existing GUID conflict rules still apply when
independently triggered.

The date remains required for normalization, under the existing primary-page
authority and date grammar. A date parse failure follows the existing
parser/source-health policy. Only its identity role changes.

### The discovery link is the identity input

For LIVE, use the official monetary-policy RSS item's `<link>`. For
HISTORICAL_BACKFILL, use the statement link in the official Federal Reserve
calendar/archive. In both cases the discovery artifact must already be durable
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
normalization, raw-first ordering and commit-bound visibility still apply.
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
| feed/archive link U1 redirects to U2 | `normalize(U1)` |
| the same U1 later redirects to U3 | the same `normalize(U1)` |
| primary HTML declares canonical U4 | still `normalize(U1)` |
| archive link U1 redirects to U2, with no GUID | `normalize(U1)` |

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
and its own actual `observed_at` and commit-bound `ingested_at`. Identity repair
never rewrites an existing revision's `observed_at`, `ingested_at` or
`declared_release_at`. D2 is metadata, not proof of when a correction occurred
or when its bytes became public. FIX1 visibility and FIX2 scope/zero semantics
remain unchanged.

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

FIX12 deliberately decides **nothing** about retries or whether redirect hops
count toward the request ceiling. A non-admitted response maps to
`SOURCE_UNAVAILABLE` *if it is the final response of the logical attempt*; what
happens before that is R13's territory.

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

## 11. Forty-seven cases, decided in advance

`FOMC01`–`FOMC82` in the JSON settle summer/winter releases, immediate release,
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
| `FOMC22` | unscheduled monetary action, different title | unresolved candidate; **zero prohibited** |
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

Invariants `F01`–`F32` state the same commitments in testable form. Revision 2
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
Independent counter-review of revision 10 remains pending; this fix authorizes
no implementation.

## 12. What this does not establish

Nothing has been captured, implemented or observed. No predictive edge, no
causal market impact, no economic significance.

`commercial_edge_established = false`.
