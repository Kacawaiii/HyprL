# FOMC DOM anchor evidence V1 — official structural observation

```
STATUS: EVIDENCE ONLY · NO CAPTURESPEC MUTATION

EVIDENCE ARTEFACT HASH
bd7d17d29c3f71efa24a503f9ffab75ff0ba2b051544dae25723daad745c6a3a

CAPTURED 2026-09-19 · www.federalreserve.gov ONLY · GET ONLY
5 statement pages at HTTP 200 · 1 failed attempt · 8 requests total

BINDS
  capture spec REV10  3dcf0a600797a9998651590389e6d933a8ae7dcbe832e8dff52cf0a40613509d
  provider rev3       e854ddb9cc8b5bc2634fe6ca6fc793289bafd6c94bb49dde71109e1d17341034
  design rev4         4d0d451494b7d4b7ea50552817f035c2da1b38a234b1e5e922a9ea4263726689
```

Canonical artefact: `docs/artifacts/fomc_dom_anchor_evidence_v1.json`.

FIX10 stopped with `INSUFFICIENT_LOCAL_EVIDENCE_FOR_DOM_ANCHOR`: the spec knew
which *values* to expect but not *where* in the HTML they live, and guessing a
selector was forbidden. This run supplies that evidence from the network, under
an explicit one-time authorisation. **It changes no spec.**

## 1. How the pages were chosen

No URL was invented. Every statement URL came from an official discovery surface
already cited in the committed artefacts:

| surface | role |
|---|---|
| `/feeds/press_monetary.xml` | already cited; recent statements |
| `/monetarypolicy/fomccalendars.htm` | already cited; offered 100 distinct statement links, 2021–2026 |

One statement URL (`monetary20260617a.htm`) was already cited directly. The rest
were hyperlinks read off the calendar page.

Rate discipline: **13 s** spacing, not 10 s. At exactly 10 s, seven requests can
land inside a 60-second window (t = 0…60) — which would breach the
six-per-rolling-60 s ceiling. 13 s caps it at five. Observed maximum: **5**.

| page | status | bytes | body SHA-256 |
|---|---|---|---|
| `monetary20260916a` | **000** (transport failure) | 0 | — |
| `monetary20260128a` | 200 | 82 109 | `d63df4e8…` |
| `monetary20251210a` | 200 | 82 679 | `4c8d26bc…` |
| `monetary20260617a` | 200 | 81 083 | `2363d432…` |
| `monetary20240320a` | 200 | 82 347 | `32c96759…` |
| `monetary20211215a` | 200 | 83 382 | `b5a25c80…` |

The 000 was a single deliberate attempt, no hidden retry, and is excluded from
positive evidence.

## 2. The structure, and it is strikingly stable

All five pages carry one container holding all three fields as consecutive
siblings:

```html
<div id="article">
  <div class="heading col-xs-12 col-sm-8 col-md-8">
    <p class="article__time">January 28, 2026</p>
    <h3 class="title">Federal Reserve issues FOMC statement</h3>
    <p class="releaseTime">For release at 2:00 p.m. EST
    <ul class="list-unstyled">          ← note: no </p>
```

| field | anchor | exactly one on | supported |
|---|---|---|---|
| primary page title | `h3.title` | **5/5** | **yes** |
| official statement date | `p.article__time` | **5/5** | **yes** |
| release line | `p.releaseTime` | **5/5** (element) | **no — see §4** |

`h3.title` text is byte-identical to the predicate on all five pages:
`Federal Reserve issues FOMC statement`.

## 3. Two negative controls, both decisive

**The document `<title>` is unusable on two independent grounds.** Its text is
`Federal Reserve Board - Federal Reserve issues FOMC statement` — prefixed, so it
fails the exact predicate. And the selector is **not unique**: every page carries
a *second* `<title>`, an inline SVG `<title>Lock</title>` inside the banner lock
icon. The R10 MEDIUM is now confirmed from the network rather than from memory,
and confirmed harder than R10 stated.

**`div#lastUpdate` is structurally separable but not value-separable.** It sits
outside `div#article`, directly under `div.row`, while `p.article__time` sits
inside `div#article > div.heading`. But its *value* equalled the statement date on
all five samples — so only structure can distinguish them, never a value check.
It stays forbidden as a causal timestamp.

## 4. The finding that blocks a clean freeze

**`<p class="releaseTime">` is never closed.** There is no `</p>`; the next markup
is `<ul class="list-unstyled">`.

A WHATWG-conformant parser implicitly closes the paragraph at `<ul>`, yielding
just the release declaration. A permissive parser does not, and keeps
accumulating — the share menu and then the entire statement body end up inside
the element. That is not hypothetical: it is why the first pass of this run's own
analyser found **zero** release-line candidates on all five pages while finding
the title and date immediately.

So the element is uniquely identifiable, but its **semantic text extent is
parser-dependent**. A FIX10 rule of the form *"extract the semantic text of
`p.releaseTime`"* would **not** be single-valued on the real pages — it would
reproduce the R10 defect in a new place.

What the evidence does support: the element's **first text node** is stable on all
5/5 pages — the release declaration followed by ASCII whitespace. Adopting
"first text node only, up to the first child element" is a *spec decision* that
this evidence backs; it is not itself an observation, so this run does not make it.

## 5. Normalization evidence

| anchor | entities | NBSP | LF | multi-space | combining |
|---|---|---|---|---|---|
| `h3.title` | no | no | no | no | no |
| `p.article__time` | no | no | no | no | no |
| `p.releaseTime` first text node | no | no | **yes** | **yes** | no |

Only the release field needs trim and ASCII-whitespace collapse. **No sample
exercised Unicode NFC, NBSP or HTML character references** in any of the three
anchors. A normalization pipeline is still required for determinism, but this
evidence must not be cited as demonstrating a need for it.

## 6. What FIX10 can now freeze without guessing

- `primary_page_title` := the single `h3.title` inside `div#article > div.heading`, cardinality exactly 1
- `official_statement_date` := the single `p.article__time` in the same container, cardinality exactly 1
- document `<title>` is **not** the classifier authority — prefixed *and* non-unique
- `div#lastUpdate` is structurally distinct and remains forbidden as a causal timestamp
- `release_line` := the single `p.releaseTime` in the same container, **with an explicit text-extent rule**

```
overall_evidence_sufficient_for_fix10 = false
blocking_item = release_line text extent
```

Two of three anchors are fully supported. The third needs one more spec decision,
which this evidence now makes safe to take.

## 7. Limitations

Five pages, 2021–2026, one page template — stability outside that range is **not**
established. No `For immediate release` sample: the 2015-03-18 statement is not
reachable from the locally cited discovery surfaces (the calendar covers
2021–2026) and inventing its URL was forbidden. Cardinality was **observed, not
guaranteed** — the Federal Reserve publishes no markup contract, so these anchors
are empirical and can change without notice.

These anchor names are structural observations for spec design. They are **not**
identity inputs and must never enter `source_item_id`.
