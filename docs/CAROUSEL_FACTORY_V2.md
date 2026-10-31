# Carousel factory v2

V2 replaces the v1 renderer and its automatic evergreen fallback. The video
factory stays unchanged. No official source is fetched by either v2 adapter.

## Design and assets

Instagram and X use 1080 × 1350; TikTok photo mode uses 1080 × 1920. Each
immutable bundle contains the three image sets, caption.txt, TikTok/X ZIPs,
font/licence digests, file digests and measured text layouts. The edition is
`carrousels v2`. The uploader takes this label from the verified bundle, so
reviewed files keep their edition when delivered later.

The prototype palette is navy `#0A0E1A`, ink `#F5F7FC`, accent `#8FB4FF`.
Large claims/numbers, series pills, a drawn arrow in the accent action block,
a full accent flip for a key slide, and horizontal bars replace the old layout.
Bars share a zero baseline and a 100% scale. IBM Plex Sans Regular/SemiBold
sets all text, including digits and percent signs; Mono is reserved for the
page counter. X preserves all three daily facts within four images. Its calendar
uses up to three detail images with two day columns at most. Calendar slides
use French dates, feed names with tickers (ticker alone if absent), and at most
six companies per day followed by the remaining count. Feed order is retained;
no size ranking is inferred. The caption and source evidence preserve the full
count and list. Unknown-hour/modification notes appear only on the cover and
in the caption. Disclaimer and site appear on every
slide. Text that cannot fit at the configured minimum is refused atomically.

`brand.logo` optionally names a local HD image; null keeps the typeset HyprL
mark. Logo paths remain private; metadata records only its digest. Changing a
logo creates a new batch identity and cannot silently reuse an old rendering.
IBM Plex Sans/Mono and the existing OFL licence are required; no font download
is performed. Keep state and generated assets outside Git.

When present beside the producer manifest, existing
`calendar-sources/nasdaq-YYYY-MM-DD.body` files supply company names and the
Nasdaq before/after-market hints. The adapter checks the complete body digest
against the recovered source, then matches the selected date and ticker. It
never expands the selected universe, fetches a body, or turns a session hint
into an exact timestamp. Missing files leave ticker-only entries; mismatched
digests stop rendering. The slide footer credits the Nasdaq calendar with its
actual collection date; full URLs and identities stay in the caption/metadata.

The four product hooks and BTC wording follow the reviewed prototype. Only
those exact supplied long hooks are exempt from the usual eight-word cover
limit. Cover actions describe the next slide. Advice and per-fact number checks
remain unchanged.

## Daily edition: facts, then copy, then verification

1. Load a fresh observed member export (maximum 24 hours old). Resolve the
   optional producer `calendar_config` pointer on each invocation. Date-marker
   recovery still binds the dataset, individual event, source digest and clocks.
   A source collection clock never becomes a claimed publication clock.
2. Select at most three actual events/releases for the edition's New York date:
   earnings in the explicit US-company editorial scope, Fed/CPI/NFP releases,
   SEC large-company headlines from the radar, or facts from the existing
   reviewed-document projection. Calendar earnings require the Nasdaq calendar
   URL for that exact day. Fed, BLS and SEC references require their exact
   official hosts and appropriate paths. Old headlines, unsupported companies,
   unavailable sources and mismatched dates are excluded. Macro availability
   still depends on the upstream verified projections actually containing it.
3. If there are no facts, produce nothing. No brief or tomorrow agenda is needed
   to publish an otherwise eligible fact. Counts are derived, never padded.
4. The editor reads local copywriting/social skills, their copy blacklist and
   carousel framework, and the configured private product context for style.
   Only the facts JSON licenses factual claims. The selected framework is
   Value-Stack. Context prices/examples do not license numbers.
5. Reserve one Sonnet call durably for the daily edition before starting
   `claude -p`. Disable tools, MCP servers, settings sources, hooks, slash
   commands and session persistence; cap the turn count at one. Use one process
   invocation, no retry. Failures, usage limits, malformed output, timeouts and
   interrupted pending reservations consume that edition's reservation.
6. The UTC exclusion window is `[11:30, 17:15)`. Check both the requested and
   actual launch clocks; limit a pre-window subprocess timeout to the remaining
   seconds before 11:30. A valid cached draft may be used during the window
   without a new call. `editorial_enabled` defaults to false.
7. Require strict output shape, exact fact IDs/order/count, short copy and a
   real subject in the hook. Check numeric/date/time tokens and tickers against
   each cited fact, including French number words and lowercase cashtags.
   Reject advice words, invisible/non-ASCII numeric evasions, relative dates,
   changed unknown hours, jargon and unsupported numbers. Source URLs/identities
   do not license arbitrary numbers in the copy. Check the hook against the
   selected facts and derived count. Source lines, edition date, final CTA and
   caption are then supplied by code.
8. Any rejection means no post. Stable facts can reuse accepted copy after the
   snapshot refreshes; changed selected evidence means no second call that day.

These are conservative lexical checks, not a proof of semantic entailment.
They cannot establish that a qualitative paraphrase preserves every nuance.
Feed integrity and human draft review still matter. A source headline alone
also cannot become four document facts; the document series keeps its reviewed
citation requirement.

## Cadence, receipts and migration

- Prepare daily copy at 11:15 UTC on exchange days. This allows the Paris summer
  posting slot to use a previously verified draft without calling the model
  during the exclusion window.
- Daily: weekdays 18:30 Paris. Calendar: Monday 08:00 Paris. New study:
  Wednesday 18:45 Paris. New reviewed document: weekdays 18:50 Paris.
- Evergreen: separate Friday 19:15 Paris slot, at most one ISO-week reservation.
  Reserve before upload; an uncertain upload consumes the week. No missing
  daily/calendar/study/document is replaced by a product post.
- Timers have no catch-up. Keep the existing locks, pinned-revision check,
  uncertain-upload reconciliation, exact attachment acknowledgements and
  retention of thirty owned posts. Retention recognizes both v1 and v2 IDs.
- Generated user units include two new slots (`prepare`, `evergreen`); generating
  files does not install or activate them. The runtime keeps low priority,
  35% CPU, 300 MB and bounded time/process limits. Real CLI resource usage is
  unproven until an allowed operator execution succeeds.

Reuse the existing operational output/state directories for a runtime upgrade;
do not discard old receipts or enable two independent factories. New v2 slot IDs
cannot republish a day that already has a delivered shared slot receipt. The
explicit review batch is the requested exception allowing four product drafts
together; it is not the scheduled evergreen rotation.

`batch` renders the four products only when their real examples exist, an
eligible calendar and the already-public BTC study. On a weekend it previews the
next week's calendar with explicit dates. It does not render an old daily brief
under today's label. A missing product proof/excerpt is skipped instead of
silently duplicating another product. Optional `excerpt_translations` maps the
SHA-256 of exact original summary text to a locally reviewed French translation;
extra numbers, tickers or advice are refused and the slide is marked translated.
This is an editorial translation, not an automatic translation service.

The default BTC carousel uses price-free aggregate counts from public study
revision `98c7189b207e857777e188318b7248e87616ddd1`. Evidence digest:
`b7583307e2b785702a06cb935c958c3f9d150e04f8175195ad2aa36d1972064e`.
It shows 251/450 Fridays versus 1567/2701 other days, and 1818/3151 for all days.
The prototype's “Un jour au hasard” label denotes that empirical all-days
frequency, as explained in the caption; it is not a newly simulated draw.
The visible verdict is “Le vendredi n'a rien de spécial.” Intervals, p-value, limitations
and immutable method links are in the caption. This imports received public
counts; it neither reruns the study nor opens candles or protected data. The
legacy result adapter retains its original plan-hash requirement. A previously
delivered study is not automatically repeated on the next Wednesday.

## Private configuration additions

```json
{
  "brand": {"logo": null},
  "calendar_config": "OPERATOR_SUPPLIED_CURRENT_PRODUCER_CONFIG",
  "editorial_enabled": false,
  "claude_binary": "claude",
  "skills_dir": "OPERATOR_SUPPLIED_SKILL_DIRECTORY",
  "product_context": "OPERATOR_SUPPLIED_PRIVATE_CONTEXT",
  "excerpt_translations": {},
  "public_study": true
}
```

These extend the existing private config; existing snapshot, font, output,
state, webhook-env and expected-commit fields remain required as applicable.
`calendar_config` is optional; absent it, the explicit calendar file and manifest
are used. The pointer convention is the current producer generation layout:
`inputs/calendar.json` and its sibling `calendar-refresh/calendar-source-manifest.json`.

After tests and operator network permission, deliver a reviewed bundle exactly:

```bash
python -m scripts.carousel_factory verify --config PRIVATE_CONFIG --bundle REVIEWED_BUNDLE
python -m scripts.carousel_factory deliver --config PRIVATE_CONFIG --bundle REVIEWED_BUNDLE --publish
```

Delivery verifies that the bundle belongs to the configured output directory
and uses its existing images/content. It does not reread the live export or
rerender the batch. Preserve and reconcile any pending receipt before retrying.
A timeout never justifies a blind repeat.
