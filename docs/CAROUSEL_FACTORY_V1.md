# Carousel factory v1

The isolated marketing factory produces Instagram PNG albums (1080 × 1350),
TikTok photo slides (1080 × 1920), a four-image X selection and a French caption.
Pillow draws IBM Plex Sans hooks and IBM Plex Mono calendar numbers. Every slide
includes the information disclaimer and site footer. Source lines distinguish
publication time from collection time. Text must fit at 28 px or larger; overflow
fails without leaving a partial bundle. The video implementation is preserved.

## Editorial selection

- **Le point marché**: five slides, exactly three observed items from that day's
  brief and calendar, followed by tomorrow's sourced agenda. An evening edition
  explicitly recaps the dated morning brief. Missing items select evergreen.
- **Chance ou pas ?**: seven slides only when the original offline study returns
  `DONE`, binds the fixed plan and identifies its input manifest. The yearly
  figure compares Fridays, other days and the empirical random-day baseline.
  A blocked study never becomes a verdict. No price data is opened here.
- **Ce que dit le document officiel**: six slides from an optional reviewed local
  `carousel-reviewed-documents-v1` projection. Each observed document needs a SEC
  archive URL, source clocks/hash, a locally reviewed well-known US symbol and
  exactly four facts with section citations. A feed headline is insufficient.
- **Les résultats US de la semaine**: the observed calendar's exact event count,
  one slide per represented weekday, and a close. All event symbols are included
  in Instagram/TikTok. Unknown release times stay unknown. The X set selects key
  slides from the full album and is explicitly a selection.
- **Four evergreen products**: a five-slide service tour, five-slide public-proof
  walkthrough, six-slide screenshot critique with its fairness pivot, and a
  five-slide morning-brief demo. The latter two data-backed demos require a real
  reveal/excerpt; unattended rotation uses the service tour when either is absent.

Brief statements are quoted verbatim from the existing observed member export.
The adapter adds reading context rather than inferred market moves. Calendar
date markers are bound to the existing parsed dataset and source manifest through
the video adapter; captured bodies are never opened or fetched. Real reveal
examples validate matching commitment hashes and clocks before the actual NYSE
opening and after its close, including DST and shortened sessions. The observed
export is trusted; this factory does not independently fetch/recompute reveal
preimages or scores. Full hashes and URLs accompany the caption/metadata.

TikTok trims the screenshot critique to five slides while preserving the fairness
pivot. Other narratives already use their minimum requested slide count. X uses
cover plus three key slides; the research selection retains chart and verdict.

## Delivery, cadence and retention

One acknowledged Discord message contains all Instagram slides as an album,
`caption.txt`, `tiktok.zip` and `x.zip`: at most ten attachments, each below 9 MB.
Messages are labelled **carrousels v1**, with mentions disabled. The caption has
its own hook, context, disclaimer, invitation and three to five hashtags.
Nothing is posted directly to Instagram, TikTok or X.

The generated **user** timers use Europe/Paris, follow DST and skip catch-up:

| Slot | Cadence |
| --- | --- |
| Market / evergreen | NYSE weekdays, 18:30 Paris |
| Weekly earnings / evergreen | Monday, 08:00 Paris |
| Chance verdict / evergreen | Wednesday, 18:45 Paris |
| New eligible reviewed filings | NYSE weekdays, 18:50 Paris |

The new explicit carousel cadence replaces the old video/post-close schedule.
Each invocation only runs in its ten-minute slot. Heavy video encoding is absent;
resources are Nice 19, idle I/O, CPUQuota 35%, MemoryMax 300M, TasksMax 16 and a
five-minute timeout. File locks serialize selection and rendering/upload. Only
this factory's user units are installed; the old video timer is disabled, with
its code, units and receipts preserved. Existing trader tools/units are untouched.

Keep the latest thirty owned delivered posts both locally and in the carousel
Discord history. Pruning deletes only acknowledged message IDs recorded by this
factory through the drafts webhook, then removes their bundles. Durable delivery
receipts survive pruning. An unacknowledged old bundle blocks retention.

Every image, pack and caption is hashed before delivery. Upload reservation is
durable before HTTP; timeouts or incomplete acknowledgements require operator
reconciliation, never a blind retry. Deletion has the same uncertainty protection.
A refreshed source cannot replace a delivered slot. Changes to an existing
content-addressed bundle are rejected. OnFailure uses the existing notifier.

## Private configuration and operation

Use external configuration fields `output_dir`, `state_dir`, `snapshot`,
`webhook_env`, `notify`, `font_dir`, `calendar_file`, `calendar_manifest`, and the
exact tested `expected_commit`. Optional fields: `study_result` (the original
aggregate study result) and `documents` (reviewed four-fact projections).
No private configuration, captures, fonts or rendered data enter Git.

`font_dir` contains licensed `IBMPlexSans-Regular.ttf`,
`IBMPlexSans-SemiBold.ttf`, `IBMPlexMono-Regular.ttf` and `LICENSE-OFL.txt`.
Prepare these offline from the site's existing WOFF2 assets, recording conversion
and licence digests. Hook strokes reinforce the installed semibold face.

```sh
python -m scripts.carousel_factory batch --config "$CAROUSEL_CONFIG"
python -m scripts.carousel_factory batch --config "$CAROUSEL_CONFIG" --publish
python -m scripts.carousel_factory run --slot daily --config "$CAROUSEL_CONFIG" --publish
python -m scripts.carousel_factory verify --config "$CAROUSEL_CONFIG" --bundle "$CAROUSEL_BUNDLE"
```

`install.units(checkout, python, config)` generates nine staging files. Verify
them with `systemd-analyze --user verify` before installing only these new user
units, disabling `hyprl-video-factory.timer`, reloading the user manager and
enabling the four carousel timers. The task authorizes this marketing switch;
the single-owner trader-engine deployment restriction remains in force.

Before pushing run the repository-mandated pytest suite, the video/carousel
tests, Ruff and `git diff --check`. Review actual IBM Plex renders, format/pack
dimensions, caption sources, acknowledgement IDs and the enabled timer clocks.
No web files change, so npm checks do not apply. Track saves and completion when
the operator publishes the drafts; likes do not determine editorial selection.
