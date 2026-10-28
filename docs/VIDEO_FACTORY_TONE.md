# HyprL product and market videos (v2)

The factory now presents the service in a calm French product tone. The five
operator scripts approved on 2026-10-28 replace the v1 rotation verbatim. Their
source SHA-256 is `5023a8ca5ffd4bacb9f52e53b34bf26aeb34960faac09fc857caa1727ad1880e`.
The previous copy remains in `content_v1.py` with no production imports. Previous
video bundles are not reused: batch identifiers are `evergreen-v2-01` through
`evergreen-v2-05`. Discord messages and caption files say
“version présentation produit”.

The design follows the site's requested tokens: IBM Plex Sans regular and
semibold, background `#FAFAF7`, text `#121417`, accent `#2F5D8A`. Left aligned
typography, generous space and a restrained reading progress line carry the
presentation. Captions are rasterized into every frame, with a fixed disclaimer,
site address and Discord invite. No simulated prices or performance curves are
drawn. There is no voice or music. Video remains 720 × 1280, 25 fps, silent H.264,
30–45 seconds and strictly below 9,000,000 bytes.

## Content selection

Monday through Thursday: **Le point marché du jour**, from the observed member-v2
export's brief with exactly the current New York date and publicly sourced
calendar entries for the following calendar day. Friday: prefer **Le bilan de
la semaine**, otherwise use the market point. Without sufficient evidence, use
the approved v2 evergreen rotation. Tuesday and Thursday no longer override
the daily brief with an evergreen or research verdict.

Market cards preserve the brief's factual summaries. Source publication dates,
times in New York and domains appear on the relevant cards; full URLs and source
digests are in metadata and the publication text. A fact from yesterday's news
in today's morning brief keeps its original timestamp. A freshly generated
export does not make an old brief current. The renderer does not invent a close,
a price move or an explanation of causality. Repeated AI mentions in a brief
cause evergreen fallback rather than rewriting the underlying facts.

Calendar kinds supported are earnings, Fed appointments, CPI and employment.
A date-only earnings marker never acquires a fabricated release time. Digest-only
calendar evidence can recover its original public URL from the existing parsed
calendar and capture manifest, when `calendar_file` and `calendar_manifest` are
configured. The dataset digest, individual event digest, symbol, kind, date,
collection clocks and captured body hash must match. Nasdaq URLs are restricted
to the same date's earnings endpoint; Fed markers bind the FOMC calendar URL.
Captured bodies are never opened. Unknown source publication times remain null;
a received timestamp is not a publication time. Unresolved calendar evidence is
insufficient for this public format. Its omission does not imply that no event
is scheduled. A source and
each availability clock must be valid and causal; contradictory evidence fails
and alerts. No source URL is fetched by the factory.

Friday recaps pool the current Monday–Friday forecast sessions with matured
published scores, restricted to the regular morning equity panel, `a_v1_as_live`,
one-day horizon. Only identical session/symbol pairs available for both public
agents are compared. Previous weeks, unmatched rows and open horizons are
excluded. Each counted row needs a score digest and a matching public reveal,
decision identity and causal commitment clocks. Counts, denominators and Brier
scores describe forecasts against SPY, never portfolio returns or a claim of
advantage. Whole-week luck statistics are not inferred from cumulative series.
Empty weekly scores select the market point or evergreen, never invented results.
The observed member export is a trusted local input produced by the existing
pipeline; the factory does not independently recompute or fetch score preimages.

## Configuration and operation

Keep the private v1 configuration fields `output_dir`, `state_dir`, `snapshot`,
`webhook_env`, `notify` and `expected_commit`. Add `font_dir`, containing
`IBMPlexSans-Regular.ttf`, `IBMPlexSans-SemiBold.ttf` and the IBM SIL OFL 1.1
licence. Use the `scripts.video_factory.fonts` CLI offline to convert the site's WOFF2 files if
needed, or supply already licensed IBM Plex TTFs. No font is bundled in Git.
Font digests and the licence identity are recorded in each video's metadata;
record the source/conversion and licence digest separately when preparing fonts.
`HYPRL_VIDEO_FONT_DIR` is an optional local test override. `verdict_content` is
ignored by daily selection; the standalone research commands remain available.

```sh
python -m scripts.video_factory batch --config "$VIDEO_CONFIG"
python -m scripts.video_factory batch --config "$VIDEO_CONFIG" --publish
python -m scripts.video_factory daily --config "$VIDEO_CONFIG" --publish
python -m scripts.video_factory.fonts --source-dir "$FONT_SOURCE" --output-dir "$FONT_DIR"
```

Only font preparation needs the optional `fonttools[woff]` Python package.
Ordinary runs use the prepared TTFs through Pillow and do not need fontTools.

Review the dry-render previews and metadata before the explicit publish command.
Existing bundles must match the selected v2 content before reuse. Upload
reservations, two-attachment acknowledgements, durable daily receipts, single
render lock and fourteen-bundle retention remain in force. The schedule remains
17:05 America/New_York on NYSE trading days, with the UTC trading-job blackout,
one encoder thread and the existing service resource limits.

To upgrade the existing marketing automation, verify the full repository checks,
pin this branch's exact tested commit in a new private config, back up the three
factory user units, generate `install.units(checkout, python, config)`, verify the
staged units with `systemd-analyze --user verify`, then atomically replace only
those factory unit files and reload the user manager. Keep the existing state
directory to preserve daily delivery idempotency. The installer for a new setup
continues to refuse overwriting existing units. No trader unit, existing tool,
website, bot, system service or source collection is changed.
