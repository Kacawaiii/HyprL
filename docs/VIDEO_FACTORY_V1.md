# Silent French video factory v1

This is a research/marketing tool. It creates caption cards locally with Pillow,
encodes a silent 720 × 1280, 25 fps H.264 MP4 with the installed ffmpeg, validates
the result with ffprobe, and sends the MP4 and caption text to the existing drafts
webhook. No model calls, broker requests, market-data requests, music, synthetic
voice, or changes to the trader, site or Discord bot are involved.

The five operator-approved evergreen scripts are preserved verbatim in
`scripts/video_factory/content.py`. Source document SHA-256:
`74cb033b8bf037c750b92b19909d8210146e1efe04e25a957813f10c3b457a7d`.
Every input card carries “Information, pas un conseil en investissement”. Every
bundle has a video, preview, caption text, SRT, and JSON metadata with a digest,
duration, size, sources, invite and conditional AI-voice labelling reminder.

No audio licence is needed for this silent version. Pillow/ffmpeg create the
original cards. The local DejaVu Sans font is covered by the permissive Bitstream
Vera/DejaVu font licence (commercial document use allowed); the local package's
copyright file is the installed licence evidence. No font files are redistributed
by this repository. See [DejaVu licences](https://dejavu-fonts.github.io/License.html).

## Private configuration and use

Supply a private JSON object through `--config "$VIDEO_CONFIG"` with these keys:

| Key | Meaning |
| --- | --- |
| `output_dir` | Dedicated private directory for the latest fourteen video bundles |
| `state_dir` | Dedicated private directory for per-day delivery records |
| `snapshot` | Existing observed member-v2 export, read only |
| `webhook_env` | Existing private env file containing `DISCORD_WEBHOOK_BROUILLONS` |
| `notify` | Existing `hyprl-notify` executable, used only for failure alerts |
| `verdict_content` | Optional private reviewed verdict content JSON |
| `expected_commit` | Exact deployed Git revision; fail on drift or tracked factory edits |
| `ffmpeg` | Optional executable, defaults to installed `ffmpeg`; ffprobe is required |

The CLI is dry by default; posting requires the explicit `--publish` flag.

```sh
python -m scripts.video_factory plan --output "$STUDY_PLAN"
python -m scripts.video_factory batch --config "$VIDEO_CONFIG"
python -m scripts.video_factory batch --config "$VIDEO_CONFIG" --publish
python -m scripts.video_factory blocked-verdict --config "$VIDEO_CONFIG" --publish
python -m scripts.video_factory daily --config "$VIDEO_CONFIG" --publish
python -m scripts.video_factory.install --config "$VIDEO_CONFIG" --enable
```

Install with the Python environment containing Pillow and
`pandas_market_calendars`; numpy/scipy are needed only for the offline study.
Installation refuses to overwrite existing units and creates only three new
user units belonging to this factory. The task explicitly authorizes these new
research/marketing units. No existing executable or unit is edited.

## Daily selection and operation

The timer runs at 17:05 America/New_York, currently 21:05 UTC and 22:05 UTC in US
standard time. The existing reveal runs at 16:15, score at 16:45, and member export
every half-hour; the five-minute buffer after 17:00 avoids knowingly consuming
the export preceding the score. NYSE holidays and weekends are skipped. Catch-up
is disabled. The job refuses the 11:30–17:15 UTC trading-job interval and reserves
ten minutes before that interval for each bounded render/upload operation.

On ordinary days, both public agent reveals for the current session are required.
Only observed equity forecasts from the regular morning, actions panel and
one-day horizon are counted. The latest common *forecast session* with published
scores is labelled explicitly, since scores often mature after the decision
day. No-decision reveals and pending scores are stated plainly. Results are
forecast classification counts and Brier scores against SPY, never trading P&L.
Reveal identity, commitment digest, causal timestamps, duplicate keys and valid
probabilities are checked. Sources include public reveal URLs and the export
digest. A published, matching cumulative luck-tail field is displayed only if
available; it is never invented or described as a probability of talent.

Tuesday rotates through evergreen scripts. Thursday uses the supplied reviewed
verdict when available. Missing, stale or synthetic daily evidence selects an
evergreen; malformed or contradictory evidence fails with an alert. The exporter
is not invoked or modified by this tool.

One daily delivery receipt survives video retention. A process lock serializes
rendering. HTTP uploads use the existing webhook secret in memory, disable
mentions, and require acknowledgement of both attachments. A reservation written
before POST prevents blind retries after an uncertain timeout. Reconcile such a
receipt in Discord before manually clearing it; do not delete it to force a retry.
Only owned, acknowledged video directories are eligible for retention deletion.
Undelivered oldest files cause a visible failure, rather than silent deletion.

The service uses Nice 19, idle I/O, 35% CPU quota, 400 MB memory and a ten-minute
timeout. Encoder threads are capped at one. The CLI alerts through `hyprl-notify
alertes`; an independent user OnFailure unit covers process death and timeouts.
Rollback is `systemctl --user disable --now hyprl-video-factory.timer`; remove only
the three factory unit files if removal is desired. Keep receipts and outputs.

## “Chance ou pas ?” #1

The historical analysis is offline and exploratory. Its plan and canonical hash
are recorded before reading any intraday observations. The fixed local range is
2018-01-01 through 2026-08-31, safely before the entire protected interval.

The claim under test is a *strict later sweep* of the Asia low. Sessions are
Europe/Paris: Asia [02:00,11:00), London [09:00,18:00), New York [14:00,23:00).
The outcome is any price below the Asia low in [11:00, next local midnight).
The London overlap before 11:00 belongs to the Asia reference, not the outcome.
The event is different from the daily low occurring during Asia, as worded in
the approved teaser. DST uses the first ambiguous 02:00 in autumn and advances
a nonexistent spring 02:00 to the first valid instant. No gaps are filled.

`study --manifest "$CANDLE_MANIFEST" --output "$STUDY_RESULT"` accepts existing
Coinbase six-field `[time, low, high, open, close, volume]` intraday candles only.
The private manifest binds `product`, `source`, `plan_sha256`, and `shards`. Each
shard declares `file`, `sha256`, `start`, `end_exclusive`,
`granularity_seconds` and the original `authorization_id`. All declared bounds
are checked before any candle file is opened; the entire input range must be
before September 2026. Hashes, UTC grids, OHLCV consistency and duplicates are
checked. Daily candles cannot establish the intraday event and are rejected.
The original collection authorization and provenance must be reviewed by the
operator; a manifest field does not authorize new network collection.

The output reports Wilson 95% intervals by year and all seven weekdays; Fridays
are compared with the six pooled other days and every named weekday. A uniform
random-day baseline samples the actual yearly observations without replacement,
matching Friday's yearly sample sizes. “Random” is not assumed to mean 50%.
The primary difference interval resamples paired ISO-week blocks. A separate
binomial test checks compatibility with a 60% Friday rate. These are different
questions. Descriptive binomial intervals assume independence; weekly blocks do
not resolve dependence lasting longer than a week. No year is selected for its
significance, no strategy is trained, and no trading profitability is inferred.

On initial installation the only research-specific Coinbase grant had expired,
and existing local BTC candles were daily. No valid Binance research grant was
found. No official endpoint was contacted. The first verdict is therefore
**BLOCKED / non vérifiable**: no rate, CI, yearly comparison, or advantage over
chance can honestly be reported yet. The sixth video and its Discord post state
this limitation. Completing the empirical verdict requires a valid scoped
collection grant (budget and expiry included), or already-authorized intraday
archives with provenance, excluding the holdout.
