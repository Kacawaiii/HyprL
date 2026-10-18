# Declared replay method and prospective forward duel

Revision 6 is additive. `replay_duel_spec_v2.json` binds the operator's method
declaration by canonical hash. Revision 5 specifications, configurations and stores
are preserved and rejected by the new execution. New replay state uses `duel-v2`;
prospective state uses `live-forward-v1` and provenance `LIVE_FORWARD`.

Reconstructed historical inputs and every historical result are labelled
`APPROXIMATE_POINT_IN_TIME`, with four residual risks: historical vendor price
corrections, headline edits within 24 hours, survivorship where vendor history is
absent, and vendor news coverage gaps. A declaration does not establish original
vintage. No assertion of unrevised archival evidence is manufactured.

Prices are SIP raw bars with timestamp and completed interval strictly before T.
Indicators never use incomplete daily aggregates. News has `created_at < T` and
`updated_at - created_at <= 24h`; an update after T within that allowance is admitted
under the declared approximate method. Only headline, source, symbols and creation
time affect selection or model input. Bodies, summaries, author and source URLs are
discarded. Citation URLs are generated opaque identifiers. EDGAR acceptance uses
the inherited conservative UTC/New York upper bound, strictly before T.

The fixed configured v1 universe needs dated trading history before July 1. News
company candidates include inactive equities, require at least 200 completed
sessions, and require a last-50-session arithmetic mean of close times volume of at
least $10M. The former $5 price floor and $1B upper liquidity band do not belong to
this new method. The inherited exclusion of fixed-universe names and known ETFs,
40-name limit, headline-keyword rank then recency, and three headlines per name
remain explicit. This is a liquidity/news proxy, with no market-cap assertion.
The primary-only variant requires an exact accession in a permitted cited headline
and a matching accepted 8-K. Missing primary evidence abstains without deleting
forecasts. A later article body or URL cannot supply evidence.

## Private archive interface

The CLI performs no source HTTP requests. An authorized collector supplies a
`replay-vendor-request-index-v1` JSON object containing an `objects` array. Each
entry has `origin`, `path`, `params`, `cache_key` and `sha256`. The cache key is
SHA256 of origin + path + the sorted URL-encoded query. Its compressed cache object
has `body`, `received_at` and `body_sha256`, in a `cache` directory beside its index.
Historical, per-session forward and forward-outcome caches therefore have separate
namespaces: refreshing a live asset response cannot overwrite a sealed replay object. The file
digest binds the exact compressed object; the retained original body hash is not
misrepresented as independently reconstructed original bytes.

Supported data is vendor-shaped: stock/crypto daily bar dictionaries, news arrays,
active and inactive asset arrays, and SEC submissions with zero-padded CIK strings
and parallel filing arrays. Asset listing dates are not invented: Alpaca assets do
not provide such a field. Candidate-history queries must cover at least 550 prior
calendar days through T, with complete pagination, to distinguish missing history
from legitimate short history. Complete all-symbol news queries must cover both
the prior close and 72-hour fixed-universe window, including long weekends.

The task does not extend an HTTP authorization. In particular, an asset-list
endpoint absent from the existing replay grant is not requested. Missing news,
asset or price coverage is a blocker. Missing SEC evidence is explicitly
`UNAVAILABLE`; the primary-only variant abstains and retains the candidate's
forecasts, rather than pretending that no filing exists.

## Seal and replay sequence

Run `python -m scripts.trading_lab.replay_seal_forward.cli` with private data-root
and authorization arguments (or their environment variables), then these commands:

1. `freeze`: write the new configuration freeze; commit before any inference.
2. `build-inputs`: read the private `archive_index.json`, construct both bundles
   for every dev/test session, and install them with a single final rename.
   A failure produces no successful partial manifest. The private input manifest
   binds every bundle file, source cache object and method declaration.
3. `seal-inputs`: emit the public digest-only seal. Commit it before any dev or
   test call. The seal artifact is excluded from the execution hash.
4. Let `night --expected-commit SHA` run only 00:30–05:30 UTC: independent 60-call
   nightly caps, subscription quota stops, no tools/web, identical inputs for both
   models. Both dev runs and their anonymized audits must be scored before either
   once-only test claim. Test rates remain frozen from dev. No test tuning.

Bundles cannot be rebuilt in place, including before a test. An incomplete staging
directory after a crash is never consumed. Changed compressed objects, declarations,
bundles, execution, model identity, prompt or configuration fail admission.

## Forward research schedule

`install_seal_forward_timers.py` installs only research user units from a clean
exact commit, validates all unit files before replacement and backs up existing
research files. It never modifies operator bin tools or trader services.

- Exchange weekdays at 09:05 America/New_York: `forward`. Holidays are checked
  against the pinned exchange calendar. First admissible day is October 19, 2026.
  Daylight-saving transitions are handled by the named timezone. There is no
  persistent missed-run backfill; dispatch closes at 09:15 local time.
- Weekdays at 16:45 local: `forward-score`, using a separate offline outcome index.
- Friday at 17:15 local: `weekly`, sending only to configured notify channel
  `brouillons`. It requires scores refreshed within two hours. Durable delivery
  intents prevent duplicate posts after uncertain delivery.

Each session reads `live-archive/SESSION/archive_index.json` from the private data
root. Every indexed response must have been received by T. The snapshot is stored
before inference and shared unchanged by both models. Each family gets one combined
analyst/news call and no reviewer or validation retry. Probabilities are mandatory
even for abstentions. A reserved call with unknown outcome is not dispatched again.
Forward calls have separate one-call counters but honor shared subscription stops.

Forward scoring uses the inherited raw fills and SPY-relative proper-score targets;
immature horizons remain pending, crypto 5d is excluded as declared. The outcome
request index is `live-archive/outcomes_index.json`. Missing fills or marks block
scoring. The weekly table reports cumulative paired counts and per-panel Brier and
log loss, with no winner claim. Historical replay and forward records never mix.

The delivery is done only when the real input seal is committed and both real
forward model decisions have completed once. Synthetic success cannot satisfy it.
