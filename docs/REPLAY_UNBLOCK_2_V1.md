# Revision 10 — replay and forward admission

Operator decision 2026-10-19, before any replay answer was read or graded.
The authoritative specification is `replay_duel_spec_v6.json` (revision 10).
Previous specifications, freezes, seals and decision stores remain rejected;
there is no implicit migration. New replay state is `duel-v6`, new forward
state is `live-forward-v4`.

Probe 4 is closed **NOT RUN**. The four original/retry envelopes remain opaque,
unchanged and digest-bound by `replay_memory_probe4_closure_v1.json`. Admission
hashes their bytes without decoding or grading any answer. No probe dispatch or
retry is allowed. Both models' test sessions and label maturities end on
2026-09-30; development remains July–August.

The shared replay/forward decoder admits the CLI's unstable-feature and disabled
code-mode-host diagnostics only after checking the entire event stream for tool,
exec and web events. A real tool event still taints the call, even alongside a
benign diagnostic or quota error. Unknown error items remain rejected. The
regression test covers both call scopes, each with and without tool events.

SEC filings are uniformly UNAVAILABLE for every day and both models. Cached
filings are withheld too. The collector rejects SEC requests before a cache read
or HTTP dispatch; the historical SEC command returns the unavailable policy
without planning candidates or retrying. Variant d retains forecasts and abstains.

Asset records are compared across all captured receipts. Duplicate UUIDs with
identical metadata are compatible; any other contradiction excludes the symbol
uniformly from the predicted universe and news candidates. Each day's manifest
contains the total excluded count and symbol-set digest. Truly absent metadata
still blocks admission; it is not a contradictory record.

All 658 remaining labels from the completed once-only retry stay excluded for
both families, explicitly bound to the earlier digest. Development label targets
are recomputed from unchanged captured source objects for the new specification.
No model decision or prior freeze is relabeled.

Research dispatch remains serial and subscription-bound: replay only
00:30–05:30 UTC with a 150-call ceiling per family and usage-limit stops;
Opus also retains the stricter existing 150/day grant across replay, probes and
forward, reserving one call for a same-day forward session. Forward has one call
per family at 09:05 America/New_York, with no validation retry. The first intended
session is 2026-10-20 if all suites pass before 12:00 UTC that day, otherwise
2026-10-21. Intent alone does not arm a timer. A committed real input seal and
green suites are required before installation.

Each real session's snapshot and both decision/failure hashes must reach the
ledger and `#journal-des-ordres` before 09:30 New York. No synthetic receipt can
satisfy this requirement. Replay is EXPLORATORY; only LIVE_FORWARD is prospective
evidence. No orders, training or live-engine deployment are authorized.
