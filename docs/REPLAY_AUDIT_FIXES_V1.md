# Replay audit corrections

Revision 8 supersedes the interrupted revision 7. Old specifications, freezes,
seals and stores remain unchanged and cannot be relabelled as this protocol.
Replay is **EXPLORATORY**: its design followed observation of the test window.
Only separately collected **LIVE_FORWARD** decisions count as prospective proof.

The configuration freeze binds one primary endpoint: `a_v1_as_live:1d` Brier
skill relative to a Laplace base rate from development labels. The freeze contains
the target counts and their digest before inference. Session block bootstraps
provide 95% intervals for each model and the paired difference. Reports retain
the full panel table, missing sessions, failures and trade exclusions. A winner
requires a paired primary interval excluding zero; replay remains exploratory
even if that condition holds.

Stocks enter on the first SIP trade in the one-second interval beginning at the
actual session open plus 30 seconds, including multi-session trades. Crypto
enters at the decision time plus 15 minutes. The dispatch window ends after
10 minutes. An entry at or before that boundary is rejected. This follows the
exchange through daylight saving time. Corporate-action windows are excluded;
unchecked actions or missing borrow flags retain the `GROSS_DIAGNOSTIC` label.
Both legs pay fees and slippage. The short leg pays 3% annual borrow on calendar
days, at least one day. An explicit hard-to-borrow flag excludes that short leg.

The anonymous development control codes headline symbols, names, executives,
products and news IDs. Ticker masks are case sensitive and word bounded. Dates
become relative day numbers; levels, including SPY and QQQ, are rescaled while
returns are preserved. Its model-facing prompt contains no replay label.

Every forward day publishes the snapshot digest and both decision digests
before the actual market open through the configured notifier's `ordres` route
and the fast-forward `ledger/forward-duel` branch. Failures have digests too.
The ledger contains only hashes and protocol/session identifiers. Scoring checks
the pushed ledger line, the confirmed Discord receipt, its deadline, the frozen
configuration, the first session and decision timestamps. Replay publishes its
test-start marker before dispatch and every response digest before scoring,
including invalid or partial responses. Long publications are split without
omitting hashes. Unknown deliveries cannot be scored.

The October memory probe uses a private `questions` list and textual `rule`,
with explicit numerical criteria. Its public artifact contains only the
preregistration digest, window identities, counts and rule. That digest must
exist in Git before either call. Calls have durable intents and never retry an
unknown outcome. A flagged model's test decisions and label maturities stop at
September 30. Non-flagging does not establish absence of memory.

The collector is GET only and bound to the collection and replay authorizations.
It records request URL hashes, response receipt times and cache digests; retries
consume request budgets. It stops on 4xx except 429. SEC hosts share a 5/s
limiter. Active and inactive assets retain their real UUID identity; identical
metadata under several UUIDs can share a symbol, while contradictory metadata
for a candidate block admission. Daily history begins in January 2025. Only
forecast names can request scoring anchors. Protected BTC/ETH data stay excluded.

News requires both `created_at < T` and `updated_at < T`. Only headline, source,
symbols and creation time enter the snapshot; later article bodies, summaries,
authors and URLs do not. Historical headlines remain the vendor's last version
edited before T, rather than a claim of original-vintage capture. This residual
risk is retained in every replay report.

Research scheduling consists of the night batch, pre-decision collection,
forward dispatch, pre-open publication retry, outcome collection, scoring and
weekly reporting. Services pin the worktree commit, have resource limits and
cannot place orders. Install only after checks pass and configuration/input
artifacts are committed. The first revised forward session is October 20; a
missing session is reported instead of being backfilled.

Subscription ceilings are 150 calls per family per night. Opus also shares the
150/day ceiling across replay, probe and forward; a same-day forward slot is
reserved. A usage limit stops that subscription. API billing is optional and
disabled by default. It requires a matching grant, a private key file, a durable
spend intent, a CLI dollar bound and known reported costs. Unknown spend and low
credit stop further API dispatch. No key is copied or enabled by this change.

Regression coverage is in `tests/replay_seal_forward/test_audit_*.py`, together
with the paired pipeline, forward and timer tests in that directory. Actual
collection, probe completion, input sealing and deployment are recorded in the
operator report; code coverage alone establishes none of those operations.
