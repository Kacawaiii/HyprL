# GPT retirement: preregistration v10, execution v5

The October 16 operator decision retires GPT from the next run after a safe
deployment. Its canonical identity is
`5136ce360c9672b685e783448d31d7c87e2c8b925343b58e978fc5f3dd6a947c`.
The new artifacts and `retirement_spec.py` bind that identity. They apply to
am, pm, recovery, weekends, holidays and both base/news arms from October 16.

Only the Claude analyst and Claude reviewer run. GPT analyst, reviewed GPT,
consensus and GPT pre-flight budgets are zero. Claude role/retry/health limits
are unchanged; the decision aggregate falls from six to four per slot and arm.
No saved budget is reassigned. Rehearsals have no GPT step. The legacy
`gpt-preflight` command returns RETIRED without launching a model. Historical
schemas/parsers and grants remain valid for reading historical evidence.

Execution preserves existing sizing, thresholds, floors, caps, collision rules,
label anchors and exits. Equities use Claude KEEP at the existing size or Claude
DOWNGRADE at half size with the original probability and existing downgrade
breaker. Crypto uses a single Claude KEEP at half size, subject to the existing
long-only and 0.55 threshold. Consensus is retired. News remains shadow only.

The v4 ledger binding stays frozen so existing lots/intents keep their original
execution and exit contracts. V5 execution requires an explicit additive
`paper-register-gpt-retirement --operator-decision DECISION_FILE` registration
using the existing private authorization/runtime arguments. The registration
binds the prior journal binding, unchanged grants, v5 spec, v10 preregistration
and operator decision. It appends once under ownership, performs no broker
request, and does not rewrite any record. An absent or mismatched registration
rejects new execution; opening an old journal never migrates it. Existing safety
and exit operations remain available. A rehearsal also checks this registration.

New decisions/predictions bind preregistration v10; new order intents bind
execution v5. New scores and cohorts have the `gpt_retired_v1` prefix, preserving
historical scores and the frozen consensus confirmatory sample. Baselines use
one reviewed-Claude asset/horizon record in the new population. Scorecards and
cockpit snapshots expose retired role states while retaining historical rows.
The cockpit offers current Claude scores and historical consensus separately.

Release requires the full repository gates, branch and integrated exact-SHA CI,
and an operator-authorized guarded deployment. Exclude 11:30–13:50,
15:00–16:30 and 19:45–20:10 UTC; require all trader services idle, ownership
locks, consistent private database backups, an explicit registration and
verified timer restoration. Update the cockpit build/snapshot from the same
release. Require a live GET-only rehearsal GREEN afterward. Never restore old
database snapshots over appended journals. Rolling code back does not authorize
resuming GPT; hold decision dispatch until an authorized compatible release.
