# Prospective trader work budgets

The work-budget overlay protects independent AM and PM admission pools. It does
not edit the research/execution grants, the v10 preregistration, historical
stores, prompts, skills, or any output validator. GPT remains retired.

| Pool, per existing dispatch budget day | Admission tokens |
| --- | ---: |
| AM base | 190,000 |
| AM news | 190,000 |
| AM probes | 20,000 |
| PM base | 190,000 |
| PM news | 190,000 |
| PM probes | 20,000 |

Each production attempt reserves 95,000 tokens before model dispatch. Each role
has one such reservation per arm/slot/day. This deliberately removes automatic
production retries, including retries after invalid output and catch-up attempts
for a role already used that day. The reviewer reservation cannot be consumed by
an analyst retry. A probe reserves 5,000 tokens, at most twice per role/slot/day.
Existing grant call ceilings still apply. Reservations survive crashes, restarts
and runtime path changes through the existing shared dispatch bank; nothing is
refunded. Previously dispatched work without an envelope prevents same-day
activation from creating a fresh pool. Historical/synthetic providers remain
usable at synthetic dates without launching real models.
The existing after-hours session-day convention is preserved across UTC midnight.

Claude receives `--max-turns 8`. Production analysts can receive at most three
`WebSearch` and three `WebFetch` permissions; reviewers get one of each; probes
get none. These are tool invocation limits. A WebSearch invocation can itself
use provider-side search operations. Explicit ask rules route every web tool
permission through the host's stdio gate. The host counts before sending ALLOW
and kills the process group on an excess, replayed permission, unauthorized tool,
taint, deadline or protocol failure. Safe mode, empty settings sources, disabled
hooks, strict empty MCP configuration, tool restrictions and the original full
JSON schemas remain in force. Only a complete structured result reaches the
unchanged schema, citation, future-source, URL, universe and reviewer checks.
The prompt byte ceiling is 260,000; the output ceiling is 32,768 tokens.

**Admission envelopes are not a hard provider token limit or a subscription quota
reservation.** The CLI cannot reserve the subscription window. It reports helper
usage at completion, after that work has happened. Receipts sum input, cache
creation and output across unique `modelUsage` entries, including web helpers;
cache reads are recorded separately. Aggregate `usage`, nested iterations and
thinking subcounts are never added again. Reported overruns, missing totals, or
a reservation left without a receipt block further dispatch in that slot. A
single in-flight request can exceed its admission estimate before the host sees
usage. Missing usage is explicitly marked incomplete, not treated as zero.
Other subscription users and a provider 429 remain outside these local pools.

`trader_work_budget_spec_v1.json` is a new hash-bound resource-method overlay.
It is not an implicit migration or execution capability. The operator must
review a separate `trader-agent-work-budget-amendment-v1.json`, with `status`
`APPROVED`, the exact spec/base-grant/parent-preregistration hashes, and explicit
approval, effective and expiry timestamps. An absent approval blocks real model
launches from 2026-10-24 00:00 UTC; a present draft, expired or mismatched approval
always blocks. The approval's canonical digest accompanies every reservation.
An append-only run binding also records the spec/amendment/parent-registration
hashes and the unchanged context hash, without changing the decision schema.
Choose a fresh UTC day to avoid unaccounted legacy work. Preserve the resource
method boundary when comparing future observations with the old population.

The operator handoff must include the amendment draft, exact release SHA,
serial trader/required-suite results, copied-context evidence, and guarded
fast-forward commands. Offline synthetic usage at the admission boundary proves
accounting and refusal behavior; it cannot establish real counterfactual token
savings, future model completion quality, or PM subscription availability.
