# Trader work budget revision 2

Revision 1 made a normal AM analyst fail at its fourth search and prevented
recovery with its single-attempt envelope. It remains immutable historical
reference. Revision 2 binds a new spec hash and separate operator amendment;
old approvals and same-day reservations cannot be reinterpreted as revision 2.
PM dispatch remains retired under the October 25 decision.

The October 21–23 AM transcript receipts contain 12 production attempts across
base and news. Analyst maxima were 21 reported CLI turns, 13 reported helper
search requests and 348,505 new-work tokens. Reviewer maxima were 3 turns,
zero reported searches and 128,903 new-work tokens. These historical result-only
transcripts do not report a reliable count of WebFetch invocations. A helper
search count is not necessarily identical to a host WebSearch invocation.
Public evidence records counts and byte digests only.

| Research ceiling | Analyst | Reviewer |
| --- | ---: | ---: |
| Unique assistant messages | 48 | 48 |
| WebSearch invocations | 32 | 8 |
| WebFetch invocations | 32 | 8 |
| Observed main-model new tokens | 750,000 | 750,000 |

A tool, turn, observed-token or research-time ceiling ends research and starts
one tool-free completion. The host terminates the research process before an
excess tool receives ALLOW. The completion receives the unchanged business
prompt and bounded observations, with explicit instructions to give the full
structured answer, preserve causal/citation rules and abstain without evidence.
It has empty tools/allowedTools; only the CLI's schema-completion mechanism is
available. The host also refuses any unexpected web permission in this phase.
No native `--max-turns` is installed: that flag can discard a complete answer
while the CLI is still finishing its schema output.

Both transport phases belong to one logical dispatch and share its original
wall-clock deadline and admission envelope. There is at most one extra CLI
process for completion, not another independent analysis or retry. This method
change is explicitly part of the operator amendment. Research reserves the
smaller of 120 seconds and a quarter of the call timeout for completion. It
never extends the grant/deadline. The original prompt is limited to 520,000 UTF-8
bytes, observations to another 520,000 bytes, and output to 32,768 tokens.
Omitted observations are counted and cannot be treated as evidence.

A production dispatch reserves 1,000,000 admission tokens. Each role retains
both original grant-authorized attempts, including validation retries/recovery:
4,000,000 per arm, 8,000,000 across base/news, plus 100,000 for rehearsal
(25,000 per probe, twice per role). These generous envelopes are runaway
admission ceilings, not targets or subscription reservations. A reservation is
never refunded. A prior incomplete/oversized usage receipt does not consume the
independently reserved reviewer/recovery capacity. Existing daily dispatch
ceilings, reviewer requirements and all output/source/execution checks remain.

Successful usage totals include model helpers once and cache reads separately.
Interrupted research reports observed main-model usage as incomplete; unknown
helper usage is never claimed to be zero. Provider-side helper work is only
reported at completion, so these envelopes cannot guarantee a token bill or
subscription availability. Quota, taint, invalid output, missing authorization
and actual decision deadlines remain genuine failures. Reaching a research
ceiling itself produces a completion, never a budget exception or fabricated
investment answer. The final model answer still has to pass every validator.

Every uncached rehearsal exercises the zero-remaining-tools transition and
records its spec identity, completion reason and zero final web permissions.
The cache binds the revision and final command, so an old unbounded success
cannot hide a broken completion path. Decision schemas and frozen skills stay
unchanged; transport completion evidence is a separate append-only record.

Deployment requires a reviewed `trader-agent-work-budget-amendment-v2.json` and
a fresh UTC dispatch day. Retain the original grants and shared bank. Do not
reset counters, migrate old reservations or restore historical journals to
activate the new method. Only the single-owner operator installs units/tools.
