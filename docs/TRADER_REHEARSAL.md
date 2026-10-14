# Daily trader rehearsal and post-run audit

The operator's October 12 rehearsal task adds infrastructure checks before each
weekday, weekend or US holiday crypto run. Execution spec v4, research bindings,
model prompts for actual decisions and the acquisition grants remain unchanged.

`python -m scripts.trading_lab.trader_agent.rehearsal` uses the trader's deployed
configuration and private grants. Its 14 independent checks cover grant/spec
admission; Yahoo daily closes for stock, ETF and benchmark paths; all amended
Coinbase products; the GDELT endpoint and every configured query; receipt-cutoff
radar/veille construction; configured read-only archives; both analysts and the
reviewer; Alpaca quotes/trades; the real executor with GET-only paper clients;
asset flags and the executor's order planner; scorer/label imports; nine cockpit
API views and the actual sanitized home/analysis snapshot exporter.

Three Yahoo requests preserve the existing 80-request daily acquisition budget
for the full universe and labels. Coinbase probes cover the configured crypto
universe. The data probes use the existing shared dispatch bank, without grants
or caps being enlarged. Fresh daily price evidence is reused on retry only when
its UTC day and latest calendar endpoint still match. GDELT reuses the radar's
causal HTTP receipt and successful parse health when the receipt is under two
hours old. A recent failure is reported without another request. The radar's
durable provider cooldown is respected even when its receipt is older. A live
fallback uses the existing source budget only when neither receipt nor cooldown
applies; HTTP 429 persists a rehearsal cooldown across restarts, honoring
Retry-After (seconds or HTTP date) and X-RateLimit-Reset with an exponential
one-hour floor. A six-hour GDELT seen-time freshness check on that fallback does
not attest publisher time.
Radar freshness uses all causal input receipts from the last two hours, since
ranked events can legitimately have older receipts. Alpaca quote freshness
matches the production 60-second rule during the regular session and for crypto;
before the equity open it uses the last completed session close. IEX can leave
an unusable book update just after the close. Before the next regular DAY market
entry (and on days without an equity session), those quotes are indicative:
their non-executable count is reported and a positive trade from the completed
session's closing window is required. Negative prices still fail. In-session,
crypto and after-close quote checks remain strict. Production after-hours limit
orders retain their fresh, positive, uncrossed quote and spread guards. Last
trades must be positive and not from the future; their ages are reported
separately. Data failures identify each asset, field and code without raw prices.

GDELT availability failures are DEGRADED / PARTIAL_CONTEXT, matching the real
run. They remain visible in reports and require a delivered Telegram alert.
Readiness is GREEN when every mandatory step passes, even with this explicitly
degraded context. Query mappings, authorization, budget, order and label failures
still BLOCK; a missing degradation alert also blocks. Optional context loss
alone does not reserve a retry or repeat model probes. The pm GREEN notification
states the context grade explicitly.
The native GDELT empty object is reported as `NEWS_EMPTY_RESPONSE`, with its
digest and zero articles. It remains an unavailable input in the real run and
grades as DEGRADED / PARTIAL_CONTEXT in rehearsal. Other malformed shapes,
including null article lists and nonempty error objects, still block.

Inference prompts contain only synthetic empty input. Claude/Opus and
Claude/Sonnet use the actual restricted runner and local schemas. GPT reuses the
day's successful web/schema preflight; a failed preflight gets one retry at the
second rehearsal. `gpt_preflight`, `rehearsal_analyst_claude` and
`rehearsal_reviewer` each have a durable cap of two calls per UTC day, separate
from the six decision calls and the news arm. Calls are reserved before dispatch
and failures never refund them. Health inference has a 120-second timeout and
64,000-character output bound. Successful Claude probes are cached by day and
command binding. Grant expiry and pause controls still apply.

The broker boundary rejects every method except GET, before dispatch, and
refuses the Claude book account. Research stores are read-only. The paper
journal's production projection, binding checks and risk logic are used with
append operations confined to memory, including alerts and halts. The existing
writer lock is held only while taking the snapshot, never during broker HTTP;
an audit defers if execution changed that snapshot. No decision, execution
intent, order or research label is recorded by a rehearsal or audit. Source
reservations, model transcripts, monitoring state and probe exports remain
private. Reports contain only identities, suffixes, codes, counts and digests.

Paper writers and rehearsal snapshots wait up to 120 seconds for the ledger owner lock before reporting
PAPER_OWNER_BUSY. Waiting reserves no intent, sends no order and emits no alert;
expiry fails closed through the existing service failure reporting.
Post-audits retain an immediate quiet deferral when ownership is busy. Snapshot
readers pin the private DB/WAL copy under the shared lock, then validate its
chain and bindings after releasing ownership. Hash validation and HTTP never
hold the execution lock. Audits report snapshot lock duration and runtime.
Radar readers also wait up to 120 seconds for the shared owner lock before
taking the receipt-cutoff SQLite transaction required by the news spec. The
collector keeps exclusive ownership; waiting never changes the store or cutoff.
`paper-execute`, `paper-exit` and `paper-audit --dry-run` use the same GET-only
snapshot boundary, so the nightly compatibility checks also release the owner
before HTTP and keep risk/order observations in memory. Paper plans use the
current broker and real prediction journal; `--synthetic-time` applies to the
synthetic research run, not future broker fills or simulated order acceptance.

Broker calendar GETs must match the pinned NYSE session. Asset GETs validate
class, active/tradable status, equity shortability and account short permission.
The production planner must produce equity DAY markets and crypto GTC markets;
the extended-hours branch must produce DAY limits. These GETs and planner checks
cannot establish broker acceptance or future fills. The report says so.

## Scheduling and failure reports

`rehearsal_schedule.render` produces only trader user units:

- 07:35 UTC every day: an early rehearsal with an independent `early` reservation,
  five minutes after the hourly crypto exit check.
- 10:45 UTC every day: the primary rehearsal and final gate.
- 11:15 UTC every day: a second rehearsal only after primary failure/interruption.
- Every ten minutes at minute 01/11/21/31/41/51, second 30: read-only checkpoints and overdue manual retry dispatch.
- `OnFailure`: interrupted/timed-out rehearsals produce an alert and failure report.

Generated model units pin the installed Claude/Codex/Node directories in
`Environment=PATH` and exclude temporary Codex session wrappers. Model launches
do not require a login shell. Verify the installed CLI paths when staging units.

Timers use `Persistent=false` and no random delay. Attempt reservations survive
restart and prevent more than early plus primary plus retry. `rehearsal-<UTC-date>.md` keeps
all results. Early failure alerts without consuming the primary or reserving a retry;
successful same-day model probes remain reusable within the existing budgets.
A manual primary after 11:15 gets its second attempt 30 minutes
later through the audit poller. Independent failures alert the operator through
the private Telegram environment. Tokens, credential URLs, HTTP exception text
and response bodies never enter reports or stdout. Notification delivery is
reserved before dispatch to avoid duplicate messages after uncertain responses;
a failed/unknown delivery remains explicitly recorded rather than called sent.

## Post-run checkpoints

The audit schedule includes the session entry plus ten minutes, each durable
intent plus ten minutes, each planned horizon exit plus ten minutes from its
execution slot, and the completed label job. Equity entry checkpoints follow
09:30:30 New York; equity exits follow close minus five minutes, including half
days and DST. Crypto horizon exits retain their existing fixed anchor. A missing
run is checked at the session entry checkpoint; a missing/failed daily label job
is checked at 21:40 UTC. Actual intent checkpoints cover recovery entries/exits
and delayed decisions. Polling bounds normal dispatch delay to ten minutes.
Audits defer quietly from two minutes before until three minutes after scheduled
entries/exits (including hourly crypto checks, DST and half days) and recorded
prediction anchors. This covers the audit service's two-minute timeout and the
writers' retry minutes. Busy paper ownership or a changed snapshot defers without
completing checkpoints or alerting; the next safe poll retries them.

Every due intent must be filled, internally settled at zero net quantity, or
explicitly canceled/not submitted with a recorded rule. Expired, rejected and
unresolved orders fail. Broker positions must reconcile to every projected lot;
overdue residual lots fail. The real selector/planner also catches eligible
entries never reserved. At label checkpoints, every matured real prediction
must have its immutable label. Checks never acquire label prices or touch
holdouts. Each checkpoint independently audits the whole ledger. Entry
reconstruction runs the complete planner once per distinct dispatch time,
rather than once per candidate at the same time. Failed orders, lots
and labels are notified once per immutable issue identity, so a historical
expired order does not alert again at every new checkpoint. A new order with the
same failure code still alerts. Newly missing labels are sent in one bounded
batch rather than one message per prediction.

The label writer dispatches each distinct source input once per job, including
failed or invalid-price responses. Reusing a typed failure leaves all affected
predictions pending; it never supplies a replacement price or weakens admission.
The next job retries under the unchanged UTC/slot budgets. A failed equity bar
shared by many analyst views therefore cannot consume the entire source quota
through duplicate fetches.

An operator can record a historical disposition with `post_audit
--acknowledge-issue CODE:SHA256 --acknowledge-reason REASON_CODE`, adding
`--unlabelable` for a missing exact label anchor. Ordinary acknowledgements accept
only exact `LABELS_MISSING` and `PAPER_ORDER_EXPIRED` identities. Reasons and timestamps
persist in monitoring state without changing research labels, score denominators,
paper events or orders. Audits keep these observations and reasons visible in
`acknowledged_issues`; only active issues block and alert. New identities still
block. Open lots, reconciliation failures and other safety checks cannot be muted.

For an operator-disposed historical missing PM run, use its exact `MISSING_PM_RUN`
identity with `--not-activated --decision-sha256 SHA256` and a reason code. This
records `NOT_ACTIVATED` once, retaining the original timestamp, reason and canonical
operator decision hash on repeated invocations. The decision hash is stored in
monitoring metadata and exposed in `acknowledged_issues`; it does not modify any
spec, grant, budget, horizon, research record or paper event. Only that day's
identity is disposed; a missing PM run on another day still blocks and alerts.

Install the rehearsal units only after the owned branch's exact SHA has green
CI and the full required local suites pass. Preserve existing trader units and
hooks, add the audit hook before the existing label report hook, and use the
operator's guarded deployment: outside 11:30–13:50, 15:00–16:30 and 19:45–20:10 UTC, all trader services idle,
owned timers stopped, saved units and consistent database backups, then a final
idle check and timer restoration. Rollback restores code/unit bindings and
holds mutation timers if needed; never restore an old backup over an appended
production journal. Neither installer starts a trader decision or order.
