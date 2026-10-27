# Forward day exclusion repair (execution revision 20)

On October 27 the 08:45 New York prebuild reported READY, but both 09:05
dispatches skipped inference with `DATA_QUALITY_EXCLUDED`. The prebuild did not
inspect the input manifest or durable model gates, and dispatch treated a shared
day exclusion as successful completion without an alert.

The exact day-level reason was `NEWS_ARCHIVE_COVERAGE_MISSING`. The collector's
72-hour query started October 24 at 13:05 UTC; the earlier prebuild required
October 24 at 12:45:11 UTC. That older missing interval preceded the candidate
decision window, which started at the previous exchange close. Turning this
partial lookback into an empty universe discarded otherwise usable inputs.

There was a separate collection failure: each five-minute poll refreshed every
page of the same 72-hour news request. Repartitioning an expanding candidate list
also repeated completed daily-history requests. The input collector reached its
unchanged 2,000-request allocation at 10:15 UTC, retaining the original 1,000
requests for outcomes within the 3,000-request daily ceiling. Collector budget
alerts were delivered. The input cache's requested future end was not proof that
later news had been received; no counterfactual can establish those missing
headlines, provider availability, model decisions or outcomes.

Revision 20 / artifact v16 changes forward inputs only. Coverage is the union of
complete, all-symbol pagination chains; missing pages and loops establish no
coverage. Gaps outside the previous-close-to-input-cutoff decision window are
recorded as partial coverage and affect news items only. Within that window a
gap of at most 1,800 seconds is also local unless it covers the entire window.
A missing entire window or a longer recent outage excludes the day. Known news
with creation or update clocks in a gap is excluded before ranking, identically
for both families. Both contexts receive the explicit missing intervals. Missing
assets, benchmark failures, systemic pipeline defects, integrity errors and all
existing eligibility rules retain their strict behavior. Historical replay
contexts and their exclusion policy are preserved by an explicit input-only
reseal into a new state directory; no prior store is implicitly admitted.
The 28 received historical envelopes from the predecessor execution are admitted
by explicit digest references, preserving their original protocol and validation
checks. Existing responses and failures cannot cause another model call merely
because the execution revision changed.

The collector now bootstraps far enough back for the earliest permitted prebuild.
After complete pagination is durably indexed, each poll resumes from the verified
contiguous coverage end with a five-minute overlap and ends at the actual poll
start. It never requests a future coverage end. Interrupted pagination cannot
advance the watermark. Existing completed daily histories are reused; only new
symbols require new history requests. New polls retain distinct response objects
and receipt times. Request limits, rate spacing, backoff, outcome reservation,
protected assets and disabled SEC collection remain unchanged.
The overlap captures recent stories; it does not establish that later edits to
older stories were observed. Contexts use the latest version actually captured.

Before READY, prebuild checks both manifests and the scheduled session's grant,
model arguments, budgets, usage/credit stops, API billing when enabled, and saved
decision/failure gates. It repeats durable gates after construction and creates
no ready pointer for a blocked result. The CLI also checks collection grants,
model executables and local publication configuration. It sends an `alertes`
notification for any blocked result. The 22:45 UTC drill checks tomorrow's gates
and applies input exclusion checks to the rebuilt archived source snapshot.
Dispatch reports BLOCKED and alerts for every skipped model, while retaining
immutable failures and the existing hash-publication protocol.

Readiness for tomorrow remains conditional on tomorrow's collection and actual
provider availability. An offline drill can demonstrate `excluded_day=false`
on its identified source session; it cannot certify future news receipts.
The lost October 27 session is never dispatched or published again.

Deployment follows the pinned research procedure in
`REPLAY_FORWARD_WINDOW_REPAIR.md`: back up research services/timers, pause only
formerly active research timers, drain research services without killing calls,
fast-forward the deployment worktree, verify staged service definitions, replace
only their expected-commit bindings atomically, reload user units and verify all
nine bindings. Keep every existing timer calendar and resource bound unchanged.
Run the Wednesday dry run and real offline drill, then restore the saved active
timers. The independently deployed reveal service is outside this change. Do not
use installer handling for unrelated login-refresh or trader units.
