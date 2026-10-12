# Second daily paper slot

The October 12 operator decision authorizes `am` at 12:00 UTC and `pm` at
16:00 UTC. The first pm slot is registered for October 13. Existing am
identities, predictions, labels, prompts, source contracts, execution v4
bindings and paper grant remain valid. No historical journal is rewritten.

The new `trader_pm_slot_spec_v2.json` and preregistration v9 are additive.
The private pm amendment binds the original grant and operator decision.
An explicit `paper-register-pm` action appends the corresponding capability
registration to the existing v4 paper ledger. Pm execution refuses an absent
or mismatched registration; loading an old ledger never migrates it.

Each slot has its own durable primary/recovery and model reservations, with
six calls including retries per base arm and six per news arm, two per role.
Source limits are Yahoo 80, GDELT 80 and Coinbase 40 per slot. Existing health
probe allowances remain separate; successful daily probes are reused. GDELT
spacing and the owner lock are shared. Failures and process restarts never
refund a reservation. News cannot acquire market inputs or trade.

Pm freezes its publication, completed-price and archive cutoffs at 16:00.
Actual input receipt, final-plan and journal timestamps are retained. A
trading day uses the existing full universe; a weekend or US holiday uses
only the registered crypto universe. Pm identity appends `:pm`, with distinct
`:catchup` and `:news` suffixes.

Pm equities enter with whole-share DAY markets immediately after the base
plan is final, only while the regular session is open and strictly before
close minus fifteen minutes. Crypto retains market/gtc execution. Both slots
share the unchanged paper account limits and forty-order daily caps.
From pm activation, an open or pending entry lot blocks another lot for the
same asset/account in either direction. For newly admitted assets, 1d takes
precedence over 5d; existing lots are retained. Suppressed views remain scored.

Primary research anchors are prospective: 17:30 UTC, or for half-day equities
regular close minus thirty minutes if earlier. Equity 1d/5d labels exit at
the close of the first/fifth complete following session; paper exits use that
close minus five minutes, with the existing overdue open sweep. Yahoo native
5-minute candles supply the exact anchor open and final session candle close.
Crypto keeps the fixed-anchor convention, with exact minute prices and
24/120-hour endpoints. Actual fills and their anchor gaps remain separate.
No nearest-price substitution is permitted.

One same-day recovery is permitted for a failed, interrupted or missed pm
primary that has issued no prediction. It retains the original 16:00 cutoff
and budget. Its separate recovery population uses a future :30 anchor in the
following UTC hour, bounded by regular close minus fifteen minutes for
stocks and 23:30 UTC on closed days. A :45 equity recovery anchor rounds
crypto forward to the next :30, preserving its existing hourly exit timer.
Both arms retain the same anchor for each asset. Recovery never enters equities after
hours. Exhausted analysts remain missing; exhausted reviewer/aggregate/data
budgets leave the slot pending. Both base and news must finish before the
same registered anchor.

Pm timers add the 16:00 run, 15:15 full-chain GET-only pre-flight, 15:45
conditional retry, and same-day recovery polling. Existing shared exit timers
cover both slots. The audit poller checks actual intents and final plans ten
minutes later and detects a missing pm slot independently of a successful am
run. Pm pre-flight sends a bounded Telegram result; failed delivery also
reports BLOCKED. Original am monitoring remains in place.

Scorecards expose `by_slot.am`, `by_slot.pm` and descriptive `combined` totals
and metrics. Execution reports also separate slots. Legacy primary scores
and the original confirmatory sample remain separate from pm. Cockpit
exports include the slot on runs, labels, timelines and their latest run,
plus both slot scorecards and the combined total. None establishes an edge.

Deployment requires the exact release SHA's branch and integrated CI,
all local gates, idle trader services, consistent database/unit backups and
an allowed operator window. The new capability is registered explicitly;
old journal snapshots must never overwrite appended evidence during rollback.
A live pre-flight or rehearsal must be GREEN after deployment. Pm spec v1
and preregistration v8 remain frozen evidence; v2 refuses their registrations
rather than migrating them implicitly.
