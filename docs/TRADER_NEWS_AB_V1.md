# Prospective news A/B and downgrade safety stop

Preregistration v5 (revision 8) binds the new `trader_news_ab_spec_v1.json` to
the operator decision digest. This is an additive contract: research v2,
weekend research v3 and execution v3 keep their frozen identities. Existing
paper bindings are not migrated. `paper-register-news-guard` explicitly journals
the extra safety contract under the paper owner lock; a mismatched registration
refuses new execution.

The base run freezes the optional news input before inference. Radar is opened
with SQLite `mode=ro`, a read transaction and a nonblocking shared lock on the
existing radar owner file. Evidence append times, receipt times and known
publication times must strictly precede the base context's frozen decision time.
The default receipt lookback is 48 hours. Events, source verification levels,
importance/evidence scores, measured priced-in moves, regime and anomalies are
recomputed from those receipts, never taken from a later digest. Collector-marked
primary sources have level OFFICIEL; other headlines have level FIL. These levels
identify provenance, not independent verification of an issuer's claims.

Veille entries require a timestamp or a dated section heading. Date-only entries
become usable after that UTC day ends. Undated entries are omitted. A file modified
at or after the cutoff is excluded in full because its earlier contents cannot be
proved. The bounded snapshot is hashed and stored privately in the run evidence;
raw article bodies and private paths are not part of the context. Prompts place
news and veille in the JSON data block, after the fixed instructions.

The scheduled service runs base inference, then base paper execution, then the
separate `news` job. News has the same analysts, reviewer, universe, input decision
time and label windows. Its recording timestamp reflects its later completion.
Its durable `news_` budget reservations allow one initial call plus one retry for
each role, at most six attempts per actual UTC day, and one news run. Base source,
model and execution budgets remain separate. News failures leave base decisions
and predictions intact. News is shadow only; the executor accepts only registered
base run summaries and base prediction variants.

The scorecard and cockpit export include `news_ab`: n, hit rate, Brier and
abstention for each source, population and horizon. Comparisons use only completed
prospective paired runs, excluding historical base sessions. Both arms need at
least 30 realized directional views. News must have strictly better SPY-relative
hit rate and strictly lower Brier. Eligibility still requires an operator decision
and a new execution registration. Crypto uses raw direction and remains
observational: there is no SPY-relative promotion claim. Before the minimum the
state is `TOO_EARLY`.

The `half_size_downgrade_v1` safety rule counts completely closed, filled entry
lots across its horizons/populations. It deduplicates cumulative exit observations,
excludes pending/unfilled/partial lots, uses actual entry/exit prices and subtracts
an additional conservative modeled roundtrip cost of 15bp of entry notional.
Separate broker fees are unavailable; this limitation is explicit in the contract
and report. At 20 completed lots, or any later cumulative prefix, a positive share
below 45% permanently latches this tier off. Exactly 45% does not trip it. The
journal records the stop and alert; restarts and improved later performance cannot
resume it. New entries and unsubmitted intents in that tier are refused; known
open entry orders containing it are canceled (a mixed immutable order is canceled
as a whole). Full KEEP entries, scheduled exits and prospective research scoring
continue. Resuming requires a new operator decision.

Validation uses native synthetic radar receipts, mocked model inference and paper
broker transports. Deployment requires the full repository gate, exact-SHA green
branch and integrated CI, an idle trader, the allowed time window, backups,
explicit guard registration, unit validation and green readiness. Rollback never
restores over appended journals; if the new safety contract was registered,
rollback holds entry/recovery timers while preserving exits and scoring.
