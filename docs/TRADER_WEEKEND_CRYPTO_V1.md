# Weekend and NYSE holiday crypto population

The October 10 operator decision authorizes `weekend_crypto_v1` on UTC dates
without a regular session in the existing pinned XNYS calendar. The daily timer
runs at 12:00Z on every date. Session dates retain the original weekday behavior,
revision 5 preregistration and execution spec v2.

Revision 6, `trader_agent_preregistration_v3.json`, applies only to this new
population and binds `trader_weekend_crypto_spec_v1.json`. Both canonical hashes
are pinned in `scripts/trading_lab/trader_agent/weekend.py`. Original artifacts,
grants, analyst/reviewer skills, schemas and paper ledger bindings stay frozen.

The private operator decision is loaded beside the base grant and checked against
its pinned canonical digest. An absent decision preserves the holiday skip; a
changed decision fails closed. The amended universe must be SOL, AVAX, LINK,
DOGE and LTC in USD. BTC and ETH remain excluded and protected. Existing grant
windows apply; this change does not renew the November 5 authorization expiry.

Context acquisition includes only Coinbase completed daily candles, crypto news,
macro news and the causal read-only FOMC archive. It does not request Yahoo,
equity/ETF prices or news, benchmarks or EDGAR. Both independent analysts and the
reviewer run serially under the same model settings, validation and retries.
All runtime paths share the existing durable UTC-day run, role, aggregate and
source budgets. There is no weekend allowance or refund.

Decisions and recording must finish before 13:30Z. Research labels use the exact
Coinbase one-minute candle open at 13:30Z on the decision date and the same UTC
anchor 24 or 120 hours later. A target is available only after its minute completes.
Missing prices or exhausted request budgets leave labels pending. The daily
21:30Z label timer includes weekends, sharing its anchor cache across predictions.
All new analyst, reviewer, consensus, baseline and portfolio scores and causal
climatology are separate from weekday scores and the confirmatory sample. Only
the unhedged crypto cohort is reported; no equity benchmark is introduced.

Paper execution uses the unchanged crypto consensus rule on `ia_crypto`: long or
flat, market GTC entry immediately after the successful decision, then exits at
the fixed label anchors through the existing hourly :30Z exit timer. Actual fills
and their outcomes differ from fixed research anchor returns. Caps, kill switch,
order reservations, idempotent IDs, reconciliation and exit-expiry checks remain
in force. Closed-day execute/exit calls cannot open `ia_actions`; the label report
hook skips account reporting on closed dates.

Before first execution, the operator decision explicitly registers the new
population in the paper journal with `paper-register-weekend --operator-decision`.
This offline operation appends one idempotent `weekend_registration` event binding
the new spec/preregistration, existing execution spec and grants. It performs no
broker request and changes no previous binding or open lot. Missing or changed
registration rejects weekend execution. Prior execution-spec stores retain their
existing rejection and explicit rebinding rules.

The recovery timer now polls every date in its existing New York time window.
Closed-day crypto recovery deliberately issues the durable transition alert
`RECOVERY_WEEKEND_CRYPTO_SKIPPED`; it makes no call or catch-up reservation.
Repeated polls are quiet. Weekend health requires a completed/degraded run by
13:30Z and a label-job record by 22:00Z. Failed runs remain visible and alerted.

`tests/trader_agent/test_weekend_crypto.py` verifies Saturday/Sunday/holiday
selection, context and API isolation, fixed labels and separate scores, shared
budgets, retries, timing/expiry, explicit registration, paper execution/exits,
kill/pause/reconciliation guards, recovery alerts and weekday preservation using
native-shape synthetic providers and an injected broker boundary.
