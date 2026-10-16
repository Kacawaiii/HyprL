# Paper execution incident recovery

The October 16 open retry read a one-share cumulative fill, then a three-share
broker position for the same short intent. Local ownership prevents concurrent
writers; it cannot make separate broker GET responses atomic.

The lock-aware second open pass now retries order/position reconciliation up to
three times. A remaining delta wholly within known outstanding signed quantities
or the progress of a fill observed in that pass
returns `PAPER_RECONCILIATION_PENDING`, blocks dispatch and leaves the account
eligible for a later reconciliation. Unknown orders, unexplained positions and
invalid or regressing fills retain their existing refusal and halt behavior.
The two New York timer calendars and the bounded paper ownership wait remain.

`paper-resume` is restricted to the operator decision with canonical digest
`435841de1bf86af67b80a4ad908002959b0b839714f3f390d0e4f63436462753`
and the October 16 reconciliation halt on `ia_actions`. It checks the complete
broker order history against durable intents, validates cumulative fills, and
reconciles current positions using GET only. Truncated history, any unknown
order, missing history, binding failure or risk breach refuses the resume.
Success appends a resume referencing the preserved halt and reconciliation;
it neither places orders nor changes any lot's quantity or exit deadline.
An overdue equity lot remains subject to the next regular-session opening sweep.
Later kill-switch handling uses the active halts since that resume, preserving
the historical incident without obstructing a new safety liquidation.

Label realization validates SPY before consuming each dependent symbol's quota.
Invalid native prices retain strict rejection and produce exact
`RETRYABLE_INVALID_PRICE` outcomes and a `skipped_invalid_price` count in the
atomic job summary. Independent outcomes continue; genuine quota or expiration
failures remain visible as BLOCKED. Retryable skips create neither labels nor
execution observations and remain eligible for a later corrected native price.
Existing explicit UNLABELABLE dispositions, issued denominators and historical
labels remain intact. No invalid price is imputed or admitted.

Regression coverage uses synthetic broker and source responses at the public
executor, resume, realization and live-configuration boundaries. The deployment
also retains the earlier audit, label quota, terminal disposition and nightly
drill fixes, and admits GPT retirement registration after populating the locked
read-only rehearsal snapshot.
