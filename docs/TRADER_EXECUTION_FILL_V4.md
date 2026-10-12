# Paper execution timing revision 4

The paper account's first OPG entry expired with zero fills. Its order response
supplied no rejection reason. CLS fills have not been demonstrated. The operator
approved DAY market execution in task `trader-exec-fill-fix`.

The immutable registration is `trader_alpaca_execution_spec_v4.json`, bound by
`paper_spec.py`, with execution preregistration `trader_agent_preregistration_v7.json`.
The private task authorization must match the canonical operator decision hash.
The v3 journal is rejected until an explicit empty-ledger `paper-rebind` checks
both AI accounts with GET only. Historical bindings, grants, registrations and
observations remain intact. Rebinding refuses pending orders or live lots.

The daily decision's post-run hook still executes crypto immediately. Equity
entries wait for `hyprl-trader-paper-open.timer` at 09:30:30 America/New_York, with
an idempotent recovery invocation at 09:31:30. The open service first sweeps
past-horizon equity lots, then executes today's preopen predictions. The accepted
entry window ends at 09:35; missing the window never shifts the research horizon.
All regular equity orders are whole-share DAY markets. Extended-hours limits and
crypto GTC orders retain their rules. Market orders have no hard slippage cap;
existing price reserves and exposure caps remain enforced.

Equity exits run at 15:55 New York and 12:55 for half days. The pinned NYSE calendar
admits only close minus five minutes through the close, and the open sweep window.
The extra half-day invocation is harmless on regular days. Holidays do not
execute equities. New York's IANA time zone supplies the DST shift.

Cumulative observed fills alone project into lots. An expired entry with no fills
has no live quantity; partial fills keep their real quantity and price. Known
stale entries are canceled and reconciled before exits, and stale exits before
retrying. A cancel acknowledgment that remains nonterminal blocks replacement.
Exit retries bind all intersecting prior terminal exit IDs, so restarts cannot
submit a second copy. Daily order caps include retry intents. Position mismatches
retain the existing persistent halt with no repair orders.

`paper-audit` uses GET only on the two AI accounts. The hourly health hook and
daily audit timer reconcile and classify every intent as filled, explicitly
canceled by rule, internally settled, or alerted. Expiry/rejection alerts are
idempotent per client ID. Unresolved orders, missed eligible entries and overdue
lots alert. A tier safety stop never cancels orders from this read-only audit.

Research labels remain open-to-close. Reports include actual fill prices and
`execution_gap_scores` separately from the research scores: signed adverse entry
gap versus the label open, exit gap versus its close, and actual-minus-label
signed return. Gaps stay pending until an available, recorded research label
contains the corresponding prices. No extra price requests are introduced.

Deployment requires local checks and the exact integrated SHA's successful CI,
an allowed time outside 11:30–12:30 UTC, inactive trader services, paused owned
timers, consistent SQLite backups and saved user units. Register under the owner
lock, verify units and synthetic readiness, and restart timers. On failure restore
units, never replace an appended journal with its backup. If the v4 binding was
appended, old code must stay unable to trade; hold mutation timers pending a
forward fix. Real DAY fills remain unproven until a scheduled eligible signal is
executed and reconciled.
