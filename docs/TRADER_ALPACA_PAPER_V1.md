# Alpaca paper execution v1

This exploratory executor uses the separate `trader-alpaca-paper-v1` grant. It
executes reviewed consensus stock and sector ETF views on `ia_actions`, and
long-or-flat crypto views on `ia_crypto`. `complet` sums their equity and P&L
against a $200,000 base. `momentum` is a GET-only benchmark; every mutation on
that account is refused before transport, including cancellation and replacement.
All order mutations are blocked before **2026-10-08 12:00Z**.

The original shadow experiment retains its primary hypothesis and labels.
Preregistration revision 4 registers `alpaca_open_entry_v1` and
`alpaca_after_hours_v1`, bound to the canonical execution specification in
`docs/artifacts/trader_alpaca_execution_spec_v1.json` and `paper_spec.py`.
Changing the specification or grant never silently rebinds an existing paper
journal. The primary ResearchStore remains immutable.

## Sizing and lots

Require directional probability at least 0.55. UP uses `p_outperform`; DOWN
uses `1-p_outperform`. For each name and horizon, signed weight equals
`name_cap * horizon_fraction * abs(p_outperform - 0.50) / 0.30`, with horizon
fractions 0.5 for 1d and 0.1 for each 5d cohort. Scale new allocations
proportionally to remaining gross and short capacity, cap the total absolute
lots per name, reserve 10% for price movement, and floor quantities to whole
shares or eight crypto decimals. Pending entry intents consume capacity.
New lots whose label exit is outside either grant's validity are excluded.

There is one lot per `(run_id, asset, horizon)`. Broker positions equal the sum
of live signed lots. Entry and exit orders submit only the resulting symbol
delta. Cumulative partial fills use largest-remainder allocation in broker
quantity units, so a partial whole-share auction cannot create fractional CLS
orders. A zero entry delta is NOT_TRADED. A zero exit delta is an internal
settlement, identified separately from broker fills.

Before admitting an entry, reserve today's exit order capacity, including new
1d lots. Select symbols in deterministic order within the 40-order allowance.
The private account's limits remain 8% per equity name, 80% gross, 30% short
gross, or 25% per crypto asset, 50% gross and no shorts. Opening gaps and marked
price movements can exceed intended weights; refuse new entries while marked
caps are breached and retain scheduled exits.

## Timing and after-hours

Daily equities enter with whole-share OPG market orders before open minus two
minutes. Exits use CLS during close minus 20 through close minus 10 minutes.
The pinned exchange calendar supplies holidays, early closes and DST.
Crypto enters at market immediately after a successful decision (maximum
five-minute decision age) and exits at the fixed research label timestamp.
Its actual entry differs from the research label's 13:30Z anchor, and the
execution observations retain that distinction.

`catchup` close-entry runs are score-only. `catchup-after-hours` requires a
recorded FAILED primary run and remaining model/reviewer budgets, starts after
the regular close with at least 80 minutes remaining, and completes before
19:30 America/New_York. Its equities have distinct predictions and retain the
catch-up exit session: next session for 1d, fifth following session for 5d.
Budgets remain pinned to the daily session across UTC midnight. The shadow
label job never imputes an open price for these predictions; actual paper
entry/exit outcomes are recorded separately.

After-hours orders require a quote no older than 60 seconds, spread at most
30bp, and a day limit with `extended_hours=true`. Limit prices remain within
10bp of the midpoint, rounded inward. Unfilled entries are NOT_TRADED and
retain the decision reference. **Paper fills in thin after-hours liquidity are
optimistic.**

The current grant allows only the paper trading API origin. Fresh bid/ask data
normally use a separate market-data origin, which is not granted. Until an
operator grants that scope, `--paper-quotes` accepts a private operator feed:
`{"quotes":{"AAPL":{"bid":100,"ask":100.1,"at":"2026-10-08T20:10:00Z"}}}`.
Missing, stale or future quotes refuse execution; no ungranted HTTP endpoint
is contacted. Automatic after-hours quote acquisition is therefore blocked.

## Commands and scheduling

Supply the parent grant, paper grant and existing trader runtime:

```sh
python -m scripts.trading_lab.trader_agent.cli paper-status \
  --authorization "$TRADER_GRANT" --paper-authorization "$PAPER_GRANT" --runtime "$TRADER_RUNTIME"
python -m scripts.trading_lab.trader_agent.cli paper-execute --dry-run \
  --authorization "$TRADER_GRANT" --paper-authorization "$PAPER_GRANT" --runtime "$TRADER_RUNTIME"
```

Other actions are `paper-exit` and `paper-report`; `--paper-account` restricts
execution to one of the two trader accounts. Paper `--dry-run` uses real GETs,
prints proposed orders without client IDs, and reserves or submits no orders.
It differs from the original trader's synthetic `run --dry-run`.

Install units with the schedule module's `--paper-authorization` option and
the usual parent/runtime/archive arguments. Installation starts timers, never
a run. `paper-execute` follows a successful daily service invocation;
`paper-report` follows label service completion, including a failed label job.
Equity exit timers fire at 15:40 and 12:40 America/New_York; the latter handles
half-days and otherwise has no due CLS exits. Crypto checks run hourly at
`:30 UTC`, matching label anchors. An additional 19:30 America/New_York check
cancels remaining after-hours entries at their cutoff. All timers are non-persistent.

The execution journal, order budget, peaks and halt state share one private
bank across runtime paths. `PAPER_PAUSED` in that bank or the trader's runtime
or shared budget `PAUSED` markers prevent mutations. A drawdown below 90% of
recorded peak persistently halts the account, cancels known pending orders,
reconciles final fills, and flattens its lots within the remaining allowance.
Unconfirmed cancellations or an exhausted budget leave an alert and require
operator attention. There is no automatic kill-switch resume.

Every start verifies the grant, suffix, evidence chains, known order identities
and broker positions. A discrepancy halts the account without repair orders.
Durable intent precedes dispatch; restart looks up its deterministic client
ID before submission. Missing orders past their auction deadline expire
locally rather than being rescheduled. Evidence contains account suffixes and
broker order digests, never credentials, full account numbers or raw bodies.

Reports contain account equity, daily and total P&L, open lots, `complet`,
fill outcomes and momentum equity/return. SPY and QQQ use the parent grant's
public OHLC source; BTC uses its public candle source only when unprotected.
Benchmarks with protected or unavailable prices remain explicitly PENDING.
SPY/QQQ comparisons begin at the first execution session's observed open;
BTC uses that day's fixed label anchor. Momentum's base is observed at the
first execution invocation. Every baseline time is reported. Unavailable
benchmark entry prices remain pending.
