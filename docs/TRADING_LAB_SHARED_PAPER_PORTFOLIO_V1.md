# Shared paper portfolio V1

```
SHADOW MODE ONLY
NO REAL MONEY
NO BROKER
NO CONFIRMED EDGE
CURRENT EXPLORATORY PORTFOLIO IS NEGATIVE GROSS AND NET
```

That last line is the point of stating it. The Phase 6B backtest of exactly
this configuration returned −0.30 % gross and −8.08 % net over the recorded
out-of-sample window: the strategy paid 7 780 to lose 296. This runtime is
plumbing for a portfolio that does not currently make money. It exists to
prove the live path is correct, not to suggest it is profitable.

## What changed

Phase 5D ran two independent shadow accounts, each with its own 100 000.
Phase 6C runs one portfolio: one cash ledger, one equity figure, one exposure
budget, positions in both instruments at once.

## Arrival order is the whole design

Live candles do not arrive in a defined order. BTC may land before ETH, or
after, or a minute later. The obvious implementation rebalances whichever
arrived — and that moves cash and equity, so the second instrument is sized
from a different number. Phase 6B established that this is not a rounding
difference but a *different portfolio*.

So arrival never triggers execution:

```
candle arrives  ->  features  ->  frozen model  ->  signal  ->  target
                ->  PendingPortfolioBatch[decision timestamp]
                                     |
                     complete?  ---- no ----> WAITING_FOR_PORTFOLIO_BATCH
                                     |
                                    yes
                                     v
                    ONE PortfolioTargetSet -> Portfolio Engine V1
                    (one shared pre-trade equity, atomic)
```

A batch executes when every session instrument has a target for that decision
timestamp **and** every price needed — to execute, and to mark every open
position — is observable. Arrival order affects nothing but the wall-clock
stamps on monitoring events. Tests permute arrival and compare fill hashes and
state.

**FLAT is a target, not a missing one.** An instrument whose signal is FLAT
produces an exposure of zero, which may well close an open position. Treating
it as "not ready" would stall every batch on a market that is mostly flat —
which this one is: in the synthetic sessions the model is under threshold
almost always.

## Batch identity

Derived, never generated:

```
batch_id = sha256(session_spec_hash, decision_at,
                  canonical instrument ids, position target hashes,
                  market frame hash)
```

A UUID would mean a restart could not recognise its own batch, and the
exactly-once constraint would have nothing to match on.

## Exactly-once

Enforced by the store, not by a code path someone must remember:

```
UNIQUE (session_id, event_type, instrument_id, natural_key)
```

A restart that re-derives a committed batch is refused by the database. The
in-memory check in front of it is an early-out that avoids redoing the work;
it is not the guarantee. Both together survive a mutation of either.

**A batch is finished when its snapshot lands, not when its target set does.**
An interrupted rebalance — target set written, snapshot not — is *resumed* on
restart rather than marked complete. Marking it done at the target set would
silently drop a trade the log says was decided. Re-appending is refused by the
database, so resuming cannot duplicate anything, and only fills this call
actually wrote are counted.

## Restart

Verify the chain, replay the tail, rebuild state and pending batches. A
half-collected batch survives: BTC ready, crash, restart, ETH completes it.

Snapshots are triggered by distance from the last snapshot, never by testing a
running total for divisibility. The Phase 5D trigger did the latter and could
not fire at all when the event stride was constant — 2 918 events, zero
snapshots, silently. A flat session is tested explicitly here for that reason.

## Holdout

The canonical guard from 6A-R. At 2026-09-01T00:00:00Z every instrument in
this session is protected, so the whole session goes `EMBARGOED`.

The subtle case is a batch decided on the last unprotected bar. Its execution
bar opens at 00:00 on 1 September — inside the window. That batch is expired
with reason `HOLDOUT_BOUNDARY`, decided **from the timestamp alone**: the
protected candle is never requested, and would be refused if it were. Without
this it would sit pending forever, waiting for a bar it must never see.

`PROTECTED_HOLDOUT_BOUNDARY_REACHED` carries no market data — asserted by a
test that plants a recognisable price and greps the event for it.

## The legacy store stays separate

`var/trading_lab/paper_v1.sqlite` is untouched, readable and exportable. The
shared portfolio writes `paper_portfolio_v1.sqlite`.

Migrating the old log in place would mean rewriting payloads whose hashes
exist precisely so they cannot be rewritten, and the result would be a chain
that verifies while describing a history that never happened: BTC and ETH
never shared a dollar in those sessions. **Their equity curves are not the
history of this portfolio and are never summed with it.**

## No financial arithmetic here

Allocation, the gross cap, quantity deltas, fees, slippage and cash all come
from the Phase 6B engine. A structural test asserts this module contains no
`fee_rate *`, no `slippage_rate *`, no sizing arithmetic — a second
implementation would agree today and drift later.

Execution timing is unchanged from 5D: a target decided on bar T is priced at
the open of T + one interval, observable only once that bar closes, so the
fill is recorded one bar after the price it is filled at. That is stated in
`PaperExecutionSpec`, not hidden.

## Commands

```
./scripts/hyprl.sh paper start|stop|restart|status
```

`paper start` now starts the shared portfolio. The 5D per-product runtime is
no longer started from here; it remains inspectable and exportable.
