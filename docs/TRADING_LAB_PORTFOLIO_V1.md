# Portfolio engine V1

Two backtests printed side by side are not a portfolio. Until Phase 6B, BTC
and ETH each ran on their own synthetic 100 000 and never competed for a
dollar, which is why adding their percentages meant nothing. This engine gives
them one cash ledger, one equity figure and one exposure budget.

It predicts nothing. It consumes the `PositionTarget` stream that Signal V1
and Risk V1 already produce and decides how several of them coexist under
shared capital. No threshold, cap or cost here was derived from any observed
P&L.

## The specification

| Field | Value |
|---|---|
| `protocol_version` | `trading-lab.portfolio.v1` |
| `base_currency` | USD |
| `initial_equity` | 100 000 |
| `max_instrument_abs_exposure` | 0.25 |
| `max_gross_exposure` | 0.50 |
| `max_net_abs_exposure` | 0.50 |
| `allocation_rule` | `proportional-gross-cap-v1` |
| `simultaneous_rebalance_rule` | `single-pretrade-equity-batch-v1` |
| `cash_model` | `shared-cash-v1` |
| `short_model` | `synthetic-linear-short-v1` |
| `optimized` | **false** |

`portfolio_spec_hash = 32ec6c9f5f62c24bd18077dda79eefb30a334edcf26b811175f9b93584cdcebf`

### Why 50 % gross

RiskSpec V1 already caps one instrument at 25 % of NAV. There are two
instruments. 50 % is therefore the gross exposure the existing rules permit —
the cap **writes down the structure that was already there**. It is not an
optimum, it was not fitted, and `PORTFOLIO_LIMIT_V1_IS_NOT_OPTIMIZED` says so
in the source.

## The batch, and why the order matters

The failure this design exists to prevent:

1. fill BTC → cash and equity move
2. size ETH from the *new* equity

That makes the portfolio depend on the order two instruments happen to occupy
in a list. It is not a rounding difference — it is a different portfolio, and
a single run would never reveal it.

So one **pre-trade equity** is computed once per timestamp, and every target in
the batch is sized from that same number:

```
A. mark every open position at this timestamp
B. pre_trade_equity = cash + Σ(quantity_i × price_i)      ← computed ONCE
C. validate each |requested_i| ≤ max_instrument_abs_exposure
D. requested_gross = Σ|requested_i|
   scale = 1 if requested_gross ≤ cap else cap / requested_gross
E. target_quantity_i = (requested_i × scale × pre_trade_equity) / price_i
F. delta_i = target_quantity_i − held_i
G. price every delta through the shared execution primitive
H. apply the aggregate cash impact
I. build the resulting positions
J. check the limits
```

No economically observable state exists between one instrument and the next.
Instrument order affects serialization only, which is asserted by permuting
the inputs and comparing the **result hash**.

### Proportional, never prioritised

When the batch exceeds the gross cap, every request is scaled by the same
factor. Giving BTC its full request and cutting ETH — or cutting whichever
arrived second — would make allocation depend on registry order or list order.
Ratios between unequal requests survive scaling.

### The net cap does not improvise

Under V1, `|net| ≤ gross ≤ 0.50`, so the net cap never binds. If a future spec
made it bind, the engine **fails closed** rather than applying a second,
unspecified scaling. A rule set that cannot satisfy its own caps is a rule set
to fix, not to paper over at runtime.

## What the caps are measured against

Targets are sized from **pre-trade** equity, so the cap is checked against
pre-trade equity too. Those differ by exactly the execution cost: checking a
25 % target against post-trade equity would report a breach on every single
trade, because the fee and slippage came out of the denominator.

Realised exposure also drifts afterwards as prices move. That is not a breach
either — no rebalance happened, and enforcing a realised band continuously
would be a different strategy that trades on price moves alone.

## Gaps: nothing is forward-filled

If an open position cannot be priced at a timestamp, the portfolio has no
honest equity, so **no snapshot is produced** and the timestamp is reported in
`unavailable_valuations`. Marking yesterday's price as today's would smooth
exactly the drawdowns a reader is looking for.

A flat position needs no price: nothing held is worth nothing at any price.

## Execution costs

Reused, not reimplemented. `apply_quantity_delta` was extracted from the
Phase 5C engine so the single-product backtest and the portfolio share one
implementation of fee, slippage and cash. All 1 636 committed fills across
BTC and ETH re-derive from it byte for byte, and the recorded result hashes
are unchanged.

The single-product primitive computes its own pre-trade equity, which is
exactly what a portfolio must not do — hence the lower-level primitive that
takes a quantity delta and knows nothing about equity.

## Attribution

* **gross P&L** — quantity held × price move between marks. An instrument's
  own arithmetic, never a share of the shared cash.
* **execution costs** — attached to that instrument's own fills.
* **net contribution** — gross − costs.

The net contributions add up to the change in portfolio equity. This is
asserted, not assumed:

```
Σ net_i  ==  final_equity − initial_equity   (to ATTRIBUTION_RELATIVE_TOLERANCE)
```

The identity is exact in arithmetic; each accumulation rounds at 34
significant digits, so the residual lands around 1E-34 relative. The tolerance
is 1E-30 — four orders looser than observed, and twenty-five orders tighter
than a real error, which would be of order cents.

## Comparing against the single-product runs

Do not add the percentages. The 5C runs used 100 000 **each** (200 000 of
synthetic capital in total, never shared); this uses 100 000 **in total**, and
the gross cap means at most half of it is ever at risk. Compare relative
return per unit of exposure, and say which capital base each figure refers to.

## Identity

Every identity entering this module is resolved through the registry before
any arithmetic. Phase 6A-R found two boundaries where a raw string comparison
let one instrument's data be used as another's, and both produced plausible,
wrong numbers. Nothing here compares a product spelling directly:
`PortfolioIdentityMismatch` is raised before any quantity, cash or fee is
computed.
