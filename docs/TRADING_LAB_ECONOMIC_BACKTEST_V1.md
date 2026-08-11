# Trading Lab — economic backtest engine V1

**STATUS: ENGINE CONTRACT, FROZEN BEFORE THE FIRST SIMULATED P&L**

This layer turns frozen position targets into a simulated portfolio: fills,
cash, quantity, costs, equity. It is the first place in HyprL where numbers
look like money, so it is the first place that can flatter itself.

## The causal boundary

A prediction at bar T is computed from the *whole* of candle T, including its
close. So a target derived from candle T is not executable at `open[T]` — that
price is already history by the time the candle exists.

```
decision from candle T
  → decision_available_at = T + 1h
  → first admissible fill  = open of candle T+1h
```

`fill_policy = next-contiguous-bar-open-after-decision-v1`. A test asserts the same-bar
fill is impossible rather than merely unused, and a mutant that fills at
`open[T]` is killed.

## Gaps

If the bar immediately after a decision is missing, the target **expires** and
is recorded in `expired_targets`. Executing it hours later would trade on an
intention the model never had at that later price.

An already-open position is **carried** across the gap. Closing before a gap
would use the knowledge that the gap was coming; the position is revalued at
the next observable open instead.

## Execution contract V1

| | |
|---|---|
| `instrument_model` | `synthetic-linear-usd-notional-v1` |
| `fee_rate` | 0.0010 (10 bps per fill, on absolute notional) |
| `slippage_rate` | 0.0005 (5 bps, always against the trade) |
| `initial_equity` | 100000 USD |
| `mark_policy` | `next-observable-open-v1` |
| `final_liquidation_policy` | `next-observable-open-after-last-fill-v1` |
| `target_quantity_basis` | `reference-market-open-v1` |
| funding / borrow | 0, and a non-zero value is refused — V1 froze no accrual schedule |
| `execution_spec_hash` | `99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb` |

```
EXECUTION_COST_V1_IS_NOT_OPTIMIZED = true
EXECUTION_COST_V1_IS_NOT_EXCHANGE_ACCOUNT_SPECIFIC = true
```

These are a **synthetic infrastructure contract**. They are not Coinbase's fee
schedule, they were not tuned, and they do not describe any real account.

Slippage moves the price against the trade in both directions: a buy fills at
`open × (1 + 0.0005)`, a sell at `open × (1 - 0.0005)`. Fees are
`|quantity × fill_price| × fee_rate`, always positive, deducted from cash, and
charged only when the traded quantity is non-zero.

## Target to quantity

At each admissible fill the engine marks the book at the bar's open, computes
`pre_trade_equity`, then

```
target_quantity = (target_exposure × pre_trade_equity) / reference_open
trade_quantity  = target_quantity − current_quantity
```

The quantity is priced at the **observable open**, not the fill price. That
keeps slippage a separable, measurable execution cost instead of silently
reshaping the intended exposure. A reversal is a single net delta, and fee and
slippage apply to the whole of it.

Because exposure is a fraction of NAV, a repeated identical target still
implies a small rebalance once costs have moved NAV. That is real, not noise.

## Accounting

`equity = cash + position_quantity × mark_price` at every snapshot, asserted by
test. Cumulative fees and slippage are monotone. Shorts are permitted and carry
no borrow or funding charge — an explicit limitation of V1, not an oversight.
If equity ever reaches zero the run **fails closed** rather than continuing
through impossible divisions.

## Metrics, fixed in advance

`initial_equity`, `final_equity`, `gross_pnl`, `net_pnl`, `gross_return`,
`net_return`, `total_fees`, `total_slippage_cost`, `total_execution_cost`,
`turnover_ratio`, `max_drawdown`, `annualized_sharpe`, `fill_count`,
`rebalance_count`, `expired_target_count`, `average_abs_exposure`,
`exposure_time_fraction`.

* **turnover** = Σ |trade notional| / pre-trade equity, summed over fills.
* **max drawdown** is negative by convention: −0.15 means a 15 % fall from the
  running peak, measured on net equity.
* **Sharpe** is descriptive: hourly returns of net equity, zero risk-free rate,
  annualised by √8760. `None` when there are fewer than two returns or zero
  variance. No significance is claimed, and none is testable here.

**Gross vs net** are two accounting views of the *same* target stream, not two
strategies: the gross path simulates the identical targets under a zero-cost
execution spec, so the difference measures exactly what execution took.

## What this engine cannot do

It cannot manufacture predictive information. V1 and V2 both measured global
rank correlations near zero. An execution simulator applied to a signal without
edge produces an honest picture of costs, nothing more.

A negative net return is not a failure of the backtester. It is the backtester
working.
