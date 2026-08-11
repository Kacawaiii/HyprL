# Portfolio backtest V1 — results

```
STATUS:              EXPLORATORY
CAPITAL MODEL:       SHARED MULTI-INSTRUMENT PORTFOLIO
SOURCE PREDICTIONS:  V2 EXPLORATORY
EXECUTION MODEL:     SYNTHETIC
COMMERCIAL EDGE:     NOT ESTABLISHED
```

BTC-USD and ETH-USD, one cash ledger of 100 000 USD, hourly, over the
recorded V2 out-of-sample window. Nothing was refitted, reselected or tuned;
the thresholds, caps and costs were frozen before this ran.

`result_hash = 6dda821d87bd725d764f9497851e7e99bd97818f686b6c23268c64a7b83e4a35`
`portfolio_spec_hash = 32ec6c9f5f62c24bd18077dda79eefb30a334edcf26b811175f9b93584cdcebf`
`portfolio_backtest_spec_hash = 520b403bd44fa68036635407f1e171401494809559a3cae2936850721c678845`

## Headline

| | |
|---|---|
| Initial equity | 100 000.00 |
| **Final equity** | **91 923.79** |
| **Net return** | **−8.0762 %** |
| Gross return (before costs) | −0.2963 % |
| Net P&L | −8 076.21 |
| Gross P&L | −296.35 |
| Fees | 5 186.57 |
| Slippage | 2 593.29 |
| **Total execution cost** | **7 779.86** |
| Max drawdown | −8.0798 % |
| Annualised Sharpe | −3.967 |
| Portfolio turnover | 51.87 × |
| Average gross exposure | 0.6926 % |
| Average absolute net exposure | 0.6549 % |
| Max observed gross exposure | 34.78 % |
| Fills | 1 636 |
| Rebalances | 7 729 |

**Costs are the entire story.** Before execution costs the portfolio is
essentially flat at −0.30 %. After them it is −8.08 %. The strategy paid
7 780 to lose 296.

## Attribution

| | BTC-USD | ETH-USD |
|---|---|---|
| Gross P&L | −1 492.995 | **+1 196.648** |
| Fees | 1 369.281 | 3 817.291 |
| Slippage | 684.640 | 1 908.646 |
| Execution cost | 2 053.922 | 5 725.937 |
| **Net contribution** | **−3 546.917** | **−4 529.290** |
| Turnover | 14.26 × | 39.47 × |
| Average absolute exposure | 0.2290 % | 0.4636 % |
| Fills | 554 | 1 082 |

ETH made money before costs and lost more than BTC after them. It traded
almost three times as much (39.5 × turnover against 14.3 ×) and its execution
bill was 2.8 × larger. A gross edge of +1 197 was consumed by 5 726 of costs.

Net contributions reconcile with the change in portfolio equity:

```
−3 546.917 + −4 529.290 = −8 076.207 = 91 923.79 − 100 000
relative residual 3.6e-33, tolerance 1e-30
```

## Alignment

Aligned by canonical instrument and timestamp, never by array index.

| | |
|---|---|
| BTC-USD targets | 7 728 |
| ETH-USD targets | 7 728 |
| Timestamps with both instruments | 7 728 |
| Timestamps with one instrument only | 0 |
| Expired targets (no contiguous next bar) | 0 |
| Valuation timestamps | 7 836 |
| Timestamps with no computable equity | 0 |
| First fill | 2025-09-08T02:00:00Z |
| Last fill | 2026-07-29T21:00:00Z |

The two OOS windows coincide exactly, so nothing was dropped to obtain a
join. Had they not, the counts above would have said so rather than the
engine quietly intersecting them.

## Comparing with the two separate accounts

**Do not add the percentages.** The Phase 5C runs were:

| | BTC-USD (5C) | ETH-USD (5C) |
|---|---|---|
| Capital | 100 000 (its own) | 100 000 (its own) |
| Net return | −3.6618 % | −4.5820 % |
| Gross return | −1.5792 % | +1.2379 % |

Those two used **200 000 of synthetic capital in total and never shared a
dollar**. This portfolio uses **100 000 in total**.

The absolute positions are the same in both settings — Risk V1 asks for up to
25 % of NAV per instrument either way — but NAV is now one shared 100 000
instead of two separate ones. So the same dollar exposure sits on half the
capital, and gross exposure per unit of capital roughly doubles: 25 % of total
capital in the split setup against up to 50 % here.

That is why −8.08 % is close to twice the −4.12 % average of the two separate
runs, and it is an artefact of the capital base, not a new finding about the
strategy. Per unit of exposure the picture is unchanged: costs dominate.

Turnover tells the same story. 51.87 × here against 14.26 × and 39.47 ×
separately — the sum, because it is now measured against a single 100 000
rather than against each sleeve's own.

## What the caps did

Nothing. `scaled_batches = 0`: the gross cap of 50 % never bound, because
maximum observed gross exposure was 34.78 % and the average was 0.69 %. The
signal is FLAT the overwhelming majority of the time, so the portfolio is
mostly in cash. The allocation rule is exercised by the synthetic tests, not
by this run.

## What this is not

* Not confirmatory. This corpus has now been observed several times.
* Not evidence of an edge. `commercial_edge_established = false`, and the
  gross figure is negative before any cost is applied.
* Not a live result. No broker, no real money, synthetic execution costs.
* Not a reason to change a threshold. Nothing here was fitted, and nothing
  here may be used to fit anything.

The V2 confirmatory holdout (2026-09-01 → 2026-11-30, BTC and ETH) remains
unobserved and unspent. This run used no network and no new market data.

## Reproducing

```
python -m scripts.trading_lab.run_portfolio_backtest --data-root data/crypto
```

Replaying from scratch reproduces the result hash exactly, as does running
with the instrument order reversed. Every headline metric re-derives from the
stored fills, equity curve and attribution alone.
