# ETH paper A/B — news overlay versus control

**Status:** paper-only experiment. No live brokerage endpoint exists in this
implementation.

## Accounts

| Variant | Purpose | Account guard |
|---|---|---|
| `control` | Frozen ETH long/cash strategy | account ID suffix `2189a1` |
| `news` | Same strategy, with a risk-only news multiplier | account ID suffix `4fbcbe` |

The account suffixes are non-secret safety guards. Credentials exist only in
GitHub Actions secrets:

- `ALPACA_ETH_CONTROL_KEY`
- `ALPACA_ETH_CONTROL_SECRET`
- `ALPACA_ETH_NEWS_KEY`
- `ALPACA_ETH_NEWS_SECRET`

## Frozen common strategy

- Instrument: `ETH/USD`, spot only.
- Completed UTC daily bars only; the current partial daily bar is excluded.
- Trend votes: close above the 64-, 128-, and 256-day moving averages.
- Momentum votes: positive 30-, 90-, and 180-day return.
- Composite: equal weight across all six votes.
- Volatility: 30-day realized volatility, annualized over 365.25 days.
- Target volatility: 20%.
- Maximum ETH exposure: 75%; otherwise the account remains in cash.
- Target rounded to 5-percentage-point increments.
- Broker rebalance band: max($250, 2% of account equity).
- Account loss gates: zero exposure at a 25% peak drawdown or a 5% daily loss.
- No leverage and no short selling.

The reproducible discovery test used one-day-lagged signals and 30 bps of
one-way turnover cost. It produced approximately 13.1% annualized return, 0.92
Sharpe, and 18.0% maximum drawdown over November 2017–July 2026. Since January
2022 the same figures were approximately 3.4%, 0.32, and 17.1%. These are
discovery numbers, not a forecast or a profitability guarantee.

## News overlay

The control account never requests news.

The news account queries Alpaca/Benzinga headlines tagged `ETHUSD` or `BTCUSD`
over the previous 12 hours. It can only reduce the common target:

- critical security/protocol headline, without market confirmation: multiplier
  0.5;
- critical headline plus ETH stress (1-hour return <= -1.5% or 6-hour return
  <= -3%): multiplier 0;
- regulatory/political risk or an attention burst plus the same market stress:
  multiplier 0.5;
- otherwise: multiplier 1.

Positive news never increases exposure. Derived event counts and hashes are
recorded; headline text is not persisted. This rule is deliberately simple and
frozen so the forward A/B result cannot be tuned after observing outcomes.

## Execution and isolation

- `scripts/eth_paper/live_eth.py` hardcodes the Alpaca paper and market-data
  hosts and rejects unexpected account IDs, assets, shorts, and foreign
  positions.
- Dry-run is the default and writes nothing.
- Open ETH orders block duplicate submission.
- Orders use deterministic client IDs and are polled through terminal fill
  states.
- The scheduled workflow runs at minute 17 each hour and commits at most one
  daily equity observation per account, plus actual order or risk-event records.
- Both accounts run sequentially in one workflow after the same 17 targeted
  tests pass.

Paper fills are simulated and do not establish achievable live execution.
