# Crypto +200% hunter

Offline Coinbase research with a separately guarded, GET-only Alpaca weekly report. This package
never imports execution clients and never sends orders. Dependencies are in `requirements.txt`.
All cache files, results, budgets, credentials and rendered units belong outside Git.

The hypothesis, candidate grid and acceptance gates were committed before validation in
`protocol.json`. `design-binding.json` binds the design selection by digest. Design receives data
truncated at 2022-12-31; unfinished forward outcomes cannot cross that boundary. The validation
uses unchanged parameters on 2023-01-01 through 2026-08-31, with trailing information expanding
each day and annual results from one continuous portfolio. There is no annual refitting.

```bash
python -m scripts.research.crypto_hunter.study design --cache "$HUNTER_CACHE" --output "$HUNTER_RESULTS"
python -m scripts.research.crypto_hunter.study validate --cache "$HUNTER_CACHE" --output "$HUNTER_RESULTS"
python -m pytest tests/research/test_crypto_hunter*.py -q
```

Validation refuses a mismatching protocol/implementation lock. Keep the original design lock and
validation evidence for an audit; changing code or rules requires a separately reviewed study,
and the already-inspected validation period can no longer be described as untouched.

Signals use completed UTC days and execute at the next calendar day's open. Costs are 0.3% per
side, with double costs and an extra day of execution delay as stress scenarios. Breakout exits
use a stop computed from the entry open or previous completed closing highs; a gap through it
fills at the worse opening price. Current-day highs never lift a same-day trailing stop.
Missing marks carry forward for seven days, then held positions are written to zero. Signal
features never fill missing history. Cash earns zero, there is no leverage, and portfolios
liquidate at the period's final close with exit costs. Initial capital is included in drawdown.

Eligible coins have at least 365 observed calendar days of history, at least 350 observations in
the last year, a complete 200-day MA and 30-day mean dollar volume of at least USD 1 million.
Dollar volume is close times base volume. Stablecoins, metals and duplicate wrapped/staked
assets listed in the protocol are excluded. Historical eligibility changes with trailing data;
today's online status is never used as a historical filter. Nonetheless, the cached universe
comes from a contemporary Coinbase catalogue and may omit historical delistings. First observed
Coinbase candles give a lower bound on age, not coin creation dates or guaranteed listing dates.
The expanding observed high is not a guaranteed full-history ATH. Sector tags are omitted
because there is no reliable dated sector taxonomy in this cache.

Forward winners are exact 180/365-calendar-day close returns of at least +200%, with at least
95% candle coverage. Unfinished or missing endpoints are censored, never treated as losses.
Daily descriptive labels overlap. Monthly eligible signal anchors are the primary capture
denominator; separate per-coin/horizon episodes suppress overlaps. Capture means exposure at
the next open, not that the portfolio realized +200%. Realized +200% position cycles are
reported separately. Hit rate and average win/loss concern closed position cycles, including
resizing cashflows and terminal liquidations. Turnover is gross purchases plus sales divided
by contemporaneous NAV per year. The carré benchmark uses hindsight-selected BTC/ETH/SOL/LINK,
inverse 60-day volatility, monthly rebalancing, individual MA200 filters and a cash allocation
for below-MA names. Missing-history names cannot be purchased. It is not a new stock-picking edge.

The result files include every candidate's design/validation metrics, nearby parameters,
double-cost and delayed execution metrics, yearly returns, forward-winner feature medians,
conditional rates, regimes, ages, liquidity, censoring and winner capture. Conditional rates
are descriptions, not independent statistical tests or newly fitted trading filters.

The weekly scan verifies that `approved-rules.json` is bound to its sibling `validation.json`,
the committed `validation-binding.json` digest and this protocol. With zero survivors it writes a French abstention report without reading
credentials or using the network. With survivors it requires a specific operator authorization:

```json
{
  "operator_signed": true,
  "purpose": "crypto-hunter-weekly-scan",
  "credential_set": "claude-book-momentum",
  "granted_at": "2026-10-10T00:00:00Z",
  "not_after": "2026-10-17T00:00:00Z",
  "scope": {
    "https://data.alpaca.markets": {
      "methods": ["GET"],
      "paths": ["/v1beta3/crypto/us/bars"],
      "max_requests": 60
    },
    "https://paper-api.alpaca.markets": {
      "methods": ["GET"],
      "paths": ["/v2/assets"],
      "max_requests": 1
    }
  }
}
```

This is a synthetic schema example, not an authorization. Only an operator-provided file enables
network access. Asset metadata can instead come from `--assets`: an operator-confirmed JSON
object with `operator_confirmed: true`, timezone-aware `as_of` and `not_after`, and a `symbols`
list of active, tradable USD pairs. No account, position, order or news endpoint is permitted.
The credential file supplies `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`; an optional base URL
must be the paper origin. Values are never printed, copied, committed or evaluated by a shell.
Budgets count attempted requests persistently across restarts; redirects are refused.

```bash
python -m scripts.research.crypto_hunter.scan \
  --approved "$HUNTER_RESULTS/approved-rules.json" --grant "$HUNTER_GRANT" \
  --credentials "$HUNTER_CREDENTIALS" --state "$HUNTER_STATE" --output "$HUNTER_REPORT"
python -m scripts.research.crypto_hunter.install_timer \
  --python "$HUNTER_PYTHON" --worktree "$PWD" \
  --approved "$HUNTER_RESULTS/approved-rules.json" --grant "$HUNTER_GRANT" \
  --credentials "$HUNTER_CREDENTIALS" --state "$HUNTER_STATE" --output "$HUNTER_REPORT"
```

The installer writes only its own `hyprl-crypto-hunter` user service and timer, with
Saturday 10:00 UTC and `KillMode=process`. Existing book/trader/site/radar units remain outside
the installer. Weekly reporting cannot reproduce daily entries/stops; Alpaca liquidity and
breadth also differ from Coinbase and that transfer is unvalidated. Sizing is a conditional
percentage of equity, capped at 0.5% planned risk and 5% per crypto order, with existing exposure,
the reserved sleeve, 35% crypto/60% gross caps, ten-name cap and 8% drawdown halt requiring human
verification. A price stop is given only for the breakout rule; strategies without a price stop
limit suggested exposure to 0.5% equity, so total loss remains inside that nominal risk budget.
