# US equity / ETF market semantics — V1

Phase 6D. The first non-crypto market in HyprL.

This document describes what was built, and — more usefully — what it
deliberately refuses to do. Nothing here trades an equity. Nothing here
captures one. The phase exists so that when equity data eventually arrives, it
arrives into a system that already knows what a session is.

---

## 1. Why a calendar is not a detail

Two numbers in Phase 1–5 quietly assume crypto: **24 bars a day** and **8760
periods a year**. Both are correct for a market that never closes and wrong for
every other asset class.

A US equity regular session is 6h30. On a 30-minute grid that is 13 bars, and a
year holds about 251 sessions — roughly **3263** periods. Annualising an equity
Sharpe ratio by 8760 instead overstates it by a factor of about 1.6.

That error is dangerous precisely because it is invisible: it produces a
plausible number that nobody re-derives. So the annualisation factor is not a
constant in a formula. It is asked of the calendar, and each calendar answers
for its own market.

```
CRYPTO_24_7        1h    24 bars/day   8760 periods/year
US_EQUITY_REGULAR  30m   13 bars/day   3263 periods/year   (251 sessions × 13)
```

`get_calendar()` **refuses** an unknown calendar. There is no fallback to 24/7,
not even temporarily, because a wrong calendar that runs is far more dangerous
than a missing one that raises.

## 2. The rules are not written here

None of the holiday logic is hand-coded. Not Thanksgiving, not Christmas, not
Juneteenth, not the early closes, not the daylight-saving shifts.

A hand-written holiday table is wrong the moment one of those changes, and the
change is silent. Juneteenth became a US market holiday in 2021; every table
written before then is now wrong and looks fine.

So the schedule comes from **`pandas_market_calendars`, pinned exactly to
5.4.0** behind the optional `[equities]` extra:

```bash
pip install 'hyprl[equities]'
```

The pin is exact rather than a floor. A newer release can move a holiday, and
every schedule, bar grid and annualisation factor derived from it would change
without a single line of code changing. `USEquityRegularCalendarSpec` records
the library, its version, the calendar name, the timezone and the session type,
and hashes them — so a corpus captured under one version is provably not the
same corpus as one captured under another.

### Calendar identity

| field | value |
| --- | --- |
| `calendar_provider` | `pandas_market_calendars` |
| `calendar_provider_version` | `5.4.0` |
| `calendar_name` | `XNYS` |
| `calendar_id` | `US_EQUITY_REGULAR` |
| `timezone` | `America/New_York` |
| `session_type` | `REGULAR` |
| `spec_hash` | `1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314` |

`XNYS` is used because it is the registered name in this library; `XNAS` is not
registered at all, and `NASDAQ` resolves to the same NYSE rule set — verified,
not assumed: both produce identical 2026 schedules, 251 sessions apiece. The
*venue* of a Nasdaq-listed instrument is still `xnas`. Venue, calendar and
provider are three different facts.

### What V1 covers

Only the **regular session**. Pre-market and after-hours have their own
liquidity, their own halts and their own reference prices; claiming them would
mean claiming bars this platform has never validated. Asking for them raises.

## 3. Session semantics

```
regular session   09:30–16:00 New York   6h30   13 × 30m bars
early close       09:30–13:00 New York   3h30    7 × 30m bars
holiday / weekend no session at all
```

Facts the implementation gets right, each verified as a property rather than
against a copied table:

* **DST.** The local open never moves (09:30). The UTC open does: 14:30Z in
  winter, 13:30Z in summer. A calendar storing "14:30 UTC" would be an hour
  wrong for eight months of the year — exactly one bar on a 30-minute grid.
* **Session length is unchanged by DST.** The clock changing must not create or
  destroy half an hour of trading.
* **A session never crosses midnight in New York**, though in UTC it routinely
  crosses into the next date.
* **An early close is shorter, not special.** It simply yields fewer bars. No
  bar is synthesised to make the count match a normal day.
* **A bar never outlives its session.** The last bar of a short session ends
  when the market does, not one interval after it opened.

### 1h is refused for equities

6h30 contains no whole number of hourly bars. `bars_per_day("1h")` raises
rather than rounding, and the equity instruments do not list `1h` among their
native timeframes at all. A bar count that is a rounding is not a fact.

## 4. Gaps

The single most useful thing in this phase.

Under the crypto rule, the next bar is always one interval later. Applied to an
equity series, that rule reports a gap every single evening, every weekend and
every holiday — thousands of false positives that train everyone to ignore the
gap detector.

`ExpectedBarGrid` fixes the definition. It enumerates every bar opening the
calendar says should exist, and gap detection compares against that and nothing
else:

* overnight — **not a gap**, no bar was expected
* weekend — **not a gap**
* holiday — **not a gap**
* after an early close — **not a gap**
* a bar missing *inside* a session — **a real gap**, and still reported
* a bar present where none was expected — reported as **unexpected**

## 5. Causality

Unchanged from crypto, and for the same reason: a bar's high, low and close do
not exist until it ends. `BarAvailability` states it explicitly because an
equity bar's close is decided by the session, not by adding an interval to the
open — which is what makes the last bar of an early close correct.

```
available_at == bar_close_at == bar_open + timeframe
```

There is deliberately no clamp to the session close. The grid never emits an
opening whose bar would outlive its session, so a clamp would be a branch that
can never run — and worse than none, because it would silently truncate a bar
instead of rejecting a grid that had started producing partial ones. The grid
membership check in `bar_availability` is what enforces the property, and an
opening that is not on the grid is refused outright.

## 6. Corporate actions

A crypto candle is a fact about a market. An equity bar is a fact about a
company as much as a market, and companies change shape.

### Adjustment policy is part of identity

| policy | meaning |
| --- | --- |
| `RAW` | prices exactly as printed on the day; a split shows as a discontinuity |
| `SPLIT_ADJUSTED` | historical prices restated in current shares; dividends **not** applied |
| `TOTAL_RETURN` | **not implemented** — refused, never silently substituted |

The policy travels with every bar and is part of its hash. Two bars with
identical numbers under different policies have **different hashes**, and an
`EquityCorpusSpec` differing only in policy is a **different corpus**.

This is not bookkeeping. The failure it prevents is a series that is raw before
a split and adjusted after it: it shows a several-hundred-percent return on one
bar, and every risk number computed from it is wrong. Both halves look
perfectly normal on their own. `require_single_policy` refuses such a series.

### Splits

Ratios are exact `Decimal` fractions — a 3-for-2 is 1.5, a 7-for-3 is not
representable, and rounding compounds across a history. The effective date is
the first session on which the new share count applies: the bar *on* that date
is already adjusted, the bar before it is not. Off by one moves a large fake
return by a day.

`apply_splits` requires a `RAW` series and emits `SPLIT_ADJUSTED`, so adjusting
twice raises instead of dividing every price by the ratio a second time — a
mistake that produces a perfectly smooth, completely wrong history.

### Dividends

Recorded, explicitly **not applied**. `applied_to_prices: false` on every one.
A price return understates a total return by the dividend; recording it makes
that understatement visible rather than unknown.

## 7. The provider

`massive-stocks-historical-v1`. Three boundaries define it.

### The transport is injected

Nothing in the module opens a socket. A provider built without a transport gets
`NoNetworkTransport`, which **raises** — it does not return empty data, because
an empty result reads as "this instrument had no bars in that window".

This is what makes Phase 6D genuinely offline. Not a mock intercepting a
request: a provider that structurally cannot make one. **No real Massive call
was made during this phase, and no API key is needed to validate it.** The
provider registered in `PROVIDERS_V1` is the offline one.

### Market data only

There is no `place_order`, no `account`, no `balance`, no `position` — not
unimplemented, absent. Mechanically enforced by an **allowlist**:

```
/v1/stocks/bars   /v1/stocks/splits   /v1/stocks/dividends   /v1/reference/tickers
```

A denylist would have to anticipate every account endpoint a vendor might ever
add, and it only has to miss one. Paths containing `account`, `order`,
`balance`, `wallet`, `position`, `withdraw`, `transfer`, `trade`, `execution` or
`portfolio` are refused with an explanation rather than a generic error.

`data_freshness = END_OF_DAY`, never `LIVE`. `get_latest_closed_bars` raises:
handing a live caller the most recent end-of-day row would be handing them a
stale price.

### The credential

`HYPRL_MASSIVE_API_KEY`, read at request time, wrapped in `Secret`, and never
held as a plain string on any object that can be printed.

Never committed · never logged · never in an API response · never in a support
bundle · never in an exception · never in the frontend.

`Secret` overrides `__repr__`, `__str__`, `__format__`, `__reduce__` (no
pickling) and `__hash__` (no dict keys). `reveal()` is the single, greppable way
out. The provider payload reports `credential_configured: true|false` and has
**no field named `api_key`** — a redacted one would be safe today and would be
the obvious place for someone to put the real one later.

A vendor exception is re-raised as our own type with the chain broken, because
a vendor's error message often quotes the request it failed on, headers
included.

Tests use a sentinel value and search whole structures — payloads, reprs, log
records, support bundles, exception text — rather than asserting that one
particular field was blanked. That is how leaks actually happen: nobody logs a
key on purpose, they log a config dict that contains one.

## 8. The seed universe

| instrument | venue | class | calendar | tradable |
| --- | --- | --- | --- | --- |
| `xnas:AAPL` | xnas | EQUITY | US_EQUITY_REGULAR | **no** |
| `xnas:MSFT` | xnas | EQUITY | US_EQUITY_REGULAR | **no** |
| `xnas:NVDA` | xnas | EQUITY | US_EQUITY_REGULAR | **no** |
| `xnas:QQQ` | xnas | ETF | US_EQUITY_REGULAR | **no** |

Venue and asset class are **proven, not assumed**: each is asserted against a
recorded vendor reference payload in `tests/fixtures/crypto/`. An unknown
exchange code or security type is refused rather than defaulted — a guessed
venue names a different instrument under a plausible id, and every artefact
referencing it would be wrong in a way no price test would catch. A non-USD
listing is refused rather than converted.

### Known is not tradable

Two registries, because they answer two different questions.

* `INSTRUMENTS_V1` — what this build **trades**: BTC-USD and ETH-USD. Unchanged.
* `CATALOGUE_V1` — what this build can **describe**: those two plus the four
  equities.

This separation is not tidiness. The shared paper portfolio builds its batch
from the tradable registry and waits until every instrument in it has supplied
a target. An equity in that list would mean **a batch that never completes and
a paper runtime that stops trading without ever raising**.

`is_tradable()` is fail-closed: anything unresolvable is not tradable.

The API marks every instrument with `tradable`, and the instrument picker in
the UI offers only tradable ones — offering AAPL in a control that drives a
backtest would promise a run that cannot happen. Reference markets get their
own section, labelled `NOT TRADED`.

## 9. Optional dependency isolation

The crypto core must keep running on an install that has no pandas. It does:

* nothing imports `pandas` at module scope outside `equity_calendar.py`
* `trading_calendar` names `US_EQUITY_REGULAR` without importing it, and loads
  it on demand
* the instrument registry, the equity provider and all corporate-action
  arithmetic work with pandas blocked
* a missing extra raises a message naming `[equities]` — and **never** returns
  the crypto calendar instead

Verified in a fresh interpreter with pandas actively blocked, because
"installed and unused" and "not required" are different claims.

## 10. API

```
GET /api/v1/calendars
GET /api/v1/calendars/{calendar_id}
GET /api/v1/instruments                       (now the full catalogue)
GET /api/v1/instruments/{instrument_id}
GET /api/v1/instruments/{instrument_id}/sessions?start&end&timeframe
```

The sessions endpoint enumerates real rows, so an unbounded range would be an
unbounded response: it is capped at **400 days**, defaults to 30, and refuses a
reversed or malformed window. A continuous market reports `continuous: true`
and `session_count: null` rather than a fabricated row per day.

Every derived number — bar counts, annualisation, session lengths — is computed
server-side. The browser formats and never computes: a second implementation of
the session rules in TypeScript would agree with Python right up until a holiday
moved, and then the page would show a schedule the backend does not believe in
with nothing reporting an error.

## 11. What this phase explicitly does not do

No live equities. No paper equities. No equity in the portfolio. No equity
model, signal, prediction or backtest. No equity corpus captured — the
`EquityCorpusSpec` is defined and hashed, and `captured: false`. No V3. No
tuning. No news, no fundamentals. No holdout touched. No broker, no order, no
real money.

`commercial_edge_established = false`.

## 12. Frozen contracts

Unchanged by this phase, and asserted so:

| contract | hash |
| --- | --- |
| SignalSpec V1 | `7f8b57f8…` |
| RiskSpec V1 | `f3be4fc6…` |
| ExecutionSpec V1 | `99295b9a…` |
| PaperModelSpec V1 | `830f5227…` |
| PaperExecutionSpec | `bb944167…` |
| PortfolioSpec V1 | `32ec6c9f…` |
| Protected holdout | `bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85` |
| BTC-USD instrument | `492c167c1e66a37a377cff8b4e135841c5a13a7c60324ec9b5c8b1976bf5701f` |
| ETH-USD instrument | `2a9e1d1c922fbb9af68ad92f8f2638e951a2fef830afb6e514a29ebfab35d7f3` |

The protected research window remains unobserved and unspent.
