# Trading Lab — paper / shadow trading V1

**NO REAL MONEY. NO BROKER. NO PRIVATE EXCHANGE API. NO EXCHANGE ACCOUNT.**

**THE SHADOW MODEL IS NOT OPTIMIZED. PAPER RESULTS ARE NOT COMMERCIAL EVIDENCE.**

**BTC/ETH PAPER MODE IS AUTOMATICALLY EMBARGOED FROM 2026-09-01 THROUGH
2026-11-30 TO PRESERVE THE V2 CONFIRMATORY HOLDOUT.**

Shadow mode runs the whole live chain with simulated money:

```
public closed candle → causal V2 features → frozen shadow model
  → Signal V1 → Risk V1 → simulated fill → paper portfolio
  → append-only event log → read-only API → cockpit
```

What it tests is **infrastructure**: ingestion, causality, orchestration,
persistence, restart safety, monitoring. It does not test whether the strategy
makes money — V1 and V2 both measured rank correlations near zero, and the
Phase 5C economic backtest lost 3.7 % (BTC) and 4.6 % (ETH) net of costs.

## The holdout embargo

The confirmatory window reserved in Phase 4D is:

| | |
|---|---|
| products | BTC-USD, ETH-USD |
| window | 2026-09-01T00:00:00Z → 2026-11-30T23:00:00Z (inclusive openings) |
| embargo truly lifts | 2026-12-01T00:00:00Z, once the final protected bar has closed |
| purpose | the single confirmatory evaluation of Benchmark V2 |

Shadow trading would walk straight into it on the first of September, so the
guard is structural rather than procedural. **Nobody has to remember to stop the
daemon.**

The window is not restated in the guard — it is read from the committed Phase 4D
contract, so the two cannot drift. Protection is applied in four places, all of
them *before* anything is kept:

1. `require_tradeable_now` — once the window is active the product stops
   polling entirely and the session moves `RUNNING → EMBARGOED`.
2. `require_unprotected_request` — a request whose range overlaps the window is
   refused before it leaves the process, so reserved data is never even asked for.
3. `_refuse_protected_payload` — the raw response is scanned before parsing,
   because a venue is free to return more than it was asked for.
4. `require_unprotected_bar` — every canonical opening, again, at ingestion and
   again at the top of the pipeline.

A protected candle therefore never reaches the store, the features, the model,
the event log or the UI. The one thing recorded at the boundary is a
`PROTECTED_HOLDOUT_BOUNDARY_REACHED` event, which carries the window and no
market data at all.

Adversarial cases covered by tests: the exact boundary hour, a `+02:00` offset
that is really midnight UTC, a naive timestamp, a clock jump over the boundary,
a range that swallows the window, a venue returning August and September in one
page, a restart during the embargo, and the last protected bar still forming at
23:59 on the 30th of November.

## The shadow model

One Ridge per product, `alpha = 1.0`, on the exact V2 feature set
(`return_1`, `return_4`, `return_12`, `ema_spread_12_26`, `rsi_14`,
`atr_pct_14`), target `forward_return` h=4.

Ridge is used because it already had a frozen contract, is deterministic and is
cheap to run hourly — **not** because it performed better. Choosing the better
performer would be one more selection pass over data that has already been spent.

Trained **once**, before the first live observation, on the already-spent
historical corpus (2025-08-01 → 2026-07-31). Nothing from August 2026 onwards
reaches the fit, and a training row past the frozen end is refused. There is no
online retraining in V1, and a test asserts the engine never calls the trainer.

The artefact is canonical JSON, not a pickle: a fitted Ridge is a handful of
Decimals, and storing them as text keeps the model inspectable and reproducible
without trusting a binary format or a library version.

```
optimized = false      research_evidence = false      shadow_only = true
```

## Execution timing, stated honestly

A target derived from candle T is priced at the open of candle T+1h — the same
rule as the backtest, and a price that exists at the instant the target does, so
nothing is backdated.

But a system that only sees **closed** candles does not *observe* that open
until T+2h. So the fill is recorded an hour after the price it is filled at
becomes observable. `PaperExecutionSpec` names that latency
(`recorded-when-the-fill-bar-closes-v1`) instead of hiding it, and the cockpit
shows it.

Two further differences from the backtest contract, both explicit:

* a live session **never liquidates** — it is open-ended;
* paper and backtest timelines are therefore not directly comparable, and the
  spec says so in `differs_from_backtest`.

Costs are identical to Phase 5C: 10 bps fee, 5 bps slippage, 100,000 USD of
synthetic initial equity, per product, never pooled. The fill accounting itself
is *the same code*: `economic_backtest.apply_position_target`, extracted in this
phase so that shadow trading could reuse it rather than grow a second economic
engine that would eventually disagree.

## Durability

An append-only SQLite log (WAL, `synchronous=FULL`) under `var/trading_lab/`,
which is git-ignored: this is operational state, not a research artefact.

Every event commits to its predecessor by hash, so truncation, reordering and
tampering are detectable rather than merely unlikely. Every event that belongs
to a candle carries a natural key, and the database refuses a duplicate — a
restart replays what is missing and nothing else. Periodic state snapshots keep
recovery bounded, and each verifies its own hash before it is trusted.

Restart scenarios covered: before the candle, after ingestion, after the
prediction, between the target and its fill, and after the fill. Each produces
exactly one pipeline and one fill.

## Control

```bash
./scripts/paper_shadow.sh start     # explicit, deliberate
./scripts/paper_shadow.sh status
./scripts/paper_shadow.sh stop
```

The read-only API has **no** start, stop, buy, sell or override endpoint. A
cockpit that can start a trading process is one XSS away from doing it by
accident, so control stays on the command line. `./scripts/dev_app.sh` still
runs the API and the cockpit alone; the shadow daemon is never started for you.

## What this does not establish

```
shadow_mode                     = true
real_money                      = false
broker_connected                = false
commercial_edge_established     = false
v2_confirmatory_holdout_observed = false
v2_confirmatory_holdout_spent    = false
```

Paper P&L under a synthetic cost model, driven by an unoptimised shadow model,
on a signal with no measured edge, is an infrastructure test. It is not
evidence about money.
