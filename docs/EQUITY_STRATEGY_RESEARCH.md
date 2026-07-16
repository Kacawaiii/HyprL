# Equity Strategy Research — honest search for a stock system

**Date**: 2026-07-11
**Context**: after the XA Turtle (BTC+ETH) regime collapse (Sharpe 0.25 since 2023)
and the idle Alpaca equities ML account, the decision was to start a **new
strategy from scratch, sourced from the trading e-book library**
(`/home/kyo/mega/Trading E-Books/`, 486 PDFs), backtested honestly on US stocks.

**Method**: pick the best book-specified systematic stock strategy, implement its
exact rules, and validate it the same way the Turtle was — costs, sub-period
regime test, parameter robustness, Monte Carlo, and (critically) **survivorship-bias
correction with point-in-time S&P 500 membership**.

Code: `scripts/momentum_stocks/`. Data: `data/momentum/` (yfinance daily,
2007-2026, 503 current + 216 recovered delisted tickers, 79 corrupted dropped).

---

## TL;DR

1. The first strategy tested — **Clenow *Stocks on the Move* momentum** — looked
   excellent on current constituents (Sharpe 0.90) but that was **survivorship
   illusion**. Point-in-time correction cut it to **Sharpe 0.59**.
2. Momentum on a volatile universe (Nasdaq-100) looked even better (0.97) but its
   least-biased sub-period (2022-2026) underperformed SPY (0.65 vs 0.74) —
   confirming the gloss is survivorship, not edge.
3. **Vol-targeting (Carver/Moreira-Muir) adds no alpha** — it only slides return
   for drawdown along the same Sharpe line.
4. Connors 2-period RSI mean-reversion looked like the winner at 0.70 — but
   **it does not survive realistic costs**. Across 19k trades, Sharpe halves at
   10 bps/side (0.70→0.39) and dies at 15 bps (0.07). The 0.70 was a 5-bps mirage.
5. **Momentum + MR do NOT diversify** — correlation 0.65 (both long-only equity
   beta). Combined Sharpe 0.71 ≈ MR alone. The Carver "free lunch" needs a
   genuinely decorrelated sleeve (different asset class / market-neutral), not a
   second long-only equity strategy.
6. **The cost-robust honest winner is momentum (PIT 0.59)** — it trades little and
   holds for months, so Sharpe barely moves from 5 to 20 bps (0.90→0.83). Modest,
   but real. Its edge is drawdown protection (~half of SPY), not excess return.

**Honest conclusion**: no long-only US-equity strategy convincingly beats SPY on
risk-adjusted return after realistic costs; the real, repeatable edge is
**drawdown reduction**. Momentum (0.59, cost-robust) is a deployable defensive
sleeve; MR is a cost mirage. A portfolio Sharpe > 0.9 requires pairing the equity
sleeve with a genuinely decorrelated non-equity sleeve, not more equity beta.

---

## The strategies, honestly

All numbers net of 5 bps/side costs, 2007-11 → 2026-07, $100k, PIT membership
where noted.

### 1. Clenow *Stocks on the Move* (momentum) — book: `0116`, ch. 7-11

Rules (exact): trade Wednesdays; rank by annualized 90-day exp-regression slope ×
R²; disqualify below 100-day MA or >15% gap in 90 days; risk-parity size
`equity*0.001/ATR20`; only buy when SPY > 200-day MA; sell when a name leaves the
top 20% of the ranking. Guards added for delisted data: min price $5, 15% per-name
cap, drop tickers with any >60% daily move (data artifacts).

| Universe | Sharpe (full) | Max DD | recent 22-26 | note |
|---|---|---|---|---|
| Current S&P 500 | 0.90 | 23% | — | **survivorship-inflated** |
| **Point-in-time S&P 500** | **0.59** | 29% | 0.50 | honest |
| Nasdaq-100 (current) | 0.97 | 18% | 0.65 | survivorship gloss; recent < SPY |

Sub-period (PIT): 2008 0.13 / recovery 0.82 / 2015-18 0.25 / covid 0.94 /
recent 0.50. Mean 0.53, CoV 59%. Regime-stable (no collapse) but modest.

Parameter robustness (current-list, pre-PIT): every config profitable, Sharpe
0.79-0.98 across reg-window {60,90,120}, risk {0.0005-0.0015}, hold {10-30%},
cost {5-20 bps}. No lucky-parameter island.

Monte Carlo (block-bootstrap): P(return>0)=100%, Sharpe p5 0.58 / median 0.96,
but max-DD median -29% with a 23% chance of exceeding -35% (worst -70%). The
realized 23% DD was on the benign side of the distribution.

### 2. Vol-targeting overlay (Moreira-Muir) — book: Carver `0234` ch. 9

Applied to the PIT momentum returns. Best config Sharpe ~0.60-0.62 vs base 0.59 —
**no material lift**. Low targets cut DD (10% target → 18.5% DD) but proportionally
cut return. Strategy vol is already moderate (16%), so no vol-timing juice. Reject
as an alpha source; keep only as a risk-dial if needed.

### 3. Connors 2-period RSI mean-reversion — book: `0114`, ch. 9

Rules (portfolio adaptation): PIT members above 200-day MA, price ≥ $5; enter when
Wilder RSI(2) < threshold, most-oversold first; buy on close, 10% each, max 10
concurrent; exit when close > 5-day MA, or 10-day time stop, or leaves index.

| RSI entry | Sharpe (full) | Max DD | recent 22-26 | trades |
|---|---|---|---|---|
| < 5 | 0.61 | 25% | 0.24 | 15,191 |
| **< 10** | **0.70** | 26% | 0.50 | 19,324 |

Sub-period (RSI<10): 2008 0.17 / recovery 1.02 / 2015-18 0.39 / covid 1.20 /
recent 0.50. Mean 0.66, CoV 59%. Survivorship-robust (3-day holds).

**Cost stress test** (the make-or-break for a 19k-trade system):

| cost/side | Sharpe FULL | Ann% | Sharpe recent 22-26 |
|---|---|---|---|
| 5 bps | 0.70 | 10.8% | 0.50 |
| **10 bps** | **0.39** | 5.2% | 0.19 |
| 15 bps | 0.07 | -0.2% | -0.11 |
| 20 bps | -0.24 | -5.2% | -0.41 |

**MR does NOT survive realistic costs.** It buys the most oversold names (RSI<10),
i.e. stocks in a sharp selloff where spreads widen and adverse selection is worst,
so 10 bps/side is optimistic if anything. At 10 bps the Sharpe halves (0.70→0.39);
at 15 bps it is dead. The 0.70 headline was entirely a 5-bps artifact across 19k
trades — the classic inflated-MR-backtest trap (Chan ch. 7). **Reject for deployment.**

By contrast momentum trades ~12k times but holds for months, so it is
cost-insensitive: Sharpe 0.90→0.83 from 5 to 20 bps (current-list sweep). The
cost-robust honest winner is therefore **momentum (PIT 0.59)**, not MR.

### 4. Momentum + MR combination

Daily-return correlation **0.648**. Both are long-only equity beta.

| Book | Sharpe | Max DD |
|---|---|---|
| Momentum PIT | 0.59 | 29% |
| Mean-reversion RSI2<10 | 0.70 | 26% |
| Combo 50/50 | 0.71 | 26% |
| SPY buy&hold | 0.62 | 55% |

No diversification benefit — the combo equals the better leg.

---

## 5. Decorrelated sleeve — ETF cross-asset trend (Clenow *Following the Trend*)

Book: `0126`, ch. 4 core rules (p.78). Diversified ETF universe (18 ETFs:
equities, bonds, commodities, FX, REIT). Long if MA50>MA100, short if MA50<MA100;
enter on 50-day close breakout; exit on 3×ATR100 trailing stop; ATR risk-parity
size at 20 bps; next-open execution; gross capped at 1.5× (cash-account realism).

| Period | Sharpe | Max DD | Lev |
|---|---|---|---|
| FULL 2007-2026 | 0.39 | 21.6% | 1.09 |
| 2008 crisis | 0.56 | 13.7% | — |
| recovery 10-14 | 0.60 | 15.6% | — |
| 2015-2018 | 0.16 | 11.5% | — |
| covid 19-21 | 0.73 | 10.2% | — |
| recent 22-26 | -0.00 | 21.6% | — |

Weak standalone (trend-following has been in a multi-year drawdown since ~2015),
BUT the point is decorrelation, not standalone Sharpe.

## 6. The portfolio — momentum + trend

Daily-return correlations: **momentum ↔ trend +0.19**, **trend ↔ SPY −0.19**
(trend hedges equities), momentum ↔ SPY +0.60.

| Book | Sharpe | Max DD |
|---|---|---|
| Equity momentum (PIT) | 0.59 | 28.8% |
| ETF trend | 0.38 | 21.6% |
| **Combo 60/40 mom/trend** | **0.65** | **16.4%** |
| Combo inverse-vol | 0.62 | 14.4% |
| SPY buy&hold | 0.62 | 54.7% |

The genuine free lunch: combining a 0.59 sleeve with a weak-but-decorrelated 0.38
sleeve barely moves Sharpe but **halves the drawdown** (29% → 14-16%, vs SPY 55%).
Calmar ~0.42 vs SPY 0.20. Trend earns its keep in crises (2008, covid).

Caveat: recent 22-26 combo Sharpe 0.35 (momentum 0.48, trend −0.04). Trend is
currently a drag — it is crisis insurance that costs premium in calm/bull markets.

## Decision

- **Winner is a two-sleeve portfolio**, not a single strategy:
  - **Equity momentum (Clenow PIT, 0.59, cost-robust)** — return engine.
  - **ETF cross-asset trend (decorrelated, −0.19 to equities)** — drawdown reducer /
    crisis hedge.
  - Combined: Sharpe ~0.65, **Max DD ~15%** (a third of SPY's), Calmar ~2× SPY.
- **Rejected**: Connors MR (dies on costs); a second long-only equity strategy
  (corr 0.65, no diversification); vol-targeting (no alpha).
- **Next**: paper-incubate the two-sleeve book on Alpaca before real capital.
  Optionally down-weight trend now (it is in drawdown) and let the regime filter
  scale it back up when trends return.
- Everything goes through paper incubation, same discipline as the Turtle.

## 7. Live-engine validation (before any real order)

`live_portfolio.py` reimplements the two sleeves for execution against Alpaca, so
it had to be proven equivalent to the validated backtest engines rather than
assumed. `validate_live.py` (signal parity) and `validate_live_walkforward.py`
(drives the live decision functions bar-by-bar over history) were written for this.

| Check | Result |
|---|---|
| Momentum score parity vs `rolling_momentum` | **exact** (max abs diff 0.00e+00, 60 tickers) |
| Trend entry-signal parity vs `etf_trend` | **exact** (0 mismatches / 90 decisions) |
| Momentum pipeline vs reference engine | **Sharpe drift 0.04** (live = 0.6× the book) |
| Trend pipeline | aligned after fixes (see below) |
| Determinism across PYTHONHASHSEED | **identical digests** over 3 seeds |

Walk-forward of the live code (current S&P list, 2010-2026 — survivorship-optimistic
and excludes 2008, so **not** an honest expectation, only a code-equivalence check):

| Book | Ann% | Sharpe | MaxDD% |
|---|---|---|---|
| LIVE momentum | 10.3 | 1.00 | 13.8 |
| ref clenow_backtest engine (100% book) | 16.2 | 0.95 | 23.4 |
| LIVE trend | 2.0 | 0.47 | 7.9 |
| LIVE combo | 12.4 | 1.04 | 13.4 |

**The honest expectation remains the PIT combo (~0.60-0.65 Sharpe, ~15% Max DD),
not these numbers.**

### Three real bugs this validation caught (pre-deployment)

1. **Non-deterministic book** — `momentum_targets` iterated `held` (a `set`); Python
   randomises string hashing per process, so when the budget bound, different names
   were kept across runs. Same inputs → different portfolio. Fixed by iterating in
   rank order (deterministic, and keeps the strongest first).
2. **Sleeve sizing** — both sleeves sized positions off *total* equity then capped the
   total at the allocation, producing a concentrated top-of-list subset rather than
   `alloc ×` the backtested book. Fixed by sizing off each sleeve's own capital;
   momentum Sharpe drift then went to 0.00.
3. **Trend resized daily** — the live engine recomputed the ATR target every day,
   whereas Clenow (and `etf_trend`) size once at entry and hold the share count until
   the exit. The daily resizing added turnover and cost ~0.15 Sharpe. Fixed by
   persisting entry share count in state.

Without this step the deployed system would have been a *different strategy* from the
validated one.

## Files of record

- `scripts/momentum_stocks/clenow_backtest.py` — momentum engine (vectorized,
  PIT membership, data-quality guards)
- `scripts/momentum_stocks/mr_backtest.py` — Connors RSI(2) MR engine
- `scripts/momentum_stocks/vol_target.py` — Moreira-Muir overlay test
- `scripts/momentum_stocks/ndx_backtest.py` — Nasdaq-100 momentum
- `scripts/momentum_stocks/robustness.py` — param sweep + Monte Carlo
- `scripts/momentum_stocks/mr_robust.py` — MR cost/position stress
- `scripts/momentum_stocks/combine.py` — momentum+MR correlation & combo
- `data/momentum/` — prices, membership, results
