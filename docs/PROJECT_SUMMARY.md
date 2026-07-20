# HyprL — Project Summary & Specs

**Last updated:** 2026-07-17
**Status:** 🟢 Two-sleeve paper portfolio LIVE on GitHub Actions (Alpaca paper (paper account id kept private), ~$112.5k)

This is the single reference document for the project's current state. It supersedes
the older VPS-based architecture described in earlier docs (those servers are dead).

---

## 1. What HyprL is

Solo systematic trading project. The goal is an honest, survivorship-corrected,
cost-tested book that actually runs and produces a verifiable track record —
after a history of strategies that looked good in backtest and died silently in
production.

**The one hard-won lesson:** every attractive backtest number this project has
produced was a mirage until proven otherwise (survivorship, costs, or regime).
The only numbers worth trusting are the ugly, honest ones.

---

## 2. The deployed system (what is running now)

A **two-sleeve portfolio**, both sleeves sourced from Clenow's books, run against
one Alpaca paper account. Honest combined metrics (survivorship-corrected, net of
costs): **Sharpe ~0.65, Max DD ~15%** vs SPY 0.62 / 55%. The edge is drawdown
reduction, not excess return.

### Sleeve A — Equity momentum (60%) — *Stocks on the Move*, ch. 7-11
| Spec | Value |
|---|---|
| Universe | Current S&P 500 |
| Cadence | **Wednesdays only** |
| Score | Annualized 90-day exp-regression slope × R² |
| Filters | Above 100-day MA; no >15% gap in 90d; price ≥ $5 |
| Sizing | Risk-parity `equity × 0.001 / ATR20`, capped 15%/position |
| Regime gate | New buys only if SPY > 200-day MA |
| Hysteresis | Hold a name while it stays in the top 20% of the ranking |
| Standalone honest Sharpe | 0.59 (point-in-time), cost-robust to 20 bps |

### Sleeve B — ETF cross-asset trend (40%) — *Following the Trend*, ch. 4
| Spec | Value |
|---|---|
| Universe | 18 ETFs: SPY QQQ IWM EEM EFA / TLT IEF LQD HYG / GLD SLV DBC USO DBA / UUP FXE FXY / VNQ |
| Cadence | Daily |
| Direction | Long if MA50>MA100, short if MA50<MA100 |
| Entry | 50-day close breakout |
| Exit | 3×ATR100 trailing stop |
| Sizing | ATR risk-parity at 20 bps, gross capped 1.5× |
| Standalone honest Sharpe | 0.39 (weak — trend-following in drawdown since ~2015) |

### Why two sleeves
Correlation momentum↔trend = **+0.19** (trend↔SPY = −0.19, it hedges equities).
Combining a 0.59 sleeve with a weak-but-decorrelated 0.38 sleeve barely moves
Sharpe but **halves the drawdown** (29% → ~15%). Carver's diversification "free
lunch", the one that actually materialized (unlike momentum+MR, which are 0.65
correlated and don't diversify).

### Execution
- Alpaca paper (paper account id kept private), fractional shares, shorting enabled, multiplier 4
- Cron **21:00 UTC** (1h after US close): signals on the completed daily close →
  market orders → filled at next open. Running during market hours would use an
  incomplete daily bar and produce wrong signals.
- Live code validated against the backtest engine by walk-forward through the same
  decision functions: **Sharpe drift 0.04**.

---

## 3. Infrastructure

**Runs on GitHub Actions, not a VPS.** Both the Oracle VPS (89.168.48.147) and the
GCP VM (34.163.37.255) are dead — free-tier reclaim of idle instances. That is what
flat-lined the account from 2026-06-18 for three weeks unnoticed. Actions is
free/unlimited for this public repo; committing state each run also keeps the repo
active, preventing the 60-day schedule auto-disable that had killed the old
track-record workflow.

| Component | Path |
|---|---|
| Workflow (merged to `main`) | `.github/workflows/portfolio-daily.yml` |
| Engine | `scripts/momentum_stocks/live_portfolio.py` |
| Watchdog | `scripts/momentum_stocks/watchdog.py` |
| Public track record | `live/portfolio/equity.jsonl` (git-timestamped) |
| Backtests | `scripts/momentum_stocks/{clenow_backtest,etf_trend,portfolio,...}.py` |
| Research writeup | `docs/EQUITY_STRATEGY_RESEARCH.md` |

**Secrets (GitHub repo):** `ALPACA_KEY`, `ALPACA_SECRET`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_ALERT_CHAT`. Never hardcoded — the repo is PUBLIC; the engine exits loudly
if they're missing.

**Watchdog** (`if: always()`): alerts Telegram **@F4llenHyprL_bot**
on engine silence (>50h) or Alpaca failure. Tested against stale-heartbeat,
never-started, and spam paths. The heartbeat is written even when the engine
crashes, so "crashed" is distinguishable from "never ran". A `pgrep` was NOT used
for liveness — it produces false positives in both directions (it matched its own
command line three times this session); the watchdog checks a self-written
heartbeat + the Alpaca account instead.

---

## 4. Everything that was tested and REJECTED (with the reason)

Rejection is the main product of the research. Each was killed by measurement, not
opinion.

| Idea | Verdict |
|---|---|
| Momentum "Sharpe 0.90" (current S&P list) | Survivorship mirage → **0.59** point-in-time |
| Momentum Nasdaq-100 "0.97" | Survivorship again; least-biased sub-period < SPY |
| Vol-targeting overlay (Carver/Moreira-Muir) | No alpha — only slides return↔DD on the same Sharpe line |
| Connors RSI(2) mean-reversion "0.70" | Dies on costs: 0.70@5bps → 0.39@10bps → 0.07@15bps across 19k trades |
| Momentum + MR combo | Correlation 0.65 (both long-equity beta) → no diversification |
| Turtle crypto (BTC+ETH, prior work) | Edge dead since 2023 (Sharpe 0.25) |
| News sentiment (FinBERT/LM) | IC **−0.014, wrong sign** — positive news predicts slight negative return |
| News price-reaction / PEAD | IC ~0 — dead too |
| News volume (n_news) | IC +0.024, t=4.6 — the only weak survivor, needs no NLP, marginal, not standalone |

Details: `docs/EQUITY_STRATEGY_RESEARCH.md`. News pipeline: `scripts/news_signal/`.

---

## 5. Prop firm challenge — analyzed and ABANDONED

Goal was: track record → prop challenge → funded account. **Verdict: prop challenges
are structurally incompatible with our edge.** Three independent blockers, any one
fatal:

1. **Calmar gap.** Our honest book has Calmar ~0.35. FTMO needs ~1.0 (+10% before
   −10%), Trade The Pool ~2.1 (+15% before −7%). We are 3-6× short. Calmar is
   scale-invariant — no sizing trick fixes it.
2. **Instrument mismatch.** Our edge lives in broad US equities + diversified ETFs
   incl. bonds. Prop firms offer FX (trend Sharpe −0.05 since 2015), indices, and
   too few stocks (FTMO: 23 stocks → cross-sectional momentum unrunnable). Trade
   The Pool has 12,000 instruments but a punishing 15%/7% ratio.
3. **CFD overnight financing ≈ 7%/yr** on long stock/index CFDs. Our gross edge is
   ~10%/yr → **net ~3%/yr, Sharpe → 0.18**. Kills any months-long-hold strategy on
   any CFD prop firm.

Monte-Carlo P(pass), honest momentum, no time limit: FTMO 76% @ small size but no
stock universe; Trade The Pool 53% (16 months) because of its ratio. **The winning
combination (FTMO's rules + a wide stock universe) does not exist in one firm.**

The Turtle EA's `InpTrailDistance=0.8` trailing stop was also shown to **destroy the
edge** (diversified Sharpe −0.03); the +113% XAUUSD result is a single-asset regime
artifact (gold $1200→$3700), not an edge.

Recommendation: **do not buy a challenge.** Own capital, however small, is the only
vehicle where this edge survives. Detail: memory `prop-firm-viability`.

---

## 6. Current state & next step

- 🟢 **34 positions, ~$112.5k, live on GitHub Actions.** First real trade since June 18.
- P&L is ±$40/line — pure noise at this stage. **Do not read anything into it for
  4-6 weeks.**

**Next step: add nothing. Let it live.** The #1 risk to this project has never been
a shortage of ideas — it's that nothing survives long enough to prove itself. What
to watch:
- 🚨 Telegram alert → engine down, fix it
- 📉 Drawdown > 15% → beyond expectation, investigate
- ✅ Silence → nominal, do nothing

The genuinely decorrelated next sleeve, if/when ever: crypto **carry / funding-rate**
(market-neutral, not the dead directional Turtle) — a 2-3 week research project with
real exchange APIs, not an account to open. Find the edge first, paper it second.

---

## 7. Memory index (cross-session)

- `deployment-architecture` — servers dead, runs on Actions now
- `equity-strategy-research` — the momentum/trend research and honest numbers
- `prop-firm-viability` — why prop is off the table
