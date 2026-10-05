# HyprL prospective trader v1

This is an operator-authorized PAPER/SHADOW experiment. The first real inference is reserved for
the **2026-10-06 12:00Z user timer**, before the US open. No broker, real order, exchange key,
model training, holdout use, or new Federal Reserve/SEC capture is implemented.

The private grant is supplied at runtime. Its expiry, universe, source hosts, model roles,
timeouts, daily calls/retries and request limits govern dispatch. An expired grant fails closed.
Daily reservations are durable SQLite FULL transactions, shared across restarts and grant updates;
failed requests, retries and interrupted runs consume their reservations. One owner lock covers
run/label jobs. Production CLI invocations share one user-level budget bank and owner lock, so changing
the runtime path cannot reset budgets or create another owner. Dry-run budgets remain isolated.
Every started experiment registers all 14 variants, including failed experiments.
The GDELT spacing clock is durable. Redirects and unapproved transport paths are refused.

The existing crypto holdout overlaps the initial run period: BTC-USD and ETH-USD are **PROTECTED**,
excluded before requests. Warmups and label endpoints are checked against the product-specific
registries. The prospective equity/ETF run can proceed with an explicit partial-context alert.
An authorization naming crypto alone does not remove the holdout protection. No protected data is
read in real mode. Synthetic fakes are explicitly labeled and bypass protection only for fabricated data.

The current grant ends before the primary experiment can reach 60 trading days. A renewed grant is
needed to complete the minimum sample; expiration never silently extends the experiment.

## Preregistration

The authoritative artifact is [trader_agent_preregistration_v1.json](artifacts/trader_agent_preregistration_v1.json),
revision 1, canonical SHA-256 `d8f6468c3460794ad950cf87757da1544ee216b1efcbc6c7f736cf0d66e6665f`.
It is registered before the first prospective inference. Its hash and the unchanged method-file hashes
are recorded in every decision and input-evidence record. A changed method or artifact fails the binding.

Primary hypothesis: reviewer-kept consensus 5d equity/ETF SPY-relative views have hit rate above 50%
and Brier better than prequential climatology. Minimum: **60 trading days and 1500 non-abstained
asset-days**. At the first weekly close after both minima, make one confirmatory look: resample
whole calendar weeks 10,000 times, seed 20261006, preserving each week's asset cross-section.
The one-sided 95% lower hit-rate bound must exceed 0.50 AND the upper paired Brier-minus-climatology
bound must be negative. The first verdict is immutable, including a negative result. Dependence across
week boundaries remains a limitation. Secondary variants are exploratory and all are counted.

Interim looks, synthetic demos, pending labels and LLM backtests of pre-cutoff outcomes do not establish
an edge. Memorized historical outcomes invalidate such backtests. Probabilities remain uncalibrated
judgments; calibration bins are descriptive, not a calibration claim. Climatology uses only consensus
asset/horizon/population outcomes already observed before the decision, initially 0.50.

## Context, inference and evidence

The Yahoo daily provider/parser and Coinbase native six-field candle adapter are reused. Context
includes the last 21 completed closes, 1/5/20-session returns and sample standard deviation of 20 daily
raw returns. Gaps, stale bars and missing prices exclude an asset; a missing SPY or empty population
fails the run. Crypto context uses UTC daily sessions. OHLC is raw; corporate actions can distort
returns. No split-adjusted or licensed redistribution claim is made.

GDELT DOC 2.0 provides an asset digest and three macro/politics/trade digests within the request cap.
GDELT `seendate` is a discovery clock, **not attested publisher time**. Analysts must verify publication
time before using a catalyst. Archive summaries reuse the read-only FOMC/EDGAR API views at each
store's last attested instant, with source-specific horizon, identity, read state and age. Archived
items are marked stale; no common global horizon or current event coverage is invented.

Analysts receive identical independent context and the original TRADER_SKILL. The reviewer receives
both outputs and the original REVIEWER_SKILL. Serial execution bounds host memory. Strict schemas
require every asset/horizon, probabilities/directions to agree, sources before the frozen context
decision time, and reviewer coverage. DOWNGRADE can only pull toward 0.50; disagreement abstains.
Consensus uses the agreeing probabilities closest to 0.50. Invalid output is never repaired.

Claude uses `-p --model opus` (Sonnet for reviewer), explicit `--tools` AND `--allowedTools`
`WebSearch,WebFetch`, `dontAsk`, safe mode, empty setting sources, disabled hooks/skills,
empty strict MCP config, structured JSON and an empty temporary cwd. No permission bypass is used.
Direct WebFetch of Federal Reserve/SEC domains is denied. Model web-search content remains subject
to the operator's scope and the prompt's prohibition on official-source fetching.

Installed Codex 0.160.0 requires global `--search` **before** `exec`:
`codex --search exec --sandbox read-only --ephemeral --output-schema SCHEMA --json`.
It uses an empty cwd, ignores user config/rules, disables stable `shell_tool`, code-mode host,
apps/plugins/hooks, agents, browser/computer/image tools and skill discovery. The installed feature
listing verifies `shell_tool=false`; `unified_exec` is enforced by this CLI version, so disabling it
is not claimed. Every command-execution/file-change/MCP event still taints the whole run, rejects
its views and raises an alert. No real model invocation is used as a deployment test.

The configured timeout kills and reaps the whole subprocess group. Retries consume role and aggregate
daily budgets. Quota/limit responses stop the day as SKIPPED_QUOTA and raise an alert. Recording after
the target open fails. CLI versions, model aliases/reported versions, skill hashes, context hash,
grant hash, preregistration hash and sources are retained. If Claude does not report a concrete model
version, this absence is explicit; an alias is not misrepresented as a fixed version.

Both original analysts, both reviewed outputs and consensus become CONTRACTS_V1 PredictionRecords
in the existing research-observability store, with actual context/features bound through PredictionEvidence.
The full daily context is stored once as immutable evidence; each input record carries the actual
asset inputs and verified context-record/hash references for shared macro and archive inputs. This
keeps the existing store's byte limit intact instead of duplicating the full universe for every view.
Analyst records retain reviewer-rejected views for false-positive measurement. Labels append separately;
original predictions and inputs never change. Raw web bodies and full CLI transcripts are private runtime
files outside Git, excluded from trader API responses. Failed outputs stay private; public code includes
only native-shape synthetic fakes.

## Shadow portfolios, labels and scorecard

Each reviewed analyst and consensus has separate shadow portfolios, also separated by horizon.
Signed weights are proportional to p_outperform minus 0.50;
directional certainty must be at least 0.55 (for DOWN, 1 minus p_outperform). Each name, including the
separate SPY hedge leg, is capped at 10%; gross is at most 100%. The hedge neutralizes net equity
exposure, not estimated beta. Each 5d daily cohort receives one fifth of that portfolio's capital.
Shadow fills are recorded later against the actual target opening mark; they are never broker fills.
Zero proposed weight is recorded as NO_FILL. Unit-return scoring of rejected original forecasts is
a hypothetical diagnostic, separate from filled shadow-portfolio P&L.

Equity labels: open(D) to close(D) for 1d, open(D) to close(D+4) for 5d, using pinned XNYS sessions,
including holidays, early closes and DST. Store both raw and raw-minus-SPY returns. Crypto labels:
13:30Z(D) to 13:30Z(D+1/D+5), using exact Coinbase minute-candle opens, available after those candles
arrive. Missing endpoint prices remain PENDING; no interpolation. Late ingestion keeps its actual
availability. Zero returns count as not-UP and ties are reported.

Paper costs are **per side**: 5 bp plus half-spread equities and 10 bp crypto. In v1 the equity
half-spread is a preregistered fixed **2.5 bp modeled assumption**, not a measured quote; roundtrip
cost is 15 bp equities / 20 bp crypto. Both unhedged and SPY-hedged cohort P&L are reported. A
partially labeled cohort stays PENDING. This is modeled P&L, not an execution-quality claim.

Scorecards separate analyst, reviewer-kept/rejected/downgraded, reviewed analyst and consensus
populations, horizons and raw/SPY-relative targets. Hit rate, false-positive rate, Brier and causal
climatology Brier, calibration bins, probability/return Pearson IC, unit P&L after costs, abstention,
pending labels, days and sample counts are explicit. Equity raw-direction scores are secondary
diagnostics because the probability estimates outperformance. Baselines: always-up, completed
20-session momentum sign, identity-seeded random direction, SPY-relative zero (0.50, abstains).
Friday's label job writes the weekly report and, if eligible, the single primary look.

## Operator commands and private files

Set `TRADER_AUTH`, `TRADER_RUNTIME`, `FOMC_ARCHIVE` and `EDGAR_ARCHIVE` to operator-owned private paths
outside the repository. Set `PYTHON` to the verified environment's interpreter. Install the optional
`[ml,trader]` dependencies; the calendar is pinned to pandas_market_calendars 5.4.0.

```sh
"$PYTHON" -m scripts.trading_lab.trader_agent.cli run --authorization "$TRADER_AUTH" --runtime "$TRADER_RUNTIME" --dry-run
"$PYTHON" -m scripts.trading_lab.trader_agent.schedule --authorization "$TRADER_AUTH" --runtime "$TRADER_RUNTIME" --fomc-store "$FOMC_ARCHIVE" --edgar-store "$EDGAR_ARCHIVE" --install
systemctl --user list-timers 'hyprl-trader-*'
"$PYTHON" -m scripts.trading_lab.trader_agent.cli status --authorization "$TRADER_AUTH" --runtime "$TRADER_RUNTIME"
"$PYTHON" -m scripts.trading_lab.trader_agent.cli pause --authorization "$TRADER_AUTH" --runtime "$TRADER_RUNTIME"
"$PYTHON" -m scripts.trading_lab.trader_agent.cli resume --authorization "$TRADER_AUTH" --runtime "$TRADER_RUNTIME"
systemctl --user stop hyprl-trader-run.timer hyprl-trader-label.timer hyprl-trader-health.timer hyprl-trader-run.service hyprl-trader-label.service hyprl-trader-health.service
```

Do not manually start the real run service: the first real run belongs to the timer. Pause prevents
subsequent dispatch at orchestration boundaries; stopping the services terminates their process groups.
Resume removes the pause marker; it does not backfill or override one-run-per-day limits. To resume
stopped timers use `systemctl --user start hyprl-trader-run.timer hyprl-trader-label.timer hyprl-trader-health.timer`.

Only these user units are installed. Run: Mon–Fri 12:00Z. Label: Mon–Fri 21:30Z. Health: hourly at :05.
Persistent=false prevents catch-up inference. Holidays skip without model/source requests. After DST,
12:00Z still precedes the calendar's 14:30Z open. No system services or other applications are changed.

The production budget bank is the user-level `.local/share/hyprl/trader-agent-budget/dispatch.sqlite`;
the owner lock is adjacent. It persists across runtime paths and grant updates. Do not delete it to retry.
Private runtime layout: `evidence/research.sqlite` (immutable records),
`raw/`, `transcripts/`, `alerts.jsonl`, latest `alert.json`, `health.json`, `last-label.json`, `weekly/`,
and private service logs. Permissions are 0700 directories / 0600 files. Dry-run uses its own
`synthetic/` subtree and budgets; it advances the fake clock to realize labels and never constructs a
real transport or CLI runner. Use a new runtime for a fresh dry-run, since daily reservations persist.

Read-only API, opt in with `--trader-root "$TRADER_RUNTIME"` when starting the existing app API:

| Endpoint | Content |
|---|---|
| `/api/v1/trader/today?date=YYYY-MM-DD` | today's run states and recorded views |
| `/api/v1/trader/ledger?after=0&limit=100` | append-only PredictionRecord page; next_after continuation |
| `/api/v1/trader/scorecard` | real-only scores, pending cohorts and multiple-testing register |
| `/api/v1/trader/alerts?limit=100` | bounded private-runtime alert summaries |

POST/PUT/PATCH/DELETE remain refused by the existing API. Raw bodies, transcripts, credentials and
runtime paths are never returned. Labels/execution histories are also available through the existing
observability API when its `--research-root` selects this runtime's `evidence/` directory.

Validation: `python -m pytest tests/trader_agent -q`, followed by the AGENTS.md gate. The new suite is
added to sources-ci without weakening existing steps. Deployment reports distinguish unavailable
checks, excluded products and future real-run behavior from verified synthetic evidence.
