# Research loop v1 — offline method and operator runbook

This loop changes research rules, tools and evidence, never model weights. It has
no broker, provider, model-call, notifier or scheduler dependency. It cannot place
an order or update a live strategy. The initial iteration is exploratory and may
finish BLOCKED. Missing evidence must not become a zero return or a simulated fill.

## Frozen contract

The authoritative new contract is `artifacts/research_loop_spec_v1.json`; eight
single-variant hypotheses are in `artifacts/research_loop_hypotheses_v1.json`.
`scripts/trading_lab/research_loop/spec.py` checks their canonical SHA-256 hashes.
Canonical JSON uses sorted keys, compact separators, UTF-8 and rejects NaN.
Preregistration commit `ad03262` precedes numeric outcome analysis. Its actual
commit clock corrects the initial rounded display clock in the register; thresholds
were unchanged. A rule change requires a new specification revision and new
bindings. Old trial databases are rejected, with no implicit migration. A ledger
also binds the implementation hash, so an analysis-code change cannot silently
reuse a previous ledger.
For a corrected implementation, `--prior-trial-ledger` explicitly carries only
hash-verified reservation counts into a **new** ledger. It checks the original
spec/registry bindings, keeps the old database read-only, and reuses no old result.
All repeated attempts still count. This count-only reference is not a migration.
Validation reservations also count. References carry spent validation rule hashes
transitively so a new implementation cannot reopen a once-only validation. A legacy
count reference without that history blocks new validation until the operator
supplies a complete audited history; it does not imply an unused validation.

The first three rules are fixed filters: the next-24-hour earnings blackout,
fresh primary 8-K/10-Q, and a USD 300m–10bn universe with a planned 2–10 day hold.
They preserve original entry/exit times, sizing and original outcomes. Excluded
entries earn cash zero. No rule gets another horizon label after seeing its result.
The remaining five require separate implementation and new counted reservations.

## Post-mortem contract

Every normalized record has an immutable decision/source identity, strategy,
instrument, decision/entry/exit clocks, original action, weight, and original
point-in-time facts. A fact is `{value, available_at, observed_at}`; BOTH clocks
must be at or before the decision. `observed_at` is the source observation vintage,
not this weekly analysis clock. Date-only news is not assigned a fabricated hour.
The outcome carries its own availability time and an explicit cost basis.

The post-mortem preserves the original forecast probability, intended weight,
review verdict, planned horizon and snapshot/feature digests. It separates:

- Broker CLOSED lots, deduplicated by lot id, with a matching FILLED entry.
- Retrospective shadow labels, including zero-weight proposals and abstentions.
- Replay/forward forecasts, including each panel and horizon, never called fills.
- Intent/acknowledgement journals, which cannot establish fills or profit.
- Open, missing or invalid records, which remain pending or quarantined.

Objective error flags include a documented pre-T sizing-limit violation and a
recorded scope violation. A positive abstention opportunity is a **counterfactual**,
not realized profit and not proof the original abstention was wrong. Other causes
stay `unidentified_requires_evidence`. In particular a loss does not prove late
news, priced-in news, an earnings gap or a sector cause. These require primary
document clocks, a pre-T expectations/price baseline, gap bars, or matched sector
returns respectively. Those diagnostic inputs must be recorded separately with
their observation/availability clocks; an after-T observation never becomes a
decision input. Existing exports do not establish those causes.

The v2 adapter verifies the original event hash chain and joins approved plans,
entry fills and exit fills. Its P&L is net shadow accounting normalized by the
original bound initial capital, never a broker fill. Unfilled/vetoed opportunities
are retained as event counts, with no fabricated opportunity return.
Book, momentum and hunter require an operator-provided offline fill export with
strategy attribution, fees, endpoints and point-in-time universes. The book remains
separate from every AI strategy/baseline. This loop reads its journal; it never
executes or modifies the book CLI.

## Development, contaminated validation and forward testing

Development is 2025-01-01 through 2026-06-30, with both entry AND exit in that
window. Calendar boundaries and session clusters use UTC after parsing aware
timestamps; a local date cannot bypass a development or protected-period boundary.
July–September 2026 are explicitly touched: July–August informed existing
replay work and lesson generation; September was already used by an existing
replay. September test answers are not opened by this iteration. A file called
`dev` in an older replay does not change these dates.

BTC/ETH 2026-09-01 through 2026-11-30 are excluded before any outcome calculation,
including intervals crossing into the holdout. The first v1 adapter conservatively
excludes all BTC/ETH records in SQL. The book skips their journal lines before
inspecting journal contents; account equity and general notes are not analyzed.
No raw frozen store, closed pilot, archive or other worktree is modified.

A trial reservation is durably appended **before** evaluating a rule. Failures,
crashes, missing data, parameter changes and repeated attempts are never refunded.
Each reservation records the data, spec, registry and implementation bindings.
The database has update/delete rejection triggers and a verified hash chain.
The family floor is eight (all proposed variants), increasing with every reserved
attempt across runs. Earlier raw p-values must be re-adjusted against the current
family count when reporting later iterations. Past unrelated historical searches
have unknown trial counts; no statistic here corrects that unknown history.
The current-run report restates all three rules against the final cumulative
count, including attempts reserved later in the same run, and appends that
restatement to the ledger without rewriting the original result events.

Missing pre-T features or net contributions block the entire eligible development
trial rather than silently dropping losers. Outside-window records are counted
separately and never tested. Duplicate decision identities are invalid input.
An empty development export reserves a trial and reports BLOCKED, never TESTED.

For eligible records, compute candidate minus original **net portfolio-return
contribution**, retaining the original number of decisions in the denominator.
Group deltas by entry session and flip one sign per session, 10,000 seeded draws.
The one-sided tail is `(1 + draws >= observed) / 10001`. Report raw p, the
Bonferroni deflated statistic `min(1, family_size * p)`, all variants and exclusions.
This is a multiplicity-adjusted randomization statistic, **not** a claimed
Deflated Sharpe Ratio. Its null requires independent session blocks and symmetric
block deltas. Across-session autocorrelation or overlapping positions can violate
that assumption; such results need a revised, preregistered blocked method and
must not be promoted on this test alone.

The fixed stress subtracts an additional 40bp roundtrip times original absolute
weight on candidate retained entries. Both ordinary and stressed mean deltas must
be positive, with adjusted p <= .05 and >=40 mature session clusters. A reservation
for the touched validation period requires explicit acknowledgement and is
single-use per rule hash, including failures. It remains exploratory, not clean
out-of-sample evidence. No validation outcome was evaluated in iteration 1.

Only a new, precommitted forward SHADOW A/B may qualify for operator review:
same opportunity/clock/cost convention, current vs candidate, >=40 unique mature
decisions in >=40 session clusters, positive ordinary and stressed effect,
adjusted p <= .05, and no risk regression. All original opportunities count,
including abstentions, missingness and failed model responses. `promotion_gate`
audits this supplied evidence; it cannot promote, deploy, schedule or trade.
Forty is an admission floor, not evidence that a small edge has sufficient power.
Gate inputs require a validation completion clock and, for each pair, the symbol,
rule registration, decision, entry, exit and outcome-availability clocks. Registration
must follow completed validation, decisions must follow September 2026, session
ids must match UTC entry dates, and outcomes must already be available. A replay
label, a future outcome or a protected instrument cannot satisfy this gate.

## Luck meter v1

All numeric luck outputs are diagnostic. A p-value is the frequency of outcomes
at least this favorable **assuming a specified null**. It is not the probability
that luck caused the strategy, nor the fraction of P&L explained by luck. The
requested wording “XX % de chances que ce soit du hasard” would assert a posterior
that these data cannot provide. Use “p sous le hasard = XX %, diagnostic” or
“hasard non chiffrable : données manquantes”. Never replace missing results by 100%.

### Matched random entry

Preserve every actual closed entry slot, its entry/exit timestamps, holding
period, side, exposure and cost convention. For each slot require the FULL
historically eligible universe and all candidate returns at exactly those
endpoints, with membership available by the original decision. Include delistings,
halts and unavailable fills under a preregistered treatment; do not choose only
surviving symbols or only symbols the model discussed. Different strategies keep
their own universes and the book never becomes an AI benchmark.

`stats.random_entry` samples a symbol uniformly for every slot, 10,000 draws,
seed 1016, and sums supplied net portfolio-return contributions. Identical
repeated exposures share a draw. It reports the real result's empirical percentile
`100 * count(random <= real) / 10000` and the upper-tail p-value with a plus-one
correction. This is conditional symbol randomization; it does not preserve every
sector/portfolio dependence. A stronger dependent null requires a new version.
The caller must certify the candidate panel's endpoints, side, weights and costs;
the function cannot infer missing data from price series.

Incomplete universe/return panels produce BLOCKED with **zero draws**. No panel
of model-selected names, book intent prices or synthetic fixtures substitutes for
the missing real universe. Iteration 1 cannot run these 10,000 real-data draws.

### Hit rate and Brier under a null

For independent binary hits, use the exact one-sided binomial tail
`sum[k=h..n] C(n,k) p0^k (1-p0)^(n-k)`. The coin null uses p0=.5. Report sample size,
success definition, fees, dependence and the correction family. Broker hit rates
are gross where fees are absent; positive-before-fees is not a net win.

For Brier, keep the original forecast vector fixed, generate 10,000 Bernoulli(p0)
label vectors, and compare mean squared error to the observed score using the
plus-one lower-tail correction. Use p0=.5, plus the already frozen historical
base rate for the forward AB 1d panel when available. Do not fit p0 on the scored
outcomes. The historical base rate is a descriptive benchmark from a previously
touched replay, not a new clean development estimate.

All forecast panels/horizons are retained; regular and intraday results remain
separate. Forecast direction hits include abstentions and differ from taken-trade
hit rates. Correlated symbols, repeated horizons and multi-day overlapping
decisions break the iid null. Consequently its p-values are labelled **diagnostic**
and cannot authorize promotion, even if small. Cohort multiplicity is reported;
the raw Brier simulations must not be presented as confirmatory significance.

### How many more decisions?

Fix the planning effect before results: hit rate .50 to .55, two-sided 95%
confidence, 80% power, Bonferroni family m. The normal-approximation target is

`ceil(((z[1-.05/(2m)] sqrt(.5*.5) + z[.8] sqrt(.55*.45)) / .05)^2)`.

For m=1 this is 783 independent decisions. Report the assumed edge, family and
remaining target after distinct session clusters, not after raw correlated rows.
This planning approximation is not an exact guarantee, a Brier sample-size
calculation, or evidence of a .55 observed edge. Correlation can increase the true
requirement. With no eligible records, credit zero observed decisions. A Brier
power target needs a separately preregistered effect/distribution; it is not
invented here. Weekly lines explicitly say the N concerns the hit-rate plan.

## Saturday operation and privacy

First review the code and supply **private**, read-only input paths via environment
variables. Run the repository interpreter, never a trading CLI:

```sh
python -m scripts.trading_lab.research_loop.cli \
  --v1 "$RL_V1_DB" --paper "$RL_PAPER_DB" --v2 "$RL_V2_DB" \
  --book "$RL_BOOK_JOURNAL" --replay "$RL_JUL_AUG_RESULTS" \
  --forward "$RL_FORWARD_RESULTS" --output "$RL_PRIVATE_OUTPUT"
```

Optionally supply `--development "$RL_CAUSAL_DEV_EXPORT"`: a normalized JSON list
using the record schema in `core.audit`, with observed/available timestamps and
full net costs. The input cannot contain a horizon outside the development dates.
The runner creates a new private run directory and an append-only trial ledger.
It writes per-record post-mortems, all panels, trial outcomes, a machine summary
and French Discord text. Existing source files are never rewritten. CLI output
prints counts and its private run directory only.

The operator may arrange a **research-only** Saturday 21:00 UTC job invoking that
same command. No timer, service, live tool, trader runtime or book is installed or
modified by this branch. Book fill export and measured unfilled/vetoed opportunity
outcomes are prerequisites to claiming complete weekly coverage. Run manually for iteration 1; label its
actual source cutoff and off-cycle date, rather than pretending it ran Saturday.

Review the generated French summary, then use the operator-authorized notifier
for `brouillons`. Keep a receipt and summary digest in the private task report.
Avoid an unattended generic notifier dependency inside the engine: notification
is a separate explicit operator action, with no automatic retry after uncertain
delivery. No tokens, contact details, private paths, raw bodies, stores, logs or
fixtures belong in Git. Public iteration evidence is limited to counts/digests.

The report must state DONE or BLOCKED, French summary first, changes/commits,
commands/results, all missing evidence, and the operator's exact next step.
