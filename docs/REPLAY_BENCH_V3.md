# Research replay bench v3

The bench is **BLOCKED before test**. No admissible development results or test score are claimed.
The original dates cannot pass the inherited prespecified memory criterion: Sonnet and Opus both
have an operational floor of 2026-11-01, after the requested test ends on 2026-10-09.
This is a conservative eligibility decision, **not proof of either model's true training cutoff**.

The forced-answer questions may be guessable. Revision 2 adjusted their threshold after observing
results. Revision 3 preserves the original threshold instead; a guessability diagnosis cannot
retroactively make those dates valid. The old specifications and all six historical trials remain
in Git. Old stores are rejected, with no implicit migration. The legacy inventory contains only
counts, dates and digests; its 9 a/b and 21 c/d development days are exploratory, not a track record.
There are no recorded historical test trials. Original private data are left in place.

The four frozen variants are (a) Sonnet analyst plus Sonnet reviewer, (b) the same analyst without
review, (c) negative-news-first liquid small/mid candidates with 2–10 session holds, and (d) c with a
cited, directly linked primary 8-K. The small/mid classification is a liquidity proxy, not verified
historical market capitalization. An unrelated filing for the same company does not qualify d.
Anonymization is a paired dev audit every fourth session; Opus is a separate paired audit of a,
with at most 15 replay dates including the 10 inherited dates. Neither audit creates a fifth
strategy or permits additional test trials.

All decision groups use Sonnet except the sampled Opus analyst. The CLI uses subscription aliases
and rejects a reported identity different from the model actually probed. No tools, MCP servers,
slash commands, hooks, automatic memory or CLAUDE.md discovery are allowed. The model sees only
the supplied prompt and causal snapshot. Invocation isolation is tested with a fake CLI; no new
real CLI call has been made because the inherited budget already reports a usage limit.

## Running and resuming

Set `REPLAY_DATA_ROOT` and `REPLAY_AUTHORIZATION` to operator-owned private locations outside Git.
Use an environment containing the repository's Python dependencies. The CLI is
`python -m scripts.trading_lab.replay_bench.cli`:

- `audit` recomputes grades and coverage from the original Sonnet/Opus answer rows, checks the
  exact event sets, model identities and answer keys, and writes a private audit.
- `status` performs no network or model calls. It reports the memory floor, available dates,
  inherited budgets and missing input proof. BLOCKED exits with code 2.
- `run dev ab|cd|anon|opus` resumes only the matching configuration. Calls are serial, only
  00:30–05:30 UTC, with a **combined 150-call night cap**, including probes, retries, reviewers,
  Opus and inherited calls. A subscription-limit response stops immediately for that UTC day.
- `score dev` writes per-view labels, hashes and summaries only for complete v3 trials. Missing
  execution data or model stages leave a partial result with BLOCKED status. Missing prices are
  never converted to successful abstentions or silently removed from coverage.
- `freeze` creates the frozen test artifact only when there are eligible test dates and a valid
  input proof. The checked-in freeze is deliberately marked BLOCKED and cannot authorize calls.
- `run test ab|cd` additionally requires the committed matching freeze, complete and scored dev
  trials and audits, and the one-shot test-start marker. Further invocations resume that same
  run. A different configuration is refused.
- `night --expected-commit SHA` verifies the installed source SHA and eligibility, then executes
  dev, dev scoring, the single test, and test scoring in that order. It never edits Git.

Each request reserves its allowance before dispatch. An intent and every returned response are
written durably. If the process dies after dispatch without a response, the next run reports
`CALL_OUTCOME_UNKNOWN` and requires inspection of the private call journal; it does not draw a
second decision. Completed analyst responses survive interruption before the reviewer. Invalid
JSON/schema/citations allow one declared retry; transport errors, identity drift, timeouts and
quota errors do not trigger blind retries. A process lock excludes simultaneous batches.

## Data and scoring controls

News require both `created_at < T` and `updated_at < T`. Daily bars require session completion by T.
Snapshots and decision records are hashed and checked on resume and before scoring. BTC/ETH price
requests overlapping the protected interval are rejected at the transport boundary. HTTP is GET
only, grant-scoped, bounded, budgeted and refuses redirects. The authorization is pinned by digest
and re-read before dispatch. No live account, order endpoint, engine unit or existing tool is used.

A modern historical download is not proof of unrevised original prices or historical membership.
`input_proof.json` must bind the v3 spec, historical universe/identity and unrevised-price evidence,
and every prediction cache object by SHA-256. Required fields are `schema: replay-input-proof-v1`,
`spec_hash`, `unrevised_prices_proven: true`, `historical_universe_and_identity_proven: true`,
`evidence: [{file, sha256}]`, and `prediction_cache: {request_url_sha256: compressed_file_sha256}`.
Evidence files must be inside the private data root. Assertions need actual supporting evidence;
creating this JSON does not establish that evidence. Prediction reads are offline and refuse
unsealed cache objects. The inherited archive does not currently have this proof.

Equity 1d fills use the first SIP trade in the exact one-second windows at open +30s and close -5m;
minute VWAP is not substituted. Multiday returns include overnight gaps, both-side 0.1% fees and
tiered spread/slippage. R is a volatility unit, not a stop-loss guarantee. P&L is marked daily on
fixed initial capital, equal weights within each entry cohort, and 1/5 or 1/10 capital per cohort
for 5-session or variable multiday holds. The calendar clock includes zero days and uses sqrt(365).
Crypto needs exact entry/exit minute opens and a midnight mark; stale fills are refused. Crypto 5d
was not declared for scoring. Horizons crossing the period end are censored and counted; dev
labels never enter the test interval. The SPY-relative label charges the asset leg only; the
absolute SPY baseline pays its own costs. Baselines use the corresponding eligible rows and
horizons. Report counts, Brier, net/gross hit rates, expectancy R, Sharpe, peak-relative drawdown,
cluster-day Kish ESS and day-ICC ESS. Wilson intervals are descriptive and ignore clustering.

## Research timer and validation

`deploy/research/install_replay_timer.py` renders only `hyprl-replay-bench.service` and its user
timer from the committed checkout. Pass `--environment` and `--claude` paths explicitly. It
refuses existing units and dirty source. The timer fires at 00:30 UTC, does not catch up missed
runs, does not restart failures, and constrains CPU, memory and duration. Every CLI call also
checks the time window. The timer remains harmlessly BLOCKED with the present evidence.

Synthetic checks: `python -m pytest tests/replay_bench tests/trader_agent -q` plus the mandatory
repository source/API suite in AGENTS.md. `sources-ci` checks out and asserts the event SHA before
running these suites. No application/web changes are included. Actual subscription execution,
original-vintage evidence and a valid test score remain unproven.

To unblock, the operator must choose a new preregistered research period genuinely after the
conservative memory floor, with adequate point-in-time inputs and a valid authorization. A new
period requires a new spec/freeze and unused test; do not reduce the memory threshold, overwrite
the blocked freeze, migrate old stores, clear same-night quota stops, or score the old test as an
honest track record. The nightly timer cannot resolve this research-design blocker by waiting.
