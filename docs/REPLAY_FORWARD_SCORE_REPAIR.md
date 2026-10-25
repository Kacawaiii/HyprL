# Forward score repair (execution revision 18)

The October 23 score was empty despite six received, prospectively published
decisions. Reproducing its original outcome cutoff yields exactly zero rows:
30 `INCOMPLETE_LABEL_COVERAGE` exclusions per model for October 20 and
216/205 `SEALED_SOURCE_DATA_MISSING` exclusions for Opus/GPT. This is a data
collection and request-boundary failure, not an unpublished decision.

The outcome index contains 64 objects, all received on October 20, and its
canonical digest is
`6a8c66ba8114698b5f8dc67813644c86aff9d547ffaac5b6252220b20e509fa2`.
That day's collector reported 30 labels. The October 21–23 collection jobs were
blocked by `BUDGET_EXHAUSTED`; the shared counter reached 3,000 each day.
Friday's scorer requested daily bars ending October 23 instead of the stored
October 20 query. Exact URL sealing correctly refused those absent objects;
even the original 30 labels became invisible.

Collector and scorer now construct outcomes with a fixed query end at each
horizon's maturity. Later runs reuse exactly those requests, without claiming
that an older response covers a longer interval. Missing requests remain
exclusions; altered objects, snapshots, decisions and publication proofs still
fail closed. Input polling leaves 1,000 requests available under the same shared
3,000 daily ceiling for the evening labels. No counter is reset, refunded or
split into a second allowance.

Each decision resolves its original registered freeze and canonical source
protocol, including costs and crypto entry offsets. Revisions 9–13 remain
immutable. October 20 is `INTRADAY`: common decision completion at
14:22:11.731018 UTC, strict entry after 14:22:41.731018 UTC. Its publication was
14:22:15.347763 UTC. October 22 and 23 retain their regular 09:05 New York
decision clocks and their original publications. All six stored decisions pass
their original binding and publication checks. October 21 retains both actual
failure codes: `FORECAST_COVERAGE_INVALID` and `MODEL_USAGE_LIMIT`.

Reports show `REGULAR_0905` and `INTRADAY` tables separately, including every
ab:1d/5d and cd:2d/10d panel, counts, Brier/log loss, trade hits and failed days.
Only regular sessions enter the regular primary endpoint. Trade hits use their
reported cost basis; they are not probability classification accuracy.
Immature horizons remain pending and are never imputed as model failures.

`forward-score --as-of <UTC timestamp>` reconstructs a past maturity cutoff
while retaining the real scoring timestamp and current verified archive proof.
`weekly --correction-week YYYY-Www --score-run <canonical score digest>` sends
an explicitly labelled correction to brouillons. It binds the original
delivered text hash and immutable score run, creates a separate at-most-once
receipt, and preserves the original weekly receipt. It still requires a fresh
score, confirmed Discord delivery and the requested reporting period.

Revision 18 changes implementation and reporting, not decision or statistical
rules. Its new freeze, input seal and explicit input-only reseal reject old
stores under the active binding; no decisions or transcripts are migrated.
The existing digest-bound replay continuation is unchanged. This repair makes
no model calls and places no orders.

Deployment retains the documented pinned-commit procedure: pause only active
research timers, drain running research jobs without killing model calls,
fast-forward the research deployment, replace exactly the eight research
service expected-commit bindings atomically after verification, reload user
units, run the next-session offline dry run, then restore the previously
active timers. Host worktree ownership restrictions still apply.
