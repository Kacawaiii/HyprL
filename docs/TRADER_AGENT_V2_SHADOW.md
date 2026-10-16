# Agent v2 shadow runtime

This is a separate **forward shadow** system, with no broker client, key loader, account selector or order endpoint.
It does not import the v1 trader. Its source package is `scripts.trading_lab.agent_v2` and its optional cockpit is
`/agent-v2`, backed by `GET /api/v1/agent-v2/summary`. Start the application API with a separate `--agent-v2-root` to read it.
Missing evidence returns 503; integrity failure returns 409. Neither path initializes a runtime.

## Activation status

**Not active.** Building this code is authorized by the operator decision identified by digest in the spec.
That decision is not an input acquisition or model-call grant with a v2 budget and expiry.
The replay authorization belongs to a different task and is not reusable here. The required replay report was unavailable
when implementation began. No external data/model request or broker order is needed for the synthetic tests.

Before activation, supply the replay report, a dedicated v2 authorization, a verified local input producer, successful
checks and exact-SHA CI. Promotion to a real paper account is a separate operator decision **after** the replay test result.
There is no automatic promotion code.

## Frozen contracts

- `docs/artifacts/trader_shadow_v2_spec_v1.json`: independent spec, canonical SHA-256 in `agent_v2/spec.py`.
- `docs/artifacts/trader_shadow_v2_preregistration_v1.json`: independent preregistration, bound to that spec.
- Both JSON schemas and no-tool prompts are hashed in the spec. Initialization also freezes the package implementation
  digest and Git SHA. Any later package change rejects the old runtime. There is no implicit store migration.
- A new, private directory is mandatory for initialization. Existing directories are refused. SQLite writes use atomic
  event batches and an exclusive writer lock. SQL triggers forbid updates/deletes, and reads verify the hash chain.
  This detects accidental modification; it is not externally anchored protection against a malicious filesystem owner.

Commands below use shell variables supplied by the operator. Runtime paths and evidence are private and never committed.

```sh
python -m scripts.trading_lab.agent_v2.cli init --runtime "$V2_RUNTIME"
python -m scripts.trading_lab.agent_v2.cli status --runtime "$V2_RUNTIME"
python -m scripts.trading_lab.agent_v2.cli decide --runtime "$V2_RUNTIME" \
  --authorization "$V2_GRANT" --inbox "$V2_RUNTIME/inbox/snapshot.json"
python -m scripts.trading_lab.agent_v2.cli watch --runtime "$V2_RUNTIME" \
  --authorization "$V2_GRANT" --inbox "$V2_RUNTIME/inbox/quotes.json"
python -m scripts.trading_lab.agent_v2.cli reconcile --runtime "$V2_RUNTIME"
```

Use a different fresh directory and `--synthetic` on **every** command for synthetic tests. A synthetic runner or quote
cannot enter a forward runtime. A failed or interrupted decision slot is not retried automatically. No retrospective
backfill is allowed. Model calls reserve budget durably before dispatch, and usage-limit responses stop the day's calls.

## Inputs and causality

The runtime deliberately consumes local inbox files. It has no network acquisition implementation and no authorization
to borrow v1's producer, accounts or keys. A live input producer is an outstanding deployment dependency, not supplied
by these commands. It must atomically publish validated snapshots and quotes, maintain completeness/provenance of the
calendar and instrument registry, and enforce its own signed endpoint/request budget and expiry. A timestamp on a local
file alone does not establish authenticity. Never point a forward runtime at replay data.

The strict normalized contracts are `SNAPSHOT` and `QUOTES` in `contracts.py`:

- Bars retain wire `t`, `c`, `h`, `l`, `v` fields and add an explicit `available_at`. A daily bar must become available
  after that NYSE session closes and before the snapshot. The last bar must be the previous completed session.
- Quotes have bid/ask, displayed sizes, quote time, received time and feed identity. Quotes older than 60 seconds,
  crossed quotes, future quotes and duplicate symbols fail. Mode/feed must agree. Forward feed is `alpaca_iex`.
- Snapshots are at most five minutes old, metadata at most four days, and calendar evidence at most 18 hours. The
  producer must report CPI/FOMC/NFP and holdings' earnings; an empty event list means a positively verified empty list,
  not an unavailable calendar. Missing calendar fields fail schema validation.
- Primary-source events include original text, digest, URL, publication and receipt times, symbol and event kind.
  8-K primary provenance requires an SEC Archives URL; earnings releases require a registered issuer domain. The
  extraction must cite an exact text substring and cannot add a symbol/source. URLs are not fetched by the model.
- Hunter input matches the observed `hunter-live` envelope (`asof`, `generated`, `candidates`) and candidate keys
  (`setup` S2/S3/S4, `two_layer`, `rev_accel_pts`, `signal_close`). Extra producer fields are removed by the adapter;
  they are never used to repair a stricter parser or to change the source's wire shape.

The LLM output is classifications only. Negative primary events block new entries and close existing longs. Positive
news entries are a separate unproven `beat_and_raise` hypothesis on liquid small/mid caps, held up to five NYSE sessions
(including overnight); stops can exit sooner. No negative-news stock short is generated. The system makes no claim
that the LLM's semantic classification or a quote-level simulation establishes a profitable strategy.

## Risk, execution and accounting

Momentum uses the book's OLS90 exponential slope x R-squared formula, MA filters, gap filter and ATR sizing on a 30%
sleeve. Hunter requires both technical and revenue-acceleration layers. Optional ETF trend is explicitly a long-only,
unlevered adaptation of the research code. P4's price-only proxy and P6 are not used as validated signals.

Risk is in code: stop-based quantities, conservative fee/slippage allowance, whole shares, displayed-size cap, 8%
notional per name, 100% gross, 4% portfolio heat and 1.5% per cluster including pending plans. Week -3% halves size;
month -6% quarters size and flags review; peak -8% halts new entries. Stops never widen. Unknown clusters fail.
The calendar blocks macro-release-day entries and earnings over the planned holding horizon.

Claude Sonnet receives immutable entry plans, their standing conditional exit instructions, and the hard-rule context. It returns exactly one ALLOW/NONE or coded VETO
for each plan. Missing, duplicate, invented or modified plans fail closed. All decisions and vetoes are journaled.
Protective exits (stop, time, negative-primary event and momentum-rank exits) execute those previously reviewed standing instructions in code independently of LLM uptime. Each exit records the supervisor plan ID; it is not a new discretionary order.

Approval does not fill an entry. A later quote at/after approval is required, with a five-minute plan expiry. Risk and
sizing run again at fill time; the quantity can only shrink. Buys pay ask plus 2bp slippage and 10bp fee. Exits receive
bid minus 2bp slippage and 10bp fee. Stops crossed in a gap fill at the observed bid, never the stop price. Entry cash,
exit cash, realized net PnL and initial-risk R are journaled. The nightly report verifies the chain, recomputes cash and
shows stale or absent valuation as BLOCKED. Scorecards report closed/winning counts, R and cluster-day counts by core/news.
Supervisor veto impact is **UNMEASURED**, with counts by code; there is no invented counterfactual performance.

Limits: minute sampling can miss intraminute stop touches; IEX is not the full market; displayed sizes do not guarantee
fills; overnight gaps can exceed planned heat. No performance claim or sample-size-adjusted inference is made here.

## Scheduling and guarded deployment

Render only the new `hyprl-agent-v2-shadow-*` user units into a new review directory:

```sh
python -m scripts.trading_lab.agent_v2.deploy --runtime "$V2_RUNTIME" \
  --authorization "$V2_GRANT" --output "$V2_UNIT_REVIEW"
```

All market schedules use `America/New_York`, with missed runs not backfilled (`Persistent=false`). Decision times are
open+10min and close-20min (including a 12:40 early-close slot); the service independently verifies the NYSE calendar and slot. Watch exits use real session
hours, including holidays and early closes. The renderer does not install, enable or modify any unit.

Activation requires the exact deployed SHA to pass local checks and CI, fast-forward integration, the replay report,
valid v2 model/input grants and a verified input producer. `deploy.check_activation` rejects missing evidence and the
UTC intervals 11:30–13:50, 15:00–16:30, 19:45–20:10. Deployment is currently blocked on the external prerequisites;
no installer bypass is provided. Existing services, v1 and the discretionary book remain outside this runtime.

```sh
python -m pytest tests/agent_v2 -q
```

The shared `sources-ci` workflow runs these regressions. The repository's full mandatory pytest set and web
`npm ci`, typecheck, lint and Vitest checks remain required before a push.

The optional `activate --activation-evidence "$V2_PROOF" --authorization "$V2_GRANT"` command journals a one-time
activation proof after checking the deployment window and prerequisites. Until this record exists, every forward
`decide`/`watch` service refuses to run. Rendering or initializing alone never activates the runtime.

The journal also refuses new writes above 512 MiB or below 256 MiB free disk. These are operational stop conditions;
records are never deleted to free room or silently migrated. A deployment must plan retention and an explicit new
runtime/spec transition before reaching that limit.

A dedicated model/runtime grant has schema `agent-v2-shadow-authorization-v1`, `operator_signed: true`,
`mode: SHADOW_FORWARD`, this preregistration hash and operator-decision hash, `granted_at`, `not_after`,
`model: sonnet`, and an integer `claude_calls_per_day` from 1 to 8. Only the operator may issue it.
The producer needs its own exact endpoint/path, symbol, request-budget and expiry authorization; the runtime
never interprets a model grant as permission to acquire data.

The reviewed deployment proof has schema `agent-v2-deploy-proof-v1`, `sha` equal to the runtime's code SHA,
`preregistration_hash`, `ci_conclusion: success`, `local_checks: PASS`, `integrated_fast_forward: true`,
`replay_report_sha256`, `input_grant_sha256`, and `forward_inputs_ready: true`. This is an operator/deployer
attestation; the checker validates its binding and required assertions, not a GitHub signature or a producer's
network behavior. Verify the actual CI run and grant contents before writing it.
