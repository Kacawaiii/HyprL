# Agent v2: prospective quality experiment (revision 3)

**Not activated by this change. No forward cycle or profitability is claimed.** This package is
`scripts.trading_lab.agent_v2_quality`. Revision one remains intact, including its store, schemas,
prompts, API and implementation binding. The new specification and preregistration are
`docs/artifacts/trader_shadow_v2_spec_v3.json` and `trader_shadow_v2_preregistration_v3.json`, with
canonical hashes bound in `agent_v2_quality/spec.py`. Opening a revision-one or revision-two store fails; no migration
or rebinding is supplied. The pinned operator grant waives the old replay prerequisite.

## Decision and execution

One cycle starts at 12:45 **UTC**, on NYSE session days only, with a three-minute start tolerance.
This is a premarket planning time, not an execution time. UTC remains fixed through DST changes.
Code selects at most three momentum and hunter S2/S3/S4 revenue-acceleration candidates using the existing pure
rules, in their deterministic proposal order. ETF and model-discovered primary-news entries are disabled. PEAD and leader pullback have no
implementation or measured-edge evidence in this revision and cannot silently become candidates.
Each enabled rule needs an operator-supplied positive after-cost expectancy, decision count and
measurement digest. This is an attestation, not an independently verified claim of statistical edge.
An edge with explicit boolean `enabled: false` contributes no proposals and needs no measured-edge
fields. Disable filtering precedes symbol deduplication, so disabled momentum cannot mask hunter.
Omitting `enabled` retains the existing evidence requirement; non-boolean flags are rejected.

Daily bars retain Alpaca's wire shape and original timestamps, and are stamped with receipt time
after the full response. The previous completed session must be present; pagination must complete.
Previous closes support **provisional sizing only**. Quotes are never invented for this step.
The same Python stop, heat, cluster, gross, cash, governor and whole-share constraints are applied at
the first fresh regular-session quote strictly later than approval. Size may only decrease. Plans
expire five minutes after the actual NYSE open, including across DST. No missed plan is backfilled.

For each candidate, the producer retrieves the most recent available 8-K, 10-Q or 10-K primary
document accepted within 120 days. It reads normalized full text, preserves text/raw digests and
source URL, and refuses empty or oversized documents rather than truncating them. It does not
retrieve issuer-hosted releases or guarantee that a separately linked earnings exhibit was read.
The SEC submissions CIK remains the actual zero-padded digit string. No research store is reused.
Missing primary evidence blocks the AI arm and is reported, while the independent baseline remains
observable. Such failures do not count as evidence for layer efficacy.

Opus extracts facts with exact text citations, then Opus as analyst and GPT as independent
contradictor return KEEP / SIZE_DOWN / VETO with cited sources and falsifiable reasons. GPT receives
the same primary evidence and extracted facts, **without the analyst judgment**. No response schema
contains a price, size, stop or proposed ticker. Schema, candidate identity and citation checks fail
closed. Calls have no tools; reservations precede dispatch; quota halts the day's calls; no retries.
Normal use is two Opus and one GPT calls, below the grant's three-per-family daily caps. GPT's served
model identity is not attested by its CLI; requested identity is recorded. Actual CLI interoperability
and forward provenance remain deployment checks.

Acquisition and inference hold the journal writer lock only for short reservations/results. The
watcher never waits on a model subprocess holding that lock. The baseline evaluation book handles
stops and five-session time exits without a new model call. Minute IEX quotes can miss stop touches;
gaps fill at the observed bid with costs, and displayed liquidity never guarantees execution.

## Paired evaluation and removal

Every filled code candidate is retained through its later observed stop or five-session exit,
including all AI vetoes. Each arm shares the baseline candidate, entry/exit quotes, stop, horizon and
costs (ask +2bp and 10bp entry fee; bid -2bp and 10bp exit fee). Unused size stays in cash.

| Arm | Frozen behavior |
|---|---|
| baseline | Python plan, no AI filter |
| extraction | Zero size on a cited guidance cut, dilution, accounting or legal flag |
| veto | Zero size if either independent judgment is VETO |
| sizing | Half size if either judgment is SIZE_DOWN; isolates size from veto |
| combined | Applies only still-active extraction, veto and sizing factors |

Disagreement produces smaller size or zero. The experiment measures paired allocation PnL, not the
capacity or reinvestment behavior of separately managed portfolios. After at least 40 **completed
valid candidate outcomes**, each layer with cumulative net paired PnL delta <=0 is permanently
removed from subsequent combined decisions. This threshold is a governance rule, not significance.
Overlapping cluster-days are reported. Removed extraction is no longer called; judgments stop when
both veto and sizing have been removed; all model calls stop once all three layers are removed.
Diagnostic history is preserved; failed AI cycles, unfilled and expired plans cannot inflate samples.

Exact-substring validation establishes citation integrity, not factual/semantic correctness. The
extraction scorecard reports financial impact separately from **UNMEASURED** semantic accuracy.
`audit-fact` records an explicit operator adjudication of a fact hash, confirmed flag and reason.
No automatic self-grading or invented red-flag accuracy is supplied. An adjudicated subset remains
labelled as a subset. Each weekly report contains counts, baseline and layer net PnL, deltas and
disabled states. Its only delivery is the task-authorized notifier channel `brouillons`.

## Required private inputs

The private configuration is supplied by the operator, never committed:

```json
{
  "authorization": "<new v2 operator grant>",
  "credentials": "<existing private Alpaca credentials>",
  "sec_authorization": "<approved SEC User-Agent grant>",
  "registry": "<verified instrument registry JSON>",
  "calendar": "<verified macro and earnings calendar JSON>",
  "hunter": "<current hunter-live JSON>",
  "radar": "<current news-radar-v3 JSON>"
}
```

- Registry: `schema: agent-v2-quality-registry-v2`, `received_at`, `instruments`, `edges`.
  Each instrument has `symbol`, `asset` (equity/etf), known risk `cluster`, `market_cap`,
  `primary_domains`, `momentum_member`, and equity `cik` (ten-digit string). Each enabled playbook's
  edge has `decisions`, `net_expectancy_after_costs`, and `evidence_sha256`.
- Calendar: `schema: agent-v2-quality-calendar-v2`, `received_at`, nonempty `source_hashes`,
  `covered_macro: [CPI, FOMC, NFP]`, `covered_symbols`, `coverage_start`, `coverage_end`, and `events`.
  Coverage must include all registry symbols through the actual five-session exit date. Events use the revision-one
  normalized `{kind, at, symbol}` shape. Empty events require positively verified coverage.
  The code does not fetch an unauthorized calendar endpoint or manufacture future earnings dates.
- Hunter: existing `{asof, generated, candidates}` envelope, including `two_layer`, `setup`,
  `rev_accel_pts`, and `signal_close`. The adapter does not change its source shape.
  The native `asof` civil date (`YYYYMMDD`) is converted only in the shared planner's view;
  the source and snapshot retain the native value and hashes.
- Radar: live `news-radar-v3` READY/PARTIAL with causal cutoff and story receipt/publication times.
  Only related stock headlines/URLs/timestamps are passed to models. Radar price panels, crypto
  events and upstream model judgments are excluded. Incomplete upstream status stays visible.
  Under the operator's uniform 2026-10-26 decision, missing publication/receipt times and
  noncausal stories are excluded and counted; they never block the whole cycle. Admission requires
  `published_at <= received_at < cutoff <= input_time`, including strict exclusion at equality
  with the cutoff. Source timestamps and digests are preserved. The same filter serves activation,
  registry projection and the cycle producer. Projection counts survive in the producer bundle,
  input review, activation proof/result and Discord success line. A missing, stale or malformed
  radar still blocks; a valid radar with no admissible headlines remains usable.

Metadata age is four days; calendar/radar age is 18 hours. The producer enforces GET-only exact
Alpaca bars/quotes paths, SEC submissions and SEC Archives document paths, refuses redirects, reserves
budget durably, and journals receipt times and body digests. Local conservative caps are 1,000 Alpaca
and 60 SEC requests/day, with SEC spacing at least one second, below the grant's 5 requests/second.
403/429 halts that source for the day. Credentials and approved SEC User-Agent never enter the journal.
The old User-Agent grant expires before the v2 grant; acquisition refuses it once expired.

## Operator deployment

The common host task rules reserve live deployment to the operator session. These commands are a
reviewable handoff, not evidence they ran. Use a clean, committed checkout at the exact successful
CI SHA; runtime initialization freezes that SHA and both packages' implementation digest. Choose a
**fresh** directory beneath the isolated v2 area, preserving the existing revision-one store.

```sh
V2_MODULE=scripts.trading_lab.agent_v2_quality.cli
"$V2_PYTHON" -m "$V2_MODULE" init --runtime "$V2_QUALITY_RUNTIME"
"$V2_PYTHON" -m "$V2_MODULE" produce --runtime "$V2_QUALITY_RUNTIME" --config "$V2_CONFIG"
"$V2_PYTHON" -m "$V2_MODULE" render --runtime "$V2_QUALITY_RUNTIME" \
  --config "$V2_CONFIG" --output "$V2_UNIT_REVIEW"
systemd-analyze --user verify "$V2_UNIT_REVIEW"/*.service "$V2_UNIT_REVIEW"/*.timer
```

Review an activation proof with schema `agent-v2-quality-deploy-proof-v2`, `sha`, `spec_hash`,
`preregistration_hash`, exact `authorization_sha256`, `ci_conclusion: success`, `ci_url`,
`local_checks: PASS`, `single_owner_reviewed: true`, `replay_waived_by_grant: true`, and
`producer_snapshot_hash` from a verified receipt no older than 24 hours. Assertions are operator
attestations, not cryptographic GitHub/producer authentication. Never manufacture PASS or sign the
operator-review field on their behalf. Activate outside the existing protected deployment intervals.

```sh
"$V2_PYTHON" -m "$V2_MODULE" activate --runtime "$V2_QUALITY_RUNTIME" \
  --config "$V2_CONFIG" --evidence "$V2_ACTIVATION_PROOF"
```

Only the operator then installs/enables the eight reviewed **new user** unit files. The daily timer
is 12:45 UTC. The separate 22:45 UTC companion runs tomorrow's native-shape synthetic cycle in a
fresh synthetic journal without reading/copying production data or calling any model/provider. It
does not edit the v1 nightly drill script or any existing timer. Its result file is under the new
runtime's `drills/`. On non-session days it explicitly skips. Existing v1 units, discretionary book,
system services and broker accounts are outside this package.

`STATUS: DONE` is justified only after a real scheduled producer/decision cycle and its journal
`cycle_complete` have been inspected. Activation and rendering alone are insufficient. Real fills,
40 decisions, extraction adjudication, source reliability, model interoperability and weekly Discord
delivery each require their own evidence; a synthetic drill proves none of those forward outcomes.

```sh
python -m pytest tests/agent_v2_quality tests/agent_v2 -q
```

Revision three freezes the operator's 2026-10-23 input admission decision before
forward outcomes. It uses `trader_shadow_v2_spec_v3.json` and
`trader_shadow_v2_preregistration_v3.json`; both earlier revisions remain intact.
A fresh runtime is mandatory. Revision-two stores remain rejected.

The offline `agent_v2_quality.registry` builder selects the first 60 names from
`agent_v2.core.ranking()` on the entire PIT momentum membership, ordered by score
descending then symbol ascending. Append current hunter symbols in symbol order,
deduplicate with momentum priority, retain the first 199 equities, and add SPY.
Selection precedes exclusions; excluded names are never backfilled. Thus the
registry always fits the unchanged 200-instrument producer limit. Exclude equities
without a granted-source earnings date, an exact ten-digit submissions CIK, a
mapped SIC, or a verified positive market cap. The existing strict bar-history
boundary also excludes unavailable, stale or invalid histories; preserve those
raw observations and count the exclusion instead of replacing zero volume. Record all reasons and disjoint
primary-reason counts. Project the calendar and native hunter candidates onto the
admitted registry; preserve original source receipts, dates and hashes.

`registry.py` commits a disjoint inclusive SIC-range risk-sector table. This is a
fixed aggregation of SIC industries for risk limits, not a claim of GICS sector
membership. It includes specific software, pharmaceuticals, medical equipment,
energy, utility and REIT ranges; missing, malformed and unmapped codes cannot
supply a cluster. A submissions CIK mismatch cannot supply one either.

The new `agent-v2-quality-snapshot-v3` contract represents SPY's corporate market
cap as null (not applicable). Equities still require a positive market cap; a null
cap on any other instrument is rejected. The frozen revision-one snapshot schema
and validator are unchanged. The new snapshot schema is canonically hashed in the
revision-three spec alongside the extraction and judgment schemas.

A valid producer receipt is a point-in-time acquisition check. It cannot prove
Monday readiness from Friday receipts: the calendar/radar freshness limit is
18 hours, metadata/hunter is four days, and the horizon is five later NYSE sessions.
For the Monday 2026-10-26 12:45 UTC cycle, coverage must reach November 2 at 21:00
UTC. Refresh and rebuild prospectively, then run `produce` in the final fresh
runtime and bind its actual receipt hash to the operator's proof. No future clock,
relabelled source receipt, inferred earnings date or copied journal is admissible.
The measured-edge requirement and operator-only activation remain in force.

## Explicit one-shot activation exception

The operator's 2026-10-25 exception permits installation and activation of **new quality-shadow
units only**. It does not cover revision-one/v1 units, broker paths, research, replay or any account.
After Monday's radar rejection, the operator re-armed the opt-in
`scripts.trading_lab.agent_v2_quality.monday` entry point for the first 2026-10-27 cycle.
The module and bootstrap unit names retain `monday` for continuity; their single allowed date is
Tuesday. Its private settings pin the reviewed SHA, grants, metadata seed, historical hunter
evidence, local check receipts and staging result. Existing units are refused rather than replaced.
No specification, preregistration, old runtime or historical evidence is migrated.

The separate `hyprl-agent-v2-quality-monday-activate.timer` has one calendar date, 2026-10-27
10:15 UTC, and `Persistent=false`. Its service has a 65-minute limit; the script checks a stricter
11:25 UTC cutoff between stages. Failure halts the pass, preserves reservations and reports the
stage and exact error through the authorized `hyprl-notify alertes` interface. Any partial timer
installation is disabled and removed. All new services retain resource limits and an explicit PATH
for the installed model CLIs; no CLI inference is dispatched during activation.

The one-shot performs these operations in order:

1. Require the pinned clean checkout, passing local checks on the exact SHA, and completed staging.
   Fetch the exact SHA's public GitHub Actions API response with GET only, preserve bytes/digest and
   receipt, and require the latest `sources-ci` push run for this repository to have completed with
   `success`. Other SHAs and earlier successful attempts cannot qualify a failed/latest run.
2. Read the already scheduled Monday-evening radar output under the shadow grant. Check its true
   cutoff both now and at Tuesday 12:45 UTC; retain PARTIAL/READY and original story timestamps.
   This refreshes the input projection without invoking or changing the radar job.
3. Refresh the granted Nasdaq/Fed/BLS calendar once, without redirects or retries. Reserve each
   request durably before dispatch; space requests by at least 1.05 seconds. The fixed horizon
   requires eight Nasdaq dates and five macro pages (13 GETs), including both monthly BLS pages,
   through November 3 at 21:00 UTC. An unrecognized response or missing date fails closed.
4. Rebuild the revision-three scope from the frozen membership in the pinned metadata seed,
   previous-session raw IEX histories and current native hunter. Preserve the top-60/no-backfill
   selection policy even though momentum proposals are disabled. Obtain submissions only for
   source-dated names, preserve the shared SEC budget/spacing, and exclude unavailable metadata or
   invalid histories. Retain hunter's historical 253-trade/+0.418 R evidence and its horizon caveat.
5. Produce a fresh snapshot, run the pure hunter-only planner, render and verify all eight units,
   write the real activation proof and hashes, activate a fresh shadow journal, then enable/check
   the four quality timers. The actual 12:45 cycle must produce another fresh snapshot; activation
   never retimestamps an earlier snapshot to claim future freshness.

Calendar exclusions can legitimately produce zero plans, particularly because source-dated
registry admission and five-session earnings exclusions overlap. A successful activation or empty
cycle proves neither model interoperability nor a positive forward edge. Future calendar/radar
refreshes after this first Tuesday are not installed by this one-shot; later stale inputs fail closed.

The same orchestration can be exercised completely offline, including real systemd unit syntax
verification, on a **fresh** staging directory:

```sh
python -m scripts.trading_lab.agent_v2_quality.monday --staging <fresh-staging-directory>
```

Only the time, provider/grant boundary, systemctl calls and notifier are substituted. Synthetic
receipts, notification and installed-unit directory are labelled and isolated; no real model,
provider, systemd timer or Discord call occurs. Production rejects staging check receipts. Tests
also exercise deadline, stale radar, failed source response, pinned-file mutation, exact-SHA CI
rejection and rollback after a partial enable failure.
