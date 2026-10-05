# Reproducible offline demonstration

Activate the repository's Python environment and run from the worktree. Each output root must be
new, beneath ignored `var/`; previous evidence, model references and archives remain intact.

```bash
bash scripts/demo_chain.sh --root var/trading_lab/demo-001
# Operator supplies archive locations as private environment values; no network request occurs.
bash scripts/demo_chain.sh --root var/trading_lab/demo-archives-001 \
  --fomc-store "$HYPRL_FOMC_ARCHIVE" --edgar-store "$HYPRL_EDGAR_ARCHIVE"
```

The command prints a single `hyprl-demo-chain-v1` JSON report and writes `demo-evidence.json`
privately in the new runtime. The report contains digests, identities, counts and explicit labels;
it does not print archive paths, bodies, feature vectors or logs. Failure exits 2 with BLOCKED.

The REAL_ARCHIVED_EVIDENCE part reads the operator's FOMC rev25 closure and EDGAR fixture-trial
closure using the existing read-only stores. It builds InformationSnapshots at each source's own
attested cutoff, verifies repeated fingerprints and source replay under that source's T/H, and
checks the full archive file fingerprint before and after. Horizons stay separate. Prices are
not opened; protected October crypto decisions remain PROTECTED and missing prices remain missing.
Without archive arguments the real part says NOT_CONFIGURED. The archive inputs must be genuine
operator-supplied closures; synthetic source directories are not real evidence.

The SYNTHETIC DEMONSTRATION part then runs:

1. 240 causal InformationSnapshots of 120 synthetic hourly bars for BTC-USD and ETH-USD, with
   synthetic at-close price evidence and explicitly absent official-source coverage.
2. One versioned dataset: 182 included decisions with causal features and explicit exclusions.
3. The shipped synthetic Ridge and local momentum adapters, each run twice in isolated workers
   under the existing resource budgets; both result hashes must reproduce bit for bit.
4. 80 immutable prediction records per model, their exact snapshot/feature evidence, validation
   references, future-label arrivals, shadow decisions and execution observations.
5. Monitoring on the test population and a labelled injected scenario demonstrating missing
   data, technical degradation, drift and performance loss. Negative model outcomes are retained.

Monitoring arrivals use a fixed demonstration clock (`2026-08-01T00:00:00Z`); model datasets use
the existing fixed June synthetic prices/seed. SHA identities reproduce in the same numeric
runtime. Different Python/numeric libraries or implementation digests can change the model
identity and are bound by the existing experiment runtime contract. A repeat in a **new** directory
preserves every prior run. The research registry `head_hash` is run-specific (the append-only ledger
stamps wall-clock `recorded_at` on some records); its record count (859) and `verified: true` reproduce.

Archive events are not attached to the June synthetic model rows. There is no admitted overlapping
real price/event training population in this demonstration. Synthetic prices, model training,
shadow costs and injected incidents establish that the software path works; they establish no
real edge. No new real training, holdout price access, external model call, broker connection or
real order occurs. Missing original inference telemetry stays unknown.

To inspect the result in the existing cockpit Lab/monitoring views, start the read-only app with
`--research-root var/trading_lab/demo-001/registry` or configure that private read-only root in
[the operations template](OPS_SUPERVISION_V1.md). `/api/v1/observability/*` reads its ledger;
`/api/v1/ops/health` serves versions and operational telemetry. The new model-control listener is
separate from this read-only presentation. Browser qualification remains the cockpit lane's task.

## Cockpit walkthrough

Start the read-only app against the demo registry (above) and the web dev server
(`cd apps/web && npm ci && npm run dev`). Every page below only reads; nothing sends a write.
The Lab pages also need the demo's job state and a local token (otherwise `/api/v1/lab` answers 503,
or 401 without the token); the token is a private value you choose, never committed:

```bash
export HYPRL_MODEL_LAB_TOKEN=<private local value>
python -m scripts.trading_lab.app_api.server --host 127.0.0.1 \
  --research-root var/trading_lab/demo-001/registry --model-lab-root var/trading_lab/demo-001/lab \
  --fomc-store "$HYPRL_FOMC_ARCHIVE" --edgar-store "$HYPRL_EDGAR_ARCHIVE"   # archive flags optional
```

1. **Events**: the FOMC/EDGAR snapshot at the store's own attested instant and horizon, with identity,
   source health, items and the verified replay. Horizons stay separate per source.
2. **Lab → Datasets** (unlock with the local operator token once): the synthetic dataset, its admissible
   decisions and each exclusion reason. The page prepares commands; it does not send them.
3. **Lab → Models**: declared capabilities of the internal model and the local adapter; frozen reference
   versus demonstration models.
4. **Lab → Experiments**: results against the baselines. The negative result is kept and labelled.
5. **Lab → Predictions**: one record with its snapshot/feature evidence; outputs a model did not provide read
   "not provided"; pending labels stay pending.
6. **Lab → Monitoring**: missing data, technical degradation, drift and performance drop, kept apart, with the
   sample and method behind every edge figure.
7. **System → Operations health**: running versions, last operations, freshness, error codes, job and EDGAR
   budgets, workers and API resources from `/api/v1/ops/health`. An unobserved value reads "not observed";
   archive age is not a live-feed guarantee.
8. **API docs**: the B2B main path with required permissions, links to `docs/API_B2B_V1.md`, the OpenAPI document and
   `examples/b2b_client.py`. The cockpit holds no project key and sends no B2B request; run
   `python -m examples.b2b_client --demo` for the same path over HTTP.

Automated: `apps/web/src/test/journey.test.tsx` walks steps 1-8 in one mounted app on real-shaped responses and
asserts no non-GET request; `ops-health.test.tsx` and `api-docs.test.tsx` cover the panel and the page
(the latter checks every listed route and permission against `docs/artifacts/b2b_openapi_v1.json`).
These run in jsdom: a browser check is still BLOCKED (no browser binary on this host).

Qualification: `python -m pytest tests/ops_supervision -q`, plus the required source/API suites,
shared snapshots/contracts, Model Lab and research tests. The CI workflow runs the new offline
qualification without private archive inputs. Public demonstration evidence contains only
identities/counts and records which real archive checks were actually run.
