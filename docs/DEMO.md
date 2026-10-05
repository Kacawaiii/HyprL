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
preserves every prior run.

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

Qualification: `python -m pytest tests/ops_supervision -q`, plus the required source/API suites,
shared snapshots/contracts, Model Lab and research tests. The CI workflow runs the new offline
qualification without private archive inputs. Public demonstration evidence contains only
identities/counts and records which real archive checks were actually run.
