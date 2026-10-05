# InformationSnapshot(T) v1

`platform/snapshot.py` composes the existing FOMC and EDGAR causal reads with price evidence and event
features V1/V2. It opens stores read-only, pins an independent H per store, and never captures, trains,
replays trades or calls a model. Source specs and frozen research artifacts are unchanged.

Use `SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=..., edgar_store=...,
prices=CorpusPrices(data_root))` as a context manager, then `build(T, products)`. `products` accepts the
six registered symbols and their catalogue aliases; duplicates are rejected. Reopen to use later
commits, or supply `horizons={"fomc": H1, "edgar": H2}` to reproduce a read. There is no common H.
The visibility mode is mandatory, as required by the existing event-intelligence design spec. V1
supports DURABLE_OBSERVED only; retrospective-source reads are not implemented.

`InformationSnapshot.to_dict()` is the complete versioned object. Its fingerprint binds selected
revisions, all observation provenance, each native snapshot identity, source specs/contracts, mapping,
feature policies, price evidence, rule method, coverage and unknown states. It never uses the time of
the API request. Product order is canonical. Repeating a pinned read gives identical bytes and hash.

| Evidence clock | Meaning |
|---|---|
| declared_publication | FOMC declared release or SEC acceptance text; metadata, never eligibility |
| observed_at | source observation clock; null when unknown/unverified |
| ingested_at | local durable transaction wall clock, preserved separately from attestation |
| attested_available_at / available_at | CAUSAL_AVAILABILITY_V3 at the observation transaction |
| revision / revision_created_ingested_at | immutable content/metadata revision and its local creation provenance |

The source FOMC snapshot's legacy `ingested_at` name represents attested transaction availability.
This new object explicitly uses the local transaction wall clock for `ingested_at`; it does not rename
or modify the source's legacy contract. Wall clocks may be skewed: only source-owned causal availability
gates events. Backfill, declared release dates and old filing dates never manufacture prior knowledge.

Every source preserves native item selection states, health, discovery/watchlist, attested bounds,
H/P and identity. NOT_CONFIGURED, NOT_OBSERVED, UNRESOLVED and INTEGRITY_ERROR survive. Per-product
NOT_APPLICABLE and UNKNOWN_MAPPING retain the underlying source state. A watched issuer without an
attested valid listing stays NOT_OBSERVED with null features. Partial onboarding remains visible;
zero feature counts do not prove that no relevant publication occurred outside the observed scope.

The dependency policy `FEATURE_HISTORY_AND_CAUSAL_ATTESTATIONS_V1` verifies acquisition-prefix raws,
all historical observations/revisions/absences and the response evidence used for availability and the
next-prefix barrier. This includes intermediate revisions omitted by a current-only source read.
Digests are reverified on every build. A dependency failure removes that source's usable events and
feature values while retaining an INTEGRITY_ERROR state. Availability selection and feature arithmetic
reuse `SourceJoin`, `event_features.features.values` and V2 classification/windows.

Event understanding uses `EVENT_UNDERSTANDING_RULES_V1`: sourced facts and versioned inputs are separate
from ordinal importance (1..3), relevance and uncertainty. Issuer links use existing verified mapping
evidence. FOMC market relevance is a scoped macro hypothesis; it is never evidence of a causal price
response. Uncertainty about economic impact is 1.0 (unknown), with no calibration claim. Exact provider,
item and revision identities support deduplication. Revision links have observation/raw evidence.
SEC amendment parents and cross-source clusters remain absent without evidence. Inventory, newly
observed items and revisions reuse `ATTESTED_OBSERVATION_CLASSES_V1`. No text interpretation is claimed.

Price selection uses latest closed bar and latest available revision; conflicting simultaneous
revisions fail closed. `MemoryPrices` requires explicitly synthetic observations and snapshots must
also be labelled synthetic. `CorpusPrices` reads the existing canonical candle corpus, checks its
content digest/count, and conservatively gates all bars on the manifest's capture completion. These
are **declared local clocks**, not server attestation or historical exchange publication. Observation
and ingestion remain null because that corpus has no authoritative clocks for them. Its gaps and
revision-history limits remain visible. Synthetic replay markers are never used as historical evidence.
Protected decision times never call a price provider. A corpus whose declared range overlaps a product
holdout is refused before candle bytes are read. No holdout, frozen model or reference result is changed.

## Read-only API

The existing `--fomc-store` / `--edgar-store` server flags configure the source roots; `--data-root`
configures the existing price corpus. Clients cannot supply paths. GET (and HEAD) endpoints:

- `/api/v1/contracts/providers`: qualified FOMC/SEC descriptors plus six future-provider proposals.
- `/api/v1/snapshots?as_of=2026-10-04T02%3A00%3A00Z&products=AAPL&visibility_mode=DURABLE_OBSERVED`:
  `{api_version, read_only, snapshot, fingerprint}`. Optional `fomc_horizon` and `edgar_horizon` name
  individual store commit sequences; a global horizon, duplicate parameters, unknown parameters,
  missing/naive instants, invalid products and invalid horizons return 400. Data failures stay visible
  in a partial snapshot. POST and other mutating methods remain refused by the existing server.

Local Python example:

```python
from scripts.trading_lab.platform.snapshot import SnapshotBuilder
from scripts.trading_lab.platform.prices import CorpusPrices

with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", fomc_store=fomc_dir,
                     edgar_store=edgar_dir, prices=CorpusPrices("data/crypto")) as reader:
    snapshot = reader.build("2026-10-04T02:00:00Z", ["AAPL", "BTC-USD"])
    print(snapshot.identity)
```

Future descriptors for macro, news, companies, crypto, regulation and market expectations declare
WAITING_AUTHORIZATION and **shape not verified against the live source**. Synthetic samples are in
`platform/synthetic.py`. SEC and Coinbase samples reuse documented native shapes (CIK digit strings;
Coinbase `[time, low, high, open, close, volume]`). Macro and expectation samples are explicitly proposed
normalized envelopes, not qualified wire schemas. No provider can be activated through these endpoints.

## Offline evidence and limits

Run with operator-configured read-only archive locations:

```bash
python -m scripts.trading_lab.platform.demo --visibility-mode DURABLE_OBSERVED \
  --fomc-store "$FOMC_ARCHIVE" --edgar-store "$EDGAR_ARCHIVE" \
  --data-root data/crypto --product AAPL --product BTC-USD \
  --as-of 2026-08-15T00:00:00Z --as-of 2026-10-02T14:00:00Z --as-of 2026-10-04T02:00:00Z
```

The CLI returns counts, states and identities only. `docs/artifacts/information_snapshot_v1_evidence.json`
records three reproducible reads: one price read before holdout with no attested events yet; six FOMC
events on October 2; 104 Apple filings on October 4. Independent archive horizons are FOMC 622 and
EDGAR 26. The October price decisions are PROTECTED for BTC; equities have no installed price corpus.
Both event windows remain partial and nonoverlapping. This proves the read-only composition and its
diagnostics; it does not qualify a joint price/event training population or an economic advantage.

Validation: `python -m pytest tests/platform -q`; source/API regression gate from AGENTS.md. The workflow
`sources-ci` runs the new platform suites as an additional step. Existing absent private FOMC fixture
and absent `hyprl_api` checks remain BLOCKED; skipped checks are never counted as passes.
