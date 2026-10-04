# Events → causal features v1

This offline slice joins official event evidence to supplied decision instants. It reads no prices,
acquires nothing, trains nothing, and changes no price model, reference result, capture spec or frozen
hash. The protected 2026-09-01 through 2026-11-30 holdout remains closed. Event-state evaluation during
that interval is permitted and explicitly flagged; it does not join market data.

Implementation: `scripts/trading_lab/event_features/`. Evidence: `docs/artifacts/event_features_matrix_v1.json`.
Source contracts: [official sources](OFFICIAL_EVENT_SOURCES.md), [FOMC](FOMC_V1_OFFLINE_SLICE.md),
[EDGAR](EDGAR_V1_OFFLINE_SLICE.md), `sources/causal.py` (CAUSAL_AVAILABILITY_V3).

## Point-in-time contract

Availability comes exclusively from the store's CAUSAL_AVAILABILITY_V3 table at a fixed commit horizon H.
FOMC declared release times, EDGAR filing dates and `acceptanceDateTime` are provenance. They neither gate
availability nor order event windows. An old filing first acquired today becomes available only when
later server evidence attests its recording transaction. The same rule applies to FOMC backfill manifests.

Coverage is the inclusive bounding interval between the availability of the first and last resolved
transactions at H. It is not a statement that every read inside the bounds resolves. At T, the source's
snapshot requires the *next* transaction to have resolved availability greater than T. Consequently a
read exactly at the last coverage boundary is already UNRESOLVED. No clock extrapolation extends coverage.

| State | Meaning | Feature values |
|---|---|---|
| NOT_OBSERVED | T precedes the first resolved transaction's availability | all null |
| UNRESOLVED | the source snapshot cannot resolve T, including at/after its final boundary; also a store without resolved transactions | all null |
| RESOLVED | the source's snapshot at (T,H) resolves | computed from its available events |
| NOT_CONFIGURED | no store supplied | all null |
| INTEGRITY_ERROR(reason) | store opening or snapshot dependencies fail closed | all null |
| NOT_APPLICABLE | product has no SEC issuer or is outside the 8-K population | EDGAR values null |
| UNKNOWN_MAPPING | issuer identity unverified | EDGAR values null |

Product rows retain both feature applicability `state` and underlying `source_state`, source snapshot
identity, H and P. A mapped issuer outside a resolved snapshot's watchlist is NOT_OBSERVED with reason
`ISSUER_NOT_IN_SNAPSHOT_WATCHLIST`, never a zero. Invalid or naive decision instants are rejected. Failure
reasons contain public error categories, never configured paths or captured text.

`SourceJoin.read_one(T)` uses the source's public snapshot API. `read_many(times)` computes the availability
table once per reader, bisects it, and assembles each distinct P using the source's own selection and raw
verification functions. It preserves the source snapshot's exact header, payload and canonical identity;
the T field and identity are recomputed for each supplied instant. Payloads are cached only within one
batch, and dependencies are verified again on the next call. A reader pins H for its lifetime; reopen it
to use later evidence. This is intended for immutable closure stores, not concurrent raw modification.
Tests compare the complete batch and public per-T results at every synthetic availability boundary ±1 µs,
including duplicates and reverse ordering. The real demo also compares both paths on its four times.

Revisions are selected at P(T). Later revisions leave earlier values unchanged. An event is one logical
FOMC statement (source item) or one EDGAR accession, rather than one observation or metadata revision.
FOMC events carry their current selected revision and the first attested verified normalized link for
that statement, including backfill. Only CURRENT_REVISION items are available statement events; other
source item selection states remain visible in the full source snapshot. EDGAR uses `first_available_at`
and the revision selected by that snapshot. An ABSENT_FROM_LISTING filing remains previously available
evidence and is counted; absence is not a proven removal or a deletion. An 8-K/A is its own accession;
no amendment parent is inferred.

## Product/source mapping

The closed machine-readable mapping is `scripts/trading_lab/event_features/mapping.json`, hashed into
every feature row and matrix. `CATALOGUE_V1` supplies product identity; equity aliases resolve to XNAS.
FOMC is macro evidence for all six products, as scoped by this task and the official-source registry.

| Product | Decision cadence | FOMC | EDGAR identity/status | Evidence |
|---|---|---|---|---|
| BTC-USD | 1h | macro | NO_SEC_ISSUER → NOT_APPLICABLE | `instrument_registry.py`: BTC_USD cryptocurrency |
| ETH-USD | 1h | macro | NO_SEC_ISSUER → NOT_APPLICABLE | `instrument_registry.py`: ETH_USD cryptocurrency |
| AAPL | 1d | macro | 0000320193 VERIFIED | `edgar_fixture_qualification_v1.json`: listed CIK, “Apple Inc.” |
| MSFT | 1d | macro | 0000789019 VERIFIED | same artifact: listed CIK, “MICROSOFT CORP” |
| NVDA | 1d | macro | null, UNVERIFIED → UNKNOWN_MAPPING | catalogue identifies product; no verified CIK in repository evidence at base aed9741 |
| QQQ | 1d | macro | 0001067839 DOCUMENTED; 8-K NOT_APPLICABLE | `TRADING_LAB_EVENT_PROVIDER_VERIFICATION.md`, “Entity semantics”: investment trust and investment-company filing population |

NVDA's CIK is never guessed. QQQ's 8-K exclusion is a scope decision based on the documented trust
population, not an empty listing converted into zero.

## Attested coverage matrix

Computed read-only from the supplied real closure stores and corpus metadata. The module reads only the
crypto manifest and equity specification/fingerprint, never candle rows. “Installed” means the declared
local corpus location exists; hashes/counts below are the committed metadata, not a new price verification.
The equity daily corpus is **not installed in this worktree**. The capture specs remain FOMC rev25
`b9d2a599…01ece` and EDGAR rev1 `98828c55…e5ce`; full hashes, store tree digests, availability-table hashes,
snapshot proof digests and product availability histograms are in the JSON artifact.

| Source | H | Attested coverage (UTC) | Resolved / unresolved transactions | Attested events / revisions |
|---|---:|---|---:|---:|
| FOMC | 622 | 2026-10-02 13:23:07.651429 → 14:51:11.319660 | 618 / 4 | 6 / 6 |
| EDGAR | 26 | 2026-10-04 01:37:18.126128 → 02:07:59.301923 | 23 / 3 | 161 / 161 |

FOMC's six normalized statements are distinct from the source snapshot's broader candidate/item inventory.
They first become attested in five groups: 13:23:19.405474 (2), 13:23:50.715277 (1),
13:24:36.550850 (1), 13:25:50.327503 (1), 13:26:47.176596 (1), all on October 2.
EDGAR's Apple group becomes available October 4 at 01:37:28.178495 (104: 102 8-K + 2 8-K/A);
Microsoft at 01:47:28.498148 (57: 56 8-K + 1 8-K/A).

| Source | Product | Price window (inclusive dates) | Metadata bars | Attested events | Events available in price window | Price/coverage overlap |
|---|---|---|---:|---:|---:|---|
| FOMC | BTC-USD | 2025-08-01 → 2026-07-31, 1h | 8,750 | 6 | 0 | none |
| FOMC | ETH-USD | same | 8,750 | 6 | 0 | none |
| FOMC | AAPL | 2024-08-01 → 2026-07-31, 1d | 501 | 6 | 0 | none |
| FOMC | MSFT | same | 501 | 6 | 0 | none |
| FOMC | NVDA | same | 501 | 6 | 0 | none |
| FOMC | QQQ | same | 501 | 6 | 0 | none |
| EDGAR | BTC-USD | 2025-08-01 → 2026-07-31, 1h | 8,750 | null: NOT_APPLICABLE | null | none |
| EDGAR | ETH-USD | same | 8,750 | null: NOT_APPLICABLE | null | none |
| EDGAR | AAPL | 2024-08-01 → 2026-07-31, 1d | 501 | 104 | 0 | none |
| EDGAR | MSFT | same | 501 | 57 | 0 | none |
| EDGAR | NVDA | same | 501 | null: UNKNOWN_MAPPING | null | none |
| EDGAR | QQQ | same | 501 | null: NOT_APPLICABLE | null | none |

These zeros are **inventory intersections**, not feature values for historical decisions. Historical
decisions precede store observation and get null features. Both coverage intervals fall completely inside
the closed holdout [2026-09-01T00:00Z, 2026-12-01T00:00Z). Thus all six FOMC and 161 EDGAR events are attested
in that interval, and **no studied price window overlaps attested coverage yet**. The listed filings'
older dates do not change this result. The existing corpora and frozen models/results/hashes stay references.

## Features v1 and use

Each source has `count_7d`, `count_30d`, `hours_since_last`; FOMC also has
`statement_within_24h`. EDGAR counts combined 8-K and 8-K/A for the mapped issuer and has `item_2_02`,
`item_5_02`, `item_7_01`, `item_8_01`. Count windows are (T−7d,T] and (T−30d,T]; the 24-hour flag includes
an event exactly 24 hours before T. Hours use elapsed UTC time since the most recent attested event.
EDGAR item flags use comma-separated exact item tokens from revisions available at T in the 7-day window.
If an optional `items` field is absent, a flag without positive evidence is null; observed positive items
remain true. Filing/acceptance dates never contribute to these features.

For RESOLVED with no available events, counts are 0, flags false and hours null. Otherwise only RESOLVED
feature states carry values. Counts describe **observed available events**, not completeness of the world's
event history. Windows reaching before observation start are left truncated; a later comparison needs
a 30-day observation warmup and source coverage/revision diagnostics, not imputed pre-observation zeros.

```python
from scripts.trading_lab.event_features import EventFeatures

with EventFeatures(fomc_store="FOMC_DIR", edgar_store="EDGAR_DIR") as features:
    rows = features.rows([("AAPL", "2026-06-01T00:00:00Z"),
                          ("BTC-USD", "2026-10-02T14:30:00Z")])
```

```bash
python -m scripts.trading_lab.event_features.demo --fomc-store DIR --edgar-store DIR
# Optional aggregate artifact: --matrix-output docs/artifacts/event_features_matrix_v1.json
python -m pytest tests/crypto/test_event_features.py -q
```

The demo prints the matrix, proof checks and AAPL/MSFT/QQQ/NVDA/BTC-USD rows at June 1, October 2 14:30,
October 4 02:05 and one second after the latest coverage. It then corrupts disposable private copies,
prints fail-closed states **and feature rows**, removes the copies, and proves the originals unchanged
by whole-tree digests. The complete real run output belongs in the operator report outside Git.

| Demo T (UTC) | FOMC, all five products | EDGAR AAPL / MSFT | Other EDGAR |
|---|---|---|---|
| 2026-06-01 00:00 | NOT_OBSERVED, null | NOT_OBSERVED / NOT_OBSERVED, null | QQQ/BTC NOT_APPLICABLE, NVDA UNKNOWN_MAPPING |
| 2026-10-02 14:30 | RESOLVED; 7d/30d 6/6, last 1.0535620567 h, within 24h true | NOT_OBSERVED / NOT_OBSERVED, null | same applicability states |
| 2026-10-04 02:05 | UNRESOLVED, null | RESOLVED; 104/104, last 0.4588393069 h / 57/57, last 0.2920838478 h | same applicability states |
| 2026-10-04 02:08:00.301923 | UNRESOLVED, null | UNRESOLVED / UNRESOLVED, null | same applicability states |
| corrupted copy at source's resolved demo T | INTEGRITY_ERROR for corrupted source, null | INTEGRITY_ERROR for corrupted source, null | applicability states retain underlying source error |

## Data and decisions needed for “prices only vs prices + events”

The comparison is **not done**. There are currently zero usable paired decisions in the studied historical
price windows. This slice proves join semantics, not predictive signal or model improvement. Before a
comparison, the operator must preregister the period, eligible products, decision clocks, observation
warmup, source-state exclusion rules, sample size/power, labels, costs, splits and evaluation, and explicitly
authorize any training. Reference models/results are preserved, with any new result carrying separate hashes.

The following is a concrete data checklist; proposed dates/volumes are requests for later operator decisions,
not authorizations or acquisition defaults.

| Data | Source/products/period/volume | Required decision or authorization |
|---|---|---|
| Forward attested events, preferred comparison route | FOMC macro for all six; EDGAR submissions for AAPL/MSFT and NVDA only after verification, QQQ excluded from 8-K. Proposed 30-day warmup 2027-03-02 → 2027-03-31, then 2027-04-01 → 2027-07-31, with later attesting responses through at least August 2 and until the final supplied decision resolves. Preserve every response, observation, revision and failure; event count cannot be known in advance. Conservative ceilings at existing cadences: FOMC at most one feed poll/minute plus budgeted statement/rechecks; EDGAR at most 144 listings/issuer/day (≥600s spacing), shared pacing and stop-on-throttle. | Explicit endpoint, issuer, request/raw-volume budgets and expiry in operator authorization; a supervised multi-month EDGAR capture is not proven by the 8-response trial and needs implementation/qualification first. Decide source-state coverage criteria and retention. No capture started here. |
| Fresh matching prices | Coinbase BTC/ETH 1h and an approved equity daily provider for AAPL/MSFT/NVDA/QQQ over the chosen forward evaluation period. Proposed April 1 → July 31 2027: 2,928 expected hourly openings/product and 84 expected daily sessions/product under the pinned calendar (computed from `USEquityCorpusV2` with that range). Request each product's complete expected grid plus needed price-feature history; verify gaps, actions, policies and content hashes. | New corpus spec/window and price acquisition authorization. These proposed dates also avoid the separate equity confirmatory 2027-Q1 holdout in `equity_research.py`. The 2026 holdout stays closed; any later reserved interval must be excluded too. Choose daily decision clock and market-data availability explicitly. |
| Installed existing equity corpus | Four daily instruments, 2024-08-01 → 2026-07-31, 501 bars each (2,004 total), corporate-action metadata and original manifests. Spec `b7ad1e33…87dae`, content `64ac4485…e024`; full references in `us_equity_corpus_v2_fingerprint.json`. Currently absent locally. | Operator provides/installs the authorized local research corpus and its usage rights, then verify the committed identities. This adds prices but does not create event overlap. No fetch or source-body redistribution here. |
| Older EDGAR pages and historical completeness | AAPL/MSFT 8-K/8-K/A for 2024-08-01 → 2026-07-31, and NVDA after verified mapping. Fixture metadata advertises 1 older page for Apple and 2 for Microsoft; actual needed pages/filings are unknown until an authorized coverage inventory. Retrieve only pages spanning the requested window plus all listing revisions available under the chosen policy; do not assume the current 161 filings are complete history. | New authorization explicitly naming older submissions pages, issuer set, volume ceiling and expiry. Existing spec/runner admit only current submissions; older-page support needs a new spec revision, canonical hash, code bindings and separately bound stores, never migration. Capturing these now remains newly attested evidence and cannot repair historical availability. |
| NVDA mapping | One official ticker/CIK lookup plus identity evidence for NVDA; zero such reads performed here. | A narrow lookup authorization naming endpoint, one response budget and expiry. Commit only identity, source URL and digest/count evidence; then version the mapping. |
| Optional historical research mode | Separate, explicitly **non-attested** declared-availability features for the existing historical price windows: FOMC statements and issuer filings/revisions spanning those windows plus 30-day warmup. FOMC declarations and EDGAR timestamp semantics require a stated research assumption and independent verification; `acceptanceDateTime` remains untrusted provenance under the current attested spec. Required event/response volume is unknown until inventory and a budget are approved. | Explicit operator decision to permit this separate research policy and label every row/result non-attested, with its own policy/mapping hashes, validation and acquisition budgets. It is never enabled by this module, never the default, and cannot be presented as store-attested history. |

The price module must supply decision timestamps without exposing holdout prices to this join. Paired
comparisons must use the same eligible decisions and economic assumptions on both sides and report
missing-source exclusions and left truncation. A four-month proposal is a bounded first collection period,
not a claim of statistical power; the operator must decide the required evidence volume before outcomes.

## Verification and limits

Synthetic stores retain real source shapes, including zero-padded string `submissions.cik`, parallel
listing columns and real FOMC feed/page grammars. Tests cover T inclusion, T−1µs decisions, 7/30-day and
24-hour edges, NOT_OBSERVED, UNRESOLVED, resolved counts, corrections, backfill, corrupt raws, applicability,
unwatched issuers, holdout flags, optional missing metadata, complete batch/public equality, determinism
and byte-identical stores. The final required repository suite is recorded in the report.

The two real stores are short closed trials, not historical or continuous observation. EDGAR older pages,
new publication/correction/removal behavior, continuous supervision and several provider semantics remain
unproven as listed in the source registries. Counts of acquired historical statements/filings clustered
at capture time describe availability of evidence, not actual publication incidence or market reactions.
There is no declared-availability fallback, price join, acquisition path, training or trading command.
