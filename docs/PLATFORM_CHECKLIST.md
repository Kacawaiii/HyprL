# HyprL platform checklist

Single source of truth for what the platform does **today**, what is buildable now, what waits for data or an
authorization, and what is future. Written 2026-10-05 against `origin/feat/phase5` = `caa5bfc` (CI run
`37250006768`, success; code identical to `d744ba3`, run `37233200078`). Rows are updated by the slice that changes them.

States: **DONE** (the path works *and* its proof exists), **IN_PROGRESS**, **BLOCKED**, **WAITING_DATA**,
**WAITING_AUTHORIZATION**, **NOT_STARTED**. A skipped or unrunnable check is BLOCKED, never a pass. "Branch/SHA" is
where the work lives: `phase5@d744ba3` means it is integrated; any other value means it is pushed but **not** in
`feat/phase5` yet.

Vision chain: Event Intelligence → InformationSnapshot → Model Lab → Model Observability → cockpit and API
(mandate: operator, 2026-10-05).

Proof shorthand: **GATE** = the AGENTS.md Python suites (`tests/crypto/test_fomc_*.py … tests/api`); **WEB** =
`cd apps/web && npm ci && npm run typecheck && npm run lint && npx vitest run`; **CI** = workflow `sources-ci`.

## 1. Delivered and verified

| Feature | State | Depends on | Branch/SHA | Proof | Next action |
|---|---|---|---|---|---|
| FOMC official source v1 (collector, causal store, snapshot P(T) under horizon H, replay, health) | DONE | `sources/` primitives | phase5@d744ba3 (spec rev 25, `fomc-store-v5`) | `tests/crypto/test_fomc_*.py`; real pilot rev25 VALIDATED, closure copy re-reads its 10 recorded snapshots identically (`mvp_check --fomc-reads`); `docs/FOMC_V1_OFFLINE_SLICE.md` | none for the slice; 24 h / 7-day rechecks and a new real publication are not proven (section 3) |
| SEC EDGAR 8-K source v1 (collector, causal store, snapshot, replay, health) | DONE | `sources/` primitives | phase5@d744ba3 (spec rev 1, hash `98828c55…ee5ce`, `edgar-store-v1`) | `tests/crypto/test_edgar_slice.py`; fixture trial closure copy read-only; digit-string `cik` fix `2385f74` | supervision hardening is a separate row (section 2) |
| Shared source primitives (append-only store, causal availability, HTTP clock, limiter, scale, canonical form) | DONE | none | phase5@d744ba3 | `tests/crypto/test_sources_primitives.py`; `docs/SOURCE_SCALE_AND_PAGINATION.md` | none |
| Read-only application API for sources (status, snapshot, replay, item/filing detail, causal events timeline, bounded pagination with bound cursors) | DONE | FOMC and EDGAR rows | phase5@d744ba3 (scale-and-pagination, hardening-review fixes) | `tests/crypto/test_app_api*.py`, `test_ops_server.py`; GATE 473 passed / 2 skipped on the night integration; CI `37233200078` | none |
| MVP acceptance check (`mvp_check`: 23 criteria per store pair + cockpit) | DONE for the two closure stores | read-only API, cockpit | phase5@d744ba3 | night run: 23 PASS / 0 FAIL / 0 BLOCKED (`~/reports/mvp-check-integration-night.json`, private); `tests/crypto/test_mvp_check.py`. `docs/MVP_ACCEPTANCE.md` itself is still **PROPOSED** (operator has not accepted it) | operator accepts or amends the criteria |
| Cockpit web (Overview, Markets, Signals, Risk, Backtests, Portfolio, Paper, Research, Events, System, Settings; source paging, timeline tab, replay section, readable numbers) | DONE for the current pages | read-only API | phase5@d744ba3 | WEB (126+ vitest cases) and CI web job on `d744ba3`; `docs/MVP_ACCEPTANCE.md` COCKPIT-01..04 | Beginner/Expert modes, Model Lab and observability views are not built (sections 2 and 4) |
| Local operations (`hyprl start/stop/status/doctor/logs/build/release/export/import/support-bundle`, `hyprl start` serves the FOMC/EDGAR stores) | DONE | none | phase5@d744ba3 (`7685ab0`) | `docs/TRADING_LAB_LOCAL_OPERATIONS_V1.md`; `tests/ops`, `tests/crypto/test_ops_server.py` | none |
| Crypto market data and datasets (Coinbase history v1, causal bars, dataset, indicators) | DONE | none | phase5@d744ba3 | corpus hash `688c250d…748b`; `tests/crypto/test_market_dataset.py`, `test_coinbase_candles.py` (CI step "Paper model, replay and engine suites") | extension to news/macro products is future (section 4) |
| Walk-forward, signal engine, risk engine, economic backtest, portfolio backtest (BTC/ETH) | DONE | datasets | phase5@d744ba3 | `tests/crypto/test_walk_forward.py`, `test_signal_engine.py`, `test_risk_engine.py`, `test_economic_backtest*.py`; frozen results under `data/crypto/*_v1` with manifests | none; results stay frozen |
| Persisted OOS signal and target runs shown on Signals and Risk | DONE | walk-forward | phase5@d744ba3 (`7e372b4`, `5875a02`) | `tests/crypto/test_app_api_signal_runs.py`; WEB | none |
| Paper / shadow engine (simulated money, no broker) | DONE | risk engine | phase5@d744ba3 | `tests/crypto/test_paper_shadow.py`, `test_paper_portfolio.py`; `docs/TRADING_LAB_PAPER_SHADOW_V1.md`; BTC/ETH embargoed 2026-09-01 → 2026-11-30 | none |
| Paper model v2 (trained to 2026-04-30) and out-of-sample replay | DONE as a mechanism; result is **no edge** | paper engine | phase5@d744ba3 | `tests/crypto/test_paper_model_v2.py`, `test_paper_replay.py`; spec hash `76fbec5a…d1d038`; `docs/PAPER_REPLAY_OOS_V2.md`: rank IC −0.073 (BTC) / −0.079 (ETH), ETH one losing round trip | none; not a commercial claim (see MORNING limits) |
| Equity research / benchmark v1 (AAPL, MSFT, NVDA, QQQ, exploratory) | DONE as exploratory | equity corpus | phase5@d744ba3 | `docs/artifacts/equity_benchmark_v1_results.json` result hash `664782bc…98291b`; `tests/crypto/test_equity_*.py` | equity is the exploratory arm of this campaign (section 3) |
| Point-in-time join of attested events to decision times, event features v1, coverage matrix | DONE (offline); no price window overlaps attested coverage | FOMC/EDGAR snapshots | phase5@d744ba3 (`0fa3de9`, `af20f28`) | `tests/crypto/test_event_features.py` (24 cases, GATE+ 528 passed / 2 skipped); matrix identity `a5382f91…917d4`; `docs/EVENT_FEATURES_V1.md` | prices-only vs prices+events comparison needs data (section 3) |
| CI (`sources-ci`: Python gate, paper/engine suites, web job with build) | DONE | none | phase5@caa5bfc | runs `37233200078` (`d744ba3`) and `37250006768` (`caa5bfc`) success | extend with each new suite |
| Unified platform checklist (this document) | DONE | none | phase5@caa5bfc | GATE 504 passed / 2 skipped (351 s, doc-only slice); CI `37250006768` success | each slice updates its rows |
| Source read documentation | DONE | none | phase5@d744ba3 | `docs/OFFICIAL_EVENT_SOURCES.md`, per-source registries | none |

## 2. Buildable now (no new data, no new authorization)

| Feature | State | Depends on | Branch/SHA | Proof | Next action |
|---|---|---|---|---|---|
| Research protection (protected intervals per product, warm-ups, admissible decision ranges) | IN_PROGRESS: done and CI-green, **not integrated** | event features | `agents/claude/protocol-features-v2`@`a6d155d` | `tests/crypto/test_research_protection.py`; protection identities `bf95ee85…af85` (crypto) and `b6ae7338…0c44` (equity); fresh-venv run 504 passed / 2 skipped; CI run `37241056023` success on `a6d155d` | integrate (task `lot6-integrate`) |
| Event features v2 (INITIAL_INVENTORY / NEWLY_OBSERVED / REVISION distinction) | IN_PROGRESS: done, **not integrated** | research protection | same branch `a6d155d` | `tests/crypto/test_event_features_v2.py`; V1 identity re-verified; on the real stores every observation is INITIAL_INVENTORY (6 FOMC, 161 EDGAR) | integrate |
| Comparison protocol v1 (prices only vs prices + events, crypto primary, equity exploratory), hash `6d8d7829…c4f` | IN_PROGRESS: written, **not integrated**, not accepted by the operator | protection, features v2 | same branch `a6d155d` | `tests/crypto/test_comparison_protocol.py`; `docs/artifacts/comparison_protocol_v1.json` | integrate; operator reviews; equity period needs revision 2 (37 test sessions vs 100 minimum) |
| EDGAR supervised service (single owner, budget across epochs, expiry before send, SIGTERM, closure with authorization identity, runbook) | IN_PROGRESS: done, **not integrated** | EDGAR source | `agents/codex/edgar-supervised`@`6419fec` | report STATUS: DONE; GATE+ 546 passed / 2 skipped in 351 s; spec hash unchanged; closure and `--check` CLI run offline. No CI run recorded yet | integrate (task `lot6-integrate`, codex retry pending after its usage limit) |
| Events timeline (separate implementation) | BLOCKED: duplicate | none | `agents/codex/events-timeline`@`cbacfc2` | its own 425-pass run | superseded by the timeline in scale-and-pagination (phase5); do not merge, operator may delete the branch |
| Provider contract (capabilities, identities, formats, clocks, limits, corrections, historical availability, health), validated with FOMC/EDGAR | NOT_STARTED | FOMC, EDGAR | none | design only: `docs/TRADING_LAB_EVENT_INTELLIGENCE_ARCHITECTURE.md` (STATUS: DESIGN ONLY) | task `contracts-snapshot` |
| InformationSnapshot(T) v1 (prices + events, per-store horizons, provenance, freshness, coverage, unknown states, policy versions, fingerprint) | NOT_STARTED as one object; its parts exist (source snapshots, PIT join) | provider contract | none | parts: see section 1 | task `contracts-snapshot` |
| Event understanding (typology, entities, dedup, fact vs interpretation, importance with version and limits) | NOT_STARTED | InformationSnapshot | none | none | after snapshot contract |
| Model contract and Model Lab path (dataset → features/events → training → validation → backtest → paper/shadow → monitoring; internal model + a second local adapter; long jobs with id, state, logs, cancel, limits, isolated workers) | NOT_STARTED | snapshot, features v2 | none | none | tasks `model-lab` (after `contracts-snapshot`) |
| Hypothesis and experiment registry (falsification, baselines, purge/embargo, fit-on-train, costs, pre-fixed criteria, budgets, null results kept) | NOT_STARTED | model lab, protocol | none | protocol v1 is the seed | task `research-observability` |
| Model Observability (per-prediction record, late labels without rewriting, freshness, drift, regimes, per-period performance, edge vs baseline) | NOT_STARTED | model lab | none | none | task `research-observability` |
| Cockpit Beginner / Expert modes (selection kept across modes, aligned chart, 4 h projection with reference price, TP/SL with origin, separate scores) | NOT_STARTED | read-only API, observability for full scope | none | none | task `cockpit-modes` (existing data first) |
| Cockpit Model Lab / research / observability views | NOT_STARTED | model lab, observability | none | none | task `cockpit-lab` |
| Cockpit journeys: navigation, pagination, loading, errors, responsive, keyboard, reduced motion, browser check | IN_PROGRESS: pagination, loading and error states exist on source pages; the rest is unproven | cockpit | phase5@d744ba3 (partial) | WEB covers the existing journeys; no browser run recorded | tasks `cockpit-modes`, `api-docs-ops-ui` |
| B2B API v1 (versioned contracts, auth, permissions, project isolation, budgets, audit log, docs, client example) | NOT_STARTED | model lab, observability | none | none | task `api-b2b`, then `api-docs-ops-ui` |
| Supervision and operations panel (versions, last operations, freshness, errors, budgets, workers, resources; start/stop/backup/restore/resume commands) | NOT_STARTED beyond `hyprl doctor` and the EDGAR runbook | EDGAR supervised | none | `hyprl doctor` only | task `ops-supervision` |
| End-to-end synthetic demonstration (snapshot → dataset → model → prediction → monitoring) | NOT_STARTED | all of the above | none | none | task `ops-supervision` |

## 3. Qualification or data needed

| Feature | State | Depends on | Branch/SHA | Proof | Next action |
|---|---|---|---|---|---|
| Prices-only vs prices + events comparison | WAITING_DATA | attested event coverage over a price window | protocol `a6d155d` | no studied price window overlaps attested coverage (matrix `a5382f91…`); backfills keep their real availability and cannot create history | capture authorization (below) after the operator accepts the protocol |
| Continuous FOMC and EDGAR capture for the protocol period (2027-03-02 → 2027-08-02) | WAITING_AUTHORIZATION | protocol, store-size ceiling | none | protocol budgets: FOMC ceiling 220,800 requests, EDGAR 66,097, Coinbase 20, equity prices 4. Raw worst case ~30 GB vs 21 GB free disk | authorization file under `~/authorizations/` with scope, budget, expiry, store-size ceiling |
| Multi-month EDGAR runner, NVDA CIK verification, older-page support | NOT_STARTED, needs authorization to verify | EDGAR supervised | none | the runner admits 1–50 requests; NVDA CIK unverified | build the runner offline, verify under authorization |
| FOMC 24 h / 7-day rechecks, a new real FOMC publication, real redirects | WAITING_AUTHORIZATION | real capture | none | NOT_PROVEN in `docs/FOMC_V1_OFFLINE_SLICE.md` | authorization for the scope |
| Equity arm of the protocol | WAITING_DATA | equity daily corpus install | none | expected INCONCLUSIVE at the proposed period (37 test sessions per instrument vs 100 minimum) | protocol revision 2 for the period |
| Confirmatory holdouts (crypto 2026-09-01 → 2026-12-01; equity sessions 2026-12-01 → 2027-02-28) | WAITING_DATA | calendar | `research_protection.py` on `a6d155d` | identities `bf95ee85…`, `b6ae7338…`; not read by any agent | untouched until a task authorizes it |
| Real model training beyond the frozen v1/v2 | WAITING_AUTHORIZATION | protocol, Model Lab | none | none | authorization per experiment |
| External model adapter calls | WAITING_AUTHORIZATION | model contract | none | none | authorization per adapter |
| Hardening-review open decisions: EDGAR UV1 exact-column rule vs doc wording (needs a new spec revision), budget scope across repeated launches | BLOCKED: operator decision | none | `agents/codex/hardening-review`@`eadb351` (fixes already in phase5) | report STATUS: BLOCKED | operator chooses |
| `docs/MVP_ACCEPTANCE.md` acceptance | WAITING_AUTHORIZATION (operator sign-off) | none | phase5@d744ba3 | 23/23 PASS | operator accepts |
| Two GATE checks that skip | BLOCKED | private official FOMC fixtures; `hyprl_api` package absent | none | 2 skipped in every GATE run | not counted as passes |

## 4. Future platform extension

| Feature | State | Depends on | Branch/SHA | Proof | Next action |
|---|---|---|---|---|---|
| Further providers: macro, news, companies, regulation, market expectations | NOT_STARTED | provider contract | none | documentation and realistic fixtures first; real activation needs authorization | after the contract |
| Crypto-native event sources (regulation, listings) | NOT_STARTED | provider contract | none | none | after the contract |
| Broker integration and real orders | NOT_STARTED, out of scope | none | none | forbidden by the mandate | not planned |
| Calibrated probabilities and intervals as cockpit claims | NOT_STARTED | model lab, observability | none | needs a named method and a calibration | after observability |
| Versioned TP/SL risk policy simulation (gaps and intrabar ambiguity stated) | NOT_STARTED | risk engine | none | must stay distinct from frozen references | after cockpit modes |
| Multi-project tenancy and billing | NOT_STARTED | B2B API | none | none | after B2B API |
