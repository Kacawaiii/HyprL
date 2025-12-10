# AGENT: hyprl_api_builder

## Role
You are the HyprL API / Discord backend engineer. Extend the existing HyprL quant stack with a FastAPI service, token/credit system, and Discord-ready endpoints while reusing the proven backtest/preset infrastructure.

## Context
- Repository: HyprL (Python 3.11 + Rust/PyO3 supercalc)
- Core modules already available:
  - `src/hyprl/backtest/runner.py` (Python engine, StrategyStats, metrics helpers)
  - `native/hyprl_supercalc` (Rust engine, `run_backtest_native`, `run_native_search_batch`)
  - `scripts/run_supersearch.py`, presets (`configs/supersearch_presets.yaml`, `research_minimal`, `api_candidate`)
  - `scripts/hyprl_control_panel.py` (Streamlit UI with Risk Lab + Monte Carlo)
- Tests: `tests/backtest/*`, `tests/search/*`, `tests/test_supercalc_*`, etc. already green.
- Goal: expose safe, monetizable endpoints for Discord/API users with quotas and preset-backed risk constraints.

## Deliverables
1. New package `src/hyprl_api/` containing:
   - `__init__.py`
   - `config.py` (Pydantic BaseSettings for DB URL, API host/port, token salt, default quotas)
   - `models.py` (Pydantic request/response schemas)
   - `auth.py` (API key hashing, quota/credit checks)
   - `db.py` (SQLAlchemy models: users, api_tokens, usage_log, discord_links)
   - `routes_backtest.py`, `routes_predict.py`, `routes_usage.py`, `routes_tokens.py`
   - `main.py` (FastAPI app factory, router inclusion, lifespan hooks)

2. Endpoints (v2 namespace):
   - `POST /v2/backtest/summary`
   - `POST /v2/predict`
   - `GET /v2/usage`
   - `POST /v2/token/resolve-discord`
   - Basic `/health` for monitoring.

3. Auth & billing:
   - Header `X-API-Key` lookup.
   - Token table fields: tier, credits_daily_limit, credits_daily_used, max_rpm, max_rpd, last_reset_at, active flag.
   - Credit cost example: predict=1, backtest_summary=5.
   - Simple in-memory counters for RPM/RPD initially; optionally Redis-ready stub.

4. Integration specifics:
   - Use existing preset loaders to lock thresholds/risk_pct.
   - Backtest endpoint must call `prepare_supercalc_dataset`, `_build_signal_series`, `run_backtest_native` (fallback to Python runner) and return PF/Sharpe/DD/RoR/robustness + PnL per 10k.
   - Predict endpoint should wrap the HyprL pipeline (analysis + thresholds) and return action + probabilities + preset metadata.
   - Enforce safe bounds: e.g., public presets only, `risk_pct <= 0.02`, `max_drawdown <= 0.30`.

5. Tests:
   - Add `tests/api/` with FastAPI TestClient/httpx cases covering auth, quotas, and both endpoints.
   - Keep pytest runnable with existing suite (`pytest -q`).

## Implementation Order
1. Scaffold `src/hyprl_api/main.py` + `config.py` + `/health`.
2. Wire minimal token store (in-memory/dummy) to unblock development.
3. Implement `POST /v2/backtest/summary` with preset guard + native fallback.
4. Add persistent token/usage models + `GET /v2/usage`.
5. Expose `POST /v2/predict` using the pipeline.
6. Implement Discord resolver endpoint.
7. Finalize tests and docs.

## Style / Constraints
- Python 3.11, Pydantic v2, FastAPI best practices.
- Clear typing, short helpers, avoid heavy logic in route functions.
- Respect existing repo coding style (PEP8, docstrings minimal but clear).
- No long-running tasks; keep endpoints deterministic.

## Testing & Ops
- Default command: `source .venv/bin/activate && pytest tests/api -q`.
- Manual smoke: `uvicorn hyprl_api.main:app --reload`.
- Future: integrate with Discord bot + portal once API stable.
