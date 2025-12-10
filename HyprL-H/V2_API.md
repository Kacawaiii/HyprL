# HyprL V2 API – Persistence & Token Management

## Storage Overview
- Database: SQLite (default `data/hyprl_v2.db`, overridable via `HYPRL_DB_URL`)
- Accounts persist credits (`credits_total`, `credits_remaining`)
- Tokens store only Argon2id hashes (`hash` column) + JSON-encoded scopes. Plain tokens are never persisted.
- Usage events are appended per successful endpoint call for audit/billing.

## Authentication / Bootstrapping
- Bearer tokens follow the format: `tok_<id>.<secret>`.
- `HYPRL_ADMIN_TOKEN` seeds the admin token at startup (scopes: `admin:*`, `read:predict`, `read:usage`, virtually unlimited credits). If unset, a dev-only fallback `tok_admin.hyprl_admin_dev_123` is provisioned and printed once.

## Endpoints

### `POST /v2/tokens` (admin only)
Creates an API token, optionally provisioning the backing account.

**Request**
```json
{
  "account_id": "acc_clientA",
  "scopes": ["read:predict", "read:usage"],
  "label": "clientA_bot",
  "credits_total": 100000,
  "expires_at": null
}
```

**Response 201**
```json
{
  "token_id": "tok_y1x2z3",
  "token_plain": "tok_y1x2z3.MFDfrWfC0O6WFevSOG8VPw",
  "scopes": ["read:predict", "read:usage"]
}
```
The `token_plain` is shown only once—clients must store it securely. Accounts are created (and credits initialized) when missing.

### `DELETE /v2/tokens/{token_id}` (admin only)
Soft-revokes a token. Subsequent calls using that token will return `401` (`{"error":"unauthorized"}`).

### `POST /v2/predict` (scope `read:predict`)
- Cost = number of symbols.
- Credits are debited transactionally; if insufficient, returns `402 {"error":"insufficient_credits"}`.
- Each successful call appends a `UsageEvent` (`endpoint="v2/predict"`) used by `/v2/usage`.
- Rate-limit headers remain unchanged (`X-RateLimit-Limit: 60`, `X-RateLimit-Remaining: <int>`).

### `GET /v2/usage` (scope `read:usage`)
Returns live credit balances and aggregated usage derived from `UsageEvent`.

**Response**
```json
{
  "account_id": "acc_clientA",
  "credits_total": 100000,
  "credits_remaining": 99997,
  "by_endpoint": {
    "v2/predict": 3
  }
}
```

## Error Codes
- `401 {"error":"unauthorized"}` – missing/invalid token, revoked or expired.
- `403 {"error":"forbidden"}` – token lacks required scopes.
- `402 {"error":"insufficient_credits"}` – quota exhausted.
- `429 {"error":"rate_limited"}` – in-memory token bucket reached 60 req/min limit.

## Redis Rate-limit
- Optional `REDIS_URL` (e.g. `redis://redis:6379/0`) enables a Redis-backed token bucket shared across API instances.
- When Redis is unreachable, the service logs a warning and automatically falls back to the in-process limiter (same contract, but per-instance).
- `deploy/docker-compose.redis.yml` spins up a local Redis (`docker compose -f deploy/docker-compose.redis.yml up -d`).

## Predict Bridge (stub vs real)
- `HYPRL_PREDICT_IMPL=stub` (default) → deterministic stub used for unit tests/regression.
- `HYPRL_PREDICT_IMPL=real` → attempts to load the HyprL ProbabilityModel stack; if it fails, the bridge logs `[PREDICT] fallback stub ...` and returns the stub output to preserve uptime.

## Environment
```
HYPRL_DB_URL=sqlite:///data/hyprl_v2.db
HYPRL_ADMIN_TOKEN=<tok_admin.secret>  # optional
HYPRL_PREDICT_IMPL=stub               # or "real"
REDIS_URL=redis://redis:6379/0        # optional shared rate-limit
```

Ensure the application has write access to the DB directory. When `HYPRL_DB_URL` changes (e.g., during tests), the engine is reconfigured automatically.

### `POST /v2/sessions` (scope `write:session`)
Starts a realtime paper session powered by the SessionManager. Payload matches the realtime CLI knobs (symbols, interval, risk, kill-switch). Costs 50 credits per session (debited before the worker is spawned). Logs are written under `data/live/sessions/<id>` (bars/predictions/fills/equity/manifest). The endpoint enforces scopes, quotas, and rate limits identical to `/v2/predict`. On success it returns the session id, log directory, implementation used (`stub` or `real`), and metadata ({"debited": 50}).

### `GET /v2/sessions/{id}` (scopes `read:usage` or `read:session`)
Provides the current session status, last event timestamp, counters derived from JSONL files, and kill-switch state. Returns `404` if the session manifest is missing. `DELETE /v2/sessions/{id}` (scope `write:session`) performs a graceful stop (signals the worker, cancels orders, waits for completion) and responds `204`.

### `GET /v2/sessions/{id}/report`
Aggregates the session JSONL logs into headline metrics (`pf`, `sharpe`, `dd`, `winrate`, `exposure`, `avg_hold_bars`) plus the top rejection reasons. Useful for dashboards/QA after a paper run.

### Autorank → Sessions Orchestrator

`POST /v2/autorank/start` (scope `write:session`) ingests one or more Supersearch CSVs (paths limited to `data/experiments/` or `docs/experiments/`), recomputes scores via `hyprl.analysis.meta_view`, applies optional Meta-ML weights + constraints, writes `_autoranked.csv`/`.SUMMARY.txt` under `data/autorank_jobs/<job_id>/`, and launches the top-K candidates as realtime sessions. Costs = `10 + 50 * top_k` credits (debited upfront; if insufficient the call returns `402` and no sessions start). Response 201:

```json
{
  "autorank_id": "ar_20251113_223015_ab12",
  "artifacts_dir": "data/autorank_jobs/ar_20251113_223015_ab12",
  "autoranked_csv": "data/autorank_jobs/.../autoranked.csv",
  "summary_txt": "data/autorank_jobs/.../autoranked.SUMMARY.txt",
  "sessions": [
    {"rank": 1, "session_id": "sess_x", "source_csv": "data/experiments/portfolio_A.csv", "config_index": 17}
  ],
  "debited_credits": 60
}
```

Rules: CSV paths are sanitized (no `..`), meta models are optional, and idempotence ensures that resubmitting the same payload within 10 minutes returns `409` referencing the existing job id. Set `dry_run=true` to compute artifacts without charging credits or launching sessions.

`GET /v2/autorank/{job_id}` (scopes `read:usage` or `read:session`) returns the job status (`running/finished/failed`), session ids + live statuses (queried via SessionManager), and artifact paths:

```json
{
  "autorank_id": "ar_20251113_223015_ab12",
  "status": "finished",
  "sessions": [
    {"session_id": "sess_x", "status": "running"}
  ],
  "artifacts": {
    "autoranked_csv": "data/autorank_jobs/.../autoranked.csv",
    "summary_txt": "data/autorank_jobs/.../autoranked.SUMMARY.txt"
  }
}
```

### CLI Companion

`python scripts/autorank_to_sessions.py --csv data/experiments/AAA.csv --top-k 3 --meta-model artifacts/meta_ml/robustness_v0/model.joblib --session-interval 1m --session-threshold 0.6 --session-risk-pct 0.1 --kill-switch-dd 0.3 --dry-run`

Runs the exact orchestrator locally/CI without HTTP, writes the same artifacts, and prints the sessions launched. Use `--dry-run` to skip credit debits/session spawns.
