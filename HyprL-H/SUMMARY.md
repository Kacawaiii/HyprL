HyprL distilled context
========================

What this bundle covers
-----------------------
- Agent rules: AGENTS.md (behavior, logging, scope).
- Project state: PROJECT_BRAIN.md (historical log, ops notes).
- API contract: V2_API.md (routes, auth, credits).
- Export/workflows: hyprl_export_overview.md (flows, presets).

Core flows (concise)
--------------------
- Auth/credits: Bearer tok_xxx with scopes; /v2/usage checks read:usage, /v2/predict requires read:predict, rate-limit + debit.
- Predict: /v2/predict → symbols, interval, threshold, risk_pct; returns results + meta; stub by default, real via HYPRL_PREDICT_IMPL=real; records UsageEvent + Prediction rows.
- Sessions: /v2/sessions (write:session) start; status/report endpoints read usage/session scopes.
- Tokens: /v2/tokens admin-only; fallback admin tok_admin.hyprl_admin_dev_123 dev-only.

Discord/bot
-----------
- bot/commands/predict.py: builds embed from /v2/predict payload (direction, prob, threshold, risk, TP/SL, status).
- bot/hyprl_client.py: HTTP client with bearer token + friendly errors.
- Session routing: bot/session_store.py + live/logs/discord_sessions.json (add plan/name/channels).
- Poster: scripts/ops/post_trade_events_to_discord.py + src/hyprl/discord_templates.py (trade/health embeds, idempotence, DLQ).

Live runners
------------
- scripts/run_live_and_notify_hourly.py orchestrates live runs → concat → monitor → Discord posts.
- Health JSON: live/logs/portfolio_live/health_asc_v2.json consumed by poster for PF/DD/Sharpe/trades status.

Trade gates / aggregation
-------------------------
- src/hyprl/analysis/trade_aggregator.py (load/normalize trades, metrics, rolling PF/MaxDD).
- src/hyprl/analysis/trade_gates.py (Gate1/Gate2 checks).
- scripts/analyze_strategy_gates.py (CLI: status CSV + optional reports).
- tests/analysis/test_trade_gates.py (synthetic gate coverage).
- docs/AI_CONTEXT.md (état courant NVDA 1h long-only v0 : fast natif 0.6 s, PF_gate~17, MaxDD_gate~1.7 %, Sharpe_gate~0.95, trade-count <300 bloquant Gate1; décider Gate1 spécial ≥150 trades pour ce strategy_id; prochaine étape = modèle v2 21 features équilibré long/short).

If adding more files
--------------------
- Include api/app.py, api/auth.py, api/routers/v2_predict.py, api/routers/v2_usage.py, api/routers/v2_tokens.py, api/repo.py, api/models.py, api/db.py for full backend code.
- Include scripts/ops/run_live_and_notify_hourly.py if sharing runner logic; redact .env.ops tokens.
