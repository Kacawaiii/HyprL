HyprL context package
---------------------

- AGENTS.md
- docs/PROJECT_BRAIN.md
- docs/hyprl_export_overview.md
- docs/V2_API.md

API v2
------
- api/app.py
- api/auth.py
- api/routers/v2_predict.py
- api/routers/v2_usage.py
- api/routers/v2_tokens.py
- api/repo.py
- api/models.py
- api/db.py
- .env.ops (redacted)

Bot & Discord
-------------
- bot/hyprl_client.py
- bot/commands/predict.py
- bot/session_store.py
- live/logs/discord_sessions.json (structure)
- scripts/ops/post_trade_events_to_discord.py
- src/hyprl/discord_templates.py

Live runners / health
---------------------
- scripts/run_live_and_notify_hourly.py
- live/logs/portfolio_live/health_asc_v2.json (structure)
