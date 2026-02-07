# Discord Server Monetization Plan (HyprL)

Objective: convert Discord community traffic into recurring revenue while staying compliant.

## Value Ladder (Plan -> Role -> Access)
- Free (Member): announcements, public insights, FAQ, community chat.
- Beta (EUR 19/mo): read-only signals, weekly recap, basic Q&A.
- Starter (EUR 29/mo): signals + monthly report + Q&A.
- Pro (EUR 79/mo): realtime alerts + API token + priority support.
- One-off: strategy audit (EUR 150-300), consulting (EUR 75-150/h).

## Server Structure (Categories/Channels)
START-HERE
- #welcome
- #rules
- #how-to-upgrade
- #faq

FREE
- #announcements (read-only)
- #public-insights
- #community-chat

SIGNALS (READ-ONLY)
- #signals-beta
- #signals-starter
- #signals-pro
- #trade-recaps

PRO SUPPORT
- #pro-support
- #api-help

OPS
- #mod-log
- #billing-ops

## Roles + Permissions
- @Visitor: read START-HERE only.
- @Member: access FREE.
- @Beta: access #signals-beta + #trade-recaps (read-only).
- @Starter: access #signals-starter + Q&A.
- @Pro: access #signals-pro + PRO SUPPORT.
- @Staff/@Bot: moderation + automation only.

Rules:
- Signals channels are read-only for all paid roles.
- No @everyone on paid channels; use role mentions only.
- Keep upgrade CTA in #how-to-upgrade and welcome message.

## Monetization Flow
1. Join Discord -> auto welcome + CTA to landing page.
2. Stripe checkout -> collect Discord ID (form field).
3. Manual role assignment while webhook is not live.
4. Webhook automation: payment event -> role grant -> API token (Pro).
5. Renewal/cancel -> role update + access revoke.

## Automation Hooks (Repo Touchpoints)
- Stripe setup: `docs/STRIPE_SETUP.md`
- Role grant/revoke checklist: `docs/BILLING_TODO.md`
- Discord posting: `scripts/ops/send_discord_bot_message.py`
- Bot commands: `bot/`
- Discord token resolve: `src/hyprl_api/v2.py` (`/v2/token/resolve-discord`)

## Content Cadence
- Daily: 1 public insight + 1 paid signal summary.
- Weekly: performance recap + risk notes.
- Monthly: roadmap update + Q&A session.

## KPIs
- Join -> upgrade conversion (%).
- MRR, churn, ARPU.
- Time-to-first-signal (activation).
- Weekly active paid members.

## Compliance Notes
- Always include "not financial advice" disclaimer.
- Use real backtests only; no fabricated performance.
- Keep paid content informational, no personalized advice.
