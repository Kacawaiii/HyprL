# Member HTTP API v1

Namespace `/api/member/v1/`. All responses are `Cache-Control: no-store`,
`X-Robots-Tag: noindex, nofollow`; no CORS. Authentication is the necessary
HttpOnly member cookie. Every resource rechecks current Discord roles on the
server. Writes require exact `Origin: <PUBLIC_URL origin>` and `X-CSRF-Token`
from `GET /session`. No user/customer/tier field is accepted from a member.

Errors are RFC 9457 `application/problem+json` with fixed error codes and no
provider details. 401: reconnect; 403: role/tier/origin/CSRF; 404: unavailable
resource/proof; 405: verb; 413: oversized body; 415: JSON required; 429: rate
budget (`Retry-After`); 503: Discord/export/router/state unavailable. A 503
is never transformed into a successful fabricated feed.

| Method / path | Minimum | Shape |
| --- | --- | --- |
| GET /session | Accès anticipé | tier, French tier label, CSRF, authorized features |
| POST /logout | Valid session | `ok`; destroys session, expires cookie; CSRF/Origin required even after role revocation |
| GET /dashboard | Accès anticipé | export UTC time/provenance/stale flag, or null metadata pending |
| GET /claims | Accès anticipé | verified claims archive |
| GET /account | Accès anticipé | own tier + private billing metadata or null |
| GET /briefs | Essentiel | dated daily brief + archive |
| GET /radar | Essentiel | own watched issuers + Fed/CPI/NFP |
| GET /calendar | Essentiel | same watchlist filter, scheduled event times |
| GET /watchlist | Essentiel | own `symbols` array |
| PUT /watchlist | Essentiel | exactly `{symbols: [<uppercase symbol>]}`; max 50, replaces only own list |
| GET /duel | Premium | revealed scored forecasts, failures, cumulative series |
| GET /proofs/:sha256 | Premium | one public forecast commitment, canonical text, receipt link and equality result |
| GET /luck | Premium | per-agent/public-variant luck series |
| GET /screener | Premium | factual SEC quarterly revenue acceleration |
| GET /alerts | Premium | own opaque destination IDs/channel/state/filter; no URL/key |
| POST /alerts | Premium | exactly channel, config, kinds, watchlist |
| PATCH /alerts | Premium | exactly `{destination: <opaque id>, paused: true/false}` |
| DELETE /alerts | Premium | exactly `{destination: <opaque id>}` |

Fondateur/Equipe inherit Premium. There is no public member admin/list API.
Snapshot data shapes are defined in `snapshot-v1.schema.json`; API collections
carry the same fields plus export metadata, without exposing the complete
snapshot. Dashboard does not expose premium counts or contents. The web server
never serves the snapshot file directly. All source URLs are inert links until
opened by the member, with no referrer to the target. All textual data is rendered
with `textContent` rather than HTML injection. Charts have equivalent data tables.

Alert body example, entirely synthetic; never deploy this destination:

```json
{
  "channel": "webhook",
  "config": {
    "url": "https://example.com/synthetic-receiver",
    "signing_key": "synthetic-key-for-documentation-only"
  },
  "kinds": ["sec_filing", "fed_release", "weekly_luck"],
  "watchlist": ["DEMO"]
}
```

Discord/Slack/Teams configs contain `url`; custom webhook also needs a 32–128
ASCII-character signing key. Telegram config is `{chat_id: "<approved id>"}`
and requires approval in the router policy. The member UI subscribes a destination
to all ten supported informational kinds; the API also accepts a chosen subset.
E-mail/API destinations are refused until the router's deferred capabilities
exist. Adding a destination does not send a message. Repeated add requests can
create multiple destinations up to the router's cap; do not retry blindly after
an ambiguous timeout. Removal deletes its pending sends, but cannot retract an
already in-flight message. The original router's delivery semantics still apply.

OAuth routes: GET `/auth/discord/start`, GET `/auth/discord/callback`, POST
`/auth/discord/complete` with exactly code/state. They are deliberately separate
from resource authorization. The callback's same-origin completion transaction
is required for Strict cookies. Their credential bodies and callback query
strings must never be logged by ingress or APM.
