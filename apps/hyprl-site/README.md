# HyprL site

Standalone Node.js >=20.6 site imported without runtime data, dependencies,
deployment configuration or operator contact details. Fonts are self-hosted.
No payment, alert delivery or trading endpoint is enabled.

```sh
npm ci
npm test
```

Use `.env.example` as a configuration checklist. Keep the populated environment
outside Git, set a long `ADMIN_PASSWORD`, and launch with:

```sh
node --env-file="$PRIVATE_SITE_ENV" server/server.js
```

`NOTIFY_EMAIL` has no code default and supplies the runtime `/contact` page.
It must be populated by the operator for rights/security requests. Do not put
an address in source, examples, reports committed to Git or page markup.

## Data handling

The single-process JSON store is private runtime data (`DATA_FILE`), never source.
Create its directory with mode 0700 and restrict an existing file to 0600. Do not
run two writers against one file. Mutations are serialized and atomically saved;
an invalid store stops startup instead of being silently replaced. Purging runs
on startup, hourly, and before API/data access, with a 24-calendar-month UTC cutoff.
Any purge failure prevents data access through the API. Backups and mailbox copies
require an operator retention policy; a restored store is purged before access.

Admin → person → **Exporter JSON** produces a single-person, token-free portability
file. Verify the request and recipient first, deliver privately and remove local
working copies. **Supprimer** and the confirmation/unsubscribe POST erase the active
record. JSON dates use ISO 8601 UTC; Paris time is presentation-only in the UI/mail.

SMTP delivery is disabled unless `SMTP_EU_APPROVED=true` **and** SMTP is configured.
The flag records the operator's approval; it cannot establish residency or a DPA.
Before enabling, choose an EU-only provider, verify its DPA/subprocessors, update
the public policy with its identity, and manage retention of notifications/copies.

## Cookies and optional code

Public visitors receive no cookie, localStorage or sessionStorage entry. The admin
session cookie is necessary, HttpOnly, SameSite=Strict, Secure with HTTPS; sessions
expire after 30 idle minutes or 12 hours total. No consent banner is shown.

`public/consent.js` is intentionally disabled (`ENABLED=false`). Before adding any
optional resource, review the processor, purposes, transfers, policy and CSP; then
register **every** optional script/iframe/resource in a purpose-specific `load`
callback and provide cleanup. Never add a direct optional `<script>`, iframe,
preload, pixel or preconnect elsewhere. Only reviewed origins can be added to CSP.
Activation is a deliberate code/configuration review, not a public query parameter.

The module has unselected purpose boxes, equal accept/refuse buttons, a persistent
settings control, and consent-gated loaders. Both decisions last six calendar months,
then require a new choice, earlier than the absolute thirteen-month backstop. Expired,
corrupt or unavailable stored choices fail closed. Withdrawal reloads the page to stop
already executed code; tabs synchronize changes. Third-party stored data must also be
removed using the chosen provider's documented cleanup and deletion mechanisms.

## Compliance material and contracts

Public: privacy/Cookies, legal notice, draft future CGV, accessibility declaration,
contact, security policy and future cancellation entry point. Placeholders must be
completed by the operator. No paid subscription is sold; a real authenticated online
cancellation flow and digital-access consent evidence are prerequisites for sales.
No certification or unperformed audit is claimed.

`docs/internal/` contains non-served register and breach-response **templates**.
Maintain completed versions and contracts privately. `docs/ACCESSIBILITY.md` records
the limited self-audit; a full RGAA audit remains outstanding.

OpenAPI 3.1 / JSON Schema 2020-12 contracts for the future API and signed webhooks
are under `public/contracts/v1/`; see `docs/WEBHOOKS.md`. HMAC-SHA256, a 300-second
replay window, full schema validation and event-ID reservation are tested reference
behavior, not an enabled service. Production receivers need a shared atomic TTL
store and durable business idempotency. `/api/v1/alerts` currently returns 404.

API errors use RFC 9457 `application/problem+json`. Discovery routes include
`/.well-known/security.txt`, `/robots.txt`, `/sitemap.xml`; HSTS is retained for HTTPS.

## Optional browser audit

Use an isolated Python environment with `test/requirements-qa.txt` and an existing
matching Chromium installation. The audit uses only loopback and synthetic data:

```sh
python -m pip install -r test/requirements-qa.txt
python "$WEBAPP_TESTING_SKILL/scripts/with_server.py" \
  --server 'node test/browser-server.js' --port 3817 -- \
  python test/browser_audit.py
```

The script runs axe WCAG 2.1 A/AA checks in light/dark modes, keyboard/mobile/reflow
checks, sign-up/admin/export/delete and disabled/opt-in consent flows. No real SMTP
or live data is loaded. The operator alone deploys after reviewing the private
handoff report; this checkout does not alter the live service.
