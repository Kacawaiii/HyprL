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
on startup, hourly, and before API/data access: seven days for unconfirmed requests,
24 calendar months from creation for confirmed signups.
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

## Espace membre v1

Member routes are disabled by default. See [member architecture, API and data
contract](docs/member/README.md) and the [operator deployment handoff](docs/member/DEPLOY.md).
No snapshot or real member credential is bundled; no Stripe/payment flow is active.

## Signup abuse controls

Production requires an independent stable `FORM_TOKEN_KEY` (32 random bytes,
64 lowercase hex characters), plus `SIGNUP_GUARD_FILE`, a private persistent
file outside code releases. Keep this file alongside external subscriber state,
with exactly one site writer. Losing it or changing the key loses suppression
history; an existing file with a different key binding refuses startup. The
release preflight checks external paths without opening personal state.

The self-hosted script obtains a signed IP-bound token (10-minute lifetime),
waits at least three seconds and solves a 16-bit SHA-256 puzzle. No tracker,
cookie or browser storage is added. POST requires the exact `PUBLIC_URL` origin;
missing/null/foreign origins, invalid/replayed proofs, honeypots and exhausted
quotas receive ordinary success with a separate private quarantine record and no email. Static HTML
has no token. Without JavaScript the form is disabled and offers Discord.

Rolling limits: 5 verified attempts/IP/hour, 30 globally/hour, 30 confirmation
emails globally/24 hours. Token minting has separate limits of 20/IP/hour and
300 globally/hour. These quotas and single-use token/address reservations are
persisted before SMTP and survive restart. Rejected fields are kept only in the separate private quarantine, never in subscriber or security state. IPs are keyed digests, not raw IPs;
expired entries are swept every minute and on guard operations. The guard file
contains only keyed digests, timestamps and counters. Confirmation reservations
remain for at most 24 calendar months, suppressing repeat email even after a
pending record or subscriber has been erased. No SMTP retry follows ambiguity.

Actions accept up to 10 comma-separated tickers/company labels (60 characters
each, 300 total), using letters, numbers, dots, hyphens and spaces. Multiword
company names use capitals and common lowercase connectors. Request phrases are quarantined with a French reason; the visitor sees ordinary success. This is a bounded lexical filter, not a
company registry or a guarantee that an entry names an actual company.

All newly accepted subscriber records are pending. One email offers an explicit POST confirmation
within 48 hours for early access and, only if separately selected, the newsletter.
GET/HEAD does not confirm. No reminder or automatic resend exists. Unconfirmed
records expire seven days after creation (startup/hourly/request-time purge).
Confirmed records retain the existing 24-month limit from creation. On first start, legacy records without early-access confirmation evidence move to quarantine, before subscriber retention runs. Known bot prose gets its specific reason; other legacy entries get an unverified reason. Their original date is preserved and their quarantine deadline is seven days from import; never mark them confirmed by inference. Duplicate submissions cannot enroll a new purpose or
undo a newsletter withdrawal. Changes require the operator contact channel.
SMTP disabled returns a clear error after proof validation, with no personal
record or address reservation, so enabling mail later does not strand addresses.

The site process checks hourly for a digest of confirmed signups from previous
UTC dates. At most one digest is attempted per rolling 24 hours. It reserves the
attempt and each included signup before SMTP; ambiguous sends are never retried.
Pending/legacy entries are never included. Existing newsletter weekly worker and
unsubscribe/erasure flows remain in place; it also requires confirmed early access.
No additional timer or service is needed for the digest. Logs contain counters
and fixed operational messages, never forms, email addresses, IPs or mail bodies.

`server/growth/merge-signup-privacy.js` updates signup behavior in the live privacy
page while preserving publisher/provider/member facts. Deploy using the existing
pinned archive, tests, preflight, code-directory swap and site-only restart
procedure. Keep SMTP disabled through deployment and rejected-bot acceptance;
the operator enables it only after reviewing the exact release. Rollback must
first disable SMTP because the previous code lacks these controls. Never restore
subscriber or guard state from a code rollback. An active service alone is not
public acceptance; use an operator-owned seed address for confirmation and erase
it afterwards. A custom solver can still pass the puzzle: quotas bound this risk,
and double opt-in prevents unconfirmed addresses receiving any follow-up.

Run the signup browser regression with the isolated growth fixture server:

```sh
python "$WEBAPP_TESTING_SKILL/scripts/with_server.py" \
  --server 'node test/growth-browser-server.js' --port 3818 -- \
  python test/antispam_browser_audit.py
```

It verifies the no-JavaScript Discord fallback, ordinary success for quarantined spam,
local proof of work under the strict CSP, a single combined confirmation,
newsletter withdrawal, erasure and suppression of another request after erasure.
All recipients and mail delivery are synthetic; external browser requests are
intercepted and refused.


### Admin triage (anti-spam release addendum)

`/admin` presents Vrais (confirmed email), En attente (confirmation reserved/sent;
uncertain delivery is shown explicitly), and Bots / suspects, each with a count.
Rows show the original date, address, channels and actions text. Suspects include
a plain French reason and their erasure deadline. `SIGNUP_QUARANTINE_FILE`
defaults to `DATA_FILE + '.quarantine.json'`; it must be a distinct private regular
file outside both code releases. It contains only bounded form fields, consent
evidence and triage metadata, never raw HTTP bodies, IPs or bearer tokens.
A repeated address does not renew its deadline. Startup, request and a dedicated
expiry timer purge it at seven days. State has one writer: the site process.

“C’est un vrai” requires recorded consent and a valid address, reserves the same
one-email/daily quota, creates a pending early-access entry, removes quarantine,
and attempts that one confirmation. It never creates newsletter membership or
marks an email confirmed. Review timestamp, reason, reservation and delivery
outcome are persisted; logs contain counters only. Missing consent, SMTP disabled,
quota/suppression or a duplicate subscriber leaves the suspect untouched with a
French error. An uncertain send is never retried. Copy-before-remove migration
and interrupted promotion recovery finish by identity on startup, with no resend.
“Supprimer” erases a suspect. CSV and bulk newsletter exports include only
confirmed entries; quarantine is never used for marketing or a digest. The
privacy policy and internal processing register disclose legitimate interest and
the seven-day limit. No real store is read during release preparation or tests.


Run the admin regression against its isolated synthetic import/mail fixture:

```sh
python "$WEBAPP_TESTING_SKILL/scripts/with_server.py" \
  --server 'node test/triage-browser-server.js' --port 3819 -- \
  python test/triage_browser_audit.py
```

This covers the three tab boundaries/counts, delayed tab responses, French reasons,
light/dark accessibility, keyboard/mobile reflow, text escaping, confirmed-only
CSV, import, review, explicit confirmation and deletion. It refuses every external
browser request. Node tests also cover exact seven-day expiry (including the timer),
restart/interruption recovery and one-email reservations through uncertain SMTP.
