> Integration update: see `docs/GROWTH_INTEGRATION.md`. It supersedes the original new-application and no-completed-study assumptions below.

# Growth engine v1 — operator handoff

This branch prepares code and user-service templates. It does not deploy,
contact Discord/Brevo/official sources, register subscribers, alter another bot,
or enable payments. Production activation belongs to the operator.

## Dedicated Discord referral service

`python -m scripts.growth.discord_bot` reuses the **existing HyprL Discord application**, as explicitly directed by the operator. It is a separate process and ledger; it never modifies kyoko-bot.
Install `scripts/growth/requirements.txt` in a dedicated virtual environment.
Render `scripts/growth/deploy/hyprl-growth-referrals.service.in` with absolute
release/Python paths, and install it as a **user** unit. Do not edit any existing
bot, trader, proxy or system unit. Environment example:
`scripts/growth/deploy/referral.env.example`; actual values remain outside Git.
`python -m scripts.growth.render_units --help` describes the offline renderer;
it writes only the three new growth units to an empty staging directory.

The existing HyprL application has Guild Members intent enabled. Do not create another application. Its identity environment is loaded separately by the unit via `--discord-env`; never copy or print its token. Registration upserts only `/parrainage`, preserving all other application commands.
Install with scopes `bot` and `applications.commands`, restricted to the HyprL
guild. Grant Create Invite on the invitation channel, Manage Guild to read
invitation counters, View Channel/Send Messages on the leaderboard channel, and
Manage Roles. Its highest role must be above both dedicated reward roles.
Administrator, message-content intent and access to other channels are unnecessary.
Use an empty, mode-0700 private ledger directory and an absolute SQLite file path.
The service holds a process lock and SQLite transactions protect reward issuance.

`/parrainage` replies privately, reuses one stored invite, and creates a new link
with `max_age=0`, `max_uses=0`, `unique=True` only for a new participant. A deleted
link is not silently replaced. One use on exactly one invite, considering **all**
guild invites, is required for attribution. Concurrent ambiguous arrivals,
counter resets, deleted invites and reconnect gaps are never guessed. These
conditions can undercount referrals; there is no retroactive attribution tool.

A member must already be seven days old at arrival and stay seven complete days.
Bots, self-invites, repeat arrivals and departures before that boundary are
excluded. Reconciliation fetches current membership and checks the original join
time, detecting leave/rejoin during downtime. API outages abort reconciliation.
Every five minutes it matures eligible arrivals and grants dedicated reward roles:

- Three valid referrals: a `premium_month_credit` reward and **one** queued month
  in `credits`, with state `pending_stripe`. The marker role does not imply a
  paid subscription. No debit, cash payout or Stripe API call occurs.
- Ten valid referrals: a `fondateur_place` reward and the dedicated place role.
  Do not map this role to an unrelated paid entitlement. The product decision
  is a reserved place, not an invented count of remaining public sale places.

Each threshold is issued once per participant record. Role writes are idempotent
and retried if they fail. Repeated reconciliations never create another credit.
A future Stripe adapter must resolve a verified Discord-to-customer mapping,
consume `pending_stripe` with a stable provider idempotency key and record actual
application. That adapter is **not done** because the product has no Stripe
integration yet. Never mark a credit applied based only on a role.

The weekly cumulative top ten starts Monday 09:00 UTC, with catch-up after a
restart. The output contains only HMAC-derived `Parrain-…` aliases and valid
counts, no Discord names/IDs/mentions/invite codes. A durable reservation avoids
duplicate posts; an uncertain send needs private review rather than blind retry.
This is pseudonymisation, not a claim of irreversible anonymity.

Records expire after 24 calendar months; repeat visits do not extend retention.
`secure_delete` is enabled. The operator must revoke obsolete/erased invites and
remove obsolete marker roles in Discord as part of the rights workflow. Old
backups must obey deletion/expiry and must never be restored without a purge.
For individual rights, **stop this one service**, export to an external private
file or erase locally, then restart:

```sh
python -m scripts.growth.admin --ledger "$REFERRAL_LEDGER" --member "$MEMBER_ID" \
  --export-to "$PRIVATE_EXPORT_FILE"
python -m scripts.growth.admin --ledger "$REFERRAL_LEDGER" --member "$MEMBER_ID" --erase
```

Exports exclude other referred members. Erasure vacuums SQLite. No contact
address, real identity, store, invitation or token belongs in Git or public logs.

## Independent newsletter consent

The original early-access purpose/version is preserved. The separate optional
checkbox has version `2026-10-newsletter-v1`, is unticked, and stores its exact
text, requested/confirmed/withdrawn timestamps independently. Choosing the email
alert channel never subscribes someone. Existing records stay unsubscribed unless
their holder explicitly requests and confirms the letter.

Set these in the existing **private site environment**, with two independent
cryptographically random 32-byte hex keys:

```dotenv
NEWSLETTER_ENABLED=true
NEWSLETTER_TOKEN_KEY=<private stable key>
NEWSLETTER_WORKER_KEY=<different private key>
NEWSLETTER_TRACKING_DISABLED=true
PUBLIC_SNAPSHOT_DIR=<absolute existing sanitized member snapshot directory>
```

Keep the token key stable: it derives per-consent unsubscribe links. Rotate the
worker key independently. `NEWSLETTER_TRACKING_DISABLED=true` is an operator
attestation: first disable **both open and click tracking in Brevo**, including
transactional defaults, then check a privately received seed message's links.
Plain-text mail has no HTML, pixel or third-party resource. The code also asks
Brevo not to track via SMTP headers, but those headers alone are not proof of
provider behavior. Existing `SMTP_EU_APPROVED` and TLS gates remain mandatory.
Review Brevo DPA, actual processing locations and public privacy facts before
activation; no source, provider or legal review was performed over the network.

The confirmation email is transactional, requested only by the explicit optional
choice. Its random token is stored as a hash, expires in 48 hours and requires a
POST from its confirmation page. Link-scanner GETs/HEADs never confirm or withdraw.
Duplicate pending requests are throttled for one hour. A fresh request invalidates
the old confirmation token. Confirmed recipients are not mailed another DOI.

Every weekly mail has a separate letter unsubscribe URL and RFC 8058 headers.
POST withdrawal preserves early access; the original general deletion link
erases both. Admin list and individual export show the newsletter state/evidence.
Authenticated `/api/admin/newsletter/export` downloads all consent states as JSON,
without tokens/hashes/delivery internals. Keep exports private and short-lived.

The existing HTTP process is the **sole writer** of subscriber state. A user timer
runs `server/growth/send-weekly.js`, which authenticates to a loopback-only POST
with the worker key. It never opens/writes the store itself. Render the newsletter
unit with the site path/env/Node executable. The timer is Monday 18:00 Europe/Paris
and catches up after downtime. `systemctl --user start hyprl-growth-newsletter`
**sends real emails** after activation; never use it as a harmless smoke check.

Eligibility is checked again per recipient. Only confirmed current-version consent
can receive a letter. The edition ID is the New York Monday date; a reservation
per subscriber/edition is durably written **before SMTP**. Success, uncertainty,
concurrent requests and restarts never send that edition again automatically.
An interrupted `reserved` or `uncertain` record needs private delivery review.
This provides at-most-once attempts, not guaranteed delivery or an SMTP receipt.
Withdrawal cannot recall a message already accepted by SMTP. No BCC address list,
tracking measurement or public recipient log is created.

All newsletter data inherits the existing 24-calendar-month inscription retention.
Enable/disable is reversible; `NEWSLETTER_ENABLED=false` refuses enrollment/sends
but existing confirmation and withdrawal links continue to work with the stable
token key. No legacy contact import or automatic bulk confirmation exists.

## Public, server-rendered French content

- `/calendrier-resultats-us`: current New York week from the read-only v2 export.
  Shows only earnings rows with real symbols/dates. Observed day markers remain
  explicitly **unconfirmed by the issuer**; they never acquire an invented time,
  source or Event schema. Publication time is not interchangeable with event time.
- `/chance-ou-pas`: one deterministic article URL per sourced claim present in the
  validated export. Latest verified-at record wins; no future/noncausal claim or
  fictitious verdict is generated. Missing claims produce an honest empty index.
  The article reports the recorded verdict/source and limits; no invented causal
  analysis or probability of luck is added. The luck counter remains pending.
- `/comprendre-depot-sec`: educational 8-K, 10-Q and Form 4 guide with original
  SEC-form links. These links were not fetched on this restricted host.
- `/parrainage`: public reward, validation and privacy rules, linked from home.

The site reuses the existing validated v2 reader/hash binding. It reads only the
sanitized member export, never research/trader stores or official sources.
Production refuses synthetic data. Exports older than 26 hours, invalid schemas,
unsafe permissions/symlinks and future timestamps do not become fresh facts.
Unavailable dynamic pages return an explanatory 503/Retry-After; absent article
IDs return 404. A newer export does not alter an event's original source clock.

Every growth page has a unique title/description, French semantic HTML, canonical,
Open Graph, JSON-LD with a per-response CSP hash, and a sitemap entry. No inline
executable script or third-party content is enabled. Token/admin/API routes stay
out of the sitemap. Home links the useful pages; robots excludes token paths.
No ranking, traffic lift, rich-result eligibility or external indexation is claimed.

The weekly letter uses that same validated factual projection: this week's
delivered brief facts, calendar markers and sourced claims. It refuses empty,
unavailable or stale content. It does not invent a weekly issue to fill a timer.
Operator source production must continue feeding the existing export; no new
source collector or authorization is added here.

## Validation and release

Run the repository's mandatory Python gate, `npm ci && npm test` in
`apps/hyprl-site`, `pytest tests/growth` with the pinned Discord dependency, and
the synthetic browser audit described in
`apps/hyprl-site/docs/growth/screenshots/README.md`. Run heavy suites sequentially.
The existing cockpit is unchanged, so its conditional web gate is not applicable.

Prepare pinned code releases with `git archive`, install/test in staging, and
review actual legal publisher/processor facts from the current site. Do not copy
old privacy HTML over the new newsletter provisions, and do not commit private
contacts. Keep all data/env outside code releases. Stop only `hyprl-site` for its
code-directory switch; then restart and check public routes without subscribing
a real address. Install only the new growth user units after configuration.
Rollback stops the two new growth services/timer and restores **code only**;
never restore deleted subscribers or an older ledger/consent file. Keep the
current private state and the token key for ongoing withdrawals.

The private operator report contains the pinned commit and host-specific commands.
