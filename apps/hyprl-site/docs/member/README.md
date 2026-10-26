# Espace membre v1

French member workspace for hyprlcore.com, served by the existing Node site.
No animation, remote font, third-party script, analytics, brokerage connection,
position sizing or account performance chart. IBM Plex and uPlot 1.6.32 (MIT)
are served locally. The public site remains usable with members disabled.

## Access and routes

Discord OAuth2 uses `identify guilds.members.read`, a five-minute single-use
state cookie, SHA-256 state storage, S256 PKCE and code exchange on the server.
The secret is environment-only. No Discord e-mail, message, avatar or username
is requested or retained. The guild/user identity and provider credentials live
in memory for the session. Persisted watchlists use a stable HMAC identifier.

| Role | Features |
| --- | --- |
| Accès anticipé | Dashboard, verified claims archive, account |
| Essentiel | Above + daily brief/archive, watchlist radar, calendar |
| Premium | Above + duel/proofs, luck, factual revenue screener, alert configuration |
| Fondateur | Premium features |
| Equipe | Premium features |

Set five distinct Discord role IDs in the private environment. The highest
recognized role wins. Each protected HTML/API request retrieves current guild
roles from Discord; provider failure denies access, with no stale role fallback.
Leaving the guild/removing every recognized role invalidates the session.
The server enforces every tier before reading a snapshot or contacting the
alert bridge. Discord limits/outages can therefore temporarily deny all access.
No bot token or privileged guild-members intent is needed for OAuth.

The external callback initially cannot receive a Strict cookie. It serves a
static same-origin completion page, removes code/state from browser history,
then POSTs them to the server with the Strict transaction cookie. The server
requires the exact configured Origin. Do not replace this with a direct GET
exchange or loosen SameSite. The callback has no reflected code/state and no
external asset. Configure ingress so callback query strings, authorization
headers and request bodies are never logged.

Session identifiers are random 256-bit values; only SHA-256 digests index the
in-memory session map. Cookies are HttpOnly, SameSite=Strict, Path=/ and Secure
with `__Host-` prefixes over HTTPS. Thirty-minute idle and twelve-hour absolute
expiry match the admin session policy. Restart invalidates sessions. OAuth
access/refresh tokens are never sent to the browser. State-changing API calls
require both exact Origin and a per-session CSRF token from `/session`.

Bounds: OAuth 10 start/completion operations per 15 minutes per IP (five full
logins); 240 member requests/minute/IP; 120 protected requests/minute/member.
Counters and session/transaction maps are bounded. Only a loopback trusted proxy
can supply a syntactically valid `X-Real-IP` when `TRUST_PROXY=true`. Configure
the ingress to overwrite that header. Otherwise the remote socket address is
used, and a shared proxy IP shares its budget.

Pages: `/membre`, `/membre/brief`, `/membre/radar`, `/membre/calendrier`,
`/membre/duel`, `/membre/chance`, `/membre/screener`,
`/membre/chance-ou-pas`, `/membre/alertes`, `/membre/compte` and
`/membre/preuves?hash=<digest>`. Lower-tier HTML is a 403 shell explaining the
required access; its APIs remain 403. Anonymous HTML redirects to Discord login.

The complete resource/verb contract is in [API.md](API.md).

## Data boundary and publishing job

The web server reads **one file**, `member-v1.json`, in `MEMBER_SNAPSHOT_DIR`.
It never imports Python research/trader code, opens those stores, computes a
portfolio or fetches SEC/Fed data. Snapshots have a closed JSON schema, size
limit, provenance, UTC export time and per-item source URL/digest/availability.
They cannot be reached through a static URL. Invalid, future, symlinked or
missing snapshots fail closed. Exports older than 26 hours are labeled old.
Empty feeds say **En cours de constitution**; no production seed data exists.
A synthetic snapshot is refused by the production reader and exporter CLI.
Only test dependency injection accepts synthetic data, with a visible banner.

The offline job `python -m scripts.member_export.export` accepts a **dedicated
editorially approved producer handoff**, not a raw database, transcript or
account export. `handoff-v1.schema.json` is its contract; `snapshot-v1.schema.json`
is the web contract. All objects reject unknown fields. The exporter refuses
private/advisory vocabulary, credentials/addresses/paths, stale/future handoffs,
noncausal availability, unscored/unrevealed forecasts, incorrect commitments,
duplicates and incomparable revenue intervals. This lexical guard does not
replace editorial review for personal data, advice or misleading claims.
The CLI has no network or synthetic mode. Publication is an fsynced atomic
replacement, file mode 0440. Only digest/count evidence is printed.

Operator job command (variables point to dedicated external runtime locations):

```bash
cd "$MEMBER_CHECKOUT"
"$MEMBER_PYTHON" -m scripts.member_export.export \
  --handoff "$MEMBER_HANDOFF_FILE" --output-dir "$MEMBER_SNAPSHOT_DIR"
```

Schedule this command in the operator's existing approved job runner after the
producer has atomically written the handoff. No timer or live producer was
installed by this branch. The handoff must already have a current UTC
`generated_at` and `provenance=observed`; never relabel synthetic data as observed.
Existing account returns, paper/replay P&L and internal model IDs have no field
in this contract. Do not copy private or frozen stores to make a feed appear.
The actual upstream mapping, receipts and editorial acceptance remain required
before calling the feeds live.

Forecast commitments contain only public agent, public variant, session,
symbol, `close_above_open` event, probability and base rate as decimal strings,
commit/reveal UTC times and a public nonce. Publish the SHA-256 of the exact
UTF-8 canonical form **before** the predicted period; canonical form has sorted
ASCII keys, no whitespace and JSON string values. The outcome and source become
available after reveal. Proof pages recompute that digest in the browser and
link to the original publication receipt. Digest equality proves content
identity, not receipt timestamp validity. Independent receipt verification is
an upstream/operator prerequisite; this app does not fetch receipt URLs.
Existing trade-decision commitments cannot be translated into forecast hashes
and presented as the original proof. New producer support is required.

Per agent/variant, ordered scored observations produce:

- Cumulative Brier skill `1 - sum((p-y)^2) / sum((base-y)^2)`. Undefined when the
  denominator is zero. Negative skill is retained and displayed.
- Hit rate of thresholded forecasts (`p >= .5`) over the last 20 observations,
  using a shorter window at startup. Failures have a separate explicit register.
- Luck tail `P(K >= observed hits)` from an exact Poisson-binomial distribution.
  Baseline hit chance is `base` for an up forecast and `1-base` otherwise.
  This assumes independence and does not correct dependence, variant selection
  or multiple testing. It is neither a return nor evidence establishing skill.

The screener uses three consecutive nonoverlapping 70–110-day quarters of the
same SEC revenue concept and currency, positive prior denominators, ten-digit
CIK strings and canonical SEC Archives source links. It computes sequential
revenue growth in percent and the difference in percentage points. It retains
negative values, sorts alphabetically, and warns about seasonality/acquisitions.
There is no recommendation or financial return ranking. Producers must resolve
restatements, filings, units and quarterly versus cumulative XBRL facts before
handoff; the exporter does not infer them from a raw SEC body.

## Private alert router API

`server/member/router-client.js` talks only to `MEMBER_ROUTER_SOCKET`, with a
separate environment HMAC key. The bridge `scripts.alert_router.member_bridge`
serves POST `/member/v1` on a mode-0600 Unix socket in a mode-0700 directory.
No TCP listener. Signature covers timestamp, 256-bit nonce and exact JSON body;
30-second skew, nonce replay cache, bounded bodies and fixed error codes.
The web process never opens the router key file, database or destination secrets.
The authenticated Discord session supplies `guild/user`; the browser cannot
supply customer, member identity, tier or policy. Router customer allowlist,
Telegram chat approval and ownership checks remain authoritative. Destinations
are encrypted by the existing Store. Lists omit addresses and signing keys.

Actions: list, add, pause/resume, remove. Discord, Slack, Telegram, Teams and
custom HTTPS webhook adapters are supported. Member e-mail collection is
**disabled** until an approved sender exists; no pretend delivery. No test
message, real outbound call or worker starts when a destination is added.
The existing worker performs SSRF/DNS/TLS/rate/retry checks on each delivery.
The member watchlist and each destination's filter are separate, labeled
preferences; editing one does not silently change the other.

## Account and future Stripe hook

No Stripe SDK, payment, inbound webhook or checkout is present. Discord remains
the only authority for access. The private hook
`MemberState.applyBillingUpdate(memberHmac, {renewal_at, manage_url, cancel_url})`
accepts only links on the configured exact HTTPS billing origin. It cannot grant
a tier. Portal URLs never enter the shared snapshot and are returned only to the
corresponding authenticated member. With no billing record, the account clearly
states payment is inactive and no renewal is known. No date is invented.

Future integration: verify the billing provider's webhook signature and replay
ID **outside this HTTP namespace**, resolve its customer to a verified Discord
identity, grant/revoke the configured role through an approved operator-owned
bot integration, compute the same `memberKey` using the private stable identity
key, then apply the billing update. Keep a durable event/idempotency register.
Do not accept a role, customer or portal URL from an unsigned member request.
Populate a working direct cancellation URL before any subscription sale. From
any member page, Compte → Résilier is two clicks; the provider's final confirmation
must fit the third click. No actual paid cancellation is claimed in this branch.
The operator must test the final provider journey before opening sales.

Watchlists/billing metadata are in a separate mode-0600 member state file,
with serialized atomic mutations. Records are purged at startup, on reads/writes
and each minute after 24 calendar months from their last modification; reads do
not extend retention. Private `get(memberHmac)` / `remove(memberHmac)` hooks
support verified rights requests. Router destinations need separate erasure
using its customer-scoped remove API and operator backup/processor checks. Single site process/bridge instance only.
Identity keys must stay stable across releases; rotation requires a reviewed
private migration and cannot be silently substituted. Backups and rights
exports/erasure remain operator-owned; never restore erased records.

## Verification and screenshots

```bash
npm --prefix apps/hyprl-site ci --ignore-scripts
npm --prefix apps/hyprl-site test
"$MEMBER_PYTHON" -m pytest tests/member tests/alerts -q
```

Browser QA uses the test-only bootstrap and boundary doubles, with every
non-loopback request blocked. It exercises the actual app, OAuth completion,
tier denial, watchlist form, proof recomputation, alert add/pause/resume/delete,
logout, responsive charts, theme toggle, skip link and axe checks.
Install the pinned `test/requirements-qa.txt` into isolated scratch; add router
Discord test dependencies separately. With Playwright's cached Chromium:

```bash
PYTHONPATH=apps/hyprl-site/.qa-deps "$MEMBER_PYTHON" \
  /path/to/webapp-testing/scripts/with_server.py \
  --server 'node apps/hyprl-site/test/member-browser-server.js' --port 3827 \
  --server 'node apps/hyprl-site/test/member-browser-server.js --empty' --port 3828 \
  -- "$MEMBER_PYTHON" apps/hyprl-site/test/member_browser_audit.py
```

The checked-in `screenshots/` are explicitly synthetic offline QA, not actual
performance evidence. `duel-empty-dark.png` shows the pending-data state.
No browser screenshot from a real member or private store is included.
Live OAuth, real messaging, producer freshness, proxy behavior, final legal
facts, provider cancellation and a complete human accessibility audit still
need operator acceptance. See `DEPLOY.md` for the staging/release sequence.
