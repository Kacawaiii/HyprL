# HyprL information alert router v1 — operator runbook

This delivery builds an isolated outbound router. It does not deploy, open a
port, register commands, modify another bot, collect official data or change
any trader runtime. Read [the French customer guide](ALERTES_CLIENT_V1.md) and
[the payload schema](alerts/payload-v1.schema.json).

## Release boundaries

Staff destination tests require the operator to resolve the host's network
restriction first: it currently permits git/pip/npm only. Official collection
always needs a separate current operator authorization naming exact scope,
budget and expiry. The router has no collector or official-source fetch path.
Never use an official-source URL as a destination. Start with synthetic tests
in `#alertes-systeme` and separately confirmed staff Slack/Telegram destinations.

No customer has been enabled by this branch. The service requires a nonempty
operator allowlist as well as the exact Discord role ID. Policy changes require
restarting both processes; Discord role checks apply on every command, modal
submission and pause-button click. The single-owner rule reserves activation
and bot integration to the operator session.

## Prepare private configuration

The following variables are operator-selected absolute paths, outside Git:

```bash
export HYPRL_ALERT_CHECKOUT='<reviewed checkout on agents/codex/alert-router-v1>'
export HYPRL_ALERT_PYTHON='<dedicated virtualenv>/bin/python'
export HYPRL_ALERT_STATE='<private state directory>'
export HYPRL_ALERT_KEY='<private key filename>'
export HYPRL_ALERT_POLICY='<private policy JSON filename>'
export HYPRL_ALERT_DISCORD_ENV='<existing approved Discord credential filename>'
# Only when a separate staff Telegram destination is configured:
export HYPRL_ALERT_TELEGRAM_ENV='<approved Telegram credential filename>'
umask 077
cd "$HYPRL_ALERT_CHECKOUT"
```

Create the dedicated environment with Python 3.12, then install the transport
dependencies (do not alter a shared environment):

```bash
python3 -m venv '<dedicated virtualenv>'
"$HYPRL_ALERT_PYTHON" -m pip install 'cryptography==50.0.2' 'discord.py==2.6.4' python-dotenv
```

Create the parent directory of the key and the policy with mode 0700; keep
their files mode 0600. Do not copy credentials or configuration into the checkout.
Write the following policy privately, replacing the synthetic IDs. The customer
key is `guild_id/user_id`; the role ID must identify **Accès anticipé**, not a
similarly named role. Use staff IDs only.

```json
{
  "schema_version": 1,
  "guild_id": 111111111111111111,
  "role_id": 222222222222222222,
  "allowed_customers": ["111111111111111111/333333333333333333"],
  "telegram_chats": {
    "111111111111111111/333333333333333333": ["-444444444444444444"]
  }
}
```

Use an empty `telegram_chats` object when Telegram is unavailable. The operator
must verify chat ownership, bot membership and staff-only visibility before
approving an ID. Do not consume the existing bot's update queue to discover IDs.
The Discord credential file supports `DISCORD_BOT_TOKEN`, `DISCORD_TOKEN` or
`BOT_TOKEN`; Telegram uses `TELEGRAM_BOT_TOKEN`. Files are parsed without shell
execution or variable interpolation and are never echoed.

```bash
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service init-key
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service status --customer '<guild_id/user_id>'
```

Key generation uses exclusive creation and refuses to overwrite a key. Existing
state refuses a wrong key. Back up the key separately from a consistent encrypted
database backup, with private permissions. Loss of the key makes the configuration
unrecoverable. This v1 does not implement key rotation or implicit store migration.

## Enable Discord and the outbound worker

Confirm which application owns the approved token and whether its gateway is
already running. Do not start a second gateway for an existing bot without an
operator-reviewed integration decision. Enable the ordinary guilds intent for
role metadata; no privileged intent is needed for the standalone gateway. Grant
`applications.commands` and the required guild access.

For an operator-reviewed standalone gateway, run in the foreground once:

```bash
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service gateway --register
```

Registration upserts only the guild's `/alertes` group. It never performs a bulk
command sync or deletes other groups. Stop the foreground gateway with Ctrl-C,
then run the gateway and worker as user processes, for example in two tmux
sessions. Pass paths explicitly so an existing tmux server's environment cannot
silently choose another configuration:

```bash
tmux new-session -d -s hyprl-alerts-gateway -c "$HYPRL_ALERT_CHECKOUT" \
  "$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service \
  --state "$HYPRL_ALERT_STATE" --key "$HYPRL_ALERT_KEY" --policy "$HYPRL_ALERT_POLICY" \
  gateway --discord-env "$HYPRL_ALERT_DISCORD_ENV"

tmux new-session -d -s hyprl-alerts-worker -c "$HYPRL_ALERT_CHECKOUT" \
  "$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service \
  --state "$HYPRL_ALERT_STATE" --key "$HYPRL_ALERT_KEY" --policy "$HYPRL_ALERT_POLICY" worker
```

For Telegram, append `--telegram-env "$HYPRL_ALERT_TELEGRAM_ENV"` after `worker`.
Do not enable system services, change nginx or expose the API. Stop these two
owned processes by sending Ctrl-C to their tmux panes. They do not auto-start
after reboot. If a launcher uses `systemd-run` around an agent-run/tmux chain,
the host requires `-p KillMode=process`; this runbook does not install that chain.

For an existing HyprL bot, the operator can instead call
`discord_gateway.install(client, store, guild_id=..., role_id=...)` during its
setup, keep its existing gateway and lifecycle, and register this group via
the same targeted upsert. Review the pinned discord.py adapter before adopting
it: targeted registration uses its `HTTPClient.upsert_guild_command` method.
This branch does not modify that bot or its deployment.

## Staff acceptance test

1. Record only staff destination identities/digests, never URLs/tokens, in the
   private acceptance report. Confirm `#alertes-systeme` is staff-only and the
   chosen webhook belongs to that exact channel. Repeat for Slack/Telegram only
   if explicitly configured and verified.
2. A staff account with the role uses `/alertes ajouter`, then `/alertes test`.
   Keep webhook URLs in the private modal. Require the synthetic label, source,
   both timestamps, public agent names and disclaimer in the received message.
3. Check `status --customer ...`: the job must be `sent` / `DELIVERED`. A 2xx
   confirms provider acceptance, while human inspection confirms the target
   channel and rendering. Inspect `dlq --customer ...` for coded failures.
4. Use the pause button, issue another test, and verify no new send is queued.
   Resume with `/alertes pause active:False`, test once, then delete the test
   destination. Verify another account and a user without the role cannot
   operate it. Confirm every bot reply is ephemeral and contains no URL.
5. Stop the worker with a pending synthetic job; restart and verify delivery,
   rate spacing and possible-duplicate handling. Confirm e-mail and API remain
   inactive. Do not enable a real customer as part of this test.

The delivery report must distinguish an offline simulated receiver from a real
provider receipt. This branch's offline tests cannot prove live Discord command
registration, actual channel ownership, Teams tenant compatibility or provider
rendering.

## Connect existing producers without collecting new data

`producers.fact_event` builds validated events from already captured structured
facts for all twelve kinds, including the staff test. Pipe its `event()` JSON to:

```bash
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service ingest --file -
```

Without `--customer`, ingestion fans out only to the policy allowlist and each
destination's subscription/watchlist. A supplied `--customer` restricts it to
one account. Do not place an event's raw text in shell arguments. File input
must be a private regular file; stdout contains only a queued count.

The `sec` adapter accepts the existing `scripts.radar.official.parse_sec` story
shape. The `radar` adapter accepts a `scripts.radar.analysis.cluster` event plus
an `available_at` supplied by the producer's observation clock:

```bash
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service ingest --adapter sec --file -
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service ingest --adapter radar --file -
```

SEC CIKs remain zero-padded strings; missing publication times are blocked. Radar
requires importance >=70 and primary/corroborated evidence, excluding retail-only
items and raw headline/summary text. The source digest covers the local input.
All other producers use the closed fact fields in the schema. This release does
not add collectors for Form 4, earnings, CPI/NFP or market moves, compute weekly
luck, or invent duel results. The operator must connect the corresponding
already-authorized structured publications before advertising those feeds as live.
No producer integration or timer was deployed by the agent.

An event is not eligible before `available_at`. The service refuses hashes at or
after their declared open and stops queued/retried hashes before that deadline,
including a five-second transport margin. Results must be available after the
declared close, match the recorded commitment for that session, and have a
confirmed outbound pre-open receipt for that customer. An internal draft alone
does not prove pre-open publication. The producer owns the correct exchange
calendar/open/close times; the router never guesses them. Never relabel protected
holdout or unverified model outputs as measured results.

## Reliability and storage

Fernet authenticated encryption protects destination configuration, watchlists,
queued payloads, feed events, API-token hashes and duel commitments. Customer
indexes are HMACs; opaque destination/job IDs, timings, counters, states and fixed
failure codes remain visible in the private database. Directory permissions are
0700 and files 0600; unsafe files/symlinks and wrong keys fail closed. Secrets
still exist in memory and on the recipient's platform while in use.

Per-destination reservations and 30-second leases are transactional, including
retry attempts. A second worker cannot reserve another message for a destination
with an active lease. Default spacing is ten seconds and 60 attempts per fixed
UTC hour. Retry backoff is 10/20/40/80 seconds plus 0–3 seconds of jitter, maximum
five attempts; numeric `Retry-After` up to one hour takes precedence when larger.
Permanent HTTP errors and SSRF refusals go directly to the dead-letter log, the
`jobs` table's `dead` records. No raw response or URL is logged. There is no
automatic dead-letter replay: review the cause and explicitly publish a new
corrected event if appropriate.

The outbox is bounded to 1,000 pending jobs per destination; a full queue rejects
that enqueue transaction. Sent/dead jobs, feed events and duel commitments expire
after 30 days via an hourly worker sweep. Pending jobs are retained until handled
or their destination is removed. Pause prevents new reservations, preserving
already queued jobs; a request already in flight can finish within five seconds.
Deletion removes the destination and its jobs. Existing encrypted feed events
remain until retention expiry. Delivery is at least once, with possible duplicates
after a crash/ambiguous network failure. Receivers should deduplicate `event_id`.

HTTPS transport checks every resolved address at every attempt, excludes special
IPv4/IPv6 ranges, pins the allowed numeric addresses while retaining TLS hostname
verification, ignores proxy environment variables, follows no redirects and bounds
DNS + connection + response to five seconds.

## Read-only REST API contract — intentionally unexposed

`api.ReadOnlyAPI.handle` is an in-process contract adapter only, with no listener.
`GET /v1/alerts?after=<cursor>&limit=<1..100>` uses `Authorization: Bearer <token>`
and returns `schema_version`, `items` (payload plus `cursor`) and `next_cursor`.
Invalid/missing tokens return 401; writes return 405; invalid pagination 400;
unknown paths 404. Tenant scoping is derived from the token, never a query field.

Issue/rotate a token to a new private file, never stdout:

```bash
"$HYPRL_ALERT_PYTHON" -m scripts.alert_router.service api-token \
  --customer '<guild_id/user_id>' --output '<new private token filename>'
```

Rotation invalidates the previous token. Removing a customer from policy and
restarting both processes also revokes access and prevents its queued sends.
Any future HTTP exposure requires a separately reviewed deployment with TLS,
request limits and a token delivery process; it is outside this activation.

## Verification

```bash
python -m pytest tests/alerts -q
python -m pytest tests/crypto/test_fomc_*.py tests/crypto/test_sources_primitives.py \
  tests/crypto/test_edgar_slice.py tests/crypto/test_app_api*.py tests/crypto/test_ops_server.py \
  tests/crypto/test_equity_api.py tests/api -q
```

Load the host profile for private FOMC identity fixtures. The designed
`tests/api/test_backtest_summary.py` skip is allowed by the operator; report any
other skip with its cause. No web application file changes are needed.
