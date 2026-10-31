# Growth integration onto the member v2 site

The integration starts at member release `1c9ffe4`, brings in the explicit
producer mapping from `b8bc4bf`, and applies only the growth changes from
`8359ed4`. It does not merge the older growth branch's unrelated base. No
member contract, frozen engine, source registry, holdout or trading behavior
changes. The operator report contains the pinned release commands and rollback.

## Public pages and newsletter

The US earnings calendar reads the existing validated v2 export and refuses
exports older than 26 hours. Day markers remain unconfirmed, without invented
publication times. The SEC guide is an explanatory page; it initiates no source
request. Official links were not fetched during integration.

The completed BTC claim check is at
`/chance-ou-pas/btc-vendredi-plus-bas-asie`, linked from the index and sitemap.
It uses price-free public aggregates from study revision
`98c7189b207e857777e188318b7248e87616ddd1`. The page displays the observed
251/450 Fridays (55.8%), 58.0% on other days, the 51.2–60.3% Wilson interval,
paired-week difference interval, p=0.075 and the exploratory limitations.
Neither a Friday advantage nor the 60% assertion is confirmed. The completed
article is independent of the daily member export; dynamic claims and earnings
retain their existing freshness checks. This does not populate the member luck
counter or rewrite its contract.

Newsletter consent remains an optional unticked box with its own version and
text, double opt-in, 48-hour hashed confirmation and withdrawal. Opening links
alone never subscribes or withdraws. Withdrawal preserves early access; general
deletion removes both. Legacy users are never enrolled. Weekly delivery retains
durable reservations and never retries uncertain SMTP sends automatically.

`NEWSLETTER_ENABLED` defaults to false. Provider tracking and processing
attestations remain operator-owned; no agent sets these attestations true.
Prepare stable private token/worker keys before activation. Only the site process
writes subscriber state. Install a weekly loopback worker only after the site
release and provider acceptance; never run another writer on the same file.

## Existing HyprL Discord application

This supersedes the original growth-v1 request for a new application. The user
unit loads the existing identity environment via `--discord-env` and then a
separate referral configuration. It uses `DISCORD_BOT_TOKEN` and
`DISCORD_GUILD_ID`, without copying or changing that identity file. It never
loads, changes or restarts kyoko-bot.

`setup_hook` upserts only the guild command `/parrainage`. It does not call
bulk command synchronization, which could remove the application's other
commands. The reply is ephemeral; invitation creation occurs only on an explicit
member command. Invite attribution, seven-day eligibility/stay requirements,
SQLite durability and once-only reward records remain unchanged.

`REFERRAL_ENABLED=false` prevents network startup, including through the unit's
ExecCondition. `REFERRAL_PUBLISH_LEADERBOARD=false` prevents every public
leaderboard send, and `REFERRAL_APPLY_REWARDS=false` prevents all role writes.
These last two switches remain false for this delivery. The ledger can record
qualified rewards without granting roles; billing credit stays `pending_stripe`.
No payment integration exists. If roles are enabled later, configure two
distinct marker roles, below the application role and separate from actual
paid membership roles. Public posting requires its own later instruction.

Manage Guild is needed to read invitation counters and Create Invite on the
chosen invitation channel is needed for private command responses. Reward and
leaderboard permissions are needed only if their separate features are enabled.
The operator must select the invitation channel; no channel or role is guessed.
Do not start a second gateway process for this application without checking the
existing application deployment and its session policy.

## Guarded site release

The source lives in an ordinary agent branch descended from `1c9ffe4`; the
supplied branch is preserved. Prepare the release with `git archive` from the
exact reviewed SHA. Install dependencies and run every site test before pruning
development dependencies. Preserve external state and the live site's legal
facts. Copy its three unaffected legal pages and use
`server/growth/merge-privacy.js` to insert only the two new growth sections into
the existing privacy page. This preserves all live publisher/processor facts;
an unexpected HTML structure is rejected for review. It copies no private file
into Git.

Run `node --env-file=... server/growth/release-preflight.js <live-directory>`
from staging. This read-only command requires loopback production config, member
v2 enabled, fresh observed v2 member/public exports, external state paths and
0440 member snapshot permissions. It creates no application/store and sends no
mail. Confirm the exact source and live/staged code digests again before the
directory switch, then restart only the site user service. Keep a private receipt
with the old code directory and new SHA for rollback. Rollback restores only
code; never restore or copy personal state, deliveries, withdrawals or ledgers.

The site switch remains operator-owned, following the `1c9ffe4` directory-swap
procedure. Run production OAuth/member acceptance under the operator's network
policy. Local screenshots use synthetic member/calendar/claim data; the BTC
article captures use only the pinned already-public aggregates. Browser requests
outside the isolated loopback servers are blocked.
