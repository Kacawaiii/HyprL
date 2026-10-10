# News radar v1

This is an independent, read-only information pipeline. It does not import the
AI trader, modify preregistration or call any broker. Paper follow-up means a
prospective observation journal, not orders or simulated fills. Its state and
reports must be outside the checkout; permissions are directory 0700/file 0600.

## Operator authorization

Every live request requires a dedicated `news-radar-v1` authorization file with
an operator signature marker, UTC start/expiry, exact HTTPS origins, GET paths
or path prefixes, and daily budgets. Existing trader grants do not apply.
The collector rechecks expiry before every dispatch. It never follows redirects
or forwards Alpaca keys to another host. SEC and Federal Reserve adapters are
absent. Source responses, keys, stores and reports must never enter Git.

Produce a **false/unsigned** draft for review:

```bash
python -m scripts.radar.grant_template ~/reports/news-radar-v1-authorization-draft.json
```

Only the operator may review/sign it and place it at
`~/authorizations/news-radar-v1.json`. A signature marker denotes a reviewed
local operator file, not a cryptographic signature. Do not self-authorize.
The draft explicitly includes Yahoo chart endpoints for yield, dollar, FX,
futures and VIX instruments which the Alpaca equity/crypto feed cannot supply.
No article is fetched: only provider news summaries, RSS/Atom, channel metadata
and market data are requested. RSS `content:encoded` is ignored; descriptions
are trimmed to short summaries. HTTP article links are preserved but not fetched.

## Running

Use the host's Python environment (requires the repository's existing test
dependencies for the repository-wide checks). The radar itself uses stdlib.

```bash
python -m pytest tests/radar -q
python -m scripts.radar.service demo --store ~/private/radar-v1-demo --reports ~/reports/radar-demo
python -m scripts.radar.service collect
python -m scripts.radar.service run --slot morning --publish-book
python -m scripts.radar.service status
python -m scripts.radar.deploy --output ~/reports/radar-unit-preview
python -m scripts.radar.deploy
```

`demo` requires an empty separate store, does not use network/LLM and cannot
publish to Claude book. `run` and `collect` require the grant; missing grants
exit BLOCKED. The credential file is configurable with `--credentials` and is
read only for the authorized Alpaca origin. Errors report fixed codes only.
No response body, key or CLI error output is printed. The default private
locations follow the task's operator conventions and can be overridden with
`--store`, `--reports`, `--authorization`, `--credentials` and `--channels`.

Channel configuration is an optional private JSON array of `handle` and/or
`channel_id` records. Fifteen French/US candidates are included. A channel ID
must be verified from metadata or supplied by the operator; no ID is guessed.
Candidates are unverified until a successful parse. HTTP 404/410 and pages
which are not feeds and redirects drop a candidate without following the
redirect; 403, 429 and grant refusals are reported as blocked. Remove the cause or append a reviewed new health record
to retry a dropped candidate; never edit/delete its historical evidence.

The installer writes only three radar service/timer pairs. It requires a valid
grant to activate them. Preview does not install or enable anything. It sets
`KillMode=process`, low priority, a memory cap and no persistent catch-up runs.
Timers are UTC: weekdays 12:20/21:20; weekends 11:30/19:30. Headlines are collected
hourly at :05, with GDELT only on even hours and never 11:45–12:30Z. A timer
that collides with another run skips via the private owner lock.

## Evidence and interpretation

SQLite has append-only news, request receipts, health, cursor/cache observations,
bars, reports and paper observations. Update/delete triggers protect evidence
and budget reservations. Dispatch is committed before the network call; failures
consume budget. Limits persist across restarts, shared by all runs using the
same store: Alpaca <=400/day, GDELT <=60/day and >=7 seconds, RSS <=300/day,
YouTube <=400/day, Yahoo <=24/day, Sonnet <=2/day including failures. Feed polls
are >=900 seconds per feed. Operator budgets can reduce these limits. A 429
stops that provider and persists a cooldown of at least one hour/Retry-After.
Alpaca news has no symbol filter and uses ascending bounded pagination. An
unfinished cursor is resumed with its original time range; backlog is visible.

Publication and receipt remain separate. Unknown/future/naive publisher times
stay unknown; GDELT `seendate` is discovery only. Source and original URL,
headline/summary digests, publisher and feed symbols are retained. Headline,
URL, shared entity and token similarity cluster repeats within a bounded window.
An article with a known publication time older than three days does not enter
the ranking merely because its feed was fetched now. The private evidence stays
append-only. Entity patterns are compiled once per dictionary for live volume.
Generic words such as maker, optimism and curve require explicit crypto context
to identify tokens. Weekend priority uses textual crypto relevance rather than
an incidental crypto ticker in a provider's broad basket of associated symbols.
The rolling "what happened in crypto today" index stays in capture evidence
but is excluded from discrete events and event-time price attribution. Reviewed
observation watches can be retired by an append-only record without deleting
their original baseline; retired watches do not acquire future labels.
When YouTube is unavailable and no retail posts were observed for an event,
the digest labels hype unknown rather than suggesting measured zero interest.
Editorial groups and explicit syndication attribution determine corroboration;
unattributed syndication remains uncertain. Repeated rumours remain rumours.
YouTube breadth affects retail hype only. The snapshot dictionary contains
500 company names, the repository's stock universe, major country/sector ETFs,
40 crypto candidates, countries and themes. Three ambiguous prospective symbols
in the repository snapshot lack a verified name; this appears in coverage.
The snapshot is not a live claim about S&P membership or Alpaca crypto support.

Importance, evidence, novelty, retail hype, observed price movement and
qualitative scenario conviction are separate fields. Transmission hypotheses
include direct companies, suppliers, competitors, sectors and country ETFs;
they are conditional mechanisms. No consensus is invented. Each ranked
scenario includes changed expectations, impact, horizon and invalidation.
Anomalies compare the latest provider daily observation (complete or explicitly
partial) with completed prior twenty-day norms. A partial day's volume is not
extrapolated, and its close is not treated as a final daily close. Stock volume
uses IEX and is labelled partial-market. A daily bar is conservatively final
only when its entire day interval completed **before the response was received**.
Advancing the clock never turns an earlier partial capture into a completed bar.

Priced-in is **observed movement**, not a remaining-upside score or proof of
causation: last closed minute before publication to the latest closed minute,
divided by ATR from completed days preceding publication. A baseline older
than fifteen minutes or an unknown publication time yields unknown. Intraday
bars are collected for the top events and active observation watches only,
so other events can legitimately lack this measure.

The regime panel names every provider instrument. It uses fresh closed minutes
for SPY/QQQ/BTC/ETH and Yahoo's timed observed quotes for the other instruments;
historical returns use calendar anchors and conservative completed daily bars.
The JSON records anchor timestamps. Yield is labelled as the provider's yield
index without guessing a scaling conversion. Gold/Brent are futures, not spot.
Missing/stale measurements stay unknown; ETFs never impersonate exact indices.
Alpaca class symbols use the provider's dotted notation on dispatch, then map
back to the repository identity. Empty bars are unverified coverage, not a
successful price-source check; unexpected returned symbols are rejected.

One Sonnet CLI invocation per radar, with no tools, hooks, MCP, skills or session
persistence, returns validated qualitative French scenario JSON. Feed data are
untrusted context. Event IDs and allowed symbol/role pairs are checked. Digits,
financial numeric expressions and unsupported output fields are refused; all
numeric report fields are rendered directly from observations/computed scores.
The CLI uses an empty temporary directory and fixed timeout, never broker keys.
If it fails, the French deterministic digest shows the failure, unknown
conviction/horizon/invalidation and measured data; there is no second LLM call.
Reports are <=60 lines, with complete evidence in companion private JSON.
Completed slots are idempotent and can rematerialize their reports after a crash.

The watch journal freezes a known observation baseline only after the scenario
is generated. It measures prospective calendar-day horizons (one and five)
at the first subsequently available completed observation. No pre-horizon
price can supply a label. These are descriptive price returns; there are no
fees, execution assumptions, causal claims or PnL estimates. All comparisons
remain separate from the trader's protected research and Claude's account.

## Release gate

Before pushing, run the full AGENTS.md Python checks and the radar tests, with
the host profile loaded for private FOMC fixtures. CI runs the radar tests in
`sources-ci`. Fast-forward integration requires green CI for the exact SHA and
an unchanged ancestor base. Feed verification, real regime coverage, an actual
Sonnet invocation and enabled timers are unproven until an operator grant is
present. A synthetic report is never called the first live radar.
