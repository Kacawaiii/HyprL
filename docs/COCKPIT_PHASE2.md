# Cockpit phase 2

The cockpit remains read-only. It neither collects source data nor submits
orders. `analysis.json` (`cockpit-analysis-v1`) joins existing radar reports,
read-only trader evidence and explicit Claude book journal fields. It is served
at `GET /api/v1/radar/analysis` using the existing `--radar-root` directory.
Snapshots, input configuration, screenshots and raw evidence stay outside Git.

## Decision provenance

All decision views use the same six steps: sourced fact, dated market
expectations and priced-in assessment, scenario, entry condition, invalidation,
then the realized result after costs. Missing fields remain explicit.

`OFFICIEL` identifies a cited primary document (SEC, Fed, BLS, FCC, or explicit
company primary provenance). `FIL` identifies a feed citation without primary
confirmation. `NON VERIFIE` means no identifiable citation. An explicit feed
flag is never upgraded solely because its link is on a primary domain. Each
citation carries its own level. The headline's level comes from its first
citation; a second primary citation does not confirm a feed headline. The
cockpit does not independently verify cited document contents.

## Markets and news comparison

The Markets panel filters by asset, horizon, observation period and overlay
kind. Radar markers use receipt time; AI decisions keep their run time and
reviewer verdict; realized outcomes remain separate later observations. Price
points come from trader reference closes, label entry/exit prices, radar marks
and dated cached book marks. Dotted connections are visual guides, not captured
continuous bars. Claude limit entries and planned stops are labeled intentions,
not broker executions. Keyboard selection and an accessible list expose the
same decision cards as the chart markers.

The news view separates explicit news tags from technical tags and unclassified
trades. `momo_v0`, `momentum` and `sleeve` are technical. Unfamiliar tags are not
inferred from prose. A close is linked only when one matching intention exists;
overlapping lots remain unresolved. Hit rate counts strictly positive explicit
net P&L. Average R uses explicit net R or net P&L divided by recorded initial
risk. Missing costs/gross-only results and open marks are excluded. Counts of
intentions, linked closes, known net outcomes and known R outcomes are separate.

AI scores come directly from the existing scorecard. Population, horizon,
target and calendar/execution variants stay separate. The random baseline uses
`random_seeded`; unavailable baselines remain visible. A with/without-news
comparison remains unavailable until matched, identified variants exist. The
30-outcome display caution is descriptive, not a validation/significance rule;
no trader preregistration or scoring policy is changed.

## Offline hourly refresh

`python -m scripts.radar.cockpit_refresh --config INPUTS.json` reads a private
configuration with these fields:

| Field | Input |
| --- | --- |
| `out` | Private snapshot directory outside the checkout |
| `radar` | Radar JSON file, or directory of existing real reports |
| `trader_root` | Active trader runtime; evidence is opened read-only |
| `book_journal` | Existing book journal, never modified |
| `paper_report` | Optional existing AI paper report |
| `radar_root` | Optional active radar ledger; SQLite `mode=ro` |

The existing `paper.json` supplies cached book marks. Their `observed_at` remains
unchanged across refreshes. Refreshing a cached account does not append equity
history or claim a new broker observation. New journal reasoning is exported
independently. No broker CLI or source HTTP request runs during refresh.

`--install-timer` installs only `hyprl-cockpit-refresh.service/.timer` in the
current user's unit directory. The timer runs hourly at :35 UTC, with
`Persistent=false`, `KillMode=process`, `MemoryMax=384M`, `CPUQuota=40%`, a
three-minute timeout and private output permissions. The runtime also skips
11:42–12:30Z to protect the full three-minute execution window before 11:45Z.
A private nonblocking lock prevents concurrent exports. User-unit health is
read through an allowlist of status properties; unit commands, grants,
credentials and raw logs are never exported. Trader, radar and book units are
observed only. Their configuration and scheduling are not changed.

## Verification

Synthetic export/refresh checks live in `tests/radar/test_cockpit_phase2.py`.
Browser behavior is covered by `apps/web/src/test/phase2.test.tsx`; the real
snapshot screenshot procedure is `apps/web/e2e/cockpit-phase2.cjs`. It blocks
non-loopback browser requests and writes evidence outside Git.
