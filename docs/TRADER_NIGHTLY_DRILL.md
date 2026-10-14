# Nightly trader drill

Install `scripts/trading_lab/trader_agent/nightly_drill.sh` as the operator's existing nightly
script only after exact-SHA CI and the guarded deployment checks pass. It
resolves the run service's deployed worktree and accepts an optional UTC date;
the default is tomorrow. Each invocation gets a fresh private directory and
isolated synthetic budgets. No production runtime is copied or deleted.

The AM and PM commands use the production decision service with native-shape
`SyntheticData`, deterministic independent analyst/reviewer fixtures and the
pinned equity calendar. `run --dry-run --defer-synthetic-labels` leaves every
new outcome pending. The separate 21:30 synthetic label step must add labels:
same-day for an equity session, next-day for a crypto-only closed session.
A second checkpoint ten days later must add the remaining horizons with zero
pending. Both run steps must publish nonempty synthetic decisions. Expiry,
pause, slot registration and budgets retain their existing checks.

The flag is rejected on production commands and non-run actions. Existing
`run --dry-run` demonstrations retain their automatic future realization.

The current paper execute/exit/audit probes still inspect today's production
research journal and broker through GET-only snapshots. Their names and report
identify that current clock explicitly; tomorrow's synthetic time does not
simulate broker orders, acceptance, fills or paper selection. Busy or changed
snapshots are unproven checks and make the drill BLOCKED. An equity execute/exit
skip passes only when its result timestamp is on a pinned closed-calendar date;
crypto and account audits still run. An open-calendar skip fails verification.

The report reads each command's top-level status/state, never a nested scoring
state. A nested news comparison may remain TOO_EARLY because its prospective
sample is small; it cannot replace a completed command's state. Zero new labels
or an empty decision also fail verification even when the command exits zero.

Reports retain the operator's `drill-<date>.md` location. Full outputs remain
private. Unattended failures retain the existing Telegram notification;
`HYPRL_DRILL_NOTIFY=0` suppresses delivery during a manual proof without changing
the result or failure exit status. This drill proves synthetic decision, recovery-no-op and label paths,
plus current GET compatibility. It does not prove tomorrow's live inference,
source availability or broker execution.
