# Public agent presentation

Member-facing agent labels are **HypR-A1** and **HypR-E1**, from the single
`configs/public_agent_names.json` table. Python renderers use
`scripts.public_agent_names`; the web app imports that same JSON. Analyst and
reviewer roles retain their distinctions. The discretionary book stays a
separate account and is never included in a research/trader baseline.

Méthode : agents propriétaires HyprL construits sur des modèles de langage tiers.

Names are applied only at presentation boundaries. Stored model identities,
transcripts, source quotations, ledger branches and entries, snapshot keys,
hashes, protocols, authorizations and audit evidence keep their original bytes.
Historical deliveries are never rewritten or resent to change their names.

The public weekly renderer (`python -m scripts.public_weekly`) consumes the same
scores and retains the existing weekly/correction receipt format, freshness,
score-hash, correction-period and at-most-once checks. It renders names before
splitting messages and hashing their actual delivery text. Corrections remain
explicit and refer to the original text hash. No historical receipt is migrated.
It uses the existing research grant, ownership lock, clean-commit and pinned-SHA
checks. Discord fallback is still unconfirmed, never a successful receipt.

Other research notifications use `scripts/public_research_notify.py`. The adapter
retains ledger/hash/URL/model-identity lines verbatim and converts owned agent
labels in prose. It forwards the original transport exit status and stdout.
On the existing `ordres` route it accepts only the frozen research hash-only
commitment template and forwards it byte-for-byte; trading messages are rejected.
Use `--render-only` for offline previews.
Source headlines and arbitrary serialized records must not be passed through
the prose converter. Future video and script exporters must use
`public_agent_label` for label fields, keeping their provenance separately.

The brief patcher stages a copy of the existing brief. Only its owned disclosure
and method text change; input assembly, source content, model invocation,
cache, metadata and delivery logic are unchanged. Anthropic and its potential
conflicts remain disclosed. The web app also converts legacy snapshot labels
while rendering; it never changes those snapshots or their identity fields.

## Research and brief deployment

Use the pause/drain/fast-forward/atomic-rebind procedure in
`REPLAY_FORWARD_WINDOW_REPAIR.md`. Save all nine research services and timers,
their active states, the research execution hash and input-seal digests. Stage
and verify units before replacing anything. Pause only active research timers;
let jobs drain, without killing or restarting a model call. Fast-forward the
research deployment checkout to the tested commit. Rebind all nine service
`--expected-commit` values, preserving schedules, resource limits and environment.

For the weekly service only, replace the module/command
`scripts.trading_lab.replay_seal_forward.cli weekly` with
`scripts.public_weekly`; all its other arguments remain. It uses the existing
notify transport because its prose is already rendered before splitting.

For the other eight research services, append a private `EnvironmentFile` with
`REPLAY_NOTIFY` pointing to the executable `scripts/public_research_notify.py`
in the deployment checkout, and `HYPRL_RESEARCH_NOTIFY_TRANSPORT` pointing to
the original notify executable. Append this file after the original environment
file so that these presentation settings take precedence. Never change the
shared notify binary or any trader unit. Verify staged user units, install them
atomically, reload user units, check all effective bindings, run the offline
`forward-dry-run`, and restore exactly the previously active timers. Confirm the
frozen execution hash and prior evidence digests still match. This presentation
change needs no spec revision or input reseal: all frozen files stay unchanged.

Stage the brief without editing its live file:

```sh
python -m scripts.public_brief_install --source "$BRIEF_TOOL" --staged "$STAGED_BRIEF"
HYPRL_BRIEF="$STAGED_BRIEF" python -m pytest "$BRIEF_TESTS" -q
```

Back up the live brief, confirm it still matches the source that was staged,
copy the tested candidate to a temporary file beside the live executable and
atomically rename it. Retain permissions. Repeat the offline tests against the
installed tool; render synthetic md/html/Discord output without sending. The
patch records the deployment checkout for importing the shared table: keep that
checkout pinned until a later coordinated deployment. Leave brief timers and
past exports, model logs and delivery ledgers intact.

## Cockpit/operator handoff

Build `apps/web` after its required checks. Deploy its static bundle through the
existing cockpit release procedure, preserving the current backend and snapshot
configuration. The source branch also contains the research predecessors it was
fast-forwarded onto; integrate intentionally. Do not use a blanket checkout or
restart of v1 trader units for a presentation change. If a v1 runtime update is
needed later, the operator owns it and must follow its guarded deployment window.

Verification: `pytest tests/public_agent_names` exercises real synthetic weekly
deliveries, message splits, unchanged numbers/score bytes, original correction
receipts, hash/staleness/weekday/delivery gates, identity preservation, the brief
patch and the frozen execution binding. Web tests cover analyst/reviewer labels,
retirement, legacy book labels, accessible chart text and immutable snapshots.
