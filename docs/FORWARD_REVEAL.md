# Forward decision reveals

`python -m scripts.forward_reveal --expected-commit <tested-sha>` audits all
recorded forward sessions locally. It writes nothing and does not fetch, push,
notify, call a model, collect sources, score, or open market price data.
`REPLAY_DATA_ROOT`, `REPLAY_LEDGER_WORKTREE` and `REPLAY_NOTIFY` use the existing
private research environment. Output contains only statuses, digests, counts,
session identities and commit identities. No decision content is printed.

The operator's `--publish` mode takes the same ownership and publisher locks as
the existing research publisher. It refreshes the ledger remote, verifies the
original receipt and committed line, then creates one append-only disclosure
commit per eligible session on `ledger/forward-duel`. It rejects unrelated
changes/unpushed commits and divergent history. Normal pushes are fast-forward;
there is no reset, rebase, merge, force push or overwrite of existing reveals.

Files are `reveals/<session>/HypR-A1.json`, `HypR-E1.json` and `README.md`.
The README links to the **earlier commitment**, lists each digest and gives the
`sha256sum` commands. Actual model identities remain inside the files: public
labels are filenames and prose only.

At creation, the existing producer applies `sha256_canonical(record)` before
adding `decision_hash`, then stores an indented envelope. Original canonical
preimages are not stored separately. The exporter therefore removes only that
self-referential hash field and obtains the preimage using the **same** shared
`canonical_bytes` function. Both the envelope hash and the earlier published
hash must equal SHA-256 of these bytes. It writes those bytes verbatim, with
no envelope, pretty printing, alias replacement, encoding change or newline.
Git blob identities are also checked against the unfiltered file before push.
A record flagged for private content blocks; it is never redacted to make its
hash pass. The existing private data and receipts are never rewritten.

Disclosure is gated on the exchange calendar's actual close plus 15 minutes,
with DST and shortened sessions. No decision is read before that boundary.
Publication and the earlier ledger commit must both predate the US open;
receipts must match the original public ledger and a pushed ancestor. Missing
or late confirmed publication produces an **UNPUBLISHED** README, without a
decision file. Published failure hashes produce **NO_DECISION**, without
exporting failure bodies. Missing/corrupt records block instead of inventing
preimages. Every run considers all sessions, so the same path backfills history.

The October 20 intraday commitment was made **after** the US open. The task's
strict pre-open rule therefore makes it UNPUBLISHED for disclosure. Its original
intraday scoring/publication protocol stays intact. Revealing it would need an
explicit operator exception to the disclosure rule; this implementation does
not silently substitute the intraday entry boundary for the US open.

After a confirmed push, a one-line notice with the disclosure commit URL goes
to `brouillons`. Private intent/delivery markers make successful reruns
idempotent. Failed pushes can resume the same commit. Ambiguous notification
delivery blocks automated resending: the operator must reconcile the Discord
delivery before changing its marker. A non-Discord fallback is unconfirmed.

## Operator deployment and backfill

The agent branch contains code, synthetic tests and this procedure only.
AGENTS.md restricts agents to pushing their own branch; the agent must not push
the ledger branch. Runtime publication is an operator action. A deployment
must also authorize publishing the exact decision records under the repository
rule that otherwise limits public evidence to digests, URLs and counts.

Use the pause/drain/fast-forward/atomic-rebind research procedure documented in
`PUBLIC_AGENT_NAMES.md` and `REPLAY_FORWARD_WINDOW_REPAIR.md`. Save all nine
existing research service/timer files, their active set, environment bindings,
the frozen execution hash and original evidence digests. Stage and verify
before installing. Pause only active research timers and let jobs drain without
killing/restarting a call. Fast-forward the research checkout to the reviewed
tested commit and rebind all nine expected-commit values. Keep every existing
schedule, resource limit, command and notification environment unchanged.

No frozen file, spec, research installer or input seal changes in this patch.
The sidecar and unit renderer are outside the frozen execution tree. Do not
run a legacy installer, reseal inputs or migrate stores for this deployment.

Stage the **two new** research user units outside Git using operator-provided
absolute paths (the environment is the private original research environment):

```sh
python -m scripts.forward_reveal_units --checkout "$RESEARCH_CHECKOUT" \
  --environment "$RESEARCH_ENV" --python "$RESEARCH_PYTHON" \
  --expected-commit "$TESTED_SHA" --output "$NEW_UNIT_STAGE"
systemd-analyze --user verify "$NEW_UNIT_STAGE/hyprl-research-forward-reveal.service" \
  "$NEW_UNIT_STAGE/hyprl-research-forward-reveal.timer"
```

The service uses the original research notifier directly, with only the
`brouillons` route. Preserve the original environment, including transport PATH.
The timer is 16:15 America/New_York (20:15 UTC during EDT, 21:15 during EST).
Short sessions are disclosed later than their minimum gate. Persistent catchup
still runs all clock/receipt/byte checks. Holidays have no eligible new session.

With the reviewed environment loaded and from the pinned clean checkout:

```sh
"$RESEARCH_PYTHON" -m scripts.forward_reveal --expected-commit "$TESTED_SHA"
```

Review its per-session audit, the four eligible October 22/23 decision digests,
the October 20 intraday UNPUBLISHED status, and failure-only days. Reconcile any
BLOCKED result first. The **operator**, after resolving the publication rules,
can backfill all closed sessions with:

```sh
"$RESEARCH_PYTHON" -m scripts.forward_reveal --expected-commit "$TESTED_SHA" --publish
```

Verify raw files with `sha256sum`, the earlier commitment links, the remote
disclosure commits and the confirmed `brouillons` delivery markers. Install the
two verified user unit files atomically, reload user units, check all ten
effective commit bindings, then restore exactly the nine formerly active
timers and enable only the new reveal timer. Run the existing offline next-day
`forward-dry-run` and confirm frozen execution/evidence digests are unchanged.
Never modify v1, operator tools, other services, or historic publications.

Rollback disables the new reveal timer and removes only its two user unit
files after it drains. Preserve already-public ledger commits and private
delivery markers; do not rewrite publication history. Restore the previous
active research timer set using the documented procedure.
