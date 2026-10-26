# Forward dispatch preparation (execution revision 19)

The October 26 regular dispatch exhausted its available preparation time building
inputs. Both families failed with `INSUFFICIENT_CALL_WINDOW`; no model decision
was received. The original failures and their publication receipts are retained.

The research prebuild runs at 08:45 America/New_York (12:45 UTC during EDT).
It reads already collected inputs offline, with its actual start as the immutable
input cutoff. Every source receipt, headline creation/update and completed bar
remains strictly before that cutoff. It holds the collector lock while checking
the index and object hashes. Later polls may continue after this lock is released.
The snapshot is stored by canonical hash with an atomic ready pointer; it does
not create the final daily decision snapshot or reserve a model call.

At 09:05, dispatch checks the cache's canonical hash, session, spec/code bindings,
clock ordering and age (at most 30 minutes). It copies the verified snapshot into
the immutable daily location. Missing, expired or previous-binding prebuilds
trigger an offline build with the original scheduled 09:05 cutoff. A corrupt
cache blocks instead of silently replacing evidence. A final daily snapshot is
never replaced. The input cutoff is separate from the scheduled decision and
entry anchor; both models receive the same contexts and the explicit clocks.
The crypto offset and all stock entry/exit clocks remain unchanged.

Dispatch requires at least the configured 900-second call timeout after snapshot
preparation, and checks again immediately before each subprocess launch. The
20-minute window remains unchanged, and publication is still required strictly
before 09:30 New York. There is one call per family, no retry or shortened timeout.
Insufficient headroom sends an alert through the existing research `alertes`
route; alert delivery failure is recorded and cannot make a blocked run succeed.

The 22:45 UTC research drill now runs `cli forward-drill` offline. It checks the
next exchange session's grants, frozen configuration, budget stops and model
arguments, then builds, writes and verifies a hash cache using the latest
collected session. It reports measured build/read times and fallback headroom;
less than the full timeout is BLOCKED and alerts. It makes zero requests, model
calls or publications. This is a rehearsal on archived inputs, not a claim that
tomorrow's receipts or subscription availability already exist.

Profiling isolated redundant bar indexing and repeated calendar queries.
Receipt-only row bookkeeping avoids serializing or rewriting superseded bars;
equal-receipt conflicts still compare complete decoded rows. Every source URL,
object hash, timestamp and revision rule remains checked. The history coverage
loop computes its invariant date boundaries once per symbol. No raw histories
are cached outside the disposable bounded-memory archive reader.

Revision 19 / artifact v15 registers the original v14 forward freeze alongside
v9–v13. Its new canonical spec, code freeze and explicit input-only reseal reject
old stores under the new binding. Historical contexts, outcomes, decisions,
transcripts, exclusions, publication proofs and base rates are never rewritten.
Model identities, schemas, costs, horizon definitions and statistical rules stay
unchanged. No model call, training or order is required for deployment validation.

Deploy using the established pinned research procedure: save the eight existing
research timers and unit copies, pause only their active timers, drain research
services without killing any model call, and fast-forward the research deployment
worktree to the tested source commit. Verify staged copies before atomically
rebinding the eight services; change the drill command to `cli forward-drill` and
add only the research prebuild service/timer rendered by the installer. Reload
user units, verify all nine expected-commit bindings and the prebuild calendar,
run `cli forward-dry-run --session 2026-10-27 --expected-commit <tested-sha>` and
the real offline drill service, then restore the formerly active timers and enable
the new prebuild timer. Preserve all original timer schedules and resources.
Do not run the installer's unrelated login-refresh handling for this deployment.
