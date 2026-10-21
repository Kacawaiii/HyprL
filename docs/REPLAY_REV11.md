# Replay revision 11

Spec artifact v11 and state directory `duel-v11` implement execution revision 15,
following the revision 14 procedure. The operator fixed these rules before scoring
any outcome, uniformly for both model families, replay and forward.

News candidates without a view receive an explicit ABSTAIN view with probability
0.5, no citation and no trade. Supplied views still undergo every existing check;
duplicates and unknown symbols are invalid, and forecasts must cover all candidates
and both horizons. The normalization records the filled count. Prompt bytes and
model-facing schemas are unchanged.

Subscription resets accept Codex try-again notices, Claude reset clocks and dated
forms. Undated clocks use their next occurrence; notices within the printed minute
expire at its end because the CLI truncates seconds. Dated notices without a year
handle New Year. Structured reset evidence retains precedence. Saved quota messages
can repair existing stop clocks without changing counters, source envelopes or
credit stops; stops without matching evidence remain unchanged and are reported.

The continuation v3 registry binds 151 original received envelopes from states v8,
v9 and v10. For the July 6 Opus news answer it admits only the original try0;
the rejected try1 is preserved separately, alongside quota evidence. Received
answers are decoded and validated locally. There is no answer selection, repeated
received call or budget refund. Scoring resolves each original response through
the admitted source and verifies its digest and publication. Pending publication
includes referenced response hashes under the current protocol as well.

The v10 forward freeze is added to the immutable historical scoring registry;
the v9 binding remains intact. The 128 input contexts, sealed cache objects,
paired exclusions and frozen development base rate are revalidated and resealed.
Old stores still fail current configuration checks. Prior test-start markers,
including v10 markers, forbid a replacement test protocol.

The research CLI's `forward-dry-run --session 2026-10-22 --expected-commit <sha>`
checks committed code, freeze and seals, both grants at the scheduled dispatch,
the real forward budget gates and no-tool command construction. It reserves no
call and performs no source request or publication. The scheduled decision is
13:05 UTC, with a 900-second call timeout inside the 1,200-second dispatch window.
Tomorrow's actual input receipts and provider availability remain unproven until
that session. The existing collector drill remains a separate data-collection job.

Deployment advances the research worktree once, after all suites pass, then binds
only its research service commands to the deployed commit. Existing schedules and
the trading engine are unchanged. No outcome is scored during this maintenance.
