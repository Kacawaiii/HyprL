# Replay revision 12: terminal model failures

Spec artifact v12, execution revision 16 and state `duel-v12` implement the
operator's October 22 decision before any outcome is scored. Each stage uses
only its original attempt pair. Two rejected validations resolve the entire
model/period/group/day cell as `MODEL_FAILURE`, with empty decisions and
forecasts. The worker advances to the next day; checkpoint recovery returns
the same terminal record. Quota, transport uncertainty and integrity errors
retain their existing stops.

Every failed attempt records its code, stage, attempt number and response
identity in the private failure journal. Revalidating the same saved response
does not add another failure event. A quota response and a subsequent received
response remain distinct events, without refunding either reservation.

Scoring accepts terminal failures as resolved cells, publishes and verifies
their decision and response hashes, and assigns no forecast or trade to them.
The other model retains its own valid sample; paired comparisons use common
forecast keys. Every panel carries its group's failed/resolved cell counts,
failure rate and planned count. The Markdown score table includes failure
counts and rates beside the performance measures. Pending cells remain pending.

Both original GPT news answers for development July 13 fail
`CITATION_NOT_IN_SNAPSHOT` in `views.news_ids`: each cites two IDs belonging to
other supplied candidates. The IDs and source IDs have the same integer type;
the candidate-specific check is correct. The dispatch journal contains two
real calls. Eleven failures represent eleven validations of that same pair,
including twenty redundant validations, rather than twenty-two real calls.
No citation rule changes and no forecast is graded during this diagnosis.

The generic `INVALID_OR_MISSING_INPUT` was a harness error in the Opus
development anonymous control for July 1. The analyst validator passed a coded
headline clock, such as the synthetic `DAY+0 11:55:08 UTC`, to an ISO parser.
The resulting `ValueError` escaped the worker. Anonymous analyst citations now
use a separately bound coded-clock schema and exact supplied clock/URL pairs.
Regular replay and forward model-facing schemas and prompt bytes stay unchanged.
The original admitted anonymous answer is reused through its original
request/schema identity, with explicit compatibility metadata and diagnostic
counts when its prompt differs from the repaired schema. No real calendar date
is invented and no valid saved answer is re-asked.

Continuation v4 adds the 79 original received envelopes from v11 to the 151
already admitted originals. Later re-asks are never admitted or graded; the
observed July 13 loop created no additional received envelopes. Source files,
reservations, old specs, freezes and seals remain unchanged. The 128 contexts,
both cache maps and all paired label exclusions are revalidated and explicitly
resealed. Stores with an old hash still fail the current configuration check.

The immutable historical forward registry also retains the v11 freeze, so
October 22 proofs keep their original protocol. Publication receipts are checked
against that original protocol too; requiring the new protocol would falsely
reject the previously published proof. Deployment follows the rev11
procedure: all suites pass, the clean research worktree advances once, its eight
service commands bind the deployed commit, and the October 23 scheduled forward
preflight checks both families without requests, calls or publication. Actual
future receipts and provider availability remain unproven until dispatch.
