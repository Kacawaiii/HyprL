# Opus replay protocol (revision 4)

The operator's 2026-10-18 decision supersedes the model-memory gate for this protocol using three new
operator-run probes with rules hashed before calls in clean sessions. Preserve revisions 1–3, the old
freeze and trial registries, and all old stores. Revision 4 rejects their decisions without migration.

All four strategies now use Opus: the shared a/b analyst, a's reviewer, c's news analyst and hence d's
primary-source subset. The anonymized b audit uses Opus on every fourth development session. Sonnet is
refused before transcript reuse or reservation; there is no longer a paired Sonnet/Opus subrun. Reported
CLI identities must match the pinned Opus identity. Non-flagging is evidence for the operator's research
floor, not proof of a training cutoff or absence of latent memory.

The private probe location is configured by `REPLAY_PROBE_ROOT`. Exactly the three preregistrations and
six Opus/Sonnet answer files are pinned by file digest. `audit` regrades complete ID sets from the raw
answers; it never reads an editable summary. Public evidence consists only of hashes and counts. These
answer files lack CLI identity/timing envelopes: pre-call hashing, model attribution and clean sessions
are operator-attested. The old forced-guess rule remains documented under revision 3 and is not rewritten.

Development covers July 1–August 31, 2026; test covers September 1–October 9 once. The prior test
configuration hashes remain in `replay_bench_test_freeze_v2.json`. For each group, the exact prior hashed
payload (spec hash, code hash, period, variant group) is retained and **only `model: opus` is added** before
rehashing. Its inherited spec/code fields identify the parent configuration; current execution spec/code
hashes are bound independently in the new freeze and every decision. The new freeze records both old and
new hashes and the amended payloads. Strategy rules, prompts, universe, periods, fills, costs and score
rules are unchanged. Synthetic tests compare these fields and verify the model-only amendment.

`replay_bench_test_freeze_v3.json` freezes the nonempty 28-session test plan. Missing input proof remains
a hard blocker. Supply evidence for unrevised prices and historical universe/identities in the existing
private `replay-input-proof-v1` shape, bound to the v4 spec, with evidence files and prediction-cache digests.
No proof is fabricated from a present-day archive. The `seal-inputs` command creates a separate public
`replay_bench_input_seal_v1.json` containing only the proof/freeze digests; commit it before test calls.
This appends input evidence without changing the frozen strategy or migrating old records. The code
hash excludes this non-spec evidence artifact, so committing that seal does not invalidate decisions.

The private `v4` directory holds new trials, call intents, results and test ownership. Task-wide budgets
and same-night quota stops remain shared with earlier runs. Dispatch is serial, only 00:30–05:30 UTC,
with at most 60 combined calls per UTC night including retries/reviewers and inherited calls. A quota
response stops the night; a timeout or unresolved intent stops without blind retry. Completed responses
resume locally, including interruption between analyst and reviewer. Saved scored periods are reused
only when current execution and all decision hashes match, avoiding another test scoring pass.

Complete and score development, including anonymized calls, before the first test call. Test ownership
is irreversible and resumes only the same freeze. Prediction transport reads only sealed offline cache
objects; BTC/ETH protected dates remain refused at the transport boundary. Actual scoring is unproven
until all original input and outcome coverage requirements pass.

The research timer installer accepts `--replace --backup-dir` for the existing replay units only. It
refuses an active service or an unrelated unit, verifies staged units, backs up the previous files and
replaces them atomically. It pins the clean source SHA, runs with bounded memory/CPU, and treats blocked
research (exit 2) as an expected stop. No live trader, operator bin tools or system services are deployed.
Set `REPLAY_REPORT` outside Git to have night batches update the private report; DONE requires a scored
test result bound to the current execution spec/code. Status and audit require no model or HTTP calls.
