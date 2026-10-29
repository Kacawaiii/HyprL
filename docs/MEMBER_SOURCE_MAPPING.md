# Member source mapping

The v2 adapter accepts the explicitly pinned revision-sixteen research baseline
only through `member_baseline_projection_spec_v1.json`. Its preimage must hash to
the source score's exact identity. The original v2 baseline preimage remains
pinned, and every field except `spec_hash` must be identical: sample count,
successes, panel and outcome digest. Unknown identities or a different sample
remain refused. Original score/decision hashes are preserved; the public member
baseline retains its original identity and reference. Export schemas and guards
are unchanged.

Brief projection first joins the delivered metadata to its original observed
report. It can join an exact story headline or a unique grouped event headline.
For a group it exports only the existing source headlines and URLs, never the
editorial group wording as a source quote. Primary source types can identify an
existing supported fact kind. Ambiguous groups, missing sources, future receipts,
unsafe prose and unsupported topics remain excluded.

A private `calendar_config` pointer can select the calendar path from a complete
daily input generation. Date markers retain `precision: date`, a null release
time and their observed dataset evidence. The adapter never upgrades a civil-date
exclusion to an attested release hour.

Only the member producer and private data need deployment. The site reader,
schemas, model outputs, source stores and research ledger remain unchanged.
