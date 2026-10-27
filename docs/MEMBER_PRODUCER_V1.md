# Observed member producer, v1

The offline producer reads the running radar, delivered brief metadata, scored
LIVE_FORWARD output and confirmed reveal receipts. It writes the existing closed
handoff v1 contract and invokes the unchanged member exporter. It never acquires
sources, calls a model, opens account/price stores or publishes research decisions.
The site continues to read only the normalized snapshot. Claims stay empty.

Run from a pinned clean checkout with the dedicated member environment:

```sh
"$MEMBER_PYTHON" -m scripts.member_export.producer \
  --config "$MEMBER_PRODUCER_CONFIG" --expected-commit "$TESTED_SHA"
```

The private JSON configuration has these absolute-path keys:

| Key | Existing observed input or dedicated output |
| --- | --- |
| `radar_db` | Running radar SQLite database; opened `mode=ro`, query-only, one read transaction |
| `brief_dir` | Existing brief output directory with `deliveries.jsonl` and dated metadata |
| `forward_dir` | Running LIVE_FORWARD version directory, including immutable score runs |
| `reveal_dir` | Confirmed research disclosure delivery markers |
| `ledger` | Existing local public ledger checkout, read through local Git commands only |
| `calendar_file` | Existing agent-v2 calendar output |
| `revenue_file` | Existing equity-hunter revenue frame output |
| `handoff` | Dedicated member handoff file |
| `output_dir` | Existing dedicated snapshot directory |
| `status_file` | Private producer status, counts, exclusion codes and score/snapshot digests |

Keep the configuration and all outputs outside Git. Do not point inputs to
synthetic outputs, account stores, holdout data or closed pilots. No credentials
are needed. No network operation exists in the producer. Output files use atomic
replacement, flushed bytes and directory fsync. The snapshot is mode 0440; the
handoff/status are 0600. A separate producer lock serializes timer and reveal
refreshes. Input/schema failure preserves the previous validated snapshot and
returns a fixed code without source bodies, paths or exception text.

## Admission and known gaps

- Radar supports primary SEC 8-K and Fed story records. An original publication
  timestamp, observed receipt time, public URL and headline digest are required.
  An Atom update timestamp is never substituted for original publication.
  Private, advisory or model-name prose is omitted with a count. Repeated
  URL/headline identities are deduplicated; at most 500 current eligible records
  are published. The original source time remains visible even for older facts.
- Briefs are **factual projections**, not the complete Discord edition. Only
  confirmed SENT deliveries with all messages sent, metadata predating delivery,
  a matching preexisting live radar report and supported source headlines enter.
  The metadata's recorded report digest is rechecked. Unsupported topics and
  grouped headlines without an exact source match are counted. A general
  inflation topic is insufficient to label an article CPI. The existing delivery
  ledger does not attest full archived Markdown bytes or expose Discord message
  links; this producer makes no stronger receipt claim.
- Forecasts remain empty: the real scorer measures benchmark-relative
  `p_outperform` at 1/2/5/10-day horizons. Handoff v1 accepts only
  `close_above_open`, a nonce, a precommitted base rate and a hash of a normalized
  per-forecast commitment. The real ledger commits original whole decisions.
  Rehashing or relabeling these observations would fabricate a different proof.
  Every actual score row is counted as incompatible, rather than transformed.
- Failures require a confirmed disclosure marker, a reveal commit present in
  the public remote-tracking ancestry, the exact committed README digest and a
  matching NO_DECISION failure digest. Failure bodies and model identities are
  never copied. Mapping is fixed: coverage invalid → invalid, usage limit →
  missing, insufficient call window → late; other codes → unscorable. The public
  agent names are HypR-A1 and HypR-E1. Availability is conservatively the confirmed
  notice timestamp. Current failures are delayed until 16:15 New York even on
  shortened sessions; the original reveal sidecar has the full exchange gate.
- The existing calendar contains all-day risk markers and Nasdaq civil-date
  guards without confirmed event hours/per-event public receipts. V1 requires
  an exact `scheduled_at` and source provenance. These entries are counted and
  withheld; midnight/23:59:59 are not sold as release times.
- Hunter revenue frames retain only symbol/concept/frame → end/value. They omit
  CIK, actual starts, filing accession/URL, original availability and revision
  identity. Three truly comparable SEC filing quarters cannot be reconstructed
  from that cache. Symbols are counted and withheld; no fabricated quarter
  starts, filing links or file-mtime provenance.

A successful partial projection exits zero with `status=PARTIAL` and explicit
counts. It does not claim that the withheld feeds are ready. An input/export
failure exits one; the dedicated systemd failure unit sends a fixed notice to
`alertes`. Unknown/fallback delivery is an alert failure, never confirmed Discord.

To complete the missing feeds, the operator must commission a new versioned
member contract and matching site support for the actual forecast event,
horizon, original preimage proofs and predeclared baseline. Calendar production
must retain source receipts and distinguish dates from confirmed times. Hunter
must retain original filing facts/receipt/intervals and revisions. The producer
cannot create that lost evidence after the fact. No v1 schema or frozen research
spec is relaxed here.

## Dedicated user jobs

Render outside Git; the renderer never installs units:

```sh
"$MEMBER_PYTHON" -m scripts.member_export.jobs \
  --checkout "$MEMBER_CHECKOUT" --python "$MEMBER_PYTHON" \
  --config "$MEMBER_PRODUCER_CONFIG" --notify "$APPROVED_NOTIFY" \
  --research-python "$RESEARCH_PYTHON" --research-environment "$RESEARCH_ENV" \
  --expected-commit "$TESTED_SHA" --output "$NEW_UNIT_STAGE"
systemd-analyze --user verify "$NEW_UNIT_STAGE"/*.service "$NEW_UNIT_STAGE"/*.timer
```

Exactly five new files are rendered: member producer service/timer, fixed-code
failure alert service, research reveal service/timer. Producer runs every thirty
minutes with AF_UNIX-only socket creation; local Git reads need no egress.
Reveal runs at 16:15 America/New_York with the original research ownership and
publisher locks, and starts the member producer through `OnSuccess`. Both jobs
route failures to the new alert unit. Code pins reject a dirty/advanced checkout.
Keep this worktree pinned while the timers use it.

Install only these five new agent user units after verification and operator
scope authorization. No existing research/trader/site units need rebinding:
the reveal sidecar can run from this separate pinned checkout, using the same
unchanged frozen execution tree and shared private locks. This avoids advancing
or editing any other worktree. Do not run a legacy research installer or reseal
inputs. The explicit reveal authorization is a named exception for public
original preimages and the ledger branch; ordinary code pushes remain confined
to the task branch. Follow [the reveal runbook](FORWARD_REVEAL.md) for byte,
pre-open receipt, close-time, append-only and notification checks. October 20
intraday remains UNPUBLISHED.

Rollback disables/stops only the two newly installed timers, drains the two new
jobs, then removes these five unit files and reloads user units. Preserve public
ledger history, confirmed delivery markers and last validated member data. Never
restore earlier personal data, mutate original research records, or restart the
site/trader/bot. Existing code/dependencies imported for member support are kept
separate from the frozen root packaging: router pins are in
`requirements-alerts.txt`, member pins in `requirements-member.txt`.
