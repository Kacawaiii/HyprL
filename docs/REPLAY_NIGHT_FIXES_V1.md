# Continuous research replay and forward recovery

Revision 13 retains the scientific panels, prompts, protected-symbol exclusions,
once-only TEST gate, 400-call UTC-day ceilings, independent subscription stops,
and no API-key fallback. Replay now runs continuously, one worker per model.
Both DEV families must finish and be scored before either TEST family starts.
The research timer checks every fifteen minutes; an active provider stop prevents
dispatch until an explicit reset, or the next UTC day when no reset is supplied.
Credit exhaustion remains a durable operator-cleared stop.

An interrupted call writes directly to private stdout/stderr files. A dead owner,
zero bytes on both streams and no previous retry permit exactly one redispatch.
Both reservations and both attempt records remain. Any bytes, unknown ownership,
second interruption or unknown API spend keep the no-retry stop.

The explicit continuation artifact binds thirteen received revision 12 envelopes
and one GPT July 7 INTENT. Original envelopes and protocol hashes are immutable.
Exact prompt/schema/family/stage matches reference received answers without another
model call. New records identify those original-protocol references. Old model
stores still fail the current configuration checks; there is no implicit migration.
The identified zero-byte INTENT alone may consume one further reservation.

The October 20 live archive contained news and assets but zero daily bars: the
collector stopped on HTTP 403 when its daily SIP query included the recent interval.
Daily queries now stop after the last completed interval. Full paginated history
still starts January 2025, covers the 200-session and 50-session rules and remains
raw; live source receipts must precede the immutable input cutoff.

More than 20% of fixed names missing or more than 50% of potential news candidates
excluded for missing/contradictory source data is a PIPELINE_DEFECT: BLOCKED and
an alert. Liquidity, company eligibility and genuinely short histories remain
separate eligibility filters. The same guard applies to live and replay admission.

For the single October 20 intraday recovery, the immutable input cutoff precedes
both calls. T is interpreted as the actual common completion of both decisions,
recorded separately from that cutoff. The first actual equity SIP trade strictly
after T+30 seconds is the entry, within the declared one-second fill coverage.
Both decision hashes and the snapshot hash must reach the pushed hash-only ledger
and Discord strictly before T+30 seconds; otherwise the run is BLOCKED. No clock
is moved after an attempted recovery. Fixed-universe 1d still exits five minutes
before close; other equity horizons keep their declared session close. Crypto
forecasts remain, but actual crypto trade labels are explicitly unavailable under
the unchanged bars-only grant; minute opens are never substituted for trades.
The regular Wednesday 09:05 New York schedule and pre-open publication remain.

The publisher explicitly pushes origin ledger/forward-duel and alerts on failure.
A separate research dry run participates in the 22:45 UTC drill, after the existing
drill service, with no model call or publication. Login refresh keeps its existing
calendars and adds 04:00 UTC. The replay has its own ownership lock, leaving the
forward pipeline available during continuous replay. No trading engine or orders
are involved.
