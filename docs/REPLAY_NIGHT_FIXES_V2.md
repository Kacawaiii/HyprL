# Independent resets and full forward call timeouts

Revision 14 corrects two execution issues measured after revision 13 deployment.
The October 20 GPT forward call lasted almost eight minutes. A four-minute
regular timeout, or a nine-hundred-second service lifetime including snapshot
construction, would cut a valid call on Wednesday.

Regular forward now submits both families concurrently, with the full 900-second
per-call timeout. Its dispatch interval is 09:05–09:25 New York, and a call is
refused before reservation unless at least 900 seconds remain. The user service
has 1,800 seconds, leaving the call's own timeout in control. Hash publication
remains strictly before 09:30. Stocks retain open plus thirty seconds; forward
crypto uses decision time plus twenty-five minutes (09:30), so publication
precedes its entry too. Replay crypto retains its original fifteen-minute offset.
The revision 14 execution amendment supersedes the inherited forward offset text.

A quota-stopped replay worker is resubmitted at its own reset while the other
worker continues. The coordinator observes completions immediately and checks
pending resets at least once a minute. Cap stops resume on UTC rollover. The
explicit provider notice for the observed Opus stop gives 17:10 UTC; parsing that
clock avoids unnecessarily waiting until midnight. Credit stops remain durable.

The maintenance checkpoint waited for all running calls; no process was killed.
Thirty received envelopes across revisions 12 and 13 are bound by digest and
referenced only for identical prompt/schema/family/stage. The original unresolved
INTENT and the received quota envelope are preserved separately, without another
zero-byte retry, an answer retry or a budget refund. Old stores still fail the
current configuration check. The 128 contexts are unchanged and explicitly resealed.

The successful October 20 intraday proof remains revision 13. The scoring registry
binds its original immutable freeze, code hash, publication and decision hashes;
no record is relabeled as revision 14. Result rows identify their source protocol.
That run's hashes reached the ledger and Discord 26.383255 seconds before its
entry boundary. Fills and matured outcomes remain to be collected.

Replay and forward publication now serialize access to the shared hash-only Git
ledger. Push timeouts alert and block; Discord timeouts block. Publication checks
its deadline before Git/Discord work as well as after confirmation. No trading
engine, orders, raw data publication or protected holdout request is involved.
