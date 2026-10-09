# GPT transport repair and daily pre-flight

The October 9 GPT failures were transport parsing failures. Both final answers
contained 70 views and passed the unchanged local schema, universe, direction,
HTTPS and publication-time checks. Their stderr files were empty. Codex 0.160.0
emitted duplicate `item.id` keys in web-search envelopes: its item ID followed by
the web call ID. `strict_json` rejected those events before reading the final
answer. The two attempts contained four and six such envelopes respectively.

Evidence (private bodies remain outside Git):

| Attempt | Stdout bytes | Stdout SHA-256 | Valid final views |
| --- | ---: | --- | ---: |
| First | 50217 | `e7c85d15fa670628fff91d02c9655a3a3ba48b456ca85d8c6525a5d2cff5ad90` | 70 |
| Retry | 67036 | `498a9a1170bcc57f723b932244b2221f94c7fd726cb8952a53cc20dd2e06eed9` | 70 |

Both stderr digests are
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
There was no truncation or invalid model answer. Analyst/reviewer instructions,
strict output schemas, local validators and preregistration are unchanged.
No historical views are repaired or reissued.

The GPT transport decoder permits exactly two string `id` values only at
`$.item` when its single `type` is `web_search`. Every other duplicate remains
invalid, including every duplicate in the final answer. The same decoder feeds
the forbidden-tool and quota checks, including failed CLI exits. Security verdicts
still precede quota handling. The regression uses synthetic web envelopes and
35 synthetic assets (70 abstentions), through `ModelRunner.infer` and the full
analyst validator, without changing any source fixture shapes.

The task authorizes a separate `gpt_preflight` dispatch, capped durably at one
model call per UTC day in the shared production budget bank. It has no retry,
a maximum 120-second model timeout, low reasoning effort and a 64,000-character
stdout limit. It does not consume the six analyst/reviewer calls. Grant expiry,
pause, the global owner lock and the restricted GPT command still apply. A failed
call stays consumed even after restart or a change of runtime path.

The probe checks the deployed strict schema against its code binding, then uses
the same GPT model, runner, schema and web tool host as the analyst. Its prompt
contains no market context or analyst skill: it asks for one Python documentation
search and exactly `{"regime":["PREFLIGHT_OK"],"views":[]}`. Success requires a
completed web search with returned search results and a valid empty-view answer;
a model claiming success without web evidence fails. Codex returns successful
search hits as `text_result` entries. Transcripts and the latest
`gpt-preflight.json` stay private. The probe never constructs a research store,
records predictions, or invokes paper execution.

Install the paired user units in `deploy/systemd/hyprl-trader-gpt-preflight.*`
alongside the other trader units. The timer runs Monday–Friday at **10:30 UTC**,
before the 12:00 run and the deployment deadline. It does not catch up missed
timers. The service loads the operator's login environment, uses the existing
deployed worktree and private grant/runtime, and has a three-minute unit timeout.
The CLI action is `gpt-preflight`; it can also be run manually for installation
verification, using the same daily cap. Its failure returns nonzero and alerts;
a later GREEN transition alerts recovery. First-time GREEN is quiet. A duplicate
invocation returns ALREADY_ATTEMPTED without replacing the day's result.

Recovery and health polling persist their last state in the budget ledger.
Repeated NO_RECOVERY_NEEDED (or another unchanged state) no longer appends alerts,
including across process restarts. State transitions still alert, and actual
recovery attempts always alert their outcome. Health alerts a fault once and its
HEALTHY recovery once. Historical alerts and dispatch reservations are retained.
