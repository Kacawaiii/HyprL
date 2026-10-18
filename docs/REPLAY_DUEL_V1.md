# Offline Opus / GPT replay duel (protocol revision 5)

This is an additive research protocol. It preserves all four parent strategies, the Jul–Aug development
period and once-only Sep 1–Oct 9 test, the point-in-time snapshot builder, universe, costs and equity fill
conventions. Previous protocols, configurations, decisions and stores are retained verbatim and never
migrated. The new spec and Python binding pin every amendment. This work authorizes no trading,
training, live GPT role, source collection or protected BTC/ETH research.

Both subscriptions receive identical analyst/news prompt bytes, schemas and sealed snapshots. Each
family supplies its own analyst and reviewer for a; b shares that analyst; c uses the news analyst; d
filters c trades using exact linked 8-K accessions accepted before the decision. Review prompts have
identical instructions/schema but contain their own family's analyst output. Every fourth dev session
also repeats b with the same anonymization seed for both families. Saved records bind spec, code,
configuration, input proof, snapshot, shown context, prompt, transcript and decision digests.

The probability contract changes for **both** models before any new test call. Every eligible asset /
fixed horizon requires a numeric probability in [0,1], including abstentions and rejected trades. UP
requires p>0.5, DOWN p<0.5; abstention permits any p. A rejected trade retains a conservative forecast;
there is no automatic conversion to p=0.5. An invalid/missing forecast is a failed day, never imputed.
A/b forecast 1d and 5d. C/d forecast fixed 2-session and 10-session panels independently of the chosen
2–10-session trading horizon. D retains every c probability; primary evidence controls trading only.
Thus a model cannot avoid probability scoring by abstaining, choosing a different horizon or failing
to establish a primary citation.

The probability target is gross outperformance of SPY over the identical interval. The parent crypto
1d fill anchors remain 13:31 UTC to +24h, with midnight marking. The new crypto-relative outcome uses
SPY trades in the same one-second anchor windows while the exchange is open and its last actual raw
session close while shut. Missing synchronized marks block scoring. Crypto 5d remains explicitly
excluded from scoring, while its probability is required. Period-crossing horizons are censored and
counted. The dev scorer never reads test prices. Outcome acquisition is also offline and digest sealed.

Models: `claude-opus-5-5` and `gpt-6.1-sol`. Sonnet is excluded. The original operator probes are regraded
from digest-pinned private files, including GPT's 0/35 exact recall and 18/36 forced direction, versus
always-up 21/36. Clean sessions, pre-call hashing and attribution of these probe files are operator
attested. Non-flagging does not rule out latent knowledge. GPT's installed Codex CLI does not normally
report server identity: evidence is labeled CLI_REQUEST_ONLY when absent, never described as a server
attestation. Any emitted conflicting identity is refused.

GPT runs `codex --no-daemon exec` with strict config, never `--search`, with `web_search="disabled"`, shell/code-mode host disabled,
empty cwd, ignored user config/rules, ephemeral sessions, disabled apps/plugins/hooks/skills/memory and
multi-agent tools, pinned model/medium reasoning and ChatGPT login. API-key environment fallbacks are
removed. Opus uses the same neutral system instruction, no tools, empty cwd, no MCP, hooks, skills,
local instruction discovery, auto-memory or session persistence. Transport output is checked for tool
use even on failure/quota. Actual inference remains unproven until the input and night gates permit it.

Each subscription has an independent durable ceiling of 60 calls per UTC night, 00:30–05:30, including
analysts, reviewers and validation retries. The same legacy budget file is used; inherited Opus usage
counts. Dispatch is serial and pairs alternate first family by day. Quota stops that subscription for
the night; the other can continue within its own budget. A completed response resumes locally; an
unresolved intent blocks without blind retry. No timeout refunds a call. The common research lock
excludes the old runner. No live timer, operator bin tool or engine is changed.

The committed test freeze pins both families' configuration payloads/hashes, common 28-session plan,
probe evidence and current execution digest. A separate committed input seal binds genuine original
vintage and universe/identity proof plus exact offline cache objects before test. An older protocol's
test-start marker prevents a replacement test. Both dev periods and the anonymized dev subset must be
complete and scored before either family claims its test. Each irreversible test claim binds the same
freeze, input seal and dev result (including the frozen base rates). Scored results are reused only
when all execution, proof and decision bindings match. DONE requires both scored tests and pairing.

Per panel, outputs include decisions/trade counts, abstention rates, hit counts and descriptive Wilson
intervals; Brier and clipped log loss on **all** eligible forecasts; constant-0.5 and base-rate scores;
net expectancy in R; daily marked fixed-initial-capital P&L Sharpe and peak-relative drawdown; cluster /
day Kish ESS and day ICC design effect. The probability base rate is Laplace (wins+1)/(n+2). During dev,
only outcomes already closed before each decision enter the expanding baseline. Test rates are fixed
from censored dev labels before any test call. No rate is fitted to test outcomes.

The head-to-head table is generated beside each private score result. Formal comparisons pair the
same day, instrument and fixed horizon, compare Opus-minus-GPT Brier and log loss, aggregate equal-weight
daily mean differences and flip signs of nonoverlapping 10-session blocks anchored at period start.
P-values are exact, two-sided, with Holm adjustment across all named panel/loss tests. With only 28 test
sessions there are at most three blocks; coarse p-values and residual cross-block dependence limit
inference. Candidate counts do not constitute independent observations. This replay cannot establish
general model superiority from an underpowered sample. C/d forecasts coincide by construction within
each family, and their duplicate comparisons are explicitly included in the conservative correction.

CLI: `python -m scripts.trading_lab.replay_duel.cli --data-root <private-data-root>
--authorization <unchanged-http-grant> status|audit|freeze|seal-inputs|night|score`.
The data root stays the parent research root so budgets/cache/ownership are shared; the new private
`duel-v1` state directory contains only new decisions. `duel_input_proof.json` requires schema
`replay-duel-input-proof-v1`, the new spec hash, true original-vintage/universe attestations, evidence
file digests and prediction-cache file digests. Scoring also needs `outcome_cache` digests. Booleans
alone do not supply the evidence. Do not fabricate a proof from a present-day archive.

`deploy/research/install_duel_timer.py` can replace only the existing replay research user units after
source commit/checks, executable/version checks, unit verification, idle-service check and a new
private backup directory. It pins the exact source SHA, serial execution, bounded CPU/memory and
00:30 scheduling. The environment supplies the parent data root, unchanged HTTP authorization,
`REPLAY_PROBE_ROOT` and `REPLAY_DUEL_REPORT`, never API keys. Missing evidence stops before dispatch.
