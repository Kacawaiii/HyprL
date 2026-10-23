# Replay revision 13: finish the original test

Execution revision 17 implements the uniform October 23 operator decision.
An unresolved original attempt pair with recorded stdout/stderr bytes, or an
already consumed zero-byte retry, resolves the cell as MODEL_FAILURE. It creates
no forecasts or trades and is never re-asked. A dead zero-byte INTENT retains its
one retry. Active owners, changed evidence, quota stops and credit stops retain
their independent gates. The envelope and partial stream digests are preserved,
verified and externally committed before scoring; partial output is never decoded.

Replay dispatch is checked in code before each reservation, in UTC. It opens at
20:30, forbids new starts at or after 06:55, and retains the full 900-second timeout
before 07:00. Consequently the latest full-timeout start is 06:45. An injected
window cannot bypass this replay guard. Forward uses its independent session
window. Cached responses remain available offline at any hour.

State duel-v13 explicitly references original first-pair responses from duel-v12
and the earlier admitted sources by digest. Original v12 test-start markers and
the already scored development result remain immutable. The new continuation
marker has a distinct publication key; old test calls must still prove their
original published start preceded dispatch. This continues the same started test;
it does not run that test again. Old stores remain rejected by configuration checks.
Input resealing preserves every context, cache binding, paired exclusion and the
frozen primary baseline. Historical forward records retain the v12 freeze too.

Models, prompts, schemas, costs, exclusions, metrics and the primary endpoint are
unchanged. Final output must include every development and test panel, per-group
MODEL_FAILURE rates, a:1d Brier skill versus the frozen baseline with its declared
moving-block bootstrap, and the paired Opus-minus-GPT difference. All replay
results remain EXPLORATORY / APPROXIMATE_POINT_IN_TIME. Winner wording requires
the primary paired confidence interval to exclude zero. A failed or incomplete
run must never be presented as a complete scored table.

Revision procedure: preserve source state and reservations; commit new spec,
continuation, freeze and input seal; run research and repository checks; push
only the task branch. Deployment and continuation must respect host ownership
rules and retain exact expected-commit checks. Production tools are not edited
in place. The scorer's cached SCORED binding prevents repeated outcome scoring.
