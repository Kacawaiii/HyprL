# Offline local triage research

This experiment ranks stored equity headlines or candidates before an LLM sees
them. It has no trader integration, inference service, deployment, network client,
order capability or price feature. Its inputs, fitted weights and detailed audit
remain private. Public evidence contains counts and digests only.

`scripts.trading_lab.local_triage` accepts JSONL rows with `session` (ISO date),
`teacher`, `text`, `label` (0/1/null), optional `headline_fingerprints` and
`payload_bytes`. Use only the headline titles available at the decision cutoff
for `text`. Keep teacher reasoning, decisions, returns and later prices out of
features. A null label is unknown, never a negative example.

The source adapter checks the teacher's shown-context hash and uses the original
news IDs. Candidate positives mean the teacher chose UP or DOWN; headline
positives additionally require that headline's ID in the active view. ABSTAIN and
unmentioned headlines supply weak negative labels. They do not prove an event is
immaterial, duplicated or already priced in. Failed or absent teacher outputs
are excluded. Claude and GPT teachers are evaluated separately.

Training is restricted to 2025 through August 2026, and testing to September and
October 2026. Actual matched training coverage is July–August 2026. The vectorizer
fits only training titles. A fixed word TF-IDF (unigrams/bigrams, at most 12,000
features) and balanced logistic regression use one CPU thread. Ranking retains
the top 10% or 20% within each session, rounding upward. Evaluation reports
precision, recall, false negatives, monthly results, random expectations, a
simple event-word comparator and intervals resampled by session. A second view
excludes every test candidate sharing a normalized headline with training.
This is exploratory: historical replay inputs are approximate point in time and
the underlying teachers may already know historical events.

Example with synthetic or operator-provided private inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.trading_lab.local_triage \
  --dataset candidates.jsonl --output metrics.json --models-dir fitted-models
```

The CLI refuses batch training during 11:30–17:15 UTC and refuses to overwrite
prior results or model artifacts. Scikit-learn, NumPy and joblib are required for
fitting; it does not install packages. Fitted artifacts are for local research.

`scripts.trading_lab.local_token_audit` accounts for a saved CLI result. It uses
the final cumulative result once, adds distinct main/helper `modelUsage` entries,
and does not add aggregate usage, streaming frames, iterations or thinking-token
subcounts again. Inconsistent totals fail. Cache share means cache-read tokens
divided by fresh input + cache creation + cache reads. New work means fresh
input + cache creation + output. Neither measure identifies subscription debit
weights or money saved. Missing helper counters are explicitly unknown.

The candidate prototype's initial 20% selection retained 36 of 61 Claude-active
cases (59.0% recall, 19.6% precision), while excluding 25 active cases. That is
insufficient evidence for automatic filtering. GPT training was blocked because
its 1,600 matched candidate labels contained no positive examples; the September
GPT news-arm outputs were absent. No positive labels were fabricated.

Headline payload reductions are measured in bytes, not provider tokens or calls.
The existing news job batches many candidates into one model call: retaining
20% of candidates does not remove 80% of calls. Fixed instructions, market
context, output and web work remain. Any token scenario must multiply the
payload reduction by the measured eligible share of total work. Prospective
shadow evaluation and a reviewed method amendment are prerequisites for any
operator deployment; this experiment changes no existing validation or spec.

Synthetic checks:

```sh
python -m pytest tests/research/test_local_triage.py \
  tests/research/test_local_token_audit.py -q
```
