# Cockpit Lab v1 (Model Lab, research registry, observability views)

Route `/lab` in `apps/web`, six tabs over the already integrated APIs. It adds no endpoint and changes no
backend rule. Data shown is the synthetic demonstration or the authorized frozen replay; nothing is captured,
trained, sent to an external model or traded from this page.

| Tab | Reads | What it answers |
|---|---|---|
| Datasets | `/api/v1/lab/jobs`, `/lab/jobs/{id}/results` | products, period, target, horizon; admissible decisions; exclusions with reasons; fingerprints |
| Experiments | `/lab/jobs`, `/lab/jobs/{id}/results`, `/lab/models` | follow a job (progress, log, limits), compare with ZERO / TRAIN_MEAN on the same test decisions, splits, artifacts, hashes, reproduction |
| Models | `/lab/models` | internal vs external adapter vs frozen reference; declared capabilities; provided and absent outputs; limits |
| Predictions | `/api/v1/observability/predictions[/{id}]` | per-prediction inputs, outputs, pending vs realized labels, append-only label history |
| Monitoring | `/observability/monitoring`, `/references`, `/health` | freshness, inference, drift, regimes, performance by product/period/regime, edge vs baselines with sample and method |
| Hypotheses | `/api/v1/research/hypotheses[/{id}]`, `/comparison`, `/proposals` | statement, mechanism, falsification, frozen criteria hash, trial history, prices-vs-events readiness |

## Decisions that follow from the server contract

- **Writes are not sent from the page.** `/api/v1/lab/*` rejects any request carrying an `Origin` header and
  a browser always sends one on POST. The page therefore *prepares* the exact command (dataset, experiment,
  reproduction, cancellation) and never sends it; the token is read from `$HYPRL_MODEL_LAB_TOKEN`, never printed.
  Nothing weakens the server rule. A same-origin GET carries no `Origin`, so reading works.
- **The operator token** (needed for every `/lab` GET) is held in module memory only: not in the URL,
  localStorage, cookies or logs. A reload forgets it; "Forget token" drops the data from view.
- Research and observability endpoints are public read-only, like the other source views.
- A server started without `--model-lab-root` / `--research-root` answers 503; the page says "Not configured".

## Beginner and Expert

The mode, product and model live in the URL and survive tab changes and the mode switch. Beginner phrases the
outcome (criterion met or not, pending or realized, which cause is raised) and keeps detail one click away.
Expert adds exact features, hashes (dataset, features, snapshot, artifact, contract, method, regime), runtime and
source hashes, per-model limits and criteria JSON.

## What the page refuses to claim

- An absent optional output (target price, class, probabilities, quantiles, scenarios) reads "not provided".
- A pending label is shown as PENDING, never filled; label versions are listed, the prediction is unchanged.
- The four diagnosis causes (missing data, technical degradation, drift, performance drop) are separate rows.
- An unmeasured inference service reads NOT_OBSERVED, "unknown, not healthy".
- Every edge sentence names its sample and method and states that nothing establishes (or refutes) an edge.
- Negative results stay visible ("criterion not met; this negative result is kept").
- Backtest numbers are labelled synthetic and "not a trading result".

## Verification

`apps/web/src/test/lab-lib.test.ts` (pure helpers) and `lab.test.tsx` (views) run over **real responses captured
from the synthetic demo** (`python -m scripts.trading_lab.research.demo --bars 120`), trimmed, in
`labFixtures.ts`. Part of the vitest run of the CI `web` job. A pending ledger row is derived from a real one by
removing its label (the demo store has none); this is the only constructed record.

Browser check: **BLOCKED** (no browser binary on the host; downloading one is outside the permitted network scope).

## Run it

```
PATH=/home/agent/venv/bin:$PATH python -m scripts.trading_lab.research.demo --root var/trading_lab/cockpit-demo --bars 120
HYPRL_MODEL_LAB_TOKEN=<32+ ascii chars> python -m scripts.trading_lab.app_api.server --port 8791 \
  --research-root var/trading_lab/cockpit-demo/registry --model-lab-root var/trading_lab/cockpit-demo/lab
cd apps/web && npm run dev   # proxies /api to 127.0.0.1:8787; use --port 8787 or edit the proxy
```
