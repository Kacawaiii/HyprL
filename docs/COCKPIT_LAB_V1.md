# Cockpit Lab v1 (Model Lab, research registry, observability views)

Route `/lab` in `apps/web`, six tabs over the integrated APIs, and a five-step journey (Data → Model →
Experiment → Results → Monitoring) that is **executable from the page** through the authenticated local
lab listener, never through the read-only API. Data is clearly labelled synthetic or the authorized frozen
replay (read-only); real-data datasets and training are shown as **WAITING_AUTHORIZATION**. Nothing is
captured, trained on real data, sent to an external model or traded from this page.

| Tab | Reads | What it answers |
|---|---|---|
| Datasets | `/api/v1/lab/jobs`, `/lab/jobs/{id}/results`; POST `/lab/datasets`, `/lab/jobs/{id}/cancel` | choose products, period, target, horizon; build in a worker; follow, cancel; admissible decisions; exclusions with reasons; fingerprints |
| Experiments | `/lab/jobs`, `/lab/jobs/{id}/results`, `/lab/models`; POST `/lab/experiments`, `/lab/monitoring`, cancel | launch on a dataset with a model whose declared contract fits (mismatches named before sending); follow a job (progress, log, limits); cancel; compare with ZERO / TRAIN_MEAN on the same test decisions; splits, artifacts, hashes; run a reproduction; open the monitoring |
| Models | `/lab/models` | internal vs local external adapter vs frozen reference; declared capabilities; provided and absent outputs; limits; how an operator registers an adapter (never from a page) |
| Predictions | `/api/v1/observability/predictions[/{id}]` | per-prediction inputs, outputs, pending vs realized labels, append-only label history |
| Monitoring | `/lab/jobs/{id}/results` (monitoring job); `/observability/monitoring`, `/references`, `/health` | the experiment's own test predictions against its validation reference; freshness, inference, drift, regimes, performance by product/period/regime, edge vs baselines with sample and method |
| Hypotheses | `/api/v1/research/hypotheses[/{id}]`, `/comparison`, `/proposals` | statement, mechanism, falsification, frozen criteria hash, trial history, prices-vs-events readiness |

## Decisions that follow from the server contract

- **Writes go to the lab listener only, from its own page.** The cockpit must be served by the same loopback
  listener that runs the lab (`--dist-root` with `--model-lab-root`). That listener admits an `Origin` only
  when it is exactly `http://<Host>`, `Host` is a literal loopback name with its own port (a DNS-rebound name
  is refused) and `Sec-Fetch-Site`, when sent, is `same-origin`; the bearer token stays required and no CORS
  grant exists. From the Vite dev server (another origin) the controls are refused with an explanation.
- **One POST per click**, no retry and no optimistic state: the server's answer is the outcome. A network
  failure says "nothing is known to have been queued". The equivalent terminal command (token from
  `$HYPRL_MODEL_LAB_TOKEN`, never printed) stays available for reproduction, open by default in Expert.
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

- `apps/web/src/test/lab-lib.test.ts` (pure helpers: contract fit, model role, journey order, refusal phrasing)
  and `lab.test.tsx` (views, including the POST bodies, bearer header, cancel, launch, monitoring, refusals)
  run over **real responses captured from the synthetic demo** in `labFixtures.ts`. Constructed records: a
  pending ledger row (label removed from a real one) and the monitoring-job envelope around the captured
  monitoring view. Part of the CI `web` job.
- `tests/model_lab/test_browser_controls.py`: the same-origin rule (DNS rebinding, cross-site, wrong port,
  https, `null`), and the page journey over HTTP with real workers.
- `apps/web/e2e/lab-journey.cjs`: headless Chromium against a real listener and real synthetic workers:
  locked page, wrong token, illegal configuration, dataset build (keyboard submit), experiment with the local
  external adapter, baselines, Expert hashes and kept selection, monitoring, cancel, foreign-origin and rebound
  refusals, refused-control message, keyboard tab navigation with reduced motion, and 390 px on four tabs
  without horizontal overflow; fails on any page or console error. CI job `lab-browser`.

## Run it

```
PATH=/home/agent/venv/bin:$PATH python -m scripts.trading_lab.research.demo --root var/trading_lab/cockpit-demo --bars 120
(cd apps/web && npm run build)
HYPRL_MODEL_LAB_TOKEN=<32+ ascii chars> python -m scripts.trading_lab.app_api.server --host 127.0.0.1 --port 8796 \
  --dist-root apps/web/dist --research-root var/trading_lab/cockpit-demo/registry --model-lab-root var/trading_lab/lab-ui
# open http://127.0.0.1:8796/lab/datasets and enter the token; browser check:
LAB_URL=http://127.0.0.1:8796 HYPRL_MODEL_LAB_TOKEN=... NODE_PATH=<playwright node_modules> node apps/web/e2e/lab-journey.cjs
```
