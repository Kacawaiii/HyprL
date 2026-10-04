# Paper replay v2: one out-of-sample window

The frozen paper model shows **no positive edge out of sample** in this window.
Both time-series rank ICs are negative. BTC makes no trades; ETH makes one
short entry and exit and loses money after the frozen synthetic costs.
Nothing was tuned or inverted after seeing these results.

| Product | Rank IC | Fills | Net return | Max drawdown | Annualized Sharpe | Final equity (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC-USD | -0.072580975304 | 0 | 0% | 0% | undefined (constant equity) | 100000.0000000000 |
| ETH-USD | -0.079268514113 | 2 | -0.039433576578% | -0.039433576578% | -2.130684997836 | 99960.56642342159949999445580871545 |

This table rounds IC, percentages and Sharpe. The [BTC result](../data/crypto/paper_replay_v2/BTC-USD.json)
and [ETH result](../data/crypto/paper_replay_v2/ETH-USD.json) contain the full decimal
values. The cockpit displays those values verbatim under **Replay hors échantillon
(v2)** on Paper. Its two accounts and curves are independent of the live shared
shadow portfolio; their equities are never combined.

## Frozen training and boundaries

One existing Ridge fit per product, with alpha 1.0 and exactly the v1 features,
label horizon, model and parameters. V2 changes only the training end to
2026-04-30T23:00:00Z. The v1 spec hash, model files and live session CLI defaults
remain unchanged.

Each fit uses 6489 usable rows from the committed Coinbase history v1 corpus.
The first fitted opening is 2025-08-02T01:00:00Z; the last is
2026-04-30T19:00:00Z, whose four-hour label ends in April. All training input bars,
including the unusable tail, are checked before features or labels are built.
The [model manifest](../data/models/paper_v2/manifest.json) records input bounds,
training hashes, fitted hashes and the frozen [v2 specification](artifacts/paper_model_spec_v2.json).

- Corpus hash: 688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b.
- V1 model spec hash: 830f52271af4eba887086d6b00885cc34f5d13562dae24cad45986a86466e033.
- V2 model spec hash: 76fbec5ab1fcd809b04cba029c86e5b1c5be1dd61ecfe7b33751365e62d1d038.
- No source bar at or after 2026-08-01T00:00:00Z is consumed. The corpus manifest
  is checked before candle-file access, and each input opening is checked before
  price/feature use. The September–November protected holdout remains unobserved.

## Replay and scoring

PaperEngine receives 2203 observed hourly bars per product in chronological order,
from 2026-05-01T00:00:00Z through 2026-07-31T23:00:00Z. Each is delivered at its
close as now. The last clock tick is therefore 2026-08-01T00:00:00Z; this is the
close of July's last bar, not an August candle. History is seeded with the same
200 pre-window bars as the live shadow CLI default.

The existing signal, risk and execution contracts are reused. A decision from T
fills at the open of the next contiguous bar and is recorded at that bar's close.
Fees are 10 bps and adverse slippage 5 bps. There is no terminal liquidation;
final equity uses the last observed open, as in the live engine. One final target
remains pending per product.

The five-hour May gap is preserved. Per product the engine records one gap, one
expired target and 25 subsequent warm-up bars without a prediction. It produces
2178 predictions, of which 2170 have a fully realized label inside the replay
window. Four predictions preceding the gap and four at the July tail are unscored.
Realized labels are computed only after the engine has stopped and never enter
a decision. Time-series Spearman rank IC, MAE and RMSE use only these 2170 observations.

| Product | MAE | RMSE | LONG / FLAT / SHORT signals and targets | Fees (USD) | Slippage (USD) |
| --- | ---: | ---: | --- | ---: | ---: |
| BTC-USD | 0.005653686480 | 0.008168726013 | 0 / 2178 / 0 | 0 | 0 |
| ETH-USD | 0.007425723031 | 0.010946672310 | 0 / 2177 / 1 | 3.528669628851 | 1.764326279280 |

Equity metrics use every observed mark, including initial capital in drawdown
and Sharpe calculations. Sharpe uses sample hourly-return variance and 8760
periods/year; a gap contributes one observed interval, not invented hourly
marks. The committed curves retain bucket endpoints/extrema and the actual
worst-drawdown peak and trough: 246 BTC points and 247 ETH points from 2203 marks.

## Determinism, evidence and read-only views

Two complete deliveries of the same window, with no second training, yield
identical canonical product results and event chains. Both chains contain
26244 events and end at
b6c1d770721f38cc50ce18e36b28ce0218a99961fe8347eb222209d5fc78340f.
The [result manifest](../data/crypto/paper_replay_v2/manifest.json) binds both
runs' result hashes, canonical file hashes, model/spec/corpus hashes and limitations.

Runtime stores are separate files under the ignored var/trading_lab/replay/
directory. The replay refuses existing files and the live database names. No
store, capture body or runtime log is committed.

- GET /api/v1/paper/replay returns summaries only.
- GET /api/v1/paper/replay/{product}/equity pages the stored downsampled curve.
- GET /api/v1/paper/replay/{product}/fills pages fills.
- Both pages default to 100 and refuse limits outside 1..1000. Cursors bind to
  product, endpoint and result hash; replacing a result invalidates a cursor.
  Artifact reads are limited to 4 MiB. These views never open a live database
  and have no training, replay or control endpoint.

Initial generation used python -m scripts.trading_lab.paper_replay train and
python -m scripts.trading_lab.paper_replay replay. They now refuse overwriting
the frozen outputs. Read-only tests reload the models, reproduce scored predictions
and validate the evidence without refitting the real models.

This is **one out-of-sample window, not confirmatory, not optimized**. The
historical window was already spent in earlier research. Candles were captured
later without point-in-time exchange revision history; bar-close delivery assumes
availability and does not reconstruct actual publication delays. The costs and
accounts are synthetic. These results establish neither profitability nor a
confirmatory finding. The next step is branch review and integration, without
changing the model or spending the protected holdout.

## Verification status

The 36 new Python v2/replay/API cases passed, including reconstruction of every
scored prediction from the frozen models without refitting. The existing paper,
portfolio, dataset, walk-forward, signal, risk and execution tests also passed;
an eight-point downsampling edge case was corrected and verified in the final run.
Independent recalculation from all 2203 marks in each private replay store agrees
with the recorded return, drawdown, Sharpe and costs for both products and both runs.

Web installation from the offline npm cache, typecheck, lint, 132 Vitest cases
(including the four replay-section cases) and the production build passed.

**Full repository validation is not done.** The required Python command initially
reported 495 passed, 7 failed and 2 skipped. Five TLS failures were caused by the
long temporary socket path; a short worktree-local path fixed them. Rechecking TLS
and the grant journal gave 12 passed and one remaining failure:
test_a_stop_while_a_continuation_waits_consumes_no_grant_and_stays_proven returns no
durable outcome. FOMC/source code and its tests are unchanged from the named base;
no rule, spec or frozen source code was modified to pass this check. The other
grant-order failure passed on retry and remains a timing concern.

BLOCKED: the official FOMC fixture proof lacks its private local fixtures, and the
legacy FastAPI summary test lacks the hyprl_api package on this branch. Neither is
reported as passed. Resolve the FOMC validation failure and review these unavailable
checks before integration; no integration or service deployment was performed.
