# US Equity Corpus V2 — locally frozen research corpus

```
STATUS: LOCALLY FROZEN RESEARCH CORPUS

SOURCE DATA NOT REDISTRIBUTED

UNOFFICIAL PROVIDER

RAW PRICE SEMANTICS

NO PREDICTIONS

NO BACKTEST RESULTS

NO COMMERCIAL EDGE CLAIM

NOT STRICT POINT-IN-TIME REVISION HISTORY
```

---

## 1. What this is, and what it is not

The first real US equity dataset in HyprL. It exists because Corpus V1 —
30-minute, split-adjusted, from Massive — is blocked on a credential that does
not exist here, and staying blocked was worse than having a different corpus
with honestly different semantics.

It is **daily**, not 30-minute. That is a fact about the source rather than a
preference: the endpoint answers HTTP 422 for a 30-minute range spanning two
years, serving intraday only for roughly the last month. A 30-minute two-year
corpus still requires a paid provider, and V1 remains its frozen spec.

It is **RAW**, not adjusted. The payload carries unadjusted OHLC in `quote` and
a separate adjusted close in `adjclose`; this corpus reads the raw quote and
records corporate actions beside it. It never claims an adjustment it did not
perform, and `adjclose` is never substituted for a close.

## 2. Identity

| | |
| --- | --- |
| corpus id | `yahoo_us_equity_daily_v1` |
| provider | `yahoo-chart-daily-v1` |
| timeframe | `1d` |
| session | `REGULAR` (US regular only) |
| adjustment | `RAW` |
| calendar | `US_EQUITY_REGULAR` |
| calendar dependency | `pandas_market_calendars==5.4.0` |
| calendar spec hash | `1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314` |
| requested range | `2024-08-01` → `2026-07-31` |
| **corpus spec hash** | `b7ad1e33b9896418e81f5e386ddc5384e0e2caca4854d467ac894eec25987dae` |
| **corpus content hash** | `64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024` |
| capture code commit | `f4d64062ca5cb4e622c2b8351ecc558a7108721f` |

The spec hash covers what was *asked for*; the content hash covers what came
back. Re-running the same request against a source that has changed its mind
gives the same spec hash and a different content hash — which is the entire
signal an audit is looking for, and it exists only because the two are
computed from different things.

## 3. What was captured

| instrument | expected | rows | missing | dup | off-grid | null | splits | content hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `xnas:AAPL` | 501 | 501 | 0 | 0 | 0 | 0 | 0 | `37ecfe78825ca78f…` |
| `xnas:MSFT` | 501 | 501 | 0 | 0 | 0 | 0 | 0 | `9a3b6d6fae0aef11…` |
| `xnas:NVDA` | 501 | 501 | 0 | 0 | 0 | 0 | 0 | `8a96035fba9e8511…` |
| `xnas:QQQ` | 501 | 501 | 0 | 0 | 0 | 0 | 0 | `79d5032705b7c91c…` |

**Total canonical rows: 2004** across 4 instruments,
501 expected sessions each.

Expected sessions come from the frozen calendar over the requested range —
never from a count of weekdays, and never hardcoded. An early close still
produces exactly one daily bar, ending when the market actually closed.

Zero splits in the window. NVDA's 10-for-1 was June 2024, before the range
opens; the others had none. That is a fact about this window, not a gap: the
corporate-action files exist and are empty, and their hash is recorded.

## 4. Storage — the source data is not in this repository

```
var/trading_lab/research/yahoo_us_equity_daily_v2/
  spec.json
  manifest.local.json
  raw/                  # the source responses, byte-for-byte
  canonical/            # derived daily bars, JSONL
  corporate_actions/    # split provenance
```

That path is gitignored **and** in the release builder's excluded directories
— both were already true before this corpus existed, so it is excluded by two
mechanisms neither of which was added for it.

What *is* committed is
`docs/artifacts/us_equity_corpus_v2_fingerprint.json`: hashes, counts, dates
and identities. It proves **which** corpus was frozen and is useless for
reconstructing it — roughly 3.7 KB standing in for 2004 bars.

The source declares `official_contract=false` and
`redistribution_permitted=false`. A release that tried to include a dataset
carrying that flag now fails to build, rather than relying on anyone
remembering to keep it out of a tuple.

## 5. Verification

Three passes, all with no network:

1. **verify** — rebuild every row from the stored raw responses and compare
   against the manifest: identity, timezone, session membership, OHLC,
   duplicates, gaps, corporate actions, every hash.
2. **rebuild** — delete the canonical files, regenerate them from raw, assert
   **byte** equality.
3. **recount** — re-derive session counts from the calendar, row counts by
   reading the files line by line, and digests from those bytes. It
   deliberately calls neither of the other two: running one function twice
   proves determinism, not correctness, because a shared bug agrees with
   itself perfectly.

```bash
python -m scripts.trading_lab.verify_yahoo_equity_corpus verify
python -m scripts.trading_lab.verify_yahoo_equity_corpus rebuild
python -m scripts.trading_lab.verify_yahoo_equity_corpus recount
```

Sockets were disabled for all three during the freeze; any network attempt
would have raised.

## 6. Point-in-time limitation

This is a **frozen snapshot of what the source returned on the capture date**.
It is not a strict point-in-time exchange revision history.

RAW does not mean PIT. RAW means unadjusted — no split restatement applied by
us or by the source. It says nothing about whether a bar was later revised, and
a revision made before the capture is already baked in invisibly. Corporate
events known today may not have been known on the dates in the series.

Causal backtesting still holds — a feature at T only ever reads sessions ≤ T —
but no future benchmark may describe a result on this corpus as **strict PIT**
without an additional layer that reconstructs revisions.

## 7. Full hashes

```
xnas:AAPL
  canonical content  37ecfe78825ca78f53f9f3c0008c0b2452263ab4f12a1ceff14acf0ad2f5cb7b
  canonical file     9d53540f58e4e774284ea440797ecbc56a1e807e85ada10cb54d85febb824cf0
  raw response       1d3bde0b7eefd3a038958286d354d3d22d2eb39d49004f4740c2ed284a64a262  (56071 bytes)
  corporate actions  4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
xnas:MSFT
  canonical content  9a3b6d6fae0aef114bd103b5ba63e2eace15296d1dcfbef1c9d310a168520364
  canonical file     280795d3e105655c4340746272a36fede6d8482bc8cc685e3e10609549acef5b
  raw response       c8c4dded9114d666e7aac9aaf6607591532354e7d7d59496d1cd970ff219c453  (54928 bytes)
  corporate actions  4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
xnas:NVDA
  canonical content  8a96035fba9e85110e6649c7d4c463ab5d001c056bb6ebc7d452132bf9a79221
  canonical file     b2f8aae8cfd8504634c1a4651063a45678b763f73779fbb81d4a7a87e314db76
  raw response       cc2816d121833b51e0e210b6a6fae058317eaf29fcdfe5513db66301271ad831  (56813 bytes)
  corporate actions  4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
xnas:QQQ
  canonical content  79d5032705b7c91c476630cdf2ca8c801e2f6a350487b5837edd5902e1f677ba
  canonical file     caa6a636e54e2f44eaca075f0cd6c8998472450b209a1f6dfef90ebc8a7e16bf
  raw response       133eadab948d79c29e4e829b5ec50b957308fa3a98867540b337064834fd2ecb  (55082 bytes)
  corporate actions  4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945

aggregate corpus     64ac4485fc2541e671b899928f804bf3a7ceed5d380cb266145904446f71e024
manifest content     b55b535e9b135d824b55d2518576e7cfa6068bf8b520525ffd67e9f4b82e9f61
```

## 8. What this phase did not do

No prediction, no feature, no model, no benchmark, no backtest, no paper
trading, no P&L. The equities remain catalogue-only and are not in the
tradable registry; the shared paper portfolio is still BTC and ETH alone.
Massive V1 is untouched and still uncaptured. The protected holdout is neither
observed nor spent.

`commercial_edge_established = false`.
