# BTC Asia low on Fridays — exploratory claim check

**No Friday advantage established.** The observed frequency is 251/450 = 55.8% (95% Wilson CI 51.2–60.3%), versus 58.0% on the six other weekdays. The paired weekly-block difference interval is −7.1 to +2.9 percentage points. The two-sided test against exactly 60% gives p=0.074918: it does not reject 60% at 5%, and does not confirm that claim.

The fixed rule is the strict later-same-day breach of the 02:00–11:00 Europe/Paris low, after 11:00 and before local midnight. Touches are excluded. DST follows the original plan. This measures event frequency, not returns, execution, costs, trading profitability or a causal weekday effect. The protected September–November 2026 BTC/ETH interval was excluded; no training or order occurred.

## Data and uncertainty

Coinbase Exchange BTC-USD hourly candles, UTC 2018-01-01 through 2026-09-01 exclusive: 264 authorized GETs, no retry, at most one request per second; 75,919 observed candles. Complete measurement windows: 3,151 days; 14 incomplete days excluded, with no filling. Year 2026 ends on August 31. Friday exceeds the other-day point estimate in 4/9 years; those subgroup comparisons are descriptive.

The empirical all-day rate is 57.7%; uniform year-stratified random samples use the Friday sample size, with a simulated 95% range of 53.6–62.0%. It is not a 50% coin baseline. The primary contrast resamples 450 paired ISO-week blocks 10,000 times with the fixed seed. Longer dependence may remain. Wilson intervals and the separate binomial test assume independent observations. This historical exploratory check is not untouched out-of-sample evidence.

| Year | Friday k/n | Friday % [95% CI] | Other days % [95% CI] | All-day random-pool % [95% CI] | Excluded days |
| --- | ---: | --- | --- | --- | ---: |
| 2018 | 26/51 | 51.0 [37.7, 64.1] | 52.6 [47.0, 58.1] | 52.4 [47.2, 57.5] | 6 |
| 2019 | 18/52 | 34.6 [23.2, 48.2] | 58.7 [53.2, 64.1] | 55.2 [50.1, 60.3] | 3 |
| 2020 | 30/52 | 57.7 [44.2, 70.1] | 51.6 [46.1, 57.1] | 52.5 [47.3, 57.6] | 2 |
| 2021 | 25/53 | 47.2 [34.4, 60.3] | 57.4 [51.8, 62.7] | 55.9 [50.8, 60.9] | 0 |
| 2022 | 37/52 | 71.2 [57.7, 81.7] | 63.3 [57.8, 68.4] | 64.4 [59.3, 69.1] | 0 |
| 2023 | 33/52 | 63.5 [49.9, 75.2] | 59.6 [54.1, 64.9] | 60.2 [55.1, 65.1] | 1 |
| 2024 | 27/52 | 51.9 [38.7, 64.9] | 60.2 [54.7, 65.5] | 59.0 [53.9, 63.9] | 0 |
| 2025 | 36/52 | 69.2 [55.7, 80.1] | 60.9 [55.4, 66.1] | 62.1 [57.0, 66.9] | 1 |
| 2026* | 19/34 | 55.9 [39.5, 71.1] | 57.7 [50.9, 64.2] | 57.4 [51.1, 63.5] | 1 |

*2026 is partial. The aggregate JSON also contains every named weekday and its sample/95% interval.

## Reproduction and identities

Collection requires the signed exact-scope operator grant; its durable shared budget includes retries and cannot be reset by choosing another output directory. Response shards stay private and retain the native six-field Coinbase shape. Every request ends before the forbidden interval. Hash/shape/coverage checks run before the study. Independent local-hour grouping reproduced all overall, yearly and weekday counts and the Friday Wilson interval.

- Plan SHA256: `2f2e51605173ad8c6a7502505fc2dfb43a09e4be0ca4571622a54621095e3209`.
- Private input manifest SHA256: `fb3f8bc813f626cfbb4c7fbf89eff72647ee60f49aa3beb4ac6419a019fa9876`.
- Analysis implementation SHA256: `10aadf5681e1e22283ceefbe8c745334a178c452e7482f31e6b4e73340e82b4c`.
- Signed authorization byte SHA256: `f078fccf9d458c3acb33f900bc9fab9cea35fc14f11eb8c2c68e679533e60b34`.
- [Aggregate result](artifacts/btc_asia_study_result_v1.json). No private source body or individual candle is published.
- [An actual in-scope Coinbase request](https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600&start=2018-01-01T00%3A00%3A00Z&end=2018-01-12T23%3A59%3A59Z).

The existing offline API is `load_candles`, `daily_events`, then `summarize` in `scripts/video_factory/study.py`. The study plan is unchanged. The only analysis correction converts the SciPy p-value to a native Python float so its derived boolean and completed result can be serialized to JSON. A regression check serializes and reads the complete result. No data-dependent rule or threshold was tuned.

The seven-slide carousel reports the observed sample/uncertainty, compares yearly empirical rates, and separates compatibility with 60%, a weekday advantage, and profitability. No video is produced.
