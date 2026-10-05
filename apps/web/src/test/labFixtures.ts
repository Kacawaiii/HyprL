/** Real responses captured from the Model Lab / research / observability API run on the SYNTHETIC demo
 *  (`python -m scripts.trading_lab.research.demo`), trimmed for size. Nothing here is a market result. */
/* eslint-disable */
export const lab: any = {
 "comparison": {
  "actions": [
   "operator reviews frozen COMPARISON_PROTOCOL_V2",
   "operator supplies scoped capture and training authorizations for the fixed windows and budgets",
   "obtain causally attested events and matching prices; backfills retain their actual acquisition availability",
   "build identical admissible A/B populations with exclusions in both arms"
  ],
  "criteria_hash": "245c8ccfa6d851b98d296c1183226a8274ed388f00af5ed54f1482653a16d99c",
  "crypto_role": "PRIMARY_CONFIRMATORY",
  "equity_role": "EXPLORATORY_NO_CLAIM",
  "execution_enabled": false,
  "hypothesis": {
   "availability": {
    "pairing": "A and B use exactly the same decisions, labels, costs and splits; a decision where a required source is not RESOLVED is excluded from both; a family's test set is the decisions common to all its products",
    "protection": {
     "admissible_ranges_price_and_label": {
      "crypto": {
       "decision_clock": "bar close = bar opening + 1h",
       "event_window_clean_from": "2026-12-31T00:00:00Z",
       "label_horizon_bars": 4,
       "price_warmup_bars": 25,
       "runs": [
        {
         "decisions": 9475,
         "first_bar_open": "2025-08-02T01:00:00Z",
         "first_decision_at": "2025-08-02T02:00:00Z",
         "last_bar_open": "2026-08-31T19:00:00Z",
         "last_decision_at": "2026-08-31T20:00:00Z",
         "run_bars": 9504
        },
        {
         "decisions": 5803,
         "first_bar_open": "2026-12-02T01:00:00Z",
         "first_decision_at": "2026-12-02T02:00:00Z",
         "last_bar_open": "2027-07-31T19:00:00Z",
         "last_decision_at": "2027-07-31T20:00:00Z",
         "run_bars": 5832
        }
       ]
      },
      "equity": {
       "decision_clock": "session close (calendar close_at, early closes honoured)",
       "event_window_clean_from": "2027-03-31T00:00:00Z",
       "label_horizon_sessions": 5,
       "price_warmup_sessions": 20,
       "runs": [
        {
         "decisions": 560,
         "first_bar_open": "2024-08-29T13:30:00Z",
         "first_decision_at": "2024-08-29T20:00:00Z",
         "last_bar_open": "2026-11-20T14:30:00Z",
         "last_decision_at": "2026-11-20T21:00:00Z",
         "run_bars": 585
        },
        {
         "decisions": 81,
         "first_bar_open": "2027-03-30T13:30:00Z",
         "first_decision_at": "2027-03-30T20:00:00Z",
         "last_bar_open": "2027-07-23T13:30:00Z",
         "last_decision_at": "2027-07-23T20:00:00Z",
         "run_bars": 106
        }
       ]
      }
     },
     "event_window_days": 30,
     "label_horizon": {
      "crypto_bars": 4,
      "equity_sessions": 5
     },
     "other_reserved_intervals": "none found: the walk-forward and paper-replay periods are exploratory/consumed, not reserved; closed pilots are archives, not intervals",
     "price_warmup": {
      "crypto_bars": 25,
      "equity_sessions": 20
     },
     "rule": "a decision is admissible only if no bar of its price features, no bar of its label window and no instant of its 30-day event window (T-30d, T] lies in a protected interval of its product",
     "table": {
      "AAPL": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "BTC-USD": [
       {
        "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
        "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
        "end_exclusive": "2026-12-01T00:00:00Z",
        "holdout_id": "coinbase_confirmatory_2026q4",
        "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
        "last_protected_bar_open": "2026-11-30T23:00:00Z",
        "observed": false,
        "products": [
         "BTC-USD",
         "ETH-USD"
        ],
        "single_use": true,
        "start": "2026-09-01T00:00:00Z",
        "timeframe": "1h",
        "unit": "bar opening instants"
       }
      ],
      "ETH-USD": [
       {
        "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
        "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
        "end_exclusive": "2026-12-01T00:00:00Z",
        "holdout_id": "coinbase_confirmatory_2026q4",
        "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
        "last_protected_bar_open": "2026-11-30T23:00:00Z",
        "observed": false,
        "products": [
         "BTC-USD",
         "ETH-USD"
        ],
        "single_use": true,
        "start": "2026-09-01T00:00:00Z",
        "timeframe": "1h",
        "unit": "bar opening instants"
       }
      ],
      "MSFT": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "NVDA": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "QQQ": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ]
     }
    },
    "warmup": {
     "capture_start_plus_30_days": "2027-04-01T00:00:00Z",
     "event": "for every required source: state RESOLVED at T, T >= attested coverage start + 30 days and T >= max(first valid read availability, last inventory availability at T) + 30 days, per issuer for EDGAR",
     "price": "the feature-complete index of the real indicators (see protection.price_warmup), inside the series"
    }
   },
   "baselines": [
    "PRICES_ONLY",
    "ZERO",
    "TRAIN_MEAN"
   ],
   "budgets": {
    "execution_enabled": false,
    "max_rows": 10000,
    "max_trials": 1,
    "protocol_budgets": {
     "capture_window": {
      "days": 153,
      "end_exclusive": "2027-08-02T00:00:00Z",
      "rule": "FOMC and EDGAR run for the whole window; stop when the last supplied decision is RESOLVED",
      "start": "2027-03-02T00:00:00Z"
     },
     "edgar": {
      "global_pacing": "10s spacing, at most 6 per 60s",
      "issuers": [
       "AAPL",
       "MSFT",
       "NVDA"
      ],
      "listings": 66096,
      "listings_per_issuer_per_day": 144,
      "nvda_identity_lookup": 1,
      "poll_interval_s": 600,
      "total_request_ceiling": 66097
     },
     "fomc": {
      "backoff_steps": 5,
      "feed_cadence_s": 60,
      "feed_polls": 220320,
      "max_statements": 16,
      "recheck_offsets_s": [
       300,
       3600,
       86400,
       604800
      ],
      "statement_pages_nominal": 80,
      "statement_pages_with_maximal_retries": 480,
      "total_request_ceiling": 220800
     },
     "prices": {
      "coinbase": {
       "bars_per_product": 2976,
       "candles_per_request": 300,
       "products": 2,
       "requests": 20,
       "series": [
        "2027-03-30T00:00:00Z",
        "2027-08-01T00:00:00Z"
       ]
      },
      "equity_daily": {
       "instruments": 4,
       "requests": 4,
       "requests_per_instrument": 1,
       "series_dates": [
        "2027-03-01",
        "2027-07-31"
       ]
      }
     },
     "raw_volume_bytes_worst_case_without_deduplication": {
      "edgar": 12244284000,
      "fomc_feed": 18637309440,
      "observed_deduplication": {
       "edgar_trial": {
        "distinct_raws": 2,
        "responses": 8
       },
       "fomc_pilot": {
        "distinct_raws": 49,
        "responses": 128
       }
      },
      "rule": "the operator sets a store-size ceiling and checks free disk before authorizing; the worst case exceeds the disk of the shared server, so the authorization must carry the ceiling and a stop",
      "sizes_used": {
       "edgar_listing_bytes_max_observed": 185250,
       "fomc_raw_bytes_max_observed": 84592
      }
     }
    },
    "wall_seconds": 180
   },
   "calendar": {
    "calendar_spec_hash": "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314",
    "capture": [
     "2027-03-02T00:00:00Z",
     "2027-08-02T00:00:00Z"
    ],
    "decision_counts_upper_bounds": {
     "crypto_per_product": {
      "expected_to_meet_minimum": true,
      "first_decision": "2027-04-01T00:00:00Z",
      "last_decision": "2027-07-31T20:00:00Z",
      "minimum_paired_test_decisions": 240,
      "purged": 3,
      "test": 1461,
      "train": 1461
     },
     "equity_per_instrument": {
      "expected_to_meet_minimum": false,
      "first_decision": "2027-04-01T20:00:00Z",
      "last_decision": "2027-07-23T20:00:00Z",
      "minimum_paired_test_decisions": 100,
      "purged": 5,
      "test": 37,
      "train": 37
     },
     "note": "upper bounds: before price gaps and source-state exclusions, which are known only after capture"
    },
    "evaluation": [
     "2027-04-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "possible": {
     "crypto": "hourly decisions, T = bar close, from the first feature-complete bar",
     "equity": "one decision per session at the calendar close (early closes honoured)"
    },
    "price_series": {
     "crypto_bar_opens": [
      "2027-03-30T00:00:00Z",
      "2027-08-01T00:00:00Z"
     ],
     "equity_sessions": [
      "2027-03-01",
      "2027-07-31"
     ],
     "rule": "series start after every protected interval, so no feature recursion reads one"
    },
    "split": {
     "purge": "train decisions whose label window ends after the split instant are dropped",
     "test": [
      "2027-06-01T00:00:00Z",
      "2027-08-01T00:00:00Z"
     ],
     "train": [
      "2027-04-01T00:00:00Z",
      "2027-06-01T00:00:00Z"
     ]
    }
   },
   "costs": {
    "cost_model": {
     "fee_rate": "0.0010",
     "fill_policy": "next-contiguous-bar-open-after-decision-v1",
     "slippage_rate": "0.0005"
    },
    "equity_costs": "no frozen equity execution or cost model exists: no equity economic metric is computed",
    "execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
    "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
    "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
   },
   "decision_criteria": {
    "exclusions_reported": [
     "protected interval or label/lookback/event window",
     "price warm-up",
     "event warm-up",
     "source state not RESOLVED, per source and state",
     "price gap or missing label",
     "unknown or unverified issuer mapping",
     "integrity errors",
     "null imputations (count)",
     "decisions kept, per product, split and variant"
    ],
    "exploratory": {
     "computed": "the same paired metric R, its bootstrap interval (blocks of 10 sessions) and the secondary prediction metrics, on the same decisions, splits and exclusions, reported as descriptive statistics",
     "effect_on_primary": "none: the equity arm cannot change, delay or condition the crypto verdict",
     "family": "EQUITY",
     "products": [
      "AAPL",
      "MSFT",
      "QQQ",
      "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
     ],
     "sample_reported": {
      "edgar_newly_observed_accessions_in_test": "counted and reported",
      "paired_test_decisions_per_instrument": "counted and reported (the V1 count expected 37, below the 100 a verdict would need)"
     },
     "status": "EXPLORATORY_NO_CLAIM",
     "verdict": "none: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE is never assigned to this arm, and no result of it is cited as evidence for or against events",
     "windows": "the equity sessions looked at here become exploratory; a later confirmatory equity claim needs its own preregistered revision on data not looked at, and the reserved equity holdout stays closed"
    },
    "primary": {
     "decision_rule": {
      "INCONCLUSIVE_INSUFFICIENT_SAMPLE": "any minimum sample not met; no verdict is drawn",
      "NOT_SUPPORTED": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
      "SUPPORTED": "R > 0, lower bound of the interval > 0, and R >= 0.005",
      "scope": "the CRYPTO family only; no pooling with any other product; no other metric can overturn it"
     },
     "families": {
      "CRYPTO": [
       "BTC-USD",
       "ETH-USD"
      ]
     },
     "label": {
      "crypto": "close[T+4 bars]/close[T]-1"
     },
     "metric": "R = 1 - MSE_B / MSE_A on test decisions, per product, equal-weighted mean over the family's products",
     "minimum_sample": {
      "blocks_per_product": 10,
      "crypto_fomc_statements_newly_observed_in_test": 2,
      "paired_test_decisions_per_product": {
       "crypto": 240
      }
     },
     "uncertainty": {
      "block_decisions": {
       "crypto": 24
      },
      "families_tested": 1,
      "family_alpha_one_sided": 0.025,
      "familywise_note": "one confirmatory family (CRYPTO) at one-sided 2.5%, the level V1 fixed for it, kept unchanged and not relaxed; the exploratory equity arm spends no alpha",
      "interval": "percentile 2.5%-97.5% (two-sided 95%): its lower bound is the one-sided 2.5% bound",
      "method": "circular block bootstrap of time-ordered paired decisions, same block starts for every product of a family and for both variants",
      "resamples": 10000,
      "seed": 20270801
     }
    },
    "secondary": {
     "annualization_periods": 8760,
     "baselines": [
      "ZERO",
      "TRAIN_MEAN"
     ],
     "economic_crypto_only": [
      "net_return",
      "sharpe",
      "max_drawdown",
      "turnover",
      "fees_paid"
     ],
     "prediction": [
      "mae",
      "rmse",
      "spearman_rank_ic",
      "directional_accuracy"
     ],
     "scope": "CRYPTO and the exploratory EQUITY arm; never part of any decision rule",
     "status": "reported with the same bootstrap; never part of the decision rule"
    },
    "stopping": "fixed periods; no optional stopping, no extension or re-run after a look"
   },
   "falsification": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
   "features": {
    "event_feature_spec_hash": "b54ae52b815dcc9a23bd728b2899f7e0e7c43cb8482f68b189ac5236e84f43fb",
    "events": {
     "AAPL": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "BTC-USD": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ],
     "ETH-USD": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ],
     "MSFT": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "NVDA": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "QQQ": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ]
    },
    "prices": {
     "crypto": [
      "return_1",
      "return_4",
      "return_12",
      "ema_spread_12_26",
      "rsi_14",
      "atr_pct_14"
     ],
     "equity": [
      "return1",
      "return5",
      "return20",
      "ema_spread10_20",
      "rsi14",
      "atr_pct14"
     ]
    },
    "transform": {
     "boolean": "0/1",
     "hours_since_last_new_*": "min(hours, 720); null (no new observation yet) -> 720",
     "null_item_flag": "0 (counted and reported)",
     "selection": "none: every listed event column enters variant B, none is dropped or added after results"
    }
   },
   "horizon_seconds": 14400,
   "hypothesis_id": "prices-vs-events-protocol-v2",
   "mechanism": "Newly observed FOMC information and revisions can change expected forward returns",
   "population": {
    "admissibility": [
     "inside the evaluation period and the product's price series",
     "no protected interval touched by lookback, label or event window",
     "price features and label exist (no gap)",
     "every required source RESOLVED, no INTEGRITY_ERROR",
     "price and event warm-ups complete"
    ],
    "new_evaluation": [
     "2027-04-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "products": {
     "crypto": [
      "BTC-USD",
      "ETH-USD"
     ],
     "equity": [
      "AAPL",
      "MSFT",
      "QQQ",
      "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
     ],
     "required_sources": {
      "AAPL": [
       "fomc",
       "edgar"
      ],
      "BTC-USD": [
       "fomc"
      ],
      "ETH-USD": [
       "fomc"
      ],
      "MSFT": [
       "fomc",
       "edgar"
      ],
      "NVDA": [
       "fomc",
       "edgar"
      ],
      "QQQ": [
       "fomc"
      ]
     },
     "roles": {
      "crypto": "PRIMARY_CONFIRMATORY",
      "equity": "EXPLORATORY_NO_CLAIM"
     }
    },
    "studied_windows": "EXPLORATORY"
   },
   "protocol_hash": "3817ff509bdc35ec53aca85bc3c558e60b3db828efa920f12191475e9883875e",
   "schema": "research-hypothesis-v1",
   "scope": "PREREGISTERED_PROTOCOL",
   "sources": [
    "fomc",
    "edgar",
    "coinbase"
   ],
   "splits": {
    "purge": "train decisions whose label window ends after the split instant are dropped",
    "test": [
     "2027-06-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "train": [
     "2027-04-01T00:00:00Z",
     "2027-06-01T00:00:00Z"
    ]
   },
   "statement": "Attested events improve crypto test MSE beyond prices on the same admissible decisions",
   "synthetic": false,
   "target": "forward_return",
   "transformations": {
    "embargo": "only the purge fixed by protocol V2; no additional embargo",
    "fit_on": "TRAIN_ONLY",
    "method": "refit on the forward train block for both variants; the split is new because the frozen walk-forward geometry (252 train sessions) does not fit a 4-month window; training needs its own authorization"
   },
   "version": "local-rule-proposals-v1"
  },
  "hypothesis_hash": "5787a940a7799947b846fca60c7818dba3cbb4cf61f45eb5d201d34048f645b2",
  "limitations": [
   "readiness is the registered campaign status, not a scan of arbitrary stores",
   "this module cannot capture, train the real variants or unlock holdouts"
  ],
  "paired_decisions": 0,
  "protocol_hash": "3817ff509bdc35ec53aca85bc3c558e60b3db828efa920f12191475e9883875e",
  "reasons": [
   "NO_ATTESTED_EVENT_PRICE_OVERLAP",
   "PROTOCOL_NOT_OPERATOR_ACCEPTED",
   "CAPTURE_AND_TRAINING_NOT_AUTHORIZED"
  ],
  "schema": "comparison-readiness-v1",
  "state": "WAITING_DATA",
  "studied_windows": "EXPLORATORY"
 },
 "datasetResult": {
  "result": {
   "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
   "manifest": {
    "counts": {
     "by_product": {
      "BTC-USD": 91,
      "ETH-USD": 91
     },
     "excluded": 58,
     "included": 182
    },
    "dataset_id": "synthetic-ffb19a95a5f2974366926255",
    "decision_end": "2026-06-06T01:00:00+00:00",
    "decision_start": "2026-06-01T00:00:00+00:00",
    "exclusions": [
     {
      "decision_at": "2026-06-01T01:00:00+00:00",
      "product": "BTC-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "7052310cf944f479309d9c3aa1ac3e00c2a9313132e51a94203a7fc10d2173e8"
     },
     {
      "decision_at": "2026-06-01T01:00:00+00:00",
      "product": "ETH-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "073c852504ef882344700a3c3bd6b736f4011a0089cbe3362140d41e112553f4"
     },
     {
      "decision_at": "2026-06-01T02:00:00+00:00",
      "product": "BTC-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "3e808b418790e982e791786684c795c9ef7d967779eabb04e00f1b02a04c8592"
     },
     {
      "decision_at": "2026-06-01T02:00:00+00:00",
      "product": "ETH-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "409f853515840541ebed2f5b318ce74a8b53d0a0f2723ae205be5ac90014e8ff"
     },
     {
      "decision_at": "2026-06-01T03:00:00+00:00",
      "product": "BTC-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "7074afb5e250f570283c54929d68468072d99a644070c80e0c3b2c8360bb8b15"
     },
     {
      "decision_at": "2026-06-01T03:00:00+00:00",
      "product": "ETH-USD",
      "reason": "PRICE_FEATURE_WARMUP_OR_GAP",
      "snapshot_hash": "3ad855db9b9b6f72a5f618c88c170a2f286ab614e38bad03dd8e0febf0bccced"
     }
    ],
    "features_hash": "c1f7d1b0228b7f752738744b67ce8c0c5eb62adc599f5e71227f0ecb040b00a5",
    "horizon_seconds": 14400,
    "policies": {
     "availability": "all price dependencies attested by decision; labels separate",
     "calendar": "coinbase-hourly-utc-v1",
     "columns": [
      "return_1",
      "return_4",
      "return_12",
      "ema_spread_12_26",
      "rsi_14",
      "atr_pct_14"
     ],
     "dataset": "MODEL_LAB_DATASET_V1",
     "event_columns": [],
     "price_features": [
      {
       "column": "return_1",
       "indicator": "simple_return",
       "parameters": {}
      },
      {
       "column": "return_4",
       "indicator": "return_over_period",
       "parameters": {
        "period": 4
       }
      },
      {
       "column": "return_12",
       "indicator": "return_over_period",
       "parameters": {
        "period": 12
       }
      },
      {
       "column": "ema_spread_12_26",
       "indicator": "ema_spread",
       "parameters": {
        "fast_period": 12,
        "slow_period": 26
       }
      },
      {
       "column": "rsi_14",
       "indicator": "rsi",
       "parameters": {
        "period": 14
       }
      },
      {
       "column": "atr_pct_14",
       "indicator": "atr_percent",
       "parameters": {
        "period": 14
       }
      }
     ],
     "protection_hash": "97b2cea6ab819d8cb75b70df9cc4bddd78b4777f1b5142a23825d72f2fb10a5d",
     "snapshot_policies": [
      "db5251193525524027e718e944b0f87f18b53cdf700e4ad9b975952fe5f87be3"
     ],
     "transformations": "fit on training only",
     "warmup_bars": 25
    },
    "products": [
     "BTC-USD",
     "ETH-USD"
    ],
    "schema": "dataset-manifest-v1",
    "snapshot_hashes": [
     "000c4187b8a2816c25668de92760553817bcc3f72366b6ce92600706b9663cb0",
     "00ee1abdbb7fff923d67164825e8f21b41d87461e95b9af847bd615a3f9e4a67",
     "0130fa34643faaa3e252742ae30fc1d2b5cb0b52937c580cc924d04a9d225b18"
    ],
    "splits": {
     "method": "temporal-purge-embargo-v1",
     "state": "UNASSIGNED"
    },
    "synthetic": true,
    "target": "forward_return",
    "version": "model-lab-dataset-v1"
   },
   "synthetic": true
  },
  "result_hash": "b6fdf8023041a4709017cce41e12d61de51f8ea34993532faa1c83a4f3babc3c",
  "state": "COMPLETE"
 },
 "experimentResult": {
  "result": {
   "backtests": {
    "BTC-USD": {
     "confirmatory": false,
     "cost_model": "synthetic",
     "economic_backtest_spec_hash": "58712fcedeb291869cbdcb978ebe36245ea1dffde192e52ef7782294c6075cf0",
     "equity_curve": [
      {
       "cash": "74962.48750000000000000000000000000",
       "cumulative_fees": "25.01250000000000000000000000000000",
       "cumulative_slippage_cost": "12.50000000000000000000000000000000",
       "equity": "99962.48750000000000000000000000000",
       "mark_price": "102.475",
       "position_quantity": "243.9619419370578189802390827030983",
       "position_value": "25000.00000000000000000000000000000",
       "realized_exposure": "0.2500938164428931402892510052833569",
       "target_exposure": "0.25",
       "timestamp": "2026-06-04T23:00:00+00:00"
      },
      {
       "cash": "74642.95311768041709868260551353989",
       "cumulative_fees": "25.33171516715243047084654793852159",
       "cumulative_slippage_cost": "12.65952781966638204440107343254452",
       "equity": "99524.09707123616240241522322517687",
       "mark_price": "100.68",
       "position_quantity": "247.1309490818012048443843634449442",
       "position_value": "24881.14395355574530373261771163698",
       "realized_exposure": "0.2500012025805832635276568302193593",
       "target_exposure": "0.25",
       "timestamp": "2026-06-05T00:00:00+00:00"
      }
     ],
     "execution_spec": {
      "allow_long": true,
      "allow_short": true,
      "borrow_rate": "0",
      "cost_model": "synthetic",
      "currency": "USD",
      "exchange_account_specific": false,
      "fee_rate": "0.0010",
      "fill_policy": "next-contiguous-bar-open-after-decision-v1",
      "final_liquidation_policy": "next-observable-open-after-last-fill-v1",
      "funding_rate": "0",
      "initial_equity": "100000",
      "instrument_model": "synthetic-linear-usd-notional-v1",
      "mark_policy": "next-observable-open-v1",
      "optimized": false,
      "protocol_version": "trading-lab.execution.v1",
      "schema_version": "trading-lab.execution.v1",
      "slippage_rate": "0.0005",
      "target_quantity_basis": "reference-market-open-v1"
     },
     "experiment_type": "exploratory",
     "expired_targets": [],
     "fills": [
      {
       "cash_after": "74962.48750000000000000000000000000",
       "equity_after": "99962.48750000000000000000000000000",
       "fee": "25.01250000000000000000000000000000",
       "fill_price": "102.5262375",
       "notional": "25012.50000000000000000000000000000",
       "position_after": "243.9619419370578189802390827030983",
       "quantity_delta": "243.9619419370578189802390827030983",
       "reference_price": "102.475",
       "side": "buy",
       "slippage_cost": "12.50000000000000000000000000000000",
       "source_position_target_hash": "0a83d59dcb0eaa58a91360e45455690fa29963c4537b3d2a8befbf015b3b5ba2",
       "timestamp": "2026-06-04T23:00:00+00:00"
      },
      {
       "cash_after": "74642.95311768041709868260551353989",
       "equity_after": "99524.09707123616240241522322517687",
       "fee": "0.3192151671524304708465479385215897",
       "fill_price": "100.730340",
       "notional": "319.2151671524304708465479385215897",
       "position_after": "247.1309490818012048443843634449442",
       "quantity_delta": "3.1690071447433858641452807418459",
       "reference_price": "100.68",
       "side": "buy",
       "slippage_cost": "0.1595278196663820444010734325445226",
       "source_position_target_hash": "5c30d23f5e4911043801cdce41dab7d8078ee010a83d68bce3054924aaaf329c",
       "timestamp": "2026-06-05T00:00:00+00:00"
      }
     ],
     "gross_metrics": {
      "annualized_sharpe": "-29.58311983912352209949267583659171",
      "average_abs_exposure": "0.2391304347826086956521739130434783",
      "expired_target_count": 0,
      "exposure_time_fraction": "0.9565217391304347826086956521739130",
      "fill_count": 23,
      "final_equity": "97720.28568275690701602943979322617",
      "gross_pnl": "-2279.71431724309298397056020677383",
      "gross_return": "-0.0227971431724309298397056020677383",
      "initial_equity": "100000",
      "max_drawdown": "-0.0311749014056640562065517423185583",
      "net_pnl": "-2279.71431724309298397056020677383",
      "net_return": "-0.0227971431724309298397056020677383",
      "periods_per_year": 8760,
      "rebalance_count": 22,
      "total_execution_cost": "0E-34",
      "total_fees": "0E-31",
      "total_slippage_cost": "0E-34",
      "turnover_ratio": "7.551766032939460678839659894403638"
     },
     "live_execution": false,
     "metrics": {
      "annualized_sharpe": "-47.04144612476324694776743852781537",
      "average_abs_exposure": "0.2392495431362315765031753172310989",
      "expired_target_count": 0,
      "exposure_time_fraction": "0.9565217391304347826086956521739130",
      "fill_count": 23,
      "final_equity": "96616.40931617835066868424711981562",
      "gross_pnl": "-2279.71431724309298397056020677383",
      "gross_return": "-0.0227971431724309298397056020677383",
      "initial_equity": "100000",
      "max_drawdown": "-0.0399685805349851831992772064290473",
      "net_pnl": "-3383.59068382164933131575288018438",
      "net_return": "-0.0338359068382164933131575288018438",
      "periods_per_year": 8760,
      "rebalance_count": 22,
      "total_execution_cost": "1108.251781981380229220339622821892",
      "total_fees": "738.8349005440704595250767644501379",
      "total_slippage_cost": "369.4168814373097696952628583717545",
      "turnover_ratio": "7.552074915720129941473390602392347"
     },
     "position_target_series_hash": "99757fe51330cff8458a67f816f7cc17cc7e7f744bb7cf690b977a1bdfceb83b",
     "schema_version": "trading-lab.economic-backtest-result.v1",
     "signal_series_hash": "2d797f023835bca1d237fc557eebda65376482772f68a815f138187db3a8ead4",
     "spec": {
      "execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "market_corpus_content_hash": "c1f7d1b0228b7f752738744b67ce8c0c5eb62adc599f5e71227f0ecb040b00a5",
      "market_corpus_spec_hash": "9755dacbc42ad0272ad8236265b110b7c27993dbeead3a93e0c75c38505ecec4",
      "product": "BTC-USD",
      "protocol_version": "trading-lab.economic-backtest.v1",
      "result_schema_version": "trading-lab.economic-backtest-result.v1",
      "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
      "schema_version": "trading-lab.economic-backtest.v1",
      "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939",
      "source_benchmark_protocol": "model-lab-experiment-v1",
      "source_benchmark_results_hash": "2bd34276057e0bd5dc2a78f918f49b07c43cd9e4ac452eca874f71ea92df67ef",
      "source_benchmark_spec_hash": "d22a3e891aa9d048e98e17af29fa00d4c0b40cf5004d1748c08b2f8784ab3669",
      "timeframe": "1h"
     },
     "window": {
      "first_fill_at": "2026-06-04T23:00:00+00:00",
      "last_fill_at": "2026-06-05T20:00:00+00:00",
      "liquidation_at": "2026-06-05T21:00:00+00:00"
     }
    },
    "ETH-USD": {
     "confirmatory": false,
     "cost_model": "synthetic",
     "economic_backtest_spec_hash": "921454818194a135fab920b6ee0dd57c69786824fa9e2181eb84d362e22aae16",
     "equity_curve": [
      {
       "cash": "74962.48750000000000000000000000000",
       "cumulative_fees": "25.01250000000000000000000000000000",
       "cumulative_slippage_cost": "12.50000000000000000000000000000000",
       "equity": "99962.48750000000000000000000000000",
       "mark_price": "152.475",
       "position_quantity": "163.9613051319888506312510247581571",
       "position_value": "25000.00000000000000000000000000000",
       "realized_exposure": "0.2500938164428931402892510052833569",
       "target_exposure": "0.25",
       "timestamp": "2026-06-04T23:00:00+00:00"
      },
      {
       "cash": "74750.81558011561808288243974422036",
       "cumulative_fees": "25.22396045942495696015740285292671",
       "cumulative_slippage_cost": "12.60567739101696999508116084604034",
       "equity": "99667.85981943763808616166584686014",
       "mark_price": "150.68",
       "position_quantity": "165.3639782275153968892966956639221",
       "position_value": "24917.04423932202000327922610263978",
       "realized_exposure": "0.2500007954867572567195339710753297",
       "target_exposure": "0.25",
       "timestamp": "2026-06-05T00:00:00+00:00"
      }
     ],
     "execution_spec": {
      "allow_long": true,
      "allow_short": true,
      "borrow_rate": "0",
      "cost_model": "synthetic",
      "currency": "USD",
      "exchange_account_specific": false,
      "fee_rate": "0.0010",
      "fill_policy": "next-contiguous-bar-open-after-decision-v1",
      "final_liquidation_policy": "next-observable-open-after-last-fill-v1",
      "funding_rate": "0",
      "initial_equity": "100000",
      "instrument_model": "synthetic-linear-usd-notional-v1",
      "mark_policy": "next-observable-open-v1",
      "optimized": false,
      "protocol_version": "trading-lab.execution.v1",
      "schema_version": "trading-lab.execution.v1",
      "slippage_rate": "0.0005",
      "target_quantity_basis": "reference-market-open-v1"
     },
     "experiment_type": "exploratory",
     "expired_targets": [],
     "fills": [
      {
       "cash_after": "74962.48750000000000000000000000000",
       "equity_after": "99962.48750000000000000000000000000",
       "fee": "25.01250000000000000000000000000000",
       "fill_price": "152.5512375",
       "notional": "25012.50000000000000000000000000000",
       "position_after": "163.9613051319888506312510247581571",
       "quantity_delta": "163.9613051319888506312510247581571",
       "reference_price": "152.475",
       "side": "buy",
       "slippage_cost": "12.50000000000000000000000000000000",
       "source_position_target_hash": "8413e777618fbca79c90bec3623503849674bb2380e0008f556160fcdf817801",
       "timestamp": "2026-06-04T23:00:00+00:00"
      },
      {
       "cash_after": "74750.81558011561808288243974422036",
       "equity_after": "99667.85981943763808616166584686014",
       "fee": "0.2114604594249569601574028529267105",
       "fill_price": "150.755340",
       "notional": "211.4604594249569601574028529267105",
       "position_after": "165.3639782275153968892966956639221",
       "quantity_delta": "1.4026730955265462580456709057650",
       "reference_price": "150.68",
       "side": "buy",
       "slippage_cost": "0.1056773910169699950811608460403351",
       "source_position_target_hash": "295acc37a2e19bdc2b649310928efebfe8985a0eb17fcef4447778719f058f6a",
       "timestamp": "2026-06-05T00:00:00+00:00"
      }
     ],
     "gross_metrics": {
      "annualized_sharpe": "-29.77244379089185904083814699140240",
      "average_abs_exposure": "0.2391304347826086956521739130434783",
      "expired_target_count": 0,
      "exposure_time_fraction": "0.9565217391304347826086956521739130",
      "fill_count": 23,
      "final_equity": "98456.35574868177707797252647938155",
      "gross_pnl": "-1543.64425131822292202747352061845",
      "gross_return": "-0.0154364425131822292202747352061845",
      "initial_equity": "100000",
      "max_drawdown": "-0.0211044229723711865171493137942709",
      "net_pnl": "-1543.64425131822292202747352061845",
      "net_return": "-0.0154364425131822292202747352061845",
      "periods_per_year": 8760,
      "rebalance_count": 22,
      "total_execution_cost": "0E-34",
      "total_fees": "0E-31",
      "total_slippage_cost": "0E-34",
      "turnover_ratio": "7.534725014429719583546503531623699"
     },
     "live_execution": false,
     "metrics": {
      "annualized_sharpe": "-56.57360122284256947523023796796148",
      "average_abs_exposure": "0.2392492759953485840069142153719202",
      "expired_target_count": 0,
      "exposure_time_fraction": "0.9565217391304347826086956521739130",
      "fill_count": 23,
      "final_equity": "97347.56299433247292042932124976148",
      "gross_pnl": "-1543.64425131822292202747352061845",
      "gross_return": "-0.0154364425131822292202747352061845",
      "initial_equity": "100000",
      "max_drawdown": "-0.0299528645645495031333622280232919",
      "net_pnl": "-2652.43700566752707957067875023852",
      "net_return": "-0.0265243700566752707957067875023852",
      "periods_per_year": 8760,
      "rebalance_count": 22,
      "total_execution_cost": "1111.770884186176928010366756024382",
      "total_fees": "741.1808462351381988986030974015871",
      "total_slippage_cost": "370.5900379510387291117636586227950",
      "turnover_ratio": "7.535024302837361801704559235501045"
     },
     "position_target_series_hash": "af0b429b934cae1549571c0e8e15fa1d10a4e56960b1f622e8aa104b33348ddb",
     "schema_version": "trading-lab.economic-backtest-result.v1",
     "signal_series_hash": "ea58d0a000bbb938f438d1faba6df9e3914eee7ee203208fbe659fdddfbb528b",
     "spec": {
      "execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "market_corpus_content_hash": "c1f7d1b0228b7f752738744b67ce8c0c5eb62adc599f5e71227f0ecb040b00a5",
      "market_corpus_spec_hash": "9755dacbc42ad0272ad8236265b110b7c27993dbeead3a93e0c75c38505ecec4",
      "product": "ETH-USD",
      "protocol_version": "trading-lab.economic-backtest.v1",
      "result_schema_version": "trading-lab.economic-backtest-result.v1",
      "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
      "schema_version": "trading-lab.economic-backtest.v1",
      "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939",
      "source_benchmark_protocol": "model-lab-experiment-v1",
      "source_benchmark_results_hash": "5e4a8b1afc5f286ffd57cefb28a3d7de82d5411965d95c4760482f1e74dcc8db",
      "source_benchmark_spec_hash": "d22a3e891aa9d048e98e17af29fa00d4c0b40cf5004d1748c08b2f8784ab3669",
      "timeframe": "1h"
     },
     "window": {
      "first_fill_at": "2026-06-04T23:00:00+00:00",
      "last_fill_at": "2026-06-05T20:00:00+00:00",
      "liquidation_at": "2026-06-05T21:00:00+00:00"
     }
    }
   },
   "criteria_met": {
    "BTC-USD": false,
    "ETH-USD": false
   },
   "fingerprint": "306bd0e6cdcba128e5fcbd29d7e05b7df44690eee9f851e6b93dde6a6981d736",
   "limitations": [
    "synthetic infrastructure demonstration; no edge claim",
    "bit-for-bit reproducibility requires the recorded numeric runtime",
    "null/negative outcomes are retained"
   ],
   "manifest": {
    "artifacts": {
     "backtest_hash": "b5d276eabedf07929c55cbd69185a7784045cb660d1415fba0204e786fd7ea95",
     "model_hashes": {
      "BTC-USD": "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941",
      "ETH-USD": "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941"
     },
     "prediction_hash": "edfc35a949267c3955deb3823e5f706e020c6fda32fe7d54b9771d290135fd6a",
     "prepared_hash": "d22a3e891aa9d048e98e17af29fa00d4c0b40cf5004d1748c08b2f8784ab3669",
     "shadow_hash": "d6a4bb05610b5b58ada7b6556560d302f90441bb727ab0e5c78a4e4292d4fa18"
    },
    "baselines": [
     "ZERO",
     "TRAIN_MEAN"
    ],
    "budgets": {
     "cpu_seconds": 60,
     "memory_mb": 1024,
     "output_mb": 32,
     "wall_seconds": 120
    },
    "costs": {
     "backtest": {
      "allow_long": true,
      "allow_short": true,
      "borrow_rate": "0",
      "cost_model": "synthetic",
      "currency": "USD",
      "exchange_account_specific": false,
      "fee_rate": "0.0010",
      "fill_policy": "next-contiguous-bar-open-after-decision-v1",
      "final_liquidation_policy": "next-observable-open-after-last-fill-v1",
      "funding_rate": "0",
      "initial_equity": "100000",
      "instrument_model": "synthetic-linear-usd-notional-v1",
      "mark_policy": "next-observable-open-v1",
      "optimized": false,
      "protocol_version": "trading-lab.execution.v1",
      "schema_version": "trading-lab.execution.v1",
      "slippage_rate": "0.0005",
      "target_quantity_basis": "reference-market-open-v1"
     },
     "backtest_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
     "calendar": "coinbase-hourly-utc-v1",
     "paper": {
      "backtest_execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "cost_model": "synthetic",
      "currency": "USD",
      "differs_from_backtest": [
       "a live session never liquidates a terminal position",
       "a fill is recorded one bar after the price it is filled at becomes observable, so paper and backtest timelines are not directly comparable"
      ],
      "fee_rate": "0.0010",
      "fill_observation_policy": "recorded-when-the-fill-bar-closes-v1",
      "fill_price_policy": "next-contiguous-bar-open-after-decision-v1",
      "initial_equity": "100000",
      "mark_policy": "latest-observed-open-v1",
      "protocol_version": "trading-lab.paper-execution.v1",
      "schema_version": "trading-lab.paper-execution.v1",
      "slippage_rate": "0.0005",
      "terminal_liquidation": false
     },
     "paper_hash": "bb944167fbcc9f35d93b269c1422e2c658235a8c49524b165206570963c19bfa",
     "risk_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
     "signal_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
    },
    "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
    "decision_criteria": {
     "commercial_claim": false,
     "metric": "test_mae",
     "rule": "strictly below ZERO and TRAIN_MEAN"
    },
    "experiment_id": "synthetic-exp-2bbebe5bd8f5a865a5f00062",
    "hypothesis": {
     "falsification": "test MAE fails to beat both ZERO and TRAIN_MEAN",
     "mechanism": "A recent synthetic price move may continue across the next four hourly bars",
     "scientific_claim": false,
     "statement": "Four times the hourly return reduces synthetic four-hour MAE relative to fixed baselines"
    },
    "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
    "parameters": {
     "alpha": null,
     "model_id": "local-momentum-v1",
     "runtime": {
      "blas_threads": 1,
      "decimal_precision": 34,
      "implementation": "model-lab-experiment-v1",
      "numpy": "2.5.3",
      "python": "3.12.3",
      "scikit-learn": "1.9.1",
      "source_hashes": {
       "economic_backtest.py": "2c526c63e2e13e3c9f027f9a63a48bdde7de4985feff3c4e000606e1d7a1fad3",
       "event_features/v2.py": "4397fb3c3ce372d8d72ad904ce4539e72579de7f1395ee9846de04ba65a640a3",
       "market_dataset.py": "8b9fc81ee2f71f8d882bdd3918d2559ace527e1679e3d413e0021563292806f0",
       "market_indicators.py": "9a13563336c1cce105eea0a712f5f97ec00c40bab383cc1b39532a4965727713",
       "models.py": "6899d8fe9d420dc75f0b969f841f991f54628e5c25617b3c2834012b612d4a48",
       "paper_engine.py": "8ca82a3147c43dc6805e427a98115c93af56273ceba12ed49c15fe9f8302d8b5",
       "paper_event_store.py": "92b9b81b32889222d0ad04aeeb202c1480837b835deb106eae265be0c351dd6e",
       "paper_model.py": "73db9d2bd04fbde9b6c03f6209170af88ceee68dc36cca1b1333b827ee574099",
       "platform/adapters.py": "762badf4ade4ab5bbc21b0fcb640cdd8761c5eda83933720da0fed539ce51e01",
       "platform/contracts.py": "3e9df25b8f28e4bc460f827443de763477f82ded17bce0ac410e1d99c77358b0",
       "platform/datasets.py": "37b9aca566081f367e2ace20852ef347c2f57b6f9e0b6ba92ef91e282d6097b2",
       "platform/experiments.py": "da6517b65c4ab75bdbdf93d6907c6c54f644dd62cca842175cf246970c548fd8",
       "platform/jobs.py": "035ecc87fa3f0af3e6c5cab64dd484d65020a8379e6bcb911ff91f5785ee645c",
       "platform/local_momentum.py": "e9f97a33bdcc1cf8fad4c9c98aedcd5194be21092bda3f2493e98ff237c097fe",
       "platform/prices.py": "1a9fef540fd4c8de96aeddaaf357a37993f63ad9e352bfcdbc1ada5cb8cb699d",
       "platform/snapshot.py": "afbe13c245a03860c7537336949c9953b82cdb2da46eedaf061f540825898a3f",
       "research_protection.py": "bc441b7134c4b4a162b6175db615c407c4c3a5318b90a1db7f697dbacf7bd4e1",
       "risk_engine.py": "56ef562fcee24f896debb39114b64f456d82ee41640e876997f9f7343af27fd2",
       "signal_engine.py": "f4033ff0cccc833d549eaeeaf95dfedb2a8510fbb226e62d93b9cd6fa8bceefb",
       "walk_forward.py": "30bab6ea43b956a7c11b7c75acb91442deb1695aa98e36ff6886189e0d369883"
      },
      "xgboost": "3.4.1"
     },
     "transformations": "training only; no validation refit"
    },
    "schema": "experiment-manifest-v1",
    "splits": {
     "embargo_seconds": 3600,
     "exclusions": [
      {
       "decision_at": "2026-06-03T19:00:00+00:00",
       "product": "BTC-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      },
      {
       "decision_at": "2026-06-03T19:00:00+00:00",
       "product": "ETH-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      },
      {
       "decision_at": "2026-06-03T20:00:00+00:00",
       "product": "BTC-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      }
     ],
     "method": "temporal-purge-embargo-v1",
     "purge_seconds": 14400,
     "refit_after_validation": false,
     "roles": {
      "test": {
       "first": "2026-06-04T23:00:00+00:00",
       "last": "2026-06-05T20:00:00+00:00",
       "population_hash": "172d1dbad8bfd2a33dd09ae7b9213fb2be0c31b676ae1ae212de31f5c7bc429c",
       "rows": 44
      },
      "train": {
       "first": "2026-06-02T02:00:00+00:00",
       "last": "2026-06-03T18:00:00+00:00",
       "population_hash": "58f19c0c6ce558737945033e03baa06523445a10269a05d22785c3802b21ac8e",
       "rows": 82
      },
      "validation": {
       "first": "2026-06-04T00:00:00+00:00",
       "last": "2026-06-04T17:00:00+00:00",
       "population_hash": "449b0f95401bd9a218f3b0028b5b8eb56d46f5145d154496225b47aaadae9508",
       "rows": 36
      }
     },
     "test_start": "2026-06-04T22:00:00+00:00",
     "train_label_available_before": "2026-06-03T23:00:00+00:00",
     "validation_label_available_before": "2026-06-04T22:00:00+00:00",
     "validation_start": "2026-06-03T23:00:00+00:00"
    },
    "status": "COMPLETE",
    "synthetic": true,
    "version": "model-lab-experiment-v1"
   },
   "metrics": {
    "BTC-USD": {
     "test": {
      "TRAIN_MEAN": {
       "count": 22,
       "mae": "0.01360100728375476320907077763833508",
       "rank_ic": null,
       "rmse": "0.01394165358428465914299365785046575"
      },
      "ZERO": {
       "count": 22,
       "mae": "0.01359754365399714288084726677141579",
       "rank_ic": null,
       "rmse": "0.01394240259485934824370010328381153"
      },
      "model": {
       "count": 22,
       "mae": "0.05801920745594743995400823289838586",
       "rank_ic": "-0.5031055900621118012422360248447206",
       "rmse": "0.06184573562999365492880401123020755"
      }
     },
     "validation": {
      "TRAIN_MEAN": {
       "count": 18,
       "mae": "0.01349113970460872358940067390446616",
       "rank_ic": null,
       "rmse": "0.01382897218018530995953012846311317"
      },
      "ZERO": {
       "count": 18,
       "mae": "0.01348690637934940985490527173378702",
       "rank_ic": null,
       "rmse": "0.01382893505809244680889790723296991"
      },
      "model": {
       "count": 18,
       "mae": "0.05837148696847297183194930539319578",
       "rank_ic": "-0.5252837977296181630546955624355003",
       "rmse": "0.06230739918623463412425404076643930"
      }
     }
    },
    "ETH-USD": {
     "test": {
      "TRAIN_MEAN": {
       "count": 22,
       "mae": "0.009137319562339387419162666832931023",
       "rank_ic": null,
       "rmse": "0.009365874053176172528549350120030634"
      },
      "ZERO": {
       "count": 22,
       "mae": "0.009131137262516414119984698969833982",
       "rank_ic": null,
       "rmse": "0.009367245227667866997171758270263886"
      },
      "model": {
       "count": 22,
       "mae": "0.03896324012305827425278063698898205",
       "rank_ic": "-0.5031055900621118012422360248447206",
       "rmse": "0.04155979721142193232302717468205185"
      }
     },
     "validation": {
      "TRAIN_MEAN": {
       "count": 18,
       "mae": "0.009060016017228983409152730278468311",
       "rank_ic": null,
       "rmse": "0.009286726565288134107592646068787552"
      },
      "ZERO": {
       "count": 18,
       "mae": "0.00905245987300090493237965844579415",
       "rank_ic": null,
       "rmse": "0.009286697190247731088217859680947127"
      },
      "model": {
       "count": 18,
       "mae": "0.03918527434630048730794196792850591",
       "rank_ic": "-0.5252837977296181630546955624355003",
       "rmse": "0.04185451092012567347156991069481126"
      }
     }
    }
   },
   "models": {
    "BTC-USD": {
     "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
     "multiplier": "4",
     "schema": "local-momentum-artifact-v1",
     "synthetic": true,
     "train_rows": 41
    },
    "ETH-USD": {
     "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
     "multiplier": "4",
     "schema": "local-momentum-artifact-v1",
     "synthetic": true,
     "train_rows": 41
    }
   },
   "predictions": [
    {
     "fingerprint": "d6c797f8910553340ac3bcf3b4af6226a482054389e6b0a893a746780fca5652",
     "record": {
      "artifact_hash": "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941",
      "costs": null,
      "decision_at": "2026-06-04T00:00:00+00:00",
      "errors": [],
      "event_ids": [],
      "execution": null,
      "features_hash": "e42f6ffc5c2814d97101fc1122b84217ef19c98db9d8f6296f00e9ca443880b3",
      "horizon_seconds": 14400,
      "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
      "model_id": "local-momentum-v1",
      "outputs": {
       "class": null,
       "probabilities": null,
       "quantiles": null,
       "return": "0.03990074441687344913151364764267990",
       "scenarios": null,
       "target_price": null
      },
      "prediction_id": "cf6539df76e073f4b3b5e7c4a652be659a3163b127014f8c542b38be0110c793",
      "product": "BTC-USD",
      "proposed_position": null,
      "risk": null,
      "schema": "prediction-record-v1",
      "signal": null,
      "snapshot_hash": "6952ed0245b6b391bc48e4b76bcd77690ff600afed8281cb06d22e4d23ea8f77",
      "synthetic": true,
      "uncertainty": null
     },
     "split": "validation"
    },
    {
     "fingerprint": "660d7c427a24092806746b39859574303fd78c071f485a88d0953958ad0ff8dc",
     "record": {
      "artifact_hash": "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941",
      "costs": null,
      "decision_at": "2026-06-04T01:00:00+00:00",
      "errors": [],
      "event_ids": [],
      "execution": null,
      "features_hash": "7b86ef7ded35749fd3f0694920a5bdcb915db46abf5d9f778a882943d79d5850",
      "horizon_seconds": 14400,
      "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
      "model_id": "local-momentum-v1",
      "outputs": {
       "class": null,
       "probabilities": null,
       "quantiles": null,
       "return": "0.03950665814947668419242297675789888",
       "scenarios": null,
       "target_price": null
      },
      "prediction_id": "bfe2e5e86bc2b2d69160fe9424d8b9acf7b23f08ab12b6724c77362211774a7c",
      "product": "BTC-USD",
      "proposed_position": null,
      "risk": null,
      "schema": "prediction-record-v1",
      "signal": null,
      "snapshot_hash": "93a91547b6217c9c832b07ef8abe126ea194aa34a29ad634fd069004afdd0656",
      "synthetic": true,
      "uncertainty": null
     },
     "split": "validation"
    }
   ],
   "prepared": {
    "artifacts": {},
    "baselines": [
     "ZERO",
     "TRAIN_MEAN"
    ],
    "budgets": {
     "cpu_seconds": 60,
     "memory_mb": 1024,
     "output_mb": 32,
     "wall_seconds": 120
    },
    "costs": {
     "backtest": {
      "allow_long": true,
      "allow_short": true,
      "borrow_rate": "0",
      "cost_model": "synthetic",
      "currency": "USD",
      "exchange_account_specific": false,
      "fee_rate": "0.0010",
      "fill_policy": "next-contiguous-bar-open-after-decision-v1",
      "final_liquidation_policy": "next-observable-open-after-last-fill-v1",
      "funding_rate": "0",
      "initial_equity": "100000",
      "instrument_model": "synthetic-linear-usd-notional-v1",
      "mark_policy": "next-observable-open-v1",
      "optimized": false,
      "protocol_version": "trading-lab.execution.v1",
      "schema_version": "trading-lab.execution.v1",
      "slippage_rate": "0.0005",
      "target_quantity_basis": "reference-market-open-v1"
     },
     "backtest_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
     "calendar": "coinbase-hourly-utc-v1",
     "paper": {
      "backtest_execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "cost_model": "synthetic",
      "currency": "USD",
      "differs_from_backtest": [
       "a live session never liquidates a terminal position",
       "a fill is recorded one bar after the price it is filled at becomes observable, so paper and backtest timelines are not directly comparable"
      ],
      "fee_rate": "0.0010",
      "fill_observation_policy": "recorded-when-the-fill-bar-closes-v1",
      "fill_price_policy": "next-contiguous-bar-open-after-decision-v1",
      "initial_equity": "100000",
      "mark_policy": "latest-observed-open-v1",
      "protocol_version": "trading-lab.paper-execution.v1",
      "schema_version": "trading-lab.paper-execution.v1",
      "slippage_rate": "0.0005",
      "terminal_liquidation": false
     },
     "paper_hash": "bb944167fbcc9f35d93b269c1422e2c658235a8c49524b165206570963c19bfa",
     "risk_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
     "signal_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
    },
    "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
    "decision_criteria": {
     "commercial_claim": false,
     "metric": "test_mae",
     "rule": "strictly below ZERO and TRAIN_MEAN"
    },
    "experiment_id": "synthetic-exp-2bbebe5bd8f5a865a5f00062",
    "hypothesis": {
     "falsification": "test MAE fails to beat both ZERO and TRAIN_MEAN",
     "mechanism": "A recent synthetic price move may continue across the next four hourly bars",
     "scientific_claim": false,
     "statement": "Four times the hourly return reduces synthetic four-hour MAE relative to fixed baselines"
    },
    "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
    "parameters": {
     "alpha": null,
     "model_id": "local-momentum-v1",
     "runtime": {
      "blas_threads": 1,
      "decimal_precision": 34,
      "implementation": "model-lab-experiment-v1",
      "numpy": "2.5.3",
      "python": "3.12.3",
      "scikit-learn": "1.9.1",
      "source_hashes": {
       "economic_backtest.py": "2c526c63e2e13e3c9f027f9a63a48bdde7de4985feff3c4e000606e1d7a1fad3",
       "event_features/v2.py": "4397fb3c3ce372d8d72ad904ce4539e72579de7f1395ee9846de04ba65a640a3",
       "market_dataset.py": "8b9fc81ee2f71f8d882bdd3918d2559ace527e1679e3d413e0021563292806f0",
       "market_indicators.py": "9a13563336c1cce105eea0a712f5f97ec00c40bab383cc1b39532a4965727713",
       "models.py": "6899d8fe9d420dc75f0b969f841f991f54628e5c25617b3c2834012b612d4a48",
       "paper_engine.py": "8ca82a3147c43dc6805e427a98115c93af56273ceba12ed49c15fe9f8302d8b5",
       "paper_event_store.py": "92b9b81b32889222d0ad04aeeb202c1480837b835deb106eae265be0c351dd6e",
       "paper_model.py": "73db9d2bd04fbde9b6c03f6209170af88ceee68dc36cca1b1333b827ee574099",
       "platform/adapters.py": "762badf4ade4ab5bbc21b0fcb640cdd8761c5eda83933720da0fed539ce51e01",
       "platform/contracts.py": "3e9df25b8f28e4bc460f827443de763477f82ded17bce0ac410e1d99c77358b0",
       "platform/datasets.py": "37b9aca566081f367e2ace20852ef347c2f57b6f9e0b6ba92ef91e282d6097b2",
       "platform/experiments.py": "da6517b65c4ab75bdbdf93d6907c6c54f644dd62cca842175cf246970c548fd8",
       "platform/jobs.py": "035ecc87fa3f0af3e6c5cab64dd484d65020a8379e6bcb911ff91f5785ee645c",
       "platform/local_momentum.py": "e9f97a33bdcc1cf8fad4c9c98aedcd5194be21092bda3f2493e98ff237c097fe",
       "platform/prices.py": "1a9fef540fd4c8de96aeddaaf357a37993f63ad9e352bfcdbc1ada5cb8cb699d",
       "platform/snapshot.py": "afbe13c245a03860c7537336949c9953b82cdb2da46eedaf061f540825898a3f",
       "research_protection.py": "bc441b7134c4b4a162b6175db615c407c4c3a5318b90a1db7f697dbacf7bd4e1",
       "risk_engine.py": "56ef562fcee24f896debb39114b64f456d82ee41640e876997f9f7343af27fd2",
       "signal_engine.py": "f4033ff0cccc833d549eaeeaf95dfedb2a8510fbb226e62d93b9cd6fa8bceefb",
       "walk_forward.py": "30bab6ea43b956a7c11b7c75acb91442deb1695aa98e36ff6886189e0d369883"
      },
      "xgboost": "3.4.1"
     },
     "transformations": "training only; no validation refit"
    },
    "schema": "experiment-manifest-v1",
    "splits": {
     "embargo_seconds": 3600,
     "exclusions": [
      {
       "decision_at": "2026-06-03T19:00:00+00:00",
       "product": "BTC-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      },
      {
       "decision_at": "2026-06-03T19:00:00+00:00",
       "product": "ETH-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      },
      {
       "decision_at": "2026-06-03T20:00:00+00:00",
       "product": "BTC-USD",
       "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
      }
     ],
     "method": "temporal-purge-embargo-v1",
     "purge_seconds": 14400,
     "refit_after_validation": false,
     "roles": {
      "test": {
       "first": "2026-06-04T23:00:00+00:00",
       "last": "2026-06-05T20:00:00+00:00",
       "population_hash": "172d1dbad8bfd2a33dd09ae7b9213fb2be0c31b676ae1ae212de31f5c7bc429c",
       "rows": 44
      },
      "train": {
       "first": "2026-06-02T02:00:00+00:00",
       "last": "2026-06-03T18:00:00+00:00",
       "population_hash": "58f19c0c6ce558737945033e03baa06523445a10269a05d22785c3802b21ac8e",
       "rows": 82
      },
      "validation": {
       "first": "2026-06-04T00:00:00+00:00",
       "last": "2026-06-04T17:00:00+00:00",
       "population_hash": "449b0f95401bd9a218f3b0028b5b8eb56d46f5145d154496225b47aaadae9508",
       "rows": 36
      }
     },
     "test_start": "2026-06-04T22:00:00+00:00",
     "train_label_available_before": "2026-06-03T23:00:00+00:00",
     "validation_label_available_before": "2026-06-04T22:00:00+00:00",
     "validation_start": "2026-06-03T23:00:00+00:00"
    },
    "status": "PREPARED",
    "synthetic": true,
    "version": "model-lab-experiment-v1"
   },
   "schema": "model-lab-result-v1",
   "shadow": {
    "broker_connected": false,
    "chain": {
     "events": 364,
     "head_hash": "5fd310c985dfe57529b1e00b3f8746783252a89168d0188923d4b8e1c7ae3d68",
     "session_id": "synthetic-exp-2bbebe5bd8f5a865a5f00062",
     "verified": true
    },
    "events": 364,
    "limitations": [
     "synthetic prices and costs; no scientific finding",
     "accounts independent per product",
     "session paper_model_spec_hash binds the new experiment, never a frozen paper model"
    ],
    "mode": "offline synthetic shadow replay",
    "products": {
     "BTC-USD": {
      "cash": "71867.29452003745553674817278774847",
      "cumulative_fees": "811.8133572348459583394969506487347",
      "cumulative_slippage_cost": "405.8999498693797645900633947576506",
      "fill_count": 25,
      "last_bar": "2026-06-05T23:00:00+00:00",
      "last_mark": "100.895",
      "pending_target": {
       "position_target_hash": "96f1185dc17e9a6d404c8fd5050097913469c178bed8199736ab9a67c11ff2a8",
       "target_exposure": "-0.25",
       "timestamp": "2026-06-05T23:00:00+00:00"
      },
      "position_quantity": "237.4341010949052115590155220222309",
      "product": "BTC-USD",
      "turnover_sum": "8.308779402709276711999001386605448"
     },
     "ETH-USD": {
      "cash": "72580.86267885664721926141551551099",
      "cumulative_fees": "814.5419883040675602315447702516804",
      "cumulative_slippage_cost": "407.2644448193117751224110389686714",
      "fill_count": 25,
      "last_bar": "2026-06-05T23:00:00+00:00",
      "last_mark": "150.895",
      "pending_target": {
       "position_target_hash": "d4b37dbfeb8da36df0b0bce89255de51b87aff91f53085df984adba35ac48ed8",
       "target_exposure": "-0.25",
       "timestamp": "2026-06-05T23:00:00+00:00"
      },
      "position_quantity": "160.3347945470124093811253346880581",
      "product": "ETH-USD",
      "turnover_sum": "8.289519905738056318438707086274237"
     }
    },
    "session_hash": "241097fc9de7613160168172018cea654668b7325c945f718efdeb2b5601fc29",
    "session_spec": {
     "broker_connected": false,
     "initial_equity": "100000",
     "model_fitted_hashes": [
      [
       "BTC-USD",
       "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941"
      ],
      [
       "ETH-USD",
       "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941"
      ]
     ],
     "paper_execution_spec_hash": "bb944167fbcc9f35d93b269c1422e2c658235a8c49524b165206570963c19bfa",
     "paper_model_spec_hash": "d22a3e891aa9d048e98e17af29fa00d4c0b40cf5004d1748c08b2f8784ab3669",
     "products": [
      "BTC-USD",
      "ETH-USD"
     ],
     "protected_holdout_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
     "real_money": false,
     "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
     "schema_version": "trading-lab.paper-session.v1",
     "shadow_mode": true,
     "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939",
     "timeframe": "1h"
    },
    "synthetic": true
   },
   "synthetic": true
  },
  "result_hash": "1edb219e98cfe5988a82d8c42e375f88997e5bd4c475516c71acbc4a0cd2e9c4",
  "state": "COMPLETE"
 },
 "health": {
  "head_hash": "94ccdb5b3fc759dce2e197d5d68210840a1f9ffdb70148250b0b5994274e650d",
  "method": {
   "bins": 5,
   "distribution": "finite-observed-summary-v1",
   "drift": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
   "edge": "paired-descriptive-mse-reduction-v1",
   "error_rate": 0.05,
   "freshness_seconds": 7200,
   "ks_threshold": 0.3,
   "latency_ms": 1000,
   "limitations": [
    "descriptive monitoring thresholds; no statistical edge verdict",
    "overlapping horizons and serial dependence are not corrected by an interval",
    "reference and current samples must share model, artifact, product and horizon"
   ],
   "minimum_sample": 10,
   "performance_drop": "edge decline > 0.05 vs reference",
   "pseudocount": 1e-06,
   "psi_threshold": 0.2,
   "uncertainty": null,
   "version": "monitoring-method-v1"
  },
  "network_requests": 0,
  "read_only": true,
  "records": 862,
  "regime_definition": {
   "clock": "features actually used at decision time; never realized labels",
   "definition": [
    "HIGH_VOL if atr_pct_14 >= 0.03",
    "otherwise UP if return_4 >= 0.01",
    "otherwise DOWN if return_4 <= -0.01",
    "otherwise RANGE",
    "UNKNOWN when either feature is unavailable"
   ],
   "inputs": [
    "return_4",
    "atr_pct_14"
   ],
   "version": "price-regimes-v1"
  },
  "schema": "research-observability-store-v1",
  "verified": true,
  "workload_execution": false
 },
 "hypothesesPage": {
  "as_of": "2026-10-05T10:30:50.310365+00:00",
  "page": {
   "has_more": true,
   "next_cursor": "eyJhZnRlciI6IjIiLCJhcGkiOiJ0cmFkaW5nLWxhYi5hcHAtYXBpLnYxIiwiZW5kcG9pbnQiOiJoeXBvdGhlc2lzIiwicHJvZHVjdCI6InJlc2VhcmNoIiwicSI6IjcwMDJlZTMyODNiMDQ4YjBkMDUwOWZjMmExZWVkODg2IiwidiI6InRyYWRpbmctbGFiLmFwcC1hcGkuY3Vyc29yLnYxIn0",
   "returned": 2
  },
  "records": [
   {
    "chain_hash": "5dbf5ad00a9ef7d6405a5443d6b9af4917315156423d4415d8ae91b8123c40e9",
    "identity": "5787a940a7799947b846fca60c7818dba3cbb4cf61f45eb5d201d34048f645b2",
    "payload": {
     "availability": {
      "pairing": "A and B use exactly the same decisions, labels, costs and splits; a decision where a required source is not RESOLVED is excluded from both; a family's test set is the decisions common to all its products",
      "protection": {
       "admissible_ranges_price_and_label": {
        "crypto": {
         "decision_clock": "bar close = bar opening + 1h",
         "event_window_clean_from": "2026-12-31T00:00:00Z",
         "label_horizon_bars": 4,
         "price_warmup_bars": 25,
         "runs": [
          {
           "decisions": 9475,
           "first_bar_open": "2025-08-02T01:00:00Z",
           "first_decision_at": "2025-08-02T02:00:00Z",
           "last_bar_open": "2026-08-31T19:00:00Z",
           "last_decision_at": "2026-08-31T20:00:00Z",
           "run_bars": 9504
          },
          {
           "decisions": 5803,
           "first_bar_open": "2026-12-02T01:00:00Z",
           "first_decision_at": "2026-12-02T02:00:00Z",
           "last_bar_open": "2027-07-31T19:00:00Z",
           "last_decision_at": "2027-07-31T20:00:00Z",
           "run_bars": 5832
          }
         ]
        },
        "equity": {
         "decision_clock": "session close (calendar close_at, early closes honoured)",
         "event_window_clean_from": "2027-03-31T00:00:00Z",
         "label_horizon_sessions": 5,
         "price_warmup_sessions": 20,
         "runs": [
          {
           "decisions": 560,
           "first_bar_open": "2024-08-29T13:30:00Z",
           "first_decision_at": "2024-08-29T20:00:00Z",
           "last_bar_open": "2026-11-20T14:30:00Z",
           "last_decision_at": "2026-11-20T21:00:00Z",
           "run_bars": 585
          },
          {
           "decisions": 81,
           "first_bar_open": "2027-03-30T13:30:00Z",
           "first_decision_at": "2027-03-30T20:00:00Z",
           "last_bar_open": "2027-07-23T13:30:00Z",
           "last_decision_at": "2027-07-23T20:00:00Z",
           "run_bars": 106
          }
         ]
        }
       },
       "event_window_days": 30,
       "label_horizon": {
        "crypto_bars": 4,
        "equity_sessions": 5
       },
       "other_reserved_intervals": "none found: the walk-forward and paper-replay periods are exploratory/consumed, not reserved; closed pilots are archives, not intervals",
       "price_warmup": {
        "crypto_bars": 25,
        "equity_sessions": 20
       },
       "rule": "a decision is admissible only if no bar of its price features, no bar of its label window and no instant of its 30-day event window (T-30d, T] lies in a protected interval of its product",
       "table": {
        "AAPL": [
         {
          "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
          "calendar_id": "US_EQUITY_REGULAR",
          "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
          "declared_end": "2027-02-28T23:59:59Z",
          "end_exclusive": "2027-03-01T00:00:00Z",
          "holdout_id": "equity_confirmatory_2027q1",
          "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
          "observed": false,
          "products": [
           "AAPL",
           "MSFT",
           "NVDA",
           "QQQ"
          ],
          "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
          "single_use": true,
          "start": "2026-12-01T00:00:00Z",
          "timeframe": "1d",
          "unit": "session dates"
         }
        ],
        "BTC-USD": [
         {
          "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
          "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
          "end_exclusive": "2026-12-01T00:00:00Z",
          "holdout_id": "coinbase_confirmatory_2026q4",
          "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
          "last_protected_bar_open": "2026-11-30T23:00:00Z",
          "observed": false,
          "products": [
           "BTC-USD",
           "ETH-USD"
          ],
          "single_use": true,
          "start": "2026-09-01T00:00:00Z",
          "timeframe": "1h",
          "unit": "bar opening instants"
         }
        ],
        "ETH-USD": [
         {
          "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
          "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
          "end_exclusive": "2026-12-01T00:00:00Z",
          "holdout_id": "coinbase_confirmatory_2026q4",
          "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
          "last_protected_bar_open": "2026-11-30T23:00:00Z",
          "observed": false,
          "products": [
           "BTC-USD",
           "ETH-USD"
          ],
          "single_use": true,
          "start": "2026-09-01T00:00:00Z",
          "timeframe": "1h",
          "unit": "bar opening instants"
         }
        ],
        "MSFT": [
         {
          "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
          "calendar_id": "US_EQUITY_REGULAR",
          "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
          "declared_end": "2027-02-28T23:59:59Z",
          "end_exclusive": "2027-03-01T00:00:00Z",
          "holdout_id": "equity_confirmatory_2027q1",
          "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
          "observed": false,
          "products": [
           "AAPL",
           "MSFT",
           "NVDA",
           "QQQ"
          ],
          "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
          "single_use": true,
          "start": "2026-12-01T00:00:00Z",
          "timeframe": "1d",
          "unit": "session dates"
         }
        ],
        "NVDA": [
         {
          "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
          "calendar_id": "US_EQUITY_REGULAR",
          "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
          "declared_end": "2027-02-28T23:59:59Z",
          "end_exclusive": "2027-03-01T00:00:00Z",
          "holdout_id": "equity_confirmatory_2027q1",
          "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
          "observed": false,
          "products": [
           "AAPL",
           "MSFT",
           "NVDA",
           "QQQ"
          ],
          "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
          "single_use": true,
          "start": "2026-12-01T00:00:00Z",
          "timeframe": "1d",
          "unit": "session dates"
         }
        ],
        "QQQ": [
         {
          "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
          "calendar_id": "US_EQUITY_REGULAR",
          "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
          "declared_end": "2027-02-28T23:59:59Z",
          "end_exclusive": "2027-03-01T00:00:00Z",
          "holdout_id": "equity_confirmatory_2027q1",
          "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
          "observed": false,
          "products": [
           "AAPL",
           "MSFT",
           "NVDA",
           "QQQ"
          ],
          "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
          "single_use": true,
          "start": "2026-12-01T00:00:00Z",
          "timeframe": "1d",
          "unit": "session dates"
         }
        ]
       }
      },
      "warmup": {
       "capture_start_plus_30_days": "2027-04-01T00:00:00Z",
       "event": "for every required source: state RESOLVED at T, T >= attested coverage start + 30 days and T >= max(first valid read availability, last inventory availability at T) + 30 days, per issuer for EDGAR",
       "price": "the feature-complete index of the real indicators (see protection.price_warmup), inside the series"
      }
     },
     "baselines": [
      "PRICES_ONLY",
      "ZERO",
      "TRAIN_MEAN"
     ],
     "budgets": {
      "execution_enabled": false,
      "max_rows": 10000,
      "max_trials": 1,
      "protocol_budgets": {
       "capture_window": {
        "days": 153,
        "end_exclusive": "2027-08-02T00:00:00Z",
        "rule": "FOMC and EDGAR run for the whole window; stop when the last supplied decision is RESOLVED",
        "start": "2027-03-02T00:00:00Z"
       },
       "edgar": {
        "global_pacing": "10s spacing, at most 6 per 60s",
        "issuers": [
         "AAPL",
         "MSFT",
         "NVDA"
        ],
        "listings": 66096,
        "listings_per_issuer_per_day": 144,
        "nvda_identity_lookup": 1,
        "poll_interval_s": 600,
        "total_request_ceiling": 66097
       },
       "fomc": {
        "backoff_steps": 5,
        "feed_cadence_s": 60,
        "feed_polls": 220320,
        "max_statements": 16,
        "recheck_offsets_s": [
         300,
         3600,
         86400,
         604800
        ],
        "statement_pages_nominal": 80,
        "statement_pages_with_maximal_retries": 480,
        "total_request_ceiling": 220800
       },
       "prices": {
        "coinbase": {
         "bars_per_product": 2976,
         "candles_per_request": 300,
         "products": 2,
         "requests": 20,
         "series": [
          "2027-03-30T00:00:00Z",
          "2027-08-01T00:00:00Z"
         ]
        },
        "equity_daily": {
         "instruments": 4,
         "requests": 4,
         "requests_per_instrument": 1,
         "series_dates": [
          "2027-03-01",
          "2027-07-31"
         ]
        }
       },
       "raw_volume_bytes_worst_case_without_deduplication": {
        "edgar": 12244284000,
        "fomc_feed": 18637309440,
        "observed_deduplication": {
         "edgar_trial": {
          "distinct_raws": 2,
          "responses": 8
         },
         "fomc_pilot": {
          "distinct_raws": 49,
          "responses": 128
         }
        },
        "rule": "the operator sets a store-size ceiling and checks free disk before authorizing; the worst case exceeds the disk of the shared server, so the authorization must carry the ceiling and a stop",
        "sizes_used": {
         "edgar_listing_bytes_max_observed": 185250,
         "fomc_raw_bytes_max_observed": 84592
        }
       }
      },
      "wall_seconds": 180
     },
     "calendar": {
      "calendar_spec_hash": "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314",
      "capture": [
       "2027-03-02T00:00:00Z",
       "2027-08-02T00:00:00Z"
      ],
      "decision_counts_upper_bounds": {
       "crypto_per_product": {
        "expected_to_meet_minimum": true,
        "first_decision": "2027-04-01T00:00:00Z",
        "last_decision": "2027-07-31T20:00:00Z",
        "minimum_paired_test_decisions": 240,
        "purged": 3,
        "test": 1461,
        "train": 1461
       },
       "equity_per_instrument": {
        "expected_to_meet_minimum": false,
        "first_decision": "2027-04-01T20:00:00Z",
        "last_decision": "2027-07-23T20:00:00Z",
        "minimum_paired_test_decisions": 100,
        "purged": 5,
        "test": 37,
        "train": 37
       },
       "note": "upper bounds: before price gaps and source-state exclusions, which are known only after capture"
      },
      "evaluation": [
       "2027-04-01T00:00:00Z",
       "2027-08-01T00:00:00Z"
      ],
      "possible": {
       "crypto": "hourly decisions, T = bar close, from the first feature-complete bar",
       "equity": "one decision per session at the calendar close (early closes honoured)"
      },
      "price_series": {
       "crypto_bar_opens": [
        "2027-03-30T00:00:00Z",
        "2027-08-01T00:00:00Z"
       ],
       "equity_sessions": [
        "2027-03-01",
        "2027-07-31"
       ],
       "rule": "series start after every protected interval, so no feature recursion reads one"
      },
      "split": {
       "purge": "train decisions whose label window ends after the split instant are dropped",
       "test": [
        "2027-06-01T00:00:00Z",
        "2027-08-01T00:00:00Z"
       ],
       "train": [
        "2027-04-01T00:00:00Z",
        "2027-06-01T00:00:00Z"
       ]
      }
     },
     "costs": {
      "cost_model": {
       "fee_rate": "0.0010",
       "fill_policy": "next-contiguous-bar-open-after-decision-v1",
       "slippage_rate": "0.0005"
      },
      "equity_costs": "no frozen equity execution or cost model exists: no equity economic metric is computed",
      "execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
      "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
     },
     "decision_criteria": {
      "exclusions_reported": [
       "protected interval or label/lookback/event window",
       "price warm-up",
       "event warm-up",
       "source state not RESOLVED, per source and state",
       "price gap or missing label",
       "unknown or unverified issuer mapping",
       "integrity errors",
       "null imputations (count)",
       "decisions kept, per product, split and variant"
      ],
      "exploratory": {
       "computed": "the same paired metric R, its bootstrap interval (blocks of 10 sessions) and the secondary prediction metrics, on the same decisions, splits and exclusions, reported as descriptive statistics",
       "effect_on_primary": "none: the equity arm cannot change, delay or condition the crypto verdict",
       "family": "EQUITY",
       "products": [
        "AAPL",
        "MSFT",
        "QQQ",
        "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
       ],
       "sample_reported": {
        "edgar_newly_observed_accessions_in_test": "counted and reported",
        "paired_test_decisions_per_instrument": "counted and reported (the V1 count expected 37, below the 100 a verdict would need)"
       },
       "status": "EXPLORATORY_NO_CLAIM",
       "verdict": "none: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE is never assigned to this arm, and no result of it is cited as evidence for or against events",
       "windows": "the equity sessions looked at here become exploratory; a later confirmatory equity claim needs its own preregistered revision on data not looked at, and the reserved equity holdout stays closed"
      },
      "primary": {
       "decision_rule": {
        "INCONCLUSIVE_INSUFFICIENT_SAMPLE": "any minimum sample not met; no verdict is drawn",
        "NOT_SUPPORTED": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
        "SUPPORTED": "R > 0, lower bound of the interval > 0, and R >= 0.005",
        "scope": "the CRYPTO family only; no pooling with any other product; no other metric can overturn it"
       },
       "families": {
        "CRYPTO": [
         "BTC-USD",
         "ETH-USD"
        ]
       },
       "label": {
        "crypto": "close[T+4 bars]/close[T]-1"
       },
       "metric": "R = 1 - MSE_B / MSE_A on test decisions, per product, equal-weighted mean over the family's products",
       "minimum_sample": {
        "blocks_per_product": 10,
        "crypto_fomc_statements_newly_observed_in_test": 2,
        "paired_test_decisions_per_product": {
         "crypto": 240
        }
       },
       "uncertainty": {
        "block_decisions": {
         "crypto": 24
        },
        "families_tested": 1,
        "family_alpha_one_sided": 0.025,
        "familywise_note": "one confirmatory family (CRYPTO) at one-sided 2.5%, the level V1 fixed for it, kept unchanged and not relaxed; the exploratory equity arm spends no alpha",
        "interval": "percentile 2.5%-97.5% (two-sided 95%): its lower bound is the one-sided 2.5% bound",
        "method": "circular block bootstrap of time-ordered paired decisions, same block starts for every product of a family and for both variants",
        "resamples": 10000,
        "seed": 20270801
       }
      },
      "secondary": {
       "annualization_periods": 8760,
       "baselines": [
        "ZERO",
        "TRAIN_MEAN"
       ],
       "economic_crypto_only": [
        "net_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "fees_paid"
       ],
       "prediction": [
        "mae",
        "rmse",
        "spearman_rank_ic",
        "directional_accuracy"
       ],
       "scope": "CRYPTO and the exploratory EQUITY arm; never part of any decision rule",
       "status": "reported with the same bootstrap; never part of the decision rule"
      },
      "stopping": "fixed periods; no optional stopping, no extension or re-run after a look"
     },
     "falsification": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
     "features": {
      "event_feature_spec_hash": "b54ae52b815dcc9a23bd728b2899f7e0e7c43cb8482f68b189ac5236e84f43fb",
      "events": {
       "AAPL": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h",
        "new_accessions_attested_7d",
        "new_accessions_attested_30d",
        "accession_revisions_attested_7d",
        "accession_revisions_attested_30d",
        "hours_since_last_new_accession",
        "new_accession_item_2_02_7d",
        "new_accession_item_5_02_7d",
        "new_accession_item_7_01_7d",
        "new_accession_item_8_01_7d"
       ],
       "BTC-USD": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h"
       ],
       "ETH-USD": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h"
       ],
       "MSFT": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h",
        "new_accessions_attested_7d",
        "new_accessions_attested_30d",
        "accession_revisions_attested_7d",
        "accession_revisions_attested_30d",
        "hours_since_last_new_accession",
        "new_accession_item_2_02_7d",
        "new_accession_item_5_02_7d",
        "new_accession_item_7_01_7d",
        "new_accession_item_8_01_7d"
       ],
       "NVDA": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h",
        "new_accessions_attested_7d",
        "new_accessions_attested_30d",
        "accession_revisions_attested_7d",
        "accession_revisions_attested_30d",
        "hours_since_last_new_accession",
        "new_accession_item_2_02_7d",
        "new_accession_item_5_02_7d",
        "new_accession_item_7_01_7d",
        "new_accession_item_8_01_7d"
       ],
       "QQQ": [
        "new_statements_attested_7d",
        "new_statements_attested_30d",
        "statement_revisions_attested_7d",
        "statement_revisions_attested_30d",
        "hours_since_last_new_statement",
        "new_statement_attested_within_24h"
       ]
      },
      "prices": {
       "crypto": [
        "return_1",
        "return_4",
        "return_12",
        "ema_spread_12_26",
        "rsi_14",
        "atr_pct_14"
       ],
       "equity": [
        "return1",
        "return5",
        "return20",
        "ema_spread10_20",
        "rsi14",
        "atr_pct14"
       ]
      },
      "transform": {
       "boolean": "0/1",
       "hours_since_last_new_*": "min(hours, 720); null (no new observation yet) -> 720",
       "null_item_flag": "0 (counted and reported)",
       "selection": "none: every listed event column enters variant B, none is dropped or added after results"
      }
     },
     "horizon_seconds": 14400,
     "hypothesis_id": "prices-vs-events-protocol-v2",
     "mechanism": "Newly observed FOMC information and revisions can change expected forward returns",
     "population": {
      "admissibility": [
       "inside the evaluation period and the product's price series",
       "no protected interval touched by lookback, label or event window",
       "price features and label exist (no gap)",
       "every required source RESOLVED, no INTEGRITY_ERROR",
       "price and event warm-ups complete"
      ],
      "new_evaluation": [
       "2027-04-01T00:00:00Z",
       "2027-08-01T00:00:00Z"
      ],
      "products": {
       "crypto": [
        "BTC-USD",
        "ETH-USD"
       ],
       "equity": [
        "AAPL",
        "MSFT",
        "QQQ",
        "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
       ],
       "required_sources": {
        "AAPL": [
         "fomc",
         "edgar"
        ],
        "BTC-USD": [
         "fomc"
        ],
        "ETH-USD": [
         "fomc"
        ],
        "MSFT": [
         "fomc",
         "edgar"
        ],
        "NVDA": [
         "fomc",
         "edgar"
        ],
        "QQQ": [
         "fomc"
        ]
       },
       "roles": {
        "crypto": "PRIMARY_CONFIRMATORY",
        "equity": "EXPLORATORY_NO_CLAIM"
       }
      },
      "studied_windows": "EXPLORATORY"
     },
     "protocol_hash": "3817ff509bdc35ec53aca85bc3c558e60b3db828efa920f12191475e9883875e",
     "schema": "research-hypothesis-v1",
     "scope": "PREREGISTERED_PROTOCOL",
     "sources": [
      "fomc",
      "edgar",
      "coinbase"
     ],
     "splits": {
      "purge": "train decisions whose label window ends after the split instant are dropped",
      "test": [
       "2027-06-01T00:00:00Z",
       "2027-08-01T00:00:00Z"
      ],
      "train": [
       "2027-04-01T00:00:00Z",
       "2027-06-01T00:00:00Z"
      ]
     },
     "statement": "Attested events improve crypto test MSE beyond prices on the same admissible decisions",
     "synthetic": false,
     "target": "forward_return",
     "transformations": {
      "embargo": "only the purge fixed by protocol V2; no additional embargo",
      "fit_on": "TRAIN_ONLY",
      "method": "refit on the forward train block for both variants; the split is new because the frozen walk-forward geometry (252 train sessions) does not fit a 4-month window; training needs its own authorization"
     },
     "version": "local-rule-proposals-v1"
    },
    "recorded_at": "2026-10-05T10:24:21.211606+00:00",
    "sequence": 1
   },
   {
    "chain_hash": "f794872f1fe695f5fca2bc198b9d99f511cb5fb23a454f4c57cbb25dfaff4d4c",
    "identity": "0192dcc19394fb4c029a796717ab76cbb98c0e9218eefedc848b732e0406f652",
    "payload": {
     "availability": {
      "exclusions_hash": "e40049861a13c41844c312eee61c42923e23f6dc668acf516b7c152c1aea7b2c",
      "protection_hash": "97b2cea6ab819d8cb75b70df9cc4bddd78b4777f1b5142a23825d72f2fb10a5d",
      "rule": "all price dependencies attested by decision; labels separate"
     },
     "baselines": [
      "ZERO",
      "TRAIN_MEAN"
     ],
     "budgets": {
      "max_rows": 600,
      "max_trials": 2,
      "wall_seconds": 120
     },
     "calendar": {
      "id": "coinbase-hourly-utc-v1"
     },
     "costs": {
      "backtest": {
       "allow_long": true,
       "allow_short": true,
       "borrow_rate": "0",
       "cost_model": "synthetic",
       "currency": "USD",
       "exchange_account_specific": false,
       "fee_rate": "0.0010",
       "fill_policy": "next-contiguous-bar-open-after-decision-v1",
       "final_liquidation_policy": "next-observable-open-after-last-fill-v1",
       "funding_rate": "0",
       "initial_equity": "100000",
       "instrument_model": "synthetic-linear-usd-notional-v1",
       "mark_policy": "next-observable-open-v1",
       "optimized": false,
       "protocol_version": "trading-lab.execution.v1",
       "schema_version": "trading-lab.execution.v1",
       "slippage_rate": "0.0005",
       "target_quantity_basis": "reference-market-open-v1"
      },
      "backtest_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
      "calendar": "coinbase-hourly-utc-v1",
      "paper": {
       "backtest_execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
       "cost_model": "synthetic",
       "currency": "USD",
       "differs_from_backtest": [
        "a live session never liquidates a terminal position",
        "a fill is recorded one bar after the price it is filled at becomes observable, so paper and backtest timelines are not directly comparable"
       ],
       "fee_rate": "0.0010",
       "fill_observation_policy": "recorded-when-the-fill-bar-closes-v1",
       "fill_price_policy": "next-contiguous-bar-open-after-decision-v1",
       "initial_equity": "100000",
       "mark_policy": "latest-observed-open-v1",
       "protocol_version": "trading-lab.paper-execution.v1",
       "schema_version": "trading-lab.paper-execution.v1",
       "slippage_rate": "0.0005",
       "terminal_liquidation": false
      },
      "paper_hash": "bb944167fbcc9f35d93b269c1422e2c658235a8c49524b165206570963c19bfa",
      "risk_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
      "signal_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
     },
     "decision_criteria": {
      "commercial_claim": false,
      "metric": "test_mae",
      "rule": "strictly below ZERO and TRAIN_MEAN"
     },
     "falsification": "test MAE fails to beat both ZERO and TRAIN_MEAN",
     "features": {
      "columns": [
       "return_1",
       "return_4",
       "return_12",
       "ema_spread_12_26",
       "rsi_14",
       "atr_pct_14"
      ],
      "definition_hash": "c1f7d1b0228b7f752738744b67ce8c0c5eb62adc599f5e71227f0ecb040b00a5"
     },
     "horizon_seconds": 14400,
     "hypothesis_id": "synthetic-hypothesis-d6acc46ff1dd2686cdc66549",
     "mechanism": "Persistent synthetic price dynamics can be represented by a regularized linear model",
     "population": {
      "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
      "end": "2026-06-06T01:00:00+00:00",
      "products": [
       "BTC-USD",
       "ETH-USD"
      ],
      "rows": 182,
      "start": "2026-06-01T00:00:00+00:00",
      "studied_window": true
     },
     "protocol_hash": null,
     "schema": "research-hypothesis-v1",
     "scope": "EXPLORATORY",
     "sources": [
      "synthetic_prices"
     ],
     "splits": {
      "embargo_seconds": 3600,
      "exclusions": [
       {
        "decision_at": "2026-06-03T19:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T19:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T20:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T20:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T21:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T21:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T22:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T22:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-03T23:00:00+00:00",
        "product": "BTC-USD",
        "reason": "TEMPORAL_EMBARGO"
       },
       {
        "decision_at": "2026-06-03T23:00:00+00:00",
        "product": "ETH-USD",
        "reason": "TEMPORAL_EMBARGO"
       },
       {
        "decision_at": "2026-06-04T18:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T18:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T19:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T19:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T20:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T20:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T21:00:00+00:00",
        "product": "BTC-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T21:00:00+00:00",
        "product": "ETH-USD",
        "reason": "PURGE_LABEL_REACHES_NEXT_BLOCK"
       },
       {
        "decision_at": "2026-06-04T22:00:00+00:00",
        "product": "BTC-USD",
        "reason": "TEMPORAL_EMBARGO"
       },
       {
        "decision_at": "2026-06-04T22:00:00+00:00",
        "product": "ETH-USD",
        "reason": "TEMPORAL_EMBARGO"
       }
      ],
      "method": "temporal-purge-embargo-v1",
      "purge_seconds": 14400,
      "refit_after_validation": false,
      "roles": {
       "test": {
        "first": "2026-06-04T23:00:00+00:00",
        "last": "2026-06-05T20:00:00+00:00",
        "population_hash": "172d1dbad8bfd2a33dd09ae7b9213fb2be0c31b676ae1ae212de31f5c7bc429c",
        "rows": 44
       },
       "train": {
        "first": "2026-06-02T02:00:00+00:00",
        "last": "2026-06-03T18:00:00+00:00",
        "population_hash": "58f19c0c6ce558737945033e03baa06523445a10269a05d22785c3802b21ac8e",
        "rows": 82
       },
       "validation": {
        "first": "2026-06-04T00:00:00+00:00",
        "last": "2026-06-04T17:00:00+00:00",
        "population_hash": "449b0f95401bd9a218f3b0028b5b8eb56d46f5145d154496225b47aaadae9508",
        "rows": 36
       }
      },
      "test_start": "2026-06-04T22:00:00+00:00",
      "train_label_available_before": "2026-06-03T23:00:00+00:00",
      "validation_label_available_before": "2026-06-04T22:00:00+00:00",
      "validation_start": "2026-06-03T23:00:00+00:00"
     },
     "statement": "Lagged price features reduce synthetic return MAE relative to fixed baselines",
     "synthetic": true,
     "target": "forward_return",
     "transformations": {
      "fit_on": "TRAIN_ONLY",
      "method": "training only; no validation refit",
      "refit_after_validation": false
     },
     "version": "local-rule-proposals-v1"
    },
    "recorded_at": "2026-10-05T10:24:35.945372+00:00",
    "sequence": 2
   }
  ],
  "schema": "research-page-v1"
 },
 "hypothesisDetail": {
  "chain_hash": "5dbf5ad00a9ef7d6405a5443d6b9af4917315156423d4415d8ae91b8123c40e9",
  "criteria_hash": "245c8ccfa6d851b98d296c1183226a8274ed388f00af5ed54f1482653a16d99c",
  "identity": "5787a940a7799947b846fca60c7818dba3cbb4cf61f45eb5d201d34048f645b2",
  "payload": {
   "availability": {
    "pairing": "A and B use exactly the same decisions, labels, costs and splits; a decision where a required source is not RESOLVED is excluded from both; a family's test set is the decisions common to all its products",
    "protection": {
     "admissible_ranges_price_and_label": {
      "crypto": {
       "decision_clock": "bar close = bar opening + 1h",
       "event_window_clean_from": "2026-12-31T00:00:00Z",
       "label_horizon_bars": 4,
       "price_warmup_bars": 25,
       "runs": [
        {
         "decisions": 9475,
         "first_bar_open": "2025-08-02T01:00:00Z",
         "first_decision_at": "2025-08-02T02:00:00Z",
         "last_bar_open": "2026-08-31T19:00:00Z",
         "last_decision_at": "2026-08-31T20:00:00Z",
         "run_bars": 9504
        },
        {
         "decisions": 5803,
         "first_bar_open": "2026-12-02T01:00:00Z",
         "first_decision_at": "2026-12-02T02:00:00Z",
         "last_bar_open": "2027-07-31T19:00:00Z",
         "last_decision_at": "2027-07-31T20:00:00Z",
         "run_bars": 5832
        }
       ]
      },
      "equity": {
       "decision_clock": "session close (calendar close_at, early closes honoured)",
       "event_window_clean_from": "2027-03-31T00:00:00Z",
       "label_horizon_sessions": 5,
       "price_warmup_sessions": 20,
       "runs": [
        {
         "decisions": 560,
         "first_bar_open": "2024-08-29T13:30:00Z",
         "first_decision_at": "2024-08-29T20:00:00Z",
         "last_bar_open": "2026-11-20T14:30:00Z",
         "last_decision_at": "2026-11-20T21:00:00Z",
         "run_bars": 585
        },
        {
         "decisions": 81,
         "first_bar_open": "2027-03-30T13:30:00Z",
         "first_decision_at": "2027-03-30T20:00:00Z",
         "last_bar_open": "2027-07-23T13:30:00Z",
         "last_decision_at": "2027-07-23T20:00:00Z",
         "run_bars": 106
        }
       ]
      }
     },
     "event_window_days": 30,
     "label_horizon": {
      "crypto_bars": 4,
      "equity_sessions": 5
     },
     "other_reserved_intervals": "none found: the walk-forward and paper-replay periods are exploratory/consumed, not reserved; closed pilots are archives, not intervals",
     "price_warmup": {
      "crypto_bars": 25,
      "equity_sessions": 20
     },
     "rule": "a decision is admissible only if no bar of its price features, no bar of its label window and no instant of its 30-day event window (T-30d, T] lies in a protected interval of its product",
     "table": {
      "AAPL": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "BTC-USD": [
       {
        "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
        "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
        "end_exclusive": "2026-12-01T00:00:00Z",
        "holdout_id": "coinbase_confirmatory_2026q4",
        "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
        "last_protected_bar_open": "2026-11-30T23:00:00Z",
        "observed": false,
        "products": [
         "BTC-USD",
         "ETH-USD"
        ],
        "single_use": true,
        "start": "2026-09-01T00:00:00Z",
        "timeframe": "1h",
        "unit": "bar opening instants"
       }
      ],
      "ETH-USD": [
       {
        "bounds": "start inclusive; last protected bar OPENING 2026-11-30T23:00:00Z inclusive; the interval ends (exclusive) one bar later, when no protected bar can still be forming",
        "contract": "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
        "end_exclusive": "2026-12-01T00:00:00Z",
        "holdout_id": "coinbase_confirmatory_2026q4",
        "identity_hash": "bf95ee8577bbb3444fa14d964ff1db951910b693ff58ebdce8ecbda2eb24af85",
        "last_protected_bar_open": "2026-11-30T23:00:00Z",
        "observed": false,
        "products": [
         "BTC-USD",
         "ETH-USD"
        ],
        "single_use": true,
        "start": "2026-09-01T00:00:00Z",
        "timeframe": "1h",
        "unit": "bar opening instants"
       }
      ],
      "MSFT": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "NVDA": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ],
      "QQQ": [
       {
        "bounds": "dates inclusive: 2026-12-01 through 2027-02-28 (the contract compares session dates: covers() reads the first ten characters)",
        "calendar_id": "US_EQUITY_REGULAR",
        "contract": "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        "declared_end": "2027-02-28T23:59:59Z",
        "end_exclusive": "2027-03-01T00:00:00Z",
        "holdout_id": "equity_confirmatory_2027q1",
        "identity_hash": "b6ae7338670c45f5de7976d5299659af3d51e89f99529ebb209561add0c244b9",
        "observed": false,
        "products": [
         "AAPL",
         "MSFT",
         "NVDA",
         "QQQ"
        ],
        "research_spec_hash": "be4270ce5e8a1d472ffcfb12745094402d1bcc748e0d1a66d2f36707a546b177",
        "single_use": true,
        "start": "2026-12-01T00:00:00Z",
        "timeframe": "1d",
        "unit": "session dates"
       }
      ]
     }
    },
    "warmup": {
     "capture_start_plus_30_days": "2027-04-01T00:00:00Z",
     "event": "for every required source: state RESOLVED at T, T >= attested coverage start + 30 days and T >= max(first valid read availability, last inventory availability at T) + 30 days, per issuer for EDGAR",
     "price": "the feature-complete index of the real indicators (see protection.price_warmup), inside the series"
    }
   },
   "baselines": [
    "PRICES_ONLY",
    "ZERO",
    "TRAIN_MEAN"
   ],
   "budgets": {
    "execution_enabled": false,
    "max_rows": 10000,
    "max_trials": 1,
    "protocol_budgets": {
     "capture_window": {
      "days": 153,
      "end_exclusive": "2027-08-02T00:00:00Z",
      "rule": "FOMC and EDGAR run for the whole window; stop when the last supplied decision is RESOLVED",
      "start": "2027-03-02T00:00:00Z"
     },
     "edgar": {
      "global_pacing": "10s spacing, at most 6 per 60s",
      "issuers": [
       "AAPL",
       "MSFT",
       "NVDA"
      ],
      "listings": 66096,
      "listings_per_issuer_per_day": 144,
      "nvda_identity_lookup": 1,
      "poll_interval_s": 600,
      "total_request_ceiling": 66097
     },
     "fomc": {
      "backoff_steps": 5,
      "feed_cadence_s": 60,
      "feed_polls": 220320,
      "max_statements": 16,
      "recheck_offsets_s": [
       300,
       3600,
       86400,
       604800
      ],
      "statement_pages_nominal": 80,
      "statement_pages_with_maximal_retries": 480,
      "total_request_ceiling": 220800
     },
     "prices": {
      "coinbase": {
       "bars_per_product": 2976,
       "candles_per_request": 300,
       "products": 2,
       "requests": 20,
       "series": [
        "2027-03-30T00:00:00Z",
        "2027-08-01T00:00:00Z"
       ]
      },
      "equity_daily": {
       "instruments": 4,
       "requests": 4,
       "requests_per_instrument": 1,
       "series_dates": [
        "2027-03-01",
        "2027-07-31"
       ]
      }
     },
     "raw_volume_bytes_worst_case_without_deduplication": {
      "edgar": 12244284000,
      "fomc_feed": 18637309440,
      "observed_deduplication": {
       "edgar_trial": {
        "distinct_raws": 2,
        "responses": 8
       },
       "fomc_pilot": {
        "distinct_raws": 49,
        "responses": 128
       }
      },
      "rule": "the operator sets a store-size ceiling and checks free disk before authorizing; the worst case exceeds the disk of the shared server, so the authorization must carry the ceiling and a stop",
      "sizes_used": {
       "edgar_listing_bytes_max_observed": 185250,
       "fomc_raw_bytes_max_observed": 84592
      }
     }
    },
    "wall_seconds": 180
   },
   "calendar": {
    "calendar_spec_hash": "1ef910eb3d4f5096ab2888ea6213df1f688870dfea02ba977bfc7faea9db6314",
    "capture": [
     "2027-03-02T00:00:00Z",
     "2027-08-02T00:00:00Z"
    ],
    "decision_counts_upper_bounds": {
     "crypto_per_product": {
      "expected_to_meet_minimum": true,
      "first_decision": "2027-04-01T00:00:00Z",
      "last_decision": "2027-07-31T20:00:00Z",
      "minimum_paired_test_decisions": 240,
      "purged": 3,
      "test": 1461,
      "train": 1461
     },
     "equity_per_instrument": {
      "expected_to_meet_minimum": false,
      "first_decision": "2027-04-01T20:00:00Z",
      "last_decision": "2027-07-23T20:00:00Z",
      "minimum_paired_test_decisions": 100,
      "purged": 5,
      "test": 37,
      "train": 37
     },
     "note": "upper bounds: before price gaps and source-state exclusions, which are known only after capture"
    },
    "evaluation": [
     "2027-04-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "possible": {
     "crypto": "hourly decisions, T = bar close, from the first feature-complete bar",
     "equity": "one decision per session at the calendar close (early closes honoured)"
    },
    "price_series": {
     "crypto_bar_opens": [
      "2027-03-30T00:00:00Z",
      "2027-08-01T00:00:00Z"
     ],
     "equity_sessions": [
      "2027-03-01",
      "2027-07-31"
     ],
     "rule": "series start after every protected interval, so no feature recursion reads one"
    },
    "split": {
     "purge": "train decisions whose label window ends after the split instant are dropped",
     "test": [
      "2027-06-01T00:00:00Z",
      "2027-08-01T00:00:00Z"
     ],
     "train": [
      "2027-04-01T00:00:00Z",
      "2027-06-01T00:00:00Z"
     ]
    }
   },
   "costs": {
    "cost_model": {
     "fee_rate": "0.0010",
     "fill_policy": "next-contiguous-bar-open-after-decision-v1",
     "slippage_rate": "0.0005"
    },
    "equity_costs": "no frozen equity execution or cost model exists: no equity economic metric is computed",
    "execution_spec_hash": "99295b9ab5e9a4e16f4cdd05855071a792e115b35b22d3f952f052d1f97e93eb",
    "risk_spec_hash": "f3be4fc63f684cd27c496bdfa1ce22b96b465de5a41bc169d9921f2c8d4c8dad",
    "signal_spec_hash": "7f8b57f892a9e05ce5d2bc60c9bb114ea7dbc029ece561b7f62e9ad2d58b2939"
   },
   "decision_criteria": {
    "exclusions_reported": [
     "protected interval or label/lookback/event window",
     "price warm-up",
     "event warm-up",
     "source state not RESOLVED, per source and state",
     "price gap or missing label",
     "unknown or unverified issuer mapping",
     "integrity errors",
     "null imputations (count)",
     "decisions kept, per product, split and variant"
    ],
    "exploratory": {
     "computed": "the same paired metric R, its bootstrap interval (blocks of 10 sessions) and the secondary prediction metrics, on the same decisions, splits and exclusions, reported as descriptive statistics",
     "effect_on_primary": "none: the equity arm cannot change, delay or condition the crypto verdict",
     "family": "EQUITY",
     "products": [
      "AAPL",
      "MSFT",
      "QQQ",
      "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
     ],
     "sample_reported": {
      "edgar_newly_observed_accessions_in_test": "counted and reported",
      "paired_test_decisions_per_instrument": "counted and reported (the V1 count expected 37, below the 100 a verdict would need)"
     },
     "status": "EXPLORATORY_NO_CLAIM",
     "verdict": "none: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE is never assigned to this arm, and no result of it is cited as evidence for or against events",
     "windows": "the equity sessions looked at here become exploratory; a later confirmatory equity claim needs its own preregistered revision on data not looked at, and the reserved equity holdout stays closed"
    },
    "primary": {
     "decision_rule": {
      "INCONCLUSIVE_INSUFFICIENT_SAMPLE": "any minimum sample not met; no verdict is drawn",
      "NOT_SUPPORTED": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
      "SUPPORTED": "R > 0, lower bound of the interval > 0, and R >= 0.005",
      "scope": "the CRYPTO family only; no pooling with any other product; no other metric can overturn it"
     },
     "families": {
      "CRYPTO": [
       "BTC-USD",
       "ETH-USD"
      ]
     },
     "label": {
      "crypto": "close[T+4 bars]/close[T]-1"
     },
     "metric": "R = 1 - MSE_B / MSE_A on test decisions, per product, equal-weighted mean over the family's products",
     "minimum_sample": {
      "blocks_per_product": 10,
      "crypto_fomc_statements_newly_observed_in_test": 2,
      "paired_test_decisions_per_product": {
       "crypto": 240
      }
     },
     "uncertainty": {
      "block_decisions": {
       "crypto": 24
      },
      "families_tested": 1,
      "family_alpha_one_sided": 0.025,
      "familywise_note": "one confirmatory family (CRYPTO) at one-sided 2.5%, the level V1 fixed for it, kept unchanged and not relaxed; the exploratory equity arm spends no alpha",
      "interval": "percentile 2.5%-97.5% (two-sided 95%): its lower bound is the one-sided 2.5% bound",
      "method": "circular block bootstrap of time-ordered paired decisions, same block starts for every product of a family and for both variants",
      "resamples": 10000,
      "seed": 20270801
     }
    },
    "secondary": {
     "annualization_periods": 8760,
     "baselines": [
      "ZERO",
      "TRAIN_MEAN"
     ],
     "economic_crypto_only": [
      "net_return",
      "sharpe",
      "max_drawdown",
      "turnover",
      "fees_paid"
     ],
     "prediction": [
      "mae",
      "rmse",
      "spearman_rank_ic",
      "directional_accuracy"
     ],
     "scope": "CRYPTO and the exploratory EQUITY arm; never part of any decision rule",
     "status": "reported with the same bootstrap; never part of the decision rule"
    },
    "stopping": "fixed periods; no optional stopping, no extension or re-run after a look"
   },
   "falsification": "sample sufficient and the SUPPORTED conditions not all met (no evidence of improvement, not evidence of no effect)",
   "features": {
    "event_feature_spec_hash": "b54ae52b815dcc9a23bd728b2899f7e0e7c43cb8482f68b189ac5236e84f43fb",
    "events": {
     "AAPL": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "BTC-USD": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ],
     "ETH-USD": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ],
     "MSFT": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "NVDA": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h",
      "new_accessions_attested_7d",
      "new_accessions_attested_30d",
      "accession_revisions_attested_7d",
      "accession_revisions_attested_30d",
      "hours_since_last_new_accession",
      "new_accession_item_2_02_7d",
      "new_accession_item_5_02_7d",
      "new_accession_item_7_01_7d",
      "new_accession_item_8_01_7d"
     ],
     "QQQ": [
      "new_statements_attested_7d",
      "new_statements_attested_30d",
      "statement_revisions_attested_7d",
      "statement_revisions_attested_30d",
      "hours_since_last_new_statement",
      "new_statement_attested_within_24h"
     ]
    },
    "prices": {
     "crypto": [
      "return_1",
      "return_4",
      "return_12",
      "ema_spread_12_26",
      "rsi_14",
      "atr_pct_14"
     ],
     "equity": [
      "return1",
      "return5",
      "return20",
      "ema_spread10_20",
      "rsi14",
      "atr_pct14"
     ]
    },
    "transform": {
     "boolean": "0/1",
     "hours_since_last_new_*": "min(hours, 720); null (no new observation yet) -> 720",
     "null_item_flag": "0 (counted and reported)",
     "selection": "none: every listed event column enters variant B, none is dropped or added after results"
    }
   },
   "horizon_seconds": 14400,
   "hypothesis_id": "prices-vs-events-protocol-v2",
   "mechanism": "Newly observed FOMC information and revisions can change expected forward returns",
   "population": {
    "admissibility": [
     "inside the evaluation period and the product's price series",
     "no protected interval touched by lookback, label or event window",
     "price features and label exist (no gap)",
     "every required source RESOLVED, no INTEGRITY_ERROR",
     "price and event warm-ups complete"
    ],
    "new_evaluation": [
     "2027-04-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "products": {
     "crypto": [
      "BTC-USD",
      "ETH-USD"
     ],
     "equity": [
      "AAPL",
      "MSFT",
      "QQQ",
      "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"
     ],
     "required_sources": {
      "AAPL": [
       "fomc",
       "edgar"
      ],
      "BTC-USD": [
       "fomc"
      ],
      "ETH-USD": [
       "fomc"
      ],
      "MSFT": [
       "fomc",
       "edgar"
      ],
      "NVDA": [
       "fomc",
       "edgar"
      ],
      "QQQ": [
       "fomc"
      ]
     },
     "roles": {
      "crypto": "PRIMARY_CONFIRMATORY",
      "equity": "EXPLORATORY_NO_CLAIM"
     }
    },
    "studied_windows": "EXPLORATORY"
   },
   "protocol_hash": "3817ff509bdc35ec53aca85bc3c558e60b3db828efa920f12191475e9883875e",
   "schema": "research-hypothesis-v1",
   "scope": "PREREGISTERED_PROTOCOL",
   "sources": [
    "fomc",
    "edgar",
    "coinbase"
   ],
   "splits": {
    "purge": "train decisions whose label window ends after the split instant are dropped",
    "test": [
     "2027-06-01T00:00:00Z",
     "2027-08-01T00:00:00Z"
    ],
    "train": [
     "2027-04-01T00:00:00Z",
     "2027-06-01T00:00:00Z"
    ]
   },
   "statement": "Attested events improve crypto test MSE beyond prices on the same admissible decisions",
   "synthetic": false,
   "target": "forward_return",
   "transformations": {
    "embargo": "only the purge fixed by protocol V2; no additional embargo",
    "fit_on": "TRAIN_ONLY",
    "method": "refit on the forward train block for both variants; the split is new because the frozen walk-forward geometry (252 train sessions) does not fit a 4-month window; training needs its own authorization"
   },
   "version": "local-rule-proposals-v1"
  },
  "recorded_at": "2026-10-05T10:24:21.211606+00:00",
  "sequence": 1,
  "trial_history": []
 },
 "labJobs": {
  "jobs": [
   {
    "cancel_requested": false,
    "created_at": 1791195886.7524383,
    "error_code": null,
    "id": "264b883445f840d3a97f9ee74b5e03d1",
    "kind": "experiment",
    "limits": {
     "cpu_seconds": 60,
     "memory_mb": 1024,
     "output_mb": 32,
     "wall_seconds": 120
    },
    "logs": [
     {
      "at": 1791195886.7524383,
      "code": "QUEUED",
      "progress": 0.0,
      "sequence": 14
     },
     {
      "at": 1791195886.952677,
      "code": "STARTED",
      "progress": 0.01,
      "sequence": 15
     },
     {
      "at": 1791195890.3647692,
      "code": "TRAINING",
      "progress": 0.1,
      "sequence": 16
     },
     {
      "at": 1791195890.461289,
      "code": "VALIDATING",
      "progress": 0.55,
      "sequence": 17
     },
     {
      "at": 1791195890.493895,
      "code": "BACKTEST",
      "progress": 0.65,
      "sequence": 18
     },
     {
      "at": 1791195890.5196998,
      "code": "SHADOW",
      "progress": 0.75,
      "sequence": 19
     },
     {
      "at": 1791195893.6770315,
      "code": "RESULT_READY",
      "progress": 0.99,
      "sequence": 20
     },
     {
      "at": 1791195893.6993759,
      "code": "COMPLETE",
      "progress": 1.0,
      "sequence": 21
     }
    ],
    "progress": 1.0,
    "result_hash": "1edb219e98cfe5988a82d8c42e375f88997e5bd4c475516c71acbc4a0cd2e9c4",
    "state": "COMPLETE",
    "updated_at": 1791195893.6993132,
    "worker_pid": 3346964
   },
   {
    "cancel_requested": false,
    "created_at": 1791195875.9854887,
    "error_code": null,
    "id": "e1565fdbaa4544078d6cdf1ca913c7cf",
    "kind": "experiment",
    "limits": {
     "cpu_seconds": 60,
     "memory_mb": 1024,
     "output_mb": 32,
     "wall_seconds": 120
    },
    "logs": [
     {
      "at": 1791195875.9854887,
      "code": "QUEUED",
      "progress": 0.0,
      "sequence": 6
     },
     {
      "at": 1791195876.2000098,
      "code": "STARTED",
      "progress": 0.01,
      "sequence": 7
     },
     {
      "at": 1791195879.4266343,
      "code": "TRAINING",
      "progress": 0.1,
      "sequence": 8
     },
     {
      "at": 1791195879.5881817,
      "code": "VALIDATING",
      "progress": 0.55,
      "sequence": 9
     },
     {
      "at": 1791195879.6185882,
      "code": "BACKTEST",
      "progress": 0.65,
      "sequence": 10
     },
     {
      "at": 1791195879.6448681,
      "code": "SHADOW",
      "progress": 0.75,
      "sequence": 11
     },
     {
      "at": 1791195882.7793207,
      "code": "RESULT_READY",
      "progress": 0.99,
      "sequence": 12
     },
     {
      "at": 1791195882.8128614,
      "code": "COMPLETE",
      "progress": 1.0,
      "sequence": 13
     }
    ],
    "progress": 1.0,
    "result_hash": "d65d0aa1ac2831bbfae1335ac96f44753b9f0111d08650ba8ffbcff8e82f4d08",
    "state": "COMPLETE",
    "updated_at": 1791195882.81277,
    "worker_pid": 3346959
   },
   {
    "cancel_requested": false,
    "created_at": 1791195861.2609217,
    "error_code": null,
    "id": "dae5cea2fe9d4e0bb2e420c4fa155bbb",
    "kind": "dataset",
    "limits": {
     "cpu_seconds": 60,
     "memory_mb": 1024,
     "output_mb": 32,
     "wall_seconds": 120
    },
    "logs": [
     {
      "at": 1791195861.2609217,
      "code": "QUEUED",
      "progress": 0.0,
      "sequence": 1
     },
     {
      "at": 1791195861.487843,
      "code": "STARTED",
      "progress": 0.01,
      "sequence": 2
     },
     {
      "at": 1791195874.9991758,
      "code": "DATASET_BUILT",
      "progress": 0.9,
      "sequence": 3
     },
     {
      "at": 1791195875.0931334,
      "code": "RESULT_READY",
      "progress": 0.99,
      "sequence": 4
     },
     {
      "at": 1791195875.106618,
      "code": "COMPLETE",
      "progress": 1.0,
      "sequence": 5
     }
    ],
    "progress": 1.0,
    "result_hash": "b6fdf8023041a4709017cce41e12d61de51f8ea34993532faa1c83a4f3babc3c",
    "state": "COMPLETE",
    "updated_at": 1791195875.1065295,
    "worker_pid": 3346919
   }
  ],
  "synthetic_only": true,
  "worker_limit": 1
 },
 "labModels": {
  "external_registration": "operator Python entry points only",
  "models": [
   {
    "contract": {
     "capabilities": [
      "train",
      "predict",
      "serialize",
      "infer"
     ],
     "horizons_seconds": [
      14400
     ],
     "implementation_version": "model-lab-adapters-v1",
     "inputs": {
      "features": [
       "return_1",
       "return_4",
       "return_12",
       "ema_spread_12_26",
       "rsi_14",
       "atr_pct_14"
      ],
      "products": [
       "BTC-USD",
       "ETH-USD"
      ],
      "target": "forward_return"
     },
     "limits": {
      "calibrated": false,
      "max_rows": 1200,
      "method": "four times the last one-hour return; no fitted transform",
      "reference_model": false,
      "remote_calls": false,
      "training": "synthetic validation only"
     },
     "model_id": "local-momentum-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "decimal simple forward return",
      "scenarios": null,
      "target_price": null
     },
     "schema": "model-contract-v1",
     "synthetic": true,
     "version": "1"
    },
    "fingerprint": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
    "registration": "external-local-entry-point"
   },
   {
    "contract": {
     "capabilities": [
      "predict",
      "serialize",
      "infer"
     ],
     "horizons_seconds": [
      14400
     ],
     "implementation_version": "model-lab-adapters-v1",
     "inputs": {
      "features": [
       "return_1",
       "return_4",
       "return_12",
       "ema_spread_12_26",
       "rsi_14",
       "atr_pct_14"
      ],
      "products": [
       "BTC-USD",
       "ETH-USD"
      ],
      "target": "forward_return"
     },
     "limits": {
      "calibrated": false,
      "max_rows": 1200,
      "paper_model_spec_hash": "830f52271af4eba887086d6b00885cc34f5d13562dae24cad45986a86466e033",
      "remote_calls": false,
      "research_evidence": false,
      "shadow_only": true,
      "training": "frozen; unavailable"
     },
     "model_id": "paper-ridge-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "decimal simple forward return",
      "scenarios": null,
      "target_price": null
     },
     "schema": "model-contract-v1",
     "synthetic": false,
     "version": "1"
    },
    "fingerprint": "7b78e00c13b6817c1dfd51fe64f766b09dae65c52afbab7bc971f5143f95c40c",
    "registration": "internal"
   },
   {
    "contract": {
     "capabilities": [
      "predict",
      "serialize",
      "infer"
     ],
     "horizons_seconds": [
      14400
     ],
     "implementation_version": "model-lab-adapters-v1",
     "inputs": {
      "features": [
       "return_1",
       "return_4",
       "return_12",
       "ema_spread_12_26",
       "rsi_14",
       "atr_pct_14"
      ],
      "products": [
       "BTC-USD",
       "ETH-USD"
      ],
      "target": "forward_return"
     },
     "limits": {
      "calibrated": false,
      "max_rows": 1200,
      "paper_model_spec_hash": "76fbec5ab1fcd809b04cba029c86e5b1c5be1dd61ecfe7b33751365e62d1d038",
      "remote_calls": false,
      "research_evidence": false,
      "shadow_only": true,
      "training": "frozen; unavailable"
     },
     "model_id": "paper-ridge-v2",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "decimal simple forward return",
      "scenarios": null,
      "target_price": null
     },
     "schema": "model-contract-v1",
     "synthetic": false,
     "version": "2"
    },
    "fingerprint": "799c5f2d4b1f43ee078b2bb9d6dcf806c363af14eaee7b1ddb51661cfb4966d8",
    "registration": "internal"
   },
   {
    "contract": {
     "capabilities": [
      "train",
      "predict",
      "serialize",
      "infer"
     ],
     "horizons_seconds": [
      14400
     ],
     "implementation_version": "model-lab-adapters-v1",
     "inputs": {
      "features": [
       "return_1",
       "return_4",
       "return_12",
       "ema_spread_12_26",
       "rsi_14",
       "atr_pct_14"
      ],
      "products": [
       "BTC-USD",
       "ETH-USD"
      ],
      "target": "forward_return"
     },
     "limits": {
      "calibrated": false,
      "max_rows": 1200,
      "reference_model": false,
      "remote_calls": false,
      "training": "synthetic only"
     },
     "model_id": "synthetic-ridge-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "decimal simple forward return",
      "scenarios": null,
      "target_price": null
     },
     "schema": "model-contract-v1",
     "synthetic": true,
     "version": "1"
    },
    "fingerprint": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
    "registration": "internal"
   }
  ]
 },
 "ledgerPage": {
  "as_of": "2026-10-05T10:30:50.171011+00:00",
  "page": {
   "has_more": true,
   "next_cursor": "eyJhZnRlciI6IjEzIiwiYXBpIjoidHJhZGluZy1sYWIuYXBwLWFwaS52MSIsImVuZHBvaW50IjoicHJlZGljdGlvbiIsInByb2R1Y3QiOiJyZXNlYXJjaCIsInEiOiJjNTFjZDk3ZWZmZjlhN2IwZDRlOTFmNGE2OWZiZGMwNiIsInYiOiJ0cmFkaW5nLWxhYi5hcHAtYXBpLmN1cnNvci52MSJ9",
   "returned": 3
  },
  "records": [
   {
    "chain_hash": "c6b313e773acea036870f8f05b485a7911842a5624dd87eee659261472867cf4",
    "identity": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
    "payload": {
     "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
     "costs": null,
     "decision_at": "2026-06-04T00:00:00+00:00",
     "errors": [],
     "event_ids": [],
     "execution": null,
     "features_hash": "e42f6ffc5c2814d97101fc1122b84217ef19c98db9d8f6296f00e9ca443880b3",
     "horizon_seconds": 14400,
     "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
     "model_id": "synthetic-ridge-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "0.003793627931969208138164078012653715",
      "scenarios": null,
      "target_price": null
     },
     "prediction_id": "20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
     "product": "BTC-USD",
     "proposed_position": null,
     "risk": null,
     "schema": "prediction-record-v1",
     "signal": null,
     "snapshot_hash": "6952ed0245b6b391bc48e4b76bcd77690ff600afed8281cb06d22e4d23ea8f77",
     "synthetic": true,
     "uncertainty": null
    },
    "recorded_at": "2026-10-05T10:24:43.202501+00:00",
    "sequence": 7,
    "view": {
     "detail_path": "/api/v1/observability/predictions/595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
     "execution_state": "PENDING",
     "input_quality": {
      "gaps": 0,
      "scope": "selected price inputs; unselected event sources retain unknown states",
      "source_states": {
       "edgar": "NOT_CONFIGURED",
       "fomc": "NOT_CONFIGURED"
      },
      "state": "VALID"
     },
     "label_state": "AVAILABLE",
     "label_versions": 1,
     "latest_decision": null,
     "latest_execution": null,
     "latest_label": {
      "available_at": "2026-06-04T04:00:00+00:00",
      "horizon_seconds": 14400,
      "identity": "571b2cb2fa5e0d6bb173be23bedabed22a7ffc9f3ed0b797224373188588c1be",
      "label_id": "lab-label-20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
      "prediction_hash": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
      "prediction_id": "20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
      "product": "BTC-USD",
      "provenance": {
       "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
       "method": "dataset-label-after-horizon-v1",
       "synthetic": true
      },
      "realized_at": "2026-06-04T04:00:00+00:00",
      "recorded_at": "2026-10-05T10:24:43.202501+00:00",
      "schema": "label-record-v1",
      "target": "forward_return",
      "value": "0.011006830131197484153112869146479",
      "version": "1"
     },
     "prediction_hash": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce"
    }
   },
   {
    "chain_hash": "0b10ac3c85a437d64245419766d77239d5b69a6c6c2d0288235be4f67d2bea03",
    "identity": "39630d3b538372df4809c2eed25cfcc14f6d983ba22d438a68eb3cdae8351c53",
    "payload": {
     "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
     "costs": null,
     "decision_at": "2026-06-04T01:00:00+00:00",
     "errors": [],
     "event_ids": [],
     "execution": null,
     "features_hash": "7b86ef7ded35749fd3f0694920a5bdcb915db46abf5d9f778a882943d79d5850",
     "horizon_seconds": 14400,
     "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
     "model_id": "synthetic-ridge-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "-0.01264906036559238623965951541624728",
      "scenarios": null,
      "target_price": null
     },
     "prediction_id": "43886a08213f783943dfdb2ee309b94fe6feb09995b6e79a8d6d1b0a77dd6837",
     "product": "BTC-USD",
     "proposed_position": null,
     "risk": null,
     "schema": "prediction-record-v1",
     "signal": null,
     "snapshot_hash": "93a91547b6217c9c832b07ef8abe126ea194aa34a29ad634fd069004afdd0656",
     "synthetic": true,
     "uncertainty": null
    },
    "recorded_at": "2026-10-05T10:24:43.202501+00:00",
    "sequence": 10,
    "view": {
     "detail_path": "/api/v1/observability/predictions/39630d3b538372df4809c2eed25cfcc14f6d983ba22d438a68eb3cdae8351c53",
     "execution_state": "PENDING",
     "input_quality": {
      "gaps": 0,
      "scope": "selected price inputs; unselected event sources retain unknown states",
      "source_states": {
       "edgar": "NOT_CONFIGURED",
       "fomc": "NOT_CONFIGURED"
      },
      "state": "VALID"
     },
     "label_state": "AVAILABLE",
     "label_versions": 1,
     "latest_decision": null,
     "latest_execution": null,
     "latest_label": {
      "available_at": "2026-06-04T05:00:00+00:00",
      "horizon_seconds": 14400,
      "identity": "463b5686320f89ede19faaaafeec6124e00cb7b43bd114310dbf14877745b2f0",
      "label_id": "lab-label-43886a08213f783943dfdb2ee309b94fe6feb09995b6e79a8d6d1b0a77dd6837",
      "prediction_hash": "39630d3b538372df4809c2eed25cfcc14f6d983ba22d438a68eb3cdae8351c53",
      "prediction_id": "43886a08213f783943dfdb2ee309b94fe6feb09995b6e79a8d6d1b0a77dd6837",
      "product": "BTC-USD",
      "provenance": {
       "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
       "method": "dataset-label-after-horizon-v1",
       "synthetic": true
      },
      "realized_at": "2026-06-04T05:00:00+00:00",
      "recorded_at": "2026-10-05T10:24:43.202501+00:00",
      "schema": "label-record-v1",
      "target": "forward_return",
      "value": "-0.0173219151420786298170494355780459",
      "version": "1"
     },
     "prediction_hash": "39630d3b538372df4809c2eed25cfcc14f6d983ba22d438a68eb3cdae8351c53"
    }
   },
   {
    "chain_hash": "cfeacc74cabd889e993b2452e99b9029c3e547c5a59a8aca1aba7147f4882e1c",
    "identity": "074d4f1795274c39c31d07d8af9c087d6d314ae0eab0c7d0e6234c6b802abee7",
    "payload": {
     "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
     "costs": null,
     "decision_at": "2026-06-04T02:00:00+00:00",
     "errors": [],
     "event_ids": [],
     "execution": null,
     "features_hash": "c1dddb54f8defebbc05089bf3b5fdf8145fc4ba6946706fe286785cee2cdef5c",
     "horizon_seconds": 14400,
     "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
     "model_id": "synthetic-ridge-v1",
     "outputs": {
      "class": null,
      "probabilities": null,
      "quantiles": null,
      "return": "0.01609328869937752305012226121888087",
      "scenarios": null,
      "target_price": null
     },
     "prediction_id": "741983db5d91e382eb3b0bc086494960b024e2dc88323ee5a09ad30a95ee945f",
     "product": "BTC-USD",
     "proposed_position": null,
     "risk": null,
     "schema": "prediction-record-v1",
     "signal": null,
     "snapshot_hash": "b2c66923dcc0971d2febf0c5d264bc55da54d342be1801b7d5b103a22a8ea2f1",
     "synthetic": true,
     "uncertainty": null
    },
    "recorded_at": "2026-10-05T10:24:43.202501+00:00",
    "sequence": 13,
    "view": {
     "detail_path": "/api/v1/observability/predictions/074d4f1795274c39c31d07d8af9c087d6d314ae0eab0c7d0e6234c6b802abee7",
     "execution_state": "PENDING",
     "input_quality": {
      "gaps": 0,
      "scope": "selected price inputs; unselected event sources retain unknown states",
      "source_states": {
       "edgar": "NOT_CONFIGURED",
       "fomc": "NOT_CONFIGURED"
      },
      "state": "VALID"
     },
     "label_state": "AVAILABLE",
     "label_versions": 1,
     "latest_decision": null,
     "latest_execution": null,
     "latest_label": {
      "available_at": "2026-06-04T06:00:00+00:00",
      "horizon_seconds": 14400,
      "identity": "32af4d4d5eabe28c2d6393b1a26a096db907854a9cf056a41f6dfd90fcfd6f1f",
      "label_id": "lab-label-741983db5d91e382eb3b0bc086494960b024e2dc88323ee5a09ad30a95ee945f",
      "prediction_hash": "074d4f1795274c39c31d07d8af9c087d6d314ae0eab0c7d0e6234c6b802abee7",
      "prediction_id": "741983db5d91e382eb3b0bc086494960b024e2dc88323ee5a09ad30a95ee945f",
      "product": "BTC-USD",
      "provenance": {
       "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
       "method": "dataset-label-after-horizon-v1",
       "synthetic": true
      },
      "realized_at": "2026-06-04T06:00:00+00:00",
      "recorded_at": "2026-10-05T10:24:43.202501+00:00",
      "schema": "label-record-v1",
      "target": "forward_return",
      "value": "0.011103950825360630545779011550092",
      "version": "1"
     },
     "prediction_hash": "074d4f1795274c39c31d07d8af9c087d6d314ae0eab0c7d0e6234c6b802abee7"
    }
   }
  ],
  "schema": "research-page-v1"
 },
 "monitoring": {
  "as_of": "2026-10-05T10:25:42.011823+00:00",
  "binding": {
   "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
   "horizon_seconds": 14400,
   "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
   "model_id": "synthetic-ridge-v1",
   "product": "BTC-USD",
   "synthetic": true
  },
  "classification": [
   {
    "category": "MISSING_DATA",
    "decision_gaps": 5,
    "freshness_seconds": 10506342.011823,
    "input_gap_unknown_rows": 0,
    "input_gaps": 0,
    "input_quality_not_valid": 0,
    "known_input_gaps": 0,
    "method": "observed-input-and-hourly-decision-gaps-v1",
    "missing_feature_rows": 0
   }
  ],
  "distributions": {
   "features": {
    "atr_pct_14": {
     "count": 40,
     "max": 0.020598882197351107,
     "mean": 0.019975809643129415,
     "min": 0.0194224058116278,
     "p50": 0.01997219504534346,
     "p95": 0.020458417325170286,
     "std": 0.00032507692534726977
    },
    "ema_spread_12_26": {
     "count": 40,
     "max": 0.002071991275923662,
     "mean": 0.000684804968124841,
     "min": -0.0008084727767637966,
     "p50": 0.0007493797232058627,
     "p95": 0.0018276454864026923,
     "std": 0.0007452007964679166
    },
    "return_1": {
     "count": 40,
     "max": 0.010006471847463533,
     "mean": 0.000704792524895963,
     "min": -0.018518518518518517,
     "p50": 0.009858500376938768,
     "p95": 0.009975607249014895,
     "std": 0.013260971656924737
    },
    "return_12": {
     "count": 40,
     "max": 0.004581445147154026,
     "mean": 0.0031167701046949364,
     "min": -0.023718104495747266,
     "p50": 0.0045251095373148,
     "p95": 0.004576397386846723,
     "std": 0.006147344460576877
    },
    "return_4": {
     "count": 40,
     "max": 0.01113873694679264,
     "mean": 0.0011249205139664716,
     "min": -0.017414273834564398,
     "p50": 0.010992786222468109,
     "p95": 0.01112617785718118,
     "std": 0.013523190498782206
    },
    "rsi_14": {
     "count": 40,
     "max": 53.85772170195261,
     "mean": 50.59410665848481,
     "min": 46.05683613699073,
     "p50": 50.79453958134829,
     "p95": 53.76223316769532,
     "std": 2.2714622437990237
    }
   },
   "prediction": {
    "count": 40,
    "max": 0.018208773340301163,
    "mean": 0.0010760974394133519,
    "min": -0.01790927369024094,
    "p50": 0.0014347119495341006,
    "p95": 0.016357697438412186,
    "std": 0.0117747038795859
   }
  },
  "drift": {
   "feature:atr_pct_14": {
    "boundaries": [
     0.019595371325113162,
     0.019897076779404055,
     0.0200467282418678,
     0.020325736797642362
    ],
    "current_count": 40,
    "ks_statistic": 0.0611111111111111,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.012396863927464193,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "feature:ema_spread_12_26": {
    "boundaries": [
     3.951956750196449e-05,
     0.0007369438574757349,
     0.001024251275093151,
     0.0014070512419204707
    ],
    "current_count": 40,
    "ks_statistic": 0.1,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.012396863927464193,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "feature:return_1": {
    "boundaries": [
     -0.018399844645111175,
     0.009843290891283055,
     0.00987666453736917,
     0.009952465834818775
    ],
    "current_count": 40,
    "ks_statistic": 0.06388888888888888,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.029311357104023456,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "feature:return_12": {
    "boundaries": [
     0.004486491758509704,
     0.004520662375313253,
     0.00453604181047234,
     0.00456575682382134
    ],
    "current_count": 40,
    "ks_statistic": 0.06666666666666671,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.00685278349707408,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "feature:return_4": {
    "boundaries": [
     -0.017302551640340218,
     0.010969637610186092,
     0.011006830131197484,
     0.011091305208952268
    ],
    "current_count": 40,
    "ks_statistic": 0.06388888888888888,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.019987097751697314,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "feature:rsi_14": {
    "boundaries": [
     48.16881439323789,
     50.52776063994573,
     51.019378123453286,
     53.479178528876915
    ],
    "current_count": 40,
    "ks_statistic": 0.05277777777777781,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.00685278349707408,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   },
   "prediction": {
    "boundaries": [
     -0.014318573011232565,
     -0.0007017033911158878,
     0.0037936279319692083,
     0.01440044292038805
    ],
    "current_count": 40,
    "ks_statistic": 0.05833333333333335,
    "method": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
    "p_value": null,
    "psi": 0.029311357104023456,
    "reference_count": 18,
    "state": "STABLE",
    "uncertainty": null
   }
  },
  "freshness": {
   "cadence_seconds": 3600,
   "decision_gaps": 5,
   "input_gap_unknown_rows": 0,
   "input_gaps": 0,
   "known_input_gaps": 0,
   "seconds_since_last_decision": 10506342.011823,
   "threshold_seconds": 7200
  },
  "inference": {
   "attempts": 0,
   "availability": null,
   "errors": 0,
   "latency_ms": {
    "count": 0,
    "max": null,
    "mean": null,
    "min": null,
    "p50": null,
    "p95": null,
    "std": null
   },
   "state": "NOT_OBSERVED"
  },
  "input_quality": {
   "missing_features": 0,
   "not_valid": 0,
   "sample": 40
  },
  "limitations": [
   "pending labels excluded; labels never used to define regimes",
   "no live freshness claim from an archived replay",
   "unobserved inference latency and availability remain null",
   "baseline advantage is descriptive and does not establish or refute an edge"
  ],
  "method": {
   "bins": 5,
   "distribution": "finite-observed-summary-v1",
   "drift": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
   "edge": "paired-descriptive-mse-reduction-v1",
   "error_rate": 0.05,
   "freshness_seconds": 7200,
   "ks_threshold": 0.3,
   "latency_ms": 1000,
   "limitations": [
    "descriptive monitoring thresholds; no statistical edge verdict",
    "overlapping horizons and serial dependence are not corrected by an interval",
    "reference and current samples must share model, artifact, product and horizon"
   ],
   "minimum_sample": 10,
   "performance_drop": "edge decline > 0.05 vs reference",
   "pseudocount": 1e-06,
   "psi_threshold": 0.2,
   "uncertainty": null,
   "version": "monitoring-method-v1"
  },
  "method_hash": "1b5ce03c0cc23654c53834fb8202dae47f22e66df2b491727852be5f3facbeae",
  "output_quality": {
   "invalid_returns": 0,
   "prediction_error_rows": 0
  },
  "performance": {
   "baselines": {
    "TRAIN_MEAN": {
     "mae": 0.013551566873139045,
     "mse": 0.0001929615497674612,
     "rmse": 0.013891060066368628,
     "sample": 40
    },
    "ZERO": {
     "mae": 0.013547756880405663,
     "mse": 0.00019297257474284963,
     "rmse": 0.013891456897778924,
     "sample": 40
    }
   },
   "edge": {
    "TRAIN_MEAN": {
     "method": "paired-descriptive-mse-reduction-v1",
     "model_mse_on_same_pairs": 5.246643551623867e-05,
     "mse_reduction": 0.7280990146510213,
     "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
     "sample": 40,
     "scope": "EXPLORATORY",
     "state": "DESCRIPTIVE_ONLY",
     "uncertainty": null
    },
    "ZERO": {
     "method": "paired-descriptive-mse-reduction-v1",
     "model_mse_on_same_pairs": 5.246643551623867e-05,
     "mse_reduction": 0.7281145489914611,
     "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
     "sample": 40,
     "scope": "EXPLORATORY",
     "state": "DESCRIPTIVE_ONLY",
     "uncertainty": null
    }
   },
   "label_policy": "latest causally available append-only correction",
   "method": "observed-forward-return-errors-v1",
   "model": {
    "mae": 0.005586417122653302,
    "mse": 5.246643551623867e-05,
    "rmse": 0.007243371833354869,
    "sample": 40
   },
   "pending": 0,
   "sample": 40,
   "uncertainty": null
  },
  "performance_by_period": {
   "2026-06": {
    "baselines": {
     "TRAIN_MEAN": {
      "mae": 0.013551566873139045,
      "mse": 0.0001929615497674612,
      "rmse": 0.013891060066368628,
      "sample": 40
     },
     "ZERO": {
      "mae": 0.013547756880405663,
      "mse": 0.00019297257474284963,
      "rmse": 0.013891456897778924,
      "sample": 40
     }
    },
    "edge": {
     "TRAIN_MEAN": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 5.246643551623867e-05,
      "mse_reduction": 0.7280990146510213,
      "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
      "sample": 40,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     },
     "ZERO": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 5.246643551623867e-05,
      "mse_reduction": 0.7281145489914611,
      "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
      "sample": 40,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     }
    },
    "label_policy": "latest causally available append-only correction",
    "method": "observed-forward-return-errors-v1",
    "model": {
     "mae": 0.005586417122653302,
     "mse": 5.246643551623867e-05,
     "rmse": 0.007243371833354869,
     "sample": 40
    },
    "pending": 0,
    "sample": 40,
    "uncertainty": null
   }
  },
  "performance_by_product": {
   "BTC-USD": {
    "baselines": {
     "TRAIN_MEAN": {
      "mae": 0.013551566873139045,
      "mse": 0.0001929615497674612,
      "rmse": 0.013891060066368628,
      "sample": 40
     },
     "ZERO": {
      "mae": 0.013547756880405663,
      "mse": 0.00019297257474284963,
      "rmse": 0.013891456897778924,
      "sample": 40
     }
    },
    "edge": {
     "TRAIN_MEAN": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 5.246643551623867e-05,
      "mse_reduction": 0.7280990146510213,
      "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
      "sample": 40,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     },
     "ZERO": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 5.246643551623867e-05,
      "mse_reduction": 0.7281145489914611,
      "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
      "sample": 40,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     }
    },
    "label_policy": "latest causally available append-only correction",
    "method": "observed-forward-return-errors-v1",
    "model": {
     "mae": 0.005586417122653302,
     "mse": 5.246643551623867e-05,
     "rmse": 0.007243371833354869,
     "sample": 40
    },
    "pending": 0,
    "sample": 40,
    "uncertainty": null
   }
  },
  "performance_by_regime": {
   "DOWN": {
    "baselines": {
     "TRAIN_MEAN": {
      "mae": 0.011098642736141738,
      "mse": 0.0001231807856978805,
      "rmse": 0.011098683962429082,
      "sample": 14
     },
     "ZERO": {
      "mae": 0.011079592772474828,
      "mse": 0.00012275829111724513,
      "rmse": 0.011079634069645312,
      "sample": 14
     }
    },
    "edge": {
     "TRAIN_MEAN": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 1.3524029793851278e-05,
      "mse_reduction": 0.8902099080044756,
      "population_hash": "a8269b29581521bef77f611ab35c8a5a98bcf1ce8850aca2ab1b71678dcbbc3d",
      "sample": 14,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     },
     "ZERO": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 1.3524029793851278e-05,
      "mse_reduction": 0.8898320457969342,
      "population_hash": "a8269b29581521bef77f611ab35c8a5a98bcf1ce8850aca2ab1b71678dcbbc3d",
      "sample": 14,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     }
    },
    "label_policy": "latest causally available append-only correction",
    "method": "observed-forward-return-errors-v1",
    "model": {
     "mae": 0.0029995954471517317,
     "mse": 1.3524029793851278e-05,
     "rmse": 0.0036775032010660817,
     "sample": 14
    },
    "pending": 0,
    "sample": 14,
    "uncertainty": null
   },
   "UP": {
    "baselines": {
     "TRAIN_MEAN": {
      "mae": 0.01487237217767606,
      "mse": 0.00023053580734338925,
      "rmse": 0.015183405656946312,
      "sample": 26
     },
     "ZERO": {
      "mae": 0.014876768323137652,
      "mse": 0.00023078026592586744,
      "rmse": 0.015191453713383306,
      "sample": 26
     }
    },
    "edge": {
     "TRAIN_MEAN": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 7.343542321290881e-05,
      "mse_reduction": 0.6814576266517904,
      "population_hash": "52589f5ec63fdfeae11fe1f6ce650beab06464c827c48cfa43377df8f53da6e4",
      "sample": 26,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     },
     "ZERO": {
      "method": "paired-descriptive-mse-reduction-v1",
      "model_mse_on_same_pairs": 7.343542321290881e-05,
      "mse_reduction": 0.681795048990462,
      "population_hash": "52589f5ec63fdfeae11fe1f6ce650beab06464c827c48cfa43377df8f53da6e4",
      "sample": 26,
      "scope": "EXPLORATORY",
      "state": "DESCRIPTIVE_ONLY",
      "uncertainty": null
     }
    },
    "label_policy": "latest causally available append-only correction",
    "method": "observed-forward-return-errors-v1",
    "model": {
     "mae": 0.006979321101769532,
     "mse": 7.343542321290881e-05,
     "rmse": 0.008569447077432056,
     "sample": 26
    },
    "pending": 0,
    "sample": 26,
    "uncertainty": null
   }
  },
  "population_hash": "dd7e5185118f65192a817f969a28b8de3530c6b0a9d4a7da8e3650630733ffb7",
  "reference_hash": "7d756087b2f9000999bb9d15b72f932710bcd7f9708dccaa09274f769cdbef63",
  "regime_definition": {
   "clock": "features actually used at decision time; never realized labels",
   "definition": [
    "HIGH_VOL if atr_pct_14 >= 0.03",
    "otherwise UP if return_4 >= 0.01",
    "otherwise DOWN if return_4 <= -0.01",
    "otherwise RANGE",
    "UNKNOWN when either feature is unavailable"
   ],
   "inputs": [
    "return_4",
    "atr_pct_14"
   ],
   "version": "price-regimes-v1"
  },
  "regime_hash": "2e93ee97a5348ca8686779654b25d60c03d723d62e979598aa323fd6f96626f4",
  "sample": 40,
  "schema": "model-monitoring-v1",
  "scope": "EXPLORATORY",
  "selection": {
   "model_id": "synthetic-ridge-v1",
   "product": "BTC-USD"
  },
  "uncertainty": null
 },
 "predictionView": {
  "decisions": [],
  "execution_state": "PENDING",
  "executions": [],
  "inputs": {
   "baselines": {
    "TRAIN_MEAN": "-0.00001904996366691180522930976805612195",
    "ZERO": "0"
   },
   "features": [
    [
     "return_1",
     "0.009975186104218362282878411910669975"
    ],
    [
     "return_4",
     "0.01112932876235902022159288517911263"
    ],
    [
     "return_12",
     "0.004541191569179130263092946344834395"
    ],
    [
     "ema_spread_12_26",
     "-0.0001434869974488636491897417864677208"
    ],
    [
     "rsi_14",
     "50.52776063994573581954290889632620"
    ],
    [
     "atr_pct_14",
     "0.02004672824186779995111848103349139"
    ]
   ],
   "input_quality": {
    "gaps": 0,
    "scope": "selected price inputs; unselected event sources retain unknown states",
    "source_states": {
     "edgar": "NOT_CONFIGURED",
     "fomc": "NOT_CONFIGURED"
    },
    "state": "VALID"
   },
   "prediction_hash": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
   "prediction_id": "20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
   "provenance": {
    "cadence_seconds": 3600,
    "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
    "experiment_hash": "2b0c8588f0639f893fd73f49eec92af1c0e27b2ebbd7c210783924b97205f3df",
    "method": "model-lab-artifact-import-v1",
    "synthetic": true
   },
   "recorded_at": "2026-10-05T10:24:43.202501+00:00",
   "schema": "prediction-evidence-v1",
   "snapshot": {
    "as_of": "2026-06-04T00:00:00+00:00",
    "companies": {
     "BTC-USD": {
      "cik": null,
      "evidence": [
       {
        "file": "scripts/trading_lab/instrument_registry.py",
        "identity": "BTC_USD: cryptocurrency, CATALOGUE_V1"
       }
      ],
      "state": "NO_SEC_ISSUER"
     }
    },
    "coverage": {
     "complete_history": false,
     "limits": [
      "source onboarding inventory is partial; no common source coverage is assumed",
      "counts include attested observations only; zero does not prove absence of news"
     ],
     "source_states": {
      "edgar": "NOT_CONFIGURED",
      "fomc": "NOT_CONFIGURED"
     },
     "state": "PARTIAL"
    },
    "events": [],
    "features": {
     "BTC-USD": {
      "edgar": {
       "history_complete": false,
       "inventory": null,
       "reason": null,
       "source_state": "NOT_CONFIGURED",
       "state": "NOT_APPLICABLE",
       "v1": {
        "count_30d": null,
        "count_7d": null,
        "hours_since_last": null,
        "item_2_02": null,
        "item_5_02": null,
        "item_7_01": null,
        "item_8_01": null
       },
       "v2": {
        "accession_revisions_attested_30d": null,
        "accession_revisions_attested_7d": null,
        "hours_since_last_new_accession": null,
        "new_accession_item_2_02_7d": null,
        "new_accession_item_5_02_7d": null,
        "new_accession_item_7_01_7d": null,
        "new_accession_item_8_01_7d": null,
        "new_accessions_attested_30d": null,
        "new_accessions_attested_7d": null
       }
      },
      "fomc": {
       "history_complete": false,
       "inventory": null,
       "reason": null,
       "source_state": "NOT_CONFIGURED",
       "state": "NOT_CONFIGURED",
       "v1": {
        "count_30d": null,
        "count_7d": null,
        "hours_since_last": null,
        "statement_within_24h": null
       },
       "v2": {
        "hours_since_last_new_statement": null,
        "new_statement_attested_within_24h": null,
        "new_statements_attested_30d": null,
        "new_statements_attested_7d": null,
        "statement_revisions_attested_30d": null,
        "statement_revisions_attested_7d": null
       }
      },
      "observations": {
       "edgar": [],
       "fomc": []
      },
      "protection": {
       "event_window_touches": [],
       "inside": []
      }
     }
    },
    "policies": {
     "causal": "CAUSAL_AVAILABILITY_V3",
     "dependencies": "FEATURE_HISTORY_AND_CAUSAL_ATTESTATIONS_V1",
     "event_features_v1": "ATTESTED_EVENT_FEATURES_V1",
     "event_features_v2": "ATTESTED_EVENT_FEATURES_V2",
     "event_understanding_hash": "efa4785e6ac2127cc5233bca97abbca1747a3ec37d157f44d818d831ad27e556",
     "mapping_hash": "1780588e6c36809145c69ea409d9bfbb38ccbe72f4a3bdf8145e446f95089c99",
     "market_causality": "PRICE_EVIDENCE_SELECTION_V1",
     "minimum_causal_quality": "SERVER_ATTESTED_EVENT_PREFIX",
     "observation_classes": "ATTESTED_OBSERVATION_CLASSES_V1",
     "prices": "PRICE_EVIDENCE_SELECTION_V1",
     "snapshot": "INFORMATION_SNAPSHOT_V1",
     "timestamp_trust": "PROVIDER_SPEC_BOUND_V1",
     "visibility": "DURABLE_OBSERVED"
    },
    "prices": {
     "BTC-USD": {
      "availability_age_seconds": 0.0,
      "freshness_seconds": 0.0,
      "policy": "PRICE_EVIDENCE_SELECTION_V1",
      "price": {
       "availability_evidence": "SYNTHETIC_AT_CLOSE_V1",
       "available_at": "2026-06-04T00:00:00+00:00",
       "bar_close_at": "2026-06-04T00:00:00+00:00",
       "bar_open_at": "2026-06-03T23:00:00+00:00",
       "close": "101.755",
       "high": "102.455",
       "identity": "3bfdb5b0ad7c38e7bc9eb1f5857d5350dbb30f7c07f56211b300b3bde389aec7",
       "ingested_at": "2026-06-04T00:00:00+00:00",
       "low": "101.055",
       "observed_at": "2026-06-04T00:00:00+00:00",
       "open": "101.655",
       "product": "BTC-USD",
       "provider_id": "model-lab-synthetic-v1",
       "revision": "c198b6f9351f9dc8dfe90e2f300f6e768866635221c3d1e0b7790f75ffb9ffc3",
       "synthetic": true,
       "volume": "103"
      },
      "quality": {
       "availability_evidence": "SYNTHETIC_AT_CLOSE_V1",
       "coverage": "PARTIAL",
       "limits": []
      },
      "state": "RESOLVED"
     }
    },
    "products": [
     "BTC-USD"
    ],
    "quality": {
     "price_states": {
      "BTC-USD": "RESOLVED"
     },
     "source_states": {
      "edgar": "NOT_CONFIGURED",
      "fomc": "NOT_CONFIGURED"
     },
     "state": "PARTIAL"
    },
    "schema": "information-snapshot-v1",
    "sources": {
     "edgar": {
      "H": null,
      "P": null,
      "coverage": {
       "attested_bounds": null,
       "complete_history": false,
       "last_boundary_inclusive_readable": false
      },
      "dependencies": null,
      "discovery": null,
      "freshness_seconds": null,
      "health": null,
      "item_states": [],
      "policy": null,
      "provider_contract_hash": "425d55371b19664f603c23e366efe2b077f98485f4ebfb8422d8b35d5fedf086",
      "provider_id": "sec_edgar_submissions_v1",
      "quality": {
       "causal_visibility": "NOT_CONFIGURED",
       "history": "PARTIAL",
       "integrity": "UNKNOWN"
      },
      "reason": null,
      "snapshot_identity": null,
      "spec_hash": "98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce",
      "spec_revision": 1,
      "state": "NOT_CONFIGURED",
      "watchlist": null
     },
     "fomc": {
      "H": null,
      "P": null,
      "coverage": {
       "attested_bounds": null,
       "complete_history": false,
       "last_boundary_inclusive_readable": false
      },
      "dependencies": null,
      "discovery": null,
      "freshness_seconds": null,
      "health": null,
      "item_states": [],
      "policy": null,
      "provider_contract_hash": "3d2e9625bc2fd173e29f1e21fb21f60391950224d9fe5033fc7536229bb08412",
      "provider_id": "federal_reserve_fomc_statements_v1",
      "quality": {
       "causal_visibility": "NOT_CONFIGURED",
       "history": "PARTIAL",
       "integrity": "UNKNOWN"
      },
      "reason": null,
      "snapshot_identity": null,
      "spec_hash": "b9d2a5997434457b5ce947c22bf80d94013d27a0e0bdb4b6be4b04a4c0c01ece",
      "spec_revision": 25,
      "state": "NOT_CONFIGURED",
      "watchlist": null
     }
    },
    "synthetic": true
   },
   "split": "validation"
  },
  "label_state": "AVAILABLE",
  "labels": [
   {
    "available_at": "2026-06-04T04:00:00+00:00",
    "horizon_seconds": 14400,
    "identity": "571b2cb2fa5e0d6bb173be23bedabed22a7ffc9f3ed0b797224373188588c1be",
    "label_id": "lab-label-20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
    "prediction_hash": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
    "prediction_id": "20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
    "product": "BTC-USD",
    "provenance": {
     "dataset_hash": "d67df70666d679f3927396d933bab36d7ff58cd7c4022b95b60da6bf52529b69",
     "method": "dataset-label-after-horizon-v1",
     "synthetic": true
    },
    "realized_at": "2026-06-04T04:00:00+00:00",
    "recorded_at": "2026-10-05T10:24:43.202501+00:00",
    "schema": "label-record-v1",
    "target": "forward_return",
    "value": "0.011006830131197484153112869146479",
    "version": "1"
   }
  ],
  "limitations": [
   "read-time enrichment; original prediction identity unchanged"
  ],
  "prediction": {
   "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
   "costs": null,
   "decision_at": "2026-06-04T00:00:00+00:00",
   "errors": [],
   "event_ids": [],
   "execution": null,
   "features_hash": "e42f6ffc5c2814d97101fc1122b84217ef19c98db9d8f6296f00e9ca443880b3",
   "horizon_seconds": 14400,
   "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
   "model_id": "synthetic-ridge-v1",
   "outputs": {
    "class": null,
    "probabilities": null,
    "quantiles": null,
    "return": "0.003793627931969208138164078012653715",
    "scenarios": null,
    "target_price": null
   },
   "prediction_id": "20d656151b35b9aa17bb7e98502b58a8ea39b880a1f2a2b92cb89af75535ffda",
   "product": "BTC-USD",
   "proposed_position": null,
   "risk": null,
   "schema": "prediction-record-v1",
   "signal": null,
   "snapshot_hash": "6952ed0245b6b391bc48e4b76bcd77690ff600afed8281cb06d22e4d23ea8f77",
   "synthetic": true,
   "uncertainty": null
  },
  "prediction_hash": "595bb64e607d9ad5f5bf58d5ccb02d04be64834aef67413aa8e06a416db8e1ce",
  "recorded_at": "2026-10-05T10:24:43.202501+00:00"
 },
 "proposals": {
  "engine": "local-rule-proposals-v1",
  "execution_enabled": false,
  "external_model_calls": 0,
  "max_proposals": 2,
  "max_trials_per_hypothesis": 8,
  "models": [
   "synthetic-ridge-v1",
   "local-momentum-v1"
  ],
  "preparation": "offline research.demo or Python propose(dataset); no HTTP write",
  "schema": "research-proposal-catalogue-v1",
  "synthetic_only": true
 },
 "references": {
  "as_of": "2026-10-05T10:30:50.294845+00:00",
  "page": {
   "has_more": true,
   "next_cursor": "eyJhZnRlciI6IjY3MSIsImFwaSI6InRyYWRpbmctbGFiLmFwcC1hcGkudjEiLCJlbmRwb2ludCI6InJlZmVyZW5jZSIsInByb2R1Y3QiOiJyZXNlYXJjaCIsInEiOiI5N2RiNDM0MWE0NTkzYzJjNDI4ZmEzZTU4YjdmNTIzNCIsInYiOiJ0cmFkaW5nLWxhYi5hcHAtYXBpLmN1cnNvci52MSJ9",
   "returned": 3
  },
  "records": [
   {
    "chain_hash": "fed5e1728109381fc8b7eadb2aab4fa4faafe967f716b8ee8217c489f2e6e26e",
    "identity": "7d756087b2f9000999bb9d15b72f932710bcd7f9708dccaa09274f769cdbef63",
    "payload": {
     "as_of": "2026-10-05T10:24:45.231472+00:00",
     "baseline_edges": {
      "TRAIN_MEAN": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 5.367472015807756e-05,
       "mse_reduction": 0.7193338851335013,
       "population_hash": "dfcdbc60c63ab5a33621c4a3bdd10fec60b513e33144dd209043637c66c70c8a",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      },
      "ZERO": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 5.367472015807756e-05,
       "mse_reduction": 0.7193323783034351,
       "population_hash": "dfcdbc60c63ab5a33621c4a3bdd10fec60b513e33144dd209043637c66c70c8a",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      }
     },
     "binding": {
      "artifact_hash": "ae124942a83b0953e133e537c91df05a0989ac0f19c2dc70a6b9e1d277c61b21",
      "horizon_seconds": 14400,
      "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
      "model_id": "synthetic-ridge-v1",
      "product": "BTC-USD",
      "synthetic": true
     },
     "features": {
      "atr_pct_14": [
       "0.02004672824186779995111848103349139",
       "0.01961791193020238127143607399232463",
       "0.02032573679764236291286516101781492",
       "0.02002343375115662830616042719628990",
       "0.01959537132511316398689902123958943",
       "0.02037274688476021962409747229316690",
       "0.01992528295162463016923897476388819",
       "0.01957335647266634264181534888049934",
       "0.02027878995044179532183117446506055",
       "0.01983776095600872595129900207995582",
       "0.01942240581162780103165231384145522",
       "0.02013360690339922475883993227807961",
       "0.01977302088106264945174978797928573",
       "0.01950016004764555822393075782512731",
       "0.02034576487366230923010378675071399",
       "0.01989707677940405360501979624932588",
       "0.02059888219735110730481106099361756",
       "0.02020901415008341314222558432966831"
      ],
      "ema_spread_12_26": [
       "-0.0001434869974488636491897417864677208",
       "0.0007979033014056280162986774240071301",
       "0.00003951956750196448428841334630234283",
       "0.0002334475900147512503511651079630208",
       "0.001170491948619157272847170746913998",
       "0.0004056577529300616333841621295198547",
       "0.0005897076066136812308944243968796191",
       "0.001514183970592168133420973098757758",
       "0.0007369438574757349434179733894181971",
       "0.0009067676318865883170194522635691011",
       "0.001815670666778875148377156741886531",
       "0.001024251275093150953147774673839153",
       "0.001178911091089763595850893247917375",
       "0.002071991275923661960021014336659135",
       "0.001266776081091421458633051690186651",
       "0.001407051241920470620481752653368327",
       "0.00001823426438045686234182516361178724",
       "-0.0002839953898762612065734767891304157"
      ],
      "return_1": [
       "0.009975186104218362282878411910669975",
       "0.009876664537369171048105744189474719",
       "-0.01844102763721292331646555079797587",
       "0.009963813017399494373667773757001933",
       "0.009865514871895553155983115735741631",
       "-0.01842041312272174969623329283110571",
       "0.009952465834818775995246583481877600",
       "0.009854390351522282688630680982497426",
       "-0.01839984464511117584231478784347995",
       "0.009941144468074583312725654087739255",
       "0.009843290891283055827619980411361410",
       "-0.01837932205033703506134523059017506",
       "0.009929848829167078351941507756150578",
       "0.009832216406593944137357530695103458",
       "-0.01835884518504165859329587289285022",
       "0.009918578830495928941524796447076240",
       "-0.01851851851851851851851851851851852",
       "0.01000647184746353362871508936127844"
      ],
      "return_12": [
       "0.004541191569179130263092946344834395",
       "0.004496578690127077223851417399804497",
       "0.004581445147154026193914645684975848",
       "0.004536041810472340005916576274529139",
       "0.004491529561099448322999560611238588",
       "0.004576203740549144448865897333863908",
       "0.004530903718295986210293031273085447",
       "0.004486491758509704476738515556422510",
       "0.004570974313111740448154220698564118",
       "0.004525777253049980322707595434868162",
       "0.004481465244288567392469189926445516",
       "0.004565756823821339950372208436724566",
       "0.004520662375313252420028499828018279",
       "0.004476449980537173997664460879719735",
       "0.004560551231844544688444951172359094",
       "0.004515559045842740748012172376558359",
       "-0.02371810449574726609963547995139733",
       "0.004555357496533967122202416320063379"
      ],
      "return_4": [
       "0.01112932876235902022159288517911263",
       "0.01101928374655647382920110192837466",
       "-0.01734132203224706512738077841102830",
       "0.01111662531017369727047146401985112",
       "0.01100683013119748415311286914647929",
       "-0.01732191514207862981704943557804593",
       "0.01110395082536063054577901155009171",
       "0.01099440463335623834298615882988122",
       "-0.01730255164034021871202916160388821",
       "0.01109130520895226777579718756189344",
       "0.01098200715791537971270284845810658",
       "-0.01728323138168754247985241285561705",
       "0.01107868836243137642811217171966962",
       "0.01096963761018609206660137120470127",
       "-0.01726395422142476116580185248048106",
       "0.01106610018772848532753680466357079",
       "-0.01741427383456439857163821356943697",
       "-0.01724472001550087192404572757217593"
      ],
      "rsi_14": [
       "50.52776063994573581954290889632620",
       "53.28621051523921160768389879713246",
       "47.86664633005570904595672254111201",
       "50.72874550652537146073542419019154",
       "53.47917852887691702684528108631287",
       "48.03431621229091609085739685958980",
       "50.89002189643886316508106857785304",
       "53.63400281840804160476497883640781",
       "48.16881439323788499942802754107799",
       "51.01937812345328912166465433050688",
       "53.75817164862333406168080769586671",
       "48.27666330706305782148392245059825",
       "51.12309532054375939486606861126575",
       "53.85772170195260955276708580375539",
       "48.36311733756793788111497437687667",
       "51.20623176227505900011169955079157",
       "46.05683613699073068437540486753265",
       "48.98676986225022629632041821459618"
      ]
     },
     "method": {
      "bins": 5,
      "distribution": "finite-observed-summary-v1",
      "drift": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
      "edge": "paired-descriptive-mse-reduction-v1",
      "error_rate": 0.05,
      "freshness_seconds": 7200,
      "ks_threshold": 0.3,
      "latency_ms": 1000,
      "limitations": [
       "descriptive monitoring thresholds; no statistical edge verdict",
       "overlapping horizons and serial dependence are not corrected by an interval",
       "reference and current samples must share model, artifact, product and horizon"
      ],
      "minimum_sample": 10,
      "performance_drop": "edge decline > 0.05 vs reference",
      "pseudocount": 1e-06,
      "psi_threshold": 0.2,
      "uncertainty": null,
      "version": "monitoring-method-v1"
     },
     "population_hash": "dfcdbc60c63ab5a33621c4a3bdd10fec60b513e33144dd209043637c66c70c8a",
     "predictions": [
      "0.003793627931969208138164078012653715",
      "-0.01264906036559238623965951541624728",
      "0.01609328869937752305012226121888087",
      "0.002073825418313075008071390792835416",
      "-0.01431857301123256412302891280224863",
      "0.01460234990625231093517213340924891",
      "0.0005656615581501159377244525105881118",
      "-0.01573641712849867664544552901667420",
      "0.01328238833544883212136401493553411",
      "-0.0007017033911158877073273720595889462",
      "-0.01697863207620924659841784281916463",
      "0.01215522510053478937529853500826947",
      "-0.001746314157668940570038257886516365",
      "-0.01790927369024094079604258772200088",
      "0.01138048186008437626896051828734473",
      "-0.002520921705382427908780600556841806",
      "0.01704993756608138606078235314774476",
      "0.01440044292038805019188323742317500"
     ],
     "reference_id": "synthetic-exp-a03df441facf1e549dcb81be-BTC-USD-validation",
     "schema": "monitoring-reference-v1",
     "selection": {
      "model_id": "synthetic-ridge-v1",
      "product": "BTC-USD",
      "split": "validation"
     },
     "synthetic": true,
     "version": "monitoring-reference-v1"
    },
    "recorded_at": "2026-10-05T10:24:45.231472+00:00",
    "sequence": 335
   },
   {
    "chain_hash": "e4caa79fb4e2129d1ca0ee1efb494097795842c2bb2600cd0010c480a3c1baf4",
    "identity": "4d8785849008555a51e6c29674efec0242cfceea557a8f0de8dbe77028df7f92",
    "payload": {
     "as_of": "2026-10-05T10:24:45.981702+00:00",
     "baseline_edges": {
      "TRAIN_MEAN": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 2.410540463376413e-05,
       "mse_reduction": 0.7204953040363835,
       "population_hash": "3ab96ca887f0dd343d74b1584467b9cde22af67b4bddedfdb90946dc449b2068",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      },
      "ZERO": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 2.410540463376413e-05,
       "mse_reduction": 0.720493535813612,
       "population_hash": "3ab96ca887f0dd343d74b1584467b9cde22af67b4bddedfdb90946dc449b2068",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      }
     },
     "binding": {
      "artifact_hash": "bbcbe0ca9e82207bac5a63dcb1fcf423b875eb3647988fe9d091f14bda32145d",
      "horizon_seconds": 14400,
      "model_contract_hash": "9bfc8016a1b82327b46f84344cf5dcc2d382964c320f27b91aedf185042f16e8",
      "model_id": "synthetic-ridge-v1",
      "product": "ETH-USD",
      "synthetic": true
     },
     "features": {
      "atr_pct_14": [
       "0.01344176358110940650407605046003701",
       "0.01319675720049487234520012413885362",
       "0.01358933776617636254403701631300767",
       "0.01343113976578867271711702586742643",
       "0.01318641913374336382765158992655937",
       "0.01362591058698560721712387569323085",
       "0.01337026668303739124130563438033449",
       "0.01317641664893069238957162416630255",
       "0.01356818074747617921215475417813492",
       "0.01331647201583491728880754840475667",
       "0.01307956729831086133939787608911035",
       "0.01347610842333863856783406881730333",
       "0.01327792483893058314693430068194784",
       "0.01313670878552391672023321252257957",
       "0.01362322567866402433662822529334938",
       "0.01336616468743134514148017954568054",
       "0.01375244280580289468646723774981873",
       "0.01353673002763115048301217164818775"
      ],
      "ema_spread_12_26": [
       "-0.00009615916824077010805672934742511516",
       "0.0005348730292502499174300030203687449",
       "0.00002648671357523914924071597935225366",
       "0.0001564702902366050015481194534745077",
       "0.0007847663038465770589354158118005280",
       "0.0002719279951754006573304250236630773",
       "0.0003953335267342202360683929301911125",
       "0.001015408018820068320233355545687519",
       "0.0004941103655481599635389235496087937",
       "0.0006080275065181953895457767806910412",
       "0.001217877785343697121541798806549287",
       "0.0006869182601737277919661496715671050",
       "0.0007907170566326946397686191731233022",
       "0.001390179272399657615633692660586851",
       "0.0008498026638624177911941142272013844",
       "0.0009440014998533820901097294528745198",
       "0.00001222912224829088480605504673647716",
       "-0.0001904489416418613308709758348778264"
      ],
      "return_1": [
       "0.006666666666666666666666666666666667",
       "0.006622516556291390728476821192052980",
       "-0.01240507986383870123068866195339094",
       "0.006661584860636993338415139363006662",
       "0.006617501810759201949035359188779878",
       "-0.01239574816026165167620605069501226",
       "0.006656510796131938005033779308517684",
       "0.006612494654077705036681251439286772",
       "-0.01238643048565265703640760834041441",
       "0.006651444455475032264469373572917701",
       "0.006607495069033530571992110453648915",
       "-0.01237712680839946441984259168544463",
       "0.006646385821043581773692216123272270",
       "0.006602503038465328646979601221955786",
       "-0.01236783709698472784231823521733455",
       "0.006641334875268461919709235090037998",
       "-0.01244009715748703472723691984507320",
       "0.006680626184066207996809253165819125"
      ],
      "return_12": [
       "0.003040417726957268911728741861925378",
       "0.003020354563361785948785292186474064",
       "0.003058408962468003058408962468003058",
       "0.003038108447262400105673337296083482",
       "0.003018075648722238624807269625693009",
       "0.003056072282753122508636726016476216",
       "0.003035802672826266292690975086619370",
       "0.003015800170458270504163115452697830",
       "0.003053739170843429481860125468848541",
       "0.003033500395673964653125824320759694",
       "0.003013528120803170755674932031838580",
       "0.003051409618573797678275290215588723",
       "0.003031201607854765905571480346611314",
       "0.003011259492013616129876931133804661",
       "0.003049083617803996950916382196003049",
       "0.003028906301442022782643049976953974",
       "-0.01596075224856909239574816026165168",
       "0.003046761160418598489866207444694662"
      ],
      "return_4": [
       "0.007435191024662263086268131576326883",
       "0.007385914006858348720654180954893168",
       "-0.01166104359789053031543778047102755",
       "0.007429519071310116086235489220563847",
       "0.007380316958255082204869691278705809",
       "-0.01165226499083529719821942916993977",
       "0.007423855765087992576144234912007424",
       "0.007374728386119707644696121683018371",
       "-0.01164349959116925592804578904333606",
       "0.007418201086236587627500331169691350",
       "0.007369148271210974767246767773135507",
       "-0.01163474736910909209752271390286947",
       "0.007412555015056752374334028260365995",
       "0.007363576594345825115055884286653517",
       "-0.01162600829496097449462786976258124",
       "0.007406917531909265260234111500562132",
       "-0.01169398548106297014091909470157343",
       "-0.01161728233912021929252055867380238"
      ],
      "rsi_14": [
       "50.52776063994573581954290889632620",
       "53.28621051523921160768389879713246",
       "47.86664633005570904595672254111201",
       "50.72874550652537146073542419019154",
       "53.47917852887691702684528108631287",
       "48.03431621229091609085739685958980",
       "50.89002189643886316508106857785304",
       "53.63400281840804160476497883640781",
       "48.16881439323788499942802754107799",
       "51.01937812345328912166465433050688",
       "53.75817164862333406168080769586671",
       "48.27666330706305782148392245059825",
       "51.12309532054375939486606861126575",
       "53.85772170195260955276708580375539",
       "48.36311733756793788111497437687667",
       "51.20623176227505900011169955079157",
       "46.05683613699073068437540486753265",
       "48.98676986225022629632041821459618"
      ]
     },
     "method": {
      "bins": 5,
      "distribution": "finite-observed-summary-v1",
      "drift": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
      "edge": "paired-descriptive-mse-reduction-v1",
      "error_rate": 0.05,
      "freshness_seconds": 7200,
      "ks_threshold": 0.3,
      "latency_ms": 1000,
      "limitations": [
       "descriptive monitoring thresholds; no statistical edge verdict",
       "overlapping horizons and serial dependence are not corrected by an interval",
       "reference and current samples must share model, artifact, product and horizon"
      ],
      "minimum_sample": 10,
      "performance_drop": "edge decline > 0.05 vs reference",
      "pseudocount": 1e-06,
      "psi_threshold": 0.2,
      "uncertainty": null,
      "version": "monitoring-method-v1"
     },
     "population_hash": "3ab96ca887f0dd343d74b1584467b9cde22af67b4bddedfdb90946dc449b2068",
     "predictions": [
      "0.002240924539998586840201589579845251",
      "-0.008622635101731750610321840707668354",
      "0.01080288565153655790198801123571894",
      "0.001015458159170759417630708837428745",
      "-0.009813095724650969469388496261816633",
      "0.009534471734338136782611016947623368",
      "0.0001667479179247352991308707471407160",
      "-0.01082081181021964457871110185028187",
      "0.008809452413619295762312991803408076",
      "-0.0005360213374712557164203161423461321",
      "-0.01131444735796650688888518154421693",
      "0.008379196062825794828532029760443580",
      "-0.001141522314718971662354925142906146",
      "-0.01225957616671805644005811120845465",
      "0.007149824695190067429432576179719280",
      "-0.002108609295697772223189737976703283",
      "0.01106695358525957899941849136606256",
      "0.009401953729000024747285876720301682"
     ],
     "reference_id": "synthetic-exp-a03df441facf1e549dcb81be-ETH-USD-validation",
     "schema": "monitoring-reference-v1",
     "selection": {
      "model_id": "synthetic-ridge-v1",
      "product": "ETH-USD",
      "split": "validation"
     },
     "synthetic": true,
     "version": "monitoring-reference-v1"
    },
    "recorded_at": "2026-10-05T10:24:45.981702+00:00",
    "sequence": 336
   },
   {
    "chain_hash": "cf3c5a8cde698987a85c991b1b4b54a22caaab9333b130d174e985961a007cc1",
    "identity": "d86bb0b98ed758277d83312e0c59534563b6fc3e0387f2b3f4bfa5bafebc9bf9",
    "payload": {
     "as_of": "2026-10-05T10:24:56.103785+00:00",
     "baseline_edges": {
      "TRAIN_MEAN": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 0.0038822119933527924,
       "mse_reduction": -19.30015907029332,
       "population_hash": "db8449042172c80e059e2b9abf3c0f17cfb5444df736bd3429afc345cbacde77",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      },
      "ZERO": {
       "method": "paired-descriptive-mse-reduction-v1",
       "model_mse_on_same_pairs": 0.0038822119933527924,
       "mse_reduction": -19.300268057051653,
       "population_hash": "db8449042172c80e059e2b9abf3c0f17cfb5444df736bd3429afc345cbacde77",
       "sample": 18,
       "scope": "EXPLORATORY",
       "state": "DESCRIPTIVE_ONLY",
       "uncertainty": null
      }
     },
     "binding": {
      "artifact_hash": "a2c1331645e362c08dc12c18d3d038f1e37ed01f8e0b5a5fc1727c9345dc6941",
      "horizon_seconds": 14400,
      "model_contract_hash": "a4ab53f042013f06d13d64a5d6caddd09f5d86ae3e99285f1bdec4d3dba19c18",
      "model_id": "local-momentum-v1",
      "product": "BTC-USD",
      "synthetic": true
     },
     "features": {
      "atr_pct_14": [
       "0.02004672824186779995111848103349139",
       "0.01961791193020238127143607399232463",
       "0.02032573679764236291286516101781492",
       "0.02002343375115662830616042719628990",
       "0.01959537132511316398689902123958943",
       "0.02037274688476021962409747229316690",
       "0.01992528295162463016923897476388819",
       "0.01957335647266634264181534888049934",
       "0.02027878995044179532183117446506055",
       "0.01983776095600872595129900207995582",
       "0.01942240581162780103165231384145522",
       "0.02013360690339922475883993227807961",
       "0.01977302088106264945174978797928573",
       "0.01950016004764555822393075782512731",
       "0.02034576487366230923010378675071399",
       "0.01989707677940405360501979624932588",
       "0.02059888219735110730481106099361756",
       "0.02020901415008341314222558432966831"
      ],
      "ema_spread_12_26": [
       "-0.0001434869974488636491897417864677208",
       "0.0007979033014056280162986774240071301",
       "0.00003951956750196448428841334630234283",
       "0.0002334475900147512503511651079630208",
       "0.001170491948619157272847170746913998",
       "0.0004056577529300616333841621295198547",
       "0.0005897076066136812308944243968796191",
       "0.001514183970592168133420973098757758",
       "0.0007369438574757349434179733894181971",
       "0.0009067676318865883170194522635691011",
       "0.001815670666778875148377156741886531",
       "0.001024251275093150953147774673839153",
       "0.001178911091089763595850893247917375",
       "0.002071991275923661960021014336659135",
       "0.001266776081091421458633051690186651",
       "0.001407051241920470620481752653368327",
       "0.00001823426438045686234182516361178724",
       "-0.0002839953898762612065734767891304157"
      ],
      "return_1": [
       "0.009975186104218362282878411910669975",
       "0.009876664537369171048105744189474719",
       "-0.01844102763721292331646555079797587",
       "0.009963813017399494373667773757001933",
       "0.009865514871895553155983115735741631",
       "-0.01842041312272174969623329283110571",
       "0.009952465834818775995246583481877600",
       "0.009854390351522282688630680982497426",
       "-0.01839984464511117584231478784347995",
       "0.009941144468074583312725654087739255",
       "0.009843290891283055827619980411361410",
       "-0.01837932205033703506134523059017506",
       "0.009929848829167078351941507756150578",
       "0.009832216406593944137357530695103458",
       "-0.01835884518504165859329587289285022",
       "0.009918578830495928941524796447076240",
       "-0.01851851851851851851851851851851852",
       "0.01000647184746353362871508936127844"
      ],
      "return_12": [
       "0.004541191569179130263092946344834395",
       "0.004496578690127077223851417399804497",
       "0.004581445147154026193914645684975848",
       "0.004536041810472340005916576274529139",
       "0.004491529561099448322999560611238588",
       "0.004576203740549144448865897333863908",
       "0.004530903718295986210293031273085447",
       "0.004486491758509704476738515556422510",
       "0.004570974313111740448154220698564118",
       "0.004525777253049980322707595434868162",
       "0.004481465244288567392469189926445516",
       "0.004565756823821339950372208436724566",
       "0.004520662375313252420028499828018279",
       "0.004476449980537173997664460879719735",
       "0.004560551231844544688444951172359094",
       "0.004515559045842740748012172376558359",
       "-0.02371810449574726609963547995139733",
       "0.004555357496533967122202416320063379"
      ],
      "return_4": [
       "0.01112932876235902022159288517911263",
       "0.01101928374655647382920110192837466",
       "-0.01734132203224706512738077841102830",
       "0.01111662531017369727047146401985112",
       "0.01100683013119748415311286914647929",
       "-0.01732191514207862981704943557804593",
       "0.01110395082536063054577901155009171",
       "0.01099440463335623834298615882988122",
       "-0.01730255164034021871202916160388821",
       "0.01109130520895226777579718756189344",
       "0.01098200715791537971270284845810658",
       "-0.01728323138168754247985241285561705",
       "0.01107868836243137642811217171966962",
       "0.01096963761018609206660137120470127",
       "-0.01726395422142476116580185248048106",
       "0.01106610018772848532753680466357079",
       "-0.01741427383456439857163821356943697",
       "-0.01724472001550087192404572757217593"
      ],
      "rsi_14": [
       "50.52776063994573581954290889632620",
       "53.28621051523921160768389879713246",
       "47.86664633005570904595672254111201",
       "50.72874550652537146073542419019154",
       "53.47917852887691702684528108631287",
       "48.03431621229091609085739685958980",
       "50.89002189643886316508106857785304",
       "53.63400281840804160476497883640781",
       "48.16881439323788499942802754107799",
       "51.01937812345328912166465433050688",
       "53.75817164862333406168080769586671",
       "48.27666330706305782148392245059825",
       "51.12309532054375939486606861126575",
       "53.85772170195260955276708580375539",
       "48.36311733756793788111497437687667",
       "51.20623176227505900011169955079157",
       "46.05683613699073068437540486753265",
       "48.98676986225022629632041821459618"
      ]
     },
     "method": {
      "bins": 5,
      "distribution": "finite-observed-summary-v1",
      "drift": "reference-quantile-psi-and-two-sample-ks-statistic-v1",
      "edge": "paired-descriptive-mse-reduction-v1",
      "error_rate": 0.05,
      "freshness_seconds": 7200,
      "ks_threshold": 0.3,
      "latency_ms": 1000,
      "limitations": [
       "descriptive monitoring thresholds; no statistical edge verdict",
       "overlapping horizons and serial dependence are not corrected by an interval",
       "reference and current samples must share model, artifact, product and horizon"
      ],
      "minimum_sample": 10,
      "performance_drop": "edge decline > 0.05 vs reference",
      "pseudocount": 1e-06,
      "psi_threshold": 0.2,
      "uncertainty": null,
      "version": "monitoring-method-v1"
     },
     "population_hash": "db8449042172c80e059e2b9abf3c0f17cfb5444df736bd3429afc345cbacde77",
     "predictions": [
      "0.03990074441687344913151364764267990",
      "0.03950665814947668419242297675789888",
      "-0.07376411054885169326586220319190348",
      "0.03985525206959797749467109502800773",
      "0.03946205948758221262393246294296652",
      "-0.07368165249088699878493317132442284",
      "0.03980986333927510398098633392751040",
      "0.03941756140608913075452272392998970",
      "-0.07359937858044470336925915137391980",
      "0.03976457787229833325090261635095702",
      "0.03937316356513222331047992164544564",
      "-0.07351728820134814024538092236070024",
      "0.03971939531666831340776603102460231",
      "0.03932886562637577654943012278041383",
      "-0.07343538074016663437318349157140088",
      "0.03967431532198371576609918578830496",
      "-0.07407407407407407407407407407407408",
      "0.04002588738985413451486035744511376"
     ],
     "reference_id": "synthetic-exp-2bbebe5bd8f5a865a5f00062-BTC-USD-validation",
     "schema": "monitoring-reference-v1",
     "selection": {
      "model_id": "local-momentum-v1",
      "product": "BTC-USD",
      "split": "validation"
     },
     "synthetic": true,
     "version": "monitoring-reference-v1"
    },
    "recorded_at": "2026-10-05T10:24:56.103785+00:00",
    "sequence": 671
   }
  ],
  "schema": "research-page-v1"
 }
};
