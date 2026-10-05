"""COMPARISON_PROTOCOL_V1: the preregistered protocol of "prices only" vs "prices + events".

The artifact `docs/artifacts/comparison_protocol_v1.json` is generated here from the registered
contracts (protected intervals, specs, cadences) and carries its canonical hash. Nothing in it is a
result: no price row is read, nothing is trained, no request is sent. Changing any rule needs a new
revision with a new hash, before any result exists.

    python -m scripts.trading_lab.comparison_protocol --write docs/artifacts/comparison_protocol_v1.json
    python -m scripts.trading_lab.comparison_protocol --revision 2 --write docs/artifacts/comparison_protocol_v2.json

Revision 2 (`build_v2`) keeps every rule of revision 1 and changes only the roles: crypto is the primary
objective with its criteria unchanged, equities an exploratory arm without a confirmatory claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from math import ceil
from pathlib import Path

from scripts.trading_lab import research_protection as rp
from scripts.trading_lab.coinbase_candles import MAX_CANDLES_PER_RESPONSE
from scripts.trading_lab.economic_backtest import PERIODS_PER_YEAR, ExecutionSpec
from scripts.trading_lab.edgar import spec as edgar_spec
from scripts.trading_lab.equity_calendar import US_EQUITY_REGULAR_SPEC, USEquityRegularCalendar
from scripts.trading_lab.equity_research import EQUITY_RESEARCH_SPEC_V1
from scripts.trading_lab.event_features import v2
from scripts.trading_lab.event_features.mapping import MAPPING_HASH
from scripts.trading_lab.fomc import spec as fomc_spec
from scripts.trading_lab.paper_model import FEATURE_SET_V2, PAPER_MODEL_SPEC_V2
from scripts.trading_lab.risk_engine import RISK_SPEC_V1
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1
from scripts.trading_lab.sources.canonical import sha256_canonical

SCHEMA = "comparison-protocol-v1"
UTC = timezone.utc
ARTIFACT = Path(__file__).resolve().parents[2] / "docs/artifacts/comparison_protocol_v1.json"

CAPTURE_START = datetime(2027, 3, 2, tzinfo=UTC)      # first authorized request
CAPTURE_END = datetime(2027, 8, 2, tzinfo=UTC)        # last attesting response (>= final decision + bound)
EVAL_START = datetime(2027, 4, 1, tzinfo=UTC)         # = CAPTURE_START + 30 days (event warm-up), inclusive
SPLIT = datetime(2027, 6, 1, tzinfo=UTC)              # train < SPLIT <= test
EVAL_END = datetime(2027, 8, 1, tzinfo=UTC)           # exclusive
CRYPTO_SERIES_START = datetime(2027, 3, 30, tzinfo=UTC)  # bar opens; 25-bar warm-up ends before EVAL_START
EQUITY_SERIES_START, EQUITY_SERIES_END = "2027-03-01", "2027-07-31"  # session dates, inclusive

BLOCK = {"crypto": 24, "equity": 10}                  # decisions per bootstrap block (>= 2x label horizon)
MIN_BLOCKS_PER_PRODUCT = 10
BOOTSTRAP = {"method": "circular block bootstrap of time-ordered paired decisions, same block starts for every "
                       "product of a family and for both variants", "resamples": 10000, "seed": 20270801,
             "interval": "percentile 2.5%-97.5% (two-sided 95%): its lower bound is the one-sided 2.5% bound",
             "families_tested": 2, "family_alpha_one_sided": 0.025,
             "familywise_note": "two co-primary families, one-sided 2.5% each: family-wise one-sided 5% (Bonferroni)"}
MARGIN = "0.005"
MAX_FOMC_STATEMENTS = 16
EDGAR_ISSUERS = ("AAPL", "MSFT", "NVDA")
SIZES = {"edgar_listing_bytes_max_observed": 185250, "fomc_raw_bytes_max_observed": 84592}
TRIAL_DEDUP = {"fomc_pilot": {"responses": 128, "distinct_raws": 49}, "edgar_trial": {"responses": 8, "distinct_raws": 2}}


def _z(moment: datetime) -> str:
    return rp.iso_z(moment)


def crypto_decisions() -> list[datetime]:
    """Admissible crypto decision instants of the evaluation period, before any source-state exclusion."""
    end = EVAL_END
    ranges = rp.crypto_ranges(CRYPTO_SERIES_START, end)
    (run,) = ranges["runs"]
    first = datetime.fromisoformat(run["first_decision_at"].replace("Z", "+00:00"))
    last = datetime.fromisoformat(run["last_decision_at"].replace("Z", "+00:00"))
    out, t = [], first
    while t <= last:
        if EVAL_START <= t < EVAL_END:
            out.append(t)
        t += rp.CRYPTO_BAR
    return out


def equity_decisions() -> list[datetime]:
    ranges = rp.equity_ranges(EQUITY_SERIES_START, EQUITY_SERIES_END)
    (run,) = ranges["runs"]
    sessions = USEquityRegularCalendar().sessions_between(
        datetime.fromisoformat(EQUITY_SERIES_START).replace(tzinfo=UTC),
        datetime.fromisoformat(EQUITY_SERIES_END).replace(tzinfo=UTC))
    lo, hi = run["first_decision_at"], run["last_decision_at"]
    return [s.close_at for s in sessions if lo <= _z(s.close_at) <= hi and EVAL_START <= s.close_at < EVAL_END]


def split(decisions: list[datetime], label_end) -> dict:
    """Train: label window ends by SPLIT (purge). Test: decisions from SPLIT on."""
    train = [t for t in decisions if t < SPLIT and label_end(t) <= SPLIT]
    test = [t for t in decisions if t >= SPLIT]
    return {"train": len(train), "purged": len([t for t in decisions if t < SPLIT]) - len(train), "test": len(test),
            "first_decision": _z(decisions[0]), "last_decision": _z(decisions[-1])}


def counts() -> dict:
    sessions = USEquityRegularCalendar().sessions_between(
        datetime.fromisoformat(EQUITY_SERIES_START).replace(tzinfo=UTC),
        datetime.fromisoformat(EQUITY_SERIES_END).replace(tzinfo=UTC))
    close_of = {s.close_at: i for i, s in enumerate(sessions)}
    equity = equity_decisions()
    horizon = rp.crypto_label_horizon()
    crypto = split(crypto_decisions(), lambda t: t + horizon * rp.CRYPTO_BAR)
    equity_split = split(equity, lambda t: sessions[close_of[t] + EQUITY_RESEARCH_SPEC_V1.target.horizon_sessions].close_at)
    for kind, row in (("crypto", crypto), ("equity", equity_split)):
        row["minimum_paired_test_decisions"] = MIN_BLOCKS_PER_PRODUCT * BLOCK[kind]
        row["expected_to_meet_minimum"] = row["test"] >= row["minimum_paired_test_decisions"]
    return {"crypto_per_product": crypto, "equity_per_instrument": equity_split,
            "note": "upper bounds: before price gaps and source-state exclusions, which are known only after capture"}


def budgets() -> dict:
    days = (CAPTURE_END - CAPTURE_START).days
    feed = days * 86400 // fomc_spec.FEED_CADENCE_S
    pages = MAX_FOMC_STATEMENTS * (1 + len(fomc_spec.OFFSETS_S))
    pages_with_retries = pages * (1 + len(fomc_spec.BACKOFF_S))
    edgar = len(EDGAR_ISSUERS) * days * 86400 // edgar_spec.POLL_INTERVAL_S
    crypto_bars = int((EVAL_END - CRYPTO_SERIES_START) / rp.CRYPTO_BAR)
    coinbase = ceil(crypto_bars / MAX_CANDLES_PER_RESPONSE) * 2
    return {
        "capture_window": {"start": _z(CAPTURE_START), "end_exclusive": _z(CAPTURE_END), "days": days,
                           "rule": "FOMC and EDGAR run for the whole window; stop when the last supplied decision is RESOLVED"},
        "fomc": {"feed_polls": feed, "feed_cadence_s": fomc_spec.FEED_CADENCE_S, "max_statements": MAX_FOMC_STATEMENTS,
                 "statement_pages_nominal": pages, "recheck_offsets_s": list(fomc_spec.OFFSETS_S),
                 "statement_pages_with_maximal_retries": pages_with_retries, "backoff_steps": len(fomc_spec.BACKOFF_S),
                 "total_request_ceiling": feed + pages_with_retries},
        "edgar": {"issuers": list(EDGAR_ISSUERS), "poll_interval_s": edgar_spec.POLL_INTERVAL_S,
                  "listings_per_issuer_per_day": 86400 // edgar_spec.POLL_INTERVAL_S, "listings": edgar,
                  "nvda_identity_lookup": 1, "total_request_ceiling": edgar + 1,
                  "global_pacing": f"{edgar_spec.SPACING_S}s spacing, at most {edgar_spec.WINDOW_MAX_STARTS} per {edgar_spec.WINDOW_S}s"},
        "prices": {"coinbase": {"products": 2, "bars_per_product": crypto_bars, "candles_per_request": MAX_CANDLES_PER_RESPONSE,
                                "requests": coinbase, "series": [_z(CRYPTO_SERIES_START), _z(EVAL_END)]},
                   "equity_daily": {"instruments": 4, "requests_per_instrument": 1, "requests": 4,
                                    "series_dates": [EQUITY_SERIES_START, EQUITY_SERIES_END]}},
        "raw_volume_bytes_worst_case_without_deduplication": {
            "edgar": edgar * SIZES["edgar_listing_bytes_max_observed"],
            "fomc_feed": feed * SIZES["fomc_raw_bytes_max_observed"], "sizes_used": SIZES,
            "observed_deduplication": TRIAL_DEDUP,
            "rule": "the operator sets a store-size ceiling and checks free disk before authorizing; the worst case "
                    "exceeds the disk of the shared server, so the authorization must carry the ceiling and a stop"},
    }


def feature_columns() -> dict:
    price = {"crypto": [f.column for f in FEATURE_SET_V2], "equity": list(EQUITY_RESEARCH_SPEC_V1.features.names)}
    fomc = list(v2.FOMC_FEATURES)
    edgar = list(v2.EDGAR_FEATURES)
    events = {"BTC-USD": fomc, "ETH-USD": fomc, "AAPL": fomc + edgar, "MSFT": fomc + edgar, "NVDA": fomc + edgar,
              "QQQ": fomc}
    return {"prices": price, "events": events,
            "event_feature_spec_hash": sha256_canonical({"policy": v2.POLICY, "classification": v2.CLASSIFICATION_POLICY,
                                                         "fomc": fomc, "edgar": edgar, "mapping_hash": MAPPING_HASH}),
            "transform": {"boolean": "0/1", "null_item_flag": "0 (counted and reported)",
                          "hours_since_last_new_*": "min(hours, 720); null (no new observation yet) -> 720",
                          "selection": "none: every listed event column enters variant B, none is dropped or added after results"}}


def build() -> dict:
    execution = ExecutionSpec()
    body = {
        "schema": SCHEMA, "revision": 1, "status": "PREREGISTERED_BEFORE_ANY_RESULT", "results_exist": False,
        "changes_need": "a new revision and canonical hash before any result exists; never an edit",
        "no_action": {"capture": False, "training": False, "trading": False, "requests_sent": 0,
                      "protected_intervals": "closed and untouched"},
        "variants": {"A": "PRICES_ONLY: the frozen price features of the product class",
                     "B": "PRICES_PLUS_EVENTS_V2: A plus every EVENT_FEATURES_V2 column of the product's sources"},
        "products": {"crypto": ["BTC-USD", "ETH-USD"],
                     "equity": ["AAPL", "MSFT", "QQQ", "NVDA (only if its EDGAR identity is verified before capture starts; else excluded from both variants)"],
                     "required_sources": {"BTC-USD": ["fomc"], "ETH-USD": ["fomc"], "AAPL": ["fomc", "edgar"],
                                          "MSFT": ["fomc", "edgar"], "NVDA": ["fomc", "edgar"], "QQQ": ["fomc"]}},
        "protection": {"table": rp.protection_table(), "event_window_days": rp.EVENT_WINDOW_DAYS,
                       "price_warmup": {"crypto_bars": rp.crypto_price_warmup(), "equity_sessions": rp.equity_price_warmup()},
                       "label_horizon": {"crypto_bars": rp.crypto_label_horizon(),
                                         "equity_sessions": EQUITY_RESEARCH_SPEC_V1.target.horizon_sessions},
                       "rule": "a decision is admissible only if no bar of its price features, no bar of its label window and "
                               "no instant of its 30-day event window (T-30d, T] lies in a protected interval of its product",
                       "admissible_ranges_price_and_label": {
                           "crypto": rp.crypto_ranges(datetime(2025, 8, 1, tzinfo=UTC), EVAL_END),
                           "equity": rp.equity_ranges("2024-08-01", EQUITY_SERIES_END)},
                       "other_reserved_intervals": "none found: the walk-forward and paper-replay periods are exploratory/consumed, "
                                                   "not reserved; closed pilots are archives, not intervals"},
        "calendar": {"possible": {"crypto": "hourly decisions, T = bar close, from the first feature-complete bar",
                                  "equity": "one decision per session at the calendar close (early closes honoured)"},
                     "calendar_spec_hash": US_EQUITY_REGULAR_SPEC.spec_hash,
                     "capture": [_z(CAPTURE_START), _z(CAPTURE_END)], "evaluation": [_z(EVAL_START), _z(EVAL_END)],
                     "split": {"train": [_z(EVAL_START), _z(SPLIT)], "test": [_z(SPLIT), _z(EVAL_END)],
                               "purge": "train decisions whose label window ends after the split instant are dropped"},
                     "price_series": {"crypto_bar_opens": [_z(CRYPTO_SERIES_START), _z(EVAL_END)],
                                      "equity_sessions": [EQUITY_SERIES_START, EQUITY_SERIES_END],
                                      "rule": "series start after every protected interval, so no feature recursion reads one"},
                     "decision_counts_upper_bounds": counts()},
        "warmup": {"price": "the feature-complete index of the real indicators (see protection.price_warmup), inside the series",
                   "event": "for every required source: state RESOLVED at T, T >= attested coverage start + 30 days and "
                            "T >= max(first valid read availability, last inventory availability at T) + 30 days, per issuer for EDGAR",
                   "capture_start_plus_30_days": _z(CAPTURE_START + timedelta(days=30))},
        "admissible_decision": ["inside the evaluation period and the product's price series",
                                "no protected interval touched by lookback, label or event window",
                                "price features and label exist (no gap)", "every required source RESOLVED, no INTEGRITY_ERROR",
                                "price and event warm-ups complete"],
        "pairing": {"rule": "A and B use exactly the same decisions, labels, costs and splits; a decision where a required "
                            "source is not RESOLVED is excluded from both; a family's test set is the decisions common to all "
                            "its products",
                    "model": {"crypto": {"spec": "ridge_regression alpha 1.0, train-fitted standardization (paper model v2)",
                                         "paper_model_spec_hash": PAPER_MODEL_SPEC_V2.paper_model_spec_hash},
                              "equity": {"spec": "EquityModelSpecV1 ridge alpha 1, one model per instrument",
                                         "research_spec_hash": EQUITY_RESEARCH_SPEC_V1.spec_hash,
                                         "model_spec_hash": EQUITY_RESEARCH_SPEC_V1.model.spec_hash,
                                         "feature_spec_hash": EQUITY_RESEARCH_SPEC_V1.features.spec_hash,
                                         "target_spec_hash": EQUITY_RESEARCH_SPEC_V1.target.spec_hash},
                              "fitting": "refit on the forward train block for both variants; the split is new because the frozen "
                                         "walk-forward geometry (252 train sessions) does not fit a 4-month window; training needs its own authorization"},
                    "features": feature_columns(),
                    "frozen_reused": {"signal_spec_hash": SIGNAL_SPEC_V1.spec_hash, "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
                                      "execution_spec_hash": execution.execution_spec_hash,
                                      "cost_model": {"fee_rate": str(execution.fee_rate), "slippage_rate": str(execution.slippage_rate),
                                                     "fill_policy": execution.fill_policy},
                                      "equity_costs": "no frozen equity execution or cost model exists: no equity economic metric is computed"}},
        "evaluation": {
            "primary": {"metric": "R = 1 - MSE_B / MSE_A on test decisions, per product, equal-weighted mean over the family's products",
                        "families": {"CRYPTO": ["BTC-USD", "ETH-USD"], "EQUITY": ["the mapped equity instruments"]},
                        "label": {"crypto": "close[T+4 bars]/close[T]-1", "equity": EQUITY_RESEARCH_SPEC_V1.target.definition},
                        "uncertainty": {**BOOTSTRAP, "block_decisions": BLOCK},
                        "minimum_sample": {"blocks_per_product": MIN_BLOCKS_PER_PRODUCT,
                                           "paired_test_decisions_per_product": {k: MIN_BLOCKS_PER_PRODUCT * v for k, v in BLOCK.items()},
                                           "crypto_fomc_statements_newly_observed_in_test": 2,
                                           "equity_edgar_newly_observed_accessions_in_test": 20},
                        "decision_rule": {"SUPPORTED": f"R > 0, lower bound of the interval > 0, and R >= {MARGIN}",
                                          "NOT_SUPPORTED": "sample sufficient and the SUPPORTED conditions not all met (no evidence of "
                                                           "improvement, not evidence of no effect)",
                                          "INCONCLUSIVE_INSUFFICIENT_SAMPLE": "any minimum sample not met; no verdict is drawn",
                                          "scope": "each family on its own; no pooling across families, no other metric can overturn it"}},
            "secondary": {"prediction": ["mae", "rmse", "spearman_rank_ic", "directional_accuracy"],
                          "baselines": ["ZERO", "TRAIN_MEAN"],
                          "economic_crypto_only": ["net_return", "sharpe", "max_drawdown", "turnover", "fees_paid"],
                          "annualization_periods": PERIODS_PER_YEAR,
                          "status": "reported with the same bootstrap; never part of the decision rule"},
            "stopping": "fixed periods; no optional stopping, no extension or re-run after a look",
            "exclusions_reported": ["protected interval or label/lookback/event window", "price warm-up", "event warm-up",
                                    "source state not RESOLVED, per source and state", "price gap or missing label",
                                    "unknown or unverified issuer mapping", "integrity errors", "null imputations (count)",
                                    "decisions kept, per product, split and variant"]},
        "requirements": {
            "authorizations": ["~/authorizations file naming FOMC endpoints, the capture window, request ceilings and expiry",
                               "same for EDGAR submissions (AAPL, MSFT, NVDA): older pages are out of scope of spec rev 1",
                               "one narrow NVDA ticker/CIK lookup (1 request), then a versioned mapping",
                               "Coinbase candles for the crypto series and an approved daily provider for the equity series",
                               "training of the two variants on the forward train block",
                               "a supervised multi-month EDGAR runner: the current runner admits 1-50 requests"],
            "data": ["attested FOMC and EDGAR stores over the capture window", "matching prices over the series windows",
                     "verified NVDA identity", "installed pandas_market_calendars pin for the equity calendar"],
            "budgets": budgets()},
        "frozen_sources": {"fomc_spec_hash": fomc_spec.SPEC_HASH, "edgar_spec_hash": edgar_spec.SPEC_HASH,
                           "event_mapping_hash": MAPPING_HASH, "features_v2_policy": v2.POLICY,
                           "classification_policy": v2.CLASSIFICATION_POLICY,
                           "features_v1_matrix_identity_prefix": "a5382f91"},
    }
    return {**body, "protocol_hash": sha256_canonical(body)}


# --- revision 2 ----------------------------------------------------------------------------------
# The operator's campaign mandate (2026-10-05): crypto is the primary objective with its fixed
# criteria unchanged; equities become an exploratory arm of this first campaign, with no
# confirmatory claim. V1 stays generated and committed exactly as it was.

SCHEMA_V2 = "comparison-protocol-v2"
ARTIFACT_V2 = Path(__file__).resolve().parents[2] / "docs/artifacts/comparison_protocol_v2.json"


def build_v2() -> dict:
    v1 = build()
    body = json.loads(json.dumps({k: v for k, v in v1.items() if k != "protocol_hash"}))
    body.update(schema=SCHEMA_V2, revision=2, supersedes={
        "revision": 1, "protocol_hash": v1["protocol_hash"], "artifact": "docs/artifacts/comparison_protocol_v1.json",
        "reason": "operator mandate 2026-10-05: crypto primary objective, equities exploratory in this first campaign",
        "changed": ["objective", "evaluation.primary (CRYPTO only)", "evaluation.exploratory (EQUITY)",
                    "evaluation.secondary.scope", "products.roles"],
        "unchanged": "calendar, capture and evaluation periods, protection, warm-ups, pairing, features, models, "
                     "costs, every crypto criterion, exclusions, requirements and budgets"})
    body["objective"] = {"primary": "CRYPTO", "primary_products": ["BTC-USD", "ETH-USD"],
                         "exploratory": "EQUITY", "exploratory_products": body["products"]["equity"]}
    body["products"]["roles"] = {"crypto": "PRIMARY_CONFIRMATORY", "equity": "EXPLORATORY_NO_CLAIM"}
    evaluation = body["evaluation"]
    primary = evaluation["primary"]
    uncertainty = primary["uncertainty"]
    primary["families"] = {"CRYPTO": primary["families"]["CRYPTO"]}
    primary["label"] = {"crypto": primary["label"]["crypto"]}
    uncertainty["block_decisions"] = {"crypto": BLOCK["crypto"]}
    uncertainty["families_tested"] = 1
    uncertainty["familywise_note"] = (
        "one confirmatory family (CRYPTO) at one-sided 2.5%, the level V1 fixed for it, kept unchanged and not "
        "relaxed; the exploratory equity arm spends no alpha")
    minimum = primary["minimum_sample"]
    minimum["paired_test_decisions_per_product"] = {"crypto": MIN_BLOCKS_PER_PRODUCT * BLOCK["crypto"]}
    del minimum["equity_edgar_newly_observed_accessions_in_test"]
    primary["decision_rule"]["scope"] = "the CRYPTO family only; no pooling with any other product; no other metric can overturn it"
    evaluation["exploratory"] = {
        "family": "EQUITY", "products": body["products"]["equity"], "status": "EXPLORATORY_NO_CLAIM",
        "computed": "the same paired metric R, its bootstrap interval (blocks of 10 sessions) and the secondary prediction "
                    "metrics, on the same decisions, splits and exclusions, reported as descriptive statistics",
        "verdict": "none: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE is never assigned to this arm, and no result of it "
                   "is cited as evidence for or against events",
        "sample_reported": {"paired_test_decisions_per_instrument": "counted and reported (the V1 count expected 37, "
                                                                      "below the 100 a verdict would need)",
                            "edgar_newly_observed_accessions_in_test": "counted and reported"},
        "effect_on_primary": "none: the equity arm cannot change, delay or condition the crypto verdict",
        "windows": "the equity sessions looked at here become exploratory; a later confirmatory equity claim needs its "
                   "own preregistered revision on data not looked at, and the reserved equity holdout stays closed"}
    evaluation["secondary"]["scope"] = "CRYPTO and the exploratory EQUITY arm; never part of any decision rule"
    return {**body, "protocol_hash": sha256_canonical(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the artifact here")
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1, help="protocol revision (V1 kept as is)")
    args = parser.parse_args()
    protocol = build() if args.revision == 1 else build_v2()
    text = json.dumps(protocol, indent=1, sort_keys=True) + "\n"
    if args.write:
        args.write.write_text(text)
    print(protocol["protocol_hash"])


if __name__ == "__main__":
    main()
