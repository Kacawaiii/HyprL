"""Combine the committed V2 exploratory predictions into one portfolio.

Refits nothing, reselects nothing, and changes no threshold, cap or cost. It
reads the recorded out-of-sample predictions, runs them through Signal V1 and
Risk V1 exactly as Phase 5C did, and hands the resulting targets to the
portfolio engine so BTC and ETH share one pot of capital.

Alignment is by canonical instrument and timestamp, never by array index. Two
lists of the same length are not evidence that their rows describe the same
hour, and quietly intersecting them to get a tidy join would drop
observations without saying so. Every count is reported instead.

Causality is unchanged from 5C: a target decided on bar T is priced at the
open of bar T + one interval, and expires if that exact bar never arrived.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from scripts.trading_lab.capture_market_history import CORPUS_ID, load_canonical_rows
from scripts.trading_lab.identity import resolve_instrument
from scripts.trading_lab.portfolio import (
    ATTRIBUTION_RELATIVE_TOLERANCE,
    PORTFOLIO_SPEC_V1,
    InstrumentPositionTarget,
    PortfolioMarketFrame,
    PortfolioTargetSet,
    run_portfolio_backtest,
)
from scripts.trading_lab.risk_engine import RISK_SPEC_V1, generate_position_target
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1, generate_signal

PORTFOLIO_BACKTEST_SCHEMA_VERSION = "trading-lab.portfolio-backtest.v1"
PRODUCTS = ("BTC-USD", "ETH-USD")
TIMEFRAME = "1h"
HOUR = timedelta(hours=1)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def load_predictions(root: pathlib.Path, product: str):
    payload = json.loads(
        (root / "benchmark_results_v2" / f"{product}.json").read_bytes())
    return payload, tuple(payload["oos_records"])


def load_prices(root: pathlib.Path, product: str) -> dict:
    manifest = json.loads((root / CORPUS_ID / "manifest.json").read_bytes())
    entry = next(item for item in manifest["products"]
                 if item["product"] == product)
    rows = load_canonical_rows(root / CORPUS_ID / entry["canonical_path"])
    return {str(row["bar_open_at"]): Decimal(str(row["open"])) for row in rows}


def build_targets(records, *, benchmark, product: str):
    """Signal V1 then Risk V1, one record at a time. No batching shortcuts."""
    selection = benchmark.get("selection", {})
    model_spec_hash = selection.get("model_spec_hash") or benchmark["dataset_hash"]
    fitted_hash = selection.get("fitted_hash") or benchmark["benchmark_results_hash"]
    targets = {}
    for record in records:
        signal = generate_signal(
            timestamp=str(record["bar_open_at"]),
            prediction=Decimal(str(record["prediction"])),
            model_spec_hash=model_spec_hash, fitted_hash=fitted_hash,
            benchmark_spec_hash=benchmark["benchmark_spec_hash"],
            signal_spec=SIGNAL_SPEC_V1)
        targets[str(record["bar_open_at"])] = generate_position_target(
            signal=signal, risk_spec=RISK_SPEC_V1)
    return targets


def build_batches(root: pathlib.Path, products=PRODUCTS):
    """Assemble (frame, target_set) pairs plus the alignment report."""
    prices, targets, benchmarks = {}, {}, {}
    for product in products:
        instrument = resolve_instrument(product).canonical_id
        benchmark, records = load_predictions(root, product)
        benchmarks[instrument] = benchmark
        prices[instrument] = load_prices(root, product)
        targets[instrument] = build_targets(records, benchmark=benchmark,
                                            product=product)

    # A target decided at T is priced at the open of T + one interval, and
    # expires if that bar never arrived. Same rule as the single-product run.
    scheduled: dict[str, dict] = {}
    expired = {instrument: 0 for instrument in targets}
    for instrument, by_timestamp in targets.items():
        for decided_at, target in by_timestamp.items():
            fill_at = _iso(_parse(decided_at) + HOUR)
            if fill_at not in prices[instrument]:
                expired[instrument] += 1
                continue
            scheduled.setdefault(fill_at, {})[instrument] = target

    # Every timestamp any instrument can be priced at, so open positions can
    # be marked even on bars where they have no new target.
    all_timestamps = sorted({stamp for table in prices.values() for stamp in table}
                            & set().union(*[set(table) for table in prices.values()]))
    first_fill = min(scheduled) if scheduled else None
    last_fill = max(scheduled) if scheduled else None
    window = [stamp for stamp in all_timestamps
              if first_fill is not None and first_fill <= stamp]

    batches = []
    for stamp in window:
        frame = PortfolioMarketFrame(
            timestamp=stamp,
            prices=tuple((instrument, table[stamp])
                         for instrument, table in prices.items()
                         if stamp in table))
        due = scheduled.get(stamp)
        target_set = None
        if due:
            target_set = PortfolioTargetSet(
                timestamp=stamp,
                portfolio_spec_hash=PORTFOLIO_SPEC_V1.portfolio_spec_hash,
                targets=tuple(
                    InstrumentPositionTarget(
                        instrument_id=instrument,
                        target_exposure=target.target_exposure,
                        source_position_target_hash=target.position_target_hash,
                        side=target.side)
                    for instrument, target in due.items()))
        batches.append((frame, target_set))

    # A final flat batch, matching the single-product convention: an open
    # position marked at the last observable price would otherwise decide the
    # headline number, and 5C liquidated. Stated rather than assumed.
    if window:
        closing = window[-1]
        batches[-1] = (
            batches[-1][0],
            PortfolioTargetSet(
                timestamp=closing,
                portfolio_spec_hash=PORTFOLIO_SPEC_V1.portfolio_spec_hash,
                targets=tuple(
                    InstrumentPositionTarget(
                        instrument_id=instrument, target_exposure=Decimal(0),
                        source_position_target_hash="terminal-liquidation",
                        side="FLAT")
                    for instrument in sorted(prices)
                    if closing in prices[instrument])))

    shared = sum(1 for due in scheduled.values() if len(due) == len(products))
    single = sum(1 for due in scheduled.values() if len(due) == 1)
    alignment = {
        "instruments": {instrument: len(table)
                        for instrument, table in targets.items()},
        "expired_targets": expired,
        "scheduled_timestamps": len(scheduled),
        "shared_timestamps": shared,
        "single_instrument_timestamps": single,
        "valuation_timestamps": len(window),
        "first_fill_at": first_fill,
        "last_fill_at": last_fill,
    }
    return batches, alignment, benchmarks


def build_backtest_spec(benchmarks: dict, root: pathlib.Path) -> dict:
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1

    corpus = json.loads((root / CORPUS_ID / "manifest.json").read_bytes())
    return {
        "schema_version": PORTFOLIO_BACKTEST_SCHEMA_VERSION,
        "protocol_version": PORTFOLIO_BACKTEST_SCHEMA_VERSION,
        "instruments": sorted(benchmarks),
        "timeframe": TIMEFRAME,
        "source_benchmark_protocol": "trading-lab.real-benchmark.v2",
        "source_benchmark_results_hashes": {
            instrument: payload["benchmark_results_hash"]
            for instrument, payload in sorted(benchmarks.items())},
        "source_benchmark_spec_hashes": {
            instrument: payload["benchmark_spec_hash"]
            for instrument, payload in sorted(benchmarks.items())},
        "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
        "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
        "execution_spec_hash": EXECUTION_SPEC_V1.execution_spec_hash,
        "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
        "market_corpus_spec_hash": corpus["corpus_spec_hash"],
        "market_corpus_content_hash": corpus["corpus_content_hash"],
        "result_schema_version": "trading-lab.portfolio-result.v1",
    }


def main(argv=None) -> int:  # pragma: no cover - entry point
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(
        description="Combine committed V2 predictions into one portfolio")
    parser.add_argument("--data-root", default="data/crypto")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)

    root = pathlib.Path(arguments.data_root)
    batches, alignment, benchmarks = build_batches(root)
    result = run_portfolio_backtest(batches, spec=PORTFOLIO_SPEC_V1)
    spec = build_backtest_spec(benchmarks, root)
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    if arguments.out:
        destination = pathlib.Path(arguments.out)
        destination.mkdir(parents=True, exist_ok=True)
        payload = result.canonical()
        payload["portfolio_backtest_spec"] = spec
        payload["portfolio_backtest_spec_hash"] = spec_hash
        payload["alignment"] = alignment
        body = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")
        result_file = "portfolio-BTC-ETH.json"
        (destination / result_file).write_bytes(body)
        manifest = {
            "schema_version": PORTFOLIO_BACKTEST_SCHEMA_VERSION,
            "protocol_version": PORTFOLIO_BACKTEST_SCHEMA_VERSION,
            "experiment_type": result.experiment_type,
            "confirmatory": result.confirmatory,
            "live_execution": False,
            "cost_model": "synthetic",
            "commercial_edge_established": False,
            "instruments": list(result.instruments),
            "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
            "portfolio_backtest_spec_hash": spec_hash,
            "result_hash": result.result_hash,
            "result_file": result_file,
            "result_file_bytes": len(body),
            "result_file_sha256": hashlib.sha256(body).hexdigest(),
            "source": {
                "benchmark_protocol": "trading-lab.real-benchmark.v2",
                "benchmark_results_hashes":
                    spec["source_benchmark_results_hashes"],
                "signal_spec_hash": spec["signal_spec_hash"],
                "risk_spec_hash": spec["risk_spec_hash"],
                "execution_spec_hash": spec["execution_spec_hash"],
            },
            "alignment": alignment,
            "reconciliation": {
                "attribution_reconciles": result.attribution_reconciles(),
                "residual": str(result.reconciliation_residual()),
                # The tolerance is relative, so report the residual that way
                # too rather than leaving a reader to compare an absolute
                # figure against a ratio.
                "relative_residual": str(
                    abs(result.reconciliation_residual())
                    / abs(result.metrics.final_equity)),
                "relative_tolerance": str(ATTRIBUTION_RELATIVE_TOLERANCE),
            },
            "equity_points": len(result.equity_curve),
            "fills": len(result.fills),
            "unavailable_valuations": len(result.unavailable_valuations),
            "scaled_batches": len(result.scaled_batches),
        }
        (destination / "manifest.json").write_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    print(json.dumps({
        "alignment": alignment,
        "portfolio_backtest_spec_hash": spec_hash,
        "result_hash": result.result_hash,
        "metrics": result.metrics.canonical(),
        "attribution": [record.canonical() for record in result.attribution],
        "reconciles": result.attribution_reconciles(),
        "unavailable_valuations": len(result.unavailable_valuations),
        "scaled_batches": len(result.scaled_batches),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
