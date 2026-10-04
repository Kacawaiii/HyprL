"""One offline OOS window through the unchanged live PaperEngine pipeline.

Training is a separate, one-time command. Replay loads the frozen v2 models,
delivers each candle only at its close, and scores predictions AFTER stopping.
It never fetches, retrains, liquidates the final position, or uses a live store.
The historical corpus was captured later, without exchange revision history:
bar-close availability here is an assumption, not evidence of publication time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

from scripts.trading_lab.capture_market_history import (
    CORPUS_ID, corpus_content_hash, load_manifest, manifest_content_sha256,
)
from scripts.trading_lab.economic_backtest import (
    ECONOMIC_PRECISION, PERIODS_PER_YEAR, annualized_sharpe,
)
from scripts.trading_lab.market_dataset import forward_return_labels
from scripts.trading_lab.paper_engine import (
    PAPER_EXECUTION_SPEC_V1, PaperEngine, build_session_spec, series_from_rows,
)
from scripts.trading_lab.paper_event_store import PaperEventStore
from scripts.trading_lab.paper_model import (
    PAPER_MODEL_SPEC_V2, load_paper_model, read_artifact, train_paper_model,
    write_artifact,
)
from scripts.trading_lab.walk_forward import (
    mean_absolute_error, rank_ic, root_mean_squared_error,
)

PRODUCTS = ("BTC-USD", "ETH-USD")
REPLAY_START = "2026-05-01T00:00:00+00:00"
REPLAY_END = "2026-07-31T23:00:00+00:00"
READ_CUTOFF = "2026-08-01T00:00:00+00:00"
SEED_BARS = 200  # same history size as the live shadow CLI default
HOUR = timedelta(hours=1)
CORPUS_HASH = "688c250dba62e4c02ef468ced4c6fbd6e004f753883167fbefb00417d374748b"
MODEL_DIR = Path("data/models/paper_v2")
RESULT_DIR = Path("data/crypto/paper_replay_v2")
RUNTIME_DIR = Path("var/trading_lab/replay")
SESSION_ID = "paper-replay-v2-20260501-20260731"
RESULT_SCHEMA = "trading-lab.paper-replay-result.v2"


class PaperReplayError(RuntimeError):
    """A replay cannot proceed within its frozen boundaries."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _moment(value) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise PaperReplayError("timestamps must carry a timezone")
    return moment


def require_allowed_open(opening: str) -> None:
    # Explicit exceptions survive python -O. Apply before price/feature access.
    if _moment(opening) >= _moment(READ_CUTOFF):
        raise PaperReplayError("no bar at or after 2026-08-01 may be read")


def load_replay_corpus(root=Path("data/crypto")) -> tuple[dict, dict]:
    """Read ONLY the named committed corpus, with bounds checked before I/O.

    Refuse a manifest extending beyond July before opening a candle file.
    Stream exactly the declared row count and refuse trailing bytes without
    decoding them, then bind every byte to the frozen canonical file digest.
    Raw captured responses and all other stores are never opened.
    """
    root = Path(root)
    manifest = load_manifest(root)
    if (manifest_content_sha256(manifest) != manifest["manifest_content_sha256"]
            or corpus_content_hash(manifest["products"]) != CORPUS_HASH
            or manifest["corpus_content_hash"] != CORPUS_HASH
            or manifest["spec"]["timeframe"] != "1h"
            or tuple(entry["product"] for entry in manifest["products"]) != PRODUCTS):
        raise PaperReplayError("replay requires the frozen Coinbase history v1 corpus")
    require_allowed_open(manifest["spec"]["requested_range"]["end"])
    rows_by_product = {}
    for entry in manifest["products"]:
        require_allowed_open(entry["last_open"])
        product = entry["product"]
        # No paths supplied by the manifest are used.
        path = root / CORPUS_ID / product / "canonical.jsonl"
        rows, file_hash = [], hashlib.sha256()
        with path.open("rb") as source:
            for _ in range(entry["canonical_rows"]):
                line = source.readline()
                row = json.loads(line)
                require_allowed_open(row["bar_open_at"])
                rows.append(row)
                file_hash.update(line)
            if source.read(1):
                raise PaperReplayError("canonical corpus has undeclared trailing bytes")
        if file_hash.hexdigest() != entry["canonical_sha256"]:
            raise PaperReplayError(f"{product}: frozen canonical hash mismatch")
        openings = [row["bar_open_at"] for row in rows]
        if (openings != sorted(set(openings))
                or openings[0] != entry["first_open"]
                or openings[-1] != entry["last_open"]):
            raise PaperReplayError(f"{product}: corpus order or endpoints mismatch")
        rows_by_product[product] = rows
    return manifest, rows_by_product


def train_v2(*, model_dir=MODEL_DIR, corpus_root=Path("data/crypto")) -> dict:
    """Exactly one existing training-path fit per product; no search or tuning."""
    model_dir = Path(model_dir)
    if model_dir.exists():
        raise PaperReplayError("v2 model directory already exists; frozen models are not overwritten")
    corpus, rows = load_replay_corpus(corpus_root)
    entries = {}
    for product in PRODUCTS:
        training = [row for row in rows[product]
                    if _moment(row["bar_open_at"]) <=
                    _moment(PAPER_MODEL_SPEC_V2.training_range_end)]
        artifact = train_paper_model(
            series_from_rows(training, product=product), product=product,
            spec=PAPER_MODEL_SPEC_V2)
        file_hash = write_artifact(artifact, model_dir / f"{product}.json")
        entries[product] = {
            "file": f"{product}.json", "file_sha256": file_hash,
            **{key: artifact[key] for key in (
                "artifact_hash", "dataset_hash", "fitted_hash", "training_rows",
                "training_first_open", "training_last_open")},
            "training_input_first_open": training[0]["bar_open_at"],
            "training_input_last_open": training[-1]["bar_open_at"],
            "training_input_hash": digest(training),
        }
    manifest = {
        "schema_version": "trading-lab.paper-model-manifest.v2",
        "corpus_content_hash": corpus["corpus_content_hash"],
        "corpus_spec_hash": corpus["corpus_spec_hash"],
        "paper_model_spec_hash": PAPER_MODEL_SPEC_V2.paper_model_spec_hash,
        "spec": PAPER_MODEL_SPEC_V2.canonical(), "products": entries,
        "research_evidence": False, "shadow_only": True, "optimized": False,
    }
    manifest["manifest_hash"] = digest(manifest)
    write_artifact(manifest, model_dir / "manifest.json")
    return manifest


def load_v2_artifacts(model_dir=MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    manifest = read_artifact(model_dir / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise PaperReplayError("v2 model manifest hash mismatch")
    if (manifest["corpus_content_hash"] != CORPUS_HASH
            or manifest["paper_model_spec_hash"] != PAPER_MODEL_SPEC_V2.paper_model_spec_hash):
        raise PaperReplayError("v2 model manifest binding mismatch")
    artifacts = {}
    for product in PRODUCTS:
        path = model_dir / f"{product}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["products"][product]["file_sha256"]:
            raise PaperReplayError("v2 model file hash mismatch")
        artifacts[product] = read_artifact(path)
    return artifacts


def downsample_equity(curve: list[dict], *, maximum: int = 500) -> list[dict]:
    """Keep bucket endpoints/extrema AND the full curve's worst-DD peak/trough."""
    if maximum < 8:
        raise PaperReplayError("equity downsampling requires at least 8 points")
    if len(curve) <= maximum:
        return curve
    equities = [Decimal(row["equity"]) for row in curve]
    trough = min(range(len(curve)), key=lambda i: Decimal(curve[i]["drawdown"]))
    peak = max(range(trough + 1), key=lambda i: equities[i])
    keep = {0, len(curve) - 1, peak, trough,
            min(range(len(curve)), key=lambda i: equities[i]),
            max(range(len(curve)), key=lambda i: equities[i])}
    buckets = (maximum - 6) // 4
    if buckets == 0:
        return [curve[index] for index in sorted(keep)]
    width = (len(curve) + buckets - 1) // buckets
    for start in range(0, len(curve), width):
        end = min(start + width, len(curve))
        keep.update((start, end - 1,
                     min(range(start, end), key=lambda i: equities[i]),
                     max(range(start, end), key=lambda i: equities[i])))
    return [curve[index] for index in sorted(keep)]


@dataclass(frozen=True)
class _Equity:
    equity: Decimal


def _product_result(*, product, rows, artifact, events, chain, session_spec) -> dict:
    replay_rows = [row for row in rows if REPLAY_START <= row["bar_open_at"] <= REPLAY_END]
    # Labels are built only now, after all engine decisions have been frozen.
    labels = forward_return_labels(series_from_rows(replay_rows, product=product),
                                   horizon=PAPER_MODEL_SPEC_V2.label_horizon)
    label_by_open = {row["bar_open_at"]: label for row, label in zip(replay_rows, labels)}
    predictions, actuals, scored = [], [], []
    signals, targets, fills, curve = Counter(), Counter(), [], []
    prediction_count = gap_count = expired_count = 0
    initial = PAPER_EXECUTION_SPEC_V1.initial_equity
    snapshots = [_Equity(initial)]
    peak = initial
    for event in events:
        payload = event.payload
        if event.event_type == "PREDICTION_CREATED":
            prediction_count += 1
            actual = label_by_open[payload["bar_open_at"]]
            if actual is not None:
                predictions.append(Decimal(payload["prediction"]))
                actuals.append(actual)
                scored.append({"bar_open_at": payload["bar_open_at"],
                               "prediction": payload["prediction"], "label": str(actual)})
        elif event.event_type == "SIGNAL_CREATED":
            signals[payload["signal"]["direction"]] += 1
        elif event.event_type == "POSITION_TARGET_CREATED":
            targets[payload["target"]["side"]] += 1
        elif event.event_type == "SIMULATED_FILL":
            # Commit derived execution evidence, never captured response bodies.
            fills.append({**payload["fill"], "available_at": event.event_at,
                          "decided_at": payload["decided_at"],
                          "fill_hash": payload["fill_hash"]})
        elif event.event_type == "GAP_DETECTED":
            if "target expired" in payload.get("reason", ""):
                expired_count += 1
            else:
                gap_count += 1
        elif event.event_type == "PORTFOLIO_SNAPSHOT":
            equity = Decimal(payload["equity"])
            peak = max(peak, equity)
            curve.append({"timestamp": payload["timestamp"], "available_at": event.event_at,
                          "equity": payload["equity"], "drawdown": str(equity / peak - 1),
                          "position_quantity": payload["position_quantity"],
                          "cumulative_fees": payload["cumulative_fees"],
                          "cumulative_slippage_cost": payload["cumulative_slippage_cost"]})
            snapshots.append(_Equity(equity))
    final = snapshots[-1].equity
    fees = Decimal(curve[-1]["cumulative_fees"])
    slippage = Decimal(curve[-1]["cumulative_slippage_cost"])
    sharpe = annualized_sharpe(snapshots)
    def text(value):
        return None if value is None else str(value)
    return {
        "schema_version": RESULT_SCHEMA, "product": product,
        "experiment_type": "one out-of-sample window", "confirmatory": False,
        "optimized": False, "research_evidence": False, "shadow_only": True,
        "live_execution": False, "cost_model": "synthetic",
        "window": {"start": REPLAY_START, "end": REPLAY_END, "read_cutoff": READ_CUTOFF,
                   "first_observed_open": replay_rows[0]["bar_open_at"],
                   "last_observed_open": replay_rows[-1]["bar_open_at"]},
        "hashes": {"corpus_content_hash": CORPUS_HASH,
                   "model_artifact_hash": artifact["artifact_hash"],
                   "fitted_hash": artifact["fitted_hash"],
                   "paper_model_spec_hash": PAPER_MODEL_SPEC_V2.paper_model_spec_hash,
                   "session_spec_hash": session_spec.session_spec_hash,
                   "signal_spec_hash": session_spec.signal_spec_hash,
                   "risk_spec_hash": session_spec.risk_spec_hash,
                   "paper_execution_spec_hash": session_spec.paper_execution_spec_hash,
                   "protected_holdout_hash": session_spec.protected_holdout_hash,
                   "scored_predictions_hash": digest(scored),
                   "full_equity_hash": digest(curve), "event_chain_head_hash": chain["head_hash"]},
        "prediction_quality": {
            "rank_ic": text(rank_ic(tuple(predictions), tuple(actuals))),
            "mae": text(mean_absolute_error(tuple(predictions), tuple(actuals))),
            "rmse": text(root_mean_squared_error(tuple(predictions), tuple(actuals))),
            "observations": len(scored), "predictions": prediction_count,
            "unscored_predictions": prediction_count - len(scored),
            "label_horizon": 4, "label": "close[T+4h] / close[T] - 1",
            "rank_ic_definition": "time-series Spearman within one product/window",
            "scoring_policy": "replay-window only; exclude gap-crossing and unavailable tail labels",
        },
        "counts": {"bars": len(replay_rows), "warmup_without_prediction": len(replay_rows) - prediction_count,
                   "signals": {side: signals[side] for side in ("LONG", "FLAT", "SHORT")},
                   "targets": {side: targets[side] for side in ("LONG", "FLAT", "SHORT")},
                   "gaps": gap_count, "expired_targets": expired_count,
                   "pending_terminal_targets": 1 if targets else 0, "fills": len(fills)},
        "metrics": {"initial_equity": str(initial), "final_equity": str(final),
                    "net_return": str(final / initial - 1), "net_pnl": str(final - initial),
                    "max_drawdown": min(curve, key=lambda row: Decimal(row["drawdown"]))["drawdown"],
                    "annualized_sharpe": text(sharpe), "periods_per_year": PERIODS_PER_YEAR,
                    "total_fees": str(fees), "total_slippage_cost": str(slippage),
                    "total_execution_cost": str(fees + slippage)},
        "execution_spec": PAPER_EXECUTION_SPEC_V1.canonical(),
        "equity_metadata": {"source_count": len(curve), "aggregation": "bucket-extrema",
                            "maximum": 500, "worst_drawdown_peak_and_trough_kept": True},
        "equity_curve": downsample_equity(curve), "fills": fills,
    }


def run_replay(rows_by_product: dict, artifacts: dict, *, database_path) -> dict:
    """Replay once. A fresh store is mandatory; even opening the live one is forbidden."""
    database_path = Path(database_path)
    if database_path.exists():
        raise PaperReplayError("replay event store must be a new file")
    if database_path.name in ("paper_v1.sqlite", "paper_portfolio_v1.sqlite"):
        raise PaperReplayError("a replay may never use a live paper database")
    if set(rows_by_product) != set(artifacts) or not set(artifacts).issubset(PRODUCTS):
        raise PaperReplayError("replay products and models must agree")
    for product, rows in rows_by_product.items():
        previous = None
        for row in rows:
            opening = row["bar_open_at"]
            require_allowed_open(opening)
            if previous is not None and _moment(opening) <= _moment(previous):
                raise PaperReplayError("corpus bars must be strictly ascending")
            previous = opening
        if not any(REPLAY_START <= row["bar_open_at"] <= REPLAY_END for row in rows):
            raise PaperReplayError(f"{product}: no replay bars")
    models = {product: load_paper_model(artifact, PAPER_MODEL_SPEC_V2, product=product)
              for product, artifact in artifacts.items()}
    spec = build_session_spec(models, products=tuple(sorted(models)), model_spec=PAPER_MODEL_SPEC_V2)
    store = PaperEventStore(database_path)
    engine = PaperEngine(store=store, models=models, session_id=SESSION_ID, session_spec=spec)
    deliveries = []
    for product, rows in rows_by_product.items():
        history = [row for row in rows if row["bar_open_at"] < REPLAY_START][-SEED_BARS:]
        engine.seed_history(product, history)
        deliveries.extend((row["bar_open_at"], product, row) for row in rows
                          if REPLAY_START <= row["bar_open_at"] <= REPLAY_END)
    engine.start(now=REPLAY_START)
    for opening, product, row in sorted(deliveries):
        require_allowed_open(opening)
        now = (_moment(opening) + HOUR).isoformat()
        engine.ingest_candle(product, row, now=now, row_product=product)
    engine.stop(now=(_moment(REPLAY_END) + HOUR).isoformat())
    chain = store.verify_chain(session_id=SESSION_ID)
    events_by_product = {product: [] for product in models}
    after = 0
    while batch := store.events(session_id=SESSION_ID, after_event_id=after, limit=1000):
        for event in batch:
            if event.product in models:
                events_by_product[event.product].append(event)
        after = batch[-1].event_id
    with localcontext() as context:
        context.prec = ECONOMIC_PRECISION
        results = {product: _product_result(
            product=product, rows=rows_by_product[product], artifact=artifacts[product],
            events=events_by_product[product], chain=chain, session_spec=spec) for product in models}
    for result in results.values():
        result["result_hash"] = digest(result)
    return {"products": results, "chain": chain}


def replay_v2(*, corpus_root=Path("data/crypto"), model_dir=MODEL_DIR,
              result_dir=RESULT_DIR, runtime_dir=RUNTIME_DIR) -> dict:
    """Two identical offline deliveries to prove determinism; no second training."""
    result_dir, runtime_dir = Path(result_dir), Path(runtime_dir)
    if result_dir.exists():
        raise PaperReplayError("frozen replay results already exist; will not overwrite")
    corpus, rows = load_replay_corpus(corpus_root)
    artifacts = load_v2_artifacts(model_dir)
    databases = [runtime_dir / "paper_replay_v2.sqlite", runtime_dir / "paper_replay_v2_check.sqlite"]
    if any(path.exists() for path in databases):
        raise PaperReplayError("replay stores already exist; use a fresh replay directory")
    first = run_replay(rows, artifacts, database_path=databases[0])
    second = run_replay(rows, artifacts, database_path=databases[1])
    if first != second:
        raise PaperReplayError("second replay differs from the first")
    entries = {}
    for product, result in first["products"].items():
        file_hash = write_artifact(result, result_dir / f"{product}.json")
        entries[product] = {"file": f"{product}.json", "file_sha256": file_hash,
                            "result_hash": result["result_hash"],
                            "second_replay_result_hash": second["products"][product]["result_hash"]}
    manifest = {
        "schema_version": "trading-lab.paper-replay-manifest.v2",
        "experiment_type": "one out-of-sample window", "confirmatory": False,
        "research_evidence": False, "optimized": False, "shadow_only": True,
        "corpus_content_hash": corpus["corpus_content_hash"],
        "corpus_spec_hash": corpus["corpus_spec_hash"],
        "paper_model_spec_hash": PAPER_MODEL_SPEC_V2.paper_model_spec_hash,
        "window": {"start": REPLAY_START, "end": REPLAY_END, "read_cutoff": READ_CUTOFF},
        "history_seed_bars": SEED_BARS, "products": entries,
        "determinism": {"verified": True, "replay_count": 2,
                        "first_chain_head_hash": first["chain"]["head_hash"],
                        "second_chain_head_hash": second["chain"]["head_hash"],
                        "events": first["chain"]["events"]},
        "limitations": ["one out-of-sample window; not confirmatory; not optimized",
                        "this historical window was already spent in earlier research",
                        "historical candles captured later; no point-in-time exchange revision history",
                        "bar-close delivery is assumed; live publication delays are not replayed",
                        "synthetic costs; independent 100000 USD accounts per product",
                        "no terminal liquidation; final mark is the last observed open",
                        "hourly Sharpe annualized at 8760; gaps contribute one observed interval"],
    }
    manifest["manifest_hash"] = digest(manifest)
    write_artifact(manifest, result_dir / "manifest.json")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "replay"))
    args = parser.parse_args(argv)
    manifest = train_v2() if args.command == "train" else replay_v2()
    print(canonical(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
