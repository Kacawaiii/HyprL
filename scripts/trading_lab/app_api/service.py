"""Read-only views over committed HyprL state.

Everything here reads artefacts that already exist -- the frozen corpus, the
committed benchmark results, the engine contracts -- and reshapes them for
display. Nothing fits a model, generates a signal, sizes a position, or writes
a byte. If a number is not already recorded on disk, this layer reports it as
unavailable rather than inventing a plausible one: a cockpit showing a
fabricated price is worse than a cockpit showing an empty panel.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import pathlib

from scripts.trading_lab.app_api.contracts import (
    APP_API_VERSION,
    CAPABILITIES,
    DEFAULT_CHART_POINTS,
    MAX_CHART_POINTS,
    SUPPORTED_PRODUCTS,
    SUPPORTED_TIMEFRAME,
    AppApiError,
    NotFoundError,
)
from scripts.trading_lab.app_api.pagination import (
    decode_cursor,
    encode_cursor,
    require_limit,
)
from scripts.trading_lab.capture_market_history import CORPUS_ID, load_canonical_rows
from scripts.trading_lab.risk_engine import (
    RISK_LIMIT_V1_IS_NOT_OPTIMIZED,
    RISK_SCALE_V1,
    RISK_SPEC_V1,
)
from scripts.trading_lab.signal_engine import (
    SIGNAL_SPEC_V1,
    SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED,
)

HOUR = timedelta(hours=1)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class AppService:
    """Reads committed state. Never writes, never computes trading logic.

    The data root is fixed at construction and every path is derived from it,
    so no client-supplied string ever reaches the filesystem.
    """

    def __init__(self, data_root):
        self._root = pathlib.Path(data_root).resolve()
        self._rows: dict[str, tuple[dict[str, str], ...]] = {}
        self._manifests: dict[str, dict] = {}

    # --- artefact access (whitelisted, never client-controlled) -----------

    def _read_json(self, relative: str) -> dict | None:
        if relative in self._manifests:
            return self._manifests[relative]
        path = self._root / relative
        if not path.is_file():
            return None
        payload = json.loads(path.read_bytes().decode("utf-8"))
        self._manifests[relative] = payload
        return payload

    def _require_product(self, product: object) -> str:
        if product not in SUPPORTED_PRODUCTS:
            raise NotFoundError(
                f"unknown product {product!r}; supported: {list(SUPPORTED_PRODUCTS)}")
        return product

    def _corpus_rows(self, product: str) -> tuple[dict[str, str], ...]:
        if product not in self._rows:
            manifest = self._read_json(f"{CORPUS_ID}/manifest.json")
            if manifest is None:
                raise NotFoundError("no market corpus is available")
            entry = next((item for item in manifest["products"]
                          if item["product"] == product), None)
            if entry is None:
                raise NotFoundError(f"{product} is not part of the corpus")
            self._rows[product] = load_canonical_rows(
                self._root / CORPUS_ID / entry["canonical_path"])
        return self._rows[product]

    # --- system --------------------------------------------------------

    def health(self) -> dict:
        corpus = self._read_json(f"{CORPUS_ID}/manifest.json")
        return {
            "status": "ok",
            "api_version": APP_API_VERSION,
            "core_status": "ready" if corpus is not None else "no_corpus",
        }

    def system(self) -> dict:
        corpus = self._read_json(f"{CORPUS_ID}/manifest.json")
        v1 = self._read_json("benchmark_results_v1/manifest.json")
        v2 = self._read_json("benchmark_results_v2/manifest.json")
        return {
            "api_version": APP_API_VERSION,
            "signal_engine": {
                "protocol": SIGNAL_SPEC_V1.version,
                "rule": SIGNAL_SPEC_V1.name,
                "spec_hash": SIGNAL_SPEC_V1.spec_hash,
                "frozen": True,
                "optimized": not SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED,
                "long_threshold": str(SIGNAL_SPEC_V1.long_threshold),
                "short_threshold": str(SIGNAL_SPEC_V1.short_threshold),
                "full_strength_excess": str(SIGNAL_SPEC_V1.full_strength_excess),
                "prediction_horizon": SIGNAL_SPEC_V1.prediction_horizon,
                "boundary_semantics": SIGNAL_SPEC_V1.boundary_semantics,
            },
            "risk_engine": {
                "protocol": RISK_SPEC_V1.protocol_version,
                "spec_hash": RISK_SPEC_V1.risk_spec_hash,
                "frozen": True,
                "optimized": not RISK_LIMIT_V1_IS_NOT_OPTIMIZED,
                "max_long_exposure": str(RISK_SPEC_V1.max_long_exposure),
                "max_short_exposure": str(RISK_SPEC_V1.max_short_exposure),
                "risk_scale": str(RISK_SCALE_V1),
                "volatility_scaling_enabled": RISK_SPEC_V1.volatility_scaling_enabled,
                "strength_mapping_version": RISK_SPEC_V1.strength_mapping_version,
            },
            "market_data": {
                "available": corpus is not None,
                "corpus_id": corpus["corpus_id"] if corpus else None,
                "corpus_spec_hash": corpus["corpus_spec_hash"] if corpus else None,
                "corpus_content_hash": corpus["corpus_content_hash"] if corpus else None,
                "products": list(SUPPORTED_PRODUCTS),
                "timeframe": SUPPORTED_TIMEFRAME,
                "point_in_time_revision_history": False,
            },
            "benchmarks": {
                "v1_available": v1 is not None,
                "v2_exploratory_available": v2 is not None,
                "v2_confirmatory_observed": False,
                "confirmatory_holdout": (v2 or {}).get("future_confirmatory_holdout"),
            },
            "capabilities": dict(CAPABILITIES),
        }

    # --- overview ------------------------------------------------------

    def overview(self) -> dict:
        corpus = self._read_json(f"{CORPUS_ID}/manifest.json")
        products = []
        for product in SUPPORTED_PRODUCTS:
            if corpus is None:
                continue
            entry = next((item for item in corpus["products"]
                          if item["product"] == product), None)
            if entry is None:
                continue
            products.append({
                "product": product,
                "timeframe": SUPPORTED_TIMEFRAME,
                "rows": entry["canonical_rows"],
                "first_open": entry["first_open"],
                "last_open": entry["last_open"],
                "missing_openings": entry["missing_count"],
                # No live price exists. Saying so beats inventing one.
                "latest_price": None,
                "latest_price_available": False,
            })
        return {
            "api_version": APP_API_VERSION,
            "system_status": "ready" if corpus is not None else "no_corpus",
            "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
            "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
            "products": products,
            "benchmarks": self.benchmark_summaries(),
            "capabilities": dict(CAPABILITIES),
        }

    # --- markets -------------------------------------------------------

    def markets(self) -> dict:
        corpus = self._read_json(f"{CORPUS_ID}/manifest.json")
        if corpus is None:
            return {"products": [], "timeframe": SUPPORTED_TIMEFRAME}
        return {
            "timeframe": SUPPORTED_TIMEFRAME,
            "corpus_id": corpus["corpus_id"],
            "corpus_content_hash": corpus["corpus_content_hash"],
            "products": [
                {
                    "product": entry["product"],
                    "rows": entry["canonical_rows"],
                    "first_open": entry["first_open"],
                    "last_open": entry["last_open"],
                    "missing_openings": entry["missing_count"],
                }
                for entry in corpus["products"]
                if entry["product"] in SUPPORTED_PRODUCTS
            ],
        }

    def _window(self, rows, start, end):
        selected = rows
        if start:
            begin = _parse(start)
            selected = [row for row in selected if _parse(row["bar_open_at"]) >= begin]
        if end:
            finish = _parse(end)
            selected = [row for row in selected if _parse(row["bar_open_at"]) <= finish]
        return list(selected)

    def market_candles(self, product, *, start=None, end=None, limit=None,
                       cursor=None) -> dict:
        """A bounded page of candles. There is no way to ask for all of them."""
        product = self._require_product(product)
        size = require_limit(limit)
        query = {"start": start or "", "end": end or "", "limit": size}
        rows = self._window(self._corpus_rows(product), start, end)
        if cursor:
            after = decode_cursor(cursor, endpoint="market_candles",
                                  product=product, query=query)
            marker = _parse(after)
            rows = [row for row in rows if _parse(row["bar_open_at"]) > marker]
        page = rows[:size]
        has_more = len(rows) > size
        next_cursor = (
            encode_cursor(endpoint="market_candles", product=product,
                          last_timestamp=page[-1]["bar_open_at"], query=query)
            if has_more and page else None
        )
        return {
            "product": product,
            "timeframe": SUPPORTED_TIMEFRAME,
            "candles": [dict(row) for row in page],
            "page": {
                "returned": len(page),
                "limit": size,
                "has_more": has_more,
                "next_cursor": next_cursor,
            },
        }

    def market_chart(self, product, *, start=None, end=None,
                     max_points=None) -> dict:
        """A bounded, viewport-sized series.

        When the window holds more points than requested, buckets are merged
        with OHLC semantics -- first open, max high, min low, last close, summed
        volume -- rather than averaged. An averaged "candle" is not a candle,
        and the metadata says plainly that the result is aggregated so no
        caller mistakes it for native 1h data.
        """
        product = self._require_product(product)
        points = require_limit(max_points, default=DEFAULT_CHART_POINTS,
                               maximum=MAX_CHART_POINTS)
        rows = self._window(self._corpus_rows(product), start, end)
        source_count = len(rows)
        if source_count <= points:
            series = [dict(row) for row in rows]
            bucket = 1
            aggregation = "none"
        else:
            bucket = -(-source_count // points)          # ceil
            series = []
            for index in range(0, source_count, bucket):
                chunk = rows[index:index + bucket]
                series.append({
                    "bar_open_at": chunk[0]["bar_open_at"],
                    "open": chunk[0]["open"],
                    "high": str(max(Decimal(row["high"]) for row in chunk)),
                    "low": str(min(Decimal(row["low"]) for row in chunk)),
                    "close": chunk[-1]["close"],
                    "volume": str(sum((Decimal(row["volume"]) for row in chunk),
                                      Decimal(0))),
                })
            aggregation = "ohlc-bucket"
        return {
            "product": product,
            "series": series,
            "metadata": {
                "source_timeframe": SUPPORTED_TIMEFRAME,
                "aggregation": aggregation,
                "bucket_size": bucket,
                "source_count": source_count,
                "returned_count": len(series),
                "max_points": points,
                "aggregated": aggregation != "none",
            },
        }

    # --- signals and risk ------------------------------------------------

    def signals(self, *, limit=None) -> dict:
        """No signal run has been persisted, and none is fabricated here."""
        require_limit(limit)
        return {
            "available": False,
            "reason": "no persisted signal run available",
            "signal_spec": {
                "protocol": SIGNAL_SPEC_V1.version,
                "rule": SIGNAL_SPEC_V1.name,
                "spec_hash": SIGNAL_SPEC_V1.spec_hash,
                "long_threshold": str(SIGNAL_SPEC_V1.long_threshold),
                "short_threshold": str(SIGNAL_SPEC_V1.short_threshold),
                "full_strength_excess": str(SIGNAL_SPEC_V1.full_strength_excess),
                "boundary_semantics": SIGNAL_SPEC_V1.boundary_semantics,
                "prediction_horizon": SIGNAL_SPEC_V1.prediction_horizon,
                "optimized": not SIGNAL_THRESHOLD_V1_IS_NOT_OPTIMIZED,
            },
            "decisions": [],
            "page": {"returned": 0, "has_more": False, "next_cursor": None},
        }

    def risk_targets(self, *, limit=None) -> dict:
        """No position-target run has been persisted, and none is fabricated."""
        require_limit(limit)
        return {
            "available": False,
            "reason": "no persisted position target run available",
            "risk_spec": {
                "protocol": RISK_SPEC_V1.protocol_version,
                "spec_hash": RISK_SPEC_V1.risk_spec_hash,
                "max_long_exposure": str(RISK_SPEC_V1.max_long_exposure),
                "max_short_exposure": str(RISK_SPEC_V1.max_short_exposure),
                "risk_scale": str(RISK_SCALE_V1),
                "volatility_scaling_enabled": RISK_SPEC_V1.volatility_scaling_enabled,
                "strength_mapping_version": RISK_SPEC_V1.strength_mapping_version,
                "risk_scale_rule_version": RISK_SPEC_V1.risk_scale_rule_version,
                "optimized": not RISK_LIMIT_V1_IS_NOT_OPTIMIZED,
            },
            "targets": [],
            "page": {"returned": 0, "has_more": False, "next_cursor": None},
        }

    # --- research --------------------------------------------------------

    def _benchmark(self, version: str) -> dict | None:
        return self._read_json(f"benchmark_results_{version}/manifest.json")

    def benchmark_summaries(self) -> list[dict]:
        summaries = []
        for version, experiment, confirmatory in (("v1", "confirmatory_pending", False),
                                                  ("v2", "exploratory", False)):
            manifest = self._benchmark(version)
            if manifest is None:
                continue
            entries = []
            for product in SUPPORTED_PRODUCTS:
                stored = self._read_json(f"benchmark_results_{version}/{product}.json")
                if stored is None:
                    continue
                metrics = stored["selection"]["global_test_metrics"]
                entries.append({
                    "product": product,
                    "rank_ic": metrics["rank_ic"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "observations": metrics["observations"],
                    "folds": stored["geometry"]["folds"],
                    "benchmark_spec_hash": stored["benchmark_spec_hash"],
                    "dataset_hash": stored["dataset_hash"],
                    "benchmark_results_hash": stored["benchmark_results_hash"],
                })
            summaries.append({
                "version": version,
                "protocol_version": manifest["benchmark_protocol_version"],
                "experiment_type": manifest.get("experiment_type", experiment),
                "confirmatory_result": manifest.get("confirmatory_result", confirmatory),
                "corpus_content_hash": manifest["corpus"]["corpus_content_hash"],
                "products": entries,
            })
        return summaries

    def benchmark_detail(self, version: str, product: str) -> dict:
        if version not in ("v1", "v2"):
            raise NotFoundError(f"unknown benchmark version {version!r}")
        product = self._require_product(product)
        stored = self._read_json(f"benchmark_results_{version}/{product}.json")
        if stored is None:
            raise NotFoundError(f"benchmark {version} has no result for {product}")
        manifest = self._benchmark(version) or {}
        return {
            "version": version,
            "product": product,
            "protocol_version": stored["protocol_version"],
            "experiment_type": manifest.get("experiment_type", "confirmatory_pending"),
            "confirmatory_result": manifest.get("confirmatory_result", False),
            "benchmark_spec_hash": stored["benchmark_spec_hash"],
            "dataset_hash": stored["dataset_hash"],
            "benchmark_results_hash": stored["benchmark_results_hash"],
            "corpus_spec_hash": stored["corpus_spec_hash"],
            "corpus_content_hash": stored["corpus_content_hash"],
            "geometry": stored["geometry"],
            "global_test_metrics": stored["selection"]["global_test_metrics"],
            "selection_counts": stored["selection"]["selection_counts"],
            "selection_reasons": stored["selection"]["selection_reasons"],
            "periods": [
                {
                    "start": period["start"],
                    "end": period["end"],
                    "rank_ic": period["metrics"]["rank_ic"],
                    "mae": period["metrics"]["mae"],
                    "rmse": period["metrics"]["rmse"],
                    "observations": period["metrics"]["observations"],
                }
                for period in stored["robustness"]["periods"]
            ],
            "scenarios": [
                {
                    "scenario_id": scenario["scenario_id"],
                    "rank_ic": scenario["global_test_metrics"]["rank_ic"],
                    "mae": scenario["global_test_metrics"]["mae"],
                    "rmse": scenario["global_test_metrics"]["rmse"],
                }
                for scenario in stored["scenarios"]
            ],
        }
