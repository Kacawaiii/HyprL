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
    DEFAULT_EQUITY_POINTS,
    DEFAULT_FILL_PAGE,
    DEFAULT_PAPER_EQUITY_POINTS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PAPER_EVENTS,
    ECONOMIC_BACKTEST_VERSIONS,
    MAX_CHART_POINTS,
    MAX_PAGE_SIZE,
    MAX_EQUITY_POINTS,
    MAX_FILL_PAGE,
    MAX_PAPER_EQUITY_POINTS,
    MAX_PAPER_EVENTS,
    PAPER_DATABASE,
    PAPER_RUNTIME_DIR,
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
from scripts.trading_lab.economic_backtest import (
    EXECUTION_COST_V1_IS_NOT_EXCHANGE_ACCOUNT_SPECIFIC,
    EXECUTION_COST_V1_IS_NOT_OPTIMIZED,
    EXECUTION_SPEC_V1,
)
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
        """Resolve any accepted spelling to the one legacy product id.

        This boundary deliberately chooses the STRICT option: the path segment
        must already be the canonical legacy id. An alias like ``btc-usd`` is
        refused rather than resolved.

        That is the narrower of the two valid choices. Accepting aliases would
        be safe here -- everything downstream would receive the resolved id --
        but it widens a read-only API's input surface for no caller that needs
        it, and one resource reachable under many URLs is a caching and
        logging nuisance. The registry still does the deciding, so the rule is
        stated once rather than inferred from a membership test, and an
        unregistered value fails closed with no fallback to a default product.
        """
        from scripts.trading_lab.identity import IdentityError, resolve_legacy_product

        try:
            resolved = resolve_legacy_product(product, context="product")
        except IdentityError as error:
            raise NotFoundError(
                f"unknown product {product!r}; supported: "
                f"{list(SUPPORTED_PRODUCTS)}") from error
        if resolved not in SUPPORTED_PRODUCTS or product != resolved:
            raise NotFoundError(
                f"unknown product {product!r}; supported: {list(SUPPORTED_PRODUCTS)} "
                "(this endpoint requires the canonical spelling)")
        return resolved

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

    # --- economic backtests ----------------------------------------------

    ECONOMIC_ROOT = "economic_backtest_{version}"

    def _economic(self, version: str, product: str | None = None):
        if version not in ECONOMIC_BACKTEST_VERSIONS:
            raise NotFoundError(f"unknown backtest version {version!r}")
        if product is None:
            return self._read_json(f"{self.ECONOMIC_ROOT.format(version=version)}"
                                   "/manifest.json")
        return self._read_json(
            f"{self.ECONOMIC_ROOT.format(version=version)}/{product}.json")

    def _backtest_summary(self, version: str, stored: dict) -> dict:
        metrics = stored["metrics"]
        gross = stored["gross_metrics"]
        return {
            "version": version,
            "product": stored["spec"]["product"],
            "experiment_type": stored["experiment_type"],
            "confirmatory": stored["confirmatory"],
            "live_execution": stored["live_execution"],
            "cost_model": stored["cost_model"],
            "source_benchmark_protocol": stored["spec"]["source_benchmark_protocol"],
            "economic_backtest_spec_hash": stored["economic_backtest_spec_hash"],
            "economic_results_hash": stored["economic_results_hash"],
            "window": stored["window"],
            "metrics": {
                "initial_equity": metrics["initial_equity"],
                "final_equity": metrics["final_equity"],
                "net_return": metrics["net_return"],
                "gross_return": metrics["gross_return"],
                "net_pnl": metrics["net_pnl"],
                "gross_pnl": gross["net_pnl"],
                "total_fees": metrics["total_fees"],
                "total_slippage_cost": metrics["total_slippage_cost"],
                "total_execution_cost": metrics["total_execution_cost"],
                "turnover_ratio": metrics["turnover_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "annualized_sharpe": metrics["annualized_sharpe"],
                "periods_per_year": metrics["periods_per_year"],
                "fill_count": metrics["fill_count"],
                "rebalance_count": metrics["rebalance_count"],
                "expired_target_count": metrics["expired_target_count"],
                "average_abs_exposure": metrics["average_abs_exposure"],
                "exposure_time_fraction": metrics["exposure_time_fraction"],
            },
        }

    def _execution_contract(self) -> dict:
        spec = EXECUTION_SPEC_V1
        return {
            "protocol": spec.protocol_version,
            "execution_spec_hash": spec.execution_spec_hash,
            "fee_rate": str(spec.fee_rate),
            "slippage_rate": str(spec.slippage_rate),
            "initial_equity": str(spec.initial_equity),
            "currency": spec.currency,
            "fill_policy": spec.fill_policy,
            "mark_policy": spec.mark_policy,
            "instrument_model": spec.instrument_model,
            "cost_model": "synthetic",
            "optimized": not EXECUTION_COST_V1_IS_NOT_OPTIMIZED,
            "exchange_account_specific":
                not EXECUTION_COST_V1_IS_NOT_EXCHANGE_ACCOUNT_SPECIFIC,
        }

    def backtests(self) -> dict:
        """A small index. No equity curve and no fill ever travels here."""
        runs = []
        for version in ECONOMIC_BACKTEST_VERSIONS:
            manifest = self._economic(version)
            if manifest is None:
                continue
            for product in SUPPORTED_PRODUCTS:
                stored = self._economic(version, product)
                if stored is None:
                    continue
                runs.append(self._backtest_summary(version, stored))
        return {
            "available": bool(runs),
            "reason": None if runs else "no persisted economic backtest available",
            "execution_spec": self._execution_contract(),
            "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
            "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
            "runs": runs,
        }

    def backtest_detail(self, version: str, product: str) -> dict:
        product = self._require_product(product)
        stored = self._economic(version, product)
        if stored is None:
            raise NotFoundError(f"no economic backtest for {version}/{product}")
        summary = self._backtest_summary(version, stored)
        summary["execution_spec"] = stored["execution_spec"]
        summary["signal_spec_hash"] = stored["spec"]["signal_spec_hash"]
        summary["risk_spec_hash"] = stored["spec"]["risk_spec_hash"]
        summary["source_benchmark_spec_hash"] = \
            stored["spec"]["source_benchmark_spec_hash"]
        summary["source_benchmark_results_hash"] = \
            stored["spec"]["source_benchmark_results_hash"]
        summary["equity_points"] = len(stored["equity_curve"])
        summary["expired_targets"] = len(stored["expired_targets"])
        return summary

    def backtest_equity(self, version: str, product: str, *, max_points=None) -> dict:
        """Bounded equity curve, downsampled so the extrema survive."""
        product = self._require_product(product)
        stored = self._economic(version, product)
        if stored is None:
            raise NotFoundError(f"no economic backtest for {version}/{product}")
        points = require_limit(max_points, default=DEFAULT_EQUITY_POINTS,
                               maximum=MAX_EQUITY_POINTS)
        curve = stored["equity_curve"]
        if len(curve) <= points:
            series, aggregation = curve, "none"
        else:
            buckets = max(1, len(curve) // max(1, points // 4))
            series, aggregation = [], "bucket-extrema"
            for start in range(0, len(curve), buckets):
                chunk = curve[start:start + buckets]
                lowest = min(chunk, key=lambda row: Decimal(row["equity"]))
                highest = max(chunk, key=lambda row: Decimal(row["equity"]))
                keep = {id(chunk[0]): chunk[0], id(lowest): lowest,
                        id(highest): highest, id(chunk[-1]): chunk[-1]}
                series.extend(sorted(keep.values(), key=lambda row: row["timestamp"]))
        return {
            "version": version, "product": product,
            "series": [{
                "timestamp": row["timestamp"], "equity": row["equity"],
                "position_quantity": row["position_quantity"],
                "target_exposure": row["target_exposure"],
                "realized_exposure": row["realized_exposure"],
                "cumulative_fees": row["cumulative_fees"],
            } for row in series],
            "metadata": {
                "source_count": len(curve), "returned_count": len(series),
                "max_points": points, "aggregation": aggregation,
                "aggregated": aggregation != "none",
                "initial_equity": stored["metrics"]["initial_equity"],
            },
        }

    def backtest_fills(self, version: str, product: str, *, limit=None,
                       cursor=None) -> dict:
        """Cursor-paginated fills. The whole history never ships at once."""
        product = self._require_product(product)
        stored = self._economic(version, product)
        if stored is None:
            raise NotFoundError(f"no economic backtest for {version}/{product}")
        size = require_limit(limit, default=DEFAULT_FILL_PAGE, maximum=MAX_FILL_PAGE)
        rows = stored["fills"]
        query = {"version": version}
        if cursor:
            after = decode_cursor(cursor, endpoint="backtest_fills", product=product,
                                  query=query)
            rows = [row for row in rows if row["timestamp"] > after]
        page = rows[:size]
        has_more = len(rows) > size
        return {
            "version": version, "product": product,
            "fills": page,
            "page": {
                "returned": len(page), "has_more": has_more,
                "total": len(stored["fills"]),
                "next_cursor": encode_cursor(
                    endpoint="backtest_fills", product=product,
                    last_timestamp=page[-1]["timestamp"], query=query)
                if has_more and page else None,
            },
        }

    # --- shadow trading ---------------------------------------------------

    def _paper_store(self):
        """The runtime log, if a session has ever written one. Read-only."""
        from scripts.trading_lab.paper_event_store import PaperEventStore
        path = pathlib.Path(PAPER_RUNTIME_DIR) / PAPER_DATABASE
        if not path.is_file():
            return None
        return PaperEventStore(path)

    def _paper_session(self):
        marker = pathlib.Path(PAPER_RUNTIME_DIR) / "paper_session.json"
        if not marker.is_file():
            return None
        try:
            return json.loads(marker.read_text())
        except (OSError, ValueError):
            return None

    def _paper_contract(self) -> dict:
        from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
        from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
        from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
        execution = PAPER_EXECUTION_SPEC_V1
        return {
            "shadow_mode": True,
            "real_money": False,
            "broker_connected": False,
            "paper_model_spec_hash": PAPER_MODEL_SPEC_V1.paper_model_spec_hash,
            "paper_model_optimized": False,
            "signal_spec_hash": SIGNAL_SPEC_V1.spec_hash,
            "risk_spec_hash": RISK_SPEC_V1.risk_spec_hash,
            "paper_execution": {
                "spec_hash": execution.paper_execution_spec_hash,
                "fee_rate": str(execution.fee_rate),
                "slippage_rate": str(execution.slippage_rate),
                "initial_equity": str(execution.initial_equity),
                "currency": execution.currency,
                "fill_price_policy": execution.fill_price_policy,
                "fill_observation_policy": execution.fill_observation_policy,
                "terminal_liquidation": execution.terminal_liquidation,
                "cost_model": execution.cost_model,
                "differs_from_backtest":
                    execution.canonical()["differs_from_backtest"],
            },
            "protected_holdout": {
                "holdout_id": PROTECTED_WINDOW_V1.holdout_id,
                "products": list(PROTECTED_WINDOW_V1.products),
                "start": PROTECTED_WINDOW_V1.start,
                "end": PROTECTED_WINDOW_V1.end,
                "holdout_hash": PROTECTED_WINDOW_V1.holdout_hash,
                "observed": PROTECTED_WINDOW_V1.observed,
            },
        }

    def paper_status(self, *, now=None) -> dict:
        from scripts.trading_lab.protected_holdout import embargo_state
        session = self._paper_session()
        store = self._paper_store()
        moment = now or _iso(datetime.now(timezone.utc))
        payload = dict(self._paper_contract())
        payload.update({
            "available": session is not None,
            "reason": None if session else "no shadow session is running",
            "session": session,
            "products": list(SUPPORTED_PRODUCTS),
            "embargo": {product: embargo_state(product, now=moment)
                        for product in SUPPORTED_PRODUCTS},
        })
        if store is not None and session:
            payload["events"] = store.count(session_id=session["session_id"])
        return payload

    def paper_products(self, *, now=None) -> dict:
        return {"products": [self.paper_product(product, now=now)
                             for product in SUPPORTED_PRODUCTS]}

    def paper_product(self, product, *, now=None) -> dict:
        from scripts.trading_lab.protected_holdout import embargo_state
        product = self._require_product(product)
        moment = now or _iso(datetime.now(timezone.utc))
        session = self._paper_session()
        store = self._paper_store()
        state = {
            "product": product,
            "available": False,
            "reason": "no shadow session is running",
            "embargo": embargo_state(product, now=moment),
            "status": "STOPPED",
            "last_candle": None, "last_prediction": None, "last_signal": None,
            "last_target": None, "last_fill": None, "portfolio": None,
            "last_event_at": None, "gap_count": 0,
        }
        if store is None or not session:
            return state
        latest = store.latest_events(session_id=session["session_id"],
                                     product=product, limit=200)
        if not latest:
            state["reason"] = "the session has produced no event for this product yet"
            state["status"] = "STARTING"
            return state

        def _last(kind):
            for event in reversed(latest):
                if event.event_type == kind:
                    return event
            return None

        candle = _last("CANDLE_INGESTED")
        prediction = _last("PREDICTION_CREATED")
        signal = _last("SIGNAL_CREATED")
        target = _last("POSITION_TARGET_CREATED")
        fill = _last("SIMULATED_FILL")
        portfolio = _last("PORTFOLIO_SNAPSHOT")
        embargoed = _last("PROTECTED_HOLDOUT_BOUNDARY_REACHED")
        state.update({
            "available": True, "reason": None,
            "status": "EMBARGOED" if embargoed else "RUNNING",
            "last_candle": candle.payload.get("bar") if candle else None,
            "last_prediction": prediction.payload if prediction else None,
            "last_signal": signal.payload.get("signal") if signal else None,
            "last_target": target.payload.get("target") if target else None,
            "last_fill": fill.payload.get("fill") if fill else None,
            "portfolio": portfolio.payload if portfolio else None,
            "last_event_at": latest[-1].event_at,
            "gap_count": sum(1 for e in latest if e.event_type == "GAP_DETECTED"),
        })
        if candle and prediction:
            state["pipeline_latency"] = {
                "bar_open_at": candle.natural_key,
                "ingested_at": candle.event_at,
                "prediction_ready_at": prediction.event_at,
            }
        return state

    def paper_events(self, product=None, *, limit=None, after_event_id=None) -> dict:
        product = self._require_product(product) if product else None
        size = require_limit(limit, default=DEFAULT_PAPER_EVENTS,
                             maximum=MAX_PAPER_EVENTS)
        session = self._paper_session()
        store = self._paper_store()
        if store is None or not session:
            return {"available": False, "reason": "no shadow session is running",
                    "events": [], "page": {"returned": 0, "last_event_id": None}}
        if after_event_id is None:
            events = store.latest_events(session_id=session["session_id"],
                                         product=product, limit=size)
        else:
            events = store.events(session_id=session["session_id"], product=product,
                                  after_event_id=int(after_event_id), limit=size)
        return {
            "available": True, "reason": None,
            "events": [{"event_id": e.event_id, "event_type": e.event_type,
                        "event_at": e.event_at, "product": e.product,
                        "natural_key": e.natural_key, "payload": e.payload,
                        "event_hash": e.event_hash} for e in events],
            "page": {"returned": len(events),
                     "last_event_id": events[-1].event_id if events else None},
        }

    def paper_equity(self, product, *, max_points=None) -> dict:
        product = self._require_product(product)
        points = require_limit(max_points, default=DEFAULT_PAPER_EQUITY_POINTS,
                               maximum=MAX_PAPER_EQUITY_POINTS)
        session = self._paper_session()
        store = self._paper_store()
        if store is None or not session:
            return {"available": False, "reason": "no shadow session is running",
                    "product": product, "series": [],
                    "metadata": {"returned_count": 0, "source_count": 0}}
        events = store.latest_events(session_id=session["session_id"], product=product,
                                     limit=MAX_PAPER_EQUITY_POINTS)
        series = [{"timestamp": e.payload["timestamp"], "equity": e.payload["equity"],
                   "position_quantity": e.payload["position_quantity"],
                   "cumulative_fees": e.payload["cumulative_fees"]}
                  for e in events if e.event_type == "PORTFOLIO_SNAPSHOT"]
        trimmed = series[-points:]
        return {"available": True, "reason": None, "product": product,
                "series": trimmed,
                "metadata": {"returned_count": len(trimmed),
                             "source_count": len(series), "max_points": points}}

    def paper_fills(self, product, *, limit=None) -> dict:
        product = self._require_product(product)
        size = require_limit(limit, default=DEFAULT_PAPER_EVENTS,
                             maximum=MAX_PAPER_EVENTS)
        session = self._paper_session()
        store = self._paper_store()
        if store is None or not session:
            return {"available": False, "reason": "no shadow session is running",
                    "product": product, "fills": []}
        events = store.latest_events(session_id=session["session_id"], product=product,
                                     limit=MAX_PAPER_EVENTS)
        fills = [dict(e.payload["fill"], decided_at=e.payload.get("decided_at"))
                 for e in events if e.event_type == "SIMULATED_FILL"]
        return {"available": True, "reason": None, "product": product,
                "fills": fills[-size:]}

    def paper_predictions(self, product, *, limit=None) -> dict:
        product = self._require_product(product)
        size = require_limit(limit, default=DEFAULT_PAPER_EVENTS,
                             maximum=MAX_PAPER_EVENTS)
        session = self._paper_session()
        store = self._paper_store()
        if store is None or not session:
            return {"available": False, "reason": "no shadow session is running",
                    "product": product, "predictions": []}
        events = store.latest_events(session_id=session["session_id"], product=product,
                                     limit=MAX_PAPER_EVENTS)
        rows = [e.payload for e in events if e.event_type == "PREDICTION_CREATED"]
        return {"available": True, "reason": None, "product": product,
                "predictions": rows[-size:]}

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

    # --- operations (read-only) -----------------------------------------

    def _layout(self):
        from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
        return RuntimeLayout(PAPER_RUNTIME_DIR)

    def _health_history(self):
        from scripts.trading_lab.ops.health import HealthHistory
        layout = self._layout()
        if not layout.ops_database.is_file():
            return None
        return HealthHistory(layout.ops_database)

    def ops_health_history(self, *, component=None, limit=None) -> dict:
        """Recent observations, newest first. Bounded like every other feed."""
        from scripts.trading_lab.ops.health import (
            COMPONENTS, HEALTH_STATES, MAX_HEALTH_RECORDS)

        if component is not None and component not in COMPONENTS:
            raise NotFoundError(
                f"unknown component {component!r}; supported: {list(COMPONENTS)}")
        size = DEFAULT_PAGE_SIZE if limit is None else int(limit)
        if size < 1 or size > MAX_PAGE_SIZE:
            raise AppApiError(f"limit must sit in 1..{MAX_PAGE_SIZE}")
        history = self._health_history()
        if history is None:
            return {"records": [], "components": list(COMPONENTS),
                    "states": list(HEALTH_STATES), "available": False,
                    "retention": MAX_HEALTH_RECORDS}
        return {
            "available": True,
            "components": list(COMPONENTS),
            "states": list(HEALTH_STATES),
            "retention": MAX_HEALTH_RECORDS,
            "latest": history.latest_per_component(),
            "records": list(history.recent(component=component, limit=size)),
        }

    def ops_runtime(self) -> dict:
        """Lifecycle and layout. No absolute path leaves this method."""
        from scripts.trading_lab.ops import recovery as recovery_module
        from scripts.trading_lab.ops import supervisor
        from scripts.trading_lab.ops.runtime_paths import RUNTIME_SCHEMA_VERSION

        layout = self._layout()
        store = self._paper_store()
        payload = {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "layout": layout.describe(),
            "app": supervisor.status(layout.pid_file),
            "paper_session": self._paper_session(),
            "snapshots": recovery_module.snapshot_pressure(store),
            "real_money": False,
            "broker_connected": False,
        }
        return payload

    def ops_recovery(self) -> dict:
        """What happened at the last shutdown, and is the log still sound."""
        from scripts.trading_lab.ops import recovery as recovery_module

        from scripts.trading_lab.ops import supervisor

        layout = self._layout()
        store = self._paper_store()
        # Whether a process is live decides how a RUNNING lifecycle marker is
        # read: a live app's "last shutdown" is the one before it started.
        live = supervisor.inspect(layout.pid_file)["state"] == supervisor.RUNNING
        clean = recovery_module.last_shutdown_clean(layout.lifecycle_file,
                                                    running=live)
        report = recovery_module.verify_runtime(store)
        return {
            "last_shutdown_clean": clean,
            # Recovery only counts as "performed" when there was something to
            # recover from: a first run is not a recovery.
            "recovery_performed": clean is False,
            "event_chain_verified": report["event_chain_verified"],
            "latest_snapshot_verified": report["latest_snapshot_verified"],
            "status": report["status"],
            "error_code": report["error_code"],
            "events": report["events"],
            "sessions": report["sessions"],
        }

    def ops_storage(self) -> dict:
        """Sizes and counts. Never a path, never a filesystem listing."""
        from scripts.trading_lab.ops.structured_log import (
            MAX_LOG_FILE_SIZE, MAX_LOG_FILES)
        from scripts.trading_lab.ops.support_bundle import storage_report

        layout = self._layout()
        store = self._paper_store()
        payload = storage_report(layout)
        payload.update({
            "events": store.count() if store is not None else 0,
            "sessions": len(store.sessions()) if store is not None else 0,
            "log_cap_bytes": MAX_LOG_FILE_SIZE * MAX_LOG_FILES,
            "paper_events_retention": "append-only; never pruned automatically",
        })
        snapshots = 0
        if store is not None:
            for session in store.sessions():
                for product in SUPPORTED_PRODUCTS:
                    try:
                        if store.latest_snapshot(session_id=session,
                                                 product=product):
                            snapshots += 1
                    except Exception:
                        continue
        payload["snapshots"] = snapshots
        # An audit trail is not pruned to save space, so the honest response to
        # a large database is to say so, not to delete evidence.
        payload["database_warning"] = (
            "the paper event log is large; export and archive it rather than "
            "deleting events" if payload["paper_database_bytes"] > 512 * 1024 * 1024
            else None)
        return payload

    def ops_settings(self) -> dict:
        """Current operational settings and the fields that are refused."""
        from scripts.trading_lab.ops import settings as settings_module

        layout = self._layout()
        payload = settings_module.describe()
        payload["current"] = settings_module.load(layout.settings_file)
        return payload

    # --- instruments and providers (read-only) ---------------------------

    def instruments(self) -> dict:
        """The registered markets, grouped by asset class.

        Served from the registry rather than a list in this file: two lists of
        products drift, and the frontend hardcoding its own would be a third.
        """
        from scripts.trading_lab.instrument_registry import (
            INSTRUMENTS_V1, PROVIDERS_V1)

        grouped = INSTRUMENTS_V1.by_asset_class()
        return {
            "api_version": APP_API_VERSION,
            "count": len(INSTRUMENTS_V1),
            "asset_classes": [
                {
                    "asset_class": asset_class,
                    "instruments": [
                        self._instrument_payload(spec, PROVIDERS_V1)
                        for spec in specs
                    ],
                }
                for asset_class, specs in sorted(grouped.items())
            ],
            "instruments": [self._instrument_payload(spec, PROVIDERS_V1)
                            for spec in INSTRUMENTS_V1.all()],
        }

    def _instrument_payload(self, spec, providers) -> dict:
        payload = dict(spec.payload())
        payload["providers"] = [provider.provider_id
                                for provider in providers.for_instrument(
                                    spec.instrument_id)]
        # What the rest of the API and every committed artefact call it.
        payload["legacy_product_id"] = spec.legacy_product_id
        return payload

    def instrument_detail(self, instrument_id) -> dict:
        from scripts.trading_lab.instrument_registry import (
            INSTRUMENTS_V1, PROVIDERS_V1, RegistryError)
        from scripts.trading_lab.trading_calendar import get_calendar

        try:
            spec = INSTRUMENTS_V1.resolve(instrument_id)
        except RegistryError as error:
            raise NotFoundError(str(error)) from error
        payload = self._instrument_payload(spec, PROVIDERS_V1)
        calendar = get_calendar(spec.trading_calendar)
        payload["calendar"] = {
            **calendar.payload(),
            "bars_per_day": calendar.bars_per_day(SUPPORTED_TIMEFRAME),
            "annualization_periods":
                calendar.annualization_periods(SUPPORTED_TIMEFRAME),
        }
        payload["provider_details"] = [
            provider.payload()
            for provider in PROVIDERS_V1.for_instrument(spec.instrument_id)]
        return payload

    def providers(self) -> dict:
        from scripts.trading_lab.instrument_registry import PROVIDERS_V1

        return {"api_version": APP_API_VERSION, **PROVIDERS_V1.payload()}

    def provider_detail(self, provider_id) -> dict:
        from scripts.trading_lab.instrument_registry import (
            INSTRUMENTS_V1, PROVIDERS_V1, RegistryError)

        try:
            provider = PROVIDERS_V1.resolve(provider_id)
        except RegistryError as error:
            raise NotFoundError(str(error)) from error
        payload = dict(provider.payload())
        payload["instrument_details"] = [
            INSTRUMENTS_V1.resolve(item).payload()
            for item in provider.supported_instruments]
        return payload

    # --- portfolio (read-only) -------------------------------------------

    PORTFOLIO_VERSIONS = ("v1",)

    def _portfolio_manifest(self):
        return self._read_json("portfolio_backtest_v1/manifest.json")

    def _portfolio_result(self, version: str):
        from scripts.trading_lab.app_api.contracts import NotFoundError

        if version not in self.PORTFOLIO_VERSIONS:
            raise NotFoundError(
                f"unknown portfolio backtest version {version!r}; supported: "
                f"{list(self.PORTFOLIO_VERSIONS)}")
        manifest = self._portfolio_manifest()
        if manifest is None:
            raise NotFoundError("no portfolio backtest has been run")
        return manifest, self._read_json(
            f"portfolio_backtest_v1/{manifest['result_file']}")

    def _portfolio_contract(self) -> dict:
        """The frozen allocation rules, available with or without a result."""
        from scripts.trading_lab.portfolio import (
            PORTFOLIO_LIMIT_V1_IS_NOT_OPTIMIZED, PORTFOLIO_SPEC_V1)

        spec = PORTFOLIO_SPEC_V1
        return {
            "protocol": spec.protocol_version,
            "portfolio_spec_hash": spec.portfolio_spec_hash,
            "frozen": True,
            "optimized": not PORTFOLIO_LIMIT_V1_IS_NOT_OPTIMIZED,
            "base_currency": spec.base_currency,
            "initial_equity": str(spec.initial_equity),
            "max_instrument_abs_exposure": str(spec.max_instrument_abs_exposure),
            "max_gross_exposure": str(spec.max_gross_exposure),
            "max_net_abs_exposure": str(spec.max_net_abs_exposure),
            "allocation_rule": spec.allocation_rule,
            "simultaneous_rebalance_rule": spec.simultaneous_rebalance_rule,
            "cash_model": spec.cash_model,
            "short_model": spec.short_model,
            "gross_cap_rationale":
                "two instruments at RiskSpec V1's 25 % each; the structure the "
                "existing rules already permit, not a fitted optimum",
        }

    def portfolio(self) -> dict:
        """The engine's contract, and whether any run exists yet."""
        manifest = self._portfolio_manifest()
        return {
            "api_version": APP_API_VERSION,
            "available": manifest is not None,
            "reason": None if manifest else "no portfolio backtest has been run",
            "portfolio": self._portfolio_contract(),
            "instruments": list(SUPPORTED_PRODUCTS),
            "shared_capital": True,
            "real_money": False,
            "broker_connected": False,
            "commercial_edge_established": False,
        }

    def portfolio_backtests(self) -> dict:
        manifest = self._portfolio_manifest()
        if manifest is None:
            return {"api_version": APP_API_VERSION, "available": False,
                    "reason": "no portfolio backtest has been run",
                    "portfolio": self._portfolio_contract(), "runs": []}
        return {
            "api_version": APP_API_VERSION,
            "available": True,
            "reason": None,
            "portfolio": self._portfolio_contract(),
            "runs": [{
                "version": "v1",
                "protocol": manifest["protocol_version"],
                "experiment_type": manifest["experiment_type"],
                "confirmatory": manifest["confirmatory"],
                "instruments": manifest["instruments"],
                "result_hash": manifest["result_hash"],
                "portfolio_backtest_spec_hash":
                    manifest["portfolio_backtest_spec_hash"],
            }],
        }

    def portfolio_backtest_detail(self, version: str) -> dict:
        manifest, result = self._portfolio_result(version)
        return {
            "api_version": APP_API_VERSION,
            "version": version,
            "available": True,
            "portfolio": self._portfolio_contract(),
            "instruments": manifest["instruments"],
            "experiment_type": result["experiment_type"],
            "confirmatory": result["confirmatory"],
            "live_execution": result["live_execution"],
            "cost_model": result["cost_model"],
            "commercial_edge_established": False,
            "metrics": result["metrics"],
            "gross_metrics": result["gross_metrics"],
            "result_hash": manifest["result_hash"],
            "source": manifest.get("source", {}),
            "alignment": manifest.get("alignment", {}),
            "equity_points": len(result["equity_curve"]),
            "fill_count": len(result["fills"]),
        }

    def portfolio_equity(self, version: str, *, max_points=None) -> dict:
        """Downsampled by buckets that keep their own extrema, never averaged.

        A mean would erase the trough of a drawdown, which is the one point on
        an equity curve nobody may hide.
        """
        manifest, result = self._portfolio_result(version)
        points = require_limit(max_points, default=DEFAULT_EQUITY_POINTS,
                               maximum=MAX_EQUITY_POINTS)
        curve = result["equity_curve"]
        if len(curve) <= points:
            kept, aggregation = curve, "none"
        else:
            # The same bucket-extrema rule the single-product curve uses: an
            # average would smooth away the trough of a drawdown, which is the
            # one point a reader is looking for.
            buckets = max(1, len(curve) // max(1, points // 4))
            kept, aggregation = [], "bucket-extrema"
            for start in range(0, len(curve), buckets):
                chunk = curve[start:start + buckets]
                lowest = min(chunk, key=lambda row: Decimal(row["equity"]))
                highest = max(chunk, key=lambda row: Decimal(row["equity"]))
                keep = {id(chunk[0]): chunk[0], id(lowest): lowest,
                        id(highest): highest, id(chunk[-1]): chunk[-1]}
                kept.extend(sorted(keep.values(), key=lambda row: row["timestamp"]))
        return {
            "api_version": APP_API_VERSION,
            "version": version,
            "series": [{"timestamp": row["timestamp"], "equity": row["equity"],
                        "gross_exposure": row["gross_exposure"],
                        "net_exposure": row["net_exposure"]} for row in kept],
            "metadata": {
                "source_count": len(curve), "returned_count": len(kept),
                "max_points": points, "aggregation": aggregation,
                "aggregated": aggregation != "none",
                "initial_equity": result["metrics"]["initial_equity"],
            },
        }

    def portfolio_fills(self, version: str, *, limit=None, cursor=None) -> dict:
        manifest, result = self._portfolio_result(version)
        size = require_limit(limit, default=DEFAULT_FILL_PAGE, maximum=MAX_FILL_PAGE)
        fills = result["fills"]
        start = 0
        query = {"limit": size}
        if cursor:
            after = decode_cursor(cursor, endpoint="portfolio_fills",
                                  product=version, query=query)
            start = next((index for index, fill in enumerate(fills)
                          if fill["timestamp"] > after), len(fills))
        page = fills[start:start + size]
        has_more = start + size < len(fills)
        return {
            "api_version": APP_API_VERSION,
            "version": version,
            "fills": page,
            "page": {
                "returned": len(page), "total": len(fills), "has_more": has_more,
                "next_cursor": encode_cursor(
                    endpoint="portfolio_fills", product=version,
                    last_timestamp=page[-1]["timestamp"], query=query)
                if has_more and page else None,
            },
        }

    def portfolio_attribution(self, version: str) -> dict:
        """Per-instrument contribution. Bounded by the registered instruments."""
        manifest, result = self._portfolio_result(version)
        return {
            "api_version": APP_API_VERSION,
            "version": version,
            "attribution": result["attribution"],
            "reconciliation": manifest.get("reconciliation", {}),
            "metrics": {
                "net_pnl": result["metrics"]["net_pnl"],
                "gross_pnl": result["metrics"]["gross_pnl"],
                "total_execution_cost": result["metrics"]["total_execution_cost"],
            },
        }

    # --- shared paper portfolio (read-only) ------------------------------

    def _portfolio_store(self):
        from scripts.trading_lab.paper_portfolio_store import PaperPortfolioStore

        layout = self._layout()
        if not layout.paper_portfolio_database.is_file():
            return None
        return PaperPortfolioStore(layout.paper_portfolio_database)

    def _portfolio_session(self):
        layout = self._layout()
        marker = layout.paper_portfolio_session_marker
        if not marker.is_file():
            return None
        try:
            return json.loads(marker.read_text())
        except (OSError, ValueError):
            return None

    def _latest_portfolio_session(self):
        store = self._portfolio_store()
        if store is None:
            return None, None
        sessions = store.sessions()
        return (store, sessions[-1]) if sessions else (store, None)

    def _portfolio_runtime_contract(self) -> dict:
        from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
        from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1
        from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1

        execution = PAPER_EXECUTION_SPEC_V1
        return {
            "mode": "SHARED_PORTFOLIO",
            "shadow_mode": True,
            "shared_capital": True,
            "real_money": False,
            "broker_connected": False,
            "commercial_edge_established": False,
            "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
            "initial_equity": str(PORTFOLIO_SPEC_V1.initial_equity),
            "max_instrument_abs_exposure":
                str(PORTFOLIO_SPEC_V1.max_instrument_abs_exposure),
            "max_gross_exposure": str(PORTFOLIO_SPEC_V1.max_gross_exposure),
            "allocation_rule": PORTFOLIO_SPEC_V1.allocation_rule,
            "simultaneous_rebalance_rule":
                PORTFOLIO_SPEC_V1.simultaneous_rebalance_rule,
            "paper_execution": {
                "spec_hash": execution.paper_execution_spec_hash,
                "fill_price_policy": execution.fill_price_policy,
                "fill_observation_policy": execution.fill_observation_policy,
            },
            "protected_holdout": {
                "holdout_id": PROTECTED_WINDOW_V1.holdout_id,
                "start": PROTECTED_WINDOW_V1.start,
                "end": PROTECTED_WINDOW_V1.end,
                "holdout_hash": PROTECTED_WINDOW_V1.holdout_hash,
                "observed": PROTECTED_WINDOW_V1.observed,
            },
        }

    def paper_portfolio(self, *, now=None) -> dict:
        """The shared portfolio's status. Never mixed with the legacy accounts."""
        from scripts.trading_lab.protected_holdout import embargo_state

        moment = now or _iso(datetime.now(timezone.utc))
        store, session_id = self._latest_portfolio_session()
        session = self._portfolio_session()
        instruments = self._portfolio_instruments()
        payload = dict(self._portfolio_runtime_contract())
        payload.update({
            "api_version": APP_API_VERSION,
            "available": session_id is not None,
            "reason": None if session_id else "no shared portfolio session recorded",
            "active_session": session,
            "session_id": session_id,
            "instruments": instruments,
            "embargo": {name: embargo_state(name, now=moment)
                        for name in instruments},
        })
        if store is None or session_id is None:
            return payload
        payload["events"] = store.count(session_id=session_id)
        try:
            payload["chain"] = store.verify_chain(session_id=session_id)
        except Exception as error:
            payload["chain"] = {"verified": False, "error": str(error)}
        snapshot = store.latest_snapshot(session_id=session_id)
        payload["snapshot_verified"] = snapshot is not None
        state = (snapshot or {}).get("state", {})
        payload["state"] = state.get("state")
        payload["pending_batches"] = state.get("pending", {})
        payload["fill_count"] = state.get("fill_count", 0)
        payload["rebalance_count"] = state.get("rebalance_count", 0)
        return payload

    def _portfolio_instruments(self) -> list:
        from scripts.trading_lab.instrument_registry import INSTRUMENTS_V1

        return [spec.canonical_id for spec in INSTRUMENTS_V1.all()]

    def paper_portfolio_positions(self) -> dict:
        payload = self.paper_portfolio()
        state = payload.get("state") or {}
        return {
            "api_version": APP_API_VERSION,
            "available": payload["available"],
            "instruments": payload["instruments"],
            "cash": state.get("cash"),
            "equity": state.get("equity"),
            "gross_exposure": state.get("gross_exposure"),
            "net_exposure": state.get("net_exposure"),
            "positions": state.get("positions", []),
        }

    def paper_portfolio_pending(self) -> dict:
        """What the runtime is waiting for. Never implies a trade happened."""
        payload = self.paper_portfolio()
        return {
            "api_version": APP_API_VERSION,
            "available": payload["available"],
            "status": "WAITING_FOR_PORTFOLIO_BATCH"
            if payload.get("pending_batches") else "IDLE",
            "pending_batches": payload.get("pending_batches", {}),
        }

    def paper_portfolio_events(self, *, limit=None, after_event_id=None) -> dict:
        store, session_id = self._latest_portfolio_session()
        size = require_limit(limit, default=DEFAULT_PAPER_EVENTS,
                             maximum=MAX_PAPER_EVENTS)
        if store is None or session_id is None:
            return {"api_version": APP_API_VERSION, "available": False,
                    "events": [], "page": {"returned": 0, "last_event_id": None}}
        if after_event_id is None:
            events = store.latest_events(session_id=session_id, limit=size)
        else:
            events = store.events(session_id=session_id,
                                  after_event_id=int(after_event_id), limit=size)
        return {
            "api_version": APP_API_VERSION,
            "available": True,
            "events": [{
                "event_id": event.event_id, "sequence": event.sequence,
                "event_type": event.event_type, "event_at": event.event_at,
                "instrument_id": event.instrument_id,
                "natural_key": event.natural_key, "payload": event.payload,
                "event_hash": event.event_hash,
            } for event in events],
            "page": {"returned": len(events),
                     "last_event_id": events[-1].event_id if events else None},
        }

    def paper_portfolio_fills(self, *, limit=None) -> dict:
        store, session_id = self._latest_portfolio_session()
        size = require_limit(limit, default=DEFAULT_FILL_PAGE,
                             maximum=MAX_FILL_PAGE)
        if store is None or session_id is None:
            return {"api_version": APP_API_VERSION, "available": False,
                    "fills": [], "page": {"returned": 0}}
        events = store.latest_events(session_id=session_id, limit=MAX_PAPER_EVENTS)
        fills = [event.payload for event in events
                 if event.event_type == "PORTFOLIO_FILL"][-size:]
        return {"api_version": APP_API_VERSION, "available": True,
                "fills": fills, "page": {"returned": len(fills)}}

    def paper_portfolio_equity(self, *, max_points=None) -> dict:
        """Equity from the recorded portfolio snapshots. Bounded."""
        store, session_id = self._latest_portfolio_session()
        points = require_limit(max_points, default=DEFAULT_PAPER_EQUITY_POINTS,
                               maximum=MAX_PAPER_EQUITY_POINTS)
        if store is None or session_id is None:
            return {"api_version": APP_API_VERSION, "available": False,
                    "series": [], "metadata": {"source_count": 0}}
        events = store.latest_events(session_id=session_id, limit=MAX_PAPER_EVENTS)
        series = [{
            "timestamp": event.payload.get("timestamp") or event.event_at,
            "equity": event.payload.get("equity"),
            "cash": event.payload.get("cash"),
            "gross_exposure": event.payload.get("gross_exposure"),
            "net_exposure": event.payload.get("net_exposure"),
        } for event in events if event.event_type == "PORTFOLIO_SNAPSHOT"]
        kept = series[-points:]
        return {"api_version": APP_API_VERSION, "available": True,
                "series": kept,
                "metadata": {"source_count": len(series),
                             "returned_count": len(kept), "max_points": points}}

    def paper_legacy(self) -> dict:
        """The pre-shared-portfolio per-product sessions. Never summed with the
        shared portfolio: two independent accounts are not its history."""
        session = self._paper_session()
        store = self._paper_store()
        return {
            "api_version": APP_API_VERSION,
            "available": store is not None,
            "label": "PRE-SHARED-PORTFOLIO",
            "shared_capital": False,
            "note": "independent per-product accounts; their equity is not the "
                    "history of the shared portfolio and is never added to it",
            "session": session,
            "sessions": len(store.sessions()) if store is not None else 0,
            "events": store.count() if store is not None else 0,
        }
