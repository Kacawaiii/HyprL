"""Realtime paper trading engine with feature-based predictions."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Literal

import joblib
import pandas as pd

from hyprl.meta.features import build_feature_frame_from_records, select_meta_features
from hyprl.meta.model import MetaRobustnessModel
from hyprl.meta.registry import resolve_model
from hyprl.model.probability import ProbabilityModel
from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.features import compute_features
from hyprl.rt.logging import LiveLogger
from hyprl.rt.risk import size_position
from hyprl.rt.tuner import Tuner


INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


def _bucket(ts: float, interval_s: int) -> int:
    return int(ts // interval_s) * interval_s


class _BarBuilder:
    def __init__(self, interval: Literal["1m", "5m", "15m", "1h"]):
        self.interval_seconds = INTERVAL_SECONDS[interval]
        self.current: dict[str, dict] = {}

    def add(self, event: MarketEvent) -> list[dict]:
        completed: list[dict] = []
        if event.get("symbol") is None or event.get("price") is None:
            return completed
        ts = float(event["ts"])
        symbol = event["symbol"].upper()
        bucket = _bucket(ts, self.interval_seconds)
        bar = self.current.get(symbol)
        if bar is None or bar["bucket"] != bucket:
            if bar is not None:
                completed.append(bar)
            bar = {
                "symbol": symbol,
                "bucket": bucket,
                "open": float(event["price"]),
                "high": float(event["price"]),
                "low": float(event["price"]),
                "close": float(event["price"]),
                "volume": float(event.get("volume") or 0.0),
            }
            self.current[symbol] = bar
        else:
            price = float(event["price"])
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += float(event.get("volume") or 0.0)
        return completed

    def flush(self) -> list[dict]:  # pragma: no cover - shutdown
        bars = list(self.current.values())
        self.current.clear()
        return bars


@dataclass(slots=True)
class LiveConfig:
    symbols: list[str]
    interval: Literal["1m", "5m", "15m", "1h"] = "1m"
    threshold: float = 0.52
    risk_pct: float = 0.25
    weighting_scheme: Literal["equal", "inv_vol"] = "equal"
    warmup_bars: int = 60
    timezone: str = "Europe/Paris"
    enable_paper: bool = False
    session_id: str | None = None
    initial_balance: float = 10_000.0
    meta_model_path: str | None = None
    meta_calibration_path: str | None = None
    max_orders_per_min: int = 10
    per_symbol_cap: int = 3
    min_qty: int = 1
    max_qty: int | None = None
    kill_switch_dd: float | None = 0.30
    resume_session: str | None = None
    tuner: Tuner | None = None


def _load_meta_models(cfg: LiveConfig) -> tuple[MetaRobustnessModel | None, object | None]:
    if not cfg.meta_model_path:
        return None, None
    model = MetaRobustnessModel.load(cfg.meta_model_path)
    calibrator = None
    if cfg.meta_calibration_path:
        calibrator = joblib.load(cfg.meta_calibration_path)
    return model, calibrator


def _resolve_path(path: str | None, registry: str | None) -> str | None:
    if path:
        return path
    if registry:
        return str(resolve_model(registry))
    return None


def _log_reason(logger: LiveLogger, base: dict, reason: str) -> None:
    payload = base.copy()
    payload["reason"] = reason
    logger.log_prediction(payload)


def _scan_jsonl_max_float(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    max_value: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = obj.get(key)
            if isinstance(val, (int, float)):
                float_val = float(val)
                max_value = float_val if max_value is None else max(max_value, float_val)
    return max_value


def _load_resume_state(session_dir: Path, manifest: dict | None = None) -> tuple[float | None, float]:
    if manifest:
        last_bar_ts = manifest.get("last_bar_ts")
        equity_peak = manifest.get("equity_peak")
        if last_bar_ts is not None or equity_peak is not None:
            return last_bar_ts, float(equity_peak or 0.0)
    last_bar_ts = _scan_jsonl_max_float(session_dir / "bars.jsonl", key="ts")
    if last_bar_ts is None:
        last_bar_ts = _scan_jsonl_max_float(session_dir / "bars.jsonl", key="bucket")
    equity_peak = _scan_jsonl_max_float(session_dir / "equity.jsonl", key="equity")
    return last_bar_ts, float(equity_peak) if equity_peak is not None else 0.0


def _log_resume_event(logger: LiveLogger, session_id: str, last_bar_ts: float | None) -> None:
    payload = {"event": "resume", "session": session_id, "last_bar_ts": last_bar_ts}
    logger.log_prediction(symbol="*", prob_up=None, threshold=None, direction=None, extra=payload)


async def run_realtime_paper(
    src: MarketSource,
    broker: PaperBroker,
    live_cfg: LiveConfig,
    logger: LiveLogger,
    meta_registry: str | None = None,
    meta_cal_registry: str | None = None,
) -> None:
    if live_cfg.interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval {live_cfg.interval}")
    if live_cfg.session_id:
        logger._manifest["session_id"] = live_cfg.session_id  # type: ignore[attr-defined]

    builder = _BarBuilder(live_cfg.interval)
    bar_frames: dict[str, pd.DataFrame] = defaultdict(lambda: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    features_history: dict[str, list[dict]] = defaultdict(list)
    prob_models: dict[str, ProbabilityModel] = defaultdict(lambda: ProbabilityModel.create())
    order_times: Deque[float] = deque()  # runtime counters reset on every start/resume
    symbol_order_times: dict[str, Deque[float]] = defaultdict(deque)
    equity = live_cfg.initial_balance
    positions: dict[str, float] = defaultdict(float)
    last_price: dict[str, float] = defaultdict(float)
    equity_peak = equity
    kill_threshold = live_cfg.kill_switch_dd
    session_dir_attr = getattr(logger, "session_dir", getattr(logger, "root", Path("data/live/sessions")))
    session_dir = Path(session_dir_attr)
    resume_last_bar_ts: float | None = None
    equity_peak_resume = 0.0
    manifest_snapshot = getattr(logger, "manifest", {})
    if live_cfg.resume_session:
        resume_last_bar_ts, equity_peak_resume = _load_resume_state(session_dir, manifest_snapshot)
        _log_resume_event(logger, live_cfg.resume_session, resume_last_bar_ts)
        logger.update_manifest({"resumed_from": live_cfg.resume_session})
    else:
        logger.update_manifest({"resumed_from": None})
    equity_peak = max(equity_peak, equity_peak_resume)
    kill_manifest = manifest_snapshot.get("killswitch") or {}
    logger.update_manifest({
        "killswitch": {
            "enabled": kill_threshold is not None,
            "dd_limit": kill_threshold,
            "triggered": bool(kill_manifest.get("triggered", False)),
            "triggered_at_ts": kill_manifest.get("triggered_at_ts"),
            "dd_at_trigger": kill_manifest.get("dd_at_trigger"),
        }
    })
    src.subscribe(live_cfg.symbols)

    simulated_brackets: dict[str, dict] = {}
    tuner = live_cfg.tuner
    symbol_bar_counts: dict[str, int] = defaultdict(int)
    pending_signals: dict[str, Deque[dict]] = defaultdict(deque)
    rolling_results: Deque[int] = deque(maxlen=50)
    equity_window: Deque[tuple[int, float]] = deque(maxlen=64)
    global_bar_index = 0
    current_drawdown = 0.0
    kill_already_triggered = bool(kill_manifest.get("triggered", False))

    meta_model_path = _resolve_path(live_cfg.meta_model_path, meta_registry)
    meta_cal_path = _resolve_path(live_cfg.meta_calibration_path, meta_cal_registry)
    if meta_model_path:
        live_cfg.meta_model_path = meta_model_path
    if meta_cal_path:
        live_cfg.meta_calibration_path = meta_cal_path
    meta_model, meta_calibrator = _load_meta_models(live_cfg)

    async for event in src.aiter():
        logger.log_event(event)
        completed = builder.add(event)
        for bar in completed:
            bar_ts = float(bar.get("ts") or bar["bucket"])
            bar["ts"] = bar_ts
            if resume_last_bar_ts is not None and bar_ts <= resume_last_bar_ts:
                continue
            ts = pd.to_datetime(bar["bucket"], unit="s")
            global_bar_index += 1
            symbol = bar["symbol"]
            symbol_bar_counts[symbol] += 1
            bar_df = bar_frames[symbol]
            row = pd.DataFrame(
                {
                    "open": [bar["open"]],
                    "high": [bar["high"]],
                    "low": [bar["low"]],
                    "close": [bar["close"]],
                    "volume": [bar["volume"]],
                },
                index=[ts],
            )
            bar_df = pd.concat([bar_df, row])
            if len(bar_df) > 600:
                bar_df = bar_df.tail(600)
            bar_frames[symbol] = bar_df
            logger.log_bar(dict(bar))
            prev_price = last_price.get(symbol)
            if prev_price:
                equity += positions[symbol] * (bar["close"] - prev_price)
            last_price[symbol] = bar["close"]

            async def finalize_bar() -> bool:
                nonlocal equity_peak, current_drawdown, kill_already_triggered
                equity_peak = max(equity_peak, equity)
                if kill_threshold is not None and equity_peak > 0:
                    drawdown = 1.0 - (equity / equity_peak)
                    current_drawdown = float(drawdown)
                    if drawdown >= kill_threshold and not kill_already_triggered:
                        kill_already_triggered = True
                        payload = {
                            "ts": ts.timestamp(),
                            "symbol": symbol,
                            "event": "kill_switch",
                            "dd": float(drawdown),
                        }
                        logger.log_prediction(payload)
                        logger.update_manifest({
                            "killswitch": {
                                "triggered": True,
                                "triggered_at_ts": ts.timestamp(),
                                "dd_at_trigger": float(drawdown),
                            },
                            "last_bar_ts": bar_ts,
                            "equity_peak": equity_peak,
                        })
                        try:
                            await broker.cancel_all()
                        except Exception:  # pragma: no cover - safety
                            pass
                        return True
                else:
                    current_drawdown = 0.0

                equity_window.append((global_bar_index, equity))
                winrate_metric = (sum(rolling_results) / len(rolling_results)) if rolling_results else None
                slope_metric = None
                if len(equity_window) >= 2:
                    first_idx, first_eq = equity_window[0]
                    last_idx, last_eq = equity_window[-1]
                    denom = max(last_idx - first_idx, 1)
                    slope_metric = (last_eq - first_eq) / denom
                metrics_payload = {
                    "winrate_rolling": winrate_metric,
                    "dd_session": current_drawdown if kill_threshold is not None else None,
                    "equity_slope": slope_metric,
                }
                if tuner:
                    delta = tuner.update(global_bar_index, metrics_payload)
                    if delta:
                        before_thr = live_cfg.threshold
                        before_risk = live_cfg.risk_pct
                        if "threshold" in delta:
                            live_cfg.threshold = delta["threshold"]
                        if "risk_pct" in delta:
                            live_cfg.risk_pct = delta["risk_pct"]
                        after_state = {"threshold": live_cfg.threshold, "risk_pct": live_cfg.risk_pct}
                        delta_payload = {}
                        if "threshold" in delta:
                            delta_payload["threshold"] = after_state["threshold"] - before_thr
                        if "risk_pct" in delta:
                            delta_payload["risk_pct"] = after_state["risk_pct"] - before_risk
                        logger.log_prediction(
                            {
                                "ts": ts.timestamp(),
                                "symbol": symbol,
                                "event": "tuner",
                                "delta": delta_payload,
                                "after": after_state,
                            }
                        )
                logger.update_manifest({"last_bar_ts": bar_ts, "equity_peak": equity_peak})
                return False

            if simulated_brackets:
                for oco_id, state in list(simulated_brackets.items()):
                    if not state.get("active", True):
                        simulated_brackets.pop(oco_id, None)
                        continue
                    if state["symbol"] != symbol:
                        continue
                    if bar_ts <= state["open_ts"]:
                        continue
                    tp_hit = False
                    sl_hit = False
                    if state["direction"] == 1:
                        if state["tp"] is not None:
                            tp_hit = bar["high"] >= state["tp"]
                        if state["sl"] is not None:
                            sl_hit = bar["low"] <= state["sl"]
                    else:
                        if state["tp"] is not None:
                            tp_hit = bar["low"] <= state["tp"]
                        if state["sl"] is not None:
                            sl_hit = bar["high"] >= state["sl"]
                    reason = None
                    if tp_hit and sl_hit:
                        reason = "tp"
                    elif tp_hit:
                        reason = "tp"
                    elif sl_hit:
                        reason = "sl"
                    if not reason:
                        continue
                    price_hit = state["tp"] if reason == "tp" else state["sl"]
                    if price_hit is None:
                        continue
                    close_qty = -state["qty"] * state["direction"]
                    pnl = (price_hit - state["entry_price"]) * state["qty"] * state["direction"]
                    equity += pnl
                    positions[symbol] += close_qty
                    last_price[symbol] = price_hit
                    logger.log_fill(
                        {
                            "symbol": symbol,
                            "qty": close_qty,
                            "price": price_hit,
                            "assumed": True,
                            "oco_group_id": oco_id,
                            "reason": reason,
                        }
                    )
                    logger.log_equity({"ts": ts.timestamp(), "equity": equity, "symbol": symbol})
                    logger.log_prediction(
                        {
                            "ts": ts.timestamp(),
                            "symbol": symbol,
                            "event": "oco_close",
                            "reason": reason,
                            "oco_group_id": oco_id,
                        }
                    )
                    simulated_brackets.pop(oco_id, None)

            feats = compute_features(bar_df)
            if not feats:
                _log_reason(logger, {"symbol": symbol, "prob_up": None, "threshold": live_cfg.threshold}, "warmup")
                if await finalize_bar():
                    return
                continue
            logger.log_features({"symbol": symbol, **feats})
            features_history[symbol].append(feats)
            if len(features_history[symbol]) > 600:
                features_history[symbol] = features_history[symbol][-600:]
            features_df = pd.DataFrame(features_history[symbol]).dropna()
            pending_queue = pending_signals[symbol]
            while pending_queue and pending_queue[0]["bar_idx"] < symbol_bar_counts[symbol]:
                signal = pending_queue.popleft()
                change = bar["close"] - signal["price"]
                win = (signal["direction"] == 1 and change >= 0) or (signal["direction"] == -1 and change <= 0)
                rolling_results.append(1 if win else 0)
            if len(features_df) < live_cfg.warmup_bars:
                payload = feats | {"symbol": symbol, "threshold": live_cfg.threshold}
                _log_reason(logger, payload, "warmup")
                if await finalize_bar():
                    return
                continue
            feature_cols = [col for col in features_df.columns if col not in {"close", "open"}]
            design = features_df[feature_cols]
            target = (features_df["close"] >= features_df["open"]).astype(int)
            model = prob_models[symbol]
            model.fit(design, target)
            prob_up = float(model.predict_proba(design.tail(1))[-1])

            meta_pred = None
            if meta_model is not None:
                meta_record = {
                    "long_threshold": live_cfg.threshold,
                    "short_threshold": max(live_cfg.threshold - 0.1, 0.0),
                    "risk_pct": live_cfg.risk_pct,
                    "min_ev_multiple": 0.0,
                    "trend_filter": False,
                    "sentiment_min": -1.0,
                    "sentiment_max": 1.0,
                    "sentiment_regime": "off",
                    "weighting_scheme": live_cfg.weighting_scheme,
                    "pf_backtest": 1.0,
                    "sharpe_backtest": 0.0,
                    "maxdd_backtest": 0.0,
                    "winrate_backtest": 0.5,
                    "equity_vol_backtest": 0.0,
                    "trades_backtest": len(features_df),
                    "correlation_mean": 0.0,
                    "correlation_max": 0.0,
                }
                feature_frame = build_feature_frame_from_records([meta_record])
                matrix, _ = select_meta_features(feature_frame)
                # Be robust to models that don't accept 'calibrator' kwarg
                try:
                    meta_pred = float(meta_model.predict(matrix, calibrator=meta_calibrator)[0])
                except TypeError:
                    meta_pred = float(meta_model.predict(matrix)[0])

            direction = 1 if prob_up >= live_cfg.threshold else -1
            prediction_payload = {
                "ts": ts.timestamp(),
                "symbol": symbol,
                "prob_up": prob_up,
                "threshold": live_cfg.threshold,
                "direction": "UP" if direction == 1 else "DOWN",
                "atr": feats.get("atr"),
                "sma_short": feats.get("sma_short"),
                "sma_long": feats.get("sma_long"),
                "rsi_raw": feats.get("rsi_raw"),
                "bb_width": feats.get("bb_width"),
                "meta_pred": meta_pred,
            }

            if not all(pd.notna(feats.get(field)) for field in ["atr", "sma_short", "sma_long", "rsi_raw"]):
                _log_reason(logger, prediction_payload, "nan_features")
                if await finalize_bar():
                    return
                continue

            qty, stop_price, take_profit = size_position(
                equity=equity,
                risk_pct=live_cfg.risk_pct,
                atr=float(feats.get("atr", 0.0) or 0.0),
                price=float(feats.get("close", bar["close"])),
                min_qty=live_cfg.min_qty,
                max_qty=live_cfg.max_qty,
            )
            if direction == -1 and stop_price is not None and take_profit is not None:
                stop_price, take_profit = take_profit, stop_price
            prediction_payload.update({"qty": qty, "stop": stop_price, "tp": take_profit})
            if qty < live_cfg.min_qty or qty == 0:
                _log_reason(logger, prediction_payload, "qty_clamp")
                if await finalize_bar():
                    return
                continue

            now = time.time()
            while order_times and now - order_times[0] > 60:
                order_times.popleft()
            sym_queue = symbol_order_times[symbol]
            while sym_queue and now - sym_queue[0] > 60:
                sym_queue.popleft()
            if len(order_times) >= live_cfg.max_orders_per_min or len(sym_queue) >= live_cfg.per_symbol_cap:
                _log_reason(logger, prediction_payload, "rate_cap")
                if await finalize_bar():
                    return
                continue

            prediction_payload["reason"] = "signal"
            logger.log_prediction(prediction_payload)

            order_times.append(now)
            sym_queue.append(now)
            pending_signals[symbol].append({"bar_idx": symbol_bar_counts[symbol], "direction": direction, "price": bar["close"]})
            side = "buy" if direction == 1 else "sell"
            order_record = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price_ref": bar["close"],
                "paper": live_cfg.enable_paper,
                "client_id": f"hyprl-{int(now)}",
            }
            if live_cfg.enable_paper:
                try:
                    response = await broker.submit_order(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        type="market",
                        limit_price=None,
                        stop_price=stop_price,
                        take_profit=take_profit,
                        client_id=order_record["client_id"],
                    )
                    order_record["broker_response"] = response
                except Exception as exc:  # pragma: no cover - network path
                    order_record["error"] = str(exc)

                if stop_price is not None and take_profit is not None:
                    bracket_result = None
                    submit_bracket = getattr(broker, "submit_bracket", None)
                    if submit_bracket is not None:
                        try:
                            bracket_result = await submit_bracket(
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                take_profit=take_profit,
                                stop_loss=stop_price,
                                client_id=order_record["client_id"],
                            )
                        except Exception as exc:  # pragma: no cover - network
                            bracket_result = {
                                "simulated": True,
                                "oco_group_id": uuid.uuid4().hex,
                                "error": str(exc),
                            }
                    if bracket_result is None:
                        bracket_result = {"simulated": True, "oco_group_id": uuid.uuid4().hex}
                    bracket_result.setdefault("submitted", now)
                    bracket_result.setdefault("symbol", symbol)
                    bracket_result.setdefault("side", side)
                    bracket_result.setdefault("qty", qty)
                    bracket_result.setdefault("tp", take_profit)
                    bracket_result.setdefault("sl", stop_price)
                    order_record["oco_group_id"] = bracket_result["oco_group_id"]
                    order_record["bracket"] = bracket_result
                    if bracket_result.get("simulated"):
                        simulated_brackets[bracket_result["oco_group_id"]] = {
                            "symbol": symbol,
                            "side": side,
                            "direction": direction,
                            "qty": qty,
                            "tp": take_profit,
                            "sl": stop_price,
                            "entry_price": bar["close"],
                            "open_ts": bar_ts,
                            "active": True,
                        }
            logger.log_order(order_record)

            fill_record = {
                "symbol": symbol,
                "qty": qty * direction,
                "price": bar["close"],
                "assumed": True,
            }
            if order_record.get("oco_group_id"):
                fill_record["oco_group_id"] = order_record["oco_group_id"]
            logger.log_fill(fill_record)
            positions[symbol] += qty * direction
            last_price[symbol] = bar["close"]
            logger.log_equity({"ts": ts.timestamp(), "equity": equity, "symbol": symbol})
            if await finalize_bar():
                return

    for bar in builder.flush():  # graceful shutdown
        bar_ts = float(bar.get("ts") or bar["bucket"])
        if resume_last_bar_ts is not None and bar_ts <= resume_last_bar_ts:
            continue
        bar["ts"] = bar_ts
        logger.log_bar(dict(bar))
