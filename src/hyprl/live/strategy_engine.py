from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

from hyprl.adaptive.engine import AdaptiveState
from hyprl.backtest.runner import BacktestConfig
from hyprl.parity.signal_trace import ParityTraceHandle, attach_parity_trace
from hyprl.strategy import decide_signals_on_bar, initial_regime_name, prepare_feature_frame
from hyprl.live.types import Bar, Position, TradeSignal


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@dataclass(slots=True)
class StrategyState:
    """Mutable buffers shared between live bars."""

    prices: pd.DataFrame = field(default_factory=_empty_price_frame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)


class StrategyEngine:
    """Shared HyprL strategy entrypoint for live/paper execution."""

    def __init__(
        self,
        config: BacktestConfig,
        trace_callback: Callable[[dict[str, object]], None] | None = None,
        precomputed_features: pd.DataFrame | None = None,
        parity_equity: dict[pd.Timestamp, float] | None = None,
    ) -> None:
        self.config = config
        self.state = StrategyState()
        self._min_history = max(config.sma_long_window, config.rsi_window, config.atr_window) + 5
        self._adaptive_cfg = config.adaptive
        self._adaptive_state = AdaptiveState(
            regime_name=initial_regime_name(self._adaptive_cfg, config.risk_profile)
        )
        self._last_equity = float(config.initial_balance)
        self._parity_trace: ParityTraceHandle | None = None
        wrapped_callback, parity_handle = attach_parity_trace(config.ticker, "replay", trace_callback)
        self._trace_callback = wrapped_callback
        self._parity_trace = parity_handle
        self._feature_cache = precomputed_features
        self._parity_equity = parity_equity

    def warmup(self, bars: Iterable[Bar]) -> None:
        records = list(bars)
        if not records:
            return
        self.state.prices = self._bars_to_frame(records)
        self._refresh_features()

    def _bars_to_frame(self, bars: Iterable[Bar]) -> pd.DataFrame:
        rows: list[dict[str, float | pd.Timestamp]] = []
        for bar in bars:
            rows.append(
                {
                    "timestamp": pd.Timestamp(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
        if not rows:
            return _empty_price_frame()
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.set_index("timestamp").sort_index()
        return df

    def _append_bar(self, bar: Bar) -> None:
        bar_df = self._bars_to_frame([bar])
        if self.state.prices.empty:
            self.state.prices = bar_df
            return
        combined = pd.concat([self.state.prices, bar_df])
        combined = combined[~combined.index.duplicated(keep="last")]  # latest tick wins
        self.state.prices = combined.sort_index()

    def _refresh_features(self) -> None:
        if self.state.prices.empty:
            self.state.features = pd.DataFrame()
            return
        if self._feature_cache is not None:
            visible_index = self.state.prices.index
            available = self._feature_cache.index.intersection(visible_index)
            if available.empty:
                self.state.features = pd.DataFrame()
                return
            self.state.features = self._feature_cache.loc[available].sort_index()
            return
        try:
            features = prepare_feature_frame(self.state.prices, self.config)
        except (RuntimeError, ValueError):
            features = pd.DataFrame()
        self.state.features = features

    def _resolve_equity(self, equity: float | None) -> float:
        if equity is None or not float(equity) > 0:
            return self._last_equity
        self._last_equity = float(equity)
        return self._last_equity

    def on_bar(
        self,
        bar: Bar,
        open_positions: list[Position],
        equity: float | None = None,
        risk_pct_override: float | None = None,
    ) -> list[TradeSignal]:
        symbol = bar.symbol.upper()
        if symbol != self.config.ticker.upper():
            return []

        self._append_bar(bar)
        self._refresh_features()
        if self.state.features.empty:
            return []

        ts = pd.Timestamp(bar.timestamp)
        if ts not in self.state.features.index:
            return []

        try:
            current_idx = int(self.state.features.index.get_loc(ts))
        except KeyError:
            return []

        if current_idx < self._min_history:
            return []

        if any(pos.symbol.upper() == symbol for pos in open_positions):
            return []

        parity_equity = None
        if self._parity_equity is not None:
            parity_equity = self._parity_equity.get(ts)
        equity_input = parity_equity if parity_equity is not None else equity
        equity_value = self._resolve_equity(equity_input)
        decision = decide_signals_on_bar(
            config=self.config,
            features=self.state.features,
            prices=self.state.prices,
            current_idx=current_idx,
            equity=equity_value,
            adaptive_cfg=self._adaptive_cfg,
            adaptive_state=self._adaptive_state,
            trace_log=self._trace_callback,
            risk_pct_override=risk_pct_override,
        )
        if decision is None:
            return []

        size = float(decision.risk_plan.position_size)
        if size <= 0:
            return []

        if getattr(self.config, "guards_config", None):
            # If guards are supplied, we rely on caller to gate via risk_pct_override or skip.
            pass

        reason = f"strategy:{decision.direction}:{decision.threshold:.3f}"
        signal = TradeSignal(
            symbol=bar.symbol,
            side=decision.direction,
            size=size,
            reason=reason,
            timestamp=bar.timestamp,
            probability_up=decision.probability_up,
            threshold=decision.threshold,
            entry_price=decision.entry_price,
            expected_pnl=decision.expected_pnl,
            risk_amount=float(decision.risk_plan.risk_amount),
            long_threshold=decision.long_threshold,
            short_threshold=decision.short_threshold,
            stop_price=float(decision.risk_plan.stop_price),
            take_profit_price=float(decision.risk_plan.take_profit_price),
            trailing_stop_activation_price=decision.risk_plan.trailing_stop_activation_price,
            trailing_stop_distance_price=decision.risk_plan.trailing_stop_distance_price,
            risk_profile=decision.profile_name,
            regime_name=decision.regime_name,
        )
        return [signal]

    def close(self) -> None:
        if self._parity_trace is not None:
            self._parity_trace.close()
            self._parity_trace = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()
