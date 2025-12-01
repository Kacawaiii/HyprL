from __future__ import annotations

from typing import Protocol

import pandas as pd
import yfinance as yf

from hyprl.live.types import Bar


class MarketDataSource(Protocol):
    """Protocol describing the minimum interface for live/paper market data feeds."""

    def get_history(self, symbol: str, interval: str, lookback: str) -> list[Bar]:
        """Return a chronological list of bars for warm-up/history."""

    def get_latest_bar(self, symbol: str, interval: str) -> Bar:
        """Fetch the most recent completed bar for the symbol/interval."""


class YFinanceSource:
    """Cheap development-grade market data source backed by yfinance."""

    def _df_to_bars(self, symbol: str, df: pd.DataFrame) -> list[Bar]:
        bars: list[Bar] = []
        for ts, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )
        if not bars:
            raise RuntimeError(f"No bars returned for {symbol}")
        return bars

    def get_history(self, symbol: str, interval: str, lookback: str) -> list[Bar]:
        df = yf.download(
            symbol,
            period=lookback,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"No history for {symbol} ({interval}, {lookback})")
        return self._df_to_bars(symbol, df)

    def get_latest_bar(self, symbol: str, interval: str) -> Bar:
        df = yf.download(
            symbol,
            period="2d",
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"No latest bar for {symbol}")
        return self._df_to_bars(symbol, df)[-1]
