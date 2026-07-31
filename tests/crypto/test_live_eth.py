from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.eth_paper.live_eth import (
    _client_order_id,
    _completed_close_series,
    _is_owned_terminal_order,
    _logged_order_ids,
    _portfolio_peak,
    _position_snapshot,
)


NOW = datetime(2026, 7, 31, 12, 34, tzinfo=timezone.utc)


def bar(timestamp: str, close: float) -> dict:
    return {"t": timestamp, "c": close}


def test_completed_daily_bars_exclude_current_partial_day() -> None:
    values = _completed_close_series(
        [
            bar("2026-07-29T00:00:00Z", 1_800),
            bar("2026-07-30T00:00:00Z", 1_900),
            bar("2026-07-31T00:00:00Z", 9_999),
        ],
        now=NOW,
        timeframe="1Day",
    )
    assert list(values) == [1_800, 1_900]
    assert values.index[-1] == pd.Timestamp("2026-07-30T00:00:00Z")


def test_completed_hourly_bars_exclude_current_partial_hour() -> None:
    values = _completed_close_series(
        [
            bar("2026-07-31T10:00:00Z", 1_800),
            bar("2026-07-31T11:00:00Z", 1_900),
            bar("2026-07-31T12:00:00Z", 9_999),
        ],
        now=NOW,
        timeframe="1Hour",
    )
    assert list(values) == [1_800, 1_900]


def test_dedicated_account_rejects_unexpected_positions_and_shorts() -> None:
    assert _position_snapshot(
        [{"symbol": "ETHUSD", "qty": "2.5", "market_value": "5000"}]
    ) == (2.5, 5000.0)

    with pytest.raises(Exception, match="unexpected positions"):
        _position_snapshot(
            [{"symbol": "BTCUSD", "qty": "1", "market_value": "100000"}]
        )
    with pytest.raises(Exception, match="short ETH"):
        _position_snapshot(
            [{"symbol": "ETH/USD", "qty": "-1", "market_value": "-2000"}]
        )


def test_portfolio_peak_ignores_null_values() -> None:
    assert _portfolio_peak(
        {"equity": [None, 100_000, "104000", None, 99_000]},
        102_000,
    ) == 104_000


def test_client_order_id_is_short_deterministic_and_variant_specific() -> None:
    first = _client_order_id("news", NOW, "buy", 0.2)
    second = _client_order_id("news", NOW, "buy", 0.2)
    control = _client_order_id("control", NOW, "buy", 0.2)
    assert first == second
    assert first != control
    assert len(first) <= 48


def test_only_owned_terminal_orders_are_reconciled() -> None:
    owned = {
        "symbol": "ETH/USD",
        "client_order_id": "hyprl-eth-n-20260731T12-b-1500",
        "status": "filled",
    }
    assert _is_owned_terminal_order(owned, "news")
    assert not _is_owned_terminal_order(owned, "control")
    assert not _is_owned_terminal_order({**owned, "symbol": "BTC/USD"}, "news")
    assert not _is_owned_terminal_order({**owned, "status": "new"}, "news")
    assert not _is_owned_terminal_order(
        {**owned, "client_order_id": "manual-order"},
        "news",
    )


def test_logged_order_ids_tolerates_invalid_and_legacy_lines(tmp_path) -> None:
    path = tmp_path / "orders.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"order":{"id":"first"}}',
                "not-json",
                '{"order":{"id":"second"}}',
                '{"legacy":true}',
            )
        ),
        encoding="utf-8",
    )
    assert _logged_order_ids(path) == {"first", "second"}
