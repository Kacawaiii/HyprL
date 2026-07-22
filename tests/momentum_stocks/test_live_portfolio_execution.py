from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    name = "hyprl_test_live_portfolio"
    path = ROOT / "scripts/momentum_stocks/live_portfolio.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lp():
    return load_module()


def test_open_short_uses_only_whole_shares(lp) -> None:
    order = lp.plan_order("GLD", desired_qty=-9.682, current_qty=0, price=375)

    assert order is not None
    assert order.intent == "open_short"
    assert order.side == "sell"
    assert order.delta_qty == -9
    assert order.target_qty == -9
    assert order.desired_qty == pytest.approx(-9.682)


def test_increase_short_adds_only_whole_shares(lp) -> None:
    order = lp.plan_order("FXY", desired_qty=-12.7, current_qty=-9, price=56)

    assert order is not None
    assert order.intent == "increase_short"
    assert order.delta_qty == -3
    assert order.target_qty == -12


def test_sub_share_short_drift_is_left_alone(lp) -> None:
    assert lp.plan_order("SLV", desired_qty=-30.4, current_qty=-30, price=53) is None


def test_side_reversal_flattens_before_opening_short(lp) -> None:
    order = lp.plan_order("GLD", desired_qty=-9.5, current_qty=2.75, price=375)

    assert order is not None
    assert order.intent == "close_long_before_short"
    assert order.side == "sell"
    assert order.delta_qty == pytest.approx(-2.75)
    assert order.target_qty == 0


def test_small_position_is_closed_even_below_rebalance_threshold(lp) -> None:
    order = lp.plan_order("TEST", desired_qty=0, current_qty=0.25, price=100)

    assert order is not None
    assert order.intent == "close_long"
    assert order.delta_qty == pytest.approx(-0.25)


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ({"tradable": True, "shortable": True, "borrow_status": "easy_to_borrow"}, None),
        (
            {"tradable": True, "shortable": True, "borrow_status": "hard_to_borrow"},
            "borrow_status=hard_to_borrow",
        ),
        ({"tradable": True, "shortable": False}, "not shortable"),
    ],
)
def test_shortability_gate(lp, asset: dict, expected: str | None) -> None:
    problem = lp.shortability_problem(asset)
    if expected is None:
        assert problem is None
    else:
        assert expected in problem


def test_reconcile_trend_state_uses_broker_as_source_of_truth(lp) -> None:
    state = {
        "trend": {
            "GLD": {"side": -1, "extreme": 370, "shares": 9.682},
            "VNQ": {"side": 1, "extreme": 100, "shares": 67.9},
            "FXY": {"side": -1, "extreme": 56, "shares": 376.1},
        }
    }
    current = {"VNQ": 67.5, "SLV": -30.0}
    data = {
        "VNQ": pd.DataFrame({"Close": [101.0]}),
        "SLV": pd.DataFrame({"Close": [53.0]}),
    }

    notes = lp.reconcile_trend_state(state, current, {"FXY"}, data)

    assert "GLD" not in state["trend"]
    assert state["trend"]["VNQ"]["shares"] == pytest.approx(67.5)
    assert state["trend"]["FXY"]["shares"] == pytest.approx(376.1)
    assert state["trend"]["SLV"] == {"side": -1, "extreme": 53.0, "shares": 30.0}
    assert any("GLD: removed stale" in note for note in notes)
    assert any("SLV: reconstructed" in note for note in notes)


def test_http_error_includes_alpaca_response_without_credentials(lp, monkeypatch) -> None:
    body = io.BytesIO(b'{"code":42210000,"message":"fractional short orders are not supported"}')
    error = urllib.error.HTTPError(
        "https://paper-api.alpaca.markets/v2/orders",
        422,
        "Unprocessable Entity",
        {},
        body,
    )
    monkeypatch.setattr(lp, "ALPACA_KEY", "secret-key-value")
    monkeypatch.setattr(lp, "ALPACA_SECRET", "secret-value")
    def raise_http_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(lp.urllib.request, "urlopen", raise_http_error)

    with pytest.raises(lp.AlpacaAPIError) as exc_info:
        lp._req("POST", "/v2/orders", {"symbol": "GLD"})

    message = str(exc_info.value)
    assert "HTTP 422" in message
    assert "fractional short orders are not supported" in message
    assert "secret-key-value" not in message
    assert "secret-value" not in message


def test_non_paper_endpoint_requires_explicit_override(lp, monkeypatch) -> None:
    monkeypatch.setattr(lp, "ALPACA_KEY", "key")
    monkeypatch.setattr(lp, "ALPACA_SECRET", "secret")
    monkeypatch.setattr(lp, "ALPACA_BASE", "https://api.alpaca.markets")
    monkeypatch.setattr(lp, "ALLOW_LIVE_ALPACA", False)

    with pytest.raises(SystemExit, match="Refusing non-paper"):
        lp._require_creds()


def test_non_wednesday_keeps_fixed_momentum_shares_and_dry_run_is_read_only(
    lp, monkeypatch, tmp_path: Path
) -> None:
    class Thursday(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 23, 21, 0, tzinfo=timezone.utc)

    state_path = tmp_path / "state.json"
    original_state = {
        "momentum": {"AAPL": 1_000.0},
        "momentum_shares": {"AAPL": 10.0},
        "trend": {},
    }
    state_path.write_text(json.dumps(original_state))
    monkeypatch.setattr(lp, "datetime", Thursday)
    monkeypatch.setattr(lp, "STATE_PATH", state_path)
    monkeypatch.setattr(lp, "LOG_PATH", tmp_path / "orders.jsonl")
    monkeypatch.setattr(lp, "HEARTBEAT_PATH", tmp_path / "heartbeat.json")
    monkeypatch.setattr(lp, "EQUITY_LOG", tmp_path / "equity.jsonl")
    monkeypatch.setattr(lp, "get_equity", lambda: 100_000.0)
    monkeypatch.setattr(lp, "get_positions", lambda: {"AAPL": 10.0})
    monkeypatch.setattr(lp, "get_open_orders", lambda: [])
    monkeypatch.setattr(lp, "fetch", lambda symbols: {})
    monkeypatch.setattr(lp, "trend_targets", lambda equity, data, state: ({}, "none"))
    # Price doubled since Wednesday. The old bug would have sold five shares to
    # restore the stale $1,000 target; the fixed engine must keep all ten shares.
    monkeypatch.setattr(lp, "latest_price", lambda symbol, cache: 200.0)

    result = lp.run_once(SimpleNamespace(live=False, force_wed=False))

    assert result["orders"] == 0
    assert json.loads(state_path.read_text()) == original_state
    assert not (tmp_path / "orders.jsonl").exists()
    assert not (tmp_path / "heartbeat.json").exists()
    assert not (tmp_path / "equity.jsonl").exists()


def test_non_wednesday_omitted_share_target_retries_exit(
    lp, monkeypatch, tmp_path: Path
) -> None:
    class Thursday(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 23, 21, 0, tzinfo=timezone.utc)

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "momentum": {"AAPL": 1_000.0},
        "momentum_shares": {},
        "trend": {},
    }))
    monkeypatch.setattr(lp, "datetime", Thursday)
    monkeypatch.setattr(lp, "STATE_PATH", state_path)
    monkeypatch.setattr(lp, "get_equity", lambda: 100_000.0)
    monkeypatch.setattr(lp, "get_positions", lambda: {"AAPL": 10.0})
    monkeypatch.setattr(lp, "get_open_orders", lambda: [])
    monkeypatch.setattr(lp, "fetch", lambda symbols: {})
    monkeypatch.setattr(lp, "trend_targets", lambda equity, data, state: ({}, "none"))
    monkeypatch.setattr(lp, "latest_price", lambda symbol, cache: 200.0)

    result = lp.run_once(SimpleNamespace(live=False, force_wed=False))

    assert result["orders"] == 1


def configure_mock_live_run(lp, monkeypatch, tmp_path: Path) -> Path:
    class Thursday(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 23, 21, 0, tzinfo=timezone.utc)

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"momentum": {}, "trend": {}}))
    monkeypatch.setattr(lp, "datetime", Thursday)
    monkeypatch.setattr(lp, "STATE_PATH", state_path)
    monkeypatch.setattr(lp, "LOG_PATH", tmp_path / "orders.jsonl")
    monkeypatch.setattr(lp, "HEARTBEAT_PATH", tmp_path / "heartbeat.json")
    monkeypatch.setattr(lp, "EQUITY_LOG", tmp_path / "equity.jsonl")
    monkeypatch.setattr(lp, "get_equity", lambda: 100_000.0)
    monkeypatch.setattr(lp, "get_positions", lambda: {})
    monkeypatch.setattr(lp, "get_open_orders", lambda: [])
    monkeypatch.setattr(lp, "fetch", lambda symbols: {})
    monkeypatch.setattr(lp, "latest_price", lambda symbol, cache: 100.0)
    monkeypatch.setattr(
        lp,
        "get_asset",
        lambda symbol: {
            "tradable": True,
            "shortable": True,
            "borrow_status": "easy_to_borrow",
        },
    )

    def trend_targets(equity, data, state):
        state["trend"] = {"GLD": {"side": -1, "extreme": 100.0, "shares": 9.5}}
        return {"GLD": -950.0}, "one short target"

    monkeypatch.setattr(lp, "trend_targets", trend_targets)
    return state_path


def test_live_short_submission_is_integer_and_healthy(lp, monkeypatch, tmp_path: Path) -> None:
    state_path = configure_mock_live_run(lp, monkeypatch, tmp_path)
    submitted = []

    def submit(symbol, qty, side):
        submitted.append((symbol, qty, side))
        return {"status": "accepted"}

    monkeypatch.setattr(lp, "submit_order", submit)

    result = lp.run_once(SimpleNamespace(live=True, force_wed=False))

    assert submitted == [("GLD", -9.0, "sell")]
    assert result["healthy"] is True
    assert json.loads((tmp_path / "heartbeat.json").read_text())["ok"] is True
    assert "GLD" in json.loads(state_path.read_text())["trend"]


def test_partial_submit_failure_is_unhealthy_and_rolls_back_state(
    lp, monkeypatch, tmp_path: Path
) -> None:
    state_path = configure_mock_live_run(lp, monkeypatch, tmp_path)

    def reject(symbol, qty, side):
        raise lp.AlpacaAPIError("Alpaca POST /v2/orders -> HTTP 422: rejected")

    monkeypatch.setattr(lp, "submit_order", reject)

    result = lp.run_once(SimpleNamespace(live=True, force_wed=False))
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())

    assert result["healthy"] is False
    assert heartbeat["ok"] is False
    assert heartbeat["failed"] == 1
    assert "GLD" not in json.loads(state_path.read_text())["trend"]
