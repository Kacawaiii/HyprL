from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    name = "hyprl_test_portfolio_watchdog"
    path = ROOT / "scripts/momentum_stocks/watchdog.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def watchdog(tmp_path: Path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "HEARTBEAT", tmp_path / "heartbeat.json")
    return module


def write_heartbeat(path: Path, **payload) -> None:
    data = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(data))


def test_healthy_heartbeat_has_no_problem(watchdog) -> None:
    write_heartbeat(watchdog.HEARTBEAT, ok=True, failed=0, deferred=0, planning_errors=0)

    assert watchdog.check_heartbeat() == []


def test_false_ok_is_reported_with_issue_details(watchdog) -> None:
    write_heartbeat(
        watchdog.HEARTBEAT,
        ok=False,
        failed=3,
        issues=["GLD [submit]: fractional short orders are not supported"],
    )

    problems = watchdog.check_heartbeat()

    assert len(problems) == 1
    assert "unhealthy run" in problems[0]
    assert "failed=3" in problems[0]
    assert "GLD" in problems[0]


def test_failure_count_is_reported_even_if_legacy_ok_was_true(watchdog) -> None:
    write_heartbeat(watchdog.HEARTBEAT, ok=True, failed=2)

    assert "failed=2" in watchdog.check_heartbeat()[0]


def test_stale_heartbeat_is_reported(watchdog) -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=watchdog.MAX_AGE_HOURS + 1)
    watchdog.HEARTBEAT.write_text(json.dumps({"ts": old.isoformat(), "ok": True}))

    assert "SILENT" in watchdog.check_heartbeat()[0]
