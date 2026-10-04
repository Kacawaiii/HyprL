"""`hyprl start` hands the official event-source stores to the read-only API (the Events page), resolved to
absolute paths, and adds nothing when none is given."""

from __future__ import annotations

import pytest

from scripts.trading_lab import hyprl_cli


class _Captured(Exception):
    pass


def _start_command(tmp_path, monkeypatch, *extra):
    seen = {}

    def fake_start(**kwargs):
        seen.update(kwargs)
        raise _Captured
    monkeypatch.setattr(hyprl_cli.supervisor, "start", fake_start)
    with pytest.raises(_Captured):
        hyprl_cli.main(["--runtime", str(tmp_path / "runtime"), *extra, "start", "--no-build", "--no-browser"])
    return seen["command"]


def test_start_passes_the_event_stores_to_the_api(tmp_path, monkeypatch):
    fomc, edgar = tmp_path / "fomc", tmp_path / "edgar"
    command = _start_command(tmp_path, monkeypatch, "--fomc-store", str(fomc), "--edgar-store", str(edgar))
    assert command[command.index("--fomc-store") + 1] == str(fomc.resolve())
    assert command[command.index("--edgar-store") + 1] == str(edgar.resolve())


def test_start_without_stores_adds_no_store_flag(tmp_path, monkeypatch):
    command = _start_command(tmp_path, monkeypatch)
    assert "--fomc-store" not in command and "--edgar-store" not in command
