"""Synthetic Coinbase wire responses and temporary grant budgets only."""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.video_factory import collect_study as collect

NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)
START = collect.instant("2018-01-01T00:00:00Z")


def grant(tmp_path):
    value = dict(
        schema="research-data-authorization-v1",
        id=collect.ID,
        status="AUTHORIZED",
        operator_signed=True,
        not_after="2026-11-15T00:00:00Z",
        scope=dict(
            origin=collect.ORIGIN,
            method="GET",
            path=collect.PATH,
            product="BTC-USD",
            granularity_seconds=3600,
            start="2018-01-01T00:00:00Z",
            end_exclusive="2026-09-01T00:00:00Z",
            max_requests_including_retries=400,
            max_requests_per_second=1,
        ),
    )
    path = tmp_path / "grant.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("fault", ["unsigned", "expired", "protected"])
def test_invalid_authority_stops_before_request_or_shard(tmp_path, fault):
    path = grant(tmp_path)
    value = json.loads(path.read_text())
    if fault == "unsigned":
        value["operator_signed"] = False
    if fault == "expired":
        value["not_after"] = "2026-10-28T00:00:00Z"
    if fault == "protected":
        value["scope"]["end_exclusive"] = "2026-09-02T00:00:00Z"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="AUTHORIZATION_REQUIRED"):
        collect.collect(
            path,
            tmp_path / "out",
            bank=tmp_path / "bank",
            clock=lambda: NOW,
            send=lambda *a: pytest.fail("request"),
        )
    assert not (tmp_path / "out").exists()


def test_exact_native_shard_and_budget_survive_resume(tmp_path, monkeypatch):
    path = grant(tmp_path)
    calls = []
    monkeypatch.setattr(collect, "windows", lambda: iter([(START, START + timedelta(hours=2))]))
    wire = [[int((START + timedelta(hours=h)).timestamp()), 99, 102, 100, 101, 2] for h in (1, 0)]

    def send(url):
        calls.append(url)
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        assert parts.hostname == "api.exchange.coinbase.com" and parts.path == collect.PATH
        assert query["granularity"] == ["3600"] and query["end"] == ["2018-01-01T01:59:59Z"]
        return 200, json.dumps(wire).encode()

    root = tmp_path / "out"
    bank = tmp_path / "bank"
    first = collect.collect(
        path, root, bank=bank, clock=lambda: NOW, send=send, pause=lambda *_: None
    )
    second = collect.collect(
        path, root, bank=bank, clock=lambda: NOW, send=send, pause=lambda *_: None
    )
    assert first == second and first["requests"] == 1 and len(calls) == 1
    assert json.loads((root / "candles-000.json").read_text()) == wire
    (root / "candles-000.json").write_text("[]")
    with pytest.raises(ValueError, match="SHARD_BINDING"):
        collect.collect(path, root, bank=bank, clock=lambda: NOW, send=send)
    assert len(calls) == 1


def test_scope_ends_before_protected_hour_and_never_inspects_its_prices():
    windows = list(collect.windows())
    assert windows[0][0] == START and windows[-1][1] == collect.instant("2026-09-01T00:00:00Z")
    assert len(windows) < 400
    protected = [int(windows[-1][1].timestamp()), {}, {}, {}, {}, {}]
    with pytest.raises(ValueError, match="PROTECTED_TIMESTAMP"):
        collect.checked_rows(json.dumps([protected]).encode(), *windows[-1])


def test_403_halts_without_retry_across_restarts(tmp_path, monkeypatch):
    path = grant(tmp_path)
    calls = []
    monkeypatch.setattr(collect, "windows", lambda: iter([(START, START + timedelta(hours=2))]))

    def send(url):
        calls.append(url)
        return 403, b""

    for index in range(2):
        with pytest.raises(ValueError, match="STUDY_HTTP_403|STUDY_SOURCE_HALTED"):
            collect.collect(
                path, tmp_path / "out", bank=tmp_path / "bank", clock=lambda: NOW, send=send
            )
    assert len(calls) == 1


def test_global_400_request_budget_cannot_be_reset_with_new_output(tmp_path, monkeypatch):
    path = grant(tmp_path)
    identity = collect.authorize(path, NOW)
    bank = tmp_path / "bank"
    bank.mkdir()
    rows = []
    previous = "0" * 64
    for i in range(400):
        row = dict(
            kind="request",
            at=NOW.isoformat(),
            authorization_sha256=identity,
            previous=previous,
            url=collect.ORIGIN + collect.PATH + "?synthetic=" + str(i),
            number=i + 1,
        )
        row["hash"] = collect.sha(collect.encoded(row))
        previous = row["hash"]
        rows.append(row)
    (bank / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(collect, "windows", lambda: iter([(START, START + timedelta(hours=2))]))
    with pytest.raises(ValueError, match="REQUEST_BUDGET"):
        collect.collect(
            path,
            tmp_path / "new-output",
            bank=bank,
            clock=lambda: NOW,
            send=lambda *_: pytest.fail("budget bypass"),
        )
