from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from scripts.research.crypto_hunter.scan import (
    ASSET_ORIGIN, BAR_PATH, DATA_ORIGIN, GrantedReader, NoRedirect, Refused,
    bars, tradable_symbols,
)
from scripts.research.crypto_hunter.install_timer import render


NOW = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)


def setup_reader(tmp_path, transport, budget=2, mutate=None):
    grant = {"operator_signed": True, "purpose": "crypto-hunter-weekly-scan",
             "credential_set": "claude-book-momentum", "granted_at": "2026-10-01T00:00:00Z",
             "not_after": "2026-10-17T00:00:00Z", "scope": {DATA_ORIGIN: {
                 "methods": ["GET"], "paths": [BAR_PATH], "max_requests": budget}}}
    if mutate:
        mutate(grant)
    (tmp_path / "grant.json").write_text(json.dumps(grant))
    (tmp_path / "credentials.env").write_text("APCA_API_KEY_ID=synthetic-id\nAPCA_API_SECRET_KEY=synthetic-secret\n")
    return GrantedReader(tmp_path / "grant.json", tmp_path / "credentials.env", tmp_path / "state",
                         now=lambda: NOW, transport=transport)


def test_wrong_account_scope_and_expiry_refused_before_network(tmp_path):
    calls = []
    for mutate in (lambda g: g.update(credential_set="other-account"),
                   lambda g: g.update(not_after="2026-10-09T00:00:00Z"),
                   lambda g: g.update(operator_signed=False)):
        with pytest.raises(Refused):
            setup_reader(tmp_path, lambda *args: calls.append(args), mutate=mutate)
    assert not calls


def test_budget_is_persistent_and_counts_failures(tmp_path):
    calls = []

    def fail(url, headers):
        calls.append(url)
        raise Refused("synthetic error")

    first = setup_reader(tmp_path, fail, budget=1)
    with pytest.raises(Refused, match="synthetic"):
        first.get(DATA_ORIGIN, BAR_PATH, {})
    second = setup_reader(tmp_path, fail, budget=1)
    with pytest.raises(Refused, match="budget"):
        second.get(DATA_ORIGIN, BAR_PATH, {})
    assert len(calls) == 1


def test_accounts_orders_and_non_granted_metadata_cannot_be_requested(tmp_path):
    calls = []
    reader = setup_reader(tmp_path, lambda *args: calls.append(args))
    for origin, path in ((DATA_ORIGIN, "/v2/orders"), (ASSET_ORIGIN, "/v2/account"),
                         (ASSET_ORIGIN, "/v2/assets"), ("https://example.invalid", BAR_PATH)):
        with pytest.raises(Refused):
            reader.get(origin, path, {})
    assert not calls


def test_redirects_cannot_forward_credentials():
    with pytest.raises(Refused):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid")


def test_operator_asset_snapshot_must_be_current_and_confirmed(tmp_path):
    reader = setup_reader(tmp_path, lambda *args: pytest.fail("network not expected"))
    snapshot = tmp_path / "assets.json"
    s = {"operator_confirmed": True, "as_of": "2026-10-01T00:00:00Z",
         "not_after": "2026-10-17T00:00:00Z", "symbols": ["BTC/USD", "USDC/USD"]}
    snapshot.write_text(json.dumps(s))
    assert tradable_symbols(reader, NOW, snapshot) == ["BTC/USD"]
    s["operator_confirmed"] = False
    snapshot.write_text(json.dumps(s))
    with pytest.raises(Refused):
        tradable_symbols(reader, NOW, snapshot)


def test_bar_shape_matches_alpaca_and_rejects_partial_day(tmp_path):
    row = {"t": "2026-10-09T00:00:00Z", "o": 100, "h": 120, "l": 90, "c": 110,
           "v": 1000, "n": 20, "vw": 105}
    reader = setup_reader(tmp_path, lambda *args: {"bars": {"BTC/USD": [row]}, "next_page_token": None})
    m = bars(reader, ["BTC/USD"], NOW)
    assert m.volume.iloc[0, 0] == 1000
    row["t"] = "2026-10-10T00:00:00Z"
    with pytest.raises(Refused, match="incomplète"):
        bars(reader, ["BTC/USD"], NOW)


def test_pagination_does_not_drop_other_symbols(tmp_path):
    rows = {s: [{"t": "2026-10-09T00:00:00Z", "o": 10, "h": 11, "l": 9,
                 "c": 10, "v": 100}] for s in ("BTC/USD", "ETH/USD")}
    answers = iter([{"bars": {"BTC/USD": rows["BTC/USD"]}, "next_page_token": "synthetic-page"},
                    {"bars": {"ETH/USD": rows["ETH/USD"]}, "next_page_token": None}])
    reader = setup_reader(tmp_path, lambda *args: next(answers))
    m = bars(reader, list(rows), NOW)
    assert list(m.close.columns) == ["BTC-USD", "ETH-USD"]


def test_zero_survivors_write_french_report_without_network_or_keys(tmp_path, monkeypatch):
    from scripts.research.crypto_hunter import scan
    monkeypatch.setattr(scan, "approved_rules", lambda path: {})
    monkeypatch.setattr(scan, "GrantedReader", lambda *args: pytest.fail("must not read credentials or use network"))
    output = tmp_path / "latest.md"
    monkeypatch.setattr("sys.argv", ["scan", "--approved", "unused", "--grant", "absent",
                                    "--credentials", "absent", "--state", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        scan.main()
    assert exit_info.value.code == 0
    assert "Aucune règle ne survit" in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600


def test_timer_is_utc_and_isolated_with_process_kill_mode(tmp_path):
    args = SimpleNamespace(python=tmp_path / "python", worktree=tmp_path / "tree",
                           approved=tmp_path / "rules", grant=tmp_path / "grant",
                           credentials=tmp_path / "keys", state=tmp_path / "state",
                           output=tmp_path / "latest", assets=None)
    service, timer = render(args)
    assert "KillMode=process" in service
    assert "Sat *-*-* 10:00:00 UTC" in timer
    assert "scripts.research.crypto_hunter.scan" in service
    assert f"WorkingDirectory={args.worktree}\n" in service
    assert "claude-book" not in service
