"""The Monday one-shot at its external I/O seams; all provider bodies are synthetic."""

from contextlib import closing
from pathlib import Path

import pytest

from scripts.trading_lab.agent_v2_quality.contracts import digest, instant
from scripts.trading_lab.agent_v2_quality.ledger import Journal
from scripts.trading_lab.agent_v2_quality.monday import (
    TIMERS,
    bootstrap_units,
    ci_receipt,
    file_hash,
    run,
)
from scripts.trading_lab.agent_v2_quality.monday_staging import prepare
from scripts.trading_lab.agent_v2_quality.producer import atomic_json, read_local


def test_complete_monday_staging_refresh_proof_activation_and_timer_enable(tmp_path):
    settings, boundary = prepare(tmp_path / "staging")
    result = run(settings, boundary=boundary)
    assert result["status"] == "ACTIVATED_SHADOW_ONLY", result
    assert result["notification"] == "staging"
    assert result["broker_orders"] == 0
    assert result["registry_count"] == 3
    root = Path(settings["output"])
    proof = read_local(root / "activation-proof.json")
    snapshot = read_local(Path(settings["runtime"]) / "inbox/quality-snapshot.json")
    assert proof["producer_snapshot_hash"] == digest(snapshot)
    assert proof["identities"]["unit_hashes"] == {
        p.name: file_hash(p) for p in (root / "units").iterdir()
    }
    assert (
        read_local(root / "calendar-refresh/calendar.json")["coverage_end"]
        == "2026-11-02T21:00:00Z"
    )
    assert read_local(root / "inputs/registry.json")["edges"]["equity_momentum"] == {
        "enabled": False
    }
    assert len([u for u in boundary.requests if "api.nasdaq.com" in u]) == 8
    assert ["systemctl", "--user", "enable", "--now", *TIMERS] in boundary.commands
    with closing(Journal(settings["runtime"], read_only=True)) as journal:
        assert len(journal.records("activation")) == 1
        assert not journal.records("model_reservation")
        assert not journal.records("cycle_complete")
    assert len(boundary.messages) == 1 and "ACTIVATED_SHADOW_ONLY" in boundary.messages[0]
    requests = len(boundary.requests)
    assert run(settings, boundary=boundary)["status"] == "BLOCKED"
    assert len(boundary.requests) == requests  # no retry under the same reservation


@pytest.mark.parametrize("failure", ["late", "stale_radar", "calendar_403", "enable", "pin"])
def test_monday_failures_notify_and_leave_timers_disabled(tmp_path, failure):
    settings, boundary = prepare(tmp_path / "staging")
    if failure == "late":
        boundary.clock.at = instant("2026-10-26T11:25:00Z")
    elif failure == "stale_radar":
        radar = read_local(settings["radar_source"])
        radar["cutoff"] = "2026-10-25T16:00:00Z"
        atomic_json(settings["radar_source"], radar)
    elif failure == "calendar_403":
        boundary.fail_url = "api.nasdaq.com"
    elif failure == "enable":
        boundary.fail_command = "enable"
    else:
        Path(settings["hunter_evidence"]).write_text("changed")
    result = run(settings, boundary=boundary)
    assert result["status"] == "BLOCKED", result
    assert len(boundary.messages) == 1
    assert result["reason"] in boundary.messages[0]
    if failure == "enable":
        assert ["systemctl", "--user", "disable", "--now", *TIMERS] in boundary.commands
        assert not list(Path(settings["unit_dir"]).glob("*.timer"))
    else:
        assert not any("enable" in c for c in boundary.commands)


def test_exact_ci_rejects_other_sha_and_latest_failed_attempt(tmp_path):
    settings, boundary = prepare(tmp_path / "staging")
    send = boundary.send

    def failed(url, headers):
        status, raw = send(url, headers)
        import json

        payload = json.loads(raw)
        old = payload["workflow_runs"][0]
        payload["workflow_runs"].append(
            {
                **old,
                "id": 2,
                "conclusion": "failure",
                "html_url": "https://github.com/Kacawaiii/HyprL/actions/runs/2",
            }
        )
        return status, json.dumps(payload).encode()

    boundary.send = failed
    with pytest.raises(Exception, match="EXACT_SHA_CI_failure"):
        ci_receipt(settings["sha"], tmp_path / "ci", boundary=boundary)

    def wrong(url, headers):
        status, raw = send(url, headers)
        import json

        payload = json.loads(raw)
        payload["workflow_runs"][0]["head_sha"] = "0" * 40
        return status, json.dumps(payload).encode()

    boundary.send = wrong
    with pytest.raises(Exception, match="EXACT_SHA_CI_MISSING"):
        ci_receipt(settings["sha"], tmp_path / "other-ci", boundary=boundary)


def test_bootstrap_is_one_calendar_date_without_catchup(monkeypatch):
    monkeypatch.setenv("PATH", "/synthetic/model-cli/bin:/usr/bin:/bin")
    units = bootstrap_units("/synthetic/settings.json", "a" * 64)
    timer = units["hyprl-agent-v2-quality-monday-activate.timer"]
    assert "OnCalendar=2026-10-26 10:15:00 UTC" in timer
    assert "Persistent=false" in timer
    assert "TimeoutStartSec=65min" in units["hyprl-agent-v2-quality-monday-activate.service"]
    from scripts.trading_lab.agent_v2_quality.deploy import render

    services = render(
        "/synthetic/repo", "/usr/bin/python3", "/synthetic/runtime", "/synthetic/config"
    )
    for name, body in (units | services).items():
        if name.endswith(".service"):
            assert 'Environment="PATH=/synthetic/model-cli/bin:/usr/bin:/bin"' in body
