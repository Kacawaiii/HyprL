import json
from pathlib import Path

import pytest

from scripts.social_publisher.__main__ import main
from scripts.social_publisher.core import FACTORY_REVISION, Ledger, atomic_json
from scripts.social_publisher.ops import stage_units
from scripts.social_publisher.runner import daily, default_config


class Credentials:
    def __init__(self, platform):
        pass

    def token(self, http):
        return "synthetic-token"


class API:
    def __init__(self, platform, grant):
        self.platform = platform

    def request(self, method, url, **kwargs):
        if self.platform == "instagram":
            raise RuntimeError("SECRET_MUST_NOT_ESCAPE")
        if "channels?" in url:
            return {"items": [{"id": "123"}]}, {}, 200
        if method == "POST":
            return (
                {},
                {
                    "location": "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=synthetic"
                },
                200,
            )
        return {"id": "synthetic-video"}, {}, 200


def config_for(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = default_config()
    config.update(media_kind="video", factory_revision=FACTORY_REVISION)
    config["state_dir"] = str(tmp_path / "runtime")
    for platform in ("instagram", "youtube"):
        config["platforms"][platform].update(enabled=True, account_id="123")
    return config


def grant_for(tmp_path, config):
    (tmp_path / "authorizations").mkdir(mode=0o700)
    atomic_json(
        Path(config["authorization"]),
        {
            "status": "AUTHORIZED",
            "purpose": "social-publisher-v1",
            "accounts": {"youtube": "123", "instagram": "123"},
            "not_before": "2000-01-01T00:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "max_requests": 30,
            "allowed_requests": [],
        },
    )


def test_dry_run_has_no_credentials_ledger_grant_or_network(tmp_path, monkeypatch, bundle, now):
    config = config_for(tmp_path, monkeypatch)

    def refuse(*args):
        pytest.fail("dry run must not construct credentials or transport")

    result = daily(
        config,
        now=now,
        loader=lambda *args: bundle,
        credential_factory=refuse,
        transport_factory=refuse,
    )
    assert result["status"] == "dry_run"
    assert result["platforms"]["youtube"] == {"status": "would_publish", "privacy": "private"}
    assert not Path(config["state_dir"]).exists()


def test_enabled_platforms_are_independent_and_uncertain_post_never_repeats(
    tmp_path, monkeypatch, bundle, now
):
    config = config_for(tmp_path, monkeypatch)
    config["dry_run"] = False
    grant_for(tmp_path, config)
    result = daily(
        config,
        now=now,
        loader=lambda *args: bundle,
        credential_factory=Credentials,
        transport_factory=API,
    )
    assert result["status"] == "blocked"
    assert result["platforms"]["youtube"]["status"] == "posted"
    assert "SECRET" not in json.dumps(result)
    result = daily(
        config,
        now=now,
        loader=lambda *args: bundle,
        credential_factory=Credentials,
        transport_factory=lambda *args: pytest.fail("must not retry"),
    )
    assert result["platforms"]["youtube"]["status"] == "already_posted"
    assert result["platforms"]["instagram"]["reason"] == "previous_attempt_needs_reconciliation"


def test_missing_grant_fails_before_credentials_or_any_remote_call(
    tmp_path, monkeypatch, bundle, now
):
    config = config_for(tmp_path, monkeypatch)
    config["dry_run"] = False

    def refuse(*args):
        pytest.fail("grant must be verified first")

    result = daily(
        config,
        now=now,
        loader=lambda *args: bundle,
        credential_factory=refuse,
        transport_factory=refuse,
    )
    assert result["status"] == "blocked"
    assert all(p["reason"] == "input_unavailable" for p in result["platforms"].values())


def test_x_without_free_entitlement_and_current_tiktok_branding_stay_blocked(
    tmp_path, monkeypatch, bundle, now
):
    config = config_for(tmp_path, monkeypatch)
    for platform in ("instagram", "youtube"):
        config["platforms"][platform]["enabled"] = False
    for platform in ("x", "tiktok"):
        config["platforms"][platform].update(enabled=True, account_id="123")
    result = daily(config, now=now, loader=lambda *args: bundle)
    assert result["platforms"]["x"]["reason"] == "x_zero_cost_access_unproven"
    assert (
        result["platforms"]["tiktok"]["reason"]
        == "factory_v2_branding_incompatible_with_tiktok_guidelines"
    )


def test_all_platforms_disabled_does_not_open_factory(tmp_path, monkeypatch, now):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = daily(
        default_config(),
        now=now.replace(hour=17, minute=42),
        loader=lambda *args: pytest.fail("no factory read needed"),
    )
    assert result["status"] == "disabled"


def test_cli_safe_defaults_and_explicit_dry_run_override(tmp_path, monkeypatch, capsys):
    path = tmp_path / "private-config" / "publisher.json"
    assert main(["--config", str(path), "init-config"]) == 0
    config = json.loads(path.read_text())
    assert config["dry_run"] is True
    assert all(p["enabled"] is False for p in config["platforms"].values())
    assert path.stat().st_mode & 0o777 == 0o600
    assert "created_disabled_dry_run" in capsys.readouterr().out


def test_cli_never_emits_exception_values_or_tokens(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"SECRET_ACTUAL_VALUE')
    path.chmod(0o600)
    assert main(["--config", str(path), "daily"]) == 1
    assert "SECRET" not in capsys.readouterr().out


def test_units_are_staged_pinned_and_resource_bounded(tmp_path):
    import shutil
    import subprocess

    output = tmp_path / "units"
    names = stage_units(output, tmp_path / "config.json", "a" * 40)
    assert len(names) == 4
    service = (output / "hyprl-social-publisher.service").read_text()
    assert "OnFailure=hyprl-social-publisher-alert.service" in service
    assert "MemoryMax=300M" in service and "UMask=0077" in service
    assert "--expected-revision" in service and "a" * 40 in service
    timer = (output / "hyprl-social-publisher.timer").read_text()
    assert "Persistent=false" in timer
    assert "18:42:00 Europe/Paris" in timer
    assert not (Path.home() / ".config/systemd/user/hyprl-social-publisher.timer").exists()
    if shutil.which("systemd-analyze"):
        checked = subprocess.run(
            ["systemd-analyze", "--user", "verify", *[str(output / name) for name in names]],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert checked.returncode == 0, checked.stderr


def test_only_existing_pending_identity_can_be_operator_reconciled(
    tmp_path, monkeypatch, bundle, capsys
):
    config = config_for(tmp_path, monkeypatch)
    directory = tmp_path / "config"
    directory.mkdir(mode=0o700)
    path = directory / "publisher.json"
    atomic_json(path, config)
    ledger = Ledger(config["state_dir"])
    ledger.reserve("youtube", bundle, "123")
    ledger.db.close()
    assert (
        main(
            [
                "--config",
                str(path),
                "reconcile-posted",
                "--platform",
                "youtube",
                "--sha256",
                bundle.sha256,
                "--remote-id",
                "confirmed-synthetic",
                "--confirmation",
                "I_VERIFIED_REMOTE_POST",
            ]
        )
        == 0
    )
    ledger = Ledger(config["state_dir"])
    assert ledger.rows()[0]["status"] == "posted"
    ledger.db.close()
    assert "operator_reconciled" in capsys.readouterr().out


def test_confirmed_post_needs_no_fresh_credentials_or_grant(tmp_path, monkeypatch, bundle, now):
    config = config_for(tmp_path, monkeypatch)
    config["dry_run"] = False
    config["platforms"]["instagram"]["enabled"] = False
    ledger = Ledger(config["state_dir"])
    ledger.reserve("youtube", bundle, "123")
    ledger.finish("youtube", bundle.sha256, "already-synthetic")
    ledger.db.close()
    result = daily(
        config,
        now=now,
        loader=lambda *args: bundle,
        credential_factory=lambda *a: pytest.fail("no new credentials"),
        transport_factory=lambda *a: pytest.fail("no new HTTP"),
    )
    assert result["platforms"]["youtube"]["status"] == "already_posted"


def test_failure_alert_uses_only_existing_notify_cli_and_fixed_text(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts.social_publisher import ops

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ops.subprocess, "run", run)
    assert ops.notify() is True
    assert calls[0][0] == [str(tmp_path / "bin/hyprl-notify"), "alertes"]
    assert "échec ou résultat incertain" in calls[0][1]["input"]
    assert calls[0][1]["capture_output"] is True
