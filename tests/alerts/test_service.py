import json

from scripts.alert_router.model import synthetic_alert
from scripts.alert_router.service import main


def test_cli_private_configuration_ingest_status_and_token_file(tmp_path, capsys, monkeypatch):
    key, policy = tmp_path / "key", tmp_path / "policy"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "guild_id": 1,
                "role_id": 3,
                "allowed_customers": ["1/2"],
                "telegram_chats": {},
            }
        )
    )
    policy.chmod(0o600)
    common = ["--state", str(tmp_path / "state"), "--key", str(key), "--policy", str(policy)]
    assert main(common + ["init-key"]) == 0
    config = tmp_path / "config"
    config.write_text(
        json.dumps(
            {
                "channel": "webhook",
                "config": {
                    "url": "https://example.org/synthetic-private-hook",
                    "signing_key": "x" * 32,
                },
            }
        )
    )
    config.chmod(0o600)
    assert main(common + ["add", "--customer", "1/2", "--file", str(config)]) == 0
    event = tmp_path / "event"
    event.write_text(json.dumps(synthetic_alert()))
    event.chmod(0o600)
    assert main(common + ["ingest", "--file", str(event)]) == 0
    assert main(common + ["status", "--customer", "1/2"]) == 0
    token_file = tmp_path / "token"
    assert main(common + ["api-token", "--customer", "1/2", "--output", str(token_file)]) == 0
    token = token_file.read_text().strip()
    assert len(token) >= 40 and token_file.stat().st_mode & 0o777 == 0o600
    text = capsys.readouterr().out
    assert "synthetic-private-hook" not in text and token not in text
    # Unknown customers and unsafe input files fail without paths or bodies.
    assert main(common + ["status", "--customer", "1/99"]) == 2
    failure = capsys.readouterr().err
    assert "BLOCKED" in failure and str(tmp_path) not in failure


def test_service_errors_never_echo_secret_input(tmp_path, capsys):
    bad = tmp_path / "synthetic-secret-path"
    assert main(["--key", str(bad / "not-present"), "init-key"]) == 2
    assert "synthetic-secret-path" not in capsys.readouterr().err
