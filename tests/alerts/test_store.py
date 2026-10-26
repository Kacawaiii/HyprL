import os

import pytest

from scripts.alert_router.model import Alert, AlertError, synthetic_alert
from scripts.alert_router.store import Store, create_key


def test_private_config_and_queue_are_encrypted_and_tenant_scoped(store):
    token = store.register("staff")
    dest = store.add(
        "staff",
        "webhook",
        {
            "url": "https://example.org/secret-hook",
            "signing_key": "synthetic-unique-secret-at-least-32",
        },
    )
    store.enqueue("staff", Alert.parse(synthetic_alert()), now=1793016000)
    assert store.list("staff")[0]["id"] == dest
    assert store.list("other") == []
    assert store.feed(token)[0]["kind"] == "system_test"
    raw = store.path.read_bytes()
    for sensitive in (
        b"secret-hook",
        b"synthetic-unique-secret",
        b"staff",
        token.encode(),
        b"example.org",
    ):
        assert sensitive not in raw
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    with pytest.raises(AlertError):
        store.pause("other", dest, True)


def test_key_not_overwritten_and_unsafe_permissions_rejected(tmp_path):
    key = tmp_path / "key"
    create_key(key)
    with pytest.raises(FileExistsError):
        create_key(key)
    key.chmod(0o644)
    with pytest.raises(AlertError):
        Store(tmp_path / "state", key, allowed_customers={"staff"})
