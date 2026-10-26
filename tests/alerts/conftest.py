import pytest

from scripts.alert_router.store import Store, create_key


@pytest.fixture
def store(tmp_path):
    key = tmp_path / "key"
    create_key(key)
    return Store(tmp_path / "state", key, allowed_customers={"staff", "other"})
