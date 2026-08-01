import socket
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden in crypto_news tests")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    yield
