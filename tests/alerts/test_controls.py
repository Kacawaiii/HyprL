import asyncio
from types import SimpleNamespace

import pytest

from scripts.alert_router.api import ReadOnlyAPI
from scripts.alert_router.discord_gateway import Access, AlertCommands, PauseView
from scripts.alert_router.model import Alert, AlertError, synthetic_alert
from scripts.alert_router.router import Router
from scripts.alert_router.store import Store, create_key
from tests.alerts.test_router import Receiver


class Response:
    def __init__(self):
        self.sent = []

    def is_done(self):
        return bool(self.sent)

    async def send_message(self, text, **kwargs):
        self.sent.append((text, kwargs))

    async def send(self, text, **kwargs):
        self.sent.append((text, kwargs))


def interaction(*, guild=1, user=2, roles=(3,)):
    response = Response()
    return SimpleNamespace(
        guild_id=guild,
        guild=object() if guild else None,
        id=100,
        user=SimpleNamespace(id=user, roles=[SimpleNamespace(id=r) for r in roles]),
        response=response,
        followup=response,
    )


@pytest.fixture
def controls(tmp_path):
    create_key(tmp_path / "key")
    store = Store(tmp_path / "state", tmp_path / "key", allowed_customers={"1/2", "1/4"})
    return store, Access(store, guild_id=1, role_id=3)


@pytest.mark.parametrize(
    "kwargs", [{"guild": None}, {"guild": 9}, {"roles": ()}, {"roles": (99,)}, {"user": 77}]
)
def test_access_requires_exact_guild_role_id_and_staff_allowlist(controls, kwargs):
    _, access = controls
    with pytest.raises(AlertError):
        access.customer(interaction(**kwargs))


def test_staff_slash_test_to_outbox_to_receiver_then_one_click_pause(controls):
    store, access = controls
    dest = store.add(
        "1/2", "discord", {"url": "https://discord.com/api/webhooks/123/synthetic-secret"}
    )
    commands = AlertCommands(access, clock=lambda: 1793016000)

    async def scenario():
        i = interaction()
        await commands.test.callback(commands, i, dest)
        assert i.response.sent[0][1]["ephemeral"] is True
        receiver = Receiver()
        assert await Router(store, receiver, clock=lambda: 1793016000).run_once()
        _, body, _ = receiver.requests[0]
        assert b"synthetic-secret" not in body
        assert "information, pas un conseil en investissement" in body.decode()
        assert "TEST STAFF" in body.decode()
        await commands.liste.callback(commands, i)
        assert all(kwargs["ephemeral"] for _, kwargs in i.response.sent)
        assert all("https://" not in text for text, _ in i.response.sent)
        view = PauseView(access, dest)
        other = interaction(user=4)
        await view.children[0].callback(other)
        assert not store.list("1/2")[0]["paused"]
        await view.children[0].callback(i)
        assert store.list("1/2")[0]["paused"]
        await commands.supprimer.callback(commands, i, dest)
        assert store.list("1/2") == []

    asyncio.run(scenario())


def test_api_auth_pagination_rotation_and_no_write_routes(controls):
    store, _ = controls
    token, other_token = store.register("1/2"), store.register("1/4")
    store.enqueue("1/2", Alert.parse(synthetic_alert()), now=1793016000)
    api = ReadOnlyAPI(store)
    assert api.handle("GET", "/v1/alerts")[0] == 401
    assert api.handle("GET", "/v1/alerts", authorization="Bearer invalid")[0] == 401
    assert api.handle("POST", "/v1/alerts", authorization=f"Bearer {token}")[0] == 405
    assert api.handle("GET", "/v1/alerts", authorization=f"Bearer {other_token}")[1]["items"] == []
    code, payload = api.handle(
        "GET", "/v1/alerts", authorization=f"Bearer {token}", query={"limit": "1"}
    )
    assert code == 200 and len(payload["items"]) == 1
    assert (
        api.handle(
            "GET",
            "/v1/alerts",
            authorization=f"Bearer {token}",
            query={"after": str(payload["next_cursor"])},
        )[1]["items"]
        == []
    )
    assert (
        api.handle("GET", "/v1/alerts", authorization=f"Bearer {token}", query={"limit": "101"})[0]
        == 400
    )
    store.register("1/2")
    assert api.handle("GET", "/v1/alerts", authorization=f"Bearer {token}")[0] == 401
