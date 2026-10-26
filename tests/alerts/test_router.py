import asyncio
import hashlib

from scripts.alert_router.model import Alert, synthetic_alert
from scripts.alert_router.router import Router
from scripts.alert_router.security import Receipt, verify


class Receiver:
    def __init__(self, statuses=(204,)):
        self.statuses = iter(statuses)
        self.requests = []

    async def post(self, url, body, headers):
        self.requests.append((url, body, headers))
        return Receipt(next(self.statuses))


def event(n=0):
    e = synthetic_alert()
    e["event_id"] = hashlib.sha256(str(n).encode()).hexdigest()
    return Alert.parse(e)


def destination(store, **kw):
    return store.add(
        "staff", "webhook", {"url": "https://example.org/hook", "signing_key": "k" * 32}, **kw
    )


def test_restart_preserves_rate_limit_and_signature(store):
    destination(store, min_interval=10, max_per_hour=1)
    receiver = Receiver((204, 204))
    now = 1793016000
    store.enqueue("staff", event(1), now=now)
    store.enqueue("staff", event(2), now=now)
    assert asyncio.run(Router(store, receiver, clock=lambda: now).run_once())
    assert not asyncio.run(Router(store, receiver, clock=lambda: now + 11).run_once())
    assert asyncio.run(Router(store, receiver, clock=lambda: now + 3600).run_once())
    _, body, headers = receiver.requests[0]
    assert verify(
        b"k" * 32, headers["X-HyprL-Timestamp"], body, headers["X-HyprL-Signature"], now=now
    )
    assert len(store.jobs("staff", state="sent")) == 2


def test_retry_backoff_dead_letter_and_pause(store):
    dest = destination(store)
    receiver = Receiver((503, 429, 400))
    now = 1793016000
    store.enqueue("staff", event(), now=now)
    router = Router(store, receiver, clock=lambda: now, jitter=lambda: 0)
    assert asyncio.run(router.run_once())
    assert store.jobs("staff")[0]["due"] == now + 10
    store.pause("staff", dest)
    now += 10
    assert not asyncio.run(router.run_once())
    store.pause("staff", dest, False)
    assert asyncio.run(router.run_once())
    assert store.jobs("staff")[0]["due"] == now + 20
    now += 20
    assert asyncio.run(router.run_once())
    assert store.jobs("staff", state="dead")[0]["code"] == "HTTP_400"


def test_enqueue_is_idempotent_and_deleted_destination_never_sends(store):
    dest = destination(store)
    now = 1793016000
    assert store.enqueue("staff", event(), now=now) == 1
    assert store.enqueue("staff", event(), now=now) == 0
    store.remove("staff", dest)
    receiver = Receiver(())
    assert not asyncio.run(Router(store, receiver, clock=lambda: now).run_once())
    assert receiver.requests == []
