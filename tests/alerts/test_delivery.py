import asyncio
import hashlib
import json

import pytest

from scripts.alert_router.destinations import request
from scripts.alert_router.model import Alert, AlertError, synthetic_alert
from scripts.alert_router.producers import sec_story
from scripts.alert_router.router import Router
from scripts.alert_router.store import Store, create_key
from tests.alerts.test_router import Receiver, destination, event


@pytest.mark.parametrize(
    "channel,config",
    [
        ("discord", {"url": "https://discord.com/api/webhooks/123/synthetic"}),
        ("slack", {"url": "https://hooks.slack.com/services/T000/B000/synthetic"}),
        ("teams", {"url": "https://tenant.webhook.office.com/synthetic"}),
        ("telegram", {"chat_id": "-123456"}),
        ("webhook", {"url": "https://example.org/hook", "signing_key": "x" * 32}),
    ],
)
def test_each_channel_keeps_source_times_disclaimer_and_no_advice(channel, config):
    url, body, _ = request(
        {"channel": channel, "config": config},
        Alert.parse(synthetic_alert()),
        now=1793016000,
        telegram_token="123456:synthetic-token",
    )
    decoded = body.decode()
    assert "information, pas un conseil en investissement" in decoded
    assert "2026-10-26T12:00:00Z" in decoded
    assert "https://example.org/hyprl/staff-test" in decoded
    assert "HypR-A1" in decoded and "HypR-E1" in decoded
    assert "buy" not in decoded.lower() and "sell" not in decoded.lower()
    if channel == "discord":
        assert json.loads(body)["allowed_mentions"] == {"parse": []}


def test_retry_limit_survives_reopening_store_and_network_error_is_redacted(store):
    destination(store)
    now = 1793016000
    store.enqueue("staff", event(), now=now)

    class Broken:
        async def post(self, *args):
            raise OSError("https://example.org/secret?token=private-response")

    for i in range(5):
        reopened = Store(store.root, store.root.parent / "key", allowed_customers={"staff"})
        assert asyncio.run(
            Router(reopened, Broken(), clock=lambda: now, jitter=lambda: 0).run_once()
        )
        now += 200
    dead = store.jobs("staff", state="dead")
    assert dead[0]["attempts"] == 5 and dead[0]["code"] == "TRANSPORT_FAILED"
    assert "secret" not in json.dumps(dead) and "private-response" not in json.dumps(dead)


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 404, 410])
def test_permanent_errors_go_to_dead_letter_without_following_redirects(store, status):
    destination(store)
    store.enqueue("staff", event(), now=1793016000)
    assert asyncio.run(Router(store, Receiver((status,)), clock=lambda: 1793016000).run_once())
    assert store.jobs("staff", state="dead")[0]["code"] == f"HTTP_{status}"


def test_retry_after_and_atomic_lease_prevent_concurrent_sends(store):
    dest = destination(store)
    now = 1793016000
    store.enqueue("staff", event(1), now=now)
    store.enqueue("staff", event(2), now=now)
    leased = store.claim(now=now)
    assert leased and store.claim(now=now + 11) is None
    # A crashed sender's lease expires; the next reservation still consumes a rate slot.
    assert store.claim(now=now + 31)
    store.pause("staff", dest)
    assert not store.can_send(leased)


def test_watchlist_kinds_availability_conflicts_and_stub_channels(store):
    destination(store, watchlist=["AAPL"], kinds=["sec_filing"])
    store.add("staff", "email", {})
    store.add("staff", "api", {})
    e = synthetic_alert()
    e.update(
        kind="sec_filing",
        symbols=["MSFT"],
        facts={"form": "4", "cik": "0000320193", "accession": "0000320193-26-000001"},
    )
    assert store.enqueue("staff", Alert.parse(e), now=1793016000) == 0
    e.update(symbols=["AAPL"], event_id=hashlib.sha256(b"changed-symbol").hexdigest())
    assert store.enqueue("staff", Alert.parse(e), now=1793016000) == 1
    with pytest.raises(AlertError, match="EVENT_ID_CONFLICT"):
        changed = {**e, "symbols": ["NVDA"]}
        store.enqueue("staff", Alert.parse(changed), now=1793016000)
    future = synthetic_alert()
    future.update(available_at="2026-10-27T12:00:00Z", event_id="f" * 64)
    with pytest.raises(AlertError, match="EVENT_NOT_AVAILABLE"):
        store.enqueue("staff", Alert.parse(future), now=1793016000)


def test_mismatched_key_refuses_existing_store(tmp_path):
    create_key(tmp_path / "key")
    Store(tmp_path / "state", tmp_path / "key", allowed_customers={"staff"})
    create_key(tmp_path / "wrong")
    with pytest.raises(AlertError, match="DECRYPTION_FAILED"):
        Store(tmp_path / "state", tmp_path / "wrong", allowed_customers={"staff"})


def test_telegram_chat_is_operator_approved_per_customer(tmp_path):
    create_key(tmp_path / "key")
    s = Store(
        tmp_path / "state",
        tmp_path / "key",
        allowed_customers={"staff", "other"},
        telegram_chats={"staff": ["-123"]},
    )
    s.add("staff", "telegram", {"chat_id": "-123"})
    with pytest.raises(AlertError, match="TELEGRAM_CHAT_NOT_APPROVED"):
        s.add("other", "telegram", {"chat_id": "-123"})


def test_removing_customer_from_policy_stops_queued_delivery(store):
    destination(store)
    store.enqueue("staff", event(), now=1793016000)
    revoked = Store(store.root, store.root.parent / "key", allowed_customers={"other"})
    assert not asyncio.run(Router(revoked, Receiver(()), clock=lambda: 1793016000).run_once())


def test_duel_hashes_are_rejected_after_open_and_queued_hashes_expire(store):
    destination(store)
    e = synthetic_alert()
    e.update(
        kind="duel_hashes",
        facts={
            "session": "2026-10-26",
            "open_at": "2026-10-26T13:30:00Z",
            "a1_sha256": "a" * 64,
            "e1_sha256": "b" * 64,
        },
    )
    alert = Alert.parse(e)
    with pytest.raises(AlertError, match="HASHES_NOT_BEFORE_OPEN"):
        store.enqueue("staff", alert, now=1793021400)
    store.enqueue("staff", alert, now=1793016000)
    assert not asyncio.run(Router(store, Receiver(()), clock=lambda: 1793021400).run_once())
    assert store.jobs("staff", state="dead")[0]["code"] == "HASH_WINDOW_CLOSED"
    e["available_at"] = "2026-10-26T13:30:00Z"
    with pytest.raises(AlertError):
        Alert.parse(e)


def test_sec_adapter_preserves_real_shape_and_excludes_raw_summary():
    story = {
        "source": "sec:8k",
        "primary": True,
        "published_at": "2026-10-26T12:00:00Z",
        "received_at": "2026-10-26T12:01:00Z",
        "symbols": ["AAPL"],
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/0000320193-26-000001-index.htm",
        "filing": {"cik": "0000320193", "form": "8-K", "accession": "0000320193-26-000001"},
        "headline": "Buy now",
        "summary": "private raw text",
    }
    result = sec_story(story).payload()
    assert result["facts"]["cik"] == "0000320193"
    assert "private raw text" not in json.dumps(result) and "Buy now" not in json.dumps(result)
    story["published_at"] = None
    with pytest.raises(AlertError, match="SOURCE_NOT_VERIFIED"):
        sec_story(story)


def test_duel_results_require_matching_preopen_delivery(store):
    destination(store)
    e = synthetic_alert()
    e.update(
        kind="duel_hashes",
        facts={
            "session": "2026-10-26",
            "open_at": "2026-10-26T13:30:00Z",
            "a1_sha256": "a" * 64,
            "e1_sha256": "b" * 64,
        },
    )
    result = synthetic_alert()
    result.update(
        kind="duel_results",
        event_id="c" * 64,
        occurred_at="2026-10-26T20:00:00Z",
        available_at="2026-10-26T20:00:00Z",
        facts={
            "session": "2026-10-26",
            "close_at": "2026-10-26T20:00:00Z",
            "a1_sha256": "a" * 64,
            "e1_sha256": "b" * 64,
            "a1_return_pct": "0.1",
            "e1_return_pct": "-0.2",
        },
    )
    close = 1793044800
    with pytest.raises(AlertError, match="DUEL_COMMITMENT_MISSING"):
        store.enqueue("staff", Alert.parse(result), now=close)
    store.enqueue("staff", Alert.parse(e), now=1793016000)
    with pytest.raises(AlertError, match="DUEL_PREOPEN_DELIVERY_UNPROVEN"):
        store.enqueue("staff", Alert.parse(result), now=close)
    assert asyncio.run(Router(store, Receiver(), clock=lambda: 1793016000).run_once())
    assert store.enqueue("staff", Alert.parse(result), now=close) == 1
    result["facts"]["e1_sha256"] = "d" * 64
    with pytest.raises(AlertError, match="DUEL_COMMITMENT_CONFLICT"):
        store.enqueue("staff", Alert.parse(result), now=close)
