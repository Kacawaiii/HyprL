"""Signed member bridge -> real encrypted router Store, offline."""

import hashlib
import hmac
import json
import secrets

import pytest

from scripts.alert_router.member_bridge import MemberAPI
from scripts.alert_router.store import Store, create_key


@pytest.fixture
def bridge(tmp_path):
    key = tmp_path / "router.key"
    create_key(key)
    store = Store(tmp_path / "store", key, allowed_customers=["800/901", "800/902"])
    return MemberAPI(store, "1" * 64, clock=lambda: 1700000000), store


def signed(api, value, nonce=None):
    raw = json.dumps(value, separators=(",", ":")).encode()
    nonce = nonce or secrets.token_urlsafe(32)
    headers = {"X-HyprL-At": "1700000000", "X-HyprL-Nonce": nonce}
    headers["X-HyprL-Signature"] = hmac.new(
        api.key, b"1700000000." + nonce.encode() + b"." + raw, hashlib.sha256
    ).hexdigest()
    return raw, headers


def test_real_store_configuration_isolation_and_secret_omission(bridge):
    api, store = bridge
    raw, headers = signed(
        api,
        {
            "customer": "800/901",
            "action": "add",
            "channel": "webhook",
            "config": {"url": "https://example.com/private-destination", "signing_key": "s" * 32},
            "kinds": ["sec_filing"],
            "watchlist": ["DEMO"],
        },
    )
    status, data = api.handle(raw, headers)
    assert status == 200
    dest = data["id"]
    status, listing = api.handle(*signed(api, {"customer": "800/901", "action": "list"}))
    assert status == 200 and len(listing["destinations"]) == 1
    assert "private-destination" not in json.dumps(listing) and "signing_key" not in json.dumps(
        listing
    )
    assert (
        api.handle(*signed(api, {"customer": "800/902", "action": "list"}))[1]["destinations"] == []
    )
    for action in ["pause", "remove"]:
        fields = {"destination": dest, **({"paused": True} if action == "pause" else {})}
        assert (
            api.handle(*signed(api, {"customer": "800/902", "action": action, **fields}))[0] == 400
        )
        assert (
            api.handle(*signed(api, {"customer": "800/901", "action": action, **fields}))[0] == 200
        )
    assert store.list("800/901") == []


def test_hmac_timestamp_replay_policy_and_fields_fail_closed(bridge):
    api, _ = bridge
    raw, headers = signed(api, {"customer": "800/901", "action": "list"})
    assert api.handle(raw + b" ", headers)[0] == 401
    assert api.handle(raw, {**headers, "X-HyprL-At": "1700000100"})[0] == 401
    assert api.handle(raw, headers)[0] == 200
    assert api.handle(raw, headers)[0] == 401
    for data in [
        {"customer": "800/903", "action": "list"},
        {"customer": "800/901", "action": "list", "owner": "800/902"},
        {
            "customer": "800/901",
            "action": "add",
            "channel": "email",
            "config": {},
            "kinds": ["sec_filing"],
            "watchlist": [],
        },
    ]:
        assert api.handle(*signed(api, data))[0] == 400


def test_node_client_to_real_unix_http_bridge(bridge, tmp_path):
    import os
    import socketserver
    import subprocess
    import threading

    from scripts.alert_router.member_bridge import Handler

    api, _ = bridge
    # Real current timestamp for the Node signing client.
    import time

    api.clock = time.time
    socket = tmp_path / "member.sock"
    with socketserver.UnixStreamServer(str(socket), Handler) as server:
        server.api = api
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            script = """
            const {RouterClient}=require('./apps/hyprl-site/server/member/router-client');
            const c=new RouterClient({routerSocket:process.env.TEST_MEMBER_SOCKET,routerKey:'1'.repeat(64)});
            (async()=>{
              const a=await c.call('800/901','add',{channel:'webhook',config:{url:'https://example.com/test',signing_key:'s'.repeat(32)},kinds:['sec_filing'],watchlist:['DEMO']});
              const b=await c.call('800/902','list');
              if(b.destinations.length) throw Error('isolation');
              await c.call('800/901','pause',{destination:a.id,paused:true});
              const own=await c.call('800/901','list');
              if(!own.destinations[0].paused) throw Error('pause');
              await c.call('800/901','remove',{destination:a.id});
              console.log('PASS');
            })().catch(()=>process.exit(1));
            """
            result = subprocess.run(
                ["node", "-e", script],
                env={**os.environ, "TEST_MEMBER_SOCKET": str(socket)},
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "PASS"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_future_skew_nonce_remains_reserved_through_inclusive_signature_window(bridge):
    api, _ = bridge
    now = 1700000000
    api.clock = lambda: now
    raw, headers = signed(api, {"customer": "800/901", "action": "list"})
    headers["X-HyprL-At"] = str(now + 30)
    headers["X-HyprL-Signature"] = hmac.new(
        api.key,
        headers["X-HyprL-At"].encode() + b"." + headers["X-HyprL-Nonce"].encode() + b"." + raw,
        hashlib.sha256,
    ).hexdigest()
    assert api.handle(raw, headers)[0] == 200
    now += 60
    assert api.handle(raw, headers)[0] == 401
