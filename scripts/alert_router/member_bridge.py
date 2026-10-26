"""Signed private Unix-socket member configuration API. Never delivers alerts.

Only the web server possesses the bridge key; customer is bound to its verified
Discord session. Existing router policy and ownership checks remain in force.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import socketserver
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from .model import AlertError
from .service import configured


class MemberAPI:
    def __init__(self, store, key, *, clock=time.time):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise AlertError("INVALID_BRIDGE_KEY")
        self.store, self.key, self.clock = store, bytes.fromhex(key), clock
        self.nonces = {}

    def handle(self, raw, headers):
        at, nonce, signature = (
            headers.get(k, "") for k in ("X-HyprL-At", "X-HyprL-Nonce", "X-HyprL-Signature")
        )
        if (
            len(raw) > 16384
            or not re.fullmatch(r"\d{10}", at)
            or abs(self.clock() - int(at)) > 30
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", nonce)
        ):
            return 401, {"error": "UNAUTHORIZED"}
        expected = hmac.new(
            self.key, at.encode() + b"." + nonce.encode() + b"." + raw, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return 401, {"error": "UNAUTHORIZED"}
        self.nonces = {n: expiry for n, expiry in self.nonces.items() if expiry > self.clock()}
        if nonce in self.nonces or len(self.nonces) >= 10000:
            return 401, {"error": "REPLAY"}
        self.nonces[nonce] = self.clock() + 61
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise AlertError("INVALID_FIELDS")
            customer, action = data.get("customer"), data.get("action")
            self.store.customer(customer)
            fields = {
                "list": set(),
                "add": {"channel", "config", "kinds", "watchlist"},
                "pause": {"destination", "paused"},
                "remove": {"destination"},
            }
            if action not in fields or set(data) != {"customer", "action"} | fields[action]:
                raise AlertError("INVALID_FIELDS")
            if action == "list":
                return 200, {"destinations": self.store.list(customer), "email_enabled": False}
            if action == "add":
                # No addresses collected while the e-mail sender is deferred.
                if data["channel"] in ("email", "api"):
                    raise AlertError("CHANNEL_NOT_ENABLED")
                dest = self.store.add(
                    customer,
                    data["channel"],
                    data["config"],
                    kinds=data["kinds"],
                    watchlist=data["watchlist"],
                )
                return 200, {"id": dest}
            if action == "pause":
                self.store.pause(customer, data["destination"], data["paused"])
            else:
                self.store.remove(customer, data["destination"])
            return 200, {"ok": True}
        except (AlertError, ValueError, TypeError, KeyError):
            return 400, {"error": "REQUEST_REFUSED"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # never log URL, headers, customer or body

    def do_POST(self):
        self.connection.settimeout(5)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if (
                self.path != "/member/v1"
                or not 0 < size <= 16384
                or self.headers.get("Transfer-Encoding")
            ):
                raise ValueError
            raw = self.rfile.read(size)
            status, data = self.server.api.handle(raw, self.headers)
        except (ValueError, OSError):
            status, data = 400, {"error": "REQUEST_REFUSED"}
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--state", default=os.environ.get("HYPRL_ALERT_STATE"))
    parser.add_argument("--key", default=os.environ.get("HYPRL_ALERT_KEY"))
    parser.add_argument("--policy", default=os.environ.get("HYPRL_ALERT_POLICY"))
    args = parser.parse_args()
    store, _ = configured(args)
    api = MemberAPI(store, os.environ.get("MEMBER_ROUTER_KEY", ""))
    if (
        not args.socket.is_absolute()
        or os.path.lexists(args.socket)
        or not args.socket.parent.is_dir()
        or args.socket.parent.stat().st_mode & 0o077
    ):
        raise AlertError("PRIVATE_SOCKET_DIRECTORY_REQUIRED")
    checkout = Path(__file__).resolve().parents[2]
    target = args.socket.resolve()
    if target == checkout or checkout in target.parents:
        raise AlertError("PRIVATE_SOCKET_INSIDE_GIT")
    os.umask(0o077)
    server = socketserver.UnixStreamServer(str(args.socket), Handler)
    try:
        with server:
            os.chmod(args.socket, 0o600)
            server.api = api
            server.serve_forever()
    finally:
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
