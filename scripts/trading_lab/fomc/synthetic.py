"""Synthetic, local-only sources for the offline slice: a simulated clock, a local HTTP provider
that stands in for www.federalreserve.gov, and fixture builders. Nothing here touches the network.
"""

from __future__ import annotations

import base64
import hashlib

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import http.client
from http.server import BaseHTTPRequestHandler
import os
import socket
import socketserver
import tempfile
import threading
from typing import Callable
from xml.sax.saxutils import escape

from scripts.trading_lab.fomc import spec


class SimClock:
    """True time advances only through sleep(); the collector wall clock may be offset from it."""

    def __init__(self, start: datetime, *, wall_offset_s: float = 0.0):
        self.true = start
        self.wall_offset_s = wall_offset_s
        self._mono = 1000.0

    def wall(self) -> datetime:
        return self.true + timedelta(seconds=self.wall_offset_s)

    def mono(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:
        seconds = max(seconds, 0.0)
        self.true += timedelta(seconds=seconds)
        self._mono += seconds


@dataclass
class SyntheticResponse:
    status: int = 200
    body: bytes = b""
    headers: list[tuple[str, str]] = field(default_factory=list)
    date: str | None = "auto"  # "auto": the true server time; None: no Date line
    stall_s: float = 0.0  # real seconds to wait before sending the status line (deadline tests)


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class LocalProvider:
    """A local HTTP server answering like the allowlisted host; routes map a path to a response or a
    callable(request_count) -> response. It records every request it serves. It listens on a Unix
    domain socket, so it needs no network at all (loopback TCP is not required)."""

    def __init__(self, clock: SimClock):
        self.clock = clock
        self.routes: dict[str, SyntheticResponse | Callable[[int], SyntheticResponse]] = {}
        self.requests: list[str] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                provider.requests.append(self.path)
                route = provider.routes.get(self.path)
                count = provider.requests.count(self.path)
                resp = route(count) if callable(route) else route
                if resp is None:
                    resp = SyntheticResponse(status=404, body=b"not found", headers=[("Content-Type", "text/plain")])
                if resp.stall_s:
                    import time as _time
                    _time.sleep(resp.stall_s)
                self.send_response_only(resp.status)
                if resp.date == "auto":
                    self.send_header("Date", format_datetime(provider.clock.true, usegmt=True))
                elif resp.date is not None:
                    self.send_header("Date", resp.date)
                for name, value in resp.headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(resp.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(resp.body)

            def log_message(self, *_args):
                pass

            def address_string(self):
                return "local"

        self._dir = tempfile.mkdtemp(prefix="fomc-provider-")
        self.path = os.path.join(self._dir, "provider.sock")
        self.server = _UnixHTTPServer(self.path, Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def connector(self) -> "LocalConnector":
        return LocalConnector(self.path)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        try:
            os.unlink(self.path)
            os.rmdir(self._dir)
        except OSError:
            pass


class LocalConnector:
    """Test connector: the logical URL stays https://www.federalreserve.gov/...; only the socket goes to
    the local provider (plain HTTP over a Unix socket instead of TLS on 443). URL admission, identity,
    redirect validation, limiter grants and the physical-attempt deadline are unchanged."""

    tls = False

    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def resolve(self):
        return [self.socket_path]

    def connect(self, address, timeout: float):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(address)
        except BaseException:
            sock.close()
            raise
        return sock


FEED_PATH = "/feeds/press_monetary.xml"


def statement_path(day: str, letter: str = "a") -> str:
    return f"/newsevents/pressreleases/monetary{day}{letter}.htm"


def url(path: str) -> str:
    return f"https://{spec.ALLOWED_HOST}{path}"


def feed_xml(items: list[dict]) -> bytes:
    parts = ['<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel><title>Monetary</title>']
    for it in items:
        parts.append("<item>")
        for name in ("title", "link", "guid"):
            if it.get(name) is not None:
                parts.append(f"<{name}>{escape(it[name])}</{name}>")
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


def statement_html(*, title: str = spec.PRIMARY_TITLE_EXACT, date_text: str = "June 17, 2026",
                   release: str = "For release at 2:00 p.m. EDT", body: str = "The Committee decided...") -> bytes:
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Federal Reserve Board - ' + title + "</title></head>"
        '<body><div class="row"><div id="article"><div class="heading col-xs-12 col-sm-8 col-md-8">'
        f'<p class="article__time">{date_text}</p><h3 class="title">{title}</h3>'
        f'<p class="releaseTime">{release}   \n<ul class="list-unstyled"><li>Share</li></ul>'
        f"</div><div class=\"col-xs-12\"><p>{body}</p></div></div>"
        f'<div id="lastUpdate">Last Update: {date_text}</div></div></body></html>'
    ).encode("utf-8")


FEED_HEADERS = [("Content-Type", "application/rss+xml; charset=utf-8")]
HTML_HEADERS = [("Content-Type", "text/html; charset=UTF-8")]


def feed_response(items: list[dict], **kw) -> SyntheticResponse:
    return SyntheticResponse(body=feed_xml(items), headers=list(FEED_HEADERS), **kw)


# A representative challenge script for synthetic pages (not Cloudflare's own bytes, which stay local):
# tests pin canon.CHALLENGE_SCRIPT_SHA256 to CF_SCRIPT_SHA256 to exercise the same context rule.
CF_SCRIPT = "<script>(function(){var p=\"window.__CF$cv$params={r:'%s',t:'%s'}\";document.cf=p;})();</script>"
CF_SCRIPT_SHA256 = hashlib.sha256((CF_SCRIPT % ("", "")).encode()).hexdigest()


def obfuscate(text: str, key: int) -> str:
    """Cloudflare's e-mail obfuscation: the key byte, then each byte XOR the key, in lowercase hex."""
    return f"{key:02x}" + "".join(f"{b ^ key:02x}" for b in text.encode())


def cloudflare_html(*, key: int, ray: str, stamp: int, email: str = "media@frb.gov", page_url: str = "",
                    script: str | None = None, **kw) -> bytes:
    """A statement page carrying the three Cloudflare spans with per-response values."""
    html = statement_html(**kw).decode("utf-8")
    share = (f'<a class="shareDL__link" href="/cdn-cgi/l/email-protection#{obfuscate("?body=" + page_url, key)}">'
             "Email</a>")
    contact = (f'<p>For media inquiries, please email <a href="/cdn-cgi/l/email-protection#{obfuscate(email, key ^ 0x5A)}">'
               f'<span class="__cf_email__" data-cfemail="{obfuscate(email, key ^ 0x33)}">[email&#160;protected]</span></a>'
               " or call 202-452-2955.</p>")
    challenge = (script or CF_SCRIPT) % (ray, base64.b64encode(str(stamp).encode()).decode())
    html = html.replace('<div id="lastUpdate">', share + contact + '<div id="lastUpdate">', 1)
    return html.replace("</body>", challenge + "</body>", 1).encode("utf-8")


def cloudflare_route(**kw):
    """A route answering each request with new Cloudflare keys and parameters (same statement otherwise)."""
    def route(count: int) -> SyntheticResponse:
        n = count + 1
        body = cloudflare_html(key=(37 * n) % 251 + 1, ray=f"{(0x9E3779B97F4A7C15 * n) % 2**64:016x}", stamp=1790879000 + n, **kw)
        return SyntheticResponse(body=body, headers=list(HTML_HEADERS))
    return route


def page_response(**kw) -> SyntheticResponse:
    date = kw.pop("date", "auto")
    return SyntheticResponse(body=statement_html(**kw), headers=list(HTML_HEADERS), date=date)


START = datetime(2026, 6, 17, 17, 59, 0, tzinfo=timezone.utc)
