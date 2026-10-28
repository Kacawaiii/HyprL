"""Short-lived capability URLs, served only on a dedicated loopback listener."""

from __future__ import annotations

import json
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .core import Blocked, atomic_json, digest, private_dir, read_regular


class MediaPool:
    def __init__(self, directory, *, clock=time.time):
        self.directory = private_dir(directory)
        self.clock = clock

    def issue(self, bundle, *, ttl=3600):
        if not 300 <= ttl <= 21600:
            raise Blocked("media_ttl_outside_limits")
        self.prune()
        token = secrets.token_hex(32)
        blob = self.directory / (bundle.sha256 + ".mp4")
        if not blob.exists():
            # Immutable validated bytes, independent of factory retention.
            with blob.open("xb") as handle:
                blob.chmod(0o600)
                handle.write(bundle.video)
        atomic_json(
            self.directory / (token + ".json"),
            {
                "sha256": bundle.sha256,
                "expires_at": self.clock() + ttl,
            },
        )
        return "https://hyprlcore.com/social-media/" + token + "/video.mp4"

    def read(self, path):
        match = re.fullmatch(r"/social-media/([a-f0-9]{64})/video\.mp4", path)
        if not match:
            raise Blocked("media_not_found")
        value = json.loads(read_regular(self.directory / (match[1] + ".json"), 1000, private=True))
        if self.clock() >= value["expires_at"] or not re.fullmatch(
            r"[a-f0-9]{64}", value["sha256"]
        ):
            raise Blocked("media_expired")
        video = read_regular(self.directory / (value["sha256"] + ".mp4"), 8_999_999, private=True)
        if digest(video) != value["sha256"]:
            raise Blocked("media_digest_mismatch")
        return video

    def prune(self):
        active = set()
        for path in self.directory.glob("*.json"):
            if not re.fullmatch(r"[a-f0-9]{64}\.json", path.name):
                continue
            value = json.loads(read_regular(path, 1000, private=True))
            if self.clock() >= value["expires_at"]:
                path.unlink()
            else:
                active.add(value["sha256"])
        for path in self.directory.glob("*.mp4"):
            if re.fullmatch(r"[a-f0-9]{64}\.mp4", path.name) and path.stem not in active:
                path.unlink()


def byte_range(value, size):
    if not value:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise Blocked("invalid_media_range")
    first, last = match.groups()
    start = int(first) if first else max(0, size - int(last))
    end = min(int(last), size - 1) if first and last else size - 1
    if start > end or start >= size:
        raise Blocked("invalid_media_range")
    return start, end, 206


def serve(directory, port=8767):
    pool = MediaPool(directory)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Capability URLs must never enter access logs.

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                video = pool.read(self.path)
            except (Blocked, ValueError, KeyError):
                self.send_error(404)
                return
            try:
                start, end, status = byte_range(self.headers.get("Range"), len(video))
            except Blocked:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(video)}")
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(video)}")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(video[start : end + 1])
                except (BrokenPipeError, ConnectionResetError):
                    pass

    # No caller can accidentally bind to 0.0.0.0.
    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.timeout = 30
        while True:
            server.handle_request()
            pool.prune()
