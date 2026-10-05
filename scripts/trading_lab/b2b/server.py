"""Dedicated loopback B2B listener: every resource authenticates and is audited."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import socket
from urllib.parse import parse_qs, urlsplit

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.b2b.api import B2BApi
from scripts.trading_lab.b2b.contracts import VERSION
from scripts.trading_lab.b2b.security import B2BError, Configuration, request_id
from scripts.trading_lab.platform.jobs import ArtifactIntegrityError


def strict_json(raw):
    def object_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=reject_constant)


class Handler(BaseHTTPRequestHandler):
    api: B2BApi
    server_version = "HyprLB2B/1"
    sys_version = ""

    def log_message(self, *args):
        pass

    def _respond(self, status, payload, *, head=False):
        raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Request-ID", payload["request_id"])
        self.send_header("Connection", "close")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.close_connection = True
        if not head:
            self.wfile.write(raw)

    def _request(self):
        rid, principal, operation, status, code = request_id(), None, "unknown", 500, "INTERNAL_ERROR"
        data = None
        try:
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or len(self.path) > 4096:
                raise B2BError("AUTH_REQUIRED", 401)
            principal = self.api.configuration.authenticate(auth[0])
            parsed = urlsplit(self.path)
            route, params = self.api.route(self.command, parsed.path)
            operation = route.operation
            project, values = self.api.admit(principal, route, params,
                parse_qs(parsed.query, keep_blank_values=True), rid, origin=self.headers.get("Origin"))
            payload = None
            if route.body is not None:
                lengths = self.headers.get_all("Content-Length", [])
                if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise B2BError("INVALID_BODY")
                length = int(lengths[0])
                if not 1 <= length <= 16384:
                    raise B2BError("PAYLOAD_TOO_LARGE", 413)
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise B2BError("JSON_REQUIRED")
                self.connection.settimeout(5)
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("incomplete body")
                    payload = strict_json(raw)
                except (ValueError, UnicodeDecodeError, TimeoutError):
                    raise B2BError("INVALID_BODY") from None
            data = self.api.execute(principal, project, route, params, values, payload, rid)
            status, code = route.status, "OK"
        except B2BError as error:
            status, code = error.status, error.code
        except ArtifactIntegrityError:
            status, code = 409, "ARTIFACT_INTEGRITY_ERROR"
        except AppApiError as error:
            # Public errors are fixed codes; provider diagnostics can contain
            # private paths or input text and never cross this boundary.
            status, code = getattr(error, "status", 400), "EVIDENCE_UNAVAILABLE"
        except KeyError:
            status, code = 404, "RESOURCE_NOT_FOUND"
        except (ValueError, TypeError, OverflowError):
            status, code = 400, "INVALID_REQUEST"
        except Exception:
            status, code = 500, "INTERNAL_ERROR"
        try:
            self.api.control.audit(rid, principal, operation, status, code)
        except Exception:
            # A missing durable audit is a failed operation, never a success.
            status, code = 503, "AUDIT_UNAVAILABLE"
        response = self.api.envelope(principal, data, rid) if status < 400 else {
            "schema": "b2b-error-v1", "api_version": VERSION, "request_id": rid, "error": code}
        self._respond(status, response, head=self.command == "HEAD")

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _request


def make_server(configuration, root, *, host="127.0.0.1", port=8791):
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("B2B v1 requires a literal loopback listener")
    if not isinstance(configuration, Configuration):
        raise ValueError("validated private configuration required")
    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET

        def server_close(self):
            super().server_close()
            if getattr(self.RequestHandlerClass, "api", None):
                self.RequestHandlerClass.api.close()

    handler = type("BoundB2BHandler", (Handler,), {})
    server = Server((host, port), handler)
    try:
        handler.api = B2BApi(configuration, root)
    except Exception:
        server.server_close()
        raise
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="private hashed-key configuration, 0600")
    parser.add_argument("--root", required=True, help="new private runtime, separate from all archives")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)
    server = make_server(Configuration.load(args.config), args.root, host=args.host, port=args.port)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
