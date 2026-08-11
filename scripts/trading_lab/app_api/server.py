"""A small local HTTP surface for the read-only application API.

Standard library only, on purpose: the trading core carries no web framework,
and a cockpit that reads committed artefacts does not need one. The router is
a fixed table of GET handlers -- there is no dynamic dispatch onto arbitrary
attributes and no filesystem path ever comes from a client.

Three boundaries are deliberate:

* **Loopback by default.** Binding 0.0.0.0 would publish a machine's research
  state to its whole network on the assumption that the network is friendly.
* **Explicit CORS origins.** A wildcard would let any page a browser happens to
  be visiting read this API.
* **GET only.** Every mutating verb is refused at the router, so no endpoint
  can grow a write path by accident.
"""

from __future__ import annotations

import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from scripts.trading_lab.app_api.contracts import APP_API_VERSION, AppApiError
from scripts.trading_lab.app_api.service import AppService

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
# The dev server origins the cockpit is served from. Never "*".
ALLOWED_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")
ALLOWED_METHODS = ("GET", "HEAD", "OPTIONS")


def build_routes(service: AppService):
    """The complete route table. Read-only by construction."""

    def _first(query, name, default=None):
        values = query.get(name)
        return values[0] if values else default

    def markets_detail(product, query):
        return service.market_candles(
            product,
            start=_first(query, "start"), end=_first(query, "end"),
            limit=_first(query, "limit"), cursor=_first(query, "cursor"))

    def chart(product, query):
        return service.market_chart(
            product,
            start=_first(query, "start"), end=_first(query, "end"),
            max_points=_first(query, "max_points"))

    def backtest_sub(version, product, leaf, query):
        if leaf == "equity":
            return service.backtest_equity(
                version, product, max_points=_first(query, "max_points"))
        if leaf == "fills":
            return service.backtest_fills(
                version, product, limit=_first(query, "limit"),
                cursor=_first(query, "cursor"))
        raise AppApiError("no such endpoint")

    return {
        "/api/v1/health": lambda query: service.health(),
        "/api/v1/system": lambda query: service.system(),
        "/api/v1/overview": lambda query: service.overview(),
        "/api/v1/markets": lambda query: service.markets(),
        "/api/v1/signals": lambda query: service.signals(
            limit=_first(query, "limit")),
        "/api/v1/risk/targets": lambda query: service.risk_targets(
            limit=_first(query, "limit")),
        "/api/v1/research/benchmarks": lambda query: {
            "benchmarks": service.benchmark_summaries()},
        "/api/v1/backtests": lambda query: service.backtests(),
    }, markets_detail, chart, backtest_sub


class AppApiHandler(BaseHTTPRequestHandler):
    service: AppService = None          # injected by make_server
    server_version = "HyprLAppAPI/1.0"
    sys_version = ""                    # do not advertise the Python build

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return                              # quiet by default; no request logging

    # --- helpers ---------------------------------------------------------

    def _cors(self):
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", ", ".join(ALLOWED_METHODS))
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, status: int, payload: dict, *, body: bool = True):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        if body:
            self.wfile.write(raw)

    def _dispatch(self, path: str, query: dict):
        routes, markets_detail, chart, backtest_sub = build_routes(self.service)
        if path in routes:
            return routes[path](query)
        parts = [segment for segment in path.strip("/").split("/") if segment]
        # /api/v1/markets/{product}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "markets"]:
            return markets_detail(parts[3], query)
        # /api/v1/markets/{product}/chart
        if len(parts) == 5 and parts[:3] == ["api", "v1", "markets"] \
                and parts[4] == "chart":
            return chart(parts[3], query)
        # /api/v1/research/benchmarks/{version}/{product}
        if len(parts) == 6 and parts[:4] == ["api", "v1", "research", "benchmarks"]:
            return self.service.benchmark_detail(parts[4], parts[5])
        # /api/v1/backtests/{version}/{product}
        if len(parts) == 5 and parts[:3] == ["api", "v1", "backtests"]:
            return self.service.backtest_detail(parts[3], parts[4])
        # /api/v1/backtests/{version}/{product}/{equity|fills}
        if len(parts) == 6 and parts[:3] == ["api", "v1", "backtests"]:
            return backtest_sub(parts[3], parts[4], parts[5], query)
        raise AppApiError("no such endpoint")

    # --- verbs -----------------------------------------------------------

    def do_GET(self, *, body: bool = True):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            payload = self._dispatch(parsed.path, query)
        except AppApiError as error:
            self._respond(getattr(error, "status", 400),
                          {"error": str(error), "api_version": APP_API_VERSION},
                          body=body)
            return
        except Exception:  # pragma: no cover - defensive
            # Never leak a traceback, a module path or a filesystem location.
            self._respond(500, {"error": "internal error",
                                "api_version": APP_API_VERSION}, body=body)
            return
        self._respond(200, payload, body=body)

    def do_HEAD(self):  # noqa: N802
        self.do_GET(body=False)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _refuse(self):
        self._respond(405, {"error": "the application API is read-only",
                            "allowed": list(ALLOWED_METHODS),
                            "api_version": APP_API_VERSION})

    do_POST = do_PUT = do_PATCH = do_DELETE = _refuse


def make_server(data_root, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Build a loopback-bound read-only server over a fixed data root."""
    service = AppService(pathlib.Path(data_root))
    handler = type("BoundAppApiHandler", (AppApiHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):  # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(description="HyprL read-only application API")
    parser.add_argument("--data-root", default="data/crypto")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arguments = parser.parse_args(argv)
    server = make_server(arguments.data_root, host=arguments.host, port=arguments.port)
    print(f"HyprL app API on http://{arguments.host}:{arguments.port}/api/v1/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
