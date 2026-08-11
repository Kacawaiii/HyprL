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

In production the same server also serves the built frontend, so the cockpit
and its API share one origin and cross-origin rules stop applying to the app
itself. The dev split (Vite on 5173, API on 8787) is unchanged; that is what
the CORS allowlist above still exists for. API paths are matched before the
static layer and never fall through to index.html -- a mistyped endpoint that
returned a page of HTML with a 200 would send every caller looking for the
bug in the wrong place.
"""

from __future__ import annotations

import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from scripts.trading_lab.app_api.contracts import (
    APP_API_VERSION,
    DEFAULT_PAPER_EVENTS,
    MAX_SSE_REPLAY_EVENTS,
    AppApiError,
)
from scripts.trading_lab.app_api.service import AppService

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
# The dev server origins the cockpit is served from. Never "*".
ALLOWED_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")
ALLOWED_METHODS = ("GET", "HEAD", "OPTIONS")
# A stream is still a read. It ends on its own so a forgotten tab cannot hold a
# thread for a week.
SSE_MAX_SECONDS = 900
SSE_POLL_SECONDS = 1.0
_SSE_KINDS = {
    "CANDLE_INGESTED": "candle",
    "PREDICTION_CREATED": "prediction",
    "SIGNAL_CREATED": "signal",
    "POSITION_TARGET_CREATED": "target",
    "SIMULATED_FILL": "fill",
    "PORTFOLIO_SNAPSHOT": "portfolio",
    "GAP_DETECTED": "gap",
    "PROTECTED_HOLDOUT_BOUNDARY_REACHED": "embargo",
    "ERROR": "error",
}


def _query_first(query: dict, name: str, default=None):
    """First value of a query parameter, for handlers outside build_routes.

    build_routes keeps its own closure of the same shape; this one exists so
    the path dispatcher does not reach into it. Calling that closure from here
    raised NameError on every request, which the handler turned into a 500 --
    including for perfectly valid paths.
    """
    values = query.get(name)
    return values[0] if values else default


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

    def paper_sub(product, leaf, query):
        if leaf == "events":
            return service.paper_events(product, limit=_first(query, "limit"),
                                        after_event_id=_first(query, "after"))
        if leaf == "equity":
            return service.paper_equity(product,
                                        max_points=_first(query, "max_points"))
        if leaf == "fills":
            return service.paper_fills(product, limit=_first(query, "limit"))
        if leaf == "predictions":
            return service.paper_predictions(product, limit=_first(query, "limit"))
        raise AppApiError("no such endpoint")

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
        "/api/v1/paper": lambda query: service.paper_status(),
        "/api/v1/paper/status": lambda query: service.paper_status(),
        "/api/v1/paper/products": lambda query: service.paper_products(),
        "/api/v1/paper/events": lambda query: service.paper_events(
            limit=_first(query, "limit"), after_event_id=_first(query, "after")),
        # Operations views. Read-only like everything else: lifecycle lives on
        # the command line, where starting a trading process takes a
        # deliberate act rather than a cross-site request.
        "/api/v1/ops/health-history": lambda query: service.ops_health_history(
            component=_first(query, "component"), limit=_first(query, "limit")),
        "/api/v1/ops/runtime": lambda query: service.ops_runtime(),
        "/api/v1/ops/recovery": lambda query: service.ops_recovery(),
        "/api/v1/ops/storage": lambda query: service.ops_storage(),
        "/api/v1/ops/settings": lambda query: service.ops_settings(),
        # The registry of markets and where their bars come from.
        "/api/v1/portfolio": lambda query: service.portfolio(),
        "/api/v1/portfolio/backtests": lambda query: service.portfolio_backtests(),
        "/api/v1/instruments": lambda query: service.instruments(),
        "/api/v1/providers": lambda query: service.providers(),
    }, markets_detail, chart, backtest_sub, paper_sub


class AppApiHandler(BaseHTTPRequestHandler):
    service: AppService = None          # injected by make_server
    site = None                         # StaticSite in production, None in dev
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
        routes, markets_detail, chart, backtest_sub, paper_sub = build_routes(
            self.service)
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
        # /api/v1/portfolio/backtests/{version}[/{leaf}]
        if len(parts) == 5 and parts[:4] == ["api", "v1", "portfolio", "backtests"]:
            return self.service.portfolio_backtest_detail(parts[4])
        if len(parts) == 6 and parts[:4] == ["api", "v1", "portfolio", "backtests"]:
            leaf = parts[5]
            if leaf == "equity":
                return self.service.portfolio_equity(
                    parts[4], max_points=_query_first(query, "max_points"))
            if leaf == "fills":
                return self.service.portfolio_fills(
                    parts[4], limit=_query_first(query, "limit"),
                    cursor=_query_first(query, "cursor"))
            if leaf == "attribution":
                return self.service.portfolio_attribution(parts[4])
            raise AppApiError("no such endpoint")
        # /api/v1/instruments/{instrument_id} -- a canonical id contains a
        # colon, which a client may send raw or percent-encoded. Decoded here
        # and nowhere else: these values are looked up in a closed registry
        # and never reach the filesystem, unlike the static layer which
        # deliberately decodes exactly once and then checks containment.
        if len(parts) == 4 and parts[:3] == ["api", "v1", "instruments"]:
            return self.service.instrument_detail(unquote(parts[3]))
        # /api/v1/providers/{provider_id}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "providers"]:
            return self.service.provider_detail(unquote(parts[3]))
        # /api/v1/paper/{product}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "paper"]:
            return self.service.paper_product(parts[3])
        # /api/v1/paper/{product}/{events|equity|fills|predictions}
        if len(parts) == 5 and parts[:3] == ["api", "v1", "paper"]:
            return paper_sub(parts[3], parts[4], query)
        raise AppApiError("no such endpoint")

    # --- verbs -----------------------------------------------------------

    def _stream_events(self, query):
        """Server-sent events: one direction, bounded replay, no control channel.

        SSE rather than a WebSocket because nothing ever travels from the
        browser to the engine -- a bidirectional socket would be a control
        path nobody asked for. A reconnect replays a bounded tail; a client
        further behind than that refetches the REST snapshot instead.
        """
        import time
        last_id = self.headers.get("Last-Event-ID")
        if last_id is None:
            values = query.get("after")
            last_id = values[0] if values else None
        try:
            cursor = int(last_id) if last_id else None
        except (TypeError, ValueError):
            cursor = None

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.end_headers()

        deadline = time.monotonic() + SSE_MAX_SECONDS
        try:
            payload = self.service.paper_status()
            self._sse("paper_status", payload)
            if cursor is None:
                recent = self.service.paper_events(limit=DEFAULT_PAPER_EVENTS)
                cursor = recent["page"]["last_event_id"]
                for event in recent["events"]:
                    self._sse(_SSE_KINDS.get(event["event_type"], "event"), event,
                              event_id=event["event_id"])
            while time.monotonic() < deadline:
                batch = self.service.paper_events(
                    limit=MAX_SSE_REPLAY_EVENTS,
                    after_event_id=cursor if cursor is not None else 0)
                for event in batch["events"]:
                    self._sse(_SSE_KINDS.get(event["event_type"], "event"), event,
                              event_id=event["event_id"])
                if batch["page"]["last_event_id"] is not None:
                    cursor = batch["page"]["last_event_id"]
                self._sse("heartbeat", {"cursor": cursor})
                time.sleep(SSE_POLL_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:                       # pragma: no cover - defensive
            try:
                self._sse("error", {"error": "stream ended"})
            except OSError:
                return

    def _sse(self, kind: str, payload, *, event_id=None) -> None:
        chunk = ""
        if event_id is not None:
            chunk += f"id: {event_id}\n"
        chunk += f"event: {kind}\n"
        chunk += f"data: {json.dumps(payload)}\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    # --- static site (production single-origin mode) ----------------------

    def _serve_static(self, path: str, *, body: bool) -> bool:
        """Serve the built frontend. Returns False if there is no build.

        Order matters: containment is checked first (a 403 must not depend on
        whether the file happens to exist), then existence, then SPA fallback
        for anything that looks like a route rather than an asset.
        """
        from scripts.trading_lab.ops.static_assets import ForbiddenPathError

        site = self.site
        if site is None or not site.available:
            return False
        try:
            served = site.serve(path)
        except ForbiddenPathError:
            self._respond(403, {"error": "forbidden"}, body=body)
            return True
        except FileNotFoundError:
            if site.looks_like_asset(path):
                # A missing script must not be answered with HTML: the browser
                # would report a syntax error instead of a 404.
                self._respond(404, {"error": "not found"}, body=body)
                return True
            served = site.spa_fallback(path)
        except OSError:                              # pragma: no cover
            self._respond(500, {"error": "internal error"}, body=body)
            return True

        try:
            payload = served["path"].read_bytes()
        except OSError:                              # pragma: no cover
            self._respond(500, {"error": "internal error"}, body=body)
            return True
        self.send_response(served["status"])
        self.send_header("Content-Type", served["content_type"])
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", served["cache_control"])
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if body:
            self.wfile.write(payload)
        return True

    def do_GET(self, *, body: bool = True):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/v1/paper/events/stream":
            if not body:
                self._respond(200, {"stream": "text/event-stream"}, body=False)
                return
            self._stream_events(query)
            return
        from scripts.trading_lab.ops.static_assets import is_api_path

        if not is_api_path(parsed.path) and self._serve_static(parsed.path, body=body):
            return
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


def make_server(data_root, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                dist_root=None):
    """Build a loopback-bound read-only server over a fixed data root.

    ``dist_root`` turns on single-origin production mode. It is resolved once,
    here, so no request can influence which directory is served.
    """
    service = AppService(pathlib.Path(data_root))
    site = None
    if dist_root is not None:
        from scripts.trading_lab.ops.static_assets import StaticSite
        site = StaticSite(dist_root)
    handler = type("BoundAppApiHandler", (AppApiHandler,),
                   {"service": service, "site": site})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):  # pragma: no cover - entry point
    import argparse
    import signal as signal_module

    parser = argparse.ArgumentParser(description="HyprL read-only application API")
    parser.add_argument("--data-root", default="data/crypto")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dist-root", default=None,
                        help="serve a frontend build from the same origin")
    # Present so the supervisor can prove a pid belongs to this application
    # before signalling it. Parsed and ignored.
    parser.add_argument("--marker", default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.host not in ("127.0.0.1", "localhost", "::1"):
        # Not a default that can drift: an explicit non-loopback bind has to be
        # stated, and it is stated loudly.
        print(f"[hyprl] WARNING: binding {arguments.host} exposes this runtime "
              "beyond the local machine")
    server = make_server(arguments.data_root, host=arguments.host,
                         port=arguments.port, dist_root=arguments.dist_root)
    if arguments.dist_root:
        print(f"HyprL on http://{arguments.host}:{arguments.port}/")
    else:
        print(f"HyprL app API on http://{arguments.host}:{arguments.port}/api/v1/health")

    def _shutdown(signum, frame):
        raise KeyboardInterrupt

    signal_module.signal(signal_module.SIGTERM, _shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
