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
* **GET only by default.** An explicitly configured, authenticated loopback
  Model Lab can POST synthetic job controls under /api/v1/lab only.

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
import os
import pathlib
import socket
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
_PORTFOLIO_SSE_KINDS = {
    "PORTFOLIO_BATCH_OPENED": "portfolio_batch_opened",
    "PORTFOLIO_INSTRUMENT_READY": "portfolio_instrument_ready",
    "PORTFOLIO_BATCH_READY": "portfolio_batch_ready",
    "PORTFOLIO_BATCH_INCOMPLETE": "portfolio_batch_incomplete",
    "PORTFOLIO_TARGET_SET_CREATED": "portfolio_target",
    "PORTFOLIO_TARGET_SCALED": "portfolio_target_scaled",
    "PORTFOLIO_FILL": "portfolio_fill",
    "PORTFOLIO_SNAPSHOT": "portfolio_state",
    "PORTFOLIO_VALUATION_UNAVAILABLE": "portfolio_valuation_unavailable",
    "PROTECTED_HOLDOUT_BOUNDARY_REACHED": "portfolio_embargo",
    "PORTFOLIO_SESSION_STARTED": "portfolio_status",
    "PORTFOLIO_SESSION_STOPPED": "portfolio_status",
    "ERROR": "error",
}

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


def _reject_json_constant(value):
    raise ValueError("JSON constants must be finite")


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

    def information_snapshot(query):
        allowed = {"as_of", "products", "visibility_mode", "fomc_horizon", "edgar_horizon"}
        if set(query) - allowed or any(len(v) != 1 for v in query.values()):
            raise AppApiError("snapshot accepts one value per as_of, products, visibility_mode and named source horizon")
        return service.snapshots.snapshot(
            as_of=_first(query, "as_of"), products=_first(query, "products"),
            visibility_mode=_first(query, "visibility_mode"),
            fomc_horizon=_first(query, "fomc_horizon"), edgar_horizon=_first(query, "edgar_horizon"))

    return {
        "/api/v1/snapshots": information_snapshot,
        "/api/v1/contracts/providers": lambda query: service.snapshots.providers(),
        "/api/v1/health": lambda query: service.health(),
        "/api/v1/system": lambda query: service.system(),
        "/api/v1/overview": lambda query: service.overview(),
        "/api/v1/markets": lambda query: service.markets(),
        "/api/v1/signals": lambda query: service.signals(
            limit=_first(query, "limit"), product=_first(query, "product"),
            cursor=_first(query, "cursor")),
        "/api/v1/risk/targets": lambda query: service.risk_targets(
            limit=_first(query, "limit"), product=_first(query, "product"),
            cursor=_first(query, "cursor")),
        "/api/v1/research/benchmarks": lambda query: {
            "benchmarks": service.benchmark_summaries()},
        # The local equity research corpus. Status only -- there is no verb
        # here that captures, downloads, repairs or rebuilds anything, and the
        # read-only method table below is what guarantees that.
        "/api/v1/research/equities/corpus":
            lambda query: service.research_equity_corpus(),
        "/api/v1/backtests": lambda query: service.backtests(),
        "/api/v1/paper": lambda query: service.paper_status(),
        "/api/v1/paper/status": lambda query: service.paper_status(),
        "/api/v1/paper/replay": lambda query: service.replay.summary(),
        "/api/v1/paper/portfolio": lambda query: service.paper_portfolio(),
        "/api/v1/paper/portfolio/positions":
            lambda query: service.paper_portfolio_positions(),
        "/api/v1/paper/portfolio/pending":
            lambda query: service.paper_portfolio_pending(),
        "/api/v1/paper/portfolio/events": lambda query:
            service.paper_portfolio_events(limit=_first(query, "limit"),
                                           after_event_id=_first(query, "after")),
        "/api/v1/paper/portfolio/fills":
            lambda query: service.paper_portfolio_fills(limit=_first(query, "limit")),
        "/api/v1/paper/portfolio/equity": lambda query:
            service.paper_portfolio_equity(max_points=_first(query, "max_points")),
        "/api/v1/paper/legacy": lambda query: service.paper_legacy(),
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
        "/api/v1/calendars": lambda query: service.calendars(),
        # Official event sources, read-only: status, a point-in-time snapshot (T, H) and its verified
        # offline replay. /api/v1/sources/fomc/items/{sid} is matched below.
        "/api/v1/sources/fomc": lambda query: service.fomc.status(),
        "/api/v1/sources/fomc/snapshot": lambda query: service.fomc.snapshot(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon"),
            limit=_first(query, "limit"), cursor=_first(query, "cursor")),
        "/api/v1/sources/fomc/timeline": lambda query: service.fomc.timeline(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon"),
            limit=_first(query, "limit"), cursor=_first(query, "cursor")),
        "/api/v1/sources/fomc/replay": lambda query: service.fomc.replay(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon")),
        # SEC EDGAR (offline slice): /api/v1/sources/edgar/filings/{accession} is matched below.
        "/api/v1/sources/edgar": lambda query: service.edgar.status(),
        "/api/v1/sources/edgar/snapshot": lambda query: service.edgar.snapshot(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon"),
            limit=_first(query, "limit"), cursor=_first(query, "cursor")),
        "/api/v1/sources/edgar/timeline": lambda query: service.edgar.timeline(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon"),
            limit=_first(query, "limit"), cursor=_first(query, "cursor")),
        "/api/v1/sources/edgar/replay": lambda query: service.edgar.replay(
            as_of=_first(query, "as_of"), horizon=_first(query, "horizon")),
    }, markets_detail, chart, backtest_sub, paper_sub


class AppApiHandler(BaseHTTPRequestHandler):
    lab = None
    research = None
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
        if path.startswith('/api/v1/observability/') or any(
                path == '/api/v1/research/' + leaf or path.startswith('/api/v1/research/' + leaf + '/')
                for leaf in ('hypotheses', 'experiments', 'proposals', 'comparison')):
            return self.research.dispatch(path, parse_qs(urlparse(self.path).query, keep_blank_values=True))
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
        # /api/v1/research/equities/{instrument_id}/bars -- canonical ids carry
        # a colon, so the segment may arrive percent-encoded. Decoded once and
        # matched against a closed four-entry set; it never reaches a path.
        if len(parts) == 6 and parts[:4] == ["api", "v1", "research", "equities"] \
                and parts[5] == "bars":
            return self.service.research_equity_bars(
                unquote(parts[4]), start=_query_first(query, "start"),
                end=_query_first(query, "end"),
                limit=_query_first(query, "limit"),
                cursor=_query_first(query, "cursor"))
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
        # /api/v1/instruments/{instrument_id}/sessions -- bounded by the
        # service, which refuses a window wider than a year rather than
        # enumerating every session since 1885.
        if len(parts) == 5 and parts[:3] == ["api", "v1", "instruments"] \
                and parts[4] == "sessions":
            return self.service.instrument_sessions(
                unquote(parts[3]), start=_query_first(query, "start"),
                end=_query_first(query, "end"),
                timeframe=_query_first(query, "timeframe"))
        # /api/v1/calendars/{calendar_id}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "calendars"]:
            return self.service.calendar_detail(unquote(parts[3]))
        # /api/v1/providers/{provider_id}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "providers"]:
            return self.service.provider_detail(unquote(parts[3]))
        # Frozen replay evidence is separate from every live shadow route.
        if len(parts) == 6 and parts[:4] == ["api", "v1", "paper", "replay"]:
            return self.service.replay.page(
                parts[4], parts[5], limit=_query_first(query, "limit"),
                cursor=_query_first(query, "cursor"))
        # /api/v1/paper/{product}
        if len(parts) == 4 and parts[:3] == ["api", "v1", "paper"]:
            return self.service.paper_product(parts[3])
        # /api/v1/paper/{product}/{events|equity|fills|predictions}
        if len(parts) == 5 and parts[:3] == ["api", "v1", "paper"]:
            return paper_sub(parts[3], parts[4], query)
        # /api/v1/sources/fomc/items/{sid}
        if len(parts) == 6 and parts[:5] == ["api", "v1", "sources", "fomc", "items"]:
            return self.service.fomc.item(parts[5], as_of=_query_first(query, "as_of"),
                                          horizon=_query_first(query, "horizon"),
                                          limit=_query_first(query, "limit"), cursor=_query_first(query, "cursor"))
        # /api/v1/sources/edgar/filings/{accession}
        if len(parts) == 6 and parts[:5] == ["api", "v1", "sources", "edgar", "filings"]:
            return self.service.edgar.filing(parts[5], as_of=_query_first(query, "as_of"),
                                             horizon=_query_first(query, "horizon"),
                                             limit=_query_first(query, "limit"), cursor=_query_first(query, "cursor"))
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
            payload = self.service.paper_portfolio()
            self._sse("portfolio_status", payload)
            if cursor is None:
                recent = self.service.paper_portfolio_events(
                    limit=DEFAULT_PAPER_EVENTS)
                cursor = recent["page"]["last_event_id"]
                for event in recent["events"]:
                    self._sse(_PORTFOLIO_SSE_KINDS.get(event["event_type"], "event"),
                              event, event_id=event["event_id"])
            while time.monotonic() < deadline:
                batch = self.service.paper_portfolio_events(
                    limit=MAX_SSE_REPLAY_EVENTS,
                    after_event_id=cursor if cursor is not None else 0)
                for event in batch["events"]:
                    self._sse(_PORTFOLIO_SSE_KINDS.get(event["event_type"], "event"),
                              event, event_id=event["event_id"])
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
        if parsed.path == "/api/v1/lab" or parsed.path.startswith("/api/v1/lab/"):
            self._lab_request("GET", parsed, body=body)
            return
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

    def _lab_request(self, method, parsed, *, body=True):
        from scripts.trading_lab.app_api.model_lab import LabApiError
        try:
            if self.lab is None:
                raise LabApiError("Model Lab is not configured", 503)
            self.lab.authorize(self.headers.get("Authorization"), self.headers.get("Origin"))
            query = parse_qs(parsed.query, keep_blank_values=True)
            payload = None
            if method == "POST":
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json" or self.headers.get("Transfer-Encoding"):
                    raise LabApiError("Model Lab requires bounded application/json", 400)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise LabApiError("invalid content length", 400) from None
                if not 1 <= length <= 16384:
                    raise LabApiError("Model Lab payload must be 1..16384 bytes", 413)
                self.connection.settimeout(5)
                try:
                    payload = json.loads(self.rfile.read(length), parse_constant=_reject_json_constant)
                except (ValueError, UnicodeDecodeError, TimeoutError):
                    raise LabApiError("invalid Model Lab JSON", 400) from None
            payload = self.lab.dispatch(method, parsed.path, query, payload)
        except AppApiError as error:
            self._respond(getattr(error, "status", 400), {"error": str(error), "api_version": APP_API_VERSION}, body=body)
            return
        except Exception:
            self._respond(500, {"error": "internal error", "api_version": APP_API_VERSION}, body=body)
            return
        self._respond(202 if method == "POST" and "job_id" in payload else 200, payload, body=body)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/v1/lab/"):
            self._lab_request("POST", parsed)
        else:
            self._refuse()

    do_PUT = do_PATCH = do_DELETE = _refuse


def make_server(data_root, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                dist_root=None, fomc_store=None, edgar_store=None,
                model_lab_root=None, model_lab_token=None, research_root=None):
    """Build a loopback-bound read-only server over a fixed data root.

    ``dist_root`` turns on single-origin production mode. It is resolved once,
    here, so no request can influence which directory is served.
    """
    if model_lab_root is not None and host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Model Lab requires a loopback listener")
    service = AppService(pathlib.Path(data_root), fomc_store=fomc_store, edgar_store=edgar_store)
    # An IPv6 loopback (::1) needs an AF_INET6 socket; the stdlib server is AF_INET only.
    class SourceServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        def server_close(self):
            super().server_close()
            if self.RequestHandlerClass.lab is not None:
                self.RequestHandlerClass.lab.close()
            service.fomc.close()
            service.edgar.close()

    server_class = SourceServer
    site = None
    if dist_root is not None:
        from scripts.trading_lab.ops.static_assets import StaticSite
        site = StaticSite(dist_root)
    from scripts.trading_lab.app_api.research import ResearchViews
    handler = type("BoundAppApiHandler", (AppApiHandler,),
                   {"service": service, "site": site, "research": ResearchViews(research_root)})
    server = server_class((host, port), handler)
    if model_lab_root is not None:
        from scripts.trading_lab.app_api.model_lab import ModelLabApi
        try:
            handler.lab = ModelLabApi(model_lab_root, token=model_lab_token)
        except Exception:
            server.server_close()
            raise
    return server


def main(argv=None):  # pragma: no cover - entry point
    import argparse
    import signal as signal_module

    parser = argparse.ArgumentParser(description="HyprL read-only application API")
    parser.add_argument("--data-root", default="data/crypto")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dist-root", default=None,
                        help="serve a frontend build from the same origin")
    parser.add_argument("--research-root", default=None, help="private registry and observability store to read only")
    parser.add_argument("--model-lab-root", default=None,
                        help="opt-in private synthetic job state; requires HYPRL_MODEL_LAB_TOKEN and loopback")
    parser.add_argument("--fomc-store", default=None,
                        help="an FOMC store directory to read (read-only: an archive or a copy)")
    parser.add_argument("--edgar-store", default=None,
                        help="an EDGAR store directory to read (read-only)")
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
                         port=arguments.port, dist_root=arguments.dist_root,
                         fomc_store=arguments.fomc_store, edgar_store=arguments.edgar_store,
                         model_lab_root=arguments.model_lab_root,
                         model_lab_token=os.environ.get("HYPRL_MODEL_LAB_TOKEN"),
                         research_root=arguments.research_root)
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
