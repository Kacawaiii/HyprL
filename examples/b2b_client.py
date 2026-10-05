"""Local synthetic B2B flow. Run from the checkout: python -m examples.b2b_client --demo.

Against an existing local server, supply HYPRL_B2B_KEY privately in the environment.
The demo generates a random in-memory key; only its hash reaches server config.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, base_url, project_id, key):
        from scripts.trading_lab.b2b.security import identifier, key_hash
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("example client requires a literal local HTTP listener")
        identifier(project_id)
        key_hash(key)
        self.base_url = base_url.rstrip("/") + "/api/b2b/v1/projects/" + project_id
        self.key = key
        # A proxy environment or redirect must never forward a key elsewhere.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, leaf="", *, payload=None, query=None):
        if leaf and (not leaf.startswith("/") or ".." in leaf or "?" in leaf or "#" in leaf):
            raise ValueError("invalid relative resource")
        url = self.base_url + leaf + ("?" + urlencode(query) if query else "")
        headers = {"Authorization": "Bearer " + self.key}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, allow_nan=False).encode()
        try:
            with self.opener.open(Request(url, headers=headers, data=data), timeout=15) as response:
                envelope = json.load(response)
        except HTTPError as error:
            detail = json.load(error)
            raise RuntimeError("B2B request failed: " + str(error.code) + " " + detail.get("error", "HTTP_ERROR")) from None
        if envelope["schema"] != "b2b-response-v1" or envelope["api_version"] != "1.0.0":
            raise RuntimeError("unexpected B2B contract version")
        return envelope["data"]

    def wait(self, job_id, *, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.request("/jobs/" + job_id)
            if status["state"] == "COMPLETE":
                return self.request("/experiments/" + job_id + "/results")["result"]
            if status["state"] in {"FAILED", "CANCELLED", "BLOCKED"}:
                raise RuntimeError("synthetic job ended " + status["state"] + ": " + str(status["error_code"]))
            time.sleep(.25)
        raise TimeoutError("synthetic job deadline exceeded")


def demonstrate(client):
    model = client.request("/models", payload={"model_id": "demo-momentum", "adapter_id": "local-momentum-v1"})
    dataset_job = client.request("/datasets", payload={"synthetic": True, "products": ["BTC-USD"],
        "start": "2026-06-01T00:00:00Z", "bars": 120, "seed": 7, "horizon_seconds": 14400})
    dataset = client.wait(dataset_job["job_id"])
    prepared = client.request("/experiments", payload={"dataset_hash": dataset["dataset_hash"], "model_id": model["model_id"]})
    result = client.wait(prepared["job_id"])
    predictions = client.request("/experiments/" + prepared["job_id"] + "/predictions", query={"limit": 200})
    return {"synthetic": True, "dataset_hash": dataset["dataset_hash"], "model_contract_hash": model["contract_hash"],
        "experiment_hash": result["fingerprint"], "result_state": result["manifest"]["status"],
        "predictions": len(predictions["predictions"]), "criteria_met": result["criteria_met"],
        "limitations": result["limitations"]}


@contextmanager
def demo_server(root):
    from scripts.trading_lab.b2b.security import Configuration, PERMISSIONS, key_hash
    from scripts.trading_lab.b2b.server import make_server
    key = secrets.token_urlsafe(32)
    configuration = Configuration({"schema": "b2b-config-v1", "projects": {"demo": {
        "request_budget": 2000, "job_budget": 16, "products": ["BTC-USD"], "sources": [], "exports": []}},
        "keys": [{"key_id": "demo-client", "project_id": "demo", "key_sha256": key_hash(key),
            "permissions": sorted(PERMISSIONS), "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), "enabled": True}]})
    server = make_server(configuration, root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Client("http://127.0.0.1:" + str(server.server_address[1]), "demo", key)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8791")
    parser.add_argument("--project", default="demo")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-root", default="var/trading_lab/b2b-demo")
    args = parser.parse_args(argv)
    if args.demo:
        with demo_server(Path(args.demo_root)) as client:
            result = demonstrate(client)
    else:
        result = demonstrate(Client(args.base_url, args.project, os.environ.get("HYPRL_B2B_KEY", "")))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
