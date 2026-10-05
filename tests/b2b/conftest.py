from contextlib import contextmanager
import json
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from jsonschema import Draft202012Validator

from scripts.trading_lab.b2b.security import Configuration, PERMISSIONS, key_hash
from scripts.trading_lab.b2b.contracts import ROUTES, openapi
from scripts.trading_lab.b2b.server import make_server
from tests.crypto import loopback

# Synthetic credentials exercise protocol shapes, never usable operator keys.
ALPHA = "synthetic-b2b-test-alpha-0000000000000000"
BETA = "synthetic-b2b-test-beta-00000000000000000"
READER = "synthetic-b2b-test-reader-00000000000000"
PREFIX = "/api/b2b/v1"


def config_payload():
    def project():
        return {"request_budget": 2000, "job_budget": 8, "products": ["BTC-USD"], "sources": [], "exports": []}
    keys = [{"key_id": name, "project_id": project_id, "key_sha256": key_hash(key),
             "permissions": sorted(rights), "expires_at": "2099-01-01T00:00:00Z", "enabled": True}
            for name, project_id, key, rights in (("alpha-admin", "alpha", ALPHA, PERMISSIONS),
                ("alpha-reader", "alpha", READER, {"read"}), ("beta-admin", "beta", BETA, PERMISSIONS))]
    return {"schema": "b2b-config-v1", "projects": {"alpha": project(), "beta": project()}, "keys": keys}


@contextmanager
def running(tmp_path, payload=None):
    server = make_server(Configuration(payload or config_payload()), tmp_path / "runtime", host=loopback.host(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.RequestHandlerClass.api, loopback.url(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def request(base, leaf="", *, project="alpha", key=ALPHA, method="GET", payload=None, query=None, headers=None, raw=None):
    path = PREFIX + "/projects/" + project + leaf if project else leaf
    values = {"Authorization": "Bearer " + key} if key else {}
    data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    if data is not None:
        values["Content-Type"] = "application/json"
    values.update(headers or {})
    url = base + path + ("?" + urlencode(query) if query else "")
    try:
        with urlopen(Request(url, method=method, data=data, headers=values), timeout=30) as response:
            status, result = response.status, json.load(response) if method != "HEAD" else response.read()
    except HTTPError as error:
        status, result = error.code, json.load(error) if method != "HEAD" else error.read()
    if method != "HEAD":
        specification = openapi()
        if status >= 400:
            schema = specification["components"]["schemas"]["Error"]
        else:
            route = next(r for r in ROUTES if r.match(path) and r.method == method)
            schema = specification["paths"][route.path][method.lower()]["responses"][str(status)]["content"]["application/json"]["schema"]
        Draft202012Validator({**schema, "components": specification["components"]}, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(result)
    return status, result


@pytest.fixture
def configuration():
    return config_payload()
