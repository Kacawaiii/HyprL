from contextlib import contextmanager
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.trading_lab.app_api.server import make_server
from scripts.trading_lab.platform.model_lab_demo import wait
from tests.crypto import loopback

TOKEN = "synthetic-test-operator-token-00000000"


@contextmanager
def server_at(root, *, enabled=True):
    server = make_server(root / "data", host=loopback.host(), port=0,
        **({"model_lab_root": root / "lab", "model_lab_token": TOKEN} if enabled else {}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, loopback.url(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def request(base, path, *, method="GET", payload=None, token=TOKEN, headers=None, raw=None):
    values = {"Authorization": "Bearer " + token} if token else {}
    if payload is not None or raw is not None:
        values["Content-Type"] = "application/json"
    values.update(headers or {})
    data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    req = Request(base + path, method=method, data=data, headers=values)
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_default_server_remains_read_only_and_controls_are_opt_in(tmp_path):
    with server_at(tmp_path, enabled=False) as (_, base):
        assert request(base, "/api/v1/lab/jobs")[0] == 503
        assert request(base, "/api/v1/snapshots", method="POST", payload={})[0] == 405
        assert request(base, "/api/v1/lab/datasets", method="POST", payload={"synthetic": True})[0] == 503


def test_authentication_origin_method_and_loopback_boundaries(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        make_server(tmp_path, host="0.0.0.0", model_lab_root=tmp_path / "lab", model_lab_token=TOKEN)
    with pytest.raises(ValueError, match="operator token"):
        make_server(tmp_path, host=loopback.host(), port=0, model_lab_root=tmp_path / "lab")
    with server_at(tmp_path) as (_, base):
        assert request(base, "/api/v1/lab/jobs", token=None)[0] == 401
        assert request(base, "/api/v1/lab/jobs", token="wrong")[0] == 401
        assert request(base, "/api/v1/lab/jobs", headers={"Origin": "http://localhost:5173"})[0] == 403
        assert request(base, "/api/v1/lab/jobs", method="PUT", payload={})[0] == 405
        code, models = request(base, "/api/v1/lab/models")
        assert code == 200 and len(models["models"]) == 4


def test_invalid_controls_and_payloads_never_expose_operator_paths(tmp_path):
    with server_at(tmp_path) as (server, base):
        for payload in ({"synthetic": False}, {"synthetic": True, "provider_url": "synthetic-unsupported"},
                        {"synthetic": True, "store": str(tmp_path)}):
            code, response = request(base, "/api/v1/lab/datasets", method="POST", payload=payload)
            assert code == 400 and str(tmp_path) not in json.dumps(response)
        assert request(base, "/api/v1/lab/datasets", method="POST", raw=b'{"synthetic":NaN}')[0] == 400
        assert request(base, "/api/v1/lab/datasets", method="POST", raw=b'not-json')[0] == 400
        assert request(base, "/api/v1/lab/datasets", method="POST", raw=b'"' + b'a' * 16384 + b'"')[0] == 413
        assert request(base, "/api/v1/lab/jobs?path=synthetic")[0] == 400
        assert request(base, "/api/v1/lab/jobs/" + "a" * 32)[0] == 404
        assert request(base, "/api/v1/lab/../snapshots")[0] == 404
        assert request(base, "/api/v1/lab/experiments", method="POST",
                       payload={"dataset_hash": "0" * 64, "entry_point": "arbitrary"})[0] == 400
        store = server.RequestHandlerClass.lab.store
        key = store.put_artifact("model", {"synthetic": True})
        with store.connect() as db:
            db.execute("UPDATE artifacts SET payload=? WHERE hash=?", ('{"synthetic":false}', key))
        code, response = request(base, "/api/v1/lab/artifacts/model/" + key)
        assert code == 409 and "integrity" in response["error"]
        assert str(tmp_path) not in json.dumps(response)


def test_http_dataset_experiment_results_and_cancel_use_workers(tmp_path, monkeypatch):
    from scripts.trading_lab.models import RidgeRegressionPredictor
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("training happened inside API process")
    monkeypatch.setattr(RidgeRegressionPredictor, "fit", forbidden_fit)
    with server_at(tmp_path) as (server, base):
        code, response = request(base, "/api/v1/lab/datasets", method="POST",
            payload={"synthetic": True, "products": ["BTC-USD"], "bars": 120})
        assert code == 202
        store = server.RequestHandlerClass.lab.store
        dataset_result = wait(store, response["job_id"], 30)
        dataset_hash = dataset_result["dataset_hash"]
        code, manifest = request(base, "/api/v1/lab/datasets/" + dataset_hash)
        assert code == 200 and manifest["fingerprint"] == dataset_hash
        assert "rows" not in manifest and manifest["synthetic"]
        code, response = request(base, "/api/v1/lab/experiments", method="POST",
            payload={"dataset_hash": dataset_hash})
        assert code == 202 and response["prepared"]["status"] == "PREPARED"
        experiment_job = response["job_id"]
        wait(store, experiment_job, 60)
        code, results = request(base, "/api/v1/lab/jobs/" + experiment_job + "/results")
        assert code == 200 and results["result"]["manifest"]["status"] == "COMPLETE"
        assert results["result"]["shadow"]["chain"]["verified"]
        model_hash = results["result"]["manifest"]["artifacts"]["model_hashes"]["BTC-USD"]
        assert request(base, "/api/v1/lab/artifacts/model/" + model_hash)[0] == 200
        assert request(base, "/api/v1/lab/experiments", method="POST", payload={
            "dataset_hash": dataset_hash, "model_id": "paper-ridge-v2"})[0] == 400
        code, response = request(base, "/api/v1/lab/datasets", method="POST", payload={"synthetic": True, "bars": 600})
        assert code == 202
        code, cancelled = request(base, "/api/v1/lab/jobs/" + response["job_id"] + "/cancel", method="POST", payload={})
        assert code == 200 and cancelled["cancel_requested"]
        assert str(tmp_path) not in json.dumps(results)
