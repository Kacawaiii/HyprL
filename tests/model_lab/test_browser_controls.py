"""The cockpit drives the Model Lab from a same-origin loopback page; foreign pages stay refused."""
import http.client
import json
from urllib.parse import urlparse

import pytest

from scripts.trading_lab.app_api.model_lab import same_loopback_origin
from scripts.trading_lab.platform.model_lab_demo import wait
from tests.model_lab.test_api import TOKEN, request, server_at


def raw_request(base, path, *, method="GET", payload=None, headers=None, token=TOKEN):
    """Send exact Host/Origin headers, as a browser (or a rebinding attacker) would."""
    target = urlparse(base)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=15)
    body = json.dumps(payload).encode() if payload is not None else None
    values = {"Authorization": "Bearer " + token} if token else {}
    if body is not None:
        values["Content-Type"] = "application/json"
    values.update(headers or {})
    connection.putrequest(method, path, skip_host=True)
    for name, value in values.items():
        connection.putheader(name, value)
    if body is not None:
        connection.putheader("Content-Length", str(len(body)))
    connection.endheaders(body)
    response = connection.getresponse()
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def own(base):
    return urlparse(base).netloc


@pytest.mark.parametrize("origin, host, fetch_site, expected", [
    ("http://127.0.0.1:8790", "127.0.0.1:8790", None, True),
    ("http://localhost:8790", "localhost:8790", "same-origin", True),
    ("http://[::1]:8790", "[::1]:8790", None, True),
    ("http://127.0.0.1:8790", "127.0.0.1:8790", "cross-site", False),
    ("http://127.0.0.1:8790", "127.0.0.1:8790", "same-site", False),
    ("http://evil.example:8790", "evil.example:8790", "same-origin", False),   # DNS rebinding
    ("http://localhost:5173", "127.0.0.1:8790", None, False),
    ("http://127.0.0.1:8790", "127.0.0.1:9999", None, False),
    ("https://127.0.0.1:8790", "127.0.0.1:8790", None, False),
    ("null", "127.0.0.1:8790", None, False),
    ("", "127.0.0.1:8790", None, False),
    ("http://127.0.0.1:8790", None, None, False),
])
def test_same_loopback_origin_is_exact(origin, host, fetch_site, expected):
    assert same_loopback_origin(origin, host, 8790, fetch_site) is expected


def test_same_origin_page_is_admitted_with_the_token_and_foreign_pages_are_not(tmp_path):
    with server_at(tmp_path) as (_, base):
        page = {"Host": own(base), "Origin": "http://" + own(base), "Sec-Fetch-Site": "same-origin"}
        assert raw_request(base, "/api/v1/lab/jobs", headers=page)[0] == 200
        assert raw_request(base, "/api/v1/lab/jobs", headers=page, token=None)[0] == 401
        assert raw_request(base, "/api/v1/lab/jobs", headers=page, token="wrong-token")[0] == 401
        port = urlparse(base).port
        rebinding = {"Host": f"evil.example:{port}", "Origin": f"http://evil.example:{port}"}
        assert raw_request(base, "/api/v1/lab/jobs", headers=rebinding)[0] == 403
        cross = {**page, "Origin": "http://localhost:5173"}
        assert raw_request(base, "/api/v1/lab/datasets", method="POST", headers=cross,
                           payload={"synthetic": True, "bars": 120})[0] == 403
        site = {**page, "Sec-Fetch-Site": "cross-site"}
        assert raw_request(base, "/api/v1/lab/datasets", method="POST", headers=site,
                           payload={"synthetic": True, "bars": 120})[0] == 403
        # No CORS grant appears for the lab: a foreign page could not read an answer anyway.
        code, listing = raw_request(base, "/api/v1/lab/jobs", headers=page)
        assert code == 200 and listing["jobs"] == []


def test_browser_journey_dataset_experiment_cancel_and_monitoring(tmp_path):
    with server_at(tmp_path) as (server, base):
        page = {"Host": own(base), "Origin": "http://" + own(base), "Sec-Fetch-Site": "same-origin"}
        store = server.RequestHandlerClass.lab.store
        code, created = raw_request(base, "/api/v1/lab/datasets", method="POST", headers=page,
                                    payload={"synthetic": True, "products": ["BTC-USD"], "bars": 120,
                                             "start": "2026-06-01T00:00:00Z", "horizon_seconds": 14400,
                                             "seed": 7, "target": "forward_return"})
        assert code == 202 and created["synthetic"]
        dataset_hash = wait(store, created["job_id"], 60)["dataset_hash"]
        code, refused = raw_request(base, "/api/v1/lab/monitoring", method="POST", headers=page,
                                    payload={"experiment_job_id": created["job_id"]})
        assert code == 409 and "completed experiment" in refused["error"]
        code, launched = raw_request(base, "/api/v1/lab/experiments", method="POST", headers=page,
                                     payload={"dataset_hash": dataset_hash, "model_id": "local-momentum-v1",
                                              "embargo_seconds": 3600})
        assert code == 202 and launched["prepared"]["status"] == "PREPARED"
        experiment = launched["job_id"]
        code, early = raw_request(base, "/api/v1/lab/monitoring", method="POST", headers=page,
                                  payload={"experiment_job_id": experiment})
        assert code in (202, 409)          # 202 only if the worker already finished
        if code == 202:
            raw_request(base, f"/api/v1/lab/jobs/{early['job_id']}/cancel", method="POST", headers=page, payload={})
        wait(store, experiment, 120)
        for payload in ({}, {"experiment_job_id": "x"}, {"experiment_job_id": experiment, "as_of": "now"},
                        {"experiment_job_id": 7}):
            assert raw_request(base, "/api/v1/lab/monitoring", method="POST", headers=page, payload=payload)[0] == 400
        assert raw_request(base, "/api/v1/lab/monitoring", method="POST", headers=page,
                           payload={"experiment_job_id": "f" * 32})[0] == 404
        code, monitoring = raw_request(base, "/api/v1/lab/monitoring", method="POST", headers=page,
                                       payload={"experiment_job_id": experiment})
        assert code == 202 and monitoring["subject"] == experiment
        result = wait(store, monitoring["job_id"], 120)
        status = store.status(monitoring["job_id"])
        assert status["kind"] == "monitoring" and status["subject"] == experiment
        assert result["schema"] == "lab-experiment-monitoring-v1" and result["synthetic"]
        assert result["experiment_job_id"] == experiment and result["ledger"]["verified"]
        assert result["reference_split"] == "validation" and result["monitored_split"] == "test"
        view = result["products"]["BTC-USD"]["view"]
        assert view["sample"] > 0 and view["reference_hash"] == result["products"]["BTC-USD"]["reference_hash"]
        assert {"MISSING_DATA", "TECHNICAL_DEGRADATION", "DRIFT", "PERFORMANCE_DROP"} >= {
            item["category"] for item in view["classification"]}
        experiment_result = store.result(experiment)["result"]
        test_predictions = [p for p in experiment_result["predictions"] if p["split"] == "test"]
        assert view["sample"] == len(test_predictions)
        # Cancelling from the page: a long dataset is cancelled before it can publish a result.
        code, long_job = raw_request(base, "/api/v1/lab/datasets", method="POST", headers=page,
                                     payload={"synthetic": True, "bars": 600})
        assert code == 202
        code, cancelled = raw_request(base, f"/api/v1/lab/jobs/{long_job['job_id']}/cancel", method="POST",
                                      headers=page, payload={})
        assert code == 200 and cancelled["cancel_requested"]
        with pytest.raises(RuntimeError, match="CANCELLED"):
            wait(store, long_job["job_id"], 60)
        assert str(tmp_path) not in json.dumps(result)


def test_monitoring_jobs_list_their_subject_only(tmp_path):
    with server_at(tmp_path) as (server, base):
        store = server.RequestHandlerClass.lab.store
        identifier = store.submit("monitoring", {"experiment_job_id": "a" * 32})
        store.cancel(identifier)
        code, listing = request(base, "/api/v1/lab/jobs")
        job = next(item for item in listing["jobs"] if item["id"] == identifier)
        assert code == 200 and job["subject"] == "a" * 32 and "payload" not in job
