import json
from pathlib import Path

import pytest

from examples.b2b_client import Client, demonstrate
from scripts.trading_lab.platform.model_lab_demo import wait
from tests.b2b.conftest import ALPHA, BETA, READER, request, running


@pytest.mark.parametrize("raw", [b'null', b'[]', b'{"synthetic":NaN}', b'{"synthetic":true,"synthetic":false}', b'not-json'])
def test_invalid_and_ambiguous_json_is_rejected_and_audited(tmp_path, raw):
    with running(tmp_path) as (api, base):
        status, result = request(base, "/datasets", method="POST", raw=raw)
        assert status == 400 and result["error"] in {"INVALID_BODY", "INVALID_REQUEST"}
        assert not api.jobs.list()
        assert api.control.audit_page("alpha")["entries"][-1]["status"] == 400


def test_body_query_exports_and_registration_boundaries(tmp_path):
    with running(tmp_path) as (api, base):
        for payload in ({"synthetic": False, "products": ["BTC-USD"]},
            {"synthetic": True, "products": ["BTC-USD"], "path": "synthetic-private"},
            {"synthetic": True, "products": ["BTC-USD"], "seed": 10001},
            {"synthetic": True, "products": ["BTC-USD"], "bars": 79}):
            assert request(base, "/datasets", method="POST", payload=payload)[0] == 400
        assert request(base, "/datasets", method="POST", raw=b'"' + b'x' * 16384 + b'"')[0] == 413
        assert request(base, "/jobs", query={"limit": ""})[0] == 400
        assert request(base, "/jobs", query={"limit": "1", "path": "synthetic-private"})[0] == 400
        assert request(base, "/datasets/" + "a" * 64 + "/export")[0] == 403
        for adapter in ("paper-ridge-v2", "https://synthetic.invalid/model", "uninstalled-entry-point"):
            assert request(base, "/models", method="POST", payload={"model_id": "synthetic", "adapter_id": adapter})[0] == 400
        registration = {"model_id": "synthetic", "adapter_id": "local-momentum-v1"}
        assert request(base, "/models", method="POST", payload=registration)[0] == 201
        assert request(base, "/models", method="POST", payload=registration)[0] == 201
        registration["adapter_id"] = "synthetic-ridge-v1"
        assert request(base, "/models", method="POST", payload=registration)[0] == 409
        assert request(base, "/models/synthetic", project="beta", key=BETA)[0] == 404
        assert not api.jobs.list()


def test_local_client_full_worker_flow_cross_project_objects_and_export_rights(tmp_path, configuration, monkeypatch):
    from scripts.trading_lab.models import RidgeRegressionPredictor
    def forbidden_training(*args, **kwargs):
        raise AssertionError("API process tried training")
    monkeypatch.setattr(RidgeRegressionPredictor, "fit", forbidden_training)
    with running(tmp_path, configuration) as (api, base):
        result = demonstrate(Client(base, "alpha", ALPHA))
        assert result["synthetic"] and result["result_state"] == "COMPLETE" and result["predictions"] > 0
        jobs = request(base, "/jobs")[1]["data"]["jobs"]
        experiment_id = next(j["id"] for j in jobs if j["kind"] == "experiment")
        dataset_id = next(j["id"] for j in jobs if j["kind"] == "dataset")
        assert all(j["worker_pid"] and j["worker_pid"] != __import__("os").getpid() for j in jobs)
        for leaf in ("/jobs/" + experiment_id, "/jobs/" + experiment_id + "/cancel",
                     "/experiments/" + experiment_id + "/results", "/experiments/" + experiment_id + "/predictions"):
            assert request(base, leaf, project="beta", key=BETA,
                           method="POST" if leaf.endswith("cancel") else "GET", payload={} if leaf.endswith("cancel") else None)[0] == 404
        assert request(base, "/jobs", project="beta", key=BETA)[1]["data"]["jobs"] == []
        dataset_hash = result["dataset_hash"]
        assert request(base, "/datasets/" + dataset_hash + "/manifest")[0] == 403
        dataset_result = request(base, "/experiments/" + dataset_id + "/results")[1]["data"]["result"]
        assert "manifest" not in dataset_result and "rows" not in dataset_result
        response = request(base, "/experiments/" + experiment_id + "/results", key=READER)[1]["data"]["result"]
        assert not ({"models", "shadow", "backtests", "predictions"} & set(response))
        project = api.configuration.projects["alpha"]
        project["exports"] = ["dataset_manifest", "dataset_rows", "model_artifact"]
        assert request(base, "/datasets/" + dataset_hash + "/manifest", key=READER)[0] == 403
        assert request(base, "/datasets/" + dataset_hash + "/manifest")[1]["data"]["manifest"]["synthetic"]
        export = request(base, "/datasets/" + dataset_hash + "/export")[1]["data"]
        assert export["fingerprint"] == dataset_hash and export["rows"]
        beta = api.configuration.projects["beta"]
        beta["exports"] = ["dataset_manifest", "dataset_rows", "model_artifact"]
        assert request(base, "/datasets/" + dataset_hash + "/export", project="beta", key=BETA)[0] == 404
        model_path = "/experiments/" + experiment_id + "/models/demo-momentum/export"
        assert request(base, model_path, query={"product": "BTC-USD"})[0] == 200
        assert request(base, model_path, project="beta", key=BETA, query={"product": "BTC-USD"})[0] == 404
        assert str(tmp_path) not in json.dumps(response)
        # Existing granted digests remain unusable after data-right revocation.
        project["products"] = []
        assert request(base, "/datasets/" + dataset_hash + "/export")[0] == 403
        assert request(base, "/experiments/" + experiment_id + "/results")[0] == 403


def test_cancel_job_budget_and_artifact_corruption(tmp_path, configuration):
    configuration["projects"]["alpha"]["job_budget"] = 1
    with running(tmp_path, configuration) as (api, base):
        status, queued = request(base, "/datasets", method="POST", payload={"synthetic": True, "products": ["BTC-USD"], "bars": 80})
        assert status == 202
        job_id = queued["data"]["job_id"]
        result = wait(api.jobs, job_id, 60)
        # Rights are only bound from verified results of an owned job.
        assert request(base, "/experiments/" + job_id + "/results")[0] == 200
        status, error = request(base, "/datasets", method="POST", payload={"synthetic": True, "products": ["BTC-USD"]})
        assert status == 429 and error["error"] == "JOB_BUDGET_EXHAUSTED"
        assert request(base, "/jobs/" + job_id + "/cancel", method="POST", payload={})[1]["data"]["state"] == "COMPLETE"
        with api.jobs.connect() as db:
            db.execute("UPDATE artifacts SET payload='{}' WHERE hash=?", (result["dataset_hash"],))
        status, error = request(base, "/experiments/" + job_id + "/results")
        # The result still binds the dataset: no false success after corruption.
        assert status == 409 and error["error"] == "ARTIFACT_INTEGRITY_ERROR"
        api.configuration.projects["alpha"]["exports"] = ["dataset_rows"]
        status, error = request(base, "/datasets/" + result["dataset_hash"] + "/export")
        assert status == 409 and error["error"] == "ARTIFACT_INTEGRITY_ERROR"


def test_example_refuses_remote_hosts_redirect_credentials_and_bad_paths():
    for base in ("https://example.com", "http://localhost:8791", "http://127.0.0.1@remote.invalid", "http://127.0.0.1/api", "http://127.0.0.1?key=private"):
        with pytest.raises(ValueError, match="local"):
            Client(base, "alpha", ALPHA)
    client = Client("http://127.0.0.1:1", "alpha", ALPHA)
    with pytest.raises(ValueError):
        client.request("/../beta/jobs")
