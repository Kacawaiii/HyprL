from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from scripts.trading_lab.b2b.security import B2BError, Configuration, ControlStore, Principal, key_hash
from scripts.trading_lab.b2b.server import make_server
from scripts.trading_lab.platform.jobs import JobStore
from tests.b2b.conftest import ALPHA, BETA, READER, config_payload, request, running


@pytest.mark.parametrize("key", [None, "wrong", "synthetic-invalid-key-000000000000000000"])
def test_all_routes_require_auth_and_failures_are_audited(tmp_path, key):
    with running(tmp_path) as (api, base):
        for leaf in ("", "/models", "/jobs", "/audit"):
            status, result = request(base, leaf, key=key)
            assert status == 401 and result["error"] == "AUTH_REQUIRED"
            assert "project_id" not in result and str(tmp_path) not in json.dumps(result)
        assert request(base, "/api/b2b/v1/openapi.json", project=None, key=key)[0] == 401
        with api.jobs.connect() as db:
            entries = [json.loads(r[0]) for r in db.execute("SELECT payload FROM b2b_audit")]
        assert len(entries) == 5 and all(e["status"] == 401 and e["project_id"] is None for e in entries)


def test_key_expiry_revocation_hash_only_configuration_and_private_file(tmp_path, configuration):
    config = Configuration(configuration)
    with pytest.raises(B2BError):
        config.authenticate("Bearer " + ALPHA, now=datetime(2099, 1, 1, tzinfo=timezone.utc))
    configuration["keys"][0]["enabled"] = False
    with pytest.raises(B2BError):
        Configuration(configuration).authenticate("Bearer " + ALPHA)
    configuration["keys"][0]["secret"] = ALPHA
    with pytest.raises(ValueError, match="hashed"):
        Configuration(configuration)
    path = tmp_path / "private.json"
    path.write_text(json.dumps(config_payload()))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        Configuration.load(path)
    path.chmod(0o600)
    assert Configuration.load(path).authenticate("Bearer " + ALPHA).project_id == "alpha"
    assert ALPHA not in path.read_text() and ALPHA != key_hash(ALPHA)
    for value in ("a" * 31, "é" * 40, "a" * 257):
        with pytest.raises(ValueError):
            key_hash(value)
    # Even a validly shaped operator hash must not authenticate an invalid
    # candidate through the dummy constant used for timing comparison.
    bad_config = config_payload()
    bad_config["keys"][0]["key_sha256"] = "0" * 64
    for authorization in (None, "Bearer short", "Bearer " + "é" * 40):
        with pytest.raises(B2BError):
            Configuration(bad_config).authenticate(authorization)


def test_project_permission_origin_method_and_data_grants(tmp_path):
    with running(tmp_path) as (_, base):
        assert request(base, "/models", project="beta")[0] == 403
        assert request(base, "/models", payload={}, method="POST", key=READER)[0] == 403
        assert request(base, headers={"Origin": "http://localhost:5173"})[0] == 403
        assert request(base, method="PUT", payload={})[0] == 405
        assert request(base, method="HEAD") == (200, b"")
        assert request(base, "/api/v1/lab/jobs", project=None)[0] == 404
        assert request(base, "/api/v1/sources/fomc", project=None, method="POST", payload={})[0] == 404
        assert request(base, "/datasets", method="POST", payload={"synthetic": True, "products": ["ETH-USD"]})[0] == 403
        assert request(base, "/observability/monitoring", query={"as_of": "2026-06-01T00:00:00Z", "product": "ETH-USD"})[0] == 403


def test_request_budget_atomic_shared_between_keys_and_persistent(tmp_path, configuration):
    configuration["projects"]["alpha"]["request_budget"] = 3
    with running(tmp_path, configuration) as (api, base):
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(lambda i: request(base, key=ALPHA if i % 2 else READER)[0], range(8)))
        assert statuses.count(200) == 3 and statuses.count(429) == 5
        assert api.control.usage("alpha", configuration["projects"]["alpha"])["requests"]["used"] == 3
        assert request(base, project="beta", key=BETA)[0] == 200
    with running(tmp_path, configuration) as (api, base):
        status, error = request(base)
        assert status == 429 and error["error"] == "REQUEST_BUDGET_EXHAUSTED"
        assert api.control.audit_page("alpha")["verified"]


def test_job_quota_and_ownership_commit_with_jobs_and_queue_failure_rolls_back(tmp_path):
    jobs = JobStore(tmp_path)
    control = ControlStore(jobs)
    principal = Principal("synthetic", "alpha", frozenset())
    project = {"request_budget": 10, "job_budget": 2}
    control.admit_request(principal, project, "synthetic-request", "create_dataset")
    def submit(i):
        try:
            return control.submit(principal, project, "synthetic-" + str(i), "dataset", {"synthetic_test": True})
        except B2BError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(submit, range(6)))
    assert values.count("JOB_BUDGET_EXHAUSTED") == 4
    accepted = [v for v in values if len(v) == 32]
    assert len(accepted) == 2
    for job_id in accepted:
        control.require_owner("alpha", "job", job_id)
        with pytest.raises(B2BError):
            control.require_owner("beta", "job", job_id)
    assert ControlStore(JobStore(tmp_path)).usage("alpha", project)["jobs"]["used"] == 2
    with jobs.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM b2b_owners").fetchone()[0] == 2
    for i in range(6):
        jobs.submit("dataset", {"synthetic": True, "i": i})
    project["job_budget"] = 3
    with pytest.raises(B2BError, match="WORKER_QUEUE_EXHAUSTED") as exhausted:
        control.submit(principal, project, "synthetic-full", "dataset", {})
    assert exhausted.value.status == 429
    assert control.usage("alpha", project)["jobs"]["used"] == 2


def test_audit_records_denials_and_successes_without_payloads_keys_or_other_projects(tmp_path):
    with running(tmp_path) as (api, base):
        request(base, "/models", project="beta")
        request(base, "/models", method="POST", payload={"model_id": "synthetic-model", "adapter_id": "local-momentum-v1"})
        request(base, project="beta", key=BETA)
        request(base, "/models", method="POST", raw=b'{"secret":"synthetic-body-content"}')
        status, response = request(base, "/audit")
        assert status == 200 and response["data"]["verified"]
        entries = response["data"]["entries"]
        assert {"PROJECT_DENIED", "OK", "INVALID_REQUEST"} <= {e["code"] for e in entries}
        assert all(e["project_id"] == "alpha" for e in entries)
        encoded = json.dumps(entries)
        for private in (ALPHA, BETA, "synthetic-body-content", str(tmp_path), "synthetic-model", "beta-admin"):
            assert private not in encoded
        with api.jobs.connect() as db:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute("UPDATE b2b_audit SET payload='{}'")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute("DELETE FROM b2b_audit")
            db.execute("DROP TRIGGER b2b_audit_no_update")
            db.execute("UPDATE b2b_audit SET chain_hash=? WHERE sequence=1", ("0" * 64,))
        status, response = request(base, "/audit")
        assert status == 409 and response["error"] == "AUDIT_INTEGRITY_ERROR"


def test_listener_roots_and_archive_isolation_fail_closed(tmp_path, configuration):
    with pytest.raises(ValueError, match="loopback"):
        make_server(Configuration(configuration), tmp_path / "runtime", host="0.0.0.0")
    configuration["projects"]["alpha"]["research_root"] = str(tmp_path / "archive")
    configuration["projects"]["beta"]["research_root"] = str(tmp_path / "archive")
    with pytest.raises(ValueError, match="separate"):
        Configuration(configuration)
    configuration["projects"]["beta"].pop("research_root")
    with pytest.raises(ValueError, match="separate"):
        make_server(Configuration(configuration), tmp_path / "archive", port=0)
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen" / "sentinel").write_text("synthetic archive sentinel")
    with pytest.raises(ValueError, match="marked"):
        make_server(Configuration(configuration), tmp_path / "frozen", port=0)
    assert (tmp_path / "frozen" / "sentinel").read_text() == "synthetic archive sentinel"
