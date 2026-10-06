from contextlib import contextmanager
from copy import deepcopy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.trading_lab.app_api.policies import PolicyApiError, PolicyViews, verify_report
from scripts.trading_lab.app_api.server import make_server
from scripts.trading_lab.policies.demo import build_report, save_report
from scripts.trading_lab.sources.canonical import canonical_bytes, sha256_canonical
from tests.crypto import loopback


@pytest.fixture(scope="module")
def envelope():
    report = build_report()
    return {"report": report, "identity": sha256_canonical(report)}


def write(root, envelope):
    root.mkdir(exist_ok=True)
    (root / "report.json").write_bytes(canonical_bytes(envelope))


@contextmanager
def serving(tmp_path, root=None):
    server = make_server(tmp_path / "data", host=loopback.host(), port=0, policy_root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield loopback.url(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def request(base, path, method="GET"):
    try:
        with urlopen(Request(base + path, method=method, data=b"{}" if method == "POST" else None), timeout=15) as response:
            return response.status, json.load(response) if method != "HEAD" else response.read()
    except HTTPError as error:
        return error.code, json.load(error)


def test_reproducible_walk_forward_train_purge_and_distinct_test_population(envelope):
    assert sha256_canonical(build_report()) == envelope["identity"]
    verified = verify_report(envelope)
    for product in verified["calibrations"]:
        previous_test_end = None
        for fold in product["folds"]:
            train, start, test = fold["artifact"]["train"], fold["artifact"]["validation_start"], fold["test"]
            assert train["last_label_available_at"] < start < test["first"]
            assert previous_test_end is None or previous_test_end < test["first"]
            assert fold["raw"]["count"] == fold["calibrated"]["count"] == test["count"]
            sample = fold["sample"]
            assert sample["prediction_hash"] == sha256_canonical({k: v for k, v in sample.items() if k != "prediction_hash"})
            assert not {"label", "label_end", "label_available_at"} & set(sample)
            previous_test_end = test["last"]
        assert sum(f["test"]["count"] for f in product["folds"]) == product["calibrated"]["count"]
    assert len(verified["simulations"]) == 6


def test_http_report_selection_provenance_and_no_mutation(tmp_path, envelope, monkeypatch):
    root = tmp_path / "evidence"
    write(root, envelope)
    before = (root / "report.json").read_bytes()
    monkeypatch.setattr("scripts.trading_lab.policies.calibration.fit_isotonic", lambda *a, **k: pytest.fail("HTTP must not fit"))
    monkeypatch.setattr("scripts.trading_lab.policies.risk.simulate", lambda *a, **k: pytest.fail("HTTP must not simulate"))
    with serving(tmp_path, root) as base:
        code, definitions = request(base, "/api/v1/policies/definitions")
        assert code == 200 and definitions["real_calibration"] == "WAITING_AUTHORIZATION"
        code, full = request(base, "/api/v1/policies/report")
        assert code == 200 and full["identity"] == envelope["identity"]
        code, selected = request(base, "/api/v1/policies/report?product=BTC-USD")
        assert code == 200 and len(selected["report"]["calibrations"]) == 1
        assert selected["selection_hash"] == sha256_canonical(selected["report"])
        assert selected["identity"] == full["identity"]
        tp = selected["report"]["simulations"][0]["plan"]["levels"]["take_profit"]
        assert tp["origin"] == "STRATEGY" and tp["method"] == "synthetic-target-v1"
        assert request(base, "/api/v1/policies/report", "POST")[0] == 405
        assert request(base, "/api/v1/policies/report", "HEAD") == (200, b"")
        for query in ("product=BTC-USD&product=ETH-USD", "product=", "limit=3"):
            assert request(base, "/api/v1/policies/report?" + query)[0] == 400
        assert request(base, "/api/v1/policies/report?product=UNKNOWN")[0] == 404
    assert (root / "report.json").read_bytes() == before


def test_unconfigured_missing_and_corrupt_are_explicit(tmp_path):
    assert PolicyViews().dispatch("/api/v1/policies/report", {})["state"] == "NOT_CONFIGURED"
    root = tmp_path / "evidence"
    with pytest.raises(PolicyApiError) as err:
        PolicyViews(root).dispatch("/api/v1/policies/report", {})
    assert err.value.status == 503
    root.mkdir()
    (root / "report.json").write_text("{corrupt")
    with pytest.raises(PolicyApiError) as err:
        PolicyViews(root).dispatch("/api/v1/policies/report", {})
    assert err.value.status == 409


@pytest.mark.parametrize("mutation", ["revision", "probability", "level", "identity"])
def test_old_revisions_and_tampered_provenance_refused(tmp_path, envelope, mutation):
    changed = deepcopy(envelope)
    report = changed["report"]
    if mutation == "revision":
        report["policy_hash"] = "b" * 64
    elif mutation == "probability":
        report["calibrations"][0]["folds"][0]["sample"]["method"] = "unidentified"
    elif mutation == "level":
        report["simulations"][0]["plan"]["levels"]["stop_loss"]["value"] = "97"
    else:
        changed["identity"] = "b" * 64
    if mutation != "identity":
        changed["identity"] = sha256_canonical(report)
    root = tmp_path / "evidence"
    write(root, changed)
    with pytest.raises(PolicyApiError) as err:
        PolicyViews(root).dispatch("/api/v1/policies/report", {})
    assert err.value.status == 409


def test_demo_refuses_overwriting_or_writing_outside_private_worktree(tmp_path):
    with pytest.raises(ValueError, match="worktree"):
        save_report(tmp_path / "runtime")


def test_cockpit_synthetic_fixture_matches_exact_api_shape():
    from tests.policies.export_views import TARGET, fixture
    assert json.loads(TARGET.read_text()) == fixture()
