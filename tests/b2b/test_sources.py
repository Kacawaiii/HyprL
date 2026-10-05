from datetime import timedelta
import hashlib
import json
from urllib.parse import urlencode

from scripts.trading_lab.b2b.contracts import PREFIX
from scripts.trading_lab.edgar import synthetic
from scripts.trading_lab.event_features.join import SourceJoin
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.research.monitoring import make_reference
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.b2b.conftest import ALPHA, BETA, request, running
from tests.crypto.test_event_features import A, ACC
from tests.platform.test_snapshot import edgar, fomc  # reusable real-shape synthetic provider fixtures
from tests.research.conftest import issue, label


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def selection(at, products="AAPL", **kwargs):
    return {"as_of": at.isoformat() if hasattr(at, "isoformat") else at,
            "products": products, "visibility_mode": "DURABLE_OBSERVED", **kwargs}


def test_versioned_events_revisions_normalized_shapes_and_read_only_archives(tmp_path, configuration, edgar):
    project = configuration["projects"]["alpha"]
    project.update(products=["AAPL"], sources=["edgar"], edgar_store=str(edgar.root), synthetic_sources=True)
    before_files = fingerprint(edgar.root)
    with running(tmp_path, configuration) as (_, base):
        query = selection(edgar.T)
        status, response = request(base, "/snapshots", query=query)
        assert status == 200
        data = response["data"]
        snapshot = data["snapshot"]
        assert data["fingerprint"] == sha256_canonical(snapshot)
        assert snapshot["synthetic"] and snapshot["sources"]["edgar"]["state"] == "RESOLVED"
        assert snapshot["sources"]["fomc"]["state"] == "NOT_CONFIGURED"
        assert snapshot["coverage"]["state"] == "PARTIAL"
        assert request(base, "/snapshots", query=selection(edgar.T, fomc_horizon=1))[0] == 403
        normalized = request(base, "/normalized-data", query=query)[1]["data"]
        assert normalized["synthetic"]
        assert normalized["events"] and all(row["cik"] == "0000320193" for row in normalized["events"])
        assert normalized["snapshot_hash"] == data["fingerprint"]
        event_view = request(base, "/events", query=query)[1]["data"]
        assert event_view["synthetic"]
        events = event_view["events"]
        assert events and all(e["observation_class"] == "INITIAL_INVENTORY" for e in events)
        event_id = events[0]["event_id"]
        revisions = request(base, "/events/" + event_id + "/revisions", query=query)[1]["data"]
        assert revisions["synthetic"]
        assert revisions["selected_revision"] == events[0]["revision"] and revisions["observations"]
        assert all(o["available_at"] <= snapshot["as_of"] for o in revisions["observations"])
        assert request(base, "/events", query=selection("2000-01-01T00:00:00Z"))[1]["data"]["events"] == []
        assert request(base, "/snapshots", query={**query, "horizon": "1"})[0] == 400
        assert request(base, "/snapshots", query=selection(edgar.T, edgar_horizon=10**9))[0] == 400
        assert str(edgar.root) not in json.dumps(snapshot)
    assert fingerprint(edgar.root) == before_files


def test_source_revision_history_remains_causal_under_pinned_horizons(tmp_path, configuration, edgar):
    configuration["projects"]["alpha"].update(products=["AAPL"], sources=["edgar"], edgar_store=str(edgar.root), synthetic_sources=True)
    with running(tmp_path, configuration) as (_, base):
        earlier = request(base, "/snapshots", query=selection(edgar.T))[1]["data"]
        event = next(e for e in earlier["snapshot"]["events"] if e["form"] == "8-K")
        old_at, horizon = edgar.T, earlier["snapshot"]["sources"]["edgar"]["H"]
        edgar.serve(A, synthetic.filing(ACC, items="9.01"))
        edgar.poll()
        current_at = edgar.settle()
        assert request(base, "/snapshots", query=selection(old_at, edgar_horizon=horizon))[1]["data"]["fingerprint"] == earlier["fingerprint"]
        current = request(base, "/events/" + event["event_id"] + "/revisions", query=selection(current_at))[1]["data"]
        assert current["selected_revision"] != event["revision"]
        assert len({o["revision"] for o in current["observations"]}) == 2


def test_fomc_fields_use_native_normalization_and_integrity_states(tmp_path, configuration, fomc):
    configuration["projects"]["alpha"].update(products=["BTC-USD"], sources=["fomc"], fomc_store=str(fomc.root), synthetic_sources=True)
    with SourceJoin("fomc", fomc.root) as reader:
        at = reader.values[-2] - timedelta(microseconds=1)
    before_files = fingerprint(fomc.root)
    with running(tmp_path, configuration) as (_, base):
        query = selection(at, "BTC-USD")
        native = request(base, "/normalized-data", query=query)[1]["data"]
        assert native["events"][0]["fields"]["provider_id"]
        assert native["events"][0]["fields"]["canonical_source_url"]
        snapshot = request(base, "/snapshots", query=query)[1]["data"]["snapshot"]
        digest = snapshot["sources"]["fomc"]["dependencies"]["raw_sha256"][0]
        assert fingerprint(fomc.root) == before_files
        (fomc.root / "raw" / digest[:2] / digest).write_bytes(b"synthetic corruption")
        data = request(base, "/events", query=query)[1]["data"]
        assert data["sources"]["fomc"]["state"] == "INTEGRITY_ERROR" and not data["events"]


def test_project_prediction_late_labels_monitoring_and_curated_archive_isolation(tmp_path, configuration):
    store = ResearchStore(tmp_path / "alpha-research")
    p, _ = issue(store)
    store.append_label(label(p))
    other, _ = issue(store, identifier="synthetic-other-product", product="ETH-USD")
    other_ref = make_reference(store, reference_id="synthetic-eth-ref", as_of="2026-06-04T00:00:00Z", product="ETH-USD", model_id=other.model_id)
    beta = ResearchStore(tmp_path / "beta-research")
    issue(beta, identifier="synthetic-beta", value=".99")
    configuration["projects"]["alpha"]["research_root"] = str(store.root)
    configuration["projects"]["beta"]["research_root"] = str(beta.root)
    before_files = fingerprint(store.root)
    with running(tmp_path, configuration) as (_, base):
        query = {"as_of": "2026-06-01T00:00:00Z", "product": "BTC-USD"}
        detail = "/observability/predictions/" + p.identity
        pending = request(base, detail, query=query)[1]["data"]
        assert pending["label_state"] == "PENDING"
        query["as_of"] = "2026-06-04T00:00:00Z"
        available = request(base, detail, query=query)[1]["data"]
        assert available["label_state"] == "AVAILABLE" and available["prediction_hash"] == p.identity
        assert request(base, detail, project="beta", key=BETA, query=query)[0] == 404
        assert request(base, "/observability/predictions/" + other.identity, query=query)[0] == 404
        page = request(base, "/observability/predictions", query={**query, "limit": 1})[1]["data"]
        assert page["records"][0]["identity"] == p.identity
        assert page["records"][0]["view"]["detail_path"].startswith(PREFIX)
        monitoring = request(base, "/observability/monitoring", query=query)[1]["data"]
        assert monitoring["schema"] == "model-monitoring-v1" and monitoring["sample"] == 1
        assert request(base, "/observability/monitoring", query={**query, "reference_hash": other_ref})[0] == 403
        assert request(base, "/observability/monitoring", query={"product": "BTC-USD"})[0] == 400
    assert fingerprint(store.root) == before_files
