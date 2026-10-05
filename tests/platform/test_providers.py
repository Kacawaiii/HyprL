import json
import pytest
from scripts.trading_lab.platform.providers import FUTURE, descriptors, future_contract, source_contract
from scripts.trading_lab.platform.synthetic import provider_fixture
from scripts.trading_lab.edgar.listing import parse_listing
from scripts.trading_lab.edgar import spec as edgar
from scripts.trading_lab.fomc import spec as fomc


@pytest.mark.parametrize("name,spec", [("fomc", fomc), ("edgar", edgar)])
def test_descriptors_bind_proven_implementation(name, spec):
    record = source_contract(name)
    assert record.provider_id == spec.PROVIDER_ID and record.identities["spec_hash"] == spec.verify_spec_binding()
    assert record.limits["spacing_seconds"] == spec.SPACING_S
    assert record.clocks["clock_error_bound_seconds"] == 92
    assert record.health["success"] == "result_state null, reason NO_FAILURE"
    assert record.activation == "READ_ONLY_ARCHIVE" and "causal_read" in record.capabilities
    assert "attests" in record.clocks["availability"] and "not server attestation" in record.clocks["ingestion"]


@pytest.mark.parametrize("kind", FUTURE)
def test_future_descriptors_and_fixtures_are_not_activation(kind):
    record, fixture = future_contract(kind), provider_fixture(kind)
    assert fixture["synthetic"] and fixture["state"] == record.activation == "WAITING_AUTHORIZATION"
    assert fixture["shape_verification"] == record.shape_verification == "shape not verified against the live source"
    assert record.limits["real_requests"] == 0 and record.capabilities == ("synthetic_fixture",)
    if kind == "companies":
        sample = fixture["sample"]
        assert sample["cik"] == "0000320193"
        assert parse_listing(json.dumps(sample).encode(), sample["cik"]).in_scope
    if kind == "crypto":
        row = fixture["sample"][0]
        assert len(row) == 6 and isinstance(row[0], int) and row[1] <= row[3] <= row[2]


def test_public_descriptors_are_stable_and_no_private_paths():
    assert descriptors() == descriptors()
    assert len(descriptors()) == 8
    assert "/home/" not in json.dumps(descriptors()) and "/srv/" not in json.dumps(descriptors())
