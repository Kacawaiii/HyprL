from pathlib import Path

from openapi_spec_validator import validate

from scripts.trading_lab.b2b.contracts import ROUTES, openapi
from scripts.trading_lab.b2b.openapi import generated
from tests.b2b.conftest import PREFIX, request, running


def test_generated_document_is_valid_openapi_and_checked_in_exactly():
    spec = openapi()
    validate(spec)
    artifact = Path(__file__).resolve().parents[2] / "docs/artifacts/b2b_openapi_v1.json"
    assert artifact.read_text() == generated()
    assert len(spec["paths"]) == len({route.path for route in ROUTES})
    assert all(operation["security"] == [{"ApiKey": []}] for path in spec["paths"].values() for operation in path.values())


def test_contract_catalogue_and_http_document_use_the_generated_code_contracts(tmp_path):
    with running(tmp_path) as (_, base):
        status, response = request(base, PREFIX + "/openapi.json", project=None)
        assert status == 200 and response["data"] == openapi()
        status, response = request(base, "/contracts")
        assert status == 200 and response["data"]["schemas"] == openapi()["components"]["schemas"]
        assert len(response["data"]["providers"]) >= 2
        assert response["data"]["adapters"]
