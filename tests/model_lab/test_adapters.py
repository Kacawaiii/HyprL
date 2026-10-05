from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts.trading_lab.platform.adapters import ModelRegistry
from scripts.trading_lab.platform.contracts import ModelContract, OUTPUTS
from scripts.trading_lab.market_dataset import DatasetRow


def model_rows(rows, *, labels=False):
    return tuple(DatasetRow(bar_open_at=r["bar_open_at"], usable=True,
        features=tuple((c, Decimal(v)) for c, v in r["features"]),
        label=Decimal(r["label"]) if labels else None) for r in rows)


def test_descriptors_explicit_capabilities_versions_and_absent_outputs():
    descriptors = ModelRegistry().descriptors()
    assert len(descriptors) == 4
    for entry in descriptors:
        model = ModelContract.from_dict(entry["contract"])
        assert model.identity == entry["fingerprint"]
        assert set(model.outputs) == set(OUTPUTS)
        assert all(model.outputs[k] is None for k in OUTPUTS if k != "return")
        assert model.horizons_seconds == (14400,)
        if model.model_id.startswith("paper-"):
            assert "train" not in model.capabilities and not model.synthetic
        if model.model_id == "local-momentum-v1":
            assert entry["registration"] == "external-local-entry-point"


@pytest.mark.parametrize("model_id", ["synthetic-ridge-v1", "local-momentum-v1"])
def test_serialization_prediction_isolation_and_real_training_refusal(dataset, model_id):
    adapter = ModelRegistry().create(model_id)
    rows = model_rows(dataset["rows"][:35], labels=True)
    with pytest.raises(ValueError, match="synthetic|real training"):
        adapter.train(rows, synthetic=False)
    adapter.train(rows, synthetic=True)
    artifact = adapter.serialize()
    other = ModelRegistry().create(model_id).restore(artifact)
    class View:
        usable = True
        bar_open_at = rows[0].bar_open_at
        features = rows[0].features
        @property
        def label(self):
            raise AssertionError("prediction accessed future label")
    expected = adapter.predict([View()])
    assert expected == other.predict([View()])
    assert isinstance(expected[0], Decimal)
    corrupted = deepcopy(artifact)
    corrupted["model_contract_hash"] = "0" * 64
    with pytest.raises(ValueError):
        other.restore(corrupted)


@pytest.mark.parametrize("version", ["1", "2"])
def test_existing_frozen_artifacts_keep_every_identity(dataset, version):
    from scripts.trading_lab.paper_model import load_paper_model, PAPER_MODEL_SPEC_V1, PAPER_MODEL_SPEC_V2
    path = Path("data/models/paper_v" + version) / "BTC-USD.json"
    artifact = json.loads(path.read_bytes())
    frozen_bytes = path.read_bytes()
    adapter = ModelRegistry().create("paper-ridge-v" + version).load(artifact, product="BTC-USD")
    spec = PAPER_MODEL_SPEC_V1 if version == "1" else PAPER_MODEL_SPEC_V2
    original = load_paper_model(artifact, spec, product="BTC-USD")
    rows = model_rows(dataset["rows"][:5])
    assert adapter.predict(rows) == original.predict(rows)
    assert adapter.model.fitted.fitted_model_hash == artifact["fitted_hash"]
    assert adapter.serialize() == artifact
    assert path.read_bytes() == frozen_bytes
    with pytest.raises(Exception, match="product|instrument|market"):
        adapter.load(artifact, product="ETH-USD")


def test_registry_rejects_overwrite_and_client_imports():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_entry_point("scripts.trading_lab.platform.local_momentum:create_adapter")
    with pytest.raises(ValueError, match="unknown"):
        registry.create("arbitrary.module:run")
