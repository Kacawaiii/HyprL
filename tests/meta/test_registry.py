from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyprl.meta.registry import REGISTRY_PATH, register_model, promote_model, resolve_model


def test_register_promote_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hyprl.meta.registry.REGISTRY_PATH", tmp_path / "registry.json")
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    model_path.write_text("model")
    meta_path.write_text(json.dumps({"dataset_hash": "abc", "cv_metrics": {"r2": 0.5}, "timestamp": "2025"}))
    record = register_model(model_path, meta_path, key="robustness", stage="Staging")
    assert record["version"].startswith("v0")
    promote_model("robustness", record["version"], stage="Production")
    resolved = resolve_model("robustness@production")
    assert resolved == model_path
    record_bis = register_model(model_path, meta_path, key="robustness", stage="Staging")
    assert record_bis["version"] != record["version"]
    with pytest.raises(ValueError):
        register_model(model_path, meta_path, key="robustness", stage="Staging", version=record_bis["version"])
