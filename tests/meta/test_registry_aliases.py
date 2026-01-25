from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyprl.meta.registry import (
    REGISTRY_PATH,
    delete_alias,
    get_history,
    list_versions,
    register_model,
    resolve_model,
    set_alias,
)


def _seed_meta(path: Path) -> None:
    path.write_text(json.dumps({"dataset_hash": "abc", "cv_metrics": {"r2": 0.9}, "timestamp": "2025"}))


def test_registry_alias_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hyprl.meta.registry.REGISTRY_PATH", tmp_path / "registry.json")
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    model_path.write_text("model")
    _seed_meta(meta_path)
    record = register_model(model_path, meta_path, key="robustness", stage="Production", version="v0.1.0")
    set_alias("robustness", "stable", record["version"])
    resolved = resolve_model("robustness@stable")
    assert resolved == model_path
    versions = list_versions("robustness")
    assert versions[0]["aliases"] == ["stable"]
    delete_alias("robustness", "stable")
    with pytest.raises(KeyError):
        resolve_model("robustness@stable")
    ops = [entry["op"] for entry in get_history()]
    assert {"register", "alias_set", "alias_delete"}.issubset(set(ops))
