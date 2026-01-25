from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hyprl.utils.provenance import hash_file, save_dataframe_with_meta, stamp_provenance


def test_save_dataframe_with_meta(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "value": [0.1, 0.2]})
    csv_path = tmp_path / "panel.csv"
    meta_path = save_dataframe_with_meta(df, csv_path, key_columns=["id"])
    meta = json.loads(meta_path.read_text())
    assert meta["rows"] == 2
    assert meta["key_columns"] == ["id"]
    assert meta["sha256"] == hash_file(csv_path)


def test_stamp_provenance(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    meta_path = stamp_provenance(
        outdir,
        meta_model={"path": "model.joblib", "version": "v0.1.0"},
        calibrator_meta=None,
        registry_snapshot={"models": {}},
        cli_args={"flag": True},
        dataset_hashes={"input.csv": "dead"},
        git_hash="abcd1234",
    )
    payload = json.loads(meta_path.read_text())
    assert payload["meta_model"]["version"] == "v0.1.0"
    assert payload["cli_args"]["flag"] is True
    assert payload["dataset_hashes"]["input.csv"] == "dead"
    assert payload["git_hash"] == "abcd1234"
