from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _hash_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataframe_with_meta(
    df: pd.DataFrame,
    path: Path,
    *,
    key_columns: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = {
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "key_columns": key_columns or [],
        "sha256": hash_file(path),
    }
    if extra:
        meta["extra"] = extra
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


def stamp_provenance(
    outdir: Path,
    *,
    meta_model: dict | None,
    calibrator_meta: dict | None,
    registry_snapshot: dict | None,
    cli_args: dict,
    dataset_hashes: dict[str, str] | None,
    git_hash: str | None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta_model": meta_model,
        "calibrator": calibrator_meta,
        "registry_snapshot": registry_snapshot,
        "cli_args": cli_args,
        "dataset_hashes": dataset_hashes or {},
        "git_hash": git_hash,
    }
    provenance_path = outdir / "provenance.json"
    with provenance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return provenance_path
