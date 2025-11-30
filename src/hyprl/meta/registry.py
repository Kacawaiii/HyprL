from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REGISTRY_PATH = Path("artifacts/meta_ml/registry.json")


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"models": {}, "history": []}
    with REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_registry(payload: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("models", {})
    payload.setdefault("history", [])
    with REGISTRY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _normalize_stage(stage: str) -> str:
    normalized = stage.strip().title()
    if normalized not in {"Staging", "Production"}:
        raise ValueError(f"Invalid stage '{stage}'. Use 'Staging' or 'Production'.")
    return normalized


def _normalize_alias(alias: str) -> str:
    token = alias.strip().lower()
    if not token:
        raise ValueError("Alias cannot be empty.")
    return token


def _parse_version(version: str) -> tuple[int, int, int]:
    token = version.strip().lstrip("v")
    parts = token.split(".")
    numeric = [int(part) if part.isdigit() else 0 for part in parts[:3]]
    while len(numeric) < 3:
        numeric.append(0)
    major, minor, patch = numeric[:3]
    return int(major), int(minor), int(patch)


def _infer_next_version(entries: list[dict]) -> str:
    if not entries:
        return "v0.1.0"
    latest = max((_parse_version(item["version"]), item["version"]) for item in entries)[0]
    major, minor, patch = latest
    patch += 1
    return f"v{major}.{minor}.{patch}"


def _append_history(registry: dict, entry: dict) -> None:
    history = registry.setdefault("history", [])
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    history.append(payload)


def register_model(
    model_path: Path,
    meta_json: Path,
    key: str,
    stage: Literal["Staging", "Production"] = "Staging",
    version: str | None = None,
) -> dict:
    registry = _load_registry()
    models = registry.setdefault("models", {})
    entry = models.setdefault(key, {"versions": [], "current": {}, "aliases": {}})
    versions = entry["versions"]
    with meta_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    stage_normalized = _normalize_stage(stage)
    version_id = (version or _infer_next_version(versions)).strip()
    if not version_id.startswith("v"):
        version_id = f"v{version_id}"
    if any(entry["version"] == version_id for entry in versions):
        raise ValueError(f"Version {version_id} already exists for key '{key}'.")
    created_at = meta.get("timestamp") or datetime.now(timezone.utc).isoformat()
    record = {
        "version": version_id,
        "path": str(model_path),
        "meta": str(meta_json),
        "dataset_hash": meta.get("dataset_hash", ""),
        "metrics": meta.get("cv_metrics", {}),
        "created_at": created_at,
        "stage": stage_normalized,
    }
    versions.append(record)
    versions.sort(key=lambda item: _parse_version(item["version"]))
    entry.setdefault("current", {})[stage_normalized] = version_id
    _append_history(
        registry,
        {"op": "register", "key": key, "version": version_id, "stage": stage_normalized},
    )
    _save_registry(registry)
    return record


def promote_model(key: str, version: str, stage: Literal["Staging", "Production"]) -> dict:
    registry = _load_registry()
    model_entry = registry.get("models", {}).get(key)
    if not model_entry:
        raise KeyError(f"Model key not found: {key}")
    versions = {item["version"]: item for item in model_entry.get("versions", [])}
    if version not in versions:
        raise KeyError(f"Version {version} missing for key {key}")
    stage_normalized = _normalize_stage(stage)
    model_entry.setdefault("current", {})[stage_normalized] = version
    versions[version]["stage"] = stage_normalized
    _append_history(
        registry,
        {"op": "promote", "key": key, "version": version, "stage": stage_normalized},
    )
    _save_registry(registry)
    return versions[version]


def resolve_model(key_or_path: str) -> Path:
    if "@" not in key_or_path:
        return Path(key_or_path)
    key, selector = key_or_path.split("@", 1)
    registry = _load_registry()
    model_entry = registry.get("models", {}).get(key)
    if not model_entry:
        raise KeyError(f"Model key not found: {key}")
    versions = model_entry.get("versions", [])
    version_lookup: str | None = None
    try:
        stage_normalized = _normalize_stage(selector)
    except ValueError:
        stage_normalized = None
    if stage_normalized is not None:
        version_lookup = model_entry.get("current", {}).get(stage_normalized)
    if version_lookup is None:
        alias_norm = _normalize_alias(selector)
        version_lookup = model_entry.get("aliases", {}).get(alias_norm)
    if not version_lookup:
        raise KeyError(f"No version registered for {key}@{selector}")
    for version_entry in versions:
        if version_entry["version"] == version_lookup:
            return Path(version_entry["path"])
    raise KeyError(f"Version {version_lookup} missing for {key}")


def set_alias(key: str, alias: str, version: str) -> dict:
    registry = _load_registry()
    models = registry.setdefault("models", {})
    entry = models.get(key)
    if not entry:
        raise KeyError(f"Model key not found: {key}")
    versions = {item["version"]: item for item in entry.get("versions", [])}
    if version not in versions:
        raise KeyError(f"Version {version} missing for key {key}")
    alias_norm = _normalize_alias(alias)
    entry.setdefault("aliases", {})[alias_norm] = version
    _append_history(registry, {"op": "alias_set", "key": key, "alias": alias_norm, "version": version})
    _save_registry(registry)
    return {"alias": alias_norm, "version": version}


def delete_alias(key: str, alias: str) -> dict:
    registry = _load_registry()
    entry = registry.get("models", {}).get(key)
    if not entry:
        raise KeyError(f"Model key not found: {key}")
    alias_norm = _normalize_alias(alias)
    aliases = entry.get("aliases", {})
    if alias_norm not in aliases:
        raise KeyError(f"Alias {alias} missing for key {key}")
    removed_version = aliases.pop(alias_norm)
    _append_history(registry, {"op": "alias_delete", "key": key, "alias": alias_norm, "version": removed_version})
    _save_registry(registry)
    return {"alias": alias_norm, "version": removed_version}


def list_versions(key: str) -> list[dict]:
    registry = _load_registry()
    entry = registry.get("models", {}).get(key)
    if not entry:
        return []
    alias_map = entry.get("aliases", {})
    alias_by_version: dict[str, list[str]] = {}
    for alias_name, version in alias_map.items():
        alias_by_version.setdefault(version, []).append(alias_name)
    current = entry.get("current", {})
    records: list[dict] = []
    for version_entry in entry.get("versions", []):
        version_id = version_entry["version"]
        stages = [stage for stage, ver in current.items() if ver == version_id]
        records.append(
            {
                "version": version_id,
                "path": version_entry.get("path"),
                "meta": version_entry.get("meta"),
                "dataset_hash": version_entry.get("dataset_hash", ""),
                "metrics": version_entry.get("metrics", {}),
                "created_at": version_entry.get("created_at", ""),
                "stages": stages,
                "aliases": sorted(alias_by_version.get(version_id, [])),
            }
        )
    return records


def get_registry_snapshot(keys: list[str] | None = None) -> dict:
    registry = _load_registry()
    if not keys:
        return registry
    snapshot = {"models": {}, "history": registry.get("history", [])}
    for key in keys:
        if key in registry.get("models", {}):
            snapshot["models"][key] = registry["models"][key]
    return snapshot


def get_history() -> list[dict]:
    registry = _load_registry()
    return list(registry.get("history", []))
